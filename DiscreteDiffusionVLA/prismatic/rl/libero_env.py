"""
libero_env.py

LIBERO environment adapter for DFM RL fine-tuning.

Wraps the LIBERO ``OffScreenRenderEnv`` and the HuggingFace processor into a
single object that produces observation dicts compatible with
``DFMRLTrainer.collect_rollouts``.

Expected obs format::

    {
        "input_ids":            (1, L)  long
        "attention_mask":       (1, L)  bool
        "pixel_values":         (1, n_img, 3, H, W)  float
        "labels":               (1, L)  long  (IGNORE_INDEX at non-action positions)
        "action_positions_mask": (1, L)  bool  (True at action token positions)
    }

Usage::

    from prismatic.rl.libero_env import LiberoRLEnv

    env = LiberoRLEnv(
        task_suite="libero_spatial",
        task_id=0,
        processor=processor,
        vla_config=vla.config,
        unnorm_key="libero_spatial",
    )
    obs = env.reset()
    obs, reward, done, info = env.step(action_cont)
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from prismatic.vla.constants import (
    ACTION_DIM,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
)


def _try_import_libero():
    """Lazy import so the module can be loaded without LIBERO installed."""
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        return benchmark, get_libero_path, OffScreenRenderEnv
    except ImportError as e:
        raise ImportError(
            "LIBERO is required for LiberoRLEnv. "
            "Install it following the instructions at "
            "https://github.com/Lifelong-Robot-Learning/LIBERO"
        ) from e


# Max episode lengths per suite (from run_libero_eval.py).
_TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) → axis-angle (3,)."""
    x, y, z, w = quat
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (angle / s) * np.array([x, y, z], dtype=np.float32)


def _prepare_image(img: np.ndarray) -> np.ndarray:
    """Rotate 180 degrees (LIBERO camera convention) and return uint8 (H,W,3)."""
    return img[::-1, ::-1].copy()


class LiberoRLEnv:
    """
    RL-compatible wrapper around a single LIBERO task.

    Parameters
    ----------
    task_suite : str
        E.g. ``"libero_spatial"``, ``"libero_object"``.
    task_id : int
        Index of the task within the suite.
    processor : transformers.ProcessorMixin
        The HuggingFace processor loaded from the VLA checkpoint
        (``AutoProcessor.from_pretrained(...)``).
    vla_config : Any
        The VLA model config (for action vocab range computation).
    unnorm_key : str
        Key for un-normalisation statistics (typically same as task_suite).
    resolution : int
        Render resolution for LIBERO images.
    init_state_idx : int
        Which pre-recorded initial state to use (0-based).
    """

    def __init__(
        self,
        task_suite: str = "libero_spatial",
        task_id: int = 0,
        processor: Any = None,
        vla_config: Any = None,
        unnorm_key: str = "libero_spatial",
        resolution: int = 256,
        init_state_idx: int = 0,
        proprio_norm_stats: Optional[dict] = None,
    ):
        benchmark_mod, get_libero_path, OffScreenRenderEnv = _try_import_libero()

        self.task_suite_name = task_suite
        self.max_steps = _TASK_MAX_STEPS.get(task_suite, 300)

        # Load benchmark task
        benchmark_dict = benchmark_mod.get_benchmark_dict()
        suite = benchmark_dict[task_suite]()
        task = suite.get_task(task_id)
        self.task_label = task.language
        # Monkey-patch torch.load for LIBERO init states (PyTorch 2.6+
        # defaults to weights_only=True which rejects numpy arrays).
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        try:
            self.init_states = suite.get_task_init_states(task_id)
        finally:
            torch.load = _orig_load
        self.init_state_idx = init_state_idx

        # Build env
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        self.env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        self.env.seed(0)

        # Processor and config
        assert processor is not None, "processor is required"
        self.processor = processor
        self.vla_config = vla_config
        self.unnorm_key = unnorm_key

        # Compute action vocab range (same logic as OpenVLAForActionPrediction)
        n_bins = int(getattr(vla_config, "n_action_bins", 256))
        anchor = getattr(vla_config, "action_vocab_anchor", "pad")
        pad_id = int(getattr(vla_config, "pad_token_id", 32000))
        if anchor == "pad":
            self.action_end = pad_id
        else:
            self.action_end = int(getattr(vla_config, "text_config", vla_config).vocab_size)
        self.action_begin = self.action_end - n_bins
        self.n_bins = n_bins
        self.mask_token_id = int(getattr(vla_config, "mask_token_id", self.action_end))

        self.step_count = 0
        self._cached_prompt_inputs: Optional[Dict[str, torch.Tensor]] = None
        self.proprio_norm_stats = proprio_norm_stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, torch.Tensor]:
        self.step_count = 0
        self.env.reset()
        raw_obs = self.env.set_init_state(self.init_states[self.init_state_idx])
        # Wait 10 dummy steps for objects to stabilise (matches eval).
        dummy = [0, 0, 0, 0, 0, 0, -1]
        for _ in range(10):
            raw_obs, _, _, _ = self.env.step(dummy)
        self._cached_prompt_inputs = None
        return self._make_obs(raw_obs)

    def step(self, action_cont: torch.Tensor):
        """
        Execute ALL NUM_ACTIONS_CHUNK (8) actions from the predicted chunk
        (matches eval's action_queue behaviour).  Accumulates env steps,
        terminates early if the task completes.

        Parameters
        ----------
        action_cont : (1, CHUNK, DIM) tensor of un-normalised actions.

        Returns
        -------
        obs (at end of chunk), reward (1,), done (1,), info dict.
        """
        chunk = action_cont[0].detach().cpu().numpy().copy()  # (CHUNK, DIM)
        total_reward = 0.0
        final_done = False
        final_info = {}
        raw_obs = None
        for step_i in range(chunk.shape[0]):
            self.step_count += 1
            act = chunk[step_i].copy()
            # Match eval's gripper postprocess:
            act[-1] = 2 * act[-1] - 1
            act[-1] = float(np.sign(act[-1]))
            act[-1] *= -1.0
            raw_obs, r, d, info = self.env.step(act.tolist())
            total_reward += float(r)
            final_info = info
            if d or self.step_count >= self.max_steps:
                final_done = True
                break

        reward_scalar = total_reward
        done_scalar = final_done
        info = final_info

        # Also done if max steps reached
        if self.step_count >= self.max_steps:
            done_scalar = True

        obs = self._make_obs(raw_obs)
        reward = torch.tensor([float(reward_scalar)])
        done = torch.tensor([bool(done_scalar)])
        info_out = {"success": torch.tensor([float(done_scalar and reward_scalar > 0)])}
        return obs, reward, done, info_out

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _make_obs(self, raw_obs: dict) -> Dict[str, torch.Tensor]:
        """Build trainer-compatible observation dict from LIBERO raw obs.

        Produces the same obs layout as the eval script:
          - 2 images: agentview + wrist (both rotated 180 deg)
          - Proprioception: eef_pos(3) + axis-angle(3) + gripper(1) = 7D
          - Prompt matches legacy format: "In: What action should..."
        """
        from PIL import Image
        agent_img = _prepare_image(raw_obs["agentview_image"])
        wrist_img = _prepare_image(raw_obs["robot0_eye_in_hand_image"])
        agent_pil = Image.fromarray(agent_img)
        wrist_pil = Image.fromarray(wrist_img)

        # Build prompt (matches training format).
        prompt = f"In: What action should the robot take to {self.task_label.lower()}?\nOut:"

        # Process primary image to get tokenised prompt
        primary = self.processor(prompt, agent_pil)
        # Process wrist image
        wrist = self.processor(prompt, wrist_pil)

        input_ids = primary["input_ids"]       # (1, L_prompt)
        attention_mask = primary["attention_mask"]
        # Concatenate both images along image dim (matches eval: num_images_in_input=2)
        primary_pv = primary["pixel_values"]   # (1, 1, C, H, W)
        wrist_pv = wrist["pixel_values"]       # (1, 1, C, H, W)
        pixel_values = torch.cat([primary_pv, wrist_pv], dim=1)  # (1, 2, C, H, W)

        # Proprio: eef_pos(3) + axis_angle(3) + gripper(1) = 7D
        proprio = np.concatenate([
            raw_obs["robot0_eef_pos"],
            _quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        ]).astype(np.float32)  # (7,)
        # Normalise using dataset statistics (matches eval pipeline).
        if self.proprio_norm_stats is not None:
            s = self.proprio_norm_stats
            if "q99" in s and "q01" in s:
                hi, lo = np.array(s["q99"]), np.array(s["q01"])
            else:
                hi, lo = np.array(s["max"]), np.array(s["min"])
            mask = np.asarray(s.get("mask", np.ones_like(lo, dtype=bool)))
            proprio = np.clip(
                np.where(mask, 2 * (proprio - lo) / (hi - lo + 1e-8) - 1, proprio),
                -1.0, 1.0,
            ).astype(np.float32)

        # Strip EOS if present at end (to match training)
        if input_ids[0, -1] == self.processor.tokenizer.eos_token_id:
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]

        L_prompt = input_ids.shape[1]
        n_act = NUM_ACTIONS_CHUNK * ACTION_DIM

        # Append placeholder action tokens (will be filled with mask tokens)
        mask_tokens = torch.full((1, n_act), self.mask_token_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, mask_tokens], dim=1)

        # Extend attention mask
        mask_ext = torch.ones(1, n_act, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, mask_ext], dim=1)

        # Labels: IGNORE_INDEX everywhere, action positions get mask_token_id
        # (actual GT ids are unknown in RL — env doesn't have them).
        L = input_ids.shape[1]
        labels = torch.full((1, L), IGNORE_INDEX, dtype=input_ids.dtype)
        # Put mask_token_id at action positions as placeholder labels
        labels[:, L_prompt:] = self.mask_token_id

        # Action positions mask
        action_pos_mask = torch.zeros(1, L, dtype=torch.bool)
        action_pos_mask[:, L_prompt:] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
            "action_positions_mask": action_pos_mask,
            "proprio": torch.from_numpy(proprio).unsqueeze(0),  # (1, 7)
        }
