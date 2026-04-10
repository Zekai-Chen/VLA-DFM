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
    ):
        benchmark_mod, get_libero_path, OffScreenRenderEnv = _try_import_libero()

        self.task_suite_name = task_suite
        self.max_steps = _TASK_MAX_STEPS.get(task_suite, 300)

        # Load benchmark task
        benchmark_dict = benchmark_mod.get_benchmark_dict()
        suite = benchmark_dict[task_suite]()
        task = suite.get_task(task_id)
        self.task_label = task.language
        self.init_states = suite.get_task_init_states(task_id)
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, torch.Tensor]:
        self.step_count = 0
        self.env.reset()
        raw_obs = self.env.set_init_state(self.init_states[self.init_state_idx])
        self._cached_prompt_inputs = None  # recompute
        return self._make_obs(raw_obs)

    def step(self, action_cont: torch.Tensor):
        """
        Parameters
        ----------
        action_cont : (1, CHUNK, DIM) tensor of continuous actions in [-1, 1].

        Returns
        -------
        obs, reward (1,), done (1,), info dict.
        """
        self.step_count += 1
        # Take first chunk step
        act = action_cont[0, 0].detach().cpu().numpy()  # (DIM,)
        # Post-process: binarise gripper, invert sign (OpenVLA convention)
        act[-1] = 1.0 if act[-1] >= 0 else -1.0
        act[-1] *= -1.0

        raw_obs, reward_scalar, done_scalar, info = self.env.step(act.tolist())

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
        """Build trainer-compatible observation dict from LIBERO raw obs."""
        img = _prepare_image(raw_obs["agentview_image"])
        from PIL import Image
        pil_img = Image.fromarray(img)

        # Build prompt (matches training format).
        prompt = f"In: What action should the robot take to {self.task_label.lower()}?\nOut:"

        # Tokenise with processor (returns input_ids, attention_mask, pixel_values)
        if self._cached_prompt_inputs is None:
            inputs = self.processor(prompt, pil_img)
            self._cached_prompt_inputs = {
                "prompt_input_ids": inputs["input_ids"],         # (1, L_prompt)
                "prompt_attention_mask": inputs["attention_mask"],
            }
            # Store pixel_values shape for consistency checks
            self._pv_shape = inputs["pixel_values"].shape

        # Re-process image (pixel values change per frame)
        inputs = self.processor(prompt, pil_img)

        input_ids = inputs["input_ids"]       # (1, L_prompt)
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]  # (1, n_img, C, H, W) or (1, C, H, W)

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
        }
