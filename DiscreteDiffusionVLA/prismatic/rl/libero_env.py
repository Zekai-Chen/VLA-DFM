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


def _resize_image_for_policy(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Match the training data pipeline exactly: JPEG encode/decode +
    lanczos3 resize + round-and-clip."""
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("tensorflow required for training-aligned image resize") from e
    x = tf.image.encode_jpeg(img)
    x = tf.io.decode_image(x, expand_animations=False, dtype=tf.uint8)
    x = tf.image.resize(x, (size, size), method="lanczos3", antialias=True)
    x = tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8)
    return x.numpy()


def _center_crop_image(pil_img):
    """Match eval's center crop (crop_scale=0.9, then resize back)."""
    import tensorflow as tf
    from PIL import Image
    x = tf.convert_to_tensor(np.array(pil_img))
    x = tf.image.convert_image_dtype(x, tf.float32)
    # crop_scale=0.9
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    new_h = tf.cast(tf.cast(h, tf.float32) * tf.sqrt(0.9), tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * tf.sqrt(0.9), tf.int32)
    x = tf.image.resize_with_crop_or_pad(x, new_h, new_w)
    x = tf.image.resize(x[None, ...], (h, w), method="lanczos3", antialias=True)[0]
    x = tf.clip_by_value(x, 0, 1)
    x = tf.image.convert_image_dtype(x, tf.uint8, saturate=True)
    return Image.fromarray(x.numpy()).convert("RGB")


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
        rotate_tasks: bool = False,
        num_tasks: int = 10,
    ):
        benchmark_mod, get_libero_path, OffScreenRenderEnv = _try_import_libero()

        self.task_suite_name = task_suite
        self.max_steps = _TASK_MAX_STEPS.get(task_suite, 300)
        self._benchmark_mod = benchmark_mod
        self._get_libero_path = get_libero_path
        self._OffScreenRenderEnv = OffScreenRenderEnv
        self._resolution = resolution
        self.rotate_tasks = rotate_tasks
        self.num_tasks = num_tasks

        # Load benchmark task
        benchmark_dict = benchmark_mod.get_benchmark_dict()
        self._suite = benchmark_dict[task_suite]()
        self._episode_counter = 0
        self.task_id = task_id
        self.init_state_idx = init_state_idx
        self._load_task(task_id)
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

    def _load_task(self, task_id: int) -> None:
        """(Re)create the underlying LIBERO env for the given task."""
        task = self._suite.get_task(task_id)
        self.task_id = task_id
        self.task_label = task.language
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        try:
            self.init_states = self._suite.get_task_init_states(task_id)
        finally:
            torch.load = _orig_load
        bddl = os.path.join(
            self._get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        # Close existing env if present
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = self._OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self._resolution,
            camera_widths=self._resolution,
        )
        self.env.seed(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, torch.Tensor]:
        self.step_count = 0
        # Rotate task_id / init_state_idx per episode if enabled
        if self.rotate_tasks:
            new_task = self._episode_counter % self.num_tasks
            new_init = (self._episode_counter // self.num_tasks) % len(self.init_states)
            if new_task != self.task_id:
                self._load_task(new_task)
            self.init_state_idx = new_init
        self._episode_counter += 1

        self.env.reset()
        # Clamp init_state_idx to available states (varies per task)
        idx = min(self.init_state_idx, len(self.init_states) - 1)
        raw_obs = self.env.set_init_state(self.init_states[idx])
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
        # Match training pipeline exactly: JPEG + lanczos3 resize + center crop
        agent_img = _resize_image_for_policy(agent_img, 224)
        wrist_img = _resize_image_for_policy(wrist_img, 224)
        agent_pil = _center_crop_image(Image.fromarray(agent_img))
        wrist_pil = _center_crop_image(Image.fromarray(wrist_img))

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

        # Pad prompt to a fixed length (so different tasks produce equal-sized
        # tensors that can be stacked in the rollout buffer).
        PAD_LEN = 48  # max prompt length across libero tasks is ~30-35 tokens
        pad_id = int(getattr(self.vla_config, "pad_token_id", 32000))
        if L_prompt < PAD_LEN:
            pad = torch.full((1, PAD_LEN - L_prompt), pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([pad, input_ids], dim=1)  # left-pad
            mask_pad = torch.zeros(1, PAD_LEN - L_prompt, dtype=attention_mask.dtype)
            attention_mask = torch.cat([mask_pad, attention_mask], dim=1)
        elif L_prompt > PAD_LEN:
            # Truncate from left (keep the tail of the prompt)
            input_ids = input_ids[:, -PAD_LEN:]
            attention_mask = attention_mask[:, -PAD_LEN:]
        L_prompt = PAD_LEN

        # Append placeholder action tokens (will be filled with mask tokens)
        mask_tokens = torch.full((1, n_act), self.mask_token_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, mask_tokens], dim=1)

        # Extend attention mask
        mask_ext = torch.ones(1, n_act, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, mask_ext], dim=1)

        # Labels: IGNORE_INDEX everywhere, action positions get mask_token_id
        L = input_ids.shape[1]
        labels = torch.full((1, L), IGNORE_INDEX, dtype=input_ids.dtype)
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
