"""VLA-DFM model server adapter for allenai/vla-evaluation-harness.

This file is the single integration point between VLA-DFM and the
vla-evaluation-harness WebSocket evaluation protocol.

It wraps the existing get_vla() + get_vla_action() inference path (from
experiments/robot/openvla_utils.py) behind the PredictModelServer interface
expected by the harness, translating observation and action formats.

Usage
-----
Start the server (from repo root, with VLA-DFM env active and vla-eval installed):

    python DiscreteDiffusionVLA/experiments/robot/vla_eval/vla_dfm_server.py \\
        --pretrained_checkpoint /path/to/checkpoint \\
        --unnorm_key libero_spatial \\
        --use_discrete_flow_matching \\
        --chunk_size 8 \\
        --port 8000

Or via the harness CLI:

    vla-eval serve --config configs/vla_eval/vla_dfm_libero.yaml

Then run the benchmark (in a separate terminal or machine):

    vla-eval run --config third_party/vla-evaluation-harness/configs/libero_90.yaml \\
                 --server-url ws://localhost:8000

Observation format (harness → this adapter)
-------------------------------------------
    obs["images"]           : dict[str, np.ndarray] – camera images [H,W,3] uint8
    obs["task_description"] : str – natural-language task instruction
    obs["state"]            : np.ndarray – proprioceptive state (optional)

    Camera key convention (harness side):
        Primary camera : first value in obs["images"]
        Wrist camera   : key containing "wrist" (if model uses num_images_in_input=2)

Action format (this adapter → harness)
---------------------------------------
    {"actions": np.ndarray}  shape [action_dim] or [chunk_size, action_dim]

    VLA-DFM returns a chunk of actions; this adapter stacks them so the
    harness's built-in ActionChunkBuffer can serve them one step at a time.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap: make VLA-DFM packages importable when run as a standalone
# script (i.e. without the caller setting PYTHONPATH).
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ROBOT_DIR = _THIS_DIR.parent              # experiments/robot/
_DFM_ROOT = _ROBOT_DIR.parent.parent       # DiscreteDiffusionVLA/

# DFM root (prismatic/) needs to be importable — insert at front.
if str(_DFM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DFM_ROOT))
# experiments/robot/ must be APPENDED, not inserted, so it cannot shadow the
# installed `vla_eval` package (our local experiments/robot/vla_eval/ dir
# would win over the harness package if placed at sys.path[0]).
if str(_ROBOT_DIR) not in sys.path:
    sys.path.append(str(_ROBOT_DIR))

# vla-eval must be installed (pip install -e third_party/vla-evaluation-harness)
from vla_eval.model_servers.base import SessionContext  # noqa: E402
from vla_eval.model_servers.predict import PredictModelServer  # noqa: E402
from vla_eval.types import Action, Observation  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal config shim: mirrors the draccus config interface expected by
# openvla_utils functions, populated with sensible VLA-DFM defaults.
# ---------------------------------------------------------------------------

def _make_cfg(**overrides: Any) -> SimpleNamespace:
    """Build a SimpleNamespace that looks like a draccus GenerateConfig."""
    defaults: dict[str, Any] = {
        # --- model loading ---
        "pretrained_checkpoint": "",
        "sync_model_logic": False,
        "action_vocab_anchor": None,
        "action_token_begin_idx": None,
        "legacy_eval_mode": False,
        # quantization (False = load in full precision)
        "load_in_8bit": False,
        "load_in_4bit": False,
        # --- inference ---
        "unnorm_key": None,
        "num_images_in_input": 1,
        "use_proprio": False,
        "center_crop": False,
        # --- optional model components (disabled by default for plain eval) ---
        "use_film": False,
        "lora_rank": 32,  # only used when use_film=True
        # --- action head type ---
        "use_l1_regression": False,
        "use_diffusion": False,
        "num_diffusion_steps_train": 100,
        "num_diffusion_steps_inference": 10,
        # --- decoding mode ---
        "use_discrete_flow_matching": False,
        "use_discrete_diffusion": False,
        # --- DFM hyperparameters (None/"auto" → resolved from checkpoint) ---
        "dfm_num_steps": None,
        "dfm_maskgit_num_steps": None,
        "dfm_schedule": "auto",
        "dfm_maskgit_schedule": "auto",
        "dfm_temperature": 1.0,
        "dfm_temperature_anneal": "none",
        "dfm_adaptive_step": True,
        "dfm_step_min": -1.0,
        "dfm_step_max": -1.0,
        "dfm_time_eps": -1.0,
        "dfm_early_exit": True,
        "dfm_early_exit_frac": 0.0,
        "dfm_corrector": False,
        "dfm_corrector_iters": 1,
        "dfm_corrector_remask_frac": 0.1,
        "dfm_clamp_values": None,
        "dfm_clamp_mask": False,
        # fall back to checkpoint config for any unset DFM params
        "use_checkpoint_defaults": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Model server
# ---------------------------------------------------------------------------

class VLADFMModelServer(PredictModelServer):
    """Adapts VLA-DFM to the vla-evaluation-harness PredictModelServer API.

    The model is loaded lazily on the first predict() call so that the
    WebSocket server starts immediately and the heavy GPU load is deferred.

    Args:
        pretrained_checkpoint:
            Local directory or HF Hub ID of the VLA-DFM checkpoint.
            Must contain config.json and checkpoint files.
        unnorm_key:
            Dataset key used to un-normalise actions (e.g. "libero_spatial").
            If None, auto-detected from the checkpoint's norm_stats dict.
        use_discrete_flow_matching:
            Enable DFM / CTMC decoding.  Should match the training mode.
        dfm_decode_mode:
            "ctmc" (default, recommended) or "maskgit".
        dfm_num_steps:
            CTMC integration steps.  0 = use checkpoint default.
        dfm_maskgit_num_steps:
            MaskGIT refinement iterations (used when dfm_decode_mode="maskgit").
        num_images_in_input:
            1 = primary camera only (default).
            2 = primary + wrist camera (model must have been trained with wrist).
        use_proprio:
            Pass obs["state"] as proprioceptive input.  Requires the checkpoint
            to have a proprio projector.
        center_crop:
            Apply center-crop preprocessing (matches training if used).
        chunk_size:
            Number of actions returned per inference call.  The harness
            ActionChunkBuffer will serve them one step at a time.
            Defaults to the model's NUM_ACTIONS_CHUNK (typically 8 for LIBERO).
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        unnorm_key: str | None = None,
        *,
        use_discrete_flow_matching: bool = False,
        dfm_decode_mode: str = "ctmc",
        dfm_num_steps: int = 0,
        dfm_maskgit_num_steps: int = 0,
        num_images_in_input: int = 1,
        use_proprio: bool = False,
        center_crop: bool = False,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, **kwargs)

        self.pretrained_checkpoint = pretrained_checkpoint
        self.unnorm_key = unnorm_key
        self.use_discrete_flow_matching = use_discrete_flow_matching
        self.dfm_decode_mode = dfm_decode_mode
        self.dfm_num_steps = dfm_num_steps
        self.dfm_maskgit_num_steps = dfm_maskgit_num_steps
        self.num_images_in_input = num_images_in_input
        self.use_proprio = use_proprio
        self.center_crop = center_crop

        # Populated on first predict()
        self._vla: Any = None
        self._processor: Any = None
        self._cfg: SimpleNamespace | None = None

    def get_observation_params(self) -> dict:
        """Tell harness/benchmark which observation fields we need.

        For SimplerEnv this controls whether `obs["states"]` (proprio) and the
        wrist image get populated; without this we'd silently get only RGB and
        the proprio_projector would receive zeros (= severe OOD for any model
        trained with proprio).
        """
        params: dict = {}
        if self.use_proprio:
            params["send_state"] = True
        if self.num_images_in_input > 1:
            params["send_wrist_image"] = True
        return params

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load VLA and processor on first call (no-op after that)."""
        if self._vla is not None:
            return

        import os
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # type: ignore[import]
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # type: ignore[import]
        from openvla_utils import DEVICE, _load_dataset_stats, get_processor  # type: ignore[import]

        # -----------------------------------------------------------------
        # Resolve checkpoint to a local directory.
        # get_vla() from openvla_utils only registers our local model classes
        # (with VLA-DFM additions like _action_vocab_range) when the path is
        # LOCAL.  For HF Hub IDs it loads the remote-cached code, which is the
        # original OpenVLA and lacks those additions.
        # We bypass get_vla() and load directly with trust_remote_code=False +
        # our registered local class to ensure the VLA-DFM class is always used.
        # -----------------------------------------------------------------
        checkpoint = self.pretrained_checkpoint
        if not os.path.isdir(checkpoint):
            logger.info("Downloading %s to local cache …", checkpoint)
            checkpoint = snapshot_download(checkpoint)
            logger.info("Resolved to local snapshot: %s", checkpoint)

        # Register VLA-DFM local classes so AutoModelForVision2Seq uses them.
        try:
            AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
        except TypeError:
            AutoConfig.register("openvla", OpenVLAConfig)
        try:
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)
        except TypeError:
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Load config with trust_remote_code=False so AutoConfig uses our registered
        # local OpenVLAConfig (not the remote-cached version from transformers_modules/).
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=False)

        logger.info("Loading model weights from %s …", checkpoint)
        vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,  # use registered local OpenVLAForActionPrediction
            low_cpu_mem_usage=True,
        )
        vla.eval()
        vla = vla.to(DEVICE)

        # Load action-normalisation statistics.
        _load_dataset_stats(vla, checkpoint)

        # Set multi-image mode (primary camera + optional wrist).
        vla.vision_backbone.set_num_images_in_input(self.num_images_in_input)

        # FiLM: if a `vision_backbone--*_checkpoint.pt` lives next to the model,
        # the model was trained with FiLM and the eval forward pass needs the
        # same wrapped backbone (otherwise FiLM weights are silently ignored).
        import glob
        vb_glob = glob.glob(os.path.join(checkpoint, "vision_backbone--*_checkpoint.pt"))
        film_checkpoint = vb_glob[0] if vb_glob else None
        if film_checkpoint is not None:
            from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
            logger.info("Detected FiLM checkpoint at %s — wrapping vision backbone.", film_checkpoint)
            vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
                vision_backbone=vla.model.vision_backbone,
                llm_dim=vla.llm_dim,
            )
            vla.model.vision_backbone.load_state_dict(torch.load(film_checkpoint, map_location="cpu"))
            vla.model.vision_backbone = vla.model.vision_backbone.to(DEVICE)
            vla.model.vision_backbone.set_num_images_in_input(self.num_images_in_input)

        self._vla = vla
        self._processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

        # Read legacy_dfm_mode from checkpoint config
        ckpt_anchor = getattr(config, "action_vocab_anchor", None)
        ckpt_legacy_train = getattr(config, "legacy_train_mode", False)

        # Build the config shim for get_vla_action()
        self._cfg = _make_cfg(
            pretrained_checkpoint=checkpoint,
            unnorm_key=self.unnorm_key,
            use_discrete_flow_matching=self.use_discrete_flow_matching,
            dfm_num_steps=self.dfm_num_steps or None,
            dfm_maskgit_num_steps=self.dfm_maskgit_num_steps or None,
            num_images_in_input=self.num_images_in_input,
            use_proprio=self.use_proprio,
            center_crop=self.center_crop,
            use_film=film_checkpoint is not None,
            action_vocab_anchor=ckpt_anchor,
            legacy_eval_mode=bool(ckpt_legacy_train or ckpt_anchor == "legacy"),
        )

        # Auto-detect unnorm_key if not provided
        if self._cfg.unnorm_key is None:
            norm_stats = getattr(self._vla, "norm_stats", {})
            if not norm_stats:
                raise RuntimeError(
                    "unnorm_key not provided and checkpoint has no norm_stats. "
                    "Pass --unnorm_key explicitly."
                )
            self._cfg.unnorm_key = next(iter(norm_stats))
            logger.info("unnorm_key auto-detected: %s", self._cfg.unnorm_key)

        logger.info(
            "VLA-DFM ready. unnorm_key=%s  dfm=%s  decode_mode=%s",
            self._cfg.unnorm_key,
            self.use_discrete_flow_matching,
            self.dfm_decode_mode,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        """Run one forward pass and return an action (or action chunk).

        The harness calls this on every observation step.  If chunk_size > 1
        the first call returns a 2-D array [chunk_size, action_dim] and the
        PredictModelServer base class buffers and serves one action at a time.
        """
        self._load_model()
        assert self._vla is not None and self._cfg is not None

        from openvla_utils import get_vla_action  # type: ignore[import]

        # --- Translate harness observation → VLA-DFM obs dict ---
        images_dict = obs.get("images", {})
        if isinstance(images_dict, dict):
            image_values = list(images_dict.values())
        else:
            # Fallback: bare array
            image_values = [images_dict]

        if not image_values:
            raise ValueError("No images found in observation dict.")

        dfm_obs: dict[str, Any] = {"full_image": image_values[0]}

        # Second image: wrist camera (only if model was trained with it)
        if self.num_images_in_input > 1:
            # Prefer a key containing "wrist"; fall back to the second image.
            wrist_img = None
            if isinstance(images_dict, dict):
                for k, v in images_dict.items():
                    if "wrist" in k.lower():
                        wrist_img = v
                        break
            if wrist_img is None and len(image_values) > 1:
                wrist_img = image_values[1]
            if wrist_img is None:
                wrist_img = image_values[0]
                logger.warning(
                    "num_images_in_input=%d but no wrist image; duplicating primary.",
                    self.num_images_in_input,
                )
            dfm_obs["wrist_image"] = wrist_img

        # Proprioception. Different benchmarks use different keys / formats:
        #   LIBERO: obs["state"] (already in xyz+euler+gripper format, matches training)
        #   SimplerEnv: obs["states"] (8D: pos3 + quat_wxyz4 + gripper1) — convert to 7D
        if self.use_proprio:
            from prismatic.vla.constants import PROPRIO_DIM
            state = None
            if "state" in obs:
                state = np.asarray(obs["state"], dtype=np.float32)
            elif "states" in obs:
                s = np.asarray(obs["states"], dtype=np.float32).flatten()
                if s.shape[0] == 8 and PROPRIO_DIM == 7:
                    # SimplerEnv 8D -> Bridge/LIBERO 7D: pos(3) + quat_wxyz(4) + gripper(1) -> pos(3) + euler(3) + gripper(1)
                    from scipy.spatial.transform import Rotation as R
                    pos, quat_wxyz, grip = s[:3], s[3:7], s[7:8]
                    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
                    euler = R.from_quat(quat_xyzw).as_euler("xyz").astype(np.float32)
                    state = np.concatenate([pos, euler, grip], axis=0)
                else:
                    state = s
            if state is None:
                logger.warning("No proprio in obs (keys=%s); falling back to zeros.", list(obs.keys()))
                state = np.zeros(PROPRIO_DIM, dtype=np.float32)
            dfm_obs["state"] = state

        task_label: str = obs.get("task_description", "")

        # --- Call existing inference utility ---
        # get_vla_action returns List[np.ndarray], one array per chunk step.
        actions_list: list[np.ndarray] = get_vla_action(
            cfg=self._cfg,
            vla=self._vla,
            processor=self._processor,
            obs=dfm_obs,
            task_label=task_label,
            use_discrete_flow_matching=self.use_discrete_flow_matching,
            dfm_decode_mode=self.dfm_decode_mode,
        )

        # --- Translate output → harness Action dict ---
        # Single action (chunk_size=1): return 1-D array so PredictModelServer
        # bypasses the chunk buffer and sends it immediately.
        # Multi-step chunk: return 2-D array; PredictModelServer buffers it.
        if len(actions_list) == 1:
            actions = np.asarray(actions_list[0], dtype=np.float32)
        else:
            actions = np.stack(actions_list, axis=0).astype(np.float32)

        return {"actions": actions}


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(VLADFMModelServer)
