"""
validate_env.py

Pure-inference validation of LiberoRLEnv: does our RL env reproduce the
~80% success rate that the official eval script gets on the same checkpoint?

If YES → env is correct, RL logic must have a bug.
If NO  → env is wrong (image pipeline, proprio, gripper, etc.).

Usage:
    python vla-scripts/validate_env.py \
        --vla_path <ckpt> \
        --dataset_name libero_object_no_noops \
        --task_suite libero_object \
        --num_tasks 10 --num_trials 2
"""
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AutoConfig.register("openvla", OpenVLAConfig)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


@dataclass
class ValidateConfig:
    vla_path: str = ""
    dataset_name: str = "libero_object_no_noops"
    task_suite: str = "libero_object"
    num_tasks: int = 10
    num_trials: int = 2
    maskgit_num_steps: int = 12
    dfm_num_steps: int = 128
    dfm_schedule: str = "cosine"
    dfm_temperature: float = 1.0
    dfm_decode_mode: str = "ctmc"


@draccus.wrap()
def main(cfg: ValidateConfig) -> None:
    # Patch torch.load for LIBERO init states (PyTorch 2.6+)
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    logger.info("Loading VLA: %s", cfg.vla_path)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device)

    # Inject fine-tune stats
    stats_path = os.path.join(cfg.vla_path, "dataset_statistics.json")
    if os.path.exists(stats_path):
        ft_stats = json.load(open(stats_path))
        vla.norm_stats.update(ft_stats)
        logger.info("Loaded fine-tune stats: %s", list(ft_stats.keys()))

    # Set vision backbone to multi-image mode
    vla.vision_backbone.set_num_images_in_input(2)

    # Load proprio_projector
    import glob as _glob
    proprio_projector = None
    for pattern in ["proprio_projector*.pt", "**/proprio_projector*.pt"]:
        matches = _glob.glob(os.path.join(cfg.vla_path, pattern), recursive=True)
        if matches:
            from prismatic.models.projectors import ProprioProjector
            from prismatic.vla.constants import PROPRIO_DIM
            proprio_projector = ProprioProjector(
                llm_dim=vla.config.text_config.hidden_size,
                proprio_dim=PROPRIO_DIM,
            ).to(device=device, dtype=dtype)
            state = torch.load(matches[0], map_location=device, weights_only=False)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            proprio_projector.load_state_dict(state)
            proprio_projector.eval()
            logger.info("Loaded proprio_projector")
            break

    # Build env
    from prismatic.rl.libero_env import LiberoRLEnv
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    proprio_stats = None
    if cfg.dataset_name in vla.norm_stats:
        proprio_stats = vla.norm_stats[cfg.dataset_name].get("proprio")

    total_successes = 0
    total_trials = 0
    per_task = {}

    for task_id in range(cfg.num_tasks):
        env = LiberoRLEnv(
            task_suite=cfg.task_suite,
            task_id=task_id,
            processor=processor,
            vla_config=vla.config,
            unnorm_key=cfg.dataset_name,
            proprio_norm_stats=proprio_stats,
        )
        task_successes = 0
        for trial in range(cfg.num_trials):
            env.init_state_idx = trial  # rotate through init states
            obs = env.reset()
            succeeded = False
            # Run up to max_steps // 8 chunks (chunk = 8 env steps)
            max_chunks = env.max_steps // 8 + 1
            for chunk_i in range(max_chunks):
                pv = obs["pixel_values"].to(device=device, dtype=dtype)
                pr = obs.get("proprio")
                if pr is not None:
                    pr = pr.to(device=device, dtype=dtype)

                # predict_action adds its own action placeholders, so we
                # must pass ONLY the prompt portion of input_ids (without the
                # mask tokens that our env appends for RL forward calls).
                from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
                n_act = NUM_ACTIONS_CHUNK * ACTION_DIM
                prompt_ids = obs["input_ids"][:, :-n_act]  # strip trailing mask tokens
                prompt_mask = obs["attention_mask"][:, :-n_act]

                with torch.inference_mode():
                    actions_np, _ = vla.predict_action(
                        input_ids=prompt_ids.to(device),
                        attention_mask=prompt_mask.to(device),
                        pixel_values=pv,
                        proprio=pr,
                        proprio_projector=proprio_projector,
                        unnorm_key=cfg.dataset_name,
                        use_discrete_flow_matching=True,
                        dfm_decode_mode=cfg.dfm_decode_mode,
                        dfm_num_steps=cfg.dfm_num_steps,
                        dfm_maskgit_num_steps=cfg.maskgit_num_steps,
                        dfm_schedule=cfg.dfm_schedule,
                        dfm_temperature=cfg.dfm_temperature,
                    )

                # Shape to (1, chunk_len, action_dim)
                a = np.asarray(actions_np)
                if a.ndim == 2:
                    a = a.reshape(1, a.shape[0], a.shape[1])
                act_tensor = torch.as_tensor(a, dtype=torch.float32)

                obs, reward, done, info = env.step(act_tensor)
                if float(info.get("success", torch.tensor(0.0)).item()) > 0.5:
                    succeeded = True
                    break
                if done.any():
                    break

            total_trials += 1
            if succeeded:
                task_successes += 1
                total_successes += 1
            logger.info(f"Task {task_id} trial {trial}: {'SUCCESS' if succeeded else 'fail'}")

        per_task[task_id] = f"{task_successes}/{cfg.num_trials}"
        logger.info(f"=== Task {task_id} ({env.task_label[:50]}): {task_successes}/{cfg.num_trials}")

    logger.info("=" * 60)
    logger.info("PER TASK:")
    for tid, result in per_task.items():
        logger.info(f"  {tid}: {result}")
    logger.info(f"OVERALL: {total_successes}/{total_trials} = {100*total_successes/max(total_trials,1):.1f}%")


if __name__ == "__main__":
    main()
