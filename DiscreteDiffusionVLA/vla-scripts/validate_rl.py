"""
Evaluate an RL-finetuned checkpoint using the same pipeline as validate_env.py.

Loads the supervised base model + applies RL LoRA weights, then runs eval
via LiberoRLEnv to directly compare against the supervised baseline (80%).
"""
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass
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
class Cfg:
    vla_path: str = ""
    rl_ckpt: str = ""
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
def main(cfg: Cfg):
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Load base model
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device)
    stats_path = os.path.join(cfg.vla_path, "dataset_statistics.json")
    if os.path.exists(stats_path):
        vla.norm_stats.update(json.load(open(stats_path)))
    vla.vision_backbone.set_num_images_in_input(2)

    # Apply LoRA (same config as training)
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules="all-linear", init_lora_weights="gaussian",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    vla = get_peft_model(vla, lora_config)

    # Load RL state into LoRA model
    logger.info("Loading RL checkpoint: %s", cfg.rl_ckpt)
    rl_state = torch.load(cfg.rl_ckpt, map_location=device, weights_only=False)
    missing, unexpected = vla.load_state_dict(rl_state["vla_state"], strict=False)
    logger.info(f"Loaded RL state: {len(rl_state['vla_state'])} keys "
                f"(missing {len(missing)} base keys, unexpected {len(unexpected)})")
    vla.eval()

    # Load proprio_projector
    proprio_projector = None
    for pattern in ["proprio_projector*.pt", "**/proprio_projector*.pt"]:
        matches = glob.glob(os.path.join(cfg.vla_path, pattern), recursive=True)
        if matches:
            from prismatic.models.projectors import ProprioProjector
            from prismatic.vla.constants import PROPRIO_DIM
            proprio_projector = ProprioProjector(
                llm_dim=vla.config.text_config.hidden_size, proprio_dim=PROPRIO_DIM,
            ).to(device=device, dtype=dtype)
            state = torch.load(matches[0], map_location=device, weights_only=False)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            proprio_projector.load_state_dict(state)
            proprio_projector.eval()
            logger.info("Loaded proprio_projector")
            break

    from prismatic.rl.libero_env import LiberoRLEnv
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    proprio_stats = vla.norm_stats.get(cfg.dataset_name, {}).get("proprio")

    total_successes = 0
    total_trials = 0
    per_task = {}
    for task_id in range(cfg.num_tasks):
        env = LiberoRLEnv(
            task_suite=cfg.task_suite, task_id=task_id,
            processor=processor, vla_config=vla.config,
            unnorm_key=cfg.dataset_name, proprio_norm_stats=proprio_stats,
        )
        task_successes = 0
        for trial in range(cfg.num_trials):
            env.init_state_idx = trial
            obs = env.reset()
            succeeded = False
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
            n_act = NUM_ACTIONS_CHUNK * ACTION_DIM
            max_chunks = env.max_steps // 8 + 1
            for _ in range(max_chunks):
                pv = obs["pixel_values"].to(device=device, dtype=dtype)
                pr = obs.get("proprio")
                if pr is not None:
                    pr = pr.to(device=device, dtype=dtype)
                prompt_ids = obs["input_ids"][:, :-n_act].to(device)
                prompt_mask = obs["attention_mask"][:, :-n_act].to(device)

                with torch.inference_mode():
                    # Access base model for predict_action
                    base = vla.base_model.model if hasattr(vla, "base_model") else vla
                    actions_np, _ = base.predict_action(
                        input_ids=prompt_ids, attention_mask=prompt_mask,
                        pixel_values=pv, proprio=pr,
                        proprio_projector=proprio_projector,
                        unnorm_key=cfg.dataset_name,
                        use_discrete_flow_matching=True,
                        dfm_decode_mode=cfg.dfm_decode_mode,
                        dfm_num_steps=cfg.dfm_num_steps,
                        dfm_maskgit_num_steps=cfg.maskgit_num_steps,
                        dfm_schedule=cfg.dfm_schedule,
                        dfm_temperature=cfg.dfm_temperature,
                    )

                a = np.asarray(actions_np)
                if a.ndim == 2:
                    a = a.reshape(1, a.shape[0], a.shape[1])
                obs, reward, done, info = env.step(torch.as_tensor(a, dtype=torch.float32))
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
        logger.info(f"=== Task {task_id}: {task_successes}/{cfg.num_trials}")

    logger.info("=" * 60)
    for tid, r in per_task.items():
        logger.info(f"  {tid}: {r}")
    logger.info(f"RL OVERALL: {total_successes}/{total_trials} = "
                f"{100*total_successes/max(total_trials,1):.1f}%")


if __name__ == "__main__":
    main()
