"""
rl_finetune.py

Entry-point for DFM-VLA RL fine-tuning (Algorithm 1).

Loads a pretrained DFM OpenVLA checkpoint (e.g. from finetune.py),
attaches a PPO ratio network and value head, then runs the
Discrete Flow Matching RL fine-tuning loop.

Usage:
    # From a locally saved DFM checkpoint:
    python vla-scripts/rl_finetune.py \
        --vla_path runs/my_dfm_run/checkpoint-50000 \
        --dataset_name libero_spatial \
        --data_root_dir datasets/rlds \
        --num_iterations 100

    # With wandb logging:
    python vla-scripts/rl_finetune.py \
        --vla_path runs/my_dfm_run/checkpoint-50000 \
        --use_wandb True \
        --wandb_project my-rl-project
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.rl.trainer import DFMRLTrainer, RLFinetuneConfig
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:
    LoraConfig = PeftModel = get_peft_model = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Register custom model class
AutoConfig.register("openvla", OpenVLAConfig)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RLFinetuneEntryConfig:
    # Pre-trained DFM checkpoint to start from
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "libero_spatial"

    # Environment
    use_dummy_env: bool = False      # True to use DummyLiberoEnv (for testing)
    task_suite: str = "libero_spatial"
    task_id: int = 0
    env_resolution: int = 256

    # RL hyperparameters (forwarded to RLFinetuneConfig)
    num_iterations: int = 100
    rollout_steps: int = 50
    rollout_episodes: int = 5
    max_transitions: int = 200
    batch_size: int = 8
    ppo_clip_eps: float = 0.2
    lambda_constraint: float = 1.0
    ppo_epochs: int = 4
    ppo_lr: float = 3e-4
    dfm_epochs: int = 4
    dfm_lr: float = 5e-6
    dfm_grad_clip: float = 1.0

    # DFM schedule
    dfm_schedule: str = "cosine"
    dfm_time_eps: float = 1e-3
    dfm_t_min: float = 0.0
    dfm_t_max: float = 1.0
    dfm_t_bias_alpha: float = 1.0
    dfm_weight_clip: float = 20.0

    # Advantage estimation
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Inference
    maskgit_num_steps: int = 12
    maskgit_schedule: str = "cosine"

    # Memory
    update_mini_batch: int = 4

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    lora_adapter_dir: Optional[str] = None  # path to lora_adapter/ for unmerged ckpt

    # Checkpointing
    run_root_dir: Path = Path("runs")
    save_interval: int = 25
    log_interval: int = 10
    resume: Optional[str] = None

    # Hardware
    torch_dtype: str = "bfloat16"

    # Logging
    use_wandb: bool = False
    wandb_project: str = "dfm-rl"


# ---------------------------------------------------------------------------
# Dummy environment (replace with real robot or simulation env)
# ---------------------------------------------------------------------------

class DummyLiberoEnv:
    """
    Placeholder environment matching the Trainer's expected API.

    Replace this with the real LIBERO / DROID / ALOHA environment.
    The environment must implement:
        obs = reset() -> dict with keys: input_ids, attention_mask,
                         pixel_values, labels, action_positions_mask
        obs, reward, done, info = step(action_cont)
    """

    def __init__(self, batch_size: int = 4, seq_len: int = 512,
                 img_size: int = 224, vocab_size: int = 32064):
        self.B = batch_size
        self.L = seq_len
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.step_count = 0
        self.max_steps = 50

        # Fixed action token range for dummy env
        self.action_begin = vocab_size - 257
        self.action_end = vocab_size - 1
        self.mask_token_id = vocab_size - 1

    def reset(self):
        self.step_count = 0
        return self._make_obs()

    def step(self, action_cont):
        self.step_count += 1
        obs = self._make_obs()
        reward = torch.zeros(self.B)
        done = torch.tensor([self.step_count >= self.max_steps] * self.B)
        if done.any():
            reward = (torch.rand(self.B) > 0.7).float()  # 30% success
        info = {"success": reward > 0.5}
        return obs, reward, done, info

    def _make_obs(self):
        B, L = self.B, self.L
        n_act = NUM_ACTIONS_CHUNK * ACTION_DIM

        input_ids = torch.randint(0, self.vocab_size, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        pixel_values = torch.rand(B, 3, self.img_size, self.img_size)

        # Fill action positions with plausible token ids (must be valid action
        # tokens in input_ids too, not just labels, so dfm_gkl_loss indexing
        # stays in-bounds after apply_mask_flow_matching).
        action_toks = torch.randint(self.action_begin, self.action_end, (B, n_act))
        input_ids[:, -n_act:] = action_toks
        labels = input_ids.clone()
        labels[:, :-n_act] = -100

        action_pos_mask = torch.zeros(B, L, dtype=torch.bool)
        action_pos_mask[:, -n_act:] = True

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            action_positions_mask=action_pos_mask,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: RLFinetuneEntryConfig) -> None:
    # ── Device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        cfg.torch_dtype, torch.float32
    )

    # ── Load pretrained DFM model ────────────────────────────────────────────
    logger.info("Loading DFM-VLA model from: %s", cfg.vla_path)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # ── Set vision backbone to multi-image mode ──────────────────────────────
    # Matches training/eval convention: num_images_in_input=2 (agentview + wrist)
    if hasattr(vla, "vision_backbone"):
        vla.vision_backbone.set_num_images_in_input(2)
        logger.info("Set vision_backbone num_images_in_input=2")

    # ── Inject fine-tuning dataset statistics into norm_stats ────────────────
    dataset_stats_path = os.path.join(cfg.vla_path, "dataset_statistics.json")
    if os.path.exists(dataset_stats_path):
        import json
        with open(dataset_stats_path) as f:
            ft_stats = json.load(f)
        vla.norm_stats.update(ft_stats)
        logger.info("Injected dataset statistics: %s", list(ft_stats.keys()))

    # ── Load LoRA adapter (unmerged checkpoint) ─────────────────────────────
    if cfg.lora_adapter_dir is not None:
        assert PeftModel is not None, "peft is required to load LoRA adapter. pip install peft"
        logger.info("Loading LoRA adapter from: %s", cfg.lora_adapter_dir)
        vla = PeftModel.from_pretrained(vla, cfg.lora_adapter_dir)
        vla = vla.merge_and_unload()
        logger.info("LoRA adapter merged into base model.")

    # ── Load proprio projector if available ──────────────────────────────────
    proprio_projector = None
    proprio_ckpt = None
    import glob as _glob
    for pattern in ["proprio_projector*.pt", "**/proprio_projector*.pt"]:
        matches = _glob.glob(os.path.join(cfg.vla_path, pattern), recursive=True)
        if matches:
            proprio_ckpt = matches[0]
            break
    if proprio_ckpt is not None:
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.constants import PROPRIO_DIM
        llm_dim = vla.config.text_config.hidden_size
        proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
        proprio_projector = proprio_projector.to(device=device, dtype=torch_dtype)
        state = torch.load(proprio_ckpt, map_location=device, weights_only=False)
        # Handle potential "module." prefix from DDP training
        state = {k.replace("module.", ""): v for k, v in state.items()}
        proprio_projector.load_state_dict(state)
        proprio_projector.eval()
        logger.info("Loaded proprio_projector from: %s", proprio_ckpt)

    # ── Apply fresh LoRA for RL fine-tuning ─────────────────────────────────
    if cfg.use_lora:
        assert get_peft_model is not None, "peft is required for LoRA. pip install peft"
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
            # DFM needs trainable embed_tokens + lm_head for mask token.
            modules_to_save=["embed_tokens", "lm_head"],
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # ── Build RL trainer config ──────────────────────────────────────────────
    rl_cfg = RLFinetuneConfig(
        num_iterations=cfg.num_iterations,
        rollout_steps=cfg.rollout_steps,
        rollout_episodes=cfg.rollout_episodes,
        max_transitions=cfg.max_transitions,
        batch_size=cfg.batch_size,
        ppo_clip_eps=cfg.ppo_clip_eps,
        lambda_constraint=cfg.lambda_constraint,
        ppo_epochs=cfg.ppo_epochs,
        ppo_lr=cfg.ppo_lr,
        dfm_epochs=cfg.dfm_epochs,
        dfm_lr=cfg.dfm_lr,
        dfm_grad_clip=cfg.dfm_grad_clip,
        dfm_schedule=cfg.dfm_schedule,
        dfm_time_eps=cfg.dfm_time_eps,
        dfm_t_min=cfg.dfm_t_min,
        dfm_t_max=cfg.dfm_t_max,
        dfm_t_bias_alpha=cfg.dfm_t_bias_alpha,
        dfm_weight_clip=cfg.dfm_weight_clip,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        maskgit_num_steps=cfg.maskgit_num_steps,
        maskgit_schedule=cfg.maskgit_schedule,
        unnorm_key=cfg.dataset_name,
        update_mini_batch=cfg.update_mini_batch,
        save_interval=cfg.save_interval,
        log_interval=cfg.log_interval,
        save_dir=str(cfg.run_root_dir / "rl_dfm"),
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
    )

    # ── Environment ──────────────────────────────────────────────────────────
    if cfg.use_dummy_env:
        def env_fn():
            return DummyLiberoEnv(batch_size=cfg.batch_size)
    else:
        from prismatic.rl.libero_env import LiberoRLEnv
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        def env_fn():
            return LiberoRLEnv(
                task_suite=cfg.task_suite,
                task_id=cfg.task_id,
                processor=processor,
                vla_config=vla.config,
                unnorm_key=cfg.dataset_name,
                resolution=cfg.env_resolution,
            )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = DFMRLTrainer(
        vla_model=vla,
        env_fn=env_fn,
        cfg=rl_cfg,
        device=device,
        proprio_projector=proprio_projector,
    )

    if cfg.resume:
        trainer.load_checkpoint(cfg.resume)

    trainer.train()


if __name__ == "__main__":
    main()
