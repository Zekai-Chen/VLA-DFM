"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""

import os
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import numpy as np
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.action_vocab import resolve_action_vocab, validate_action_vocab_alignment
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 256_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur) 100_000 

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    use_discrete_diffusion: bool = True             # If True, uses discrete diffusion (instead of continuous) for action generation
    use_discrete_flow_matching: bool = False        # If True, uses discrete flow matching for action generation
    legacy_train_mode: Optional[bool] = None        # If True, use legacy prompt/tokenization/masks (forced for discrete diffusion)
    legacy_dfm_mode: bool = False                   # If True, use legacy prompt/tokenization/masks for DFM (opt-in)

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    torch_dtype: str = "bfloat16"                    # bfloat16 | float16 | float32
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # DFM configuration
    dfm_schedule: str = "cosine"                     # Schedule for kappa(t)
    dfm_time_eps: float = 1e-3                       # Avoid t at endpoints
    dfm_t_min: float = 0.0                           # Min t for sampling
    dfm_t_max: float = 1.0                           # Max t for sampling (capped by 1 - dfm_time_eps)
    dfm_loss_mode: str = "generalized_kl"            # generalized_kl | masked_ce
    dfm_weight_clip: float = 20.0                    # Clamp kappa_dot/(1-kappa)
    dfm_train_mode: str = "flow"                     # flow | diffusion_like
    dfm_maskgit_num_steps: int = 12                  # MaskGIT iterations (for inference config)
    dfm_maskgit_schedule: str = "cosine"             # MaskGIT schedule (for inference config)
    dfm_t_bias_alpha: float = 1.0                    # Low-t bias for DFM sampling (>1 biases low-t)
    dfm_log_mask_stats: bool = False                # If True, emit per-batch DFM mask trace JSONL
    dfm_log_mask_every: int = 1                      # Log every N gradient steps
    dfm_log_mask_max_samples: int = 5000             # Max number of per-sample traces to write

    # fmt: on


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    """Resolve a string dtype name to a torch dtype."""
    normalized = dtype_str.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {dtype_str}")


def resolve_legacy_tokenization_mode(cfg: FinetuneConfig) -> Tuple[bool, Optional[bool]]:
    """Resolve legacy tokenization mode across DD/DFM and enforce configuration constraints."""
    legacy_train_mode_arg = cfg.legacy_train_mode
    if cfg.legacy_dfm_mode and not cfg.use_discrete_flow_matching:
        raise ValueError("legacy_dfm_mode is only supported when use_discrete_flow_matching=True.")
    if cfg.use_discrete_flow_matching and cfg.legacy_train_mode:
        raise ValueError("legacy_train_mode is only supported when use_discrete_diffusion=True.")

    if cfg.use_discrete_diffusion:
        if cfg.legacy_train_mode is False:
            print("[legacy_train] forcing legacy_train_mode=True for discrete diffusion training")
        cfg.legacy_train_mode = True
    else:
        if cfg.legacy_train_mode is None:
            cfg.legacy_train_mode = False
        elif cfg.legacy_train_mode:
            raise ValueError("legacy_train_mode is only supported when use_discrete_diffusion=True.")

    legacy_tokenization_mode = cfg.legacy_train_mode if cfg.use_discrete_diffusion else bool(cfg.legacy_dfm_mode)
    return legacy_tokenization_mode, legacy_train_mode_arg


def apply_legacy_tokenization_overrides(
    cfg: FinetuneConfig,
    model_config,
    processor,
    legacy_tokenization_mode: bool,
) -> None:
    """Force legacy config fields to match 2026-02-27 behavior for DD/DFM tokenization."""
    if not legacy_tokenization_mode:
        return
    if cfg.use_discrete_flow_matching:
        print("[legacy_dfm] applying legacy tokenization overrides (anchor=legacy)")
        if processor.tokenizer.mask_token_id is None:
            processor.tokenizer.add_special_tokens({"mask_token": "<mask>"})
        if processor.tokenizer.mask_token_id is None:
            raise ValueError(
                "legacy_dfm_mode requires a mask token in the tokenizer. "
                "Tried to add '<mask>' but mask_token_id is still None. "
                "Re-save the tokenizer with a mask token or use a base model that includes one."
            )
    else:
        print("[legacy_train] applying legacy tokenization overrides (anchor=legacy)")
    model_config.legacy_train_mode = True
    model_config.legacy_eval_mode = True
    model_config.action_vocab_anchor = "legacy"
    model_config.action_token_begin_idx = int(ACTION_TOKEN_BEGIN_IDX)
    n_bins = int(getattr(model_config, "n_action_bins", 256))
    expected_begin = int(processor.tokenizer.vocab_size - (n_bins + 1))
    if expected_begin != int(ACTION_TOKEN_BEGIN_IDX):
        raise ValueError(
            "legacy tokenization requires ACTION_TOKEN_BEGIN_IDX alignment. "
            f"Expected begin={expected_begin} from vocab_size and n_bins, "
            f"but ACTION_TOKEN_BEGIN_IDX={ACTION_TOKEN_BEGIN_IDX}. "
            "Update the tokenizer/vocab or constants for legacy training."
        )


def apply_legacy_dd_overrides(cfg, model_config, processor) -> None:
    """Backward-compatible wrapper for legacy DD overrides."""
    apply_legacy_tokenization_overrides(cfg, model_config, processor, legacy_tokenization_mode=bool(cfg.legacy_train_mode))


def validate_saved_checkpoint_config(cfg, checkpoint_dir: Path) -> None:
    """Validate legacy config fields on disk after saving a checkpoint."""
    if not ((cfg.legacy_train_mode and cfg.use_discrete_diffusion) or (cfg.legacy_dfm_mode and cfg.use_discrete_flow_matching)):
        return
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"[legacy_train] config.json not found at {cfg_path}")
    raw = json.load(open(cfg_path))
    cfg_dict = raw.get("vla", raw) if isinstance(raw, dict) else {}
    errors = []
    if cfg.use_discrete_flow_matching:
        if cfg_dict.get("use_discrete_flow_matching") is not True:
            errors.append("use_discrete_flow_matching != True")
        if cfg_dict.get("use_discrete_diffusion") is True:
            errors.append("use_discrete_diffusion should be False for DFM legacy checkpoints")
    else:
        if cfg_dict.get("use_discrete_diffusion") is not True:
            errors.append("use_discrete_diffusion != True")
    if cfg_dict.get("legacy_train_mode") is not True:
        errors.append("legacy_train_mode != True")
    if cfg_dict.get("legacy_eval_mode") is not True:
        errors.append("legacy_eval_mode != True")
    if cfg_dict.get("action_vocab_anchor") != "legacy":
        errors.append(f"action_vocab_anchor != 'legacy' (found {cfg_dict.get('action_vocab_anchor')})")
    if cfg_dict.get("action_token_begin_idx") != ACTION_TOKEN_BEGIN_IDX:
        errors.append(
            f"action_token_begin_idx != {ACTION_TOKEN_BEGIN_IDX} "
            f"(found {cfg_dict.get('action_token_begin_idx')})"
        )
    if errors:
        raise ValueError("[legacy_train] checkpoint config validation failed: " + "; ".join(errors))
    print("[legacy_train] checkpoint config validation passed")


def _apply_finetune_cfg_to_model_config(cfg, model_config, processor) -> None:
    """Apply finetune CLI args to model config so checkpoints are self-describing."""
    if cfg.use_discrete_diffusion or cfg.use_discrete_flow_matching:
        if processor.tokenizer.mask_token_id is None:
            processor.tokenizer.add_special_tokens({"mask_token": "<mask>"})
        if processor.tokenizer.mask_token_id is None:
            raise ValueError("mask_token_id is None after tokenizer setup; cannot proceed.")
        model_config.set_mask_token_id(processor.tokenizer.mask_token_id)
        if hasattr(model_config, "use_mask_token"):
            model_config.use_mask_token = True

    model_config.use_discrete_diffusion = cfg.use_discrete_diffusion
    model_config.use_discrete_flow_matching = cfg.use_discrete_flow_matching
    model_config.dfm_schedule = cfg.dfm_schedule
    model_config.dfm_time_eps = cfg.dfm_time_eps
    model_config.dfm_t_min = cfg.dfm_t_min
    model_config.dfm_t_max = cfg.dfm_t_max
    model_config.dfm_loss_mode = cfg.dfm_loss_mode
    model_config.dfm_weight_clip = cfg.dfm_weight_clip
    model_config.dfm_train_mode = cfg.dfm_train_mode
    model_config.dfm_maskgit_num_steps = cfg.dfm_maskgit_num_steps
    model_config.dfm_maskgit_schedule = cfg.dfm_maskgit_schedule
    model_config.dfm_t_bias_alpha = cfg.dfm_t_bias_alpha


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        torch_dtype (torch.dtype): Dtype used for model inputs and autocast.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_dtype: Optional[torch.dtype] = None,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_dtype (torch.dtype): Optional dtype to cast module parameters.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_dtype is not None:
        module = module.to(to_dtype)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    torch_dtype: torch.dtype,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    legacy_tokenization_mode: bool = False,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
    use_discrete_diffusion=False,
    use_discrete_flow_matching=False,
    dfm_schedule: str = "cosine",
    dfm_time_eps: float = 1e-3,
    dfm_t_min: float = 0.0,
    dfm_t_max: float = 1.0,
    dfm_loss_mode: str = "generalized_kl",
    dfm_weight_clip: float = 20.0,
    dfm_train_mode: str = "flow",
    dfm_t_bias_alpha: float = 1.0,
    dfm_log_mask_stats: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        legacy_tokenization_mode (bool): Whether to use legacy action masking logic.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict, dfm_trace)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
            dfm_trace: Optional dict containing per-batch t/kappa/mask_frac tensors.
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch_dtype)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    autocast_dtype = torch_dtype if torch_dtype in (torch.float16, torch.bfloat16) else None
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch_dtype).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
            dfm_schedule=dfm_schedule,
            dfm_time_eps=dfm_time_eps,
            dfm_t_min=dfm_t_min,
            dfm_t_max=dfm_t_max,
            dfm_loss_mode=dfm_loss_mode,
            dfm_weight_clip=dfm_weight_clip,
            dfm_train_mode=dfm_train_mode,
            dfm_t_bias_alpha=dfm_t_bias_alpha,
            dfm_log_mask_stats=dfm_log_mask_stats,
        )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    if use_discrete_diffusion or use_discrete_flow_matching:
        # For discrete diffusion, we only need to calculated masked action tokens
        ground_truth_token_ids = output.labels[:, 1:].to(device_id)
    if legacy_tokenization_mode:
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    else:
        current_action_mask = get_current_action_mask(
            ground_truth_token_ids, action_tokenizer.action_token_begin_idx, action_tokenizer.action_token_end_idx
        )
        next_actions_mask = get_next_actions_mask(
            ground_truth_token_ids, action_tokenizer.action_token_begin_idx, action_tokenizer.action_token_end_idx
        )

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
        if use_discrete_flow_matching and getattr(output, "dfm_stats", None) is not None:
            dfm_stats = output.dfm_stats
            metrics.update(
                {
                    "dfm_kappa_mean": dfm_stats["kappa_mean"].item(),
                    "dfm_mask_frac_mean": dfm_stats["mask_frac_mean"].item(),
                    "dfm_w_mean": dfm_stats["w_mean"].item(),
                    "dfm_frac_w_clipped": dfm_stats["frac_w_clipped"].item(),
                    "dfm_num_supervised_tokens": dfm_stats["num_supervised_tokens"].item(),
                }
            )
            if "t_mean" in dfm_stats:
                metrics["dfm_t_mean"] = dfm_stats["t_mean"].item()
            if "t_min" in dfm_stats:
                metrics["dfm_t_min"] = dfm_stats["t_min"].item()
            if "t_max" in dfm_stats:
                metrics["dfm_t_max"] = dfm_stats["t_max"].item()
            if "mask_ratio_mean" in dfm_stats:
                metrics["dfm_mask_ratio_mean"] = dfm_stats["mask_ratio_mean"].item()
            if "mask_ratio_min" in dfm_stats:
                metrics["dfm_mask_ratio_min"] = dfm_stats["mask_ratio_min"].item()
            if "mask_ratio_max" in dfm_stats:
                metrics["dfm_mask_ratio_max"] = dfm_stats["mask_ratio_max"].item()
            if "w_min" in dfm_stats:
                metrics["dfm_w_min"] = dfm_stats["w_min"].item()
            if "w_max" in dfm_stats:
                metrics["dfm_w_max"] = dfm_stats["w_max"].item()
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]

        if use_discrete_diffusion:
            # reset action mask to get correct hidden states for action portion
            ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
            if legacy_tokenization_mode:
                current_action_mask = get_current_action_mask(ground_truth_token_ids)
                next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
            else:
                current_action_mask = get_current_action_mask(
                    ground_truth_token_ids, action_tokenizer.action_token_begin_idx, action_tokenizer.action_token_end_idx
                )
                next_actions_mask = get_next_actions_mask(
                    ground_truth_token_ids, action_tokenizer.action_token_begin_idx, action_tokenizer.action_token_end_idx
                )

        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch_dtype)
        )  # (B, act_chunk_len, D)

        if use_l1_regression:
            # Predict action
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_diffusion:
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                        torch_dtype=torch_dtype,
                    )

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    dfm_trace = getattr(output, "dfm_trace", None)
    return loss, metrics, dfm_trace


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
    torch_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        torch_dtype (torch.dtype): Dtype for model inputs/autocast.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch_dtype,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        autocast_dtype = torch_dtype if torch_dtype in (torch.float16, torch.bfloat16) else None
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch_dtype).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch_dtype)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.config.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()
    validate_saved_checkpoint_config(cfg, checkpoint_dir)

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            config=vla.module.config,
            torch_dtype=resolve_torch_dtype(cfg.torch_dtype),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.config = vla.module.config
            merged_vla.save_pretrained(checkpoint_dir)
            vla.module.config.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # Wait for merged model to be saved
        dist.barrier()


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    torch_dtype: torch.dtype,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
    legacy_tokenization_mode: bool,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        torch_dtype (torch.dtype): Dtype used for model inputs/autocast.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics, _ = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                torch_dtype=torch_dtype,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                legacy_tokenization_mode=legacy_tokenization_mode,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                use_discrete_diffusion=cfg.use_discrete_diffusion,
                use_discrete_flow_matching=cfg.use_discrete_flow_matching,
                dfm_schedule=cfg.dfm_schedule,
                dfm_time_eps=cfg.dfm_time_eps,
                dfm_t_min=cfg.dfm_t_min,
                dfm_t_max=cfg.dfm_t_max,
                dfm_loss_mode=cfg.dfm_loss_mode,
                dfm_weight_clip=cfg.dfm_weight_clip,
                dfm_train_mode=cfg.dfm_train_mode,
                dfm_t_bias_alpha=cfg.dfm_t_bias_alpha,
                dfm_log_mask_stats=False,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )
    assert not (cfg.use_discrete_diffusion and cfg.use_discrete_flow_matching), (
        "Cannot enable both discrete diffusion and discrete flow matching!"
    )
    assert not (cfg.use_discrete_flow_matching and (cfg.use_l1_regression or cfg.use_diffusion)), (
        "DFM is not compatible with continuous action heads (L1 regression or diffusion)."
    )
    legacy_tokenization_mode, legacy_train_mode_arg = resolve_legacy_tokenization_mode(cfg)

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    torch_dtype = resolve_torch_dtype(cfg.torch_dtype)

    # Initialize wandb logging (placeholder entity/project cause W&B 404 upsertBucket)
    if distributed_state.is_main_process:
        wb_entity = cfg.wandb_entity if cfg.wandb_entity not in ("", "your-wandb-entity") else None
        wb_project = cfg.wandb_project if cfg.wandb_project not in ("", "your-wandb-project") else None
        wandb.init(entity=wb_entity, project=wb_project, name=f"ft+{run_id}")

    # Optional DFM mask-trace logging
    dfm_trace_path = None
    dfm_trace_samples = 0
    if cfg.dfm_log_mask_stats and cfg.use_discrete_flow_matching and distributed_state.is_main_process:
        dfm_trace_path = run_dir / "dfm_train_mask_trace.jsonl"
        print(f"[dfm_trace] logging per-batch mask stats to {dfm_trace_path}")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )
    if cfg.use_discrete_diffusion:
        if cfg.legacy_train_mode:
            if legacy_train_mode_arg is None:
                print("[legacy_train] auto-enabled for discrete diffusion")
            else:
                print("[legacy_train] enabled: using legacy prompt/tokenization/masks")
        elif legacy_train_mode_arg is False:
            print("[legacy_train] disabled by explicit flag")
    elif cfg.legacy_dfm_mode:
        print("[legacy_dfm] enabled: using legacy tokenizer/prompt/masks")

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    # Load the model configuration
    try:
        model_config = AutoConfig.from_pretrained(cfg.vla_path, trust_remote_code=True)
    except AttributeError as exc:
        # Some HF repos ship config modules without OpenVLAConfig; fall back to local class.
        if "OpenVLAConfig" not in str(exc):
            raise
        logger.warning(
            "OpenVLAConfig not found in remote configuration module; falling back to local OpenVLAConfig."
        )
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig as LocalOpenVLAConfig

        model_config = LocalOpenVLAConfig.from_pretrained(cfg.vla_path, trust_remote_code=True)

    if legacy_tokenization_mode:
        apply_legacy_tokenization_overrides(cfg, model_config, processor, legacy_tokenization_mode)
        # Legacy path: warn (do not fail) if model vocab (after padding) doesn't match tokenizer.vocab_size.
        text_cfg = getattr(model_config, "text_config", None)
        text_vocab = getattr(text_cfg, "vocab_size", None) if text_cfg is not None else None
        pad_multiple = getattr(model_config, "pad_to_multiple_of", 0) or 0
        if text_vocab is not None:
            base_vocab = int(text_vocab) - int(pad_multiple)
            tok_vocab = int(processor.tokenizer.vocab_size)
            if tok_vocab != base_vocab:
                print(
                    "[legacy_train] WARNING: tokenizer.vocab_size does not match base model vocab. "
                    f"tokenizer.vocab_size={tok_vocab} text_config.vocab_size={text_vocab} "
                    f"pad_to_multiple_of={pad_multiple} (base_vocab={base_vocab}). "
                    "Proceeding to preserve legacy behavior."
                )

    _apply_finetune_cfg_to_model_config(cfg, model_config, processor)

    # For DFM, stamp explicit action vocab range into config to avoid eval mismatches.
    if cfg.use_discrete_flow_matching:
        n_bins = int(getattr(model_config, "n_action_bins", 256))
        if legacy_tokenization_mode:
            action_range = resolve_action_vocab(
                processor.tokenizer,
                n_bins,
                "legacy",
                ACTION_TOKEN_BEGIN_IDX,
            )
            print(
                "[dfm_vocab] "
                f"anchor=legacy begin={action_range.begin} end={action_range.end} "
                f"pad_id={action_range.pad_token_id} vocab_size={action_range.vocab_size}"
            )
        else:
            anchor = getattr(model_config, "action_vocab_anchor", "pad")
            action_range = resolve_action_vocab(processor.tokenizer, n_bins, anchor)
            model_config.action_vocab_anchor = anchor
            model_config.action_token_begin_idx = action_range.begin
            print(
                "[dfm_vocab] "
                f"anchor={anchor} begin={action_range.begin} end={action_range.end} "
                f"pad_id={action_range.pad_token_id} vocab_size={action_range.vocab_size}"
            )

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        config=model_config,  # Pass the updated config
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    if cfg.use_discrete_flow_matching:
        # Keep runtime config consistent with stamped vocab range.
        vla.config.action_vocab_anchor = model_config.action_vocab_anchor
        vla.config.action_token_begin_idx = model_config.action_token_begin_idx

    # LoRA setup
    if cfg.use_lora:
        lora_kwargs = {}
        if cfg.use_discrete_flow_matching:
            # Ensure mask/pad embeddings are trainable and saved in the adapter for DFM
            lora_kwargs["modules_to_save"] = ["embed_tokens", "lm_head"]
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
            **lora_kwargs,
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_dtype=torch_dtype,
        )

    action_head = None
    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_dtype=torch_dtype,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_dtype=torch_dtype,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector,
            "noisy_action_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim},
            to_dtype=torch_dtype,
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )

    if cfg.resume and cfg.resume_step is not None:
        for _ in range(cfg.resume_step):
            scheduler.step()

    # Create Action Tokenizer
    model_cfg = getattr(vla, "module", vla)
    n_action_bins = getattr(getattr(model_cfg, "config", None), "n_action_bins", None)
    action_vocab_anchor = getattr(getattr(model_cfg, "config", None), "action_vocab_anchor", "pad")
    action_token_begin_idx = getattr(getattr(model_cfg, "config", None), "action_token_begin_idx", None)
    if legacy_tokenization_mode:
        action_tokenizer = ActionTokenizer(
            processor.tokenizer,
            bins=n_action_bins if n_action_bins is not None else 256,
            action_vocab_anchor="legacy",
            legacy_bins=True,
            action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX,
        )
        expected_begin = int(processor.tokenizer.vocab_size - ((n_action_bins or 256) + 1))
        if expected_begin != ACTION_TOKEN_BEGIN_IDX:
            raise ValueError(
                "legacy tokenization requires ACTION_TOKEN_BEGIN_IDX alignment. "
                f"Expected begin={expected_begin} from vocab_size and n_bins, "
                f"but ACTION_TOKEN_BEGIN_IDX={ACTION_TOKEN_BEGIN_IDX}. "
                "Update the tokenizer/vocab or constants for legacy training."
            )
    else:
        action_tokenizer = ActionTokenizer(
            processor.tokenizer,
            bins=n_action_bins if n_action_bins is not None else 256,
            action_vocab_anchor=action_vocab_anchor,
            action_token_begin_idx=action_token_begin_idx,
        )
    if cfg.use_discrete_flow_matching:
        # Fail fast on action vocab misalignment.
        model_config = getattr(model_cfg, "config", None)
        n_bins = int(getattr(model_config, "n_action_bins", n_action_bins or 256))
        anchor = getattr(model_config, "action_vocab_anchor", "pad")
        begin_override = getattr(model_config, "action_token_begin_idx", None)
        action_range = resolve_action_vocab(processor.tokenizer, n_bins, anchor, begin_override)
        if action_tokenizer.action_token_begin_idx != action_range.begin:
            raise ValueError(
                "DFM action vocab mismatch: "
                f"tokenizer_begin={action_tokenizer.action_token_begin_idx} "
                f"config_begin={action_range.begin} "
                f"anchor={anchor}"
            )
        # Validate encoded tokens fall within range.
        sample_actions = np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
        sample_ids = action_tokenizer.encode_actions_to_token_ids(sample_actions)
        validate_action_vocab_alignment(action_range, sample_ids.tolist())

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        legacy_mode=legacy_tokenization_mode,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # TODO [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_kappa_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_mask_frac_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_w_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_w_min": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_w_max": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_frac_w_clipped": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_num_supervised_tokens": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_t_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_t_min": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_t_max": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_mask_ratio_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_mask_ratio_min": deque(maxlen=cfg.grad_accumulation_steps),
        "dfm_mask_ratio_max": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics, dfm_trace = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                torch_dtype=torch_dtype,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                legacy_tokenization_mode=legacy_tokenization_mode,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                use_discrete_diffusion=cfg.use_discrete_diffusion,
                use_discrete_flow_matching=cfg.use_discrete_flow_matching,
                dfm_schedule=cfg.dfm_schedule,
                dfm_time_eps=cfg.dfm_time_eps,
                dfm_t_min=cfg.dfm_t_min,
                dfm_t_max=cfg.dfm_t_max,
                dfm_loss_mode=cfg.dfm_loss_mode,
                dfm_weight_clip=cfg.dfm_weight_clip,
                dfm_train_mode=cfg.dfm_train_mode,
                dfm_t_bias_alpha=cfg.dfm_t_bias_alpha,
                dfm_log_mask_stats=cfg.dfm_log_mask_stats if cfg.use_discrete_flow_matching else False,
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Compute global log step index
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

            # Optional per-batch DFM trace logging
            if (
                dfm_trace_path is not None
                and dfm_trace is not None
                and cfg.dfm_log_mask_every > 0
                and (cfg.dfm_log_mask_every == 1 or (log_step % cfg.dfm_log_mask_every == 0))
                and dfm_trace_samples < cfg.dfm_log_mask_max_samples
            ):
                try:
                    t_vals = dfm_trace.get("t")
                    kappa_vals = dfm_trace.get("kappa")
                    mask_frac_vals = dfm_trace.get("mask_frac")
                    if t_vals is not None and kappa_vals is not None and mask_frac_vals is not None:
                        t_list = t_vals.detach().float().cpu().view(-1).tolist()
                        kappa_list = kappa_vals.detach().float().cpu().view(-1).tolist()
                        mask_frac_list = mask_frac_vals.detach().float().cpu().view(-1).tolist()
                        remaining = max(cfg.dfm_log_mask_max_samples - dfm_trace_samples, 0)
                        n = min(len(t_list), len(kappa_list), len(mask_frac_list), remaining)
                        if n > 0:
                            row = {
                                "step": int(log_step),
                                "t": t_list[:n],
                                "kappa": kappa_list[:n],
                                "mask_frac": mask_frac_list[:n],
                            }
                            with open(dfm_trace_path, "a", encoding="utf-8") as handle:
                                handle.write(json.dumps(row) + "\n")
                            dfm_trace_samples += n
                except Exception as exc:
                    print(f"[dfm_trace] failed to log mask stats: {exc}")

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                warmup_step = log_step if cfg.resume else gradient_step_idx
                lr_progress = min((warmup_step + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    torch_dtype=torch_dtype,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                    legacy_tokenization_mode=legacy_tokenization_mode,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
