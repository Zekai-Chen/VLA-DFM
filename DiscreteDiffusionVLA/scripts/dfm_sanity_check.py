#!/usr/bin/env python3
"""
Offline DFM sanity check: run DFM action prediction on training samples and compare
predictions against ground-truth action tokens/actions. This isolates "bad model"
vs "bad eval path".
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure repo root is on sys.path for "experiments.*" imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.openvla_utils import (  # noqa: E402
    get_processor,
    get_proprio_projector,
    prepare_images_for_vla,
)
from experiments.robot.robot_utils import get_model  # noqa: E402
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import compute_token_accuracy, get_current_action_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    STOP_INDEX,
    NormalizationType,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


class SanityBatchTransform(RLDSBatchTransform):
    """RLDS batch transform that also returns raw images for parity checks."""

    def __call__(self, rlds_batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out = super().__call__(rlds_batch)
        out["_raw_image"] = rlds_batch["observation"]["image_primary"][0]
        if self.use_wrist_image:
            raw_wrist = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    raw_wrist.append(rlds_batch["observation"][k][0])
            out["_raw_wrist_images"] = raw_wrist
        return out


def _resolve_unnorm_key(vla, dataset_name: str) -> str:
    if dataset_name in vla.norm_stats:
        return dataset_name
    alt = f"{dataset_name}_no_noops"
    if alt in vla.norm_stats:
        return alt
    raise KeyError(f"Unnorm key not found for dataset={dataset_name}: keys={list(vla.norm_stats.keys())}")


def _unnormalize_actions(actions_norm: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    mask = stats.get("mask", np.ones_like(stats["min"], dtype=bool))
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        high = np.asarray(stats["max"])
        low = np.asarray(stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        high = np.asarray(stats["q99"])
        low = np.asarray(stats["q01"])
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")
    return np.where(
        mask,
        0.5 * (actions_norm + 1.0) * (high - low + 1e-8) + low,
        actions_norm,
    )


def _normalize_actions(actions_unnorm: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    mask = stats.get("mask", np.ones_like(stats["min"], dtype=bool))
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        high = np.asarray(stats["max"])
        low = np.asarray(stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        high = np.asarray(stats["q99"])
        low = np.asarray(stats["q01"])
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")
    normed = np.where(
        mask,
        2 * (actions_unnorm - low) / (high - low + 1e-8) - 1.0,
        actions_unnorm,
    )
    return np.clip(normed, -1.0, 1.0)


def _stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan, "clip_frac": math.nan}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "clip_frac": float(np.mean((arr <= -1.0) | (arr >= 1.0))),
    }


def _stat_value(stats: Dict[str, np.ndarray], key: str, idx: int) -> Optional[float]:
    arr = stats.get(key)
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if idx >= arr.shape[0]:
        return None
    return float(arr[idx])


def _histogram(values: List[int], bins: int) -> Tuple[List[float], List[int]]:
    if not values:
        return [], []
    hist = np.histogram(values, bins=bins)
    return hist[1].tolist(), hist[0].astype(int).tolist()


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _compare_image_transforms(
    raw_image: np.ndarray,
    processor,
    cfg,
) -> Tuple[float, float, float]:
    pil = Image.fromarray(raw_image).convert("RGB")
    train_t = processor.image_processor.apply_transform(pil)
    eval_img = prepare_images_for_vla([raw_image], cfg)[0]
    eval_t = processor.image_processor.apply_transform(eval_img)
    train_t = _tensor_to_numpy(train_t).astype(np.float32)
    eval_t = _tensor_to_numpy(eval_t).astype(np.float32)
    mean_diff = float(np.mean(train_t) - np.mean(eval_t))
    std_diff = float(np.std(train_t) - np.std(eval_t))
    l2 = float(np.sqrt(np.mean((train_t - eval_t) ** 2)))
    return mean_diff, std_diff, l2


def _as_bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y")


def _compute_num_patches(vla, use_proprio: bool) -> int:
    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    if use_proprio:
        num_patches += 1
    return num_patches


def _teacher_forced_metrics(
    vla,
    batch: Dict[str, torch.Tensor],
    action_tokenizer: ActionTokenizer,
    unnorm_key: str,
    proprio_projector,
    use_proprio: bool,
    dfm_schedule: str,
    mask_token_id: int,
    compute_masked_denoise: bool = True,
) -> Dict[str, float]:
    device = next(vla.parameters()).device
    pixel_dtype = torch.bfloat16
    if hasattr(vla, "vision_backbone") and hasattr(vla.vision_backbone, "half_precision_dtype"):
        pixel_dtype = vla.vision_backbone.half_precision_dtype

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    pixel_values = batch["pixel_values"].to(device, dtype=pixel_dtype)
    proprio = batch.get("proprio")
    if proprio is not None:
        proprio = proprio.to(device)
        if proprio_projector is not None:
            try:
                proj_dtype = next(proprio_projector.parameters()).dtype
                proprio = proprio.to(dtype=proj_dtype)
            except StopIteration:
                pass

    def _forward(ids: torch.Tensor):
        return vla(
            input_ids=ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=False,
            proprio=proprio if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            use_film=False,
            dfm_schedule=dfm_schedule,
        )

    with torch.no_grad():
        output = _forward(input_ids)

    num_patches = _compute_num_patches(vla, use_proprio)
    logits = output.logits[:, num_patches:-1, :]
    gt_tokens = labels[:, 1:]
    # Align shapes if needed
    min_len = min(logits.shape[1], gt_tokens.shape[1])
    logits = logits[:, :min_len]
    gt_tokens = gt_tokens[:, :min_len]
    pred_tokens = logits.argmax(dim=-1)

    action_begin = action_tokenizer.action_token_begin_idx
    action_end = action_tokenizer.action_token_end_idx
    action_mask = (gt_tokens >= action_begin) & (gt_tokens < action_end)

    metrics: Dict[str, float] = {}
    if action_mask.sum().item() > 0:
        action_logits = logits[action_mask]
        action_targets = gt_tokens[action_mask]
        action_ce = F.cross_entropy(action_logits, action_targets, reduction="mean")
        action_acc = compute_token_accuracy(pred_tokens, gt_tokens, action_mask)
        metrics["teacher_forced_action_ce"] = float(action_ce.item())
        metrics["teacher_forced_action_acc"] = float(action_acc.item())
    else:
        metrics["teacher_forced_action_ce"] = float("nan")
        metrics["teacher_forced_action_acc"] = float("nan")

    stop_mask = gt_tokens == STOP_INDEX
    if stop_mask.sum().item() > 0:
        stop_acc = (pred_tokens[stop_mask] == gt_tokens[stop_mask]).float().mean()
        metrics["teacher_forced_stop_acc"] = float(stop_acc.item())
    else:
        metrics["teacher_forced_stop_acc"] = float("nan")

    if compute_masked_denoise and mask_token_id is not None:
        # Replace action tokens with mask token and recompute logits
        masked_input_ids = input_ids.clone()
        action_positions = (input_ids >= action_begin) & (input_ids < action_end)
        masked_input_ids[action_positions] = mask_token_id
        with torch.no_grad():
            masked_out = _forward(masked_input_ids)
        masked_logits = masked_out.logits[:, num_patches:-1, :]
        masked_logits = masked_logits[:, :min_len]
        masked_pred = masked_logits.argmax(dim=-1)
        if action_mask.sum().item() > 0:
            masked_action_logits = masked_logits[action_mask]
            masked_targets = gt_tokens[action_mask]
            masked_ce = F.cross_entropy(masked_action_logits, masked_targets, reduction="mean")
            masked_acc = compute_token_accuracy(masked_pred, gt_tokens, action_mask)
            metrics["masked_denoise_action_ce"] = float(masked_ce.item())
            metrics["masked_denoise_action_acc"] = float(masked_acc.item())
        else:
            metrics["masked_denoise_action_ce"] = float("nan")
            metrics["masked_denoise_action_acc"] = float("nan")

    return metrics


def _fixed_mask_denoise_metrics(
    vla,
    batch: Dict[str, torch.Tensor],
    action_tokenizer: ActionTokenizer,
    proprio_projector,
    use_proprio: bool,
    dfm_schedule: str,
    mask_token_id: int,
    mask_ratio: float,
    gripper_offsets: List[int],
) -> Dict[str, float]:
    device = next(vla.parameters()).device
    pixel_dtype = torch.bfloat16
    if hasattr(vla, "vision_backbone") and hasattr(vla.vision_backbone, "half_precision_dtype"):
        pixel_dtype = vla.vision_backbone.half_precision_dtype

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    pixel_values = batch["pixel_values"].to(device, dtype=pixel_dtype)
    proprio = batch.get("proprio")
    if proprio is not None:
        proprio = proprio.to(device)
        if proprio_projector is not None:
            try:
                proj_dtype = next(proprio_projector.parameters()).dtype
                proprio = proprio.to(dtype=proj_dtype)
            except StopIteration:
                pass

    action_begin = action_tokenizer.action_token_begin_idx
    action_end = action_tokenizer.action_token_end_idx
    action_mask_labels = (labels >= action_begin) & (labels < action_end)

    if mask_ratio <= 0.0:
        masked_mask = torch.zeros_like(action_mask_labels, dtype=torch.bool)
        eval_override = action_mask_labels
    else:
        num_action = action_mask_labels.sum(dim=1)
        num_mask = torch.round(num_action.float() * mask_ratio).long()
        num_mask = torch.clamp(num_mask, min=1)
        rand = torch.rand_like(action_mask_labels.float())
        rand = torch.where(action_mask_labels, rand, torch.full_like(rand, 2.0))
        perm = rand.argsort(dim=1)
        ranks = perm.argsort(dim=1)
        masked_mask = (ranks < num_mask[:, None]) & action_mask_labels

    masked_input_ids = input_ids.clone()
    masked_input_ids[masked_mask] = mask_token_id

    with torch.no_grad():
        output = vla(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=False,
            proprio=proprio if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            use_film=False,
            dfm_schedule=dfm_schedule,
        )

    num_patches = _compute_num_patches(vla, use_proprio)
    logits = output.logits[:, num_patches:-1, :]
    gt_tokens = labels[:, 1:]
    min_len = min(logits.shape[1], gt_tokens.shape[1])
    logits = logits[:, :min_len]
    gt_tokens = gt_tokens[:, :min_len]
    pred_tokens = logits.argmax(dim=-1)

    action_mask = (gt_tokens >= action_begin) & (gt_tokens < action_end)
    masked_eval = masked_mask[:, 1:][:, :min_len]
    if mask_ratio <= 0.0:
        masked_action_mask = eval_override[:, 1:][:, :min_len] & action_mask
    else:
        masked_action_mask = masked_eval & action_mask

    metrics: Dict[str, float] = {}
    if masked_action_mask.sum().item() > 0:
        masked_action_logits = logits[masked_action_mask]
        masked_targets = gt_tokens[masked_action_mask]
        masked_ce = F.cross_entropy(masked_action_logits, masked_targets, reduction="mean")
        masked_acc = compute_token_accuracy(pred_tokens, gt_tokens, masked_action_mask)
        metrics["masked_denoise_action_ce"] = float(masked_ce.item())
        metrics["masked_denoise_action_acc"] = float(masked_acc.item())
    else:
        metrics["masked_denoise_action_ce"] = float("nan")
        metrics["masked_denoise_action_acc"] = float("nan")

    # Gripper-only metrics (restricted to gripper token positions)
    gripper_mask = torch.zeros_like(action_mask_labels, dtype=torch.bool)
    for b in range(action_mask_labels.shape[0]):
        positions = torch.nonzero(action_mask_labels[b]).flatten()
        if positions.numel() >= (ACTION_DIM * NUM_ACTIONS_CHUNK):
            gripper_positions = positions[gripper_offsets]
            gripper_mask[b, gripper_positions] = True
    gripper_mask_eval = gripper_mask[:, 1:][:, :min_len]
    if mask_ratio <= 0.0:
        masked_gripper_mask = gripper_mask_eval
    else:
        masked_gripper_mask = masked_eval & gripper_mask_eval
    if masked_gripper_mask.sum().item() > 0:
        gripper_logits = logits[masked_gripper_mask]
        gripper_targets = gt_tokens[masked_gripper_mask]
        gripper_ce = F.cross_entropy(gripper_logits, gripper_targets, reduction="mean")
        gripper_acc = compute_token_accuracy(pred_tokens, gt_tokens, masked_gripper_mask)
        metrics["masked_denoise_gripper_ce"] = float(gripper_ce.item())
        metrics["masked_denoise_gripper_acc"] = float(gripper_acc.item())
    else:
        metrics["masked_denoise_gripper_ce"] = float("nan")
        metrics["masked_denoise_gripper_acc"] = float("nan")

    return metrics


def _evaluate_checkpoint(
    label: str,
    checkpoint: str,
    data_root: str,
    dataset_name: str,
    num_batches: int,
    use_discrete_flow_matching: bool,
    use_discrete_diffusion: bool,
    dfm_decode_mode: str,
    dfm_num_steps: int,
    dfm_maskgit_num_steps: int,
    dfm_maskgit_schedule: str,
    dfm_early_exit: bool,
    center_crop: bool,
    use_proprio: bool,
    check_image_parity: bool,
    log_gripper_hist: bool,
    gripper_hist_bins: int,
    mask_embed_override: str = "none",
    compute_masked_denoise: bool = True,
    mask_ratios: Optional[List[float]] = None,
) -> Dict[str, float]:
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=checkpoint,
        use_film=False,
        num_images_in_input=2,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=center_crop,
        dfm_num_steps=int(dfm_num_steps),
        dfm_maskgit_num_steps=int(dfm_maskgit_num_steps),
        dfm_maskgit_schedule=dfm_maskgit_schedule,
        dfm_early_exit=dfm_early_exit,
    )

    processor = get_processor(cfg)
    vla = get_model(cfg)
    proprio_projector = None
    if use_proprio:
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)

    n_action_bins = getattr(vla.config, "n_action_bins", 256)
    anchor = getattr(vla.config, "action_vocab_anchor", "pad")
    begin_override = getattr(vla.config, "action_token_begin_idx", None)
    action_tokenizer = ActionTokenizer(
        processor.tokenizer,
        bins=n_action_bins,
        action_vocab_anchor=anchor,
        action_token_begin_idx=begin_override,
    )
    unnorm_key = _resolve_unnorm_key(vla, dataset_name)
    action_stats = vla.get_action_stats(unnorm_key)
    if log_gripper_hist:
        gripper_idx = ACTION_DIM - 1
        gripper_stat = {
            "min": _stat_value(action_stats, "min", gripper_idx),
            "max": _stat_value(action_stats, "max", gripper_idx),
            "mean": _stat_value(action_stats, "mean", gripper_idx),
            "std": _stat_value(action_stats, "std", gripper_idx),
            "q01": _stat_value(action_stats, "q01", gripper_idx),
            "q99": _stat_value(action_stats, "q99", gripper_idx),
        }
        print(f"[{label}] dataset_gripper_stats={gripper_stat} (unnorm_key={unnorm_key})")
    mask_token_id = processor.tokenizer.mask_token_id
    pad_token_id = processor.tokenizer.pad_token_id

    # Optional mask embedding override (eval-only)
    orig_mask_embed = None
    orig_out_mask_embed = None
    if mask_embed_override == "pad":
        if mask_token_id is None or pad_token_id is None:
            raise ValueError("mask_embed_override=pad requires mask_token_id and pad_token_id to be set.")
        with torch.no_grad():
            in_emb = vla.get_input_embeddings().weight
            orig_mask_embed = in_emb[mask_token_id].detach().clone()
            in_emb[mask_token_id].copy_(in_emb[pad_token_id])
            out_emb = vla.get_output_embeddings()
            if out_emb is None and hasattr(vla, "language_model"):
                out_emb = vla.language_model.get_output_embeddings()
            if out_emb is not None and hasattr(out_emb, "weight"):
                orig_out_mask_embed = out_emb.weight[mask_token_id].detach().clone()
                out_emb.weight[mask_token_id].copy_(out_emb.weight[pad_token_id])

    use_wrist_image = getattr(vla.vision_backbone, "get_num_images_in_input", lambda: 1)() > 1
    batch_transform = SanityBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=use_proprio,
    )
    dataset = RLDSDataset(
        data_root_dir=data_root,
        data_mix=dataset_name,
        batch_transform=batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=10_000,
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    if check_image_parity:
        first = next(iter(dataset))
        raw_img = first.get("_raw_image")
        if raw_img is not None:
            mean_diff, std_diff, l2 = _compare_image_transforms(raw_img, processor, cfg)
            print(
                f"[{label}] [image_parity] mean_diff={mean_diff:+.6f} std_diff={std_diff:+.6f} l2={l2:.6f}"
            )

    # Accumulators
    token_match_sum = 0.0
    seq_match_sum = 0.0
    l2_norm_sum = 0.0
    l2_unnorm_sum = 0.0
    tf_ce_sum = 0.0
    tf_acc_sum = 0.0
    tf_stop_sum = 0.0
    tf_count = 0
    masked_ce_sum = 0.0
    masked_acc_sum = 0.0
    masked_count = 0
    mask_curve = {}
    gripper_offsets = [ACTION_DIM - 1 + i * ACTION_DIM for i in range(NUM_ACTIONS_CHUNK)]
    gripper_pred_token_ids: List[int] = []
    gripper_gt_token_ids: List[int] = []
    gripper_pred_norm_vals: List[float] = []
    gripper_gt_norm_vals: List[float] = []
    gripper_pred_unnorm_vals: List[float] = []
    gripper_gt_unnorm_vals: List[float] = []
    gripper_offset_max = max(gripper_offsets) if gripper_offsets else 0
    gripper_token_warned = False
    if mask_ratios:
        for ratio in mask_ratios:
            mask_curve[ratio] = {
                "ce_sum": 0.0,
                "acc_sum": 0.0,
                "count": 0,
                "g_ce_sum": 0.0,
                "g_acc_sum": 0.0,
                "g_count": 0,
            }
    count = 0

    data_iter = iter(dataset)
    for _ in range(num_batches):
        sample = next(data_iter)
        batch = collator([sample])
        device = next(vla.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_dtype = torch.bfloat16
        if hasattr(vla, "vision_backbone") and hasattr(vla.vision_backbone, "half_precision_dtype"):
            pixel_dtype = vla.vision_backbone.half_precision_dtype
        pixel_values = batch["pixel_values"].to(device, dtype=pixel_dtype)
        actions_gt_norm = batch["actions"].numpy()

        # Determine prompt length and strip action tokens + stop
        action_chunk_len = actions_gt_norm.shape[1] * actions_gt_norm.shape[2]
        prompt_len = input_ids.shape[1] - (action_chunk_len + 1)
        input_ids_prompt = input_ids[:, :prompt_len]
        attention_mask_prompt = attention_mask[:, :prompt_len]

        proprio = batch.get("proprio")
        if proprio is not None:
            proprio = proprio.to(device)
            if proprio_projector is not None:
                try:
                    proj_dtype = next(proprio_projector.parameters()).dtype
                    proprio = proprio.to(dtype=proj_dtype)
                except StopIteration:
                    pass

        with torch.no_grad():
            pred_actions_unnorm, _, debug = vla.predict_action(
                input_ids=input_ids_prompt,
                attention_mask=attention_mask_prompt,
                pixel_values=pixel_values,
                unnorm_key=unnorm_key,
                proprio=proprio,
                proprio_projector=proprio_projector,
                use_film=False,
                use_discrete_diffusion=use_discrete_diffusion,
                use_discrete_flow_matching=use_discrete_flow_matching,
                dfm_num_steps=int(dfm_num_steps),
                dfm_maskgit_num_steps=int(dfm_maskgit_num_steps),
                dfm_maskgit_schedule=dfm_maskgit_schedule,
                dfm_schedule=getattr(vla.config, "dfm_schedule", "cosine"),
                dfm_early_exit=dfm_early_exit,
                dfm_decode_mode=dfm_decode_mode,
                return_debug=True,
                dfm_debug_level=1,
            )

        pred_actions_unnorm = np.asarray(pred_actions_unnorm)
        pred_actions_norm = _normalize_actions(pred_actions_unnorm, action_stats)
        gt_actions_norm = actions_gt_norm
        gt_actions_unnorm = _unnormalize_actions(gt_actions_norm, action_stats)

        pred_token_ids = action_tokenizer.encode_actions_to_token_ids(pred_actions_norm)
        gt_token_ids = action_tokenizer.encode_actions_to_token_ids(gt_actions_norm)
        pred_token_ids = pred_token_ids.reshape(-1)
        gt_token_ids = gt_token_ids.reshape(-1)

        if log_gripper_hist:
            pred_actions_norm_r = pred_actions_norm.reshape(-1, ACTION_DIM)
            gt_actions_norm_r = gt_actions_norm.reshape(-1, ACTION_DIM)
            pred_actions_unnorm_r = pred_actions_unnorm.reshape(-1, ACTION_DIM)
            gt_actions_unnorm_r = gt_actions_unnorm.reshape(-1, ACTION_DIM)
            gripper_pred_norm_vals.extend(pred_actions_norm_r[:, -1].tolist())
            gripper_gt_norm_vals.extend(gt_actions_norm_r[:, -1].tolist())
            gripper_pred_unnorm_vals.extend(pred_actions_unnorm_r[:, -1].tolist())
            gripper_gt_unnorm_vals.extend(gt_actions_unnorm_r[:, -1].tolist())

            if pred_token_ids.shape[0] > gripper_offset_max and gt_token_ids.shape[0] > gripper_offset_max:
                gripper_pred_token_ids.extend(pred_token_ids[gripper_offsets].tolist())
                gripper_gt_token_ids.extend(gt_token_ids[gripper_offsets].tolist())
            elif not gripper_token_warned:
                gripper_token_warned = True
                print(f"[{label}] WARNING: gripper token offsets exceed token id length; skipping token hist.")

        token_match = (pred_token_ids == gt_token_ids)
        token_match_sum += float(token_match.mean())
        seq_match_sum += float(token_match.all())

        l2_norm = float(np.sqrt(np.mean((pred_actions_norm - gt_actions_norm) ** 2)))
        l2_unnorm = float(np.sqrt(np.mean((pred_actions_unnorm - gt_actions_unnorm) ** 2)))
        l2_norm_sum += l2_norm
        l2_unnorm_sum += l2_unnorm
        count += 1

        tf_metrics = _teacher_forced_metrics(
            vla=vla,
            batch=batch,
            action_tokenizer=action_tokenizer,
            unnorm_key=unnorm_key,
            proprio_projector=proprio_projector,
            use_proprio=use_proprio,
            dfm_schedule=getattr(vla.config, "dfm_schedule", "cosine"),
            mask_token_id=mask_token_id,
            compute_masked_denoise=compute_masked_denoise,
        )
        if not math.isnan(tf_metrics["teacher_forced_action_ce"]):
            tf_ce_sum += tf_metrics["teacher_forced_action_ce"]
            tf_acc_sum += tf_metrics["teacher_forced_action_acc"]
            tf_stop_sum += tf_metrics["teacher_forced_stop_acc"]
            tf_count += 1
        if "masked_denoise_action_ce" in tf_metrics and not math.isnan(tf_metrics["masked_denoise_action_ce"]):
            masked_ce_sum += tf_metrics["masked_denoise_action_ce"]
            masked_acc_sum += tf_metrics["masked_denoise_action_acc"]
            masked_count += 1

        if mask_ratios:
            for ratio in mask_ratios:
                ratio_metrics = _fixed_mask_denoise_metrics(
                    vla=vla,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    proprio_projector=proprio_projector,
                    use_proprio=use_proprio,
                    dfm_schedule=getattr(vla.config, "dfm_schedule", "cosine"),
                    mask_token_id=mask_token_id,
                    mask_ratio=ratio,
                    gripper_offsets=gripper_offsets,
                )
                if not math.isnan(ratio_metrics["masked_denoise_action_ce"]):
                    mask_curve[ratio]["ce_sum"] += ratio_metrics["masked_denoise_action_ce"]
                    mask_curve[ratio]["acc_sum"] += ratio_metrics["masked_denoise_action_acc"]
                    mask_curve[ratio]["count"] += 1
                if not math.isnan(ratio_metrics["masked_denoise_gripper_ce"]):
                    mask_curve[ratio]["g_ce_sum"] += ratio_metrics["masked_denoise_gripper_ce"]
                    mask_curve[ratio]["g_acc_sum"] += ratio_metrics["masked_denoise_gripper_acc"]
                    mask_curve[ratio]["g_count"] += 1

        dfm_stats = (debug or {}).get("dfm_stats", {})
        print(
            f"[{label} batch {count}] token_match={token_match.mean():.3f} seq_match={token_match.all()} "
            f"l2_norm={l2_norm:.4f} l2_unnorm={l2_unnorm:.4f} "
            f"tf_ce={tf_metrics['teacher_forced_action_ce']:.4f} "
            f"tf_acc={tf_metrics['teacher_forced_action_acc']:.4f} "
            f"masked_ce={tf_metrics.get('masked_denoise_action_ce', float('nan')):.4f} "
            f"masked_acc={tf_metrics.get('masked_denoise_action_acc', float('nan')):.4f} "
            f"mask_embed_override={mask_embed_override} "
            f"dfm_decode_mode={dfm_stats.get('dfm_decode_mode')} "
            f"n_masked_init={dfm_stats.get('dfm_n_masked_initial')} "
            f"n_action_pos={dfm_stats.get('dfm_n_action_positions')}"
        )
        print(f"[{label} batch {count}] pred_norm_stats={_stats(pred_actions_norm)}")
        print(f"[{label} batch {count}] pred_unnorm_stats={_stats(pred_actions_unnorm)}")

    summary = {}
    if count > 0:
        summary["token_match_mean"] = token_match_sum / count
        summary["seq_match_mean"] = seq_match_sum / count
        summary["l2_norm_mean"] = l2_norm_sum / count
        summary["l2_unnorm_mean"] = l2_unnorm_sum / count
    if tf_count > 0:
        summary["teacher_forced_action_ce"] = tf_ce_sum / tf_count
        summary["teacher_forced_action_acc"] = tf_acc_sum / tf_count
        summary["teacher_forced_stop_acc"] = tf_stop_sum / tf_count
    if masked_count > 0:
        summary["masked_denoise_action_ce"] = masked_ce_sum / masked_count
        summary["masked_denoise_action_acc"] = masked_acc_sum / masked_count
    if mask_ratios:
        for ratio in mask_ratios:
            entry = mask_curve[ratio]
            if entry["count"] > 0:
                summary[f"mask_ratio_{ratio}_action_ce"] = entry["ce_sum"] / entry["count"]
                summary[f"mask_ratio_{ratio}_action_acc"] = entry["acc_sum"] / entry["count"]
            if entry["g_count"] > 0:
                summary[f"mask_ratio_{ratio}_gripper_ce"] = entry["g_ce_sum"] / entry["g_count"]
                summary[f"mask_ratio_{ratio}_gripper_acc"] = entry["g_acc_sum"] / entry["g_count"]
    if log_gripper_hist:
        def _mean_std(vals: List[float]) -> Tuple[float, float]:
            if not vals:
                return float("nan"), float("nan")
            arr = np.asarray(vals, dtype=np.float32)
            return float(arr.mean()), float(arr.std())

        pred_norm_mean, pred_norm_std = _mean_std(gripper_pred_norm_vals)
        gt_norm_mean, gt_norm_std = _mean_std(gripper_gt_norm_vals)
        pred_unnorm_mean, pred_unnorm_std = _mean_std(gripper_pred_unnorm_vals)
        gt_unnorm_mean, gt_unnorm_std = _mean_std(gripper_gt_unnorm_vals)

        summary["gripper_pred_norm_mean"] = pred_norm_mean
        summary["gripper_pred_norm_std"] = pred_norm_std
        summary["gripper_gt_norm_mean"] = gt_norm_mean
        summary["gripper_gt_norm_std"] = gt_norm_std
        summary["gripper_pred_unnorm_mean"] = pred_unnorm_mean
        summary["gripper_pred_unnorm_std"] = pred_unnorm_std
        summary["gripper_gt_unnorm_mean"] = gt_unnorm_mean
        summary["gripper_gt_unnorm_std"] = gt_unnorm_std

        pred_bins, pred_counts = _histogram(gripper_pred_token_ids, gripper_hist_bins)
        gt_bins, gt_counts = _histogram(gripper_gt_token_ids, gripper_hist_bins)
        summary["gripper_pred_token_hist_bins"] = pred_bins
        summary["gripper_pred_token_hist_counts"] = pred_counts
        summary["gripper_gt_token_hist_bins"] = gt_bins
        summary["gripper_gt_token_hist_counts"] = gt_counts

        def _quartile_fracs(token_ids: List[int]) -> Tuple[float, float]:
            if not token_ids:
                return float("nan"), float("nan")
            token_ids_arr = np.asarray(token_ids, dtype=np.int64)
            bin_index = action_tokenizer.action_token_end_idx - token_ids_arr
            n_bins = int(action_tokenizer.n_bins)
            low_thresh = int(n_bins * 0.25)
            high_thresh = int(n_bins * 0.75)
            return float(np.mean(bin_index <= low_thresh)), float(np.mean(bin_index >= high_thresh))

        pred_low, pred_high = _quartile_fracs(gripper_pred_token_ids)
        gt_low, gt_high = _quartile_fracs(gripper_gt_token_ids)
        summary["gripper_pred_token_low_quartile_frac"] = pred_low
        summary["gripper_pred_token_high_quartile_frac"] = pred_high
        summary["gripper_gt_token_low_quartile_frac"] = gt_low
        summary["gripper_gt_token_high_quartile_frac"] = gt_high

        pred_top = Counter(gripper_pred_token_ids).most_common(5) if gripper_pred_token_ids else []
        gt_top = Counter(gripper_gt_token_ids).most_common(5) if gripper_gt_token_ids else []
        summary["gripper_pred_token_top5"] = pred_top
        summary["gripper_gt_token_top5"] = gt_top
    summary["mask_embed_override"] = mask_embed_override

    print(f"\n=== {label} Summary ===")
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    # Restore mask embedding if overridden
    if mask_embed_override == "pad" and orig_mask_embed is not None:
        with torch.no_grad():
            in_emb = vla.get_input_embeddings().weight
            in_emb[mask_token_id].copy_(orig_mask_embed)
            out_emb = vla.get_output_embeddings()
            if out_emb is None and hasattr(vla, "language_model"):
                out_emb = vla.language_model.get_output_embeddings()
            if orig_out_mask_embed is not None and out_emb is not None and hasattr(out_emb, "weight"):
                out_emb.weight[mask_token_id].copy_(orig_out_mask_embed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--dfm_decode_mode", type=str, default="maskgit")
    parser.add_argument("--dfm_num_steps", type=int, default=128)
    parser.add_argument("--dfm_maskgit_num_steps", type=int, default=12)
    parser.add_argument("--dfm_maskgit_schedule", type=str, default="cosine")
    parser.add_argument("--dfm_early_exit", type=str, default="False")
    parser.add_argument("--compare_dd", type=str, default="False")
    parser.add_argument("--dd_checkpoint", type=str, default="")
    parser.add_argument("--dd_num_steps", type=int, default=64)
    parser.add_argument("--mask_embed_override", type=str, default="none")
    parser.add_argument("--mask_ratios", type=str, default="")
    parser.add_argument("--primary_mode", type=str, default="dfm", choices=["dfm", "dd"])
    parser.add_argument("--center_crop", type=str, default="True")
    parser.add_argument("--use_proprio", type=str, default="True")
    parser.add_argument("--check_image_parity", type=str, default="False")
    parser.add_argument("--log_gripper_hist", type=str, default="True")
    parser.add_argument("--gripper_hist_bins", type=int, default=16)
    args = parser.parse_args()

    check_image_parity = _as_bool(args.check_image_parity)
    log_gripper_hist = _as_bool(args.log_gripper_hist)
    primary_mode = args.primary_mode.lower()
    if primary_mode not in ("dfm", "dd"):
        raise ValueError(f"Unknown primary_mode: {primary_mode}")

    primary_label = "DFM" if primary_mode == "dfm" else "DD"
    mask_ratios = []
    if args.mask_ratios:
        mask_ratios = [float(x) for x in args.mask_ratios.split(",") if x.strip() != ""]

    dfm_summary = _evaluate_checkpoint(
        label=primary_label,
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        num_batches=args.num_batches,
        use_discrete_flow_matching=(primary_mode == "dfm"),
        use_discrete_diffusion=(primary_mode == "dd"),
        dfm_decode_mode=args.dfm_decode_mode,
        dfm_num_steps=int(args.dfm_num_steps),
        dfm_maskgit_num_steps=int(args.dfm_maskgit_num_steps),
        dfm_maskgit_schedule=str(args.dfm_maskgit_schedule),
        dfm_early_exit=_as_bool(args.dfm_early_exit),
        center_crop=_as_bool(args.center_crop),
        use_proprio=_as_bool(args.use_proprio),
        check_image_parity=check_image_parity,
        log_gripper_hist=log_gripper_hist,
        gripper_hist_bins=int(args.gripper_hist_bins),
        mask_embed_override=args.mask_embed_override,
        compute_masked_denoise=(primary_mode == "dfm"),
        mask_ratios=mask_ratios,
    )

    compare_dd = _as_bool(args.compare_dd) or bool(args.dd_checkpoint)
    if compare_dd and primary_mode == "dd":
        raise ValueError("--compare_dd is only supported when --primary_mode dfm")
    if compare_dd:
        if not args.dd_checkpoint:
            raise ValueError("--dd_checkpoint is required when --compare_dd True")
        _evaluate_checkpoint(
            label="DD",
            checkpoint=args.dd_checkpoint,
            data_root=args.data_root,
            dataset_name=args.dataset_name,
            num_batches=args.num_batches,
            use_discrete_flow_matching=False,
            use_discrete_diffusion=True,
            dfm_decode_mode="ctmc",
            dfm_num_steps=int(args.dd_num_steps),
            dfm_maskgit_num_steps=int(args.dfm_maskgit_num_steps),
            dfm_maskgit_schedule=str(args.dfm_maskgit_schedule),
            dfm_early_exit=False,
            center_crop=_as_bool(args.center_crop),
            use_proprio=_as_bool(args.use_proprio),
            check_image_parity=False,
            log_gripper_hist=log_gripper_hist,
            gripper_hist_bins=int(args.gripper_hist_bins),
            mask_ratios=mask_ratios,
        )


if __name__ == "__main__":
    main()
