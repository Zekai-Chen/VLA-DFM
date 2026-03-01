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
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from experiments.robot.openvla_utils import (
    get_model,
    get_processor,
    get_proprio_projector,
    prepare_images_for_vla,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--dfm_decode_mode", type=str, default="maskgit")
    parser.add_argument("--dfm_num_steps", type=int, default=128)
    parser.add_argument("--dfm_early_exit", type=str, default="False")
    parser.add_argument("--center_crop", type=str, default="True")
    parser.add_argument("--use_proprio", type=str, default="True")
    parser.add_argument("--check_image_parity", type=str, default="False")
    args = parser.parse_args()

    def _as_bool(v: str) -> bool:
        return str(v).lower() in ("1", "true", "yes", "y")

    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.checkpoint,
        use_film=False,
        num_images_in_input=2,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=_as_bool(args.center_crop),
        dfm_num_steps=int(args.dfm_num_steps),
        dfm_early_exit=_as_bool(args.dfm_early_exit),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processor = get_processor(cfg)
    vla = get_model(cfg)
    proprio_projector = None
    if _as_bool(args.use_proprio):
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)

    # Action tokenizer + unnorm key
    n_action_bins = getattr(vla.config, "n_action_bins", 256)
    anchor = getattr(vla.config, "action_vocab_anchor", "pad")
    action_tokenizer = ActionTokenizer(processor.tokenizer, bins=n_action_bins, action_vocab_anchor=anchor)
    unnorm_key = _resolve_unnorm_key(vla, args.dataset_name)
    action_stats = vla.get_action_stats(unnorm_key)

    use_wrist_image = getattr(vla.vision_backbone, "get_num_images_in_input", lambda: 1)() > 1
    batch_transform = SanityBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=_as_bool(args.use_proprio),
    )
    dataset = RLDSDataset(
        data_root_dir=args.data_root,
        data_mix=args.dataset_name,
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

    # Optional parity check for the first sample
    if _as_bool(args.check_image_parity):
        first = next(iter(dataset))
        raw_img = first.get("_raw_image")
        if raw_img is not None:
            mean_diff, std_diff, l2 = _compare_image_transforms(raw_img, processor, cfg)
            print(
                f"[image_parity] mean_diff={mean_diff:+.6f} std_diff={std_diff:+.6f} l2={l2:.6f}"
            )

    # Metrics accumulators
    token_match_sum = 0.0
    token_total = 0
    seq_match_sum = 0
    l2_norm_sum = 0.0
    l2_unnorm_sum = 0.0
    count = 0

    data_iter = iter(dataset)
    for _ in range(args.num_batches):
        sample = next(data_iter)
        batch = collator([sample])
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        actions_gt_norm = batch["actions"].numpy()

        # Determine prompt length and strip action tokens + stop
        action_chunk_len = actions_gt_norm.shape[1] * actions_gt_norm.shape[2]
        prompt_len = input_ids.shape[1] - (action_chunk_len + 1)
        input_ids_prompt = input_ids[:, :prompt_len]
        attention_mask_prompt = attention_mask[:, :prompt_len]

        # Proprio (if present)
        proprio = batch.get("proprio")
        if proprio is not None:
            proprio = proprio.numpy()

        # DFM predict
        with torch.no_grad():
            pred_actions_unnorm, _, debug = vla.predict_action(
                input_ids=input_ids_prompt,
                attention_mask=attention_mask_prompt,
                pixel_values=pixel_values,
                unnorm_key=unnorm_key,
                proprio=proprio,
                proprio_projector=proprio_projector,
                use_film=False,
                use_discrete_diffusion=False,
                use_discrete_flow_matching=True,
                dfm_num_steps=int(args.dfm_num_steps),
                dfm_schedule=getattr(vla.config, "dfm_schedule", "cosine"),
                dfm_early_exit=_as_bool(args.dfm_early_exit),
                dfm_decode_mode=args.dfm_decode_mode,
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

        token_match = (pred_token_ids == gt_token_ids)
        token_match_sum += float(token_match.mean())
        token_total += 1
        seq_match_sum += float(token_match.all())

        l2_norm = float(np.sqrt(np.mean((pred_actions_norm - gt_actions_norm) ** 2)))
        l2_unnorm = float(np.sqrt(np.mean((pred_actions_unnorm - gt_actions_unnorm) ** 2)))
        l2_norm_sum += l2_norm
        l2_unnorm_sum += l2_unnorm
        count += 1

        dfm_stats = (debug or {}).get("dfm_stats", {})
        print(
            f"[batch {count}] token_match={token_match.mean():.3f} seq_match={token_match.all()} "
            f"l2_norm={l2_norm:.4f} l2_unnorm={l2_unnorm:.4f} "
            f"dfm_decode_mode={dfm_stats.get('dfm_decode_mode')} "
            f"n_masked_init={dfm_stats.get('dfm_n_masked_initial')} "
            f"n_action_pos={dfm_stats.get('dfm_n_action_positions')}"
        )
        print(f"[batch {count}] pred_norm_stats={_stats(pred_actions_norm)}")
        print(f"[batch {count}] pred_unnorm_stats={_stats(pred_actions_unnorm)}")

    if count > 0:
        print("\n=== Summary ===")
        print(f"token_match_mean: {token_match_sum / count:.4f}")
        print(f"seq_match_mean:   {seq_match_sum / count:.4f}")
        print(f"l2_norm_mean:     {l2_norm_sum / count:.6f}")
        print(f"l2_unnorm_mean:   {l2_unnorm_sum / count:.6f}")


if __name__ == "__main__":
    main()
