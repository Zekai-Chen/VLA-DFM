#!/bin/bash
set -euo pipefail

# Discrete Diffusion VLA training script
# Paper configuration: 8x A100-80GB, batch=8/GPU, 320k steps, lora_rank=32
#
# Usage:
#   bash scripts/train_ddvla.sh                                    # fresh start
#   bash scripts/train_ddvla.sh --resume /path/to/checkpoint-7000  # resume

# ── Paths (edit these) ─────────────────────────────────────────────
VLA_PATH="${VLA_PATH:-$HOME/data/models/openvla-7b}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/RLDS/modified_libero_rlds}"
RUN_ROOT="${RUN_ROOT:-$HOME/checkpoints/ddvla-320k}"
DATASET_NAME="${DATASET_NAME:-libero_object_no_noops}"

# ── Hardware ───────────────────────────────────────────────────────
NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"

# ── Training params ────────────────────────────────────────────────
BATCH_SIZE=8
LR=5e-4
MAX_STEPS=320000
SAVE_FREQ=7000          # ~3h per checkpoint on 8xA100
DECAY_START=100000
LORA_RANK=32

# ── Resume handling ────────────────────────────────────────────────
# Usage:
#   --resume /path/to/checkpoint-7000    Resume from specific checkpoint
#   --resume latest                      Auto-find latest checkpoint in RUN_ROOT
#   --resume                             Same as --resume latest
RESUME_ARGS=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_TARGET="${2:-latest}"
    if [[ "$RESUME_TARGET" == "latest" ]]; then
        FOUND=$(find "$RUN_ROOT" -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
        if [[ -z "$FOUND" ]]; then
            echo "No checkpoint found in $RUN_ROOT — starting fresh"
            RESUME_TARGET=""
        else
            RESUME_TARGET="$FOUND"
        fi
    fi
    if [[ -n "$RESUME_TARGET" ]]; then
        RESUME_STEP=$(basename "$RESUME_TARGET" | grep -oP '\d+')
        VLA_PATH="$RESUME_TARGET"
        RESUME_ARGS="--resume True --resume_step $RESUME_STEP"
        echo "Resuming from: $RESUME_TARGET (step $RESUME_STEP)"
    fi
fi

# ── Diagnostics ────────────────────────────────────────────────────
echo "========================================="
echo "DDVLA Training"
echo "  GPUs: $NPROC"
echo "  VLA path: $VLA_PATH"
echo "  Dataset: $DATASET_NAME"
echo "  Batch: ${BATCH_SIZE}/GPU x ${NPROC} GPUs = $((BATCH_SIZE * NPROC)) total"
echo "  Steps: $MAX_STEPS"
echo "  Save freq: $SAVE_FREQ (~$((SAVE_FREQ * 16 / 36000))h on 8xA100)"
echo "  LoRA rank: $LORA_RANK"
echo "========================================="

# ── Launch ─────────────────────────────────────────────────────────
torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC" vla-scripts/finetune.py \
    --vla_path "$VLA_PATH" \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "$RUN_ROOT" \
    --use_discrete_diffusion True \
    --use_l1_regression False \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --num_steps_before_decay $DECAY_START \
    --max_steps $MAX_STEPS \
    --save_freq $SAVE_FREQ \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank $LORA_RANK \
    $RESUME_ARGS
