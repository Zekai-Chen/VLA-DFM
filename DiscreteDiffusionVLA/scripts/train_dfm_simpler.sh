#!/bin/bash
set -euo pipefail

# DFM-VLA Supervised Fine-tuning script for SimplerEnv benchmarks
# Trains on Bridge V2 (WidowX) or Fractal/RT-1 (Google Robot) for SimplerEnv eval.
#
# Usage:
#   # Bridge V2 (WidowX)
#   DATASET_NAME=bridge_orig RUN_ROOT=$HOME/checkpoints/dfm-vla-bridge-320k \
#       bash scripts/train_dfm_simpler.sh
#
#   # Fractal / RT-1 (Google Robot)
#   DATASET_NAME=fractal20220817_data RUN_ROOT=$HOME/checkpoints/dfm-vla-fractal-320k \
#       bash scripts/train_dfm_simpler.sh
#
#   bash scripts/train_dfm_simpler.sh --resume   # auto-find latest checkpoint

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${DFM_USE_SHARED_HF_CACHE:-0}" != "1" ]]; then
    export HF_HOME="${DFM_HF_HOME:-$REPO_ROOT/.hf_cache}"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
fi

# ── Paths ──────────────────────────────────────────────────────────
VLA_PATH="${VLA_PATH:-$REPO_ROOT/data/models/openvla-7b}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/RLDS}"
DATASET_NAME="${DATASET_NAME:?DATASET_NAME required (bridge_orig | fractal20220817_data)}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT required}"

# ── Embodiment-specific defaults (SimplerEnv: single primary camera, no wrist) ─
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-1}"

# ── Hardware ───────────────────────────────────────────────────────
NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"

# ── Training params (matching DDVLA paper scale) ───────────────────
BATCH_SIZE=8
LR=5e-4
MAX_STEPS=320001
SAVE_FREQ=8000
DECAY_START=100000
LORA_RANK=32

# ── DFM-specific params ───────────────────────────────────────────
DFM_SCHEDULE="cosine"
DFM_LOSS_MODE="generalized_kl"
DFM_TRAIN_MODE="flow"
DFM_TIME_EPS="1e-3"
DFM_T_MIN="0.0"
DFM_T_MAX="0.7"
DFM_WEIGHT_CLIP="20.0"

# ── Resume handling ────────────────────────────────────────────────
RESUME_ARGS=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_TARGET="${2:-latest}"
    if [[ "$RESUME_TARGET" == "latest" ]]; then
        FOUND=$(find "$RUN_ROOT" -maxdepth 3 -name "*_chkpt" -type d 2>/dev/null | sed 's/.*--\([0-9]*\)_chkpt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)
        if [[ -z "$FOUND" ]]; then
            echo "No checkpoint found in $RUN_ROOT — starting fresh"
            RESUME_TARGET=""
        else
            RESUME_TARGET="$FOUND"
        fi
    fi
    if [[ -n "$RESUME_TARGET" ]]; then
        RESUME_STEP=$(basename "$RESUME_TARGET" | grep -oP '(\d+)_chkpt' | grep -oP '\d+')
        VLA_PATH="$RESUME_TARGET"
        RESUME_ARGS="--resume True --resume_step $RESUME_STEP"
        echo "Resuming from: $RESUME_TARGET (step $RESUME_STEP)"
    fi
fi

# ── Diagnostics ────────────────────────────────────────────────────
echo "========================================="
echo "DFM-VLA Supervised Fine-tuning (SimplerEnv)"
echo "  GPUs: $NPROC"
echo "  VLA path: $VLA_PATH"
echo "  Dataset: $DATASET_NAME"
echo "  Num images: $NUM_IMAGES_IN_INPUT"
echo "  Batch: ${BATCH_SIZE}/GPU x ${NPROC} GPUs = $((BATCH_SIZE * NPROC)) total"
echo "  Steps: $MAX_STEPS"
echo "  Save freq: $SAVE_FREQ"
echo "  LoRA rank: $LORA_RANK"
echo "  DFM schedule: $DFM_SCHEDULE, t_max: $DFM_T_MAX"
echo "========================================="

# ── Launch ─────────────────────────────────────────────────────────
torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC" vla-scripts/finetune.py \
    --vla_path "$VLA_PATH" \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "$RUN_ROOT" \
    --use_discrete_diffusion False \
    --use_discrete_flow_matching True \
    --use_l1_regression False \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input "$NUM_IMAGES_IN_INPUT" \
    --use_proprio True \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --num_steps_before_decay $DECAY_START \
    --max_steps $MAX_STEPS \
    --save_freq $SAVE_FREQ \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank $LORA_RANK \
    --dfm_schedule $DFM_SCHEDULE \
    --dfm_loss_mode $DFM_LOSS_MODE \
    --dfm_train_mode $DFM_TRAIN_MODE \
    --dfm_time_eps $DFM_TIME_EPS \
    --dfm_t_min $DFM_T_MIN \
    --dfm_t_max $DFM_T_MAX \
    --dfm_weight_clip $DFM_WEIGHT_CLIP \
    --legacy_dfm_mode True \
    $RESUME_ARGS
