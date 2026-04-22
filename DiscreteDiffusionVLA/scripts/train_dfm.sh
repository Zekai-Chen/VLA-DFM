#!/bin/bash
set -euo pipefail

# DFM-VLA Supervised Fine-tuning script
# Matches DDVLA paper training scale (320k steps × batch 64)
# but uses Discrete Flow Matching instead of Discrete Diffusion
#
# Usage:
#   bash scripts/train_dfm.sh                                    # fresh start
#   bash scripts/train_dfm.sh --resume                           # auto-find latest checkpoint
#   bash scripts/train_dfm.sh --resume /path/to/checkpoint-7000  # specific checkpoint

# ── Paths (edit these) ─────────────────────────────────────────────
# VLA_PATH="${VLA_PATH:-$HOME/data/models/openvla-7b}"
# DATA_ROOT="${DATA_ROOT:-$HOME/data/RLDS/modified_libero_rlds}"
# RUN_ROOT="${RUN_ROOT:-$HOME/checkpoints/dfm-vla-320k}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Prefer this repo's prismatic/ over any pip-installed openvla copy (avoids ModuleNotFoundError: prismatic.vla.action_vocab).
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Hugging Face: keep hub + trust_remote_code modules under this repo instead of
# shared Lustre home_cache (avoids stale transformers_modules/openvla-7b).
# Opt back into cache_env.sh HF paths: DFM_USE_SHARED_HF_CACHE=1
if [[ "${DFM_USE_SHARED_HF_CACHE:-0}" != "1" ]]; then
    export HF_HOME="${DFM_HF_HOME:-$REPO_ROOT/.hf_cache}"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
fi

# ── Paths (edit these or use env overrides) ────────────────────────
VLA_PATH="${VLA_PATH:-$REPO_ROOT/data/models/openvla-7b}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/RLDS/modified_libero_rlds}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/checkpoints/dfm-vla-320k}"
DATASET_NAME="${DATASET_NAME:-libero_object_no_noops}"

# ── Hardware ───────────────────────────────────────────────────────
NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"

# ── Training params (matching DDVLA paper scale) ───────────────────
BATCH_SIZE=8
LR=5e-4
MAX_STEPS=320000
SAVE_FREQ=7000          # ~3h per checkpoint on 8xA100
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
# Usage:
#   --resume /path/to/checkpoint-7000    Resume from specific checkpoint
#   --resume latest                      Auto-find latest checkpoint in RUN_ROOT
#   --resume                             Same as --resume latest
RESUME_ARGS=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_TARGET="${2:-latest}"
    if [[ "$RESUME_TARGET" == "latest" ]]; then
        FOUND=$(find "$RUN_ROOT" -maxdepth 3 -name "*_chkpt" -type d 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
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
echo "DFM-VLA Supervised Fine-tuning"
echo "  GPUs: $NPROC"
echo "  VLA path: $VLA_PATH"
echo "  Dataset: $DATASET_NAME"
echo "  Batch: ${BATCH_SIZE}/GPU x ${NPROC} GPUs = $((BATCH_SIZE * NPROC)) total"
echo "  Steps: $MAX_STEPS"
echo "  Save freq: $SAVE_FREQ"
echo "  LoRA rank: $LORA_RANK"
echo "  DFM schedule: $DFM_SCHEDULE, t_max: $DFM_T_MAX"
echo "  Legacy DFM mode: True"
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
    --dfm_schedule $DFM_SCHEDULE \
    --dfm_loss_mode $DFM_LOSS_MODE \
    --dfm_train_mode $DFM_TRAIN_MODE \
    --dfm_time_eps $DFM_TIME_EPS \
    --dfm_t_min $DFM_T_MIN \
    --dfm_t_max $DFM_T_MAX \
    --dfm_weight_clip $DFM_WEIGHT_CLIP \
    --legacy_dfm_mode True \
    $RESUME_ARGS
