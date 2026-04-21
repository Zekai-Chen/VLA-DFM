#!/bin/bash
set -euo pipefail

# Evaluate DDVLA checkpoint on LIBERO-Object
# Usage: bash scripts/eval_ddvla.sh <checkpoint_path> [num_trials]
#
# Examples:
#   bash scripts/eval_ddvla.sh ~/checkpoints/ddvla-320k/<run>/checkpoint-320000
#   bash scripts/eval_ddvla.sh ~/checkpoints/ddvla-320k/<run>/checkpoint-320000 50

CKPT="${1:?Usage: bash scripts/eval_ddvla.sh <checkpoint_path> [num_trials]}"
NUM_TRIALS="${2:-50}"
TASK_SUITE="${TASK_SUITE:-libero_object}"

echo "========================================="
echo "DDVLA Evaluation"
echo "  Checkpoint: $CKPT"
echo "  Task suite: $TASK_SUITE"
echo "  Trials per task: $NUM_TRIALS"
echo "  Total trials: $((NUM_TRIALS * 10))"
echo "========================================="

export MUJOCO_GL=osmesa

python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$TASK_SUITE" \
    --num_trials_per_task "$NUM_TRIALS" \
    --use_l1_regression False \
    --use_diffusion False \
    --use_discrete_diffusion True \
    --use_film False \
    --center_crop True \
    --num_images_in_input 2 \
    --num_open_loop_steps 8 \
    --use_proprio True \
    --use_wandb False
