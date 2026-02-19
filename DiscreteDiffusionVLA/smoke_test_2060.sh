#!/bin/bash
set -euo pipefail

# --- Simple single-GPU smoke test for low VRAM (e.g., RTX 2060) ---
# Set these env vars before running if needed:
#   export VLA_PATH="openvla/openvla-7b"
#   export DATA_ROOT="/path/to/modified_libero_rlds"
#   export DATASET_NAME="libero_object_no_noops"
#   export RUN_ROOT_DIR="/tmp/openvla_smoke"

export WANDB_MODE=offline
export WANDB_DISABLED=true

VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
DATA_ROOT="${DATA_ROOT:-/path/to/modified_libero_rlds}"
DATASET_NAME="${DATASET_NAME:-libero_object_no_noops}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/tmp/openvla_smoke}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: DATA_ROOT does not exist: ${DATA_ROOT}"
  echo "Set DATA_ROOT to your RLDS root directory."
  exit 1
fi

cd /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
TORCH_DTYPE=${TORCH_DTYPE:-float16} \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --use_discrete_diffusion False \
  --use_discrete_flow_matching True \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --num_steps_before_decay 2 \
  --max_steps 2 \
  --save_freq 1 \
  --save_latest_checkpoint_only True \
  --image_aug False \
  --shuffle_buffer_size 1000 \
  --use_lora True \
  --lora_rank 4 \
  --torch_dtype "${TORCH_DTYPE}" \
  --dfm_schedule cosine \
  --dfm_loss_mode generalized_kl \
  --wandb_entity "offline" \
  --wandb_project "offline" \
  --run_id_note "smoke2060-$(date +%Y%m%d_%H%M)"
