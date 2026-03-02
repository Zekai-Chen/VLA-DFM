#!/bin/bash
#SBATCH --account=p32222
#SBATCH --partition=gengpu              # GPU partition (48 h max)
#SBATCH --gres=gpu:a100:2               # 2×A100 GPUs
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --time=47:00:00                 # smoke run
#SBATCH --job-name=openvla-ft-smoke-3k-dd
#SBATCH --output=logs/openvla_ft_smoke_3k_dd_%j.out
#SBATCH --error=logs/openvla_ft_smoke_3k_dd_%j.err

set -euo pipefail

# --- Environment (cluster-standard) ---
module load gcc/12.4.0-gcc-8.5.0 && module load cuda/12.4.0-gcc-12.4.0
module load git

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO_ROOT_DEFAULT=${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}
REPO_ROOT=${REPO_ROOT:-${REPO_ROOT_DEFAULT}}

# Resolve repo root (search upwards for expected scripts)
if [ ! -f "${REPO_ROOT}/vla-scripts/finetune.py" ] && [ ! -f "${REPO_ROOT}/experiments/robot/libero/run_libero_eval.py" ]; then
  SEARCH_DIR="${REPO_ROOT}"
  for _ in 1 2 3 4; do
    SEARCH_DIR=$(dirname "${SEARCH_DIR}")
    if [ -f "${SEARCH_DIR}/vla-scripts/finetune.py" ] || [ -f "${SEARCH_DIR}/experiments/robot/libero/run_libero_eval.py" ]; then
      REPO_ROOT="${SEARCH_DIR}"
      break
    fi
  done
fi

if [ ! -d "${REPO_ROOT}" ] || { [ ! -f "${REPO_ROOT}/vla-scripts/finetune.py" ] && [ ! -f "${REPO_ROOT}/experiments/robot/libero/run_libero_eval.py" ]; }; then
  echo "ERROR: Repo root not found or missing expected scripts at ${REPO_ROOT}" 1>&2
  exit 1
fi
cd "${REPO_ROOT}"

# Activate a virtualenv if available (robust check)
VENV_ACTIVATE=${VENV_ACTIVATE:-"${REPO_ROOT}/.venv/bin/activate"}
if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "ERROR: Virtualenv activate script not found at ${VENV_ACTIVATE}." 1>&2
  ls -la "${REPO_ROOT}"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV_ACTIVATE}"

# --- Basic diagnostics ---
mkdir -p logs

echo "SLURM_JOBID=${SLURM_JOBID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
export NCCL_IB_DISABLE=1
which python
python -V || true
nvcc --version || true
python -c "import torch; print('CUDA:', torch.version.cuda, 'GPUs:', torch.cuda.device_count())" || true

export WANDB_CACHE_DIR=/projects/p32222/.cache
export WANDB_MODE=online
export WANDB_DISABLED=false
export WANDB_NAME="openvla_ft_smoke_3k_dd_${SLURM_JOBID:-local}"

# --- Job params (adjust paths if needed) ---
BASE_DIR="/scratch/ywn1043/VLA-DFM"
VLA_PATH="${BASE_DIR}/models/openvla-7b"
DATA_ROOT="${BASE_DIR}/RLDS/modified_libero_rlds"
DATASET_NAME="libero_object_no_noops"
RUN_ROOT_DIR="${BASE_DIR}/checkpoints/ddopenvla-libero-object-smoke"

# --- Training params (short) ---
BATCH_SIZE=2
LEARNING_RATE=5e-4
NUM_STEPS_BEFORE_DECAY=10000
MAX_STEPS=3000
SAVE_FREQ=1000
SHUFFLE_BUFFER_SIZE=10000
LORA_RANK=16
TORCH_DTYPE="bfloat16"

NPROC=${SLURM_GPUS_ON_NODE:-2}


# --- Launch (PyTorch DDP) ---
torchrun --standalone --nnodes 1 --nproc-per-node ${NPROC} vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --use_discrete_diffusion True \
  --use_discrete_flow_matching False \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --num_steps_before_decay ${NUM_STEPS_BEFORE_DECAY} \
  --max_steps ${MAX_STEPS} \
  --save_freq ${SAVE_FREQ} \
  --shuffle_buffer_size ${SHUFFLE_BUFFER_SIZE} \
  --save_latest_checkpoint_only True \
  --image_aug False \
  --lora_rank ${LORA_RANK} \
  --torch_dtype "${TORCH_DTYPE}" \
  --wandb_entity "a10v-1" \
  --wandb_project "VLA-DFM" \
  --run_id_note "smoke-2xA100-dd-3k--$(date +%Y%m%d_%H%M)" \
  | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
