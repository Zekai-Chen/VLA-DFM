#!/bin/bash
#SBATCH --account=p32222
#SBATCH --partition=gengpu              # GPU partition (48 h max)
#SBATCH --gres=gpu:a100:4               # 4×A100 GPUs
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --time=47:59:00                 # <= 48h limit on gengpu
#SBATCH --job-name=openvla-ft
#SBATCH --output=logs/openvla_ft_%j.out
#SBATCH --error=logs/openvla_ft_%j.err

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
if [ ! -f "${VENV_ACTIVATE}" ] && [ -f "${REPO_ROOT}/.venv-ubuntu-nvidia/bin/activate" ]; then
  VENV_ACTIVATE="${REPO_ROOT}/.venv-ubuntu-nvidia/bin/activate"
fi
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
export WANDB_NAME="openvla_ft_${SLURM_JOBID:-local}"

# --- Job params (adjust paths if needed) ---
BASE_DIR="/scratch/ywn1043/VLA-DFM"
VLA_PATH="${BASE_DIR}/models/openvla-7b"
DATA_ROOT="${BASE_DIR}/RLDS/modified_libero_rlds"
DATASET_NAME="libero_object_no_noops"
RUN_ROOT_DIR="${BASE_DIR}/checkpoints/ddopenvla-libero-object"
WANDB_ENTITY="a10v-1"
WANDB_PROJECT="openvla-ft"

# --- Training params ---
BATCH_SIZE=8
LEARNING_RATE=5e-4
NUM_STEPS_BEFORE_DECAY=100000
MAX_STEPS=320005
SAVE_FREQ=10000
SHUFFLE_BUFFER_SIZE=256000
LORA_RANK=32
TORCH_DTYPE="bfloat16"   # bfloat16 | float16 | float32

# --- DFM params ---
USE_DFM=true
DFM_SCHEDULE="cosine"
DFM_LOSS_MODE="generalized_kl"
DFM_TIME_EPS=1e-3
DFM_T_MIN=0.0
DFM_T_MAX=1.0
DFM_WEIGHT_CLIP=20.0

NPROC=${SLURM_GPUS_ON_NODE:-1}


# --- Launch (PyTorch DDP) ---
if [[ "${USE_DFM}" == "true" ]]; then
  torchrun --standalone --nnodes 1 --nproc-per-node ${NPROC} vla-scripts/finetune.py \
    --vla_path "${VLA_PATH}" \
    --data_root_dir "${DATA_ROOT}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --use_discrete_diffusion False \
    --use_discrete_flow_matching True \
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
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank ${LORA_RANK} \
    --torch_dtype "${TORCH_DTYPE}" \
    --dfm_schedule ${DFM_SCHEDULE} \
    --dfm_loss_mode ${DFM_LOSS_MODE} \
    --dfm_time_eps ${DFM_TIME_EPS} \
    --dfm_t_min ${DFM_T_MIN} \
    --dfm_t_max ${DFM_T_MAX} \
    --dfm_weight_clip ${DFM_WEIGHT_CLIP} \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --run_id_note "dfm--ctmc--$(date +%Y%m%d_%H%M)" \
    | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
else
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
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank ${LORA_RANK} \
    --torch_dtype "${TORCH_DTYPE}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --run_id_note "diffusion--$(date +%Y%m%d_%H%M)" \
    | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
fi
