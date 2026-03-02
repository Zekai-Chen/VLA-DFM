#!/bin/bash
#SBATCH --account=p32222
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=dd-sanity-smoke
#SBATCH --output=logs/dd_sanity_smoke_%j.out
#SBATCH --error=logs/dd_sanity_smoke_%j.err

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

# --- Run params ---
CHECKPOINT_ROOT="/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke/openvla-7b+libero_object_no_noops+b4+lr-0.0005+lora-r16+dropout-0.0--smoke-2xA100-dd-3k--20260302_0220"
DATA_ROOT="/scratch/ywn1043/VLA-DFM/RLDS/modified_libero_rlds"
DATASET_NAME="libero_object_no_noops"
NUM_BATCHES=${NUM_BATCHES:-5}
DD_NUM_STEPS=${DD_NUM_STEPS:-64}
CHECK_IMAGE_PARITY=${CHECK_IMAGE_PARITY:-True}

LOG_DIR="/scratch/ywn1043/VLA-DFM/logs/dd_sanity/$(date +'%m%d_%H%M')"
mkdir -p "$LOG_DIR"

echo "Running DD sanity check..."
python "${REPO_ROOT}/scripts/dfm_sanity_check.py" \
  --primary_mode dd \
  --checkpoint "${CHECKPOINT_ROOT}" \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --num_batches "${NUM_BATCHES}" \
  --dfm_num_steps "${DD_NUM_STEPS}" \
  --check_image_parity "${CHECK_IMAGE_PARITY}" \
  > "${LOG_DIR}/dd_sanity.log" 2>&1

echo "Sanity check finished. Log: ${LOG_DIR}/dd_sanity.log"
