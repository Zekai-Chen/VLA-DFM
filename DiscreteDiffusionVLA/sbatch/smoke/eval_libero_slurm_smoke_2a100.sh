#!/bin/bash
#SBATCH --account=p32222
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=openvla-eval-smoke
#SBATCH --output=logs/openvla_eval_smoke_%j.out
#SBATCH --error=logs/openvla_eval_smoke_%j.err

set -euo pipefail

# --- Environment (cluster-standard) ---
module load python-miniconda3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
VENV_PATH=${VENV_PATH:-${REPO_ROOT}/.venv}
if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Venv not found at ${VENV_PATH}. Run scripts/uv_setup_linux_cuda.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

module load gcc/12.4.0-gcc-8.5.0 && module load cuda/12.4.0-gcc-12.4.0
module load git

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

# --- Base path + logging ---
BASE_DIR="/scratch/ywn1043/VLA-DFM"
LOG_DIR="${BASE_DIR}/logs/eval_smoke/$(date +'%m%d_%H%M')"
mkdir -p "$LOG_DIR"

# --- Smoke eval params ---
CHECKPOINT_ROOT="${BASE_DIR}/checkpoints/ddopenvla-libero-object/openvla-7b+libero_object_no_noops"
TASK_SUITE="libero_object"
NUM_TRIALS=2

# Use 1 job per GPU for smoke
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
MAX_PER_GPU=1
TOTAL_SLOTS=$((NUM_GPUS * MAX_PER_GPU))

GPUS=()
for ((i=0; i<NUM_GPUS; i++)); do GPUS+=("$i"); done

# Evaluate a small set of checkpoints
STEPS=(
  20000
)

declare -a JOB_PIDS
for ((i=0; i<TOTAL_SLOTS; i++)); do
  JOB_PIDS[i]=0
 done

start_job() {
  local STEP=$1
  local SLOT=$2
  local GPU_INDEX=$(( SLOT / MAX_PER_GPU ))
  local GPU=${GPUS[$GPU_INDEX]}

  echo "[$(date +'%H:%M:%S')] START STEP=${STEP} on GPU=${GPU} (slot ${SLOT})"
  CUDA_VISIBLE_DEVICES=$GPU \
    python ../../experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${CHECKPOINT_ROOT}--${STEP}_chkpt" \
      --task_suite_name ${TASK_SUITE} \
      --num_trials_per_task ${NUM_TRIALS} \
      --use_l1_regression False \
      --use_diffusion False \
      --use_discrete_diffusion True \
      --use_film False \
      --num_images_in_input 2 \
      --use_proprio True \
      --topk_filter_thres 0.0 \
      --use_wandb False \
    > "$LOG_DIR/eval_${STEP}.log" 2>&1 &

  JOB_PIDS[$SLOT]=$!
}

for STEP in "${STEPS[@]}"; do
  while :; do
    for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
      pid=${JOB_PIDS[slot]}
      if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
        start_job "$STEP" "$slot"
        break 2
      fi
    done
    sleep 2
  done
done

for pid in "${JOB_PIDS[@]}"; do
  [[ $pid -ne 0 ]] && wait "$pid"
 done

echo "Eval smoke finished."
