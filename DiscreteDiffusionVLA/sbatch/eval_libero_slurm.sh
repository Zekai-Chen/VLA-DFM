#!/bin/bash
#SBATCH --account=p32222
#SBATCH --partition=gengpu              # GPU partition (48 h max)
#SBATCH --gres=gpu:a100:4               # 4×A100 GPUs
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --time=47:59:00
#SBATCH --job-name=openvla-eval
#SBATCH --output=logs/openvla_eval_%j.out
#SBATCH --error=logs/openvla_eval_%j.err

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
export LIBERO_CONFIG_PATH=/projects/p32222/aTester/VLA-DFM/DiscreteDiffusionVLA/.libero

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

# --- GPU slotting config ---
# Use SLURM-provided GPU count by default; fall back to 4
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
MAX_PER_GPU=2
TOTAL_SLOTS=$((NUM_GPUS * MAX_PER_GPU))

# Dtype (eval path uses bf16 in code, keep explicit here)
TORCH_DTYPE="bfloat16"
export TORCH_DTYPE

# Optional: override GPU list
if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
else
  GPUS=()
  for ((i=0; i<NUM_GPUS; i++)); do GPUS+=("$i"); done
fi

# --- Base path + logging ---
BASE_DIR="/scratch/ywn1043/VLA-DFM"
LOG_DIR="${BASE_DIR}/logs/discrete_diffusion_libero_spatial/$(date +'%m%d_%H%M')"
mkdir -p "$LOG_DIR"

# --- Steps to evaluate ---
STEPS=(
  300000
  290000
  280000
  270000
  260000
  250000
  # add more checkpoints here
  70000
  60000
  50000
)

# --- Eval params (adjust paths) ---
CHECKPOINT_ROOT="/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke/openvla-7b+libero_object_no_noops+b2+lr-0.0005+lora-r16+dropout-0.0--smoke-2xA100--20260219_1208"
TASK_SUITE="libero_object"

# initialization
declare -a JOB_PIDS
for ((i=0; i<TOTAL_SLOTS; i++)); do
  JOB_PIDS[i]=0
 done

start_job() {
  local STEP=$1
  local SLOT=$2
  local GPU_INDEX=$(( SLOT / MAX_PER_GPU ))
  local GPU=${GPUS[$GPU_INDEX]}

  local CKPT_PATH="${CHECKPOINT_ROOT}--${STEP}_chkpt"
  if [[ -d "${CHECKPOINT_ROOT}" && -f "${CHECKPOINT_ROOT}/config.json" ]]; then
    CKPT_PATH="${CHECKPOINT_ROOT}"
  elif [[ -d "${CHECKPOINT_ROOT}/${STEP}_chkpt" ]]; then
    CKPT_PATH="${CHECKPOINT_ROOT}/${STEP}_chkpt"
  fi

  echo "[$(date +'%H:%M:%S')] START STEP=${STEP} on GPU=${GPU} (slot ${SLOT})"
  CUDA_VISIBLE_DEVICES=$GPU \
    python "${REPO_ROOT}/experiments/robot/libero/run_libero_eval.py" \
      --pretrained_checkpoint "${CKPT_PATH}" \
      --task_suite_name ${TASK_SUITE} \
      --use_l1_regression False \
      --use_diffusion False \
      --use_discrete_diffusion True \
      --use_film False \
      --num_images_in_input 2 \
      --use_proprio True \
      --topk_filter_thres 0.0 \
      --use_wandb True \
    > "$LOG_DIR/eval_${STEP}.log" 2>&1 &

  JOB_PIDS[$SLOT]=$!
}

# traverse all steps, start if there are empty slots
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

# wait
for pid in "${JOB_PIDS[@]}"; do
  [[ $pid -ne 0 ]] && wait "$pid"
 done

echo "All evaluations finished."
