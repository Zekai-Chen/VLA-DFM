#!/bin/bash
#SBATCH --account=p32222
#SBATCH --partition=gengpu              # GPU partition (48 h max)
#SBATCH --gres=gpu:a100:4               # 4×A100 GPUs
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --time=47:59:00
#SBATCH --job-name=openvla-eval-dd
#SBATCH --output=logs/openvla_eval_dd_%j.out
#SBATCH --error=logs/openvla_eval_dd_%j.err

set -euo pipefail

# --- Environment (cluster-standard) ---
module load python-miniconda3
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate ddopenvla
fi

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
BASE_DIR="/scratch/ywn1043/finetune"
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
CHECKPOINT_ROOT="${BASE_DIR}/checkpoints/ddopenvla-libero-object/openvla-7b+libero_object_no_noops"    # expects ${CHECKPOINT_ROOT}--${STEP}_chkpt
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

  echo "[$(date +'%H:%M:%S')] START STEP=${STEP} on GPU=${GPU} (slot ${SLOT})"
  CUDA_VISIBLE_DEVICES=$GPU \
    python ../experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${CHECKPOINT_ROOT}--${STEP}_chkpt" \
      --task_suite_name ${TASK_SUITE} \
      --use_l1_regression False \
      --use_diffusion False \
      --use_discrete_diffusion True \
      --use_film False \
      --num_images_in_input 2 \
      --use_proprio True \
      --topk_filter_thres 0.0 \
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
