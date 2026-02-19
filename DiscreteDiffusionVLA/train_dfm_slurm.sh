#!/bin/bash
#SBATCH --account=REPLACE_ME
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --time=47:59:00
#SBATCH --job-name=openvla-dfm
#SBATCH --output=logs/openvla_dfm_%j.out
#SBATCH --error=logs/openvla_dfm_%j.err

set -euo pipefail

# --- Environment (customize for your cluster) ---
module load python-miniconda3
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate ddopenvla
fi

module load gcc/12.4.0-gcc-8.5.0 || true
module load cuda/12.4.0-gcc-12.4.0 || true
module load git || true

# --- Diagnostics ---
mkdir -p logs

echo "SLURM_JOBID=${SLURM_JOBID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"

python -V || true
python -c "import torch; print('CUDA:', torch.version.cuda, 'GPUs:', torch.cuda.device_count())" || true

# --- Paths & params (edit these) ---
VLA_PATH="openvla/openvla-7b"
DATA_ROOT="/path/to/modified_libero_rlds"
DATASET_NAME="libero_object_no_noops"
RUN_ROOT_DIR="/path/to/checkpoints/ddopenvla-libero-object-dfm"
WANDB_ENTITY="your-wandb-entity"
WANDB_PROJECT="your-wandb-project"

# --- Training params ---
BATCH_SIZE=8
LEARNING_RATE=5e-4
NUM_STEPS_BEFORE_DECAY=100000
MAX_STEPS=320005
SAVE_FREQ=10000
SHUFFLE_BUFFER_SIZE=256000
LORA_RANK=32

# --- DFM params ---
DFM_SCHEDULE="cosine"
DFM_LOSS_MODE="generalized_kl"
DFM_TIME_EPS=1e-3
DFM_T_MIN=0.0
DFM_T_MAX=1.0
DFM_WEIGHT_CLIP=20.0

# --- Optional: quick smoke test ---
# Set SMOKE_TEST=1 to run 2 steps with tiny buffers
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  MAX_STEPS=2
  SAVE_FREQ=1
  BATCH_SIZE=1
  SHUFFLE_BUFFER_SIZE=1000
  export WANDB_MODE=offline
  export WANDB_DISABLED=true
fi

NPROC=${SLURM_GPUS_ON_NODE:-1}

# --- Launch (PyTorch DDP) ---
cd /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA

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
