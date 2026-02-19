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

# --- DFM params (disabled for discrete diffusion) ---
USE_DFM=false
DFM_SCHEDULE="cosine"
DFM_LOSS_MODE="generalized_kl"
DFM_TIME_EPS=1e-3
DFM_T_MIN=0.0
DFM_T_MAX=1.0
DFM_WEIGHT_CLIP=20.0

NPROC=${SLURM_GPUS_ON_NODE:-1}

cd /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA

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
    --run_id_note "discrete-diffusion--$(date +%Y%m%d_%H%M)" \
    | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
fi
