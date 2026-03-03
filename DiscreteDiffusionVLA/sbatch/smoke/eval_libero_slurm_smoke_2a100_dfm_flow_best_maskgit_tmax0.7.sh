#!/bin/bash
#SBATCH --account=p32222
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --time=47:00:00
#SBATCH --job-name=openvla-eval-dfm-maskgit-tmax0.7
#SBATCH --output=logs/openvla_eval_dfm_maskgit_tmax0.7_%j.out
#SBATCH --error=logs/openvla_eval_dfm_maskgit_tmax0.7_%j.err

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
WANDB_ENTITY=${WANDB_ENTITY:-a10v-1}
WANDB_PROJECT=${WANDB_PROJECT:-VLA-DFM}
export WANDB_ENTITY WANDB_PROJECT
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

# --- Base path + logging ---
BASE_DIR="/scratch/ywn1043/VLA-DFM"
LOG_DIR="${BASE_DIR}/logs/eval_dfm_maskgit_tmax0.7/$(date +'%m%d_%H%M')"
mkdir -p "$LOG_DIR"

# --- Smoke eval params ---
CHECKPOINT_ROOT="/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-20k-maskfix-moremask-tmax0p7/openvla-7b+libero_object_no_noops+b4+lr-0.0005+lora-r16+dropout-0.0--smoke-2xA100-20k-tmax0p7--20260302_1313"
TASK_SUITE="libero_object"
NUM_TRIALS=50
DFM_DEBUG=${DFM_DEBUG:-True}
DFM_DEBUG_LEVEL=${DFM_DEBUG_LEVEL:-1}
DFM_FAIL_FAST=${DFM_FAIL_FAST:-False}
DFM_NUM_STEPS=${DFM_NUM_STEPS:-0}
DFM_MASKGIT_NUM_STEPS=${DFM_MASKGIT_NUM_STEPS:-0}
DFM_MASKGIT_SCHEDULE=${DFM_MASKGIT_SCHEDULE:-auto}
DFM_TEMPERATURE=${DFM_TEMPERATURE:-0.0}
DFM_SCHEDULE=${DFM_SCHEDULE:-auto}
DFM_EARLY_EXIT=${DFM_EARLY_EXIT:-False}
DFM_DECODE_MODE=${DFM_DECODE_MODE:-maskgit}
NUM_OPEN_LOOP_STEPS=${NUM_OPEN_LOOP_STEPS:-8}
SYNC_MODEL_LOGIC=${SYNC_MODEL_LOGIC:-False}
USE_CHECKPOINT_DEFAULTS=${USE_CHECKPOINT_DEFAULTS:-True}

# Use 1 job per GPU for smoke
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
MAX_PER_GPU=1
TOTAL_SLOTS=$((NUM_GPUS * MAX_PER_GPU))

GPUS=()
for ((i=0; i<NUM_GPUS; i++)); do GPUS+=("$i"); done

# Evaluate a small set of checkpoints
STEPS=(
  3000
)

# --- Preflight: validate checkpoint config matches DFM eval expectations ---
FIRST_STEP="${STEPS[0]}"
CKPT_PATH="${CHECKPOINT_ROOT}--${FIRST_STEP}_chkpt"
if [[ -d "${CHECKPOINT_ROOT}" && -f "${CHECKPOINT_ROOT}/config.json" ]]; then
  CKPT_PATH="${CHECKPOINT_ROOT}"
elif [[ -d "${CHECKPOINT_ROOT}/${FIRST_STEP}_chkpt" ]]; then
  CKPT_PATH="${CHECKPOINT_ROOT}/${FIRST_STEP}_chkpt"
fi
export CKPT_PATH

python - <<'PY'
import json, os, sys
ckpt = os.environ.get("CKPT_PATH")
if not ckpt:
    print("CKPT_PATH not set; skipping config validation.")
    sys.exit(0)
cfg_path = os.path.join(ckpt, "config.json")
if not os.path.exists(cfg_path):
    print(f"WARNING: config.json not found at {cfg_path}")
    sys.exit(0)
cfg = json.load(open(cfg_path))
print("Checkpoint config summary:")
for k in [
    "use_discrete_flow_matching",
    "use_discrete_diffusion",
    "use_mask_token",
    "dfm_schedule",
    "dfm_loss_mode",
    "dfm_train_mode",
    "dfm_time_eps",
    "dfm_t_min",
    "dfm_t_max",
    "dfm_weight_clip",
]:
    print(f"  {k}: {cfg.get(k)}")
if not cfg.get("use_discrete_flow_matching", False):
    print("WARNING: config.use_discrete_flow_matching is False. Eval may mismatch training.")
PY

python - <<'PY'
import os
from transformers import AutoProcessor

ckpt = os.environ.get("CKPT_PATH")
if not ckpt:
    print("CKPT_PATH not set; skipping tokenizer validation.")
    raise SystemExit(0)

try:
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    tok = processor.tokenizer
    print("Tokenizer summary:")
    print("  vocab_size:", tok.vocab_size)
    print("  vocab_len:", len(tok))
    print("  pad_token_id:", tok.pad_token_id)
    print("  mask_token_id:", tok.mask_token_id)
    if tok.pad_token_id is None or tok.mask_token_id is None:
        print("WARNING: pad_token_id or mask_token_id is None.")
    elif tok.pad_token_id >= len(tok) or tok.mask_token_id >= len(tok):
        print("WARNING: pad/mask token id out of range for len(tokenizer).")
    # derive action range from config + tokenizer
    cfg = getattr(processor, "config", None)
except Exception:
    # Fallback: load config.json directly if processor init fails
    import json
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tok = None

def _get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return cfg.get(key, default)

anchor = _get(cfg, "action_vocab_anchor", "pad")
n_bins = _get(cfg, "n_action_bins", 256)
pad_id = _get(cfg, "pad_token_id", None)
vocab_len = len(tok) if tok is not None else None
if anchor == "pad" and pad_id is not None:
    action_end = int(pad_id)
elif anchor == "vocab_size" and vocab_len is not None:
    action_end = int(vocab_len)
else:
    action_end = None
action_begin = action_end - int(n_bins) if action_end is not None else None
print("Action vocab range:")
print("  anchor:", anchor)
print("  n_action_bins:", n_bins)
print("  action_begin:", action_begin)
print("  action_end:", action_end)
PY

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
      --sync_model_logic ${SYNC_MODEL_LOGIC} \
      --use_checkpoint_defaults ${USE_CHECKPOINT_DEFAULTS} \
      --task_suite_name ${TASK_SUITE} \
      --num_trials_per_task ${NUM_TRIALS} \
      --use_l1_regression False \
      --use_diffusion False \
      --use_discrete_diffusion False \
      --use_discrete_flow_matching True \
      --use_film False \
      --center_crop True \
      --num_images_in_input 2 \
      --num_open_loop_steps ${NUM_OPEN_LOOP_STEPS} \
      --use_proprio True \
      --topk_filter_thres 0.0 \
      --dfm_debug ${DFM_DEBUG} \
      --dfm_debug_level ${DFM_DEBUG_LEVEL} \
      --dfm_fail_fast ${DFM_FAIL_FAST} \
      --dfm_num_steps ${DFM_NUM_STEPS} \
      --dfm_maskgit_num_steps ${DFM_MASKGIT_NUM_STEPS} \
      --dfm_maskgit_schedule ${DFM_MASKGIT_SCHEDULE} \
      --dfm_schedule ${DFM_SCHEDULE} \
      --dfm_temperature ${DFM_TEMPERATURE} \
      --dfm_early_exit ${DFM_EARLY_EXIT} \
      --dfm_decode_mode ${DFM_DECODE_MODE} \
      --local_log_dir "${LOG_DIR}" \
      --use_wandb True \
      --wandb_entity "${WANDB_ENTITY}" \
      --wandb_project "${WANDB_PROJECT}" \
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
