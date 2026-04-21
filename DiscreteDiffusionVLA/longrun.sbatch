#!/bin/bash
#SBATCH -A coreai_dlalgo_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -J dfm_longrun
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# DFM: always runs scripts/train_dfm.sh --resume (latest checkpoint under RUN_ROOT, see train_dfm.sh).
#
# Usage:
#   sbatch longrun.sbatch
#
# Optional: export CONDA_ENV=ddvla (default), VLA_PATH, DATA_ROOT, RUN_ROOT before sbatch.

DDVLA_ROOT="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/Programs/zk/VLA-DFM/DiscreteDiffusionVLA"
CONDA_SH="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/Apps/miniforge3/etc/profile.d/conda.sh"
CACHE_ENV="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/cache_env.sh"
CONDA_ENV="${CONDA_ENV:-dfm-vla}"
NGPU=8

mkdir -p logs

echo "========== Job =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)  PWD=${PWD}"
echo "DDVLA_ROOT=${DDVLA_ROOT}  (train_ddvla.sh --resume)"
echo "========================="

SRUN_PID=""

graceful_exit() {
    echo "[Graceful Exit] Caught termination signal."

    if [[ -n "${SRUN_PID}" ]] && kill -0 "${SRUN_PID}" 2>/dev/null; then
        echo "Forwarding TERM to srun process ${SRUN_PID} ..."
        kill -TERM "${SRUN_PID}" 2>/dev/null || true
        wait "${SRUN_PID}" || true
    fi

    exit 0
}

trap graceful_exit SIGTERM SIGINT

echo "Starting workload..."

set +e
srun \
    --ntasks=1 \
    --time="${SLURM_TIMELIMIT:-04:00:00}" \
    --gpus="${NGPU}" \
    bash -lc "
        set -euo pipefail

        source ${CACHE_ENV}
        source ${CONDA_SH}
        # conda activate.d hooks (e.g. cuda-nvcc) reference unset vars; nounset breaks them.
        set +u
        conda activate ${CONDA_ENV}
        cd ${DDVLA_ROOT}
        set -u
        # export WANDB_API_KEY=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
        # zekai‘s wandb key
        export WANDB_API_KEY=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
        # Single-quoted bash -lc: do NOT use \${CONDA_PREFIX} — that leaves a literal ${CONDA_PREFIX} in CUDA_HOME.
        export CUDA_HOME=\"\${CONDA_PREFIX}\"
        export CMAKE_PREFIX_PATH=\"\${CONDA_PREFIX}\"
        export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"
        export CPATH=\"\${CONDA_PREFIX}/targets/x86_64-linux/include:\${CONDA_PREFIX}/include:\${CPATH:-}\"
        export LIBRARY_PATH=\"\${CONDA_PREFIX}/targets/x86_64-linux/lib:\${CONDA_PREFIX}/lib:\${LIBRARY_PATH:-}\"

        bash scripts/train_dfm.sh --resume
    " &
SRUN_PID=$!

wait "${SRUN_PID}"
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo "Workload exited successfully."
    exit 0
else
    echo "Workload failed with EXIT_CODE=${EXIT_CODE}"
    exit "${EXIT_CODE}"
fi
