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
# Absolute chdir + stdout/stderr: relative logs/ breaks when submit CWD is Slurm spool
# (/cm/local/.../var/spool/job<JOBID>/ → mkdir .../logs permission denied).
#SBATCH --chdir=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/Programs/zk/VLA-DFM/DiscreteDiffusionVLA
#SBATCH -o /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/Programs/zk/VLA-DFM/DiscreteDiffusionVLA/logs/%x_%j.out
#SBATCH -e /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/Programs/zk/VLA-DFM/DiscreteDiffusionVLA/logs/%x_%j.err

set -euo pipefail

# DFM: always runs scripts/train_dfm.sh --resume (latest checkpoint under RUN_ROOT, see train_dfm.sh).
#
# Arg1 SEG — chain segment index (1..MAX_SEG). Passed through by sbatch requeue; no state files / folders.
#
# Usage:
#   sbatch longrun_dfm.sbatch
#   sbatch longrun_dfm.sbatch 1
#
# Parallel chains: use distinct --job-name so singleton does not block each other.
#
# Optional: export CONDA_ENV=dfm-vla (default), MAX_SEG (default 25), VLA_PATH, DATA_ROOT, RUN_ROOT before sbatch.
# HF cache: scripts/train_dfm.sh redirects HF_HOME to REPO/.hf_cache (not home_cache).
#           Set DFM_USE_SHARED_HF_CACHE=1 before sbatch to keep cache_env.sh HF paths.

# Slurm: batch file lives in spool, so dirname(BASH_SOURCE) is wrong. #SBATCH --chdir
# already ran — PWD is the repo. Local bash: use this script's directory.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    DFM_ROOT="${DFM_ROOT:-${PWD}}"
else
    DFM_ROOT="${DFM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
fi
# Chain must re-submit this file on the shared FS (default was wrong: longrun.sbatch does not exist).
LONGRUN_SBATCH="${LONGRUN_SBATCH:-${DFM_ROOT}/longrun_dfm.sbatch}"
CONDA_SH="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/Apps/miniforge3/etc/profile.d/conda.sh"
CACHE_ENV="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/cache_env.sh"
CONDA_ENV="${CONDA_ENV:-dfm-vla}"
NGPU=8
MAX_SEG="${MAX_SEG:-25}"
SEG="${1:-1}"

mkdir -p "${DFM_ROOT}/logs"

if ! [[ "${SEG}" =~ ^[0-9]+$ ]] || [[ "${SEG}" -lt 1 ]]; then
    echo "Invalid SEG='${SEG}' (need positive integer). Usage: sbatch longrun_dfm.sbatch [SEG]" >&2
    exit 1
fi

if [[ "${SEG}" -gt "${MAX_SEG}" ]]; then
    echo "SEG=${SEG} > MAX_SEG=${MAX_SEG}; chain finished, exiting."
    exit 0
fi

echo "========== Job =========="
echo "SEG=${SEG}/${MAX_SEG}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)  PWD=${PWD}"
echo "DFM_ROOT=${DFM_ROOT}  (train_dfm.sh --resume)"
echo "========================="

############################
# Pre-schedule next segment (same pattern as Psi0-hojune / DDVLA longrun: survives SIGTERM / pre-timeout)
############################
if [[ "${SEG}" -lt "${MAX_SEG}" ]]; then
    echo "Scheduling follow-up: SEG=$((SEG + 1)) / MAX_SEG=${MAX_SEG}"
    sbatch \
        --dependency=singleton \
        --job-name="${SLURM_JOB_NAME:-dfm_longrun}" \
        -D "${DFM_ROOT}" \
        -o "${DFM_ROOT}/logs/%x_%j.out" \
        -e "${DFM_ROOT}/logs/%x_%j.err" \
        "${LONGRUN_SBATCH}" "$((SEG + 1))"
fi

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
        cd ${DFM_ROOT}
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
