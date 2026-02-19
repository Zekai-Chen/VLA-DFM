#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION=${PYTHON_VERSION:-3.10}
VENV_DIR=${VENV_DIR:-.venv}
CUDA_WHL=${CUDA_WHL:-cu121}  # CUDA 12.4 module -> use cu121 wheels

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"

export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/${CUDA_WHL}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it first: https://astral.sh/uv" >&2
  exit 1
fi

uv venv -p "${PYTHON_VERSION}" "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [ ! -f "uv.lock" ]; then
  echo "uv.lock not found. Generating for linux + CUDA (${CUDA_WHL})..."
  uv lock \
    --python "${PYTHON_VERSION}" \
    --extra-index-url "${UV_EXTRA_INDEX_URL}"
fi

uv sync --frozen --extra-index-url "${UV_EXTRA_INDEX_URL}"

# Optional: install Flash Attention 2 for training
uv pip install packaging ninja
ninja --version; echo $?
uv pip install "flash-attn==2.5.5" --no-build-isolation
