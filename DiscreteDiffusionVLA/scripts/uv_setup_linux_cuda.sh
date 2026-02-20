#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION=${PYTHON_VERSION:-3.10}
VENV_DIR=${VENV_DIR:-.venv}
CUDA_WHL=${CUDA_WHL:-cu121}  # CUDA 12.4 module -> use cu121 wheels
UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.org/simple}
UV_INDEX=${UV_INDEX:-https://download.pytorch.org/whl/${CUDA_WHL}}
UV_INDEX_STRATEGY=${UV_INDEX_STRATEGY:-unsafe-best-match}

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"

export UV_DEFAULT_INDEX UV_INDEX UV_INDEX_STRATEGY

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
    --default-index "${UV_DEFAULT_INDEX}" \
    --index "${UV_INDEX}" \
    --index-strategy "${UV_INDEX_STRATEGY}"
fi

uv sync --frozen \
  --default-index "${UV_DEFAULT_INDEX}" \
  --index "${UV_INDEX}" \
  --index-strategy "${UV_INDEX_STRATEGY}"

# Pin PEFT + diffusers + hub + accelerate to compatible versions
uv pip install "peft==0.11.1" "diffusers==0.24.0" "accelerate==0.23.0" "huggingface_hub==0.19.4"

# Optional: install Flash Attention 2 for training
uv pip install packaging ninja
ninja --version; echo $?
uv pip install "flash-attn==2.5.5" --no-build-isolation

uv pip install -e LIBERO
uv pip install -r experiments/robot/libero/libero_requirements.txt