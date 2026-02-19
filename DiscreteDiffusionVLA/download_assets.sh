#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/base_dir"
  exit 1
fi

BASE_DIR="$1"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="${MODEL_ID:-openvla/openvla-7b}"
DOWNLOAD_BASE_MODEL="${DOWNLOAD_BASE_MODEL:-1}"
DOWNLOAD_LIBERO_DATASET="${DOWNLOAD_LIBERO_DATASET:-1}"
DOWNLOAD_LIBERO_REPO="${DOWNLOAD_LIBERO_REPO:-1}"
DOWNLOAD_FINETUNED="${DOWNLOAD_FINETUNED:-0}"

# Subfolders (customize if desired)
MODELS_DIR="${BASE_DIR}/models"
RLDS_DIR="${BASE_DIR}/RLDS"
LIBERO_DIR="${BASE_DIR}/LIBERO"
CHECKPOINTS_DIR="${BASE_DIR}/checkpoints"

# Optional: set HF token via HF_TOKEN env var
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

mkdir -p "${MODELS_DIR}" "${RLDS_DIR}" "${LIBERO_DIR}" "${CHECKPOINTS_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found"
  exit 1
fi

HF_CLI="huggingface-cli"
if ! command -v huggingface-cli >/dev/null 2>&1; then
  if python -m huggingface_hub.cli.hf --help >/dev/null 2>&1; then
    HF_CLI="python -m huggingface_hub.cli.hf"
  else
    echo "ERROR: huggingface-cli not found. Install with: pip install -U huggingface_hub"
    exit 1
  fi
fi

if [[ "${DOWNLOAD_BASE_MODEL}" == "1" ]]; then
  echo "Downloading base model: ${MODEL_ID}"
  ${HF_CLI} download "${MODEL_ID}" \
    --local-dir "${MODELS_DIR}/${MODEL_ID##*/}"
fi

if [[ "${DOWNLOAD_LIBERO_REPO}" == "1" ]]; then
  if [[ ! -d "${LIBERO_DIR}/LIBERO" ]]; then
    echo "Cloning LIBERO repo"
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}/LIBERO"
  else
    echo "LIBERO repo already exists: ${LIBERO_DIR}/LIBERO"
  fi
fi

if [[ "${DOWNLOAD_LIBERO_DATASET}" == "1" ]]; then
  if ! command -v git-lfs >/dev/null 2>&1; then
    if command -v git >/dev/null 2>&1; then
      if ! git lfs --version >/dev/null 2>&1; then
        echo "ERROR: git-lfs not found. Install git-lfs and re-run."
        exit 1
      fi
    fi
  fi

  if [[ ! -d "${RLDS_DIR}/modified_libero_rlds" ]]; then
    echo "Cloning LIBERO RLDS dataset (openvla/modified_libero_rlds)"
    git lfs install
    git clone https://huggingface.co/datasets/openvla/modified_libero_rlds \
      "${RLDS_DIR}/modified_libero_rlds"
  else
    echo "Dataset already exists: ${RLDS_DIR}/modified_libero_rlds"
  fi
fi

if [[ "${DOWNLOAD_FINETUNED}" == "1" ]]; then
  FINETUNED_IDS=(
    "moojink/openvla-7b-oft-finetuned-libero-spatial"
    "moojink/openvla-7b-oft-finetuned-libero-object"
    "moojink/openvla-7b-oft-finetuned-libero-goal"
    "moojink/openvla-7b-oft-finetuned-libero-10"
    "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
  )
  for ID in "${FINETUNED_IDS[@]}"; do
    echo "Downloading finetuned checkpoint: ${ID}"
    ${HF_CLI} download "${ID}" \
      --local-dir "${MODELS_DIR}/${ID##*/}"
  done
fi

echo "Done."
