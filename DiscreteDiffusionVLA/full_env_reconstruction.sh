#!/usr/bin/env bash
set -euo pipefail

module load gcc/12.4.0-gcc-8.5.0 && module load cuda/12.4.0-gcc-12.4.0
module load git

REPO_ROOT=${1:-/projects/p32222/aTester/VLA-DFM/DiscreteDiffusionVLA}
cd "${REPO_ROOT}"

PYTHON_VERSION=3.10 CUDA_WHL=cu121 ./scripts/uv_setup_linux_cuda.sh

module load ca-certificates-mozilla/2023-05-30-gcc-8.5.0
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE
export GIT_SSL_CAINFO=$SSL_CERT_FILE

source .venv/bin/activate
uv pip install --no-cache "flash-attn==2.5.5" --no-build-isolation
