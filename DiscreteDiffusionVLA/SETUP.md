# Setup Instructions

## Set Up UV Environment (Recommended for Clusters)

The easiest way to keep installs reproducible on Ubuntu/CUDA machines is to use `uv` with a lockfile.

```bash
# From the repo root
./scripts/uv_setup_linux_cuda.sh

# Optional overrides
# PYTHON_VERSION=3.10 CUDA_WHL=cu121 VENV_DIR=.venv ./scripts/uv_setup_linux_cuda.sh
# For CUDA 11.8, set CUDA_WHL=cu118
```

Notes:
- The lockfile is platform-specific. Generate `uv.lock` on the same OS/arch as your target machine (Linux x86_64).
- `uv` will use the PyTorch CUDA wheels via `https://download.pytorch.org/whl/<cuXXX>`.
- If your cluster loads `cuda/12.4.0`, keep the default `CUDA_WHL=cu121` (recommended).

### Flash-Attn (Optional) + Cluster SSL Fixes

`flash-attn` is optional and only improves speed/memory. If install fails due to SSL or UV cache issues on clusters, use:

```bash
module load ca-certificates-mozilla/2023-05-30-gcc-8.5.0

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE
export GIT_SSL_CAINFO=$SSL_CERT_FILE

rm -rf ~/.uv/sdists-v9
uv cache clean || true

source .venv/bin/activate
uv pip install --no-cache "flash-attn==2.5.5" --no-build-isolation
```

## Set Up Conda Environment (Alternative)

```bash
# Create and activate conda environment
conda create -n ddopenvla python=3.10 -y
conda activate ddopenvla

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/Liang-ZX/DiscreteDiffusionVLA.git
cd DiscreteDiffusionVLA
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
