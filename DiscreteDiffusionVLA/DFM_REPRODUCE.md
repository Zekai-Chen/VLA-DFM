# Reproducing DFM-VLA SFT on LIBERO-Object

This guide reproduces supervised fine-tuning (SFT) of the **Discrete Flow Matching (DFM)** variant on LIBERO-Object, using the same training scale as the DDVLA paper (320k steps × batch 64).

**Repo**: https://github.com/a10v/VLA-DFM (master branch)

## 1. Setup Environment

```bash
# Create conda env
conda create -n dfm-vla python=3.10 -y
conda activate dfm-vla

# Clone DFM-VLA repo and install (this pulls torch 2.2.0 as dependency)
git clone https://github.com/a10v/VLA-DFM.git
cd VLA-DFM/DiscreteDiffusionVLA
pip install -e .

# IMPORTANT: upgrade PyTorch AFTER pip install -e . (which downgrades to 2.2.0)
# A100/H100 (sm_80/sm_90):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Fix dependency versions
pip install peft==0.18.1 transformers==4.49.0 "diffusers>=0.27" "numpy<2.0"
pip install tensorflow tensorflow_datasets dlimp draccus

# Flash Attention (optional, ~10% speedup, A100/H100 only)
pip install packaging ninja
pip install flash-attn --no-build-isolation  # compiles from source, ~15 min

# IMPORTANT: numpy must be <2.0 — install LAST (other packages may upgrade it)
pip install "numpy<2.0"
```

## 2. Setup LIBERO Evaluation Environment

```bash
# LIBERO is already included in the repo at DiscreteDiffusionVLA/LIBERO/
pip install -e LIBERO
# Fix: copy libero to site-packages
cp -r LIBERO/libero $(python -c "import site; print(site.getsitepackages()[0])")/libero

# Install LIBERO requirements
pip install robosuite==1.4.1 mujoco bddl easydict cloudpickle gym "imageio[ffmpeg]"
sudo apt-get install -y libosmesa6-dev libegl1-mesa-dev libgl1-mesa-dev

# IMPORTANT: robosuite pulls numpy>=2.0 — must downgrade AGAIN
pip install "numpy<2.0"

# Setup LIBERO config (LIBERO_ROOT = path to DiscreteDiffusionVLA/LIBERO)
LIBERO_ROOT="$(pwd)/LIBERO"
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << YAML
assets: ${LIBERO_ROOT}/libero/libero/./assets
bddl_files: ${LIBERO_ROOT}/libero/libero/./bddl_files
benchmark_root: ${LIBERO_ROOT}/libero/libero
datasets: ${LIBERO_ROOT}/libero/libero/../datasets
init_states: ${LIBERO_ROOT}/libero/libero/./init_files
init_files: ${LIBERO_ROOT}/libero/libero/./init_files
YAML
```

## 3. Download Data

```bash
# Base model (~15GB)
huggingface-cli download openvla/openvla-7b --local-dir ~/data/models/openvla-7b

# LIBERO RLDS dataset (~20GB, needs git-lfs)
sudo apt-get install -y git-lfs && git lfs install
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds ~/data/RLDS/modified_libero_rlds
```

## 4. Verify Installation

```bash
python -c "
import torch, peft, transformers, numpy
print('torch:', torch.__version__)       # should be 2.5.1+cu121
print('peft:', peft.__version__)         # should be 0.18.1
print('transformers:', transformers.__version__)  # should be 4.49.0
print('numpy:', numpy.__version__)       # should be 1.x (not 2.x)
print('gpus:', torch.cuda.device_count())
"
```

**Troubleshooting**: If torch version shows wrong or imports fail, check for stale user site-packages:
```bash
python -c "import torch; print(torch.__file__)"
# If path is ~/.local/... instead of conda env, either:
#   pip uninstall torch  (from ~/.local)
# or prefix all commands with:
#   export PYTHONNOUSERSITE=1
```

## 5. Setup Weights & Biases

```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

## 6. Training

### Fresh start (paper configuration)

```bash
bash scripts/train_dfm.sh
```

### Resume from checkpoint

```bash
# Auto-find latest checkpoint:
bash scripts/train_dfm.sh --resume

# Resume from specific checkpoint:
bash scripts/train_dfm.sh --resume ~/checkpoints/dfm-vla-320k/<run_dir>/checkpoint-7000
```

### Training configuration

The script `scripts/train_dfm.sh` runs with these parameters:

- GPUs: 8 (DDP, auto-detected)
- Batch size: 8 per GPU × 8 GPUs = 64 total
- Max steps: 320,000
- Learning rate: 5e-4
- LR decay starts: step 100,000
- LoRA rank: 32
- Image augmentation: True (random crop 90% area)
- DFM schedule: cosine
- DFM t_max: 0.7
- DFM loss: generalized KL
- Legacy DFM mode: True (action vocab anchor=legacy, range [31743, 31999))
- Checkpoint saved every 7,000 steps (~3 hours on 8×A100-80GB)

### Training time estimates

| Hardware | Batch/GPU | Total Batch | Speed | Time |
|----------|-----------|-------------|-------|------|
| 8× A100-80GB SXM4 | 8 | 64 | ~1.55 s/it | ~6 days |
| 1× B200-183GB | 24 | 24 | ~1.1 s/it | ~4 days* |

*B200 single-GPU has lower total batch (24 vs 64), so total samples seen = 37.5% of paper config.

## 7. Evaluation

```bash
# Standard LIBERO eval (500 trials = 10 tasks × 50 episodes)
export MUJOCO_GL=osmesa
bash scripts/eval_ddvla.sh ~/checkpoints/dfm-vla-320k/<run_dir>/checkpoint-320000

# Quick eval (20 trials for fast iteration)
bash scripts/eval_ddvla.sh ~/checkpoints/dfm-vla-320k/<run_dir>/checkpoint-320000 2
```

Note: `eval_ddvla.sh` works for both DD and DFM checkpoints — it reads the model config to determine the decode method.

## 8. Key Differences: DFM vs DD

| | Discrete Diffusion (DD) | Discrete Flow Matching (DFM) |
|---|---|---|
| Flag | `--use_discrete_diffusion True` | `--use_discrete_flow_matching True` |
| Masking | Random binary mask | Mixture path with kappa(t) schedule |
| Loss | Masked cross-entropy | Generalized KL divergence |
| Schedule | N/A | cosine (kappa) |
| Decode | Iterative unmasking | MaskGIT or CTMC |
| Paper | DDVLA (Liang et al.) | Our method |

Both methods share the same base model (OpenVLA-7B), LoRA fine-tuning, and LIBERO evaluation pipeline.
