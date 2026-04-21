# Reproducing Discrete Diffusion VLA on LIBERO-Object

## 1. Setup Environment

```bash
# Create conda env
conda create -n ddvla python=3.10 -y
conda activate ddvla

# PyTorch (use cu121 for A100/H100, nightly for B200)
# A100/H100:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# B200 (sm_100, needs nightly):
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Clone and install DDVLA
git clone https://github.com/Liang-ZX/DiscreteDiffusionVLA.git DDVLA
cd DDVLA
pip install -e .

# Fix dependency versions
pip install "peft>=0.17,<0.19" "transformers>=4.45,<4.50" "diffusers>=0.27" "numpy<2.0"
pip install tensorflow tensorflow_datasets dlimp draccus

# Flash Attention (A100/H100 only, optional but recommended)
pip install packaging ninja
pip install "flash-attn>=2.5" --no-build-isolation

# LIBERO evaluation environment
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
# Fix: copy libero package to site-packages
cp -r LIBERO/libero $(python -c "import site; print(site.getsitepackages()[0])")/libero
pip install robosuite==1.4.1 mujoco bddl easydict cloudpickle gym imageio[ffmpeg]
sudo apt-get install -y libosmesa6-dev libegl1-mesa-dev libgl1-mesa-dev

# Setup LIBERO config
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'YAML'
assets: /path/to/LIBERO/libero/libero/./assets
bddl_files: /path/to/LIBERO/libero/libero/./bddl_files
benchmark_root: /path/to/LIBERO/libero/libero
datasets: /path/to/LIBERO/libero/libero/../datasets
init_states: /path/to/LIBERO/libero/libero/./init_files
init_files: /path/to/LIBERO/libero/libero/./init_files
YAML
# NOTE: update /path/to/LIBERO to your actual LIBERO repo path
```

## 2. Download Data

```bash
# Base model (~15GB)
huggingface-cli download openvla/openvla-7b --local-dir ~/data/models/openvla-7b

# LIBERO RLDS dataset (~20GB, needs git-lfs)
sudo apt-get install -y git-lfs
git lfs install
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds ~/data/RLDS/modified_libero_rlds
```

## 3. Training

### Paper configuration (8x A100-80GB)

```bash
bash scripts/train_ddvla.sh
```

Or manually:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
    --vla_path ~/data/models/openvla-7b \
    --data_root_dir ~/data/RLDS/modified_libero_rlds \
    --dataset_name libero_object_no_noops \
    --run_root_dir ~/checkpoints/ddvla-320k \
    --use_discrete_diffusion True \
    --use_l1_regression False \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 320000 \
    --save_freq 7000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32
```

### Resume from checkpoint

```bash
bash scripts/train_ddvla.sh --resume ~/checkpoints/ddvla-320k/<run_dir>/checkpoint-7000
```

Or manually:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
    --vla_path ~/checkpoints/ddvla-320k/<run_dir>/checkpoint-7000 \
    --data_root_dir ~/data/RLDS/modified_libero_rlds \
    --dataset_name libero_object_no_noops \
    --run_root_dir ~/checkpoints/ddvla-320k \
    --use_discrete_diffusion True \
    --use_l1_regression False \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 320000 \
    --save_freq 7000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32
```

### Training time estimates

| Hardware | Batch/GPU | Total Batch | Speed | Time (320k steps) |
|----------|-----------|-------------|-------|--------------------|
| 8x A100-80GB | 8 | 64 | ~1.6 s/it | ~6 days |
| 1x B200-183GB | 24 | 24 | ~1.1 s/it | ~4 days |

Checkpoints saved every 7000 steps (~3 hours on 8xA100).

## 4. Evaluation

```bash
bash scripts/eval_ddvla.sh <checkpoint_path>
```

Standard LIBERO evaluation: 10 tasks x 50 trials = 500 total.

## 5. Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| use_discrete_diffusion | True | DD method (not DFM) |
| lora_rank | 32 | Paper uses r=32 |
| image_aug | True | Random crop 90% area |
| num_steps_before_decay | 100000 | LR decay starts here |
| max_steps | 320000 | Paper's full training |
| batch_size | 8 | Per GPU |
| learning_rate | 5e-4 | |
| num_images_in_input | 2 | agentview + wrist |
| use_proprio | True | proprioception state |
