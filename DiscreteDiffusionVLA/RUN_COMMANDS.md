# Run Guide (Downloads + Finetune + Eval)

This is a minimal command list to get the repo ready and run training/eval.

## 0) Prereqs
- `git` and `git-lfs` installed
- Python env with `pip`
- Optional: Hugging Face token if downloading gated models (set `HF_TOKEN=...`)

## 1) Clone repo (if needed)
```bash
git clone <YOUR_REPO_URL>
cd /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA
```

## 2) Download assets to a base dir
Example base path: `~/Downloads`

```bash
cd /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA

# core assets: base model + LIBERO repo + RLDS dataset
/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/download_assets.sh ~/Downloads
```

Optional: include finetuned checkpoints
```bash
DOWNLOAD_FINETUNED=1 /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/download_assets.sh ~/Downloads
```

If you need a Hugging Face token:
```bash
HF_TOKEN=your_token_here /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/download_assets.sh ~/Downloads
```

## 3) Update SLURM scripts to your base dir
Default base dir inside scripts is `/scratch/ywn1043/finetune`.
Edit these files if you want to use `~/Downloads` (or another path):
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/finetune_slurm.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/eval_libero_slurm.sh`

For example, set:
```
BASE_DIR="~/Downloads"
```

## 4) Submit finetune job (SLURM)
```bash
sbatch /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/finetune_slurm.sh
```

## 5) Submit eval job (SLURM)
```bash
sbatch /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/eval_libero_slurm.sh
```

## 6) Optional local smoke tests
CPU smoke test (DFM unit tests only):
```bash
/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/smoke_test_cpu.sh
```

Low‑VRAM smoke test (RTX 2060 style):
```bash
export DATA_ROOT=~/Downloads/RLDS/modified_libero_rlds
export DATASET_NAME=libero_object_no_noops
export RUN_ROOT_DIR=~/Downloads/checkpoints/openvla_smoke
/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/smoke_test_2060.sh
```

## Notes
- The download script uses `huggingface-cli` if available, otherwise `python -m huggingface_hub.cli.hf`.
- RLDS dataset download requires `git-lfs`.
- For full training, A100‑class GPUs are recommended.
