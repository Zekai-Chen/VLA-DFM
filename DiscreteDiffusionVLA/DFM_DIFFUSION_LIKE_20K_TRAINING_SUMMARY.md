# DFM Diffusion‑Like 20K Smoke Training Summary

**Script**: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/finetune_slurm_smoke_2a100_wandb_dfm_diffusion_like_20k.sh`

**Purpose**
Train a DFM checkpoint with **diffusion‑aligned masking + masked‑CE loss** for 20k steps to validate that the objective matches discrete diffusion behavior while keeping the DFM code path and inference options available.

**High‑Level Behavior**
- Uses the **DFM training branch** (`--use_discrete_flow_matching True`).
- **Training corruption** is diffusion‑style masking (`dfm_train_mode=diffusion_like`).
- **Loss** uses the HF shifted LM loss (DFM override is bypassed in diffusion_like).
- **STOP/EOS is supervised** (aligned with discrete diffusion).
- No hazard weighting is applied (masked‑CE with uniform weights).

**Core Training Flags**
- `--use_discrete_flow_matching True`
- `--use_discrete_diffusion False`
- `--use_diffusion False`
- `--use_l1_regression False`
- `--dfm_train_mode diffusion_like`
- `--dfm_loss_mode masked_ce`
- `--dfm_schedule cosine`
- `--dfm_time_eps 1e-3`
- `--dfm_t_min 0.0`
- `--dfm_t_max 1.0`
- `--dfm_weight_clip 20.0`

**Training Hyperparameters**
- **Steps**: `--max_steps 20000`
- **Batch size**: `--batch_size 6`
- **LR**: `--learning_rate 5e-4`
- **LR decay start**: `--num_steps_before_decay 10000`
- **Checkpoint cadence**: `--save_freq 10000`
- **Shuffle buffer**: `--shuffle_buffer_size 10000`
- **LoRA rank**: `--lora_rank 16`
- **dtype**: `--torch_dtype bfloat16`

**Model and Data Paths**
- **VLA base**: `/scratch/ywn1043/VLA-DFM/models/openvla-7b`
- **Dataset root**: `/scratch/ywn1043/VLA-DFM/RLDS/modified_libero_rlds`
- **Dataset name**: `libero_object_no_noops`
- **Checkpoint root**: `/scratch/ywn1043/VLA-DFM/checkpoints/dfm-diffusion_like-20k-smoke`

**WandB / Logging**
- **Project**: `VLA-DFM`
- **Entity**: `a10v-1`
- **Run ID note**: `dfm-diffusion_like-20k--smoke-2xA100--<timestamp>`
- **Logs**: `logs/openvla_ft_smoke_<jobid>.out|err`

**Environment & Runtime**
- SLURM: 2×A100, 120G RAM, 47h wall time, `gengpu` partition.
- Modules: `gcc/12.4.0`, `cuda/12.4.0`, `git`.
- Uses repo venv: `${REPO_ROOT}/.venv/bin/activate`.

**Optional Debug (Training‑Time)**
- `VLA_DFM_DEBUG=1`
- `VLA_DFM_DEBUG_EVERY=200`

These emit DFM mask ratio, weight stats, and (if enabled) HF loss parity checks for the diffusion_like path.

**Behavior vs Discrete Diffusion**
- **Training**: Equivalent masking + masked‑CE loss (diffusion‑aligned).
- **Inference**: Still uses DFM decoding options (CTMC or MaskGIT), depending on eval config.
- Net effect: DFM infrastructure with diffusion‑style training semantics.

**How to Run**
Submit the script as‑is:

```bash
sbatch /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/finetune_slurm_smoke_2a100_wandb_dfm_diffusion_like_20k.sh
```

**Expected Signals**
- Train `curr_action_accuracy` should move above zero within early steps.
- `curr_action_l1_loss` should drop below ~0.5 as training progresses.
- `dfm_mask_ratio_mean` should stay in a reasonable band (not near 0 or 1).
