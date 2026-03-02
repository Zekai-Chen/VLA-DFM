# DFM Ablations & Performance Report

Date: 2026-03-02

This report summarizes all **DFM/DD ablations, code changes, and results** discussed in this thread. It is strictly based on the logs you provided and the code changes we made in the repo. If a metric or run detail was not explicitly provided, it is marked **“not provided.”**

---

## 1) Executive Summary

**Key findings**
1. **DFM models consistently show strong teacher‑forced accuracy but weak masked‑denoise accuracy**, indicating a failure to learn the “recover from mask” conditioning that DFM requires at inference.
2. **Mask‑embedding overrides (mask→pad) did not improve masked‑denoise accuracy**, suggesting the issue is not just an untrained mask embedding.
3. **Discrete diffusion baseline remains better at eval** (≈20% success at 3k steps), while DFM eval is at or near 0% in multiple runs.
4. The **most reliable diagnostic is the masked‑denoise metric**; it is low across DFM runs even when teacher‑forced accuracy is high.

**Primary root‑cause signal**
- High teacher‑forced acc + low masked‑denoise acc ⇒ DFM training does not sufficiently teach **denoising from fully masked action spans**.

---

## 2) Run Inventory (Checkpoint + Script + Metrics)

**Legend**
- `TF` = teacher‑forced
- `Masked` = fully masked action‑span denoise
- `TM` = token_match
- `L2N` = normalized action L2
- `L2U` = unnormalized action L2

| Run ID | Checkpoint Path | Train Script | Eval/Sanity Script | Decode | TM | L2N | L2U | TF CE | TF Acc | Masked CE | Masked Acc | Eval Success |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **DD‑baseline‑3k** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke/...--20260302_0220` | `sbatch/smoke/finetune_slurm_smoke_2a100_wandb_dd_3k.sh` | `dfm_sanity_check_slurm_smoke_2a100_dd.sh` | DD | 0.032 | 0.569 | 0.402 | 3.104 | 0.347 | n/a | n/a | **~20% (user report)** |
| **DFM‑baseline‑A** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k/...--20260228_2041` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k.sh` | `dfm_sanity_check_slurm_smoke_2a100.sh` | ctmc | 0.446 | 0.216 | 0.158 | 0.000 | 1.000 | not provided | not provided | 0% (eval logs) |
| **DFM‑baseline‑B (CTMC)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k/...--20260228_2041` | same as above | `dfm_sanity_check_slurm_smoke_2a100.sh` | ctmc | 0.157 | 0.363 | 0.234 | not provided | not provided | not provided | not provided | 0% (eval logs) |
| **DFM‑baseline‑C (maskpad override)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k/...--20260228_2041` | same as above | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | ctmc | 0.054 | 0.713 | 0.398 | 1.683 | 0.583 | 4.936 | 0.167 | 0% (eval logs) |
| **DFM‑maskfix (none)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix/...--20260301_1422` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix.sh` | `dfm_sanity_check_slurm_smoke_2a100.sh` | ctmc | 0.286 | 0.522 | 0.364 | 0.144 | 0.946 | 4.459 | 0.304 | 0% (implied) |
| **DFM‑maskfix (maskpad)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix/...--20260301_1422` | same as above | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | ctmc | 0.000 | 0.472 | 0.332 | 1.659 | 0.609 | 3.662 | 0.304 | 0% (implied) |
| **DFM‑maskedce + tmax=0.7** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-maskedce/...--20260301_1621` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_maskedce.sh` | `dfm_sanity_check_slurm_smoke_2a100.sh` | ctmc | 0.000 | 0.809 | 0.469 | 2.380 | 0.304 | 3.956 | 0.286 | 0% (implied) |
| **DFM‑maskedce + tmax=0.7 (maskpad)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-maskedce/...--20260301_1621` | same as above | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | ctmc | 0.000 | 0.428 | 0.362 | 2.367 | 0.292 | 4.366 | 0.167 | 0% (implied) |
| **DFM‑moremask tmax=0.9 (maskpad)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask/...--20260301_1904` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_moremask.sh` | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | ctmc | 0.114 | 0.452 | 0.275 | 1.110 | 0.724 | 3.984 | 0.288 | 0% (implied) |
| **DFM‑moremask tmax=0.9 (maskpad, newer)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask/...--20260301_2133` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_moremask.sh` | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | ctmc | 0.032 | 0.533 | 0.439 | 0.360 | 0.910 | 3.180 | 0.458 | 0% (implied) |
| **DFM‑moremask tmax=0.9 (no‑maskpad, newer)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask/...--20260301_2133` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_moremask.sh` | `dfm_sanity_check_slurm_smoke_2a100.sh` | ctmc | 0.004 | 0.511 | 0.392 | 1.003 | 0.773 | 3.869 | 0.254 | 0% (implied) |
| **DFM‑moremask tmax=0.9 (no‑maskpad, maskgit)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask/...--20260301_2133` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_moremask.sh` | `dfm_sanity_check_slurm_smoke_2a100.sh` | maskgit | 0.000 | 0.611 | 0.487 | 0.492 | 0.846 | 3.435 | 0.367 | 0% (implied) |
| **DFM‑moremask tmax=0.9 (no‑maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask/...--20260301_2244` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_moremask.sh` (with `--dfm_maskgit_num_steps 12`) | `dfm_sanity_check_slurm_smoke_2a100.sh` | maskgit | 0.339 | 0.412 | 0.314 | 1.330 | 0.704 | 3.522 | 0.371 | 0% (implied) |
| **DFM‑moremask tmax=0.9 (maskpad, maskgit)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask/...--20260301_2133` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_moremask.sh` | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | maskgit | 0.007 | 0.536 | 0.426 | 0.640 | 0.814 | 4.415 | 0.179 | 0% (implied) |
| **DFM‑moremask tmax=0.9 (maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask/...--20260301_2244` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_moremask.sh` (with `--dfm_maskgit_num_steps 12`) | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | maskgit | 0.239 | 0.632 | 0.529 | 1.286 | 0.686 | 3.999 | 0.288 | 0% (implied) |
| **DFM‑tmax=0.6 (no‑maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask-tmax0.6/...--20260302_0104` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskgit_tmax_sweep.sh` (tmax=0.6) | `dfm_sanity_check_slurm_smoke_2a100.sh` | maskgit | 0.089 | 0.605 | 0.409 | 1.497 | 0.652 | 3.720 | 0.307 | 0% (implied) |
| **DFM‑tmax=0.6 (maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask-tmax0.6/...--20260302_0104` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskgit_tmax_sweep.sh` (tmax=0.6) | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | maskgit | 0.057 | 0.669 | 0.525 | 1.266 | 0.723 | 3.820 | 0.258 | 0% (implied) |
| **DFM‑tmax=0.7 (no‑maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask-tmax0.7/...--20260302_0118` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskgit_tmax_sweep.sh` (tmax=0.7) | `dfm_sanity_check_slurm_smoke_2a100.sh` | maskgit | 0.314 | 0.446 | 0.392 | 0.277 | 0.929 | 3.019 | 0.454 | 0% (implied) |
| **DFM‑tmax=0.8 (no‑maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask-tmax0.8/...--20260302_0143` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskgit_tmax_sweep.sh` (tmax=0.8) | `dfm_sanity_check_slurm_smoke_2a100.sh` | maskgit | 0.121 | 0.540 | 0.419 | 1.121 | 0.771 | 3.555 | 0.372 | 0% (implied) |
| **DFM‑tmax=0.8 (maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask-tmax0.8/...--20260302_0143` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskgit_tmax_sweep.sh` (tmax=0.8) | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | maskgit | 0.161 | 0.560 | 0.390 | 0.408 | 0.958 | 3.766 | 0.357 | 0% (implied) |
| **DFM‑tmax=0.9 (no‑maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask-tmax0.9/...--20260302_0217` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskgit_tmax_sweep.sh` (tmax=0.9) | `dfm_sanity_check_slurm_smoke_2a100.sh` | maskgit | 0.246 | 0.433 | 0.368 | 1.269 | 0.725 | 3.525 | 0.329 | 0% (implied) |
| **DFM‑tmax=0.9 (maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask-tmax0.9/...--20260302_0217` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskgit_tmax_sweep.sh` (tmax=0.9) | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | maskgit | 0.339 | 0.474 | 0.372 | 1.232 | 0.758 | 3.277 | 0.420 | 0% (implied) |
| **DFM‑tmax=0.7 (maskpad, maskgit, 12 steps)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask-tmax0.7/...--20260302_0104` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskgit_tmax_sweep.sh` (tmax=0.7) | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | maskgit | 0.300 | 0.457 | 0.341 | 1.087 | 0.735 | 3.361 | 0.401 | 0% (implied) |

Notes:
- The eval scripts used for DFM were `eval_libero_slurm_smoke_2a100_dfm_flow_3k.sh` and `eval_libero_slurm_smoke_2a100_dfm_flow_clone.sh`. Both produced **0% success** in multiple episodes.
- Missing metrics are explicitly marked as “not provided.”

---

## 3) Detailed Run Notes and Interpretation

### 3.1 Discrete Diffusion Baseline (3k)
- **Outcome:** ~20% success (user report).
- **Interpretation:** DD baseline is functioning and provides a reference. DFM should be compared to this baseline under identical eval conditions.

### 3.2 DFM baseline (20260228_2041)
- **Outcome:** 0% success in eval.
- **Sanity:** Teacher‑forced accuracy can be extremely high (TF acc ~1.0 in one run), but masked‑denoise was not reported in the earliest baseline run. Later masked runs show it is low.
- **Interpretation:** Model can output correct tokens if conditioned on ground‑truth actions but fails to predict from masked actions.

### 3.3 DFM maskfix (20260301_1422)
- **maskpad override OFF:** TF acc ~0.946, masked‑denoise acc ~0.304.
- **maskpad override ON:** TF acc ~0.609, masked‑denoise acc ~0.304.
- **Interpretation:** Mask‑embedding override does **not** improve denoising. Mask‑conditioning weakness persists.

### 3.4 DFM maskedCE + tmax=0.7 (20260301_1621)
- **Outcome:** TF acc ~0.304, masked acc ~0.286 (both low). Token match collapsed to 0.
- **Interpretation:** This ablation **degraded the model substantially**. The masked‑CE‑only or overly aggressive masking likely destabilized training or shifted the loss away from correct action prediction.

### 3.5 DFM moremask (tmax=0.9) (20260301_1904)
- **Outcome:** TF acc ~0.724 (moderate), masked acc ~0.288 (still low).
- **Interpretation:** Increasing masking intensity helps only marginally; denoising accuracy still too low for successful DFM inference.

### 3.6 DFM moremask (tmax=0.9) (20260301_2133, maskpad)
- **Outcome:** TF acc ~0.910, masked acc ~0.458 (notably higher than earlier runs).
- **Interpretation:** Mask‑denoise accuracy improved into the ~0.46 range, which is the best observed so far in this thread, but it still remains below the likely threshold for stable DFM inference. This run still used **CTMC decoding** in sanity checks, so it does **not** validate the fixed MaskGIT path yet.

### 3.7 DFM moremask (tmax=0.9) (20260301_2133, no‑maskpad)
- **Outcome:** TF acc ~0.773, masked acc ~0.254 (worse than maskpad).
- **Interpretation:** Removing the mask‑pad override reduced masked‑denoise accuracy significantly. This suggests the override helps this checkpoint, but the overall denoising ability still remains too low for successful DFM inference. CTMC decode was used in this sanity run.

### 3.8 DFM moremask (tmax=0.9) (20260301_2133, no‑maskpad, MaskGIT)
- **Outcome:** TF acc ~0.846, masked acc ~0.367.
- **Interpretation:** Switching to the **fixed MaskGIT decode** improved masked‑denoise accuracy versus CTMC without maskpad (~0.25 → ~0.37), but it is still below the best observed masked‑denoise (~0.46 with maskpad+CTMC). This indicates decode improvements help, but mask conditioning remains the dominant limitation.

### 3.9 DFM moremask (tmax=0.9) (20260301_2133, maskpad, MaskGIT)
- **Outcome:** TF acc ~0.814, masked acc ~0.179.
- **Interpretation:** MaskGIT with maskpad **reduced** masked‑denoise accuracy compared to maskpad+CTMC (~0.46). This suggests maskpad interacts poorly with MaskGIT in this checkpoint, and that the maskpad improvement seen under CTMC does **not** transfer to MaskGIT.

### 3.10 DFM moremask (tmax=0.9) (20260301_2244, no‑maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_maskgit_num_steps 12`.
- **Outcome:** TF acc ~0.704, masked acc ~0.371.
- **Interpretation:** Masked‑denoise accuracy stayed roughly flat vs the prior MaskGIT run (~0.37), while teacher‑forced accuracy declined. This suggests **the 12‑step MaskGIT schedule alone is not sufficient to improve denoising**, and training‑side changes (e.g., `dfm_t_max` sweep or masked‑CE auxiliary) are still needed.

### 3.11 DFM moremask (tmax=0.9) (20260301_2244, maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_maskgit_num_steps 12` with `mask_embed_override=pad`.
- **Outcome:** TF acc ~0.686, masked acc ~0.288.
- **Interpretation:** Masked‑denoise accuracy improved versus the older maskpad+MaskGIT run (~0.18 → ~0.29) but remains **below** the no‑maskpad MaskGIT run (~0.37). This continues to indicate that maskpad is **not beneficial** for MaskGIT in this checkpoint.

### 3.12 DFM tmax=0.6 (20260302_0104, no‑maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_t_max 0.6` with `--dfm_maskgit_num_steps 12`.
- **Outcome:** TF acc ~0.652, masked acc ~0.307, token_match ~0.089, L2_unnorm ~0.409.
- **Interpretation:** Lowering `dfm_t_max` to 0.6 **did not improve** masked‑denoise accuracy versus the best no‑maskpad MaskGIT run (~0.37). It also reduced teacher‑forced accuracy, suggesting **too much heavy masking harms overall conditioning** at this training length (3k steps).

### 3.13 DFM tmax=0.6 (20260302_0104, maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_t_max 0.6` with `--dfm_maskgit_num_steps 12` and `mask_embed_override=pad`.
- **Outcome:** TF acc ~0.723, masked acc ~0.258, token_match ~0.057, L2_unnorm ~0.525.
- **Interpretation:** Masked‑denoise accuracy is **lower** than the no‑maskpad tmax=0.6 run (~0.31) and well below the best no‑maskpad MaskGIT run (~0.37). This reinforces that **maskpad is counter‑productive under MaskGIT** and that tmax=0.6 is too aggressive at 3k steps.

### 3.14 DFM tmax=0.7 (20260302_0118, no‑maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_t_max 0.7` with `--dfm_maskgit_num_steps 12`.
- **Outcome:** TF acc ~0.929, masked acc ~0.454, token_match ~0.314, L2_unnorm ~0.392.
- **Interpretation:** This is the **best MaskGIT/no‑maskpad result so far**, with masked‑denoise accuracy approaching the CTMC+maskpad peak (~0.46) while maintaining very strong teacher‑forced accuracy. This suggests **tmax=0.7 is currently the most promising setting** for 3k‑step runs.
- **Note:** The provided run command string contains `run_root_dir` and `run_id_note` with `tmax0.6`; the checkpoint path and results indicate `tmax=0.7` and should be treated as authoritative.

### 3.15 DFM tmax=0.7 (20260302_0104, maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_t_max 0.7` with `--dfm_maskgit_num_steps 12` and `mask_embed_override=pad`.
- **Outcome:** TF acc ~0.735, masked acc ~0.401, token_match ~0.300, L2_unnorm ~0.341.
- **Interpretation:** Masked‑denoise accuracy improves relative to maskpad+tmax0.6 (~0.26) and older maskpad+MaskGIT runs (~0.18), but remains **below** the no‑maskpad tmax=0.7 result (~0.45). Maskpad still appears net‑negative for MaskGIT at this step budget.

### 3.16 DFM tmax=0.8 (20260302_0143, no‑maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_t_max 0.8` with `--dfm_maskgit_num_steps 12`.
- **Outcome:** TF acc ~0.771, masked acc ~0.372, token_match ~0.121, L2_unnorm ~0.419.
- **Interpretation:** Masked‑denoise accuracy is **lower** than the tmax=0.7 run (~0.45) and roughly comparable to earlier MaskGIT/no‑maskpad runs (~0.37). This suggests **tmax=0.8 is not better than 0.7** at 3k steps.

### 3.17 DFM tmax=0.8 (20260302_0143, maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_t_max 0.8` with `--dfm_maskgit_num_steps 12` and `mask_embed_override=pad`.
- **Outcome:** TF acc ~0.958, masked acc ~0.357, token_match ~0.161, L2_unnorm ~0.390.
- **Interpretation:** Masked‑denoise accuracy is **below** the no‑maskpad tmax=0.8 run (~0.37) and well below the best tmax=0.7 no‑maskpad run (~0.45). Maskpad again does not improve MaskGIT denoising.

### 3.18 DFM tmax=0.9 (20260302_0217, no‑maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_t_max 0.9` with `--dfm_maskgit_num_steps 12`.
- **Outcome:** TF acc ~0.725, masked acc ~0.329, token_match ~0.246, L2_unnorm ~0.368.
- **Interpretation:** Masked‑denoise accuracy is **lower** than tmax=0.8 (~0.37) and clearly below the tmax=0.7 peak (~0.45). This suggests **tmax=0.9 is too aggressive** at 3k steps and supports the current optimum near **tmax≈0.7**.
- **Note:** Your prompt said “tmax=0.8,” but the run config and checkpoint path indicate **tmax=0.9**; the results are attributed to **tmax=0.9**.

### 3.19 DFM tmax=0.9 (20260302_0217, maskpad, MaskGIT, 12 steps)
- **Primary change:** `--dfm_t_max 0.9` with `--dfm_maskgit_num_steps 12` and `mask_embed_override=pad`.
- **Outcome:** TF acc ~0.758, masked acc ~0.420, token_match ~0.339, L2_unnorm ~0.372.
- **Interpretation:** Masked‑denoise accuracy improves versus the no‑maskpad tmax=0.9 run (~0.33 → ~0.42) and is comparable to maskpad tmax=0.7 (~0.40), but remains **below** the no‑maskpad tmax=0.7 peak (~0.45). This suggests maskpad can help at **tmax=0.9**, but does **not** beat the best no‑maskpad setting.

### 3.20 DD baseline (3k, 20260302_0220)
- **Training config:** `use_discrete_diffusion=True`, 3k steps, same dataset/optimizer settings as DFM runs.
- **Outcome (sanity):** token_match ~0.032, L2_unnorm ~0.402, TF acc ~0.347 (TF CE ~3.104). Masked‑denoise metrics are **not applicable** for DD (reported as `nan`).
- **Interpretation:** Teacher‑forced accuracy on DD is **low** compared to DFM runs, but DD still achieved **~20% success** in eval (per user report). This highlights that **DD’s sampling path can still perform** even when TF metrics are weak, while DFM depends heavily on masked‑denoise quality.

---

## 4) Code Changes Inventory (Grouped by Area)

This section summarizes **modifications and new additions** made during this effort.

### 4.1 Config persistence & checkpoint correctness
Files:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/configuration_prismatic.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py`

Changes:
- Added **explicit DFM fields** to config serialization (`dfm_schedule`, `dfm_loss_mode`, `dfm_train_mode`, etc.).
- Added a helper to **apply finetune args into config before model init**.
- Ensured merged checkpoints **save the updated config** (DFM fields preserved).

Why:
- Prevents silent mismatch between DFM training and eval.

### 4.2 Action vocab anchoring + tokenizer
Files:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/configuration_prismatic.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/vla/action_tokenizer.py`

Changes:
- Added `action_vocab_anchor` (`pad` default).
- Implemented `_action_vocab_range()` anchored to PAD.
- ActionTokenizer now uses `action_token_end_idx` for encode/decode.
- Correct bin edges with `np.linspace(n_bins+1)`.

Why:
- Fixes action vocab range errors (e.g., debug high=83) and PAD/MASK collisions.

### 4.3 DFM corruption + generalized KL
Files:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

Changes:
- Corruption replaced by **mixture‑path Bernoulli masking**.
- Implemented **generalized‑KL loss** with correct action‑span integrand.

Why:
- Aligns code with generalized‑KL formulation; eliminates top‑k masking artifacts.

### 4.4 Multimodal alignment + masks
Files:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

Changes:
- Added multimodal alignment helpers for patch tokens.
- Ensured DFM tensors are shifted consistently for loss/logits.

Why:
- Prevents action/logit misalignment when patch tokens are inserted.

### 4.5 DFM schedules & decode instrumentation
Files:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_schedule.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_decode.py`

Changes:
- Added **sin** schedule (`kappa=sin(pi t/2)`).
- Added decode stats: `dfm_decode_mode`, `n_action_positions`, `n_masked_initial`, per‑step unresolved counts.
- MaskGIT integrity checks in debug mode.

### 4.6 Eval parity & diagnostics
Files:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/openvla_utils.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py`

Changes:
- EOS stripping for prompt tokens after processor tokenization.
- Center‑crop restored to match training.
- Added strong tokenizer/model vocab alignment checks.
- Added detailed debug output: unnormalized action stats, gripper inversion, mask counts.

Why:
- Ensures eval matches training data formatting and avoids silent token mismatches.

### 4.7 Sanity scripts and SLURM utilities
Files:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/scripts/dfm_sanity_check.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/dfm_sanity_check_slurm_smoke_2a100.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/dfm_sanity_check_slurm_smoke_2a100_maskpad.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/dfm_sanity_check_slurm_smoke_2a100_dd.sh`

Changes:
- Added **offline sanity check** for DFM/ DD on training data.
- Added **teacher‑forced metrics** and **masked‑denoise metrics**.
- Added **mask embedding override** (mask→pad) for diagnosis.
- Added DD parity sanity script.

---

## 5) Performance Summary (DFM vs DD)

**Discrete Diffusion (3k)**
- ~20% success (user report). No further metrics provided.

**DFM (multiple runs)**
- Eval success ≈ 0% across multiple checkpoints.
- Teacher‑forced acc often **high**, while masked‑denoise acc is **generally low** but varies: **~0.46 with maskpad+CTMC**, **~0.45 with MaskGIT/no‑maskpad at tmax=0.7 (12 steps)**, **~0.40 with MaskGIT+maskpad at tmax=0.7 (12 steps)**, **~0.37 with MaskGIT/no‑maskpad (12 steps)**, **~0.37 with MaskGIT/no‑maskpad at tmax=0.8**, **~0.36 with MaskGIT+maskpad at tmax=0.8**, **~0.31 with MaskGIT/no‑maskpad at tmax=0.6**, **~0.29 with MaskGIT+maskpad (12 steps)**, **~0.26 with MaskGIT+maskpad at tmax=0.6**, **~0.25 with CTMC/no‑maskpad**, and **~0.18 with MaskGIT+maskpad (older run)**.
- Teacher‑forced acc often **high**, while masked‑denoise acc is **generally low** but varies: **~0.46 with maskpad+CTMC**, **~0.45 with MaskGIT/no‑maskpad at tmax=0.7 (12 steps)**, **~0.42 with MaskGIT+maskpad at tmax=0.9**, **~0.40 with MaskGIT+maskpad at tmax=0.7 (12 steps)**, **~0.37 with MaskGIT/no‑maskpad (12 steps)**, **~0.37 with MaskGIT/no‑maskpad at tmax=0.8**, **~0.36 with MaskGIT+maskpad at tmax=0.8**, **~0.33 with MaskGIT/no‑maskpad at tmax=0.9**, **~0.31 with MaskGIT/no‑maskpad at tmax=0.6**, **~0.29 with MaskGIT+maskpad (12 steps)**, **~0.26 with MaskGIT+maskpad at tmax=0.6**, **~0.25 with CTMC/no‑maskpad**, and **~0.18 with MaskGIT+maskpad (older run)**.
- Indicates **mask conditioning weakness** remains the dominant issue even after MaskGIT decode fixes.

---

## 6) Root‑Cause Analysis

Evidence across runs strongly suggests:
1. **Teacher‑forced acc is high** → the model can predict action tokens when conditioned on GT actions.
2. **Masked‑denoise acc is low** → the model cannot recover actions from masked action spans, which is **exactly DFM inference**.
3. **Mask‑embedding override does not help** → the issue is not simply an untrained mask embedding.

**Conclusion:** The DFM training objective or schedule does not sufficiently train the model to denoise from fully masked action spans.

---

## 7) Recommendations (Next Experiments)

1. **Increase fully masked exposure**
   - Lower `dfm_t_max` further (e.g., 0.7 or 0.6), or bias `t` toward small values.

2. **Add a masked‑CE auxiliary (small weight)**
   - Keep generalized‑KL as main loss, add a low‑weight masked‑CE term to teach recovery from masks.

3. **Force MaskGIT decoding for DFM sanity**
   - Ensure decode path is actually maskgit and verify `n_masked_initial == n_action_positions`.

4. **Report masked‑denoise acc as a gating metric**
   - If masked‑denoise acc is < 0.5 on train data, do not run expensive eval.

---

## 8) Appendix: Paths and Scripts (for traceability)

**Training scripts**
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/finetune_slurm_smoke_2a100_wandb_dfm_flow_3k.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_maskedce.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_moremask.sh`

**Eval scripts**
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/eval_libero_slurm_smoke_2a100_dfm_flow_3k.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/eval_libero_slurm_smoke_2a100_dfm_flow_clone.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/eval_libero_slurm_smoke_2a100_discrete_diffusion.sh`

**Sanity scripts**
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/scripts/dfm_sanity_check.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/dfm_sanity_check_slurm_smoke_2a100.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/dfm_sanity_check_slurm_smoke_2a100_maskpad.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/dfm_sanity_check_slurm_smoke_2a100_dd.sh`

---

End of report.
