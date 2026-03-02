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
| **DD‑baseline‑3k** | *(not provided)* | *(not provided)* | *(eval script for DD baseline)* | *(DD)* | not provided | not provided | not provided | not provided | not provided | not provided | not provided | **~20% (user report)** |
| **DFM‑baseline‑A** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k/...--20260228_2041` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k.sh` | `dfm_sanity_check_slurm_smoke_2a100.sh` | ctmc | 0.446 | 0.216 | 0.158 | 0.000 | 1.000 | not provided | not provided | 0% (eval logs) |
| **DFM‑baseline‑B (CTMC)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k/...--20260228_2041` | same as above | `dfm_sanity_check_slurm_smoke_2a100.sh` | ctmc | 0.157 | 0.363 | 0.234 | not provided | not provided | not provided | not provided | 0% (eval logs) |
| **DFM‑baseline‑C (maskpad override)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k/...--20260228_2041` | same as above | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | ctmc | 0.054 | 0.713 | 0.398 | 1.683 | 0.583 | 4.936 | 0.167 | 0% (eval logs) |
| **DFM‑maskfix (none)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix/...--20260301_1422` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix.sh` | `dfm_sanity_check_slurm_smoke_2a100.sh` | ctmc | 0.286 | 0.522 | 0.364 | 0.144 | 0.946 | 4.459 | 0.304 | 0% (implied) |
| **DFM‑maskfix (maskpad)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix/...--20260301_1422` | same as above | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | ctmc | 0.000 | 0.472 | 0.332 | 1.659 | 0.609 | 3.662 | 0.304 | 0% (implied) |
| **DFM‑maskedce + tmax=0.7** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-maskedce/...--20260301_1621` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_maskedce.sh` | `dfm_sanity_check_slurm_smoke_2a100.sh` | ctmc | 0.000 | 0.809 | 0.469 | 2.380 | 0.304 | 3.956 | 0.286 | 0% (implied) |
| **DFM‑maskedce + tmax=0.7 (maskpad)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-maskedce/...--20260301_1621` | same as above | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | ctmc | 0.000 | 0.428 | 0.362 | 2.367 | 0.292 | 4.366 | 0.167 | 0% (implied) |
| **DFM‑moremask tmax=0.9 (maskpad)** | `/scratch/ywn1043/VLA-DFM/checkpoints/ddopenvla-libero-object-smoke-3k-maskfix-moremask/...--20260301_1904` | `finetune_slurm_smoke_2a100_wandb_dfm_flow_3k_maskfix_moremask.sh` | `dfm_sanity_check_slurm_smoke_2a100_maskpad.sh` | ctmc | 0.114 | 0.452 | 0.275 | 1.110 | 0.724 | 3.984 | 0.288 | 0% (implied) |

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
- Teacher‑forced acc often **high**, while masked‑denoise acc stays **low** (~0.17–0.31).
- Indicates **mask conditioning failure**, not necessarily action tokenization failure.

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
