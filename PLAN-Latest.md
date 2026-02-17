# Plan: Full CTMC Discrete Flow Matching (DFM) Integration

## Summary
Implement a **full DFM path** (CTMC hazard/tau‑leaping solver, adaptive step sizing, generalized KL loss) alongside existing discrete diffusion. The new path is **mask‑source + convex mixture**, **time‑independent denoiser**, and **mask‑only corruption** for parity and stability. Discrete diffusion remains intact and selectable via flags.

---

## Decisions Locked (from open-question-responses.md)
- **Objective**: speed–quality tradeoff with no success‑rate regressions.
- **Scope**: full CTMC now (no staging).
- **Path**: mask‑source + convex mixture only (no uniform‑noise path).
- **Coupling**: U‑coupling with constant mask source.
- **Corruption**: mask‑only.
- **Time conditioning**: **no explicit t input** (time‑independent denoiser).
- **Loss**: **generalized KL / weighted CE** aligned to DFM (keep masked‑CE as optional ablation).
- **Sampling**: CTMC hazard/tau‑leaping with **adaptive step size** and **early exit**.
- **Inference**: CTMC replaces MaskGIT; hybrid corrector optional for debugging.
- **Compatibility**: preserve discrete diffusion + checkpoints.

---

## Public API / Config Changes
Add new flags and parameters:

**Training (FinetuneConfig in** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py`**):**
- `use_discrete_flow_matching: bool`
- `dfm_schedule: str` (e.g., `cosine`, `linear`, `poly2`)
- `dfm_time_eps: float` (avoid t=0/1)
- `dfm_loss_mode: str` (`generalized_kl` | `masked_ce`)
- `dfm_weight_clip: float` (clamp for `kappa_dot/(1-kappa)`)

**Inference (GenerateConfig in** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py`**):**
- `use_discrete_flow_matching: bool`
- `dfm_num_steps: int`
- `dfm_schedule: str`
- `dfm_temperature: float`
- `dfm_adaptive_step: bool`
- `dfm_step_min: float`, `dfm_step_max: float`
- `dfm_early_exit: bool`
- `dfm_corrector: bool` (optional remask pass)
- `dfm_clamp_mask: bool` (if clamping is used)

**Model Config (OpenVLAConfig in** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/configuration_prismatic.py`**):**
- `use_discrete_flow_matching: bool` + setter `set_discrete_flow_matching()`  
- Keep `use_discrete_diffusion` intact; enforce exclusivity.

**Inference API (predict_action in** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`**):**
- add `use_discrete_flow_matching` + DFM solver kwargs.

---

## Implementation Plan

### 1) Scheduling & Path Utilities
**Add** a DFM schedule module (new file or extend existing):  
`/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_schedule.py`
- `kappa(t, schedule_type)` and `kappa_dot(t, schedule_type)`
- `time_grid(num_steps, eps)`  
- Support `cosine`, `linear`, `poly2` (default cosine).

### 2) Training: Masked Path + Generalized KL Loss
**In** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`:

**New method** `apply_mask_flow_matching(...)`:
- Sample `t ~ Uniform(eps, 1-eps)`
- Compute `kappa(t)`; mask ratio = `1 - kappa`
- Produce `masked_input_ids`, `masked_input_embeddings`, `masked_labels`, `loss_mask`
- Return `(t, kappa, kappa_dot, loss_mask)` for loss weighting

**Forward path changes**:
- Add `self.use_discrete_flow_matching = config.use_discrete_flow_matching`
- If `use_discrete_flow_matching`:
  - call `apply_mask_flow_matching`
  - run LM forward to get logits
  - compute per-token CE and apply `weight = clamp(kappa_dot/(1-kappa))`
  - apply `loss_mask` and reduce (sum/num_mask)
  - set `loss` in `PrismaticCausalLMOutputWithPast` to computed DFM loss
- If `dfm_loss_mode == masked_ce`, skip weighting but still use `loss_mask`.

**Notes**:
- Do **not** inject `t` into model inputs (time‑independent denoiser).
- Mask only action-token positions (consistent with current diffusion training).

### 3) CTMC Solver (Inference)
**Add** CTMC solver:  
`/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_decode.py`

Algorithm (vectorized):
- Initialize `cur_seqs` = all `[MASK]` action tokens
- For each time step:
  - Compute logits → posterior probs (`softmax(logits/temperature)`)
  - Sample `x1_i ~ posterior` per position
  - Compute hazard `λ = kappa_dot/(1-kappa)`
  - Compute update probability `p = 1 - exp(-h * λ)`
  - Update positions where `rand < p` and `x1_i != cur`
  - Optional clamping mask applied after updates
  - Early exit if no changes (or below threshold)
- Use **adaptive step** if enabled:
  - `h = min(base_h, (1-kappa)/kappa_dot)` with eps guards
  - Clamp `h` within `[dfm_step_min, dfm_step_max]`

### 4) DFM Prediction Path in Model
**In** `modeling_prismatic.py`:
- Add `_discrete_flow_matching_prediction(...)` mirroring `_discrete_diffusion_prediction(...)`
  - Build `tokens_to_logits` same as discrete diffusion
  - Call `dfm_decode` to get final action token ids
  - Map token ids to actions using existing bin mapping
- Update `predict_action()` to route to DFM when `use_discrete_flow_matching=True`.

### 5) Training Script Integration
**In** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py`:
- Add DFM flags to `FinetuneConfig`
- Add mutual exclusivity checks:
  - `use_discrete_diffusion` XOR `use_discrete_flow_matching`
- When DFM enabled:
  - add mask token to tokenizer
  - set `model_config.set_discrete_flow_matching(True)`
- Ensure `run_forward_pass()` receives DFM flag and uses masked token metrics similar to diffusion.

### 6) Evaluation + Inference Wiring
**Update**:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/openvla_utils.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/robot_utils.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py`

Add DFM flags + hyperparams and pass them into `vla.predict_action(...)`.

### 7) Scripts & Docs
- Add DFM examples to:
  - `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/finetune.sh`
  - `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/finetune_from_ckpt.sh`
- Update `/Users/ali/dev/VLA-DFM/PLAN.md` and `/Users/ali/dev/VLA-DFM/Design-plan.md` to reflect CTMC scope.

---

## Tests & Validation
1. **Unit sanity**
   - `dfm_schedule.kappa/kappa_dot` shape + boundary checks
   - `dfm_decode` step stability (no NaNs, no negative probs)
2. **Training smoke**
   - 1–2 steps on small batch with `use_discrete_flow_matching=True`
3. **Inference smoke**
   - Run `run_libero_eval.py` with DFM for 1–2 rollouts
4. **A/B sanity**
   - Discrete diffusion path unchanged when DFM disabled

---

## Assumptions & Defaults
- **No external dependencies** (no `flow_matching` library) — implement solver locally.
- Default schedule: **cosine**; `dfm_num_steps=12` for parity.
- `dfm_loss_mode = generalized_kl` with weight clipping.
- Adaptive step enabled by default.

---

If you want me to proceed to implement, I’ll start by adding the schedule + CTMC solver modules and wiring config flags end‑to‑end.