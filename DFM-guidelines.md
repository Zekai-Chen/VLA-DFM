# DFM Missing Behaviors — Priority Guidelines

This document captures DFM behaviors that are missing or underspecified across `/Users/ali/dev/VLA-DFM/deep-research-report.md`, `/Users/ali/dev/VLA-DFM/open-question-responses.md`, and `/Users/ali/dev/VLA-DFM/PLAN-Latest.md`. Each item includes a priority tag and concrete implementation guidance. Priority rubric: **High = correctness/parity/compatibility**, **Low = diagnostics/monitoring/tuning/secondary robustness**. Early exit and dtype safety are explicitly **Low**.

## Priority Rubric
- **High**: Impacts correctness, parity with discrete diffusion, or backward compatibility.
- **Low**: Debug/metrics/tuning/secondary stability (nice-to-have, not blocking).

---

## Per-Item Playbook

### 1) Tokenizer + Config Injection (DFM)
- **Priority**: High
- **Behavior Gap**: The docs don’t specify the exact `[MASK]` token injection flow or how it is persisted in model config for DFM, including pad-to-multiple behavior.
- **Guideline (What to implement)**: Mirror the discrete diffusion path exactly: add `[MASK]` via the processor/tokenizer, ensure `mask_token_id` is set, set `use_mask_token=True`, and persist these values into `config.json`. Preserve any `pad_to_multiple_of` logic already used by diffusion.
- **Notes/Defaults**: The DFM path must not change tokenization/vocab sizing beyond adding `[MASK]` if it is missing.

### 2) Exact Loss Construction over Multimodal Sequence
- **Priority**: High
- **Behavior Gap**: Missing exact masking rules for multimodal sequences (vision patches + language + action tokens), and where loss is computed.
- **Guideline (What to implement)**: Build labels only for action tokens, set `IGNORE_INDEX` for vision patch tokens and language tokens, and compute loss only on action-token positions that are supervised (typically masked positions). Ensure the loss mask is expanded to the multimodal layout the same way discrete diffusion does.
- **Notes/Defaults**: Keep the action-token supervision consistent with discrete diffusion for parity.

### 3) Token-Count Normalization for DFM Loss
- **Priority**: High
- **Behavior Gap**: Loss normalization is not explicitly defined.
- **Guideline (What to implement)**: Use token-count normalization: `loss_sum / max(num_supervised_tokens, 1)` (or weighted denom if using DFM weighting). Avoid sequence-length normalization which varies with mask ratio.
- **Notes/Defaults**: If using per-example weights, normalize by `sum(weights * mask)`.

### 4) Per-Example Time Sampling + Kappa-to-Mask Count
- **Priority**: High
- **Behavior Gap**: No concrete rule for sampling one `t` per example or converting `kappa(t)` into a discrete mask count.
- **Guideline (What to implement)**: Sample one `t` per batch element in `[t_min, 1 - t_eps]`. Compute `mask_frac = 1 - kappa(t)` and `n_mask = round(mask_frac * L_action)`; clamp `n_mask` to `[1, L_action]`. Mask positions uniformly among unclamped action tokens.
- **Notes/Defaults**: `t_eps = 1e-3` to avoid endpoints; clamp to at least 1 to avoid empty supervision.

### 5) Weight Clipping Policy for `kappa_dot/(1-kappa)`
- **Priority**: High
- **Behavior Gap**: No explicit clipping policy or defaults.
- **Guideline (What to implement)**: Compute `w = kappa_dot / (1 - kappa + eps)` with `eps = 1e-8`; clamp `w` to `[0, w_clip]` with default `w_clip = 20`. Log `frac_w_clipped` for monitoring.
- **Notes/Defaults**: Use `t_eps` to avoid `kappa -> 1` spikes.

### 6) CTMC Step-Size Precedence Rules
- **Priority**: High
- **Behavior Gap**: No precise rule for combining base grid, adaptive safety cap, step min/max, or remaining time.
- **Guideline (What to implement)**: Define `dt_grid` from a monotone time grid. Compute `dt_safe = (1 - kappa) / (kappa_dot + 1e-12)`. Set `dt = min(dt_grid, dt_safe, step_max, remaining_time)`. If `dt < step_min`, only raise to `step_min` when it does not violate `dt_safe` or remaining time; otherwise keep `dt` and record `dt_under_min=True`.
- **Notes/Defaults**: Safety cap always takes precedence.

### 7) Corrector Definition (if retained)
- **Priority**: High
- **Behavior Gap**: No concrete definition of when/how a corrector runs or how it selects tokens.
- **Guideline (What to implement)**: Run corrector after CTMC by default. Compute confidence via `max softmax prob`, remask the bottom `corrector_remask_frac` among unclamped tokens, then resample with 1–2 passes. Make it MaskGIT-style for debuggability.
- **Notes/Defaults**: `corrector_iters = 1`, `corrector_remask_frac = 0.1`.

### 8) Clamp Semantics
- **Priority**: High
- **Behavior Gap**: Clamping behavior is not explicitly defined.
- **Guideline (What to implement)**: Accept `clamp_mask` and `clamp_values`. Apply clamping immediately after initialization and after every CTMC step. Exclude clamped positions from masking, updates, and loss. Allow a convenience path that derives `clamp_mask` from initial tokens if provided.
- **Notes/Defaults**: Clamping should be an invariant (re-applied each step).

### 9) Temperature Usage and Annealing
- **Priority**: High
- **Behavior Gap**: Unclear how temperature should interact with CTMC updates.
- **Guideline (What to implement)**: Apply temperature to logits only (`softmax(logits / T)`) and never to hazard/jump probabilities. Optionally support an anneal schedule applied to logits.
- **Notes/Defaults**: Default `T=1.0`; simple linear or polynomial anneal acceptable.

### 10) Compatibility Defaults + Warnings for Old Checkpoints
- **Priority**: High
- **Behavior Gap**: No defined behavior when loading older configs missing DFM fields.
- **Guideline (What to implement)**: If DFM fields are missing, default them (e.g., `dfm_num_steps=12`, `dfm_schedule=cosine`, `dfm_temperature=1.0`, `dfm_adaptive_step=True`). If a diffusion-trained checkpoint is used with DFM, log a warning about expected quality drop.
- **Notes/Defaults**: DFM remains opt-in; diffusion behavior is unchanged.

### 11) Early-Exit Rules
- **Priority**: Low
- **Behavior Gap**: Exact early-exit conditions are not specified.
- **Guideline (What to implement)**: Stop if there are no unresolved `[MASK]` tokens among unclamped positions, or if a step produces `num_changed_tokens == 0`. Optionally support `early_exit_frac` threshold.
- **Notes/Defaults**: Treat as non-blocking; do not block parity work on this.

### 12) Dtype Safety (fp32 Prob Math)
- **Priority**: Low
- **Behavior Gap**: No explicit guidance on probability math precision.
- **Guideline (What to implement)**: Run model forward in bf16/fp16, but cast logits to fp32 for softmax, sampling, and CTMC update math.
- **Notes/Defaults**: This is stability-focused and can be added after parity.

### 13) DFM-Specific Logging/Diagnostics
- **Priority**: Low
- **Behavior Gap**: Missing explicit DFM metrics for debugging solver behavior.
- **Guideline (What to implement)**: Add `kappa_mean`, `mask_frac_mean`, `w_mean`, `frac_w_clipped`, `dfm_nfe_realized`, `dfm_early_exit_iter`, `dfm_dt_safe_hits`, `dfm_dt_under_min`, and per-step `num_changed_tokens` summaries.
- **Notes/Defaults**: Useful for tuning and debugging; not a correctness blocker.

---

## Coverage Checklist
- [ ] Tokenizer/config injection for DFM mirrors diffusion and is persisted
- [ ] Multimodal loss masking is action-only with `IGNORE_INDEX`
- [ ] Token-count (not length) normalization is used
- [ ] Per-example time sampling and `n_mask` clamping are defined
- [ ] `kappa_dot/(1-kappa)` is clipped with explicit defaults
- [ ] CTMC step-size precedence rule is defined
- [ ] Corrector spec (if retained) is explicitly defined
- [ ] Clamp semantics are explicit and invariant
- [ ] Temperature applies to logits only; anneal is optional
- [ ] Compatibility defaults and warnings for old checkpoints are defined
- [ ] Early exit rules are documented (Low)
- [ ] Dtype safety is documented (Low)
- [ ] DFM logging metrics are documented (Low)
