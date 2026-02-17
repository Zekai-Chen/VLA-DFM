# Full CTMC Discrete Flow Matching (DFM) Integration Plan

**Summary**
Implement a full DFM path alongside discrete diffusion: CTMC hazard/tau‑leaping solver, adaptive step sizing, generalized KL / weighted CE training. Use mask‑source + convex mixture, time‑independent denoiser, and mask‑only corruption for parity and stability. Discrete diffusion remains intact and selectable.

## Public API / Config Changes
- **Training (`vla-scripts/finetune.py`)**
  - `use_discrete_flow_matching: bool`
  - `dfm_schedule: str` (`cosine|linear|poly2`)
  - `dfm_time_eps: float`
  - `dfm_loss_mode: str` (`generalized_kl|masked_ce`)
  - `dfm_weight_clip: float`

- **Inference (`experiments/robot/libero/run_libero_eval.py`)**
  - `use_discrete_flow_matching: bool`
  - `dfm_num_steps: int`
  - `dfm_schedule: str`
  - `dfm_temperature: float`
  - `dfm_adaptive_step: bool`
  - `dfm_step_min: float`, `dfm_step_max: float`
  - `dfm_time_eps: float`
  - `dfm_early_exit: bool`
  - `dfm_corrector: bool`
  - `dfm_clamp_mask: bool`

- **Model config (`prismatic/extern/hf/configuration_prismatic.py`)**
  - `use_discrete_flow_matching: bool` + `set_discrete_flow_matching()`

## Core Implementation Steps
1. **Scheduling utilities**
   - Add `prismatic/discrete_flow/dfm_schedule.py` with `kappa(t)`, `kappa_dot(t)`, and `time_grid()`.

2. **Training forward path**
   - Add `apply_mask_flow_matching()` to `PrismaticForConditionalGeneration`.
   - Integrate DFM masking into `forward()`; compute weighted CE with `kappa_dot/(1-kappa)` (clamped).

3. **CTMC inference solver**
   - Add `prismatic/discrete_flow/dfm_decode.py` implementing hazard/tau‑leaping updates, adaptive step, early exit, optional corrector.

4. **Prediction path**
   - Add `_discrete_flow_matching_prediction()` in `modeling_prismatic.py` using `dfm_decode`.
   - Route `predict_action()` to DFM when enabled.

5. **Training integration**
   - Wire DFM flags into `FinetuneConfig` and `run_forward_pass()`.
   - Enforce mutual exclusivity with discrete diffusion and continuous action heads.

6. **Evaluation / inference wiring**
   - Propagate DFM flags through `openvla_utils.py`, `robot_utils.py`, and `run_libero_eval.py`.

7. **Scripts / docs**
   - Update `finetune.sh` and `finetune_from_ckpt.sh` with DFM examples.

## Tests & Validation
- **Unit sanity**: schedule functions; `dfm_decode` stability.
- **Training smoke**: 1–2 steps with DFM enabled.
- **Inference smoke**: 1–2 rollouts in LIBERO eval.
- **A/B sanity**: discrete diffusion unchanged when DFM disabled.

## Assumptions & Defaults
- No external `flow_matching` dependency.
- Default schedule: `cosine`, `dfm_num_steps=12`.
- DFM loss defaults to `generalized_kl` with weight clipping.
- Adaptive step enabled by default.
