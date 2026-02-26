# Discrete Flow Matching (DFM) Implementation and Codebase Structure — DiscreteDiffusionVLA

**Scope:** This document covers DFM in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA` only.

**Goal:** Provide a detailed, end-to-end description of the discrete flow matching implementation and a directory-level map of the codebase, with explicit file references.

---

## High-Level Summaries

### Discrete Diffusion VLA (What It Does)

Discrete Diffusion VLA uses a **discrete tokenization of actions** and learns to generate those action tokens **by iterative denoising**. Instead of predicting continuous actions directly, the model fills in masked action tokens conditioned on the multimodal context (language prompt + images + optional proprioception). In training, action tokens are selectively masked and the model learns to reconstruct them. At inference, it starts from masked action tokens and iteratively refines them until a full action chunk is produced, which is then de-tokenized into continuous control values. This is the discrete diffusion path that serves as the main alternative to DFM in the action-decoding stack.

Key touchpoints for discrete diffusion (for contrast with DFM) include the action-token masking and inference branches in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py` and the training configuration toggles in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py`.

### Discrete Flow Matching (What It Tries To Do)

Discrete Flow Matching (DFM) aims to produce the same **discrete action tokens** as discrete diffusion, but via a **continuous-time flow** rather than a fixed-step denoising chain. It introduces a schedule `kappa(t)` and uses **CTMC hazard / tau-leaping updates** to move masked tokens toward their final values. The goal is **efficient, stable discrete generation** that can exit early if tokens converge, while maintaining strong alignment with the model’s conditional distribution. In short, DFM is an alternative decoding path that tries to match the same end distribution as discrete diffusion, but through a continuous-time, hazard-driven update mechanism.

The core mechanics are implemented in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_decode.py` and scheduled by `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_schedule.py`, with an optional MaskGIT-style corrector in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/parallel_decode.py`.

### LIBERO Evaluation (What It Does)

The LIBERO evaluation pipeline runs trained policies on the LIBERO simulation benchmarks and reports success rates across task suites. The top-level script `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py` validates configuration constraints (including mutual exclusivity of DFM vs discrete diffusion or continuous heads), seeds the environment, loads the model and its auxiliary components, initializes the task suite, and then iterates over tasks and episodes to compute per-task and aggregate success rates. The script logs results locally (and optionally to Weights & Biases), and it can also emit DFM-specific statistics like NFE realized, early-exit iterations, **final mask fraction** (`DFM/Mask Frac Final`), and **in-action fraction** (`DFM/InActionFracFinal`) when DFM is enabled. Current eval defaults for DFM are tuned for stability: `dfm_schedule="linear"`, `dfm_num_steps=64`, and `dfm_early_exit=False`.

The evaluation loop runs per task and per episode: it resets the environment, optionally loads fixed initial states, prepares observations (third-person and wrist images plus proprio), queries the policy to get an **action chunk**, and then executes actions open-loop for a fixed number of steps before requerying. This “action queue” behavior aligns with `NUM_ACTIONS_CHUNK` in the model. Core helpers and preprocessing live in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/libero_utils.py` and `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/robot_utils.py`. The model-side inference path (including setting the mask token if needed and passing DFM parameters into `predict_action`) is handled in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/openvla_utils.py`. The high-level setup instructions for LIBERO are summarized in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/LIBERO.md`.

---

## DFM Overview (What Is Implemented Here)

The DFM implementation in this repo performs **discrete token sampling with a continuous-time Markov chain (CTMC)** style update rule (hazard / tau-leaping). The main flow is:

1. Start from a masked action-token sequence (or a partially clamped sequence).
2. At each time step `t`, query the model for token logits and sample candidate token ids (mask token is never sampled if it is inside the logits vocab range).
3. Compute a **hazard rate** from the schedule `kappa(t)` and `kappa_dot(t)`.
4. Convert hazard into a per-step update probability `p_update = 1 - exp(-h * hazard)`.
5. Update only **unresolved (masked)** positions stochastically with the sampled ids (respecting clamps).
6. Optionally run a **MaskGIT-style corrector** pass that re-masks low-confidence tokens.

Core implementation files:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_decode.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_schedule.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/parallel_decode.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/mask_schedule.py`

---

## DFM Decode (CTMC Hazard / Tau-Leaping)

**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_decode.py`

### Function Signature and Roles
`dfm_decode(...)` runs the CTMC update loop. It expects:

- `init_ids`: initial token ids `[B, L]`.
- `tokens_to_logits`: callable that maps token ids to `(logits, hidden_states)`.
- `mask_token_id`: token id for mask positions.

It returns `(final_ids, actions_hidden_states, stats)`.

### Key Steps (Implementation Detail)
1. **Initialize:** copy `init_ids` to `cur`, set clamp mask/value if provided.
2. **Time grid:** `time_grid(num_steps, eps=time_eps)` yields `t_grid` and `dt_grid` in `(eps, 1-eps)`.
3. **Per-step loop:** for each `t` in `t_grid`:
   - **Early exit:** if no unresolved tokens remain (mask tokens not clamped), break.
   - **Model query:** `tokens_to_logits(cur)` → logits and last hidden states.
   - **Temperature:** optional linear anneal (`temperature_anneal == "linear"`).
   - **Sample ids:** multinomial sample from `softmax(logits)` (parallel categorical), with the mask token suppressed if it is in-range. In DFM inference, `tokens_to_logits` additionally hard-masks logits to the action-token vocab range before sampling (see Inference section).
   - **Hazard:** `kappa(t)` and `kappa_dot(t)` set `hazard = kdot / (1 - kappa)`.
   - **Step size:** start with `dt_grid[step]`, optionally cap by the “safe” step `(1 - kappa)/kdot`, then clamp to `step_max`, and enforce `step_min` if possible.
   - **Update probability:** `p_update = 1 - exp(-h * hazard)`, broadcast to `[B, L]`.
   - **Apply update:** update positions where `rand < p_update` AND sampled id differs AND not clamped AND **unresolved**.
   - **Per-step stats:** record number of changed tokens; early exit triggers only when unresolved tokens are gone (no early exit on “no changes”).
4. **Corrector (optional):** if `corrector=True`, re-mask low-confidence tokens and run a short MaskGIT-style decode (see `parallel_decode.decode`).
5. **Stats:** returns `dfm_nfe_realized`, `dfm_early_exit_iter`, `dfm_dt_safe_hits`, `dfm_dt_under_min`, `dfm_num_changed_tokens`, `dfm_mask_frac_final`, and `dfm_unresolved_final`.

### DFM Decode Parameters (Summary Table)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `num_steps` | `12` | Number of CTMC steps (time grid length). |
| `schedule` | `"cosine"` | `kappa(t)` schedule (`cosine`, `linear`, `poly2`). |
| `temperature` | `1.0` | Sampling temperature for logits. |
| `temperature_anneal` | `"none"` | Optional linear anneal to 1.0. |
| `adaptive_step` | `True` | Enforce safe step upper bound `(1-kappa)/kdot`. |
| `step_min` | `1e-4` | Minimum allowed step size if feasible. |
| `step_max` | `0.2` | Maximum allowed step size. |
| `time_eps` | `1e-3` | Avoid exact endpoints for t-grid. |
| `early_exit` | `True` | Exit only when unresolved tokens are gone (no early-exit on “no changes”). |
| `early_exit_frac` | `0.0` | Exit if change fraction below threshold. |
| `corrector` | `False` | Enable MaskGIT re-mask corrector. |
| `corrector_iters` | `1` | Number of corrector iterations. |
| `corrector_remask_frac` | `0.1` | Fraction of lowest-confidence tokens to re-mask. |
| `clamp_mask` | `None` | Boolean mask to freeze positions. |
| `clamp_values` | `None` | Values to force at clamped positions. |

---

## DFM Schedules

**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_schedule.py`

### Functions
- `kappa(t, schedule)`
- `kappa_dot(t, schedule)`
- `time_grid(num_steps, eps)`

### Implemented Schedules
- `cosine`: `kappa(t) = 1 - cos(0.5 * pi * t)`
- `linear`: `kappa(t) = t`
- `poly2`: `kappa(t) = t^2`

`kappa_dot` returns the derivative for each schedule. `time_grid` returns `(t, dt)` with `t` in `(eps, 1-eps)` and `sum(dt) = 1 - eps`.

---

## MaskGIT Components (Used by DFM Corrector)

**Files:**
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/parallel_decode.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/mask_schedule.py`

### Purpose
The DFM decoder can optionally run a **corrector** stage that re-masks low-confidence tokens and runs a short MaskGIT-style parallel decode. This uses:

- `mask_schedule.schedule(...)` to decide how many tokens to re-mask per step.
- `parallel_decode.decode(...)` to iteratively sample and mask tokens in parallel.

### Key Notes
- The corrector uses `mask_scheduling_method` derived from the DFM `schedule` (e.g., `cosine` or `powX`).
- It re-masks the **lowest-confidence** tokens and refines them with a short, fixed number of iterations.

---

## Training Integration (DFM in Model Forward Pass)

**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

### Where DFM Enters the Training Graph
During multimodal forward:

1. **Action mask extraction:** `all_actions_mask = self._process_action_masks(labels)` identifies the action-token positions.
2. **DFM branch:** when `self.use_discrete_flow_matching` is true:
   - `apply_mask_flow_matching(...)` corrupts action tokens according to DFM schedule.
   - It returns `dfm_loss_mask`, `kappa_t`, `kdot_t`, and masked inputs/labels.
   - `dfm_weight = (kdot_t / (1 - kappa_t)).clamp(max=dfm_weight_clip)`.
3. **Language model forward:** the masked multimodal sequence is fed through the LLM.
4. **Loss override:** DFM overrides the standard LM loss using `dfm_loss_mask` and optional weighting.

### `apply_mask_flow_matching` Behavior
**Location:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

Key logic:
- Sample `t ~ Uniform([t_min, t_max])`, clamped into `(eps, 1-eps)`.
- Compute `kappa_t` and `kdot_t`.
- Derive mask ratio `mask_ratio = 1 - kappa_t`.
- Select a per-example number of masked tokens (at least 1).
- Build `masked_input_ids`, `masked_labels`, `loss_mask`.

Outputs:
- `dfm_loss_mask`: float mask (`1` on supervised tokens, `0` elsewhere).
- `kappa_t`, `kdot_t`: used to weight the loss.

### DFM Loss Computation
**Location:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

- Token-level CE loss is computed for the multimodal sequence.
- The mask is expanded to include visual patch tokens (inserted zeros).
- Weighting depends on `dfm_loss_mode`:
  - `masked_ce`: weights are all `1` on masked positions.
  - `generalized_kl`: weights are `dfm_weight` expanded over positions.
- Final loss is `sum(masked_loss) / sum(mask * weight)`.

### Debug and Safeguards
- Env vars: `VLA_DFM_DEBUG=1` and `VLA_DFM_DEBUG_EVERY` control periodic logging of mask stats.
- The model raises if both discrete diffusion and DFM are enabled at the same time.

---

## Inference / Prediction Path (DFM at Test Time)

**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

### Entry Point
`OpenVLAForActionPrediction.predict_action(...)` controls the inference path. It routes to DFM when:

- `use_discrete_flow_matching=True` and
- `use_discrete_diffusion=False` and
- diffusion action heads are not used.

### `_discrete_flow_matching_prediction` Flow
**Location:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

1. Build masked sequences where all action tokens are set to `mask_token_id`.
2. Prepare a `tokens_to_logits` closure that:
   - Splices `suffix_seq` into the full sequence.
   - Builds multimodal embeddings.
   - Calls the language model to obtain logits and hidden states.
   - Hard-masks logits to the action-token vocab range (the last `n_bins` tokens) before sampling; this applies to both the main decode and the corrector because they share the same closure.
3. Apply clamp logic if `dfm_clamp_mask` or `dfm_clamp_values` are provided.
4. Call `dfm_decode(...)` with CTMC parameters.
5. Convert final token ids to discrete action bins, then to normalized actions.

### Clamp and Early Exit Behavior
- If `dfm_clamp_values` is set, tokens are forced to those values at clamped positions.
- `dfm_early_exit` stops decoding only when all masked positions are resolved (no early exit on “no changes”).

### Mask-Token Collision Warning
In `_discrete_flow_matching_prediction`, the model emits a warning if `mask_token_id` overlaps the action-token range (i.e., it falls inside the discretized action vocabulary). This is a safety check to flag misconfiguration that can corrupt decoding. See `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`.

### Corrector
If `dfm_corrector=True`, the decoder re-masks low-confidence positions and runs a short MaskGIT pass.

---

## Configuration and Flags

### Model Configuration
**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/configuration_prismatic.py`

Key flags:
- `use_discrete_diffusion: bool`
- `use_discrete_flow_matching: bool`
- `mask_token_id`, `use_mask_token`

Methods:
- `set_discrete_flow_matching(...)`
- `set_mask_token_id(...)`

### Training Configuration
**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py`

Important DFM fields in `FinetuneConfig`:
- `use_discrete_flow_matching`
- `dfm_schedule`
- `dfm_time_eps`
- `dfm_t_min`, `dfm_t_max`
- `dfm_loss_mode` (`generalized_kl` or `masked_ce`)
- `dfm_weight_clip`

### Scripts and CLI Flags

DFM training examples:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/finetune.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/train_dfm_slurm.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/finetune_slurm.sh`

These pass:
- `--use_discrete_flow_matching True`
- `--use_discrete_diffusion False`
- DFM hyperparameters: `--dfm_schedule`, `--dfm_loss_mode`, `--dfm_time_eps`, `--dfm_t_min`, `--dfm_t_max`, `--dfm_weight_clip`

### Inference / Evaluation Configuration
**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py`

`GenerateConfig` exposes DFM inference controls:
- `dfm_num_steps`, `dfm_schedule`, `dfm_temperature`, `dfm_temperature_anneal`
- `dfm_adaptive_step`, `dfm_step_min`, `dfm_step_max`, `dfm_time_eps`
- `dfm_early_exit`, `dfm_early_exit_frac`
- `dfm_corrector`, `dfm_corrector_iters`, `dfm_corrector_remask_frac`
- `dfm_clamp_mask`, `dfm_clamp_values`

### Mutual-Exclusion Constraints
- Discrete diffusion and DFM are mutually exclusive; the model raises if both are enabled.
- DFM is not supported with diffusion action heads in `predict_action`.

---

## DFM Metrics and Tests

### Decode Stats
**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_decode.py`

Returned stats dictionary includes:
- `dfm_nfe_realized`: number of steps actually executed.
- `dfm_early_exit_iter`: step where early exit occurred (or `-1`).
- `dfm_dt_safe_hits`: count of times adaptive-step safety bound was active.
- `dfm_dt_under_min`: count of steps where safe step < `step_min`.
- `dfm_num_changed_tokens`: list of changed token counts per step.
- `dfm_mask_frac_final`: fraction of tokens still equal to `mask_token_id` at the end (should be near 0 when decoding succeeds).
- `dfm_unresolved_final`: count of unresolved (masked) tokens at the end.
- `dfm_in_action_frac_final`: fraction of decoded tokens that fall inside the action-token vocab range (should be ~1.0 with action-vocab masking).

### Unit Tests
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/tests/test_dfm_decode.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/tests/test_dfm_schedule.py`

These validate:
- `kappa` endpoints and monotonic properties.
- `kappa_dot` non-negativity.
- `time_grid` shape and bounds.
- DFM decode invariants and stats bounds.

### Perf Smoke Test
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/tests/perf/test_dfm_decode_perf.py`

Uses `pytest -m perf` with `RUN_PERF=1` to check that `dfm_nfe_realized` is bounded under a small config.

---

## Codebase Structure (DiscreteDiffusionVLA)

Top-level directories and their purpose:

- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/`: Core model code, including DFM implementation and HF integration.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/`: Evaluation / robot environment integration (LIBERO, ALOHA).
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/`: Training and utilities (fine-tune, deploy, merge LoRA).
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/scripts/`: Helper shell scripts (eval, setup).
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/`: SLURM launch scripts (includes DFM toggles).
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/tests/`: Unit and perf tests (DFM tests here).
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/assets/`: Project assets (figures).
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/LIBERO/`: External benchmark support folder.

### DFM-Relevant Subtrees

- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/`: DFM core logic and schedules.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/`: HuggingFace model integration, including DFM training and inference logic.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/`: Evaluation and inference wrappers that pass DFM params to the model.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py`: DFM training config and CLI flags.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/`: DFM-enabled training scripts.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/tests/`: DFM unit/perf tests.

---

## Quick Navigation (Key Files)

| Area | File | Responsibility |
| --- | --- | --- |
| DFM Decode | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_decode.py` | CTMC hazard / tau-leaping decoder; updates only masked tokens and suppresses mask-token sampling, with adaptive step, clamp, and corrector. |
| DFM Schedule | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_schedule.py` | `kappa`, `kappa_dot`, `time_grid` schedule definitions. |
| MaskGIT | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/parallel_decode.py` | Parallel decode for optional corrector stage. |
| Mask Schedule | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/mask_schedule.py` | Masking ratio schedules used by MaskGIT. |
| Training Integration | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py` | DFM masking, weighting, loss override, prediction path, action-vocab masking, in-action telemetry, and mask-token collision warning in DFM inference. |
| Model Config | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/configuration_prismatic.py` | DFM flags in model config and setters. |
| Finetune CLI | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py` | Training config for DFM and run-time flags. |
| DFM Training Script | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/train_dfm_slurm.sh` | Example SLURM launch with DFM hyperparameters. |
| DFM Fine-tune Example | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/finetune.sh` | Example torchrun invocation with DFM toggles. |
| Evaluation Config | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py` | DFM inference config and runtime parameters. |
| Inference Helper | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/openvla_utils.py` | Passes DFM parameters into `predict_action` and ensures mask token is set. |
| DFM Tests | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/tests/test_dfm_decode.py` | Decode correctness and stat bounds. |
| DFM Tests | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/tests/test_dfm_schedule.py` | Schedule correctness checks. |
| DFM Perf Test | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/tests/perf/test_dfm_decode_perf.py` | Perf smoke test gated by `RUN_PERF=1`. |

---

## Notes and Practical Pitfalls

- DFM and discrete diffusion are **mutually exclusive**. Confirm your config and scripts only enable one.
- DFM relies on a **mask token** being present in the tokenizer; evaluation helpers set it if missing.
- DFM loss weighting uses `kdot / (1-kappa)` and is clipped to avoid unstable gradients.
- The corrector step is optional; it adds compute but can refine low-confidence tokens.

---

## Suggested Reading Order

1. DFM core: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_decode.py`
2. Schedule logic: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/dfm_schedule.py`
3. Training integration: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`
4. Finetuning config: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py`
5. Evaluation config: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py`
