# Discrete Diffusion Implementation and Codebase Structure — DiscreteDiffusionVLA

**Scope:** This document covers the **discrete diffusion (mask-based action-token)** pipeline in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA`. It explicitly **excludes the continuous diffusion action head (DDIM)** and focuses only on the discrete token path.

**Goal:** Provide a detailed, end-to-end description of the discrete diffusion implementation and a directory-level map of the codebase, with explicit file references.

---

## High-Level Summary (What Discrete Diffusion Does)

Discrete Diffusion VLA converts continuous actions into **discrete action tokens** and learns to generate those tokens using a **masked language modeling** style objective. During training, the model randomly masks a subset of action tokens and learns to reconstruct them given the rest of the multimodal context (language + images + optional proprioception). During inference, it starts from an all-mask action token sequence and **iteratively refines** it using a MaskGIT-style parallel decoding loop. The final action tokens are then de-tokenized back into continuous actions using the action bin centers.

Core files for discrete diffusion:
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/parallel_decode.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/mask_schedule.py`

---

## Discrete Diffusion Overview (What Is Implemented Here)

The discrete diffusion implementation uses a **mask-only corruption path** and a **parallel token refinement decoder**:

1. **Training:** mask a subset of action tokens using a schedule, then apply a standard masked-CE loss on the masked positions.
2. **Inference:** initialize all action tokens to `mask_token_id`, then iteratively sample replacements using a MaskGIT-style loop.
3. **De-tokenization:** convert final action token ids into discrete bins and map them to normalized action values via `bin_centers`.

This design is intentionally aligned with the discrete action-token vocabulary. It **does not** use the continuous diffusion action head (DDIM) and does **not** incorporate DFM hazard updates.

---

## Training Integration (Discrete Diffusion)

**Primary file:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

### `apply_mask_diffusion(...)`
This is the training-time corruption routine for discrete diffusion. It:

1. Samples a random **time ratio**: `rand_time ~ Uniform(0, 1)`.
2. Computes a **mask ratio** using `mask_schedule(rand_time, total_unknown, method="cosine")`.
3. Computes `num_mask >= 1` per example based on `total_unknown * mask_ratio`.
4. Selects the lowest-ranked random positions among maskable tokens.
5. Builds:
   - `masked_input_ids`: masked positions replaced with `mask_token_id`.
   - `masked_labels`: masked positions keep original ids, others set to `IGNORE_INDEX`.
   - `loss_mask`: float mask (`1.0` on masked positions, `0.0` elsewhere).

Key references:
- `apply_mask_diffusion(...)` in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`
- `mask_schedule(...)` in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/mask_schedule.py`

### Forward-pass Integration
Inside the multimodal forward pass:

- The `use_discrete_diffusion` branch calls `apply_mask_diffusion(...)`.
- The model then runs the standard language model forward pass.
- The **default CE loss** is applied using `masked_labels`. There is **no DFM weighting** and no alternate loss mode for discrete diffusion.
- The EOS position is computed and updated for masked sequences using `_get_eos_pos(...)` and `STOP_INDEX`.

Mutual-exclusion constraints:
- Discrete diffusion cannot be enabled alongside DFM or continuous diffusion action heads.
- Enforced in `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py` and `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`.

---

## Inference / Prediction Path (Discrete Diffusion)

**Primary file:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

### `_discrete_diffusion_prediction(...)`
This is the discrete diffusion inference path for action tokens. The flow is:

1. **Mask all action tokens** by setting them to `mask_token_id` while leaving EOS/STOP tokens intact.
2. **Construct `tokens_to_logits` closure** that:
   - Inserts the candidate suffix into the full sequence.
   - Builds multimodal embeddings and attention masks.
   - Runs the language model and extracts logits for the action-token span.
3. **MaskGIT decode** via `parallel_decode.decode(...)` with:
   - `num_iter=12`
   - `mask_scheduling_method="cosine"`
   - `choice_temperature=1.0`
   - `use_remask=False`
4. **Select final tokens** using `final_iters[:, -1, :]`.
5. **De-tokenize**:
   - `discretized_actions = self.vocab_size - predicted_action_token_ids`
   - `np.clip(discretized_actions - 1, 0, n_bins-1)`
   - Lookup `bin_centers` to produce normalized actions.

### Mask-Token Collision Warning
The same safety warning used in DFM is also triggered here if `mask_token_id` overlaps the action-token range. This protects against silently treating a valid action token as a mask.

### Note on Top-k Filtering
A `topk_filter_thres` configuration exists, but **top-k filtering is commented out** in `_discrete_diffusion_prediction`. The current discrete diffusion path uses **full logits** without top-k pruning.

---

## Mask Schedule + MaskGIT Decode

### Mask Schedule
**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/mask_schedule.py`

`mask_schedule.schedule(ratio, total_unknown, method)` produces a **mask ratio** in `(0, 1]`:
- Supported methods: `uniform`, `linear`, `powX`, `cosine`, `log`, `exp`.
- Outputs are clamped to `[1e-6, 1.0]` for numerical stability.

### MaskGIT Parallel Decode
**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/parallel_decode.py`

`parallel_decode.decode(...)` implements the non-autoregressive decoding loop:

1. Compute logits and convert to probabilities with `softmax`.
2. Sample all positions in parallel (`multinomial`).
3. Only update positions that are still masked.
4. Compute the next mask length using the schedule and initial unknown count.
5. Use `mask_by_random_topk(...)` (Gumbel + top-k) to decide which positions remain masked.
6. Iterate for `num_iter` steps and return all iterations.

`mask_by_random_topk` uses Gumbel noise to approximate sampling based on confidence scores and selects which tokens to keep masked.

---

## Configuration & Flags (Discrete Diffusion)

### Model Configuration
**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/configuration_prismatic.py`

Key flags:
- `use_discrete_diffusion: bool`
- `set_dicrete_diffusion(...)` (note spelling)
- `mask_token_id`, `use_mask_token`

### Training Configuration
**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py`

Key training fields:
- `use_discrete_diffusion`
- Mutual exclusion assertions with DFM and continuous heads:
  - No discrete diffusion + DFM.
  - No discrete diffusion + continuous diffusion action head.

### Evaluation Configuration
**File:** `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py`

Key eval flags:
- `use_discrete_diffusion`
- `topk_filter_thres` (currently not applied in inference path)

---

## Evaluation / Scripts

Discrete diffusion evaluation is enabled by toggling `--use_discrete_diffusion True` and disabling diffusion/DFM in the eval scripts:

- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/scripts/eval_libero_object_batch.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/eval_libero_slurm.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/eval_libero_slurm_discrete_diffusion.sh`
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/smoke/eval_libero_slurm_smoke_2a100_discrete_diffusion.sh`

---

## Metrics & Tests

- There are **no discrete-diffusion-specific unit tests** under `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/tests/`.
- Discrete diffusion relies on the standard LM cross-entropy loss (via masked labels) and does not emit special training stats beyond the default language-model outputs.

---

## Codebase Structure (DiscreteDiffusionVLA)

Top-level directories relevant to discrete diffusion:

- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/`: HuggingFace integration and discrete diffusion training/inference logic.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/`: MaskGIT decode and mask schedules (shared with DFM corrector).
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/`: Evaluation pipeline and model wrappers.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/`: Training configs and CLI flags.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/`: SLURM launch scripts.
- `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/scripts/`: Local evaluation helpers.

---

## Quick Navigation (Key Files)

| Area | File | Responsibility |
| --- | --- | --- |
| Discrete Diffusion Training | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py` | `apply_mask_diffusion`, discrete diffusion branch in multimodal forward, and masked-label CE loss. |
| Discrete Diffusion Inference | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py` | `_discrete_diffusion_prediction` and de-tokenization to action bins. |
| MaskGIT Decode | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/parallel_decode.py` | Parallel sampling loop, Gumbel top-k masking, iterative refinement. |
| Mask Schedule | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/mask_schedule.py` | Mask ratio schedules used during training and decoding. |
| Training Config | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py` | Discrete diffusion flags and mutual-exclusion checks. |
| Eval Config | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py` | Discrete diffusion toggles and eval wiring. |
| Eval Scripts | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/sbatch/eval_libero_slurm_discrete_diffusion.sh` | SLURM launch for discrete diffusion eval. |
| Eval Scripts | `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/scripts/eval_libero_object_batch.sh` | Local batch eval for discrete diffusion. |

---

## Notes and Practical Pitfalls

- Discrete diffusion is **mutually exclusive** with DFM and continuous diffusion heads.
- The action-token vocabulary must be stable; ensure `mask_token_id` does not collide with action-token ids.
- Top-k filtering exists as a config but is currently not applied in the inference path.

---

## Suggested Reading Order

1. Discrete diffusion training/inference paths: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`
2. MaskGIT decoder: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/parallel_decode.py`
3. Mask schedules: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/prismatic/discrete_flow/mask_schedule.py`
4. Training configs: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/vla-scripts/finetune.py`
5. Eval configs/scripts: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/experiments/robot/libero/run_libero_eval.py`
