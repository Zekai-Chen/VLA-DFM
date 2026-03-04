# Eval Pipeline Changes (Checkpoint-First) — 2026-03-03

## Why This Change
Recent eval runs showed misalignment between checkpoint logic (modeling/config) and repo code, which can silently degrade performance. The goal of this update is to make eval **checkpoint-first** by default and ensure inference parameters are **config-first** unless explicitly overridden, while keeping legacy checkpoints usable.

## Behavioral Changes
- **Checkpoint-first model logic**: `sync_model_logic` now defaults to `False`, so eval does **not** overwrite checkpoint code by default.
- **Config-first inference defaults**: DFM params default to `auto/0/-1` so checkpoint config is used unless set explicitly.
- **Prompt/EOS alignment**: eval uses `PurePromptBuilder("openvla")` and strips the trailing EOS token to match training-time prompt formatting.
- **Fail-fast tokenizer check**: eval raises if `mask_token_id` is missing (no silent tokenizer mutation).

## New / Updated Flags
- `--sync_model_logic` (default `False`)
- `--use_checkpoint_defaults` (default `True`)
- DFM params defaulting to checkpoint config via `auto/0/-1`:
  - `--dfm_schedule auto`
  - `--dfm_maskgit_schedule auto`
  - `--dfm_maskgit_num_steps 0`
  - `--dfm_num_steps 0`
  - `--dfm_time_eps -1`

## Scripts Updated
- `sbatch/smoke/eval_libero_slurm_smoke_2a100_discrete_diffusion.sh`
- `sbatch/smoke/eval_libero_slurm_smoke_2a100_dfm_flow_best_maskgit_tmax0.7.sh`
- `scripts/eval_libero_object_batch.sh`

## Notes / Compatibility
- **Legacy checkpoints** without new config fields are still supported.
- If `modeling_prismatic.py` is missing inside a checkpoint, a **warning** is emitted and the repo version is copied for compatibility.

## Public API / Interface Changes
`run_libero_eval.py` now exposes:
- `--sync_model_logic` (default `False`)
- `--use_checkpoint_defaults` (default `True`)
