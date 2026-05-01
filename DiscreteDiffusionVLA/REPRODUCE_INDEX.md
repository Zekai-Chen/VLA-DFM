# DFM-VLA: Benchmark Reproduction Index

| Benchmark | Doc | Dataset | Steps | Time on 4×A100 |
|-----------|-----|---------|-------|---------------|
| LIBERO (Spatial / Object / Goal / Long) | [DFM_REPRODUCE.md](DFM_REPRODUCE.md) | `modified_libero_rlds` (~20 GB) | 320k | ~6 days |
| SimplerEnv (Bridge / Fractal) | [SIMPLER_REPRODUCE.md](SIMPLER_REPRODUCE.md) | `bridge_oxe` (~390 GB) + `fractal20220817_data` (~110 GB) | 100k | ~2 days each |
| ManiSkill2 | [MANISKILL_REPRODUCE.md](MANISKILL_REPRODUCE.md) | `maniskill_dataset_converted_externally_to_rlds` (~150 GB) | 100k | ~2.3 days |
| CALVIN ABC→D | [CALVIN_REPRODUCE.md](CALVIN_REPRODUCE.md) | `calvin_abc_rlds` (HF, ~80 GB) | 100k | ~2.5 days |

## Common setup (all benchmarks share)

1. **Conda env**: install once per `DFM_REPRODUCE.md` Section 1. Used for all 4 benchmarks.
2. **Base model**: `~/data/models/openvla-7b` (~15 GB).
3. **Repo**: `git clone --branch rl-dfm-finetune https://github.com/a10v/VLA-DFM.git`.
4. **Convention**: every script uses `--resume` from day one. Same command works for fresh start (no checkpoint → fresh) and resume (checkpoint exists → auto-resume from latest).
5. **Save format**: checkpoints written to `$RUN_ROOT/<run_id>--<step>_chkpt/`.
6. **Evaluation**: shared adapter at `experiments/robot/vla_eval_adapter/vla_dfm_server.py`; harness at `https://github.com/allenai/vla-evaluation-harness`.

## Quick run-all (parallel, 4 machines × 4 GPUs each)

Machine 1: `bash scripts/train_dfm.sh --resume` (LIBERO-Object, default)
Machine 2: `DATASET_NAME=bridge_oxe RUN_ROOT=... USE_FILM=True bash scripts/train_dfm_simpler.sh --resume`
Machine 3: `DATASET_NAME=fractal20220817_data RUN_ROOT=... bash scripts/train_dfm_simpler.sh --resume`
Machine 4: `DATASET_NAME=maniskill_dataset_converted_externally_to_rlds RUN_ROOT=... bash scripts/train_dfm_maniskill.sh --resume`
+ CALVIN: `DATASET_NAME=calvin_abc_rlds RUN_ROOT=... bash scripts/train_dfm_calvin.sh --resume`

See per-benchmark docs for full env-var list and download commands.
