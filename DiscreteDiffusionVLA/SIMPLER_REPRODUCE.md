# Reproducing DFM-VLA on SimplerEnv (Bridge V2 + Fractal/RT-1)

This guide trains DFM-VLA checkpoints for SimplerEnv evaluation. SimplerEnv has
two embodiment branches:

| Branch | Training data | Eval task suite |
|--------|--------------|-----------------|
| **WidowX** | Bridge V2 (`bridge_oxe`, ~60k traj) | `widowx_*` tasks (stack_cube, carrot_on_plate, spoon_on_tablecloth, eggplant_in_basket) |
| **Google Robot** | Fractal / RT-1 (`fractal20220817_data`, ~130k traj) | `google_robot_*` tasks (pick_coke_can, move_near, open_drawer, etc.) |

Each requires a **separate** 100k-step training run (per concurrent DDVLA paper
Appendix C). Same env / repo / install as `DFM_REPRODUCE.md` — only the dataset
and a few embodiment-specific constants change.

## 1. Environment

Same conda env as LIBERO training. See `DFM_REPRODUCE.md` Section 1 for setup.

## 2. Download datasets

Both datasets are RLDS format from Open-X-Embodiment.

```bash
# Bridge V2 (WidowX, ~600 GB)
gsutil -m cp -r gs://gresearch/robotics/bridge/0.1.0 \
    $REPO_ROOT/data/RLDS/bridge_oxe/

# Fractal / RT-1 (Google Robot, ~110 GB)
gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data/0.1.0 \
    $REPO_ROOT/data/RLDS/fractal20220817_data/
```

If `gsutil` is unavailable, both are also on Hugging Face (`openvla/modified_libero_rlds`-style mirrors exist for OXE datasets). Confirm the dataset directory matches the layout `<DATA_ROOT>/<dataset_name>/<version>/...`.

Verify after download:

```bash
ls $REPO_ROOT/data/RLDS/bridge_oxe/0.1.0/ | head
ls $REPO_ROOT/data/RLDS/fractal20220817_data/0.1.0/ | head
```

## 3. Embodiment constants

`prismatic/vla/constants.py` already includes both embodiments. The runtime
detector picks the right set from CLI args (`bridge` → BRIDGE, `fractal`/`google_robot`/`rt_1` → GOOGLE_ROBOT).

| Embodiment | NUM_ACTIONS_CHUNK | ACTION_DIM | PROPRIO_DIM | Norm |
|-----------|-------------------|------------|-------------|------|
| Bridge V2 (WidowX) | **3** | 7 | 7 | bounds_q99 |
| Fractal (Google Robot) | **8** | 7 | 8 | bounds_q99 |

**Chunk sizes match the concurrent Discrete Diffusion VLA paper (Sec 4.2)**:
LIBERO=8, Fractal=8, Bridge=3.

## 4. Training

**Config matches the concurrent Discrete Diffusion VLA paper (ICLR 2026 sub. #6223, Appendix C):**
- **100k steps** (vs 150k–300k for LIBERO)
- **Batch size 32 total** (paper used 4×A800)
- Bridge uses **FiLM** for stronger language grounding on WidowX manipulation; Fractal does not.

**Use `--resume` from day one — same command works for fresh start and resume.**

### Bridge V2 (WidowX) — with FiLM

```bash
DATASET_NAME=bridge_oxe \
RUN_ROOT=$HOME/checkpoints/dfm-vla-bridge-100k \
NUM_IMAGES_IN_INPUT=1 \
BATCH_SIZE=8 \
MAX_STEPS=100001 \
SAVE_FREQ=5000 \
DECAY_START=50000 \
USE_FILM=True \
bash scripts/train_dfm_simpler.sh --resume
```

### Fractal / RT-1 (Google Robot)

```bash
DATASET_NAME=fractal20220817_data \
RUN_ROOT=$HOME/checkpoints/dfm-vla-fractal-100k \
NUM_IMAGES_IN_INPUT=1 \
BATCH_SIZE=8 \
MAX_STEPS=100001 \
SAVE_FREQ=5000 \
DECAY_START=50000 \
bash scripts/train_dfm_simpler.sh --resume
```

> **Note**: With `BATCH_SIZE=8` per GPU on 4 GPUs → total batch 32 (paper config).
> If you have 8 GPUs, use `BATCH_SIZE=4` to keep total batch = 32.
> Estimated wall-clock: ~2 days on 4×A100-80GB at 100k steps.
> First checkpoint saves at step 5000 (~2-3 h); 20 checkpoints total (5000, 10000, …, 100000).

### Key differences vs LIBERO training

| Param | LIBERO | SimplerEnv (Bridge / Fractal) |
|-------|--------|-------------------------------|
| `--num_images_in_input` | 2 (agentview + wrist) | **1** (single primary camera) |
| `--dataset_name` | `libero_object_no_noops` etc. | `bridge_oxe` / `fractal20220817_data` |
| Embodiment constants | LIBERO_CONSTANTS | BRIDGE_CONSTANTS / GOOGLE_ROBOT_CONSTANTS |

All other hyperparameters match `train_dfm.sh`: LoRA rank 32, cosine DFM
schedule, t_max=0.7, generalized KL loss, legacy DFM vocab.

### Expected training time

~2 days per branch on 4×A100-80GB at 100k steps. Two branches in parallel
on separate machines = 2 days total.

## 5. Evaluation

Use the same `vla_eval_adapter/vla_dfm_server.py` infrastructure as for
LIBERO eval.

```bash
# Start server (terminal 1)
python experiments/robot/vla_eval_adapter/vla_dfm_server.py \
    --pretrained_checkpoint $HOME/checkpoints/dfm-vla-bridge-100k/<run_dir>/<step>_chkpt \
    --unnorm_key bridge_oxe \
    --use_discrete_flow_matching \
    --dfm_decode_mode ctmc \
    --num_images_in_input 1 \
    --use_proprio \
    --center_crop \
    --chunk_size 3 \
    --port 8000

# Run eval (terminal 2)
cd ~/vla-evaluation-harness
vla-eval run --config configs/simpler_all_tasks.yaml --server-url ws://localhost:8000
# (use simpler_google_robot_tasks.yaml for the Google Robot branch)
```

## 6. Implementation Notes

**Action tokenizer bins**: The 256-bin action tokenizer was originally fitted on
LIBERO action stats (99-percentile). When training on Bridge / Fractal, the bins
need re-fitting on those datasets — finetune.py does this automatically via the
RLDS dataset's stored statistics, but verify the `dataset_statistics.json`
written into the `RUN_ROOT` matches your dataset.

**Proprio dim mismatch**: If `PROPRIO_DIM` reported at startup doesn't match the
embodiment (e.g. `BRIDGE` shows `PROPRIO_DIM=7` not `8`), either the detector
mis-fired or the dataset key list in `oxe/configs.py` changed. Fix manually in
`prismatic/vla/constants.py`.

**Single camera**: SimplerEnv eval only provides one primary camera. Train with
`NUM_IMAGES_IN_INPUT=1` to match — otherwise the model expects a wrist image
that doesn't exist at eval time.
