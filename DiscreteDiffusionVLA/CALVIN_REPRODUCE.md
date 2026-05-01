# Reproducing DFM-VLA on CALVIN ABC→D

This guide trains DFM-VLA on CALVIN's ABC→D split (4 environments,
chained long-horizon language-conditioned tasks).

| Item | Value |
|------|-------|
| Robot | Franka Panda (CALVIN sim) |
| Eval suite | CALVIN ABC→D (1000 sequences × 5 chained subtasks) |
| Training data | `zhouhongyi/calvin_abc_rlds` (HF community RLDS conversion) |
| Action dim | 7 (delta xyz + delta euler + gripper) |
| State dim | 15 (full CALVIN proprio) |
| Cameras | 2 (`rgb_static` 200×200 + `rgb_gripper` 84×84) |

## 1. Environment

Same conda env as `DFM_REPRODUCE.md`. No extra installs needed.

## 2. Download dataset

CALVIN data is the **ABC→D split** (paper-standard, harder than ABCD→D),
hosted as a community RLDS conversion on Hugging Face
(`zhouhongyi/calvin_abc_rlds`, ~80 GB, 512 train + 32 validation shards).

> **Use `git clone` (LFS), not `huggingface-cli download`** — the HF CLI
> repeatedly fails with 416 Range errors on a couple of shards.

```bash
sudo apt-get install -y git-lfs && git lfs install   # if not already installed
cd $REPO_ROOT/data/RLDS
git clone https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds
```

After clone, **reorganize to TFDS layout** (`<dataset>/<version>/...`):

```bash
cd $REPO_ROOT/data/RLDS/calvin_abc_rlds
mkdir -p 1.0.0
mv calvin_abc-train.tfrecord-* calvin_abc-validation.tfrecord-* \
   features.json dataset_info.json 1.0.0/
```

Verify:

```bash
ls 1.0.0/calvin_abc-train.tfrecord-*      | wc -l   # expect 512
ls 1.0.0/calvin_abc-validation.tfrecord-* | wc -l   # expect 32
ls 1.0.0/features.json 1.0.0/dataset_info.json
```

If git clone misses a couple shards (HF CDN occasionally serves bad bytes for
certain files), patch them directly:

```bash
cd $REPO_ROOT/data/RLDS/calvin_abc_rlds/1.0.0
for f in 00100 00101; do
  [ -f calvin_abc-train.tfrecord-${f}-of-00512 ] || \
    wget -q "https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds/resolve/main/calvin_abc-train.tfrecord-${f}-of-00512" \
         -O calvin_abc-train.tfrecord-${f}-of-00512
done
```

(After download, you can `rm -rf $REPO_ROOT/data/RLDS/calvin_abc_rlds/.git` to
reclaim ~30 GB of git-lfs object storage that's no longer needed.)

> **Reliability check**: After the first epoch, inspect
> `$RUN_ROOT/<run_dir>/dataset_statistics.json` and verify `action.min` / `action.max`
> match CALVIN's expected ranges (delta xyz ≈ ±0.02, delta euler ≈ ±0.05,
> gripper ∈ {-1, 1}). If wildly off, the conversion may be miscalibrated and
> we'd need to fall back to converting from the official HDF5.

## 3. Embodiment constants

Detected automatically (`calvin` → `CALVIN_CONSTANTS`):

| Param | Value |
|-------|-------|
| NUM_ACTIONS_CHUNK | 8 |
| ACTION_DIM | 7 |
| PROPRIO_DIM | 15 (CALVIN's full state, kept as-is) |
| Norm | bounds_q99 |

## 4. Training

```bash
DATASET_NAME=calvin_abc_rlds \
RUN_ROOT=$HOME/checkpoints/dfm-vla-calvin-100k \
NUM_IMAGES_IN_INPUT=2 \
BATCH_SIZE=8 \
MAX_STEPS=100001 \
SAVE_FREQ=5000 \
DECAY_START=50000 \
bash scripts/train_dfm_calvin.sh --resume
```

**Use `--resume` from day one — same command works for fresh start and resume.**

### Configuration rationale (no published 1:1 reference)

CALVIN is not part of the concurrent DDVLA paper (which only reports
LIBERO + SimplerEnv-Fractal + SimplerEnv-Bridge). We reuse the SimplerEnv
config as a sensible default:

| Param | Value | Rationale |
|-------|-------|-----------|
| Total batch | 32 | Matches SimplerEnv config |
| Action chunk | 8 | Franka, same as LIBERO/Fractal |
| Max steps | 100k | Same as SimplerEnv |
| LR / decay | 5e-4 → 5e-5 at step 50k | Same as SimplerEnv |
| Save freq | 5000 (~2-3 h on 4×A100) | 20 checkpoints |

### Expected training time

~2.5 days on 4×A100-80GB (slightly slower than SimplerEnv due to
2 cameras + 15-dim proprio).

## 5. Evaluation

```bash
# Server (terminal 1)
python experiments/robot/vla_eval_adapter/vla_dfm_server.py \
    --pretrained_checkpoint $HOME/checkpoints/dfm-vla-calvin-100k/<run_dir>/<step>_chkpt \
    --unnorm_key calvin_abc_rlds \
    --use_discrete_flow_matching \
    --dfm_decode_mode ctmc \
    --num_images_in_input 2 \
    --use_proprio \
    --center_crop \
    --chunk_size 8 \
    --port 8000

# Eval (terminal 2, in vla-evaluation-harness)
vla-eval run --config configs/calvin_eval.yaml --server-url ws://localhost:8000
```

CALVIN eval = 1000 sequences × up to 5 chained subtasks.

## 6. Implementation Notes

**Two cameras at different resolutions**: `rgb_static` is 200×200, `rgb_gripper`
is 84×84. The frame-transform pipeline resizes both to 224×224 internally —
no manual handling needed.

**15-dim proprio**: Kept as-is (CALVIN's full state). The `proprio_projector`
maps it to model embedding dim. No truncation/padding.

**Language instruction**: CALVIN ABC→D demonstrations are language-annotated
(e.g. "open the drawer", "push the red block"). The harness eval also
provides language for each subtask.
