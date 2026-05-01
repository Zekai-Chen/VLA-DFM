# Reproducing DFM-VLA on ManiSkill2

This guide trains DFM-VLA on the OXE-converted ManiSkill2 dataset for evaluation
on the ManiSkill2 benchmark (5 tasks × 50 episodes).

| Item | Value |
|------|-------|
| Robot | Franka Panda |
| Tasks (eval) | PickCube, StackCube, PickSingleYCB, PickSingleEGAD, PickClutterYCB |
| Training data | `maniskill_dataset_converted_externally_to_rlds` (~5k demos) |
| Action dim | 7 (xyz + euler + gripper) |
| State dim | 8 (tcp_pose 7 + gripper_state 1) |
| Cameras | 2 (primary + wrist) |

## 1. Environment

Same conda env as `DFM_REPRODUCE.md`.

## 2. Download dataset

```bash
# ManiSkill2 OXE version (~30 GB)
gsutil -m cp -r gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0 \
    $REPO_ROOT/data/RLDS/maniskill_dataset_converted_externally_to_rlds/

# Verify
ls $REPO_ROOT/data/RLDS/maniskill_dataset_converted_externally_to_rlds/0.1.0/ | head
```

Install `gsutil` first if missing: `pip install gsutil` (no sudo) or
`sudo snap install google-cloud-cli`.

## 3. Embodiment constants

Detected automatically from CLI args (`maniskill` → `MANISKILL_CONSTANTS`):

| Param | Value |
|-------|-------|
| NUM_ACTIONS_CHUNK | 8 (Franka Panda, same as LIBERO/Fractal) |
| ACTION_DIM | 7 |
| PROPRIO_DIM | 8 |
| Norm | bounds_q99 |

## 4. Training

**Use `--resume` from day one — same command works for fresh start and resume.**

```bash
DATASET_NAME=maniskill_dataset_converted_externally_to_rlds \
RUN_ROOT=$HOME/checkpoints/dfm-vla-maniskill-100k \
NUM_IMAGES_IN_INPUT=2 \
BATCH_SIZE=8 \
MAX_STEPS=100001 \
SAVE_FREQ=5000 \
DECAY_START=50000 \
bash scripts/train_dfm_maniskill.sh --resume
```

### Configuration

| Param | Value | Notes |
|-------|-------|-------|
| Total batch | 32 | Same as paper config (4 GPU × batch 8) |
| Action chunk | 8 | Franka Panda, matches LIBERO |
| Max steps | 100k | Similar dataset size to SimplerEnv |
| LR | 5e-4 | Same as paper |
| LR decay | step 50k → 5e-5 | 50% of training |
| Save freq | 5000 (~2-3h on 4×A100) | 20 checkpoints total |

### Expected training time

~2 days on 4×A100-80GB at 100k steps (similar to SimplerEnv).

## 5. Evaluation

Use `vla_eval_adapter/vla_dfm_server.py` + the harness `maniskill2_eval.yaml`:

```bash
# Start server (terminal 1)
python experiments/robot/vla_eval_adapter/vla_dfm_server.py \
    --pretrained_checkpoint $HOME/checkpoints/dfm-vla-maniskill-100k/<run_dir>/<step>_chkpt \
    --unnorm_key maniskill_dataset_converted_externally_to_rlds \
    --use_discrete_flow_matching \
    --dfm_decode_mode ctmc \
    --num_images_in_input 2 \
    --use_proprio \
    --center_crop \
    --chunk_size 8 \
    --port 8000

# Run eval (terminal 2, in vla-evaluation-harness repo)
vla-eval run --config configs/maniskill2_eval.yaml --server-url ws://localhost:8000
```

Eval covers 5 tasks × 50 episodes = 250 total trials.

## 6. Implementation Notes

**Action tokenizer bins**: re-fit on ManiSkill2 actions automatically. Verify
`dataset_statistics.json` is written to `RUN_ROOT` after first epoch.

**Two cameras**: ManiSkill2 RLDS has primary + wrist (`image`, `wrist_image`).
Train with `NUM_IMAGES_IN_INPUT=2`. Eval harness should provide both at runtime.

**No FiLM**: Default since ManiSkill2 task descriptions are simple ("pick the
red cube"). Can enable with `USE_FILM=True` if needed.
