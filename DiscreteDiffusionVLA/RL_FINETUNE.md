# RL Fine-Tuning for Discrete Flow Matching VLA

This document describes the RL fine-tuning extension added on branch `rl-dfm-finetune`.

It implements **Algorithm 1** from:

> **RL Fine-Tuning for Discrete Flow Matching Model** (2026, under review)  
> Maojiang Su, Han Liu et al., Northwestern University

on top of the existing [Discrete Diffusion VLA](https://arxiv.org/abs/2508.20072) codebase.

---

## Overview

The supervised DFM training in `vla-scripts/finetune.py` minimises a masked prediction loss.
RL fine-tuning replaces this with an **importance-weighted DFM objective**, steering the policy
toward high-reward actions while preserving the learned distribution via a KL constraint.

```
Algorithm 1: Discrete Flow Matching RL Fine-tuning
───────────────────────────────────────────────────
Require: Env MDP M; pretrained DFM-VLA u_ϕ; advantage estimator Â;
         PPO clip ε; ratio net r_β; K iterations; regulariser λ.

θ₁ = ϕ

for k = 1 … K:
  1. Collect rollouts D ← rollout(π_k, M)
  2. Estimate advantages  A^{π_k} ← Â(D)   [GAE, γ=0.99, λ_GAE=0.95]
  3. PPO update for ratio network β:
       β_{k+1} ← argmax_β E_{(s,a)∼D}[
           min(r_β A, clip(r_β,1-ε,1+ε)A)
           − λ(E_{a'∼π_old}[r_β(s,a')]−1)²
       ]
  4. Weighted DFM update:
       For each (s,a)∈D  sample  t∼U[0,1],  x_t∼q_t(·|a)
       θ_{k+1} ← argmin_θ E[ r_{β_{k+1}}(s,a) · L_DFM(θ; x_t, a, s, t) ]

Output: policy π_K
```

### Why a separate ratio network?

Instead of computing `π_θ(a|s)/π_ref(a|s)` analytically (expensive for long token sequences),
we train a small transformer `r_β` to approximate the ratio, following the same PPO-clip trick.
This keeps the DFM model gradient clean and avoids storing a frozen copy of the base model.

---

## New files on this branch

```
DiscreteDiffusionVLA/
├── prismatic/rl/
│   ├── __init__.py
│   ├── ratio_network.py     # r_β: LLM-hidden-state + action-token → scalar ratio
│   ├── rollout_buffer.py    # Transition storage + GAE advantage computation
│   └── trainer.py           # Algorithm 1 training loop (DFMRLTrainer)
└── vla-scripts/
    └── rl_finetune.py       # Entry-point script
```

### `prismatic/rl/ratio_network.py`

| Symbol | Implementation |
|--------|----------------|
| `r_β(s, a)` | Cross-attention transformer over action tokens, conditioned on LLM hidden states at action positions. Scalar output via softplus (always > 0). |
| `ppo_ratio_loss` | PPO clipped surrogate + `(E[r_β]−1)²` constraint. |

### `prismatic/rl/rollout_buffer.py`

Stores `Transition` objects (obs tensors, action tokens, reward, done, value, log-prob).
`compute_advantages()` runs backward-pass GAE.

### `prismatic/rl/trainer.py`

`DFMRLTrainer` wires everything together:

| Method | Algorithm step |
|--------|----------------|
| `collect_rollouts()` | Step 1-2: run MaskGIT inference, store to buffer, compute GAE |
| `update_ratio_network()` | Step 3: PPO update for β |
| `update_policy()` | Step 4: weighted DFM loss using existing `apply_mask_flow_matching` + `_dfm_generalized_kl_loss` per-sample variant |
| `update_value_network()` | MSE update for V(s) (used in GAE) |

The weighted DFM update re-uses VLA-DFM's existing infrastructure:
- `apply_mask_flow_matching()` → samples `(t, x_t)` pair
- `dfm_gkl_loss_per_sample()` (new helper in `trainer.py`) → per-sample loss (B,)
- Weighted mean: `(r_β · loss).mean()`

---

## Quick start

### 1. Install dependencies (same environment as supervised training)

```bash
cd DiscreteDiffusionVLA
pip install -e .
```

### 2. Obtain a supervised DFM checkpoint

Run the supervised fine-tuning first, or download a pretrained checkpoint:

```bash
python vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --dataset_name libero_spatial \
    --use_discrete_flow_matching True \
    --dfm_schedule cosine
```

### 3. RL fine-tuning

```bash
python vla-scripts/rl_finetune.py \
    --vla_path runs/libero_spatial+dfm+..../checkpoint-50000 \
    --dataset_name libero_spatial \
    --num_iterations 100 \
    --rollout_steps 50 \
    --dfm_lr 5e-6 \
    --ppo_clip_eps 0.2 \
    --lambda_constraint 1.0
```

With WandB:

```bash
python vla-scripts/rl_finetune.py \
    --vla_path <ckpt_path> \
    --use_wandb True \
    --wandb_project my-rl-project
```

Resume from RL checkpoint:

```bash
python vla-scripts/rl_finetune.py \
    --vla_path <ckpt_path> \
    --resume runs/rl_dfm/rl_ckpt_iter_0025.pt
```

---

## Hyperparameter guide

| Argument | Default | Notes |
|----------|---------|-------|
| `num_iterations` | 100 | Outer RL loop iterations K |
| `rollout_steps` | 50 | Env steps collected per iteration |
| `ppo_clip_eps` | 0.2 | ε — PPO trust-region |
| `lambda_constraint` | 1.0 | λ — E[r_β]=1 penalty strength |
| `ppo_epochs` | 4 | Inner gradient steps for β |
| `ppo_lr` | 3e-4 | Learning rate for ratio network |
| `dfm_epochs` | 4 | Inner gradient steps for θ |
| `dfm_lr` | 5e-6 | Fine-tuning LR (much lower than supervised) |
| `dfm_grad_clip` | 1.0 | Gradient clipping norm |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE λ |
| `dfm_schedule` | cosine | Same as supervised training |
| `dfm_weight_clip` | 20.0 | Clip kappa_dot/(1−kappa) in GKL loss |
| `maskgit_num_steps` | 12 | MaskGIT iterations at inference |

---

## Connecting a real environment

Replace `DummyLiberoEnv` in `vla-scripts/rl_finetune.py` with your actual environment.
The required API:

```python
class MyEnv:
    def reset(self) -> dict:
        # Returns observation dict with keys:
        #   input_ids          (B, L)  – tokenised prompt + masked actions
        #   attention_mask     (B, L)
        #   pixel_values       (B, C, H, W)
        #   labels             (B, L)  – IGNORE_INDEX=-100 at non-action positions
        #   action_positions_mask (B, L)  – True at action token positions
        ...

    def step(self, action_cont: torch.Tensor):
        # action_cont: (B, H, D) continuous actions
        # Returns: obs, reward (B,), done (B,), info dict
        ...
```

For LIBERO, adapt `experiments/robot/libero/run_libero_eval.py` to this interface.
For real robot (DROID/ALOHA), adapt `experiments/robot/aloha/run_aloha_eval.py`.

---

## Theory: Schrödinger Bridge interpretation

The paper shows that the reweighted DFM objective admits a
**KL-regularised Stochastic Optimal Control** interpretation via the Schrödinger Bridge:

```
min_θ  KL(π_θ || π_ref)  s.t.  E_{π_θ}[R] is maximised
```

The importance weight `r_β(s,a) = π_θ(a|s)/π_ref(a|s)` is the Radon–Nikodym derivative
connecting the controlled process to the reference DFM path measure.
Optimising the weighted DFM loss is equivalent to minimising this KL-regularised objective.

---

## Citation

If you use this RL extension, please cite both papers:

```bibtex
@article{liang2025discrete,
  title   = {Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding
             in Vision-Language-Action Policies},
  author  = {Liang, Zhixuan and others},
  journal = {arXiv:2508.20072},
  year    = {2025}
}

@article{su2026rl,
  title   = {RL Fine-Tuning for Discrete Flow Matching Model},
  author  = {Su, Maojiang and Liu, Han and others},
  journal = {arXiv preprint},
  year    = {2026}
}
```
