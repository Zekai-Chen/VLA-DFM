"""
trainer.py

DFM RL fine-tuning trainer — implements Algorithm 1 from:
  "RL Fine-Tuning for Discrete Flow Matching Model" (2026)

Algorithm 1: Discrete Flow Matching RL Fine-tuning
────────────────────────────────────────────────────
Require: Environment MDP M; Pre-trained DFM-VLA model u_ϕ;
         advantage estimator Â; PPO clip ε; ratio net r_β;
         iterations K; regulariser λ.

Initialize: θ_1 = ϕ;  β_1
for k = 1 … K:
  1. Collect rollout batch D using π_k (induced by u_{θ_k})
  2. Estimate advantages A^{π_k} ← Â(D)  [GAE]
  3. PPO update for ratio network β_{k+1}:
       argmax_β E_{(s,a)~D}[min(r_β A, clip(r_β,1-ε,1+ε)A)]
               - λ(E_{a'~π_old}[r_β(s,a')]-1)²
  4. Weighted DFM update for θ_{k+1}:
       For each (s,a)∈D sample t~U[0,1], x_t~q_t(·|a)
       argmin_θ E[r_{β_{k+1}}(s,a) · L_DFM(θ; x_t, a, s, t)]
Output: π_K

Integration with VLA-DFM:
  - Uses `OpenVLAForActionPrediction.apply_mask_flow_matching()` for
    the x_t ~ q_t(·|a) sampling step.
  - Uses `_dfm_generalized_kl_loss_per_sample()` (added in this PR) for
    the per-sample DFM loss that gets reweighted by r_β.
  - Uses MaskGIT / CTMC decoding from `prismatic.discrete_flow` at inference.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from prismatic.rl.ratio_network import RatioNetwork, ppo_ratio_loss
from prismatic.rl.rollout_buffer import RolloutBuffer, Transition

try:
    # Real package import (used at training time).
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
except Exception:  # pragma: no cover - smoke tests with mock model
    ACTION_DIM, NUM_ACTIONS_CHUNK = 7, 8


@contextlib.contextmanager
def _disable_dfm(model):
    """Temporarily turn off use_discrete_flow_matching so that forward()
    does not apply internal DFM masking (avoids double-masking and crashes
    when labels=None).

    Handles PEFT-wrapped models by setting the flag on ALL layers that
    have it (PeftModel, LoraModel, and the underlying OpenVLA model)."""
    # Collect all objects in the wrapper chain that have the flag
    targets = []
    for path in [
        [],                          # model itself
        ["base_model"],              # LoraModel
        ["model"],                   # OpenVLA (via PeftModel.model)
        ["base_model", "model"],     # OpenVLA (via PeftModel.base_model.model)
        ["module"],                  # DDP wrapper
        ["module", "base_model", "model"],
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "use_discrete_flow_matching"):
                targets.append(obj)
        except AttributeError:
            pass

    # Save and disable
    saved = [(t, t.use_discrete_flow_matching) for t in targets]
    for t in targets:
        t.use_discrete_flow_matching = False
    try:
        yield
    finally:
        for t, flag in saved:
            t.use_discrete_flow_matching = flag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RLFinetuneConfig:
    # ── RL outer loop ──────────────────────────────────────────────────────
    num_iterations: int = 100         # K
    rollout_steps: int = 50           # env steps collected per iteration
    batch_size: int = 8               # samples per gradient update

    # ── PPO / ratio network ────────────────────────────────────────────────
    ppo_clip_eps: float = 0.2         # ε
    lambda_constraint: float = 1.0   # λ
    ppo_epochs: int = 4
    ppo_lr: float = 3e-4
    ratio_hidden_dim: int = 512
    ratio_num_layers: int = 3
    ratio_num_heads: int = 8

    # ── DFM policy update ──────────────────────────────────────────────────
    dfm_epochs: int = 4
    dfm_lr: float = 5e-6             # lower than supervised fine-tuning LR
    dfm_grad_clip: float = 1.0

    # DFM schedule parameters (passed through to apply_mask_flow_matching)
    dfm_schedule: str = "cosine"
    dfm_time_eps: float = 1e-3
    dfm_t_min: float = 0.0
    dfm_t_max: float = 1.0
    dfm_t_bias_alpha: float = 1.0
    dfm_weight_clip: float = 20.0

    # ── Advantage estimation ───────────────────────────────────────────────
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ── Inference ─────────────────────────────────────────────────────────
    maskgit_num_steps: int = 12
    maskgit_schedule: str = "cosine"
    unnorm_key: Optional[str] = None   # dataset key for action un-normalisation

    # ── Logging / checkpointing ────────────────────────────────────────────
    log_interval: int = 10
    save_interval: int = 25
    save_dir: str = "checkpoints/rl_dfm"
    use_wandb: bool = False
    wandb_project: str = "dfm-rl"


# ---------------------------------------------------------------------------
# Value network (for advantage estimation)
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """
    Lightweight MLP value head V(s) that takes the mean-pooled
    LLM hidden states (at non-action positions) as state representation.
    """

    def __init__(self, llm_hidden_dim: int = 4096, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(llm_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: LLM last hidden states (B, L, D).
            lang_mask:     True at language/vision token positions (B, L).
        Returns:
            value: (B,)
        """
        # Mean-pool over language tokens (keep dtype of hidden_states)
        dt = hidden_states.dtype
        denom = lang_mask.to(dt).sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # (B, 1)
        state = (hidden_states * lang_mask.unsqueeze(-1).to(dt)).sum(dim=1) / denom  # (B, D)
        return self.net(state).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Per-sample DFM loss helper
# ---------------------------------------------------------------------------

def dfm_gkl_loss_per_sample(
    shift_logits: torch.Tensor,    # (B, T-1, V_full)
    x1: torch.Tensor,              # (B, T-1) clean action label ids
    xt: torch.Tensor,              # (B, T-1) masked input ids
    action_mask: torch.Tensor,     # (B, T-1) True at action positions
    kappa_t: torch.Tensor,         # (B,)
    kdot_t: torch.Tensor,          # (B,)
    action_begin: int,
    action_end: int,
    mask_id: int,
    weight_clip: float = 20.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized KL DFM loss returned **per sample** (B,) for RL reweighting.

    Identical math to `_dfm_generalized_kl_loss` in modeling_prismatic.py
    except we return per-sample means instead of the global mean.
    """
    B = shift_logits.shape[0]
    am = action_mask.bool()

    action_ids = torch.arange(action_begin, action_end, device=shift_logits.device)
    allowed_ids = torch.cat([action_ids, torch.tensor([mask_id], device=shift_logits.device)])
    K = allowed_ids.numel()

    logits = shift_logits.index_select(dim=-1, index=allowed_ids)

    x1_safe = torch.where(am, x1, torch.full_like(x1, action_begin))
    xt_safe = torch.where(am, xt, torch.full_like(xt, action_begin))

    x1_idx = (x1_safe - action_begin).clamp(0, K - 1)
    xt_idx = torch.where(xt_safe == mask_id, torch.full_like(xt_safe, K - 1), (xt_safe - action_begin).clamp(0, K - 2))

    log_p = torch.log_softmax(logits, dim=-1)
    log_p_x1 = log_p.gather(-1, x1_idx.unsqueeze(-1)).squeeze(-1)
    log_p_xt = log_p.gather(-1, xt_idx.unsqueeze(-1)).squeeze(-1)
    p_xt = torch.exp(log_p_xt)

    delta = (xt_safe == x1_safe).to(log_p.dtype)

    denom = (1.0 - kappa_t).clamp(min=eps)
    w = (kdot_t / denom).clamp(min=0.0, max=weight_clip).view(B, 1)

    loss_pos = -w * (p_xt - delta + (1.0 - delta) * log_p_x1)  # (B, T-1)

    # Per-sample mean (over action positions only)
    n_act = am.float().sum(dim=-1).clamp(min=1.0)  # (B,)
    per_sample_loss = (loss_pos * am).sum(dim=-1) / n_act  # (B,)
    return per_sample_loss


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class DFMRLTrainer:
    """
    Implements Algorithm 1: Discrete Flow Matching RL Fine-tuning.

    Wraps a pretrained `OpenVLAForActionPrediction` model and adds:
      - RatioNetwork r_β   (Algorithm 1, step 3)
      - ValueNetwork V(s)  (for GAE advantage estimation)
      - Weighted DFM loss  (Algorithm 1, step 4)

    Args:
        vla_model:  Pretrained OpenVLAForActionPrediction (LoRA fine-tuned DFM).
        env_fn:     Callable returning an environment with reset() / step() API.
        cfg:        RLFinetuneConfig.
        device:     Training device.
    """

    def __init__(
        self,
        vla_model,
        env_fn: Callable,
        cfg: RLFinetuneConfig | None = None,
        device: str | torch.device = "cuda",
    ):
        self.cfg = cfg or RLFinetuneConfig()
        self.device = torch.device(device) if isinstance(device, str) else device

        # Pre-trained DFM-VLA (θ)
        self.vla = vla_model.to(self.device)
        self._model_dtype = next(self.vla.parameters()).dtype
        model_cfg = vla_model.config

        # Infer LLM hidden dim and action vocab from model config
        llm_hidden_dim = getattr(model_cfg, "text_config", model_cfg).hidden_size
        n_action_bins = int(getattr(model_cfg, "n_action_bins", 256))
        action_begin, action_end, _ = self.vla._action_vocab_range()
        self.action_begin = action_begin
        self.action_end = action_end
        self.mask_token_id = int(self.vla.mask_token_id)
        # Action token grid: NUM_ACTIONS_CHUNK steps × ACTION_DIM dims (constants
        # come from prismatic.vla.constants; cannot be read from config because
        # `vla.config` is an HF PretrainedConfig, which has no `.get`).
        n_action_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM

        # Ratio network (β)
        self.ratio_net = RatioNetwork(
            llm_hidden_dim=llm_hidden_dim,
            n_action_tokens=n_action_tokens,
            action_vocab=n_action_bins,
            hidden_dim=self.cfg.ratio_hidden_dim,
            num_heads=self.cfg.ratio_num_heads,
            num_layers=self.cfg.ratio_num_layers,
        ).to(device=self.device, dtype=self._model_dtype)

        # Value network
        self.value_net = ValueNetwork(
            llm_hidden_dim=llm_hidden_dim,
            hidden_dim=512,
        ).to(device=self.device, dtype=self._model_dtype)

        # Environment
        self.env_fn = env_fn
        self.env = env_fn()

        # Optimizers
        self.vla_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.vla.parameters()),
            lr=self.cfg.dfm_lr,
            weight_decay=1e-4,
        )
        self.ratio_optimizer = optim.AdamW(
            self.ratio_net.parameters(),
            lr=self.cfg.ppo_lr,
            weight_decay=1e-4,
        )
        self.value_optimizer = optim.AdamW(
            self.value_net.parameters(),
            lr=1e-4,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            capacity=self.cfg.rollout_steps * 4,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        self.global_step = 0

        if self.cfg.use_wandb:
            import wandb
            wandb.init(project=self.cfg.wandb_project, config=vars(self.cfg))

    # ------------------------------------------------------------------
    # Step 1: Collect rollouts
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollouts(self) -> Dict[str, float]:
        """
        Run current policy π_k in the environment, store transitions.
        Algorithm 1, step 3.
        """
        self.vla.eval()
        self.value_net.eval()
        self.buffer.clear()

        obs = self.env.reset()
        total_reward = 0.0
        success_count = 0
        num_episodes = 0

        for _ in range(self.cfg.rollout_steps):
            # --- Get hidden states from a clean forward pass ---
            input_ids = obs["input_ids"].to(self.device)
            attention_mask = obs["attention_mask"].to(self.device)
            pixel_values = obs["pixel_values"].to(device=self.device, dtype=self._model_dtype)
            labels = obs["labels"].to(self.device)
            action_pos_mask = obs["action_positions_mask"].to(self.device)

            # Plain forward — we only need hidden states.  Disable DFM
            # masking so forward() does not apply internal DFM corruption
            # (which crashes when labels=None and adds unwanted masking).
            with _disable_dfm(self.vla):
                vla_out = self.vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                )
            hidden_states = self._strip_patches(
                vla_out.hidden_states[-1], input_ids.shape[1]
            )  # (B, L_text, D)

            # Value estimate
            lang_mask = attention_mask.bool() & (~action_pos_mask)
            value = self.value_net(hidden_states, lang_mask)  # (B,)

            # --- Predict actions using MaskGIT inference ---
            B = input_ids.shape[0]
            action_cont, action_token_ids = self._predict_actions(
                input_ids, attention_mask, pixel_values, labels, action_pos_mask
            )

            # Log-probability is not consumed by the current PPO loss (which
            # uses r_β directly), so store zeros to avoid extra forwards.
            log_prob = torch.zeros(B, device=self.device)

            # Step environment
            next_obs, reward, done, info = self.env.step(action_cont)

            # Store
            trans = Transition(
                input_ids=input_ids.cpu(),
                attention_mask=attention_mask.cpu(),
                pixel_values=pixel_values.cpu(),
                labels=labels.cpu(),
                action_positions_mask=action_pos_mask.cpu(),
                action_token_ids=action_token_ids.cpu(),
                action_cont=action_cont.cpu(),
                reward=reward,
                done=done,
                value=value.detach().cpu(),
                log_prob=log_prob.detach().cpu(),
            )
            self.buffer.push(trans)

            total_reward += reward.mean().item()
            if "success" in info:
                success_count += info["success"].float().mean().item()
                num_episodes += 1

            obs = next_obs
            if done.all():
                obs = self.env.reset()

        # Bootstrap last value
        last_obs_input = obs["input_ids"].to(self.device)
        with _disable_dfm(self.vla):
            last_vla_out = self.vla(
                input_ids=last_obs_input,
                attention_mask=obs["attention_mask"].to(self.device),
                pixel_values=obs["pixel_values"].to(device=self.device, dtype=self._model_dtype),
                labels=obs["labels"].to(self.device),
                output_hidden_states=True,
            )
        last_hs = self._strip_patches(
            last_vla_out.hidden_states[-1], last_obs_input.shape[1]
        )
        last_lang_mask = obs["attention_mask"].to(self.device).bool() & (~obs["action_positions_mask"].to(self.device))
        last_value = self.value_net(last_hs, last_lang_mask)

        # Compute advantages
        self.buffer.compute_advantages(last_value=last_value.detach().cpu())

        return {
            "rollout/mean_reward": total_reward / self.cfg.rollout_steps,
            "rollout/success_rate": success_count / max(num_episodes, 1),
        }

    # ------------------------------------------------------------------
    # Step 2: PPO update for ratio network β
    # ------------------------------------------------------------------

    def update_ratio_network(self, batch: dict) -> Dict[str, float]:
        """Algorithm 1, step 5-6."""
        self.ratio_net.train()

        advantages = batch["advantages"].to(self.device)
        action_token_ids = batch["action_token_ids"].to(self.device)
        action_pos_mask = batch["action_positions_mask"].to(self.device)

        with torch.no_grad(), _disable_dfm(self.vla):
            vla_out = self.vla(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=batch["pixel_values"].to(device=self.device, dtype=self._model_dtype),
                labels=batch["labels"].to(self.device),
                output_hidden_states=True,
            )
            hidden_states = self._strip_patches(
                vla_out.hidden_states[-1], batch["input_ids"].shape[1]
            ).detach()

        # Relative action token ids: map [action_begin, action_end) -> [0, n_bins)
        rel_action_ids = (action_token_ids - self.action_begin).clamp(0, self.ratio_net.action_vocab)

        total_loss = 0.0
        for _ in range(self.cfg.ppo_epochs):
            ratio = self.ratio_net(hidden_states, rel_action_ids, action_pos_mask)
            losses = ppo_ratio_loss(
                ratio=ratio,
                advantage=advantages,
                clip_eps=self.cfg.ppo_clip_eps,
                lambda_constraint=self.cfg.lambda_constraint,
            )
            self.ratio_optimizer.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(self.ratio_net.parameters(), 1.0)
            self.ratio_optimizer.step()
            total_loss += losses["loss"].item()

        return {
            "ratio_net/loss": total_loss / self.cfg.ppo_epochs,
            "ratio_net/mean_ratio": ratio.mean().item(),  # type: ignore[possibly-undefined]
        }

    # ------------------------------------------------------------------
    # Step 3: Weighted DFM update for θ
    # ------------------------------------------------------------------

    def update_policy(self, batch: dict) -> Dict[str, float]:
        """Algorithm 1, steps 7-10."""
        self.vla.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(device=self.device, dtype=self._model_dtype)
        labels = batch["labels"].to(self.device)
        action_pos_mask = batch["action_positions_mask"].to(self.device)
        action_token_ids = batch["action_token_ids"].to(self.device)

        # Get importance weights from (now updated) ratio network — detached
        with torch.no_grad(), _disable_dfm(self.vla):
            vla_out_eval = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=True,
            )
            hs = self._strip_patches(vla_out_eval.hidden_states[-1], input_ids.shape[1])
            rel_ids = (action_token_ids - self.action_begin).clamp(0, self.ratio_net.action_vocab)
            weights = self.ratio_net(hs, rel_ids, action_pos_mask).detach()  # (B,)

        total_loss = 0.0
        for _ in range(self.cfg.dfm_epochs):
            # Sample t ~ U[0,1] and x_t ~ q_t(·|a) via apply_mask_flow_matching
            embeddings = self.vla.get_input_embeddings()(input_ids)
            (
                masked_input_ids,
                masked_embeddings,
                masked_labels,
                loss_mask,
                kappa_t,
                kdot_t,
                t,
            ) = self.vla.apply_mask_flow_matching(
                input_ids=input_ids,
                input_embeddings=embeddings,
                labels=labels,
                loss_mask_full=action_pos_mask,
                mask_token_id=self.mask_token_id,
                schedule=self.cfg.dfm_schedule,
                time_eps=self.cfg.dfm_time_eps,
                t_min=self.cfg.dfm_t_min,
                t_max=self.cfg.dfm_t_max,
                t_bias_alpha=self.cfg.dfm_t_bias_alpha,
            )

            # Forward pass with DFM DISABLED — we already applied masking
            # externally via apply_mask_flow_matching above. Leaving DFM
            # enabled would cause the model to mask AGAIN internally.
            with _disable_dfm(self.vla):
                vla_out = self.vla(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=masked_labels,
                    output_hidden_states=False,
                )

            # Per-sample DFM loss (B,) then reweight by r_β
            # xt must be the *corrupted* input ids (mask_token_id at masked
            # positions, original token id elsewhere), NOT masked_labels
            # (which uses IGNORE_INDEX at unmasked positions).
            # Strip vision patch positions from logits to align with text masks.
            logits_text = self._strip_patches(vla_out.logits, masked_input_ids.shape[1])
            shift_logits = logits_text[:, :-1, :]
            shift_xt = masked_input_ids[:, 1:]
            shift_x1 = labels[:, 1:]
            shift_act_mask = action_pos_mask[:, 1:]

            per_sample_loss = dfm_gkl_loss_per_sample(
                shift_logits=shift_logits,
                x1=shift_x1,
                xt=shift_xt,
                action_mask=shift_act_mask,
                kappa_t=kappa_t,
                kdot_t=kdot_t,
                action_begin=self.action_begin,
                action_end=self.action_end,
                mask_id=self.mask_token_id,
                weight_clip=self.cfg.dfm_weight_clip,
            )  # (B,)

            loss = (weights * per_sample_loss).mean()

            self.vla_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.vla.parameters()),
                self.cfg.dfm_grad_clip,
            )
            self.vla_optimizer.step()
            total_loss += loss.item()

        return {
            "policy/weighted_dfm_loss": total_loss / self.cfg.dfm_epochs,
            "policy/mean_weight": weights.mean().item(),
        }

    # ------------------------------------------------------------------
    # Value network update
    # ------------------------------------------------------------------

    def update_value_network(self, batch: dict) -> Dict[str, float]:
        self.value_net.train()
        returns = batch["returns"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        action_pos_mask = batch["action_positions_mask"].to(self.device)

        with torch.no_grad(), _disable_dfm(self.vla):
            vla_out = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=batch["pixel_values"].to(device=self.device, dtype=self._model_dtype),
                labels=batch["labels"].to(self.device),
                output_hidden_states=True,
            )
            hs = self._strip_patches(vla_out.hidden_states[-1], input_ids.shape[1])

        lang_mask = attention_mask.bool() & (~action_pos_mask)
        values = self.value_net(hs, lang_mask)
        value_loss = ((values - returns.detach()) ** 2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return {"value_net/loss": value_loss.item()}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Execute Algorithm 1 for K iterations."""
        logger.info("Starting DFM RL fine-tuning for %d iterations", self.cfg.num_iterations)

        for iteration in range(1, self.cfg.num_iterations + 1):
            all_metrics: Dict[str, float] = {}

            # --- Steps 1-2 ---
            rollout_metrics = self.collect_rollouts()
            all_metrics.update(rollout_metrics)

            batch = self.buffer.get_batch(device=self.device)

            # --- Step 3: ratio network ---
            all_metrics.update(self.update_ratio_network(batch))

            # --- Step 4: DFM policy ---
            all_metrics.update(self.update_policy(batch))

            # --- Value network ---
            all_metrics.update(self.update_value_network(batch))

            self.global_step += 1

            if iteration % self.cfg.log_interval == 0:
                msg = "  ".join(f"{k}={v:.4f}" for k, v in all_metrics.items())
                logger.info("[iter %d] %s", iteration, msg)
                if self.cfg.use_wandb:
                    import wandb
                    wandb.log({"iteration": iteration, **all_metrics})

            if iteration % self.cfg.save_interval == 0:
                self._save_checkpoint(iteration)

        logger.info("RL fine-tuning complete.")

    def _save_checkpoint(self, iteration: int) -> None:
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        path = os.path.join(self.cfg.save_dir, f"rl_ckpt_iter_{iteration:04d}.pt")
        torch.save(
            {
                "iteration": iteration,
                # Save trainable params only (LoRA adapters or full model).
                "vla_state": {
                    n: self.vla.state_dict()[n]
                    for n, p in self.vla.named_parameters() if p.requires_grad
                },
                "ratio_net_state": self.ratio_net.state_dict(),
                "value_net_state": self.value_net.state_dict(),
                "vla_optimizer": self.vla_optimizer.state_dict(),
                "ratio_optimizer": self.ratio_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)

    @staticmethod
    def _strip_patches(hidden_states: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Extract text-only hidden states from multimodal output.

        The multimodal forward prepends vision patches after the BOS token:
            [BOS, patch_1..patch_N, text_token_2..text_token_L]
        This helper removes the patch positions so the output aligns with
        the original input_ids / masks of length ``seq_len``.
        """
        L_mm = hidden_states.shape[1]
        if L_mm == seq_len:
            return hidden_states  # no patches (mock model)
        n_patches = L_mm - seq_len
        return torch.cat([hidden_states[:, :1, :], hidden_states[:, n_patches + 1:, :]], dim=1)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        missing, unexpected = self.vla.load_state_dict(ckpt["vla_state"], strict=False)
        self.ratio_net.load_state_dict(ckpt["ratio_net_state"])
        self.value_net.load_state_dict(ckpt["value_net_state"])
        self.vla_optimizer.load_state_dict(ckpt["vla_optimizer"])
        self.ratio_optimizer.load_state_dict(ckpt["ratio_optimizer"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer"])
        logger.info("Loaded checkpoint from %s (iter %d)", path, ckpt["iteration"])
        return ckpt["iteration"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_action_token_ids(
        labels: torch.Tensor,
        action_pos_mask: torch.Tensor,
        n_act: int,
    ) -> torch.Tensor:
        """Extract the first `n_act` action token ids per sample from labels."""
        B = labels.shape[0]
        device = labels.device
        out = torch.zeros(B, n_act, device=device, dtype=labels.dtype)
        for i in range(B):
            idx = action_pos_mask[i].nonzero(as_tuple=False).squeeze(-1)
            k = min(len(idx), n_act)
            if k > 0:
                out[i, :k] = labels[i, idx[:k]]
        return out

    @torch.no_grad()
    def _predict_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
        action_pos_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict actions and recover discrete token ids.

        If the VLA has a real ``predict_action`` with DFM support, call it
        (batch_size=1 loop) and reverse-map the continuous actions back to
        token ids.  Otherwise (mock model or models without predict_action),
        fall back to extracting token ids from labels.

        Returns:
            action_cont:      (B, NUM_ACTIONS_CHUNK, ACTION_DIM) continuous.
            action_token_ids: (B, N_act) discrete token ids.
        """
        B = input_ids.shape[0]
        n_act = NUM_ACTIONS_CHUNK * ACTION_DIM
        has_predict = hasattr(self.vla, "predict_action") and hasattr(self.vla, "bin_centers")

        if has_predict:
            all_actions = []
            for i in range(B):
                single_ids = input_ids[i : i + 1]
                single_mask = attention_mask[i : i + 1]
                single_pv = pixel_values[i : i + 1]
                actions_np, _ = self.vla.predict_action(
                    input_ids=single_ids,
                    unnorm_key=self.cfg.unnorm_key,
                    attention_mask=single_mask,
                    pixel_values=single_pv,
                    use_discrete_flow_matching=True,
                    dfm_maskgit_num_steps=self.cfg.maskgit_num_steps,
                    dfm_maskgit_schedule=self.cfg.maskgit_schedule,
                    dfm_schedule=self.cfg.dfm_schedule,
                    dfm_time_eps=self.cfg.dfm_time_eps,
                )
                all_actions.append(actions_np)

            # Stack and convert back to token ids
            actions_np = np.stack(all_actions, axis=0)  # (B, CHUNK, DIM)
            # Reverse the token→action mapping:
            #   token → disc = action_end - token → clip(disc-1) → bin_centers[disc]
            # Inverse: bin_idx = argmin(|centers - val|), token = end - bin_idx - 1
            bin_centers = self.vla.bin_centers  # (n_bins,) numpy
            flat = actions_np.reshape(B, -1)  # (B, n_act)
            # Vectorised nearest-bin lookup
            diffs = np.abs(bin_centers[None, None, :] - flat[:, :, None])  # (B, n_act, n_bins)
            bin_idx = diffs.argmin(axis=-1)  # (B, n_act)
            token_ids_np = self.action_end - bin_idx - 1

            action_cont = torch.as_tensor(
                actions_np, dtype=torch.float32, device=self.device
            )
            action_token_ids = torch.as_tensor(
                token_ids_np, dtype=torch.long, device=self.device
            )
        else:
            # Fallback: extract from labels (for mock models / smoke tests).
            action_token_ids = self._extract_action_token_ids(labels, action_pos_mask, n_act)
            n_bins = self.action_end - self.action_begin
            bin_centers_t = torch.linspace(-1.0, 1.0, n_bins, device=self.device)
            rel = (action_token_ids - self.action_begin).clamp(0, n_bins - 1)
            action_cont = bin_centers_t[rel].view(B, NUM_ACTIONS_CHUNK, ACTION_DIM)

        return action_cont, action_token_ids
