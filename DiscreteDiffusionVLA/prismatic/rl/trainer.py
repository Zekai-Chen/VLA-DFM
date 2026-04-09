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

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from prismatic.rl.ratio_network import RatioNetwork, ppo_ratio_loss
from prismatic.rl.rollout_buffer import RolloutBuffer, Transition

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
        # Mean-pool over language tokens
        denom = lang_mask.float().sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # (B, 1)
        state = (hidden_states * lang_mask.unsqueeze(-1).float()).sum(dim=1) / denom  # (B, D)
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

    x1_idx = x1_safe - action_begin
    xt_idx = torch.where(xt_safe == mask_id, torch.full_like(xt_safe, K - 1), xt_safe - action_begin)

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
        model_cfg = vla_model.config

        # Infer LLM hidden dim and action vocab from model config
        llm_hidden_dim = getattr(model_cfg, "text_config", model_cfg).hidden_size
        n_action_bins = int(getattr(model_cfg, "n_action_bins", 256))
        action_begin, action_end, _ = self.vla._action_vocab_range()
        self.action_begin = action_begin
        self.action_end = action_end
        self.mask_token_id = int(self.vla.mask_token_id)
        n_action_tokens = self.vla.config.get("NUM_ACTIONS_CHUNK", 56)  # fallback

        # Ratio network (β)
        self.ratio_net = RatioNetwork(
            llm_hidden_dim=llm_hidden_dim,
            n_action_tokens=n_action_tokens,
            action_vocab=n_action_bins,
            hidden_dim=self.cfg.ratio_hidden_dim,
            num_heads=self.cfg.ratio_num_heads,
            num_layers=self.cfg.ratio_num_layers,
        ).to(self.device)

        # Value network
        self.value_net = ValueNetwork(
            llm_hidden_dim=llm_hidden_dim,
            hidden_dim=512,
        ).to(self.device)

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
            # --- Run inference (MaskGIT decoding) ---
            input_ids = obs["input_ids"].to(self.device)
            attention_mask = obs["attention_mask"].to(self.device)
            pixel_values = obs["pixel_values"].to(self.device)
            labels = obs["labels"].to(self.device)
            action_pos_mask = obs["action_positions_mask"].to(self.device)

            # Generate action tokens + get hidden states for value/ratio nets
            vla_out = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=True,
                dfm_schedule=self.cfg.dfm_schedule,
                dfm_time_eps=self.cfg.dfm_time_eps,
                dfm_t_min=self.cfg.dfm_t_min,
                dfm_t_max=self.cfg.dfm_t_max,
                dfm_loss_mode="generalized_kl",
                dfm_weight_clip=self.cfg.dfm_weight_clip,
                dfm_t_bias_alpha=self.cfg.dfm_t_bias_alpha,
            )
            hidden_states = vla_out.hidden_states[-1]  # (B, L, D)

            # Value estimate
            lang_mask = attention_mask.bool() & (~action_pos_mask)
            value = self.value_net(hidden_states, lang_mask)  # (B,)

            # Predict actions (MaskGIT inference)
            action_cont, action_token_ids = self.vla.predict_action(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                maskgit_num_steps=self.cfg.maskgit_num_steps,
                maskgit_schedule=self.cfg.maskgit_schedule,
            )

            # Log-probability estimate via MC sampling of DFM time
            log_prob = self._estimate_log_prob(
                input_ids, attention_mask, pixel_values, labels, action_pos_mask
            )

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
        last_vla_out = self.vla(
            input_ids=last_obs_input,
            attention_mask=obs["attention_mask"].to(self.device),
            pixel_values=obs["pixel_values"].to(self.device),
            output_hidden_states=True,
        )
        last_hs = last_vla_out.hidden_states[-1]
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

        with torch.no_grad():
            vla_out = self.vla(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=batch["pixel_values"].to(self.device),
                output_hidden_states=True,
            )
            hidden_states = vla_out.hidden_states[-1].detach()

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
        pixel_values = batch["pixel_values"].to(self.device)
        labels = batch["labels"].to(self.device)
        action_pos_mask = batch["action_positions_mask"].to(self.device)
        action_token_ids = batch["action_token_ids"].to(self.device)

        # Get importance weights from (now updated) ratio network — detached
        with torch.no_grad():
            vla_out_eval = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
            hs = vla_out_eval.hidden_states[-1]
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

            # Forward pass
            vla_out = self.vla(
                input_ids=None,
                inputs_embeds=masked_embeddings,
                attention_mask=attention_mask,
                labels=masked_labels,
                output_hidden_states=False,
            )

            # Per-sample DFM loss (B,) then reweight by r_β
            shift_logits = vla_out.logits[:, :-1, :]
            shift_labels = masked_labels[:, 1:]
            shift_x1 = labels[:, 1:]
            shift_act_mask = action_pos_mask[:, 1:]

            per_sample_loss = dfm_gkl_loss_per_sample(
                shift_logits=shift_logits,
                x1=shift_x1,
                xt=shift_labels,
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

        with torch.no_grad():
            vla_out = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=batch["pixel_values"].to(self.device),
                output_hidden_states=True,
            )
            hs = vla_out.hidden_states[-1]

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
                "vla_state": {k: v for k, v in self.vla.state_dict().items() if "lora" in k},
                "ratio_net_state": self.ratio_net.state_dict(),
                "value_net_state": self.value_net.state_dict(),
                "vla_optimizer": self.vla_optimizer.state_dict(),
                "ratio_optimizer": self.ratio_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)

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

    @torch.no_grad()
    def _estimate_log_prob(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        labels,
        action_pos_mask,
        n_mc: int = 4,
    ) -> torch.Tensor:
        """
        Monte-Carlo estimate of log π_θ(a|s) via:
            E_{t~U[0,1], x_t~q_t(·|a)} [log p^θ_{1|t}(a|x_t, s)]
        """
        B = input_ids.shape[0]
        log_prob_accum = torch.zeros(B, device=self.device)

        for _ in range(n_mc):
            embeddings = self.vla.get_input_embeddings()(input_ids)
            _, masked_emb, masked_labels, _, kappa_t, kdot_t, _ = self.vla.apply_mask_flow_matching(
                input_ids=input_ids,
                input_embeddings=embeddings,
                labels=labels,
                loss_mask_full=action_pos_mask,
                mask_token_id=self.mask_token_id,
                schedule=self.cfg.dfm_schedule,
                time_eps=self.cfg.dfm_time_eps,
            )
            vla_out = self.vla(
                inputs_embeds=masked_emb,
                attention_mask=attention_mask,
                labels=masked_labels,
                output_hidden_states=False,
            )
            shift_logits = vla_out.logits[:, :-1, :]
            shift_labels = masked_labels[:, 1:]
            shift_x1 = labels[:, 1:]
            shift_act_mask = action_pos_mask[:, 1:]

            per_sample = dfm_gkl_loss_per_sample(
                shift_logits=shift_logits,
                x1=shift_x1,
                xt=shift_labels,
                action_mask=shift_act_mask,
                kappa_t=kappa_t,
                kdot_t=kdot_t,
                action_begin=self.action_begin,
                action_end=self.action_end,
                mask_id=self.mask_token_id,
                weight_clip=self.cfg.dfm_weight_clip,
            )
            log_prob_accum -= per_sample  # loss = -log_prob

        return log_prob_accum / n_mc
