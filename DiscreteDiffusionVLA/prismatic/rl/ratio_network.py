"""
ratio_network.py

Learned importance-ratio network r_β(s, a) for DFM RL fine-tuning.

Algorithm 1 (paper), step 5-6:
    β_{k+1} ← argmax_β E_{(s,a)~D} [
        min(r_β(s,a) A, clip(r_β(s,a), 1-ε, 1+ε) A)
        - λ (E_{a'~π_old}[r_β(s,a')] - 1)²
    ]

The ratio network takes:
  - LLM hidden states (last layer) at action token positions → state summary
  - Clean action token ids → action encoding

and produces a scalar r_β(s, a) > 0 representing the likelihood ratio
between the current policy and the reference policy.

Integration note:
  Call `model.forward(..., output_hidden_states=True)` and pass the last
  hidden state to RatioNetwork.forward() together with the action token ids.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RatioNetwork(nn.Module):
    """
    Importance-ratio network r_β(s, a).

    Architecture:
      1. Project LLM hidden states at action positions → action-aware state repr.
      2. Embed clean action token ids, add learnable positional bias.
      3. Cross-attend action embeddings to state representation.
      4. Global mean-pool → MLP → softplus scalar.

    Args:
        llm_hidden_dim:  Hidden dimension of the LLM (e.g. 4096 for Llama-2-7B).
        n_action_tokens: Total action token positions = NUM_ACTIONS_CHUNK * ACTION_DIM.
        action_vocab:    Number of clean action bins (e.g. 256).
        hidden_dim:      Internal dimension of the ratio network.
        num_heads:       Attention heads.
        num_layers:      Transformer encoder depth.
    """

    def __init__(
        self,
        llm_hidden_dim: int = 4096,
        n_action_tokens: int = 56,       # 8 steps × 7 dims
        action_vocab: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()
        self.n_action_tokens = n_action_tokens
        self.action_vocab = action_vocab
        # mask token id = action_vocab (just outside clean range)
        self.mask_token_id_offset = action_vocab

        # State: project LLM hidden states at action positions to hidden_dim
        self.state_proj = nn.Linear(llm_hidden_dim, hidden_dim)

        # Action: embed clean token ids (include mask token)
        self.action_embed = nn.Embedding(action_vocab + 1, hidden_dim)
        self.pos_embed = nn.Embedding(n_action_tokens, hidden_dim)

        # Cross-attend action repr → state context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            kdim=hidden_dim,
            vdim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Self-attend within action tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Scalar head: pool → MLP → softplus
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_token_ids: torch.Tensor,
        action_positions_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance ratio r_β(s, a).

        Args:
            hidden_states:        LLM last-layer hidden states (B, L, llm_hidden_dim).
            action_token_ids:     Clean action token ids (B, N_act), values in
                                  [action_begin, action_end) mapped to [0, action_vocab).
            action_positions_mask: Boolean mask (B, L); True at action token positions.

        Returns:
            ratio: Positive scalar per sample (B,).
        """
        B, L, _ = hidden_states.shape
        N = self.n_action_tokens
        device = hidden_states.device

        # --- State context: hidden states at action positions ---
        # Gather the action-position hidden states (B, N_act, llm_hidden_dim)
        # We take the first N_act True positions per sample.
        # If the sequence is padded unevenly, fall back to mean-pooling.
        action_hs = self._gather_action_hidden(hidden_states, action_positions_mask, N)
        ctx = self.state_proj(action_hs)  # (B, N, hidden_dim)

        # --- Action encoding ---
        pos = torch.arange(N, device=device)
        # Clamp token ids into [0, action_vocab] range (action_vocab = mask)
        a_ids = action_token_ids.clamp(0, self.action_vocab)  # (B, N)
        x = self.action_embed(a_ids) + self.pos_embed(pos).unsqueeze(0)  # (B, N, D)

        # Cross-attend to state context
        x_ca, _ = self.cross_attn(query=x, key=ctx, value=ctx)
        x = self.cross_norm(x + x_ca)

        # Self-attend within action tokens
        x = self.self_attn(x)  # (B, N, D)

        # Pool and project to scalar
        x_pool = self.out_norm(x.mean(dim=1))     # (B, D)
        logit = self.out_mlp(x_pool).squeeze(-1)  # (B,)
        return F.softplus(logit)                  # (B,) > 0

    @staticmethod
    def _gather_action_hidden(
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """
        Extract exactly n hidden states at True positions of mask per sample.
        Falls back to zero-padding if fewer than n positions are available.
        """
        B, L, D = hidden_states.shape
        out = torch.zeros(B, n, D, device=hidden_states.device, dtype=hidden_states.dtype)
        for i in range(B):
            idx = mask[i].nonzero(as_tuple=False).squeeze(-1)  # positions of action tokens
            k = min(len(idx), n)
            if k > 0:
                out[i, :k] = hidden_states[i, idx[:k]]
        return out


# ---------------------------------------------------------------------------
# PPO loss for the ratio network
# ---------------------------------------------------------------------------

def ppo_ratio_loss(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
    lambda_constraint: float = 1.0,
    old_ratio_samples: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    PPO clipped surrogate loss for ratio network β (Algorithm 1, step 6).

    Objective (maximise):
        L_PPO(β) = E[min(r_β A, clip(r_β, 1-ε, 1+ε) A)]
                   - λ (E_{a'~π_old}[r_β(s,a')] - 1)²

    Args:
        ratio:              Current r_β(s, a) estimates (B,).
        advantage:          Estimated advantages A^{π_k}(s, a) (B,).
        clip_eps:           PPO clip parameter ε.
        lambda_constraint:  Regularizer weight λ.
        old_ratio_samples:  r_β evaluated on π_old samples for constraint (B,).

    Returns:
        dict with 'loss' (scalar, negated for minimisation), 'surr_loss', 'constraint'.
    """
    adv = advantage.detach()

    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    surr_obj = torch.min(surr1, surr2).mean()

    if old_ratio_samples is not None:
        constraint = lambda_constraint * (old_ratio_samples.mean() - 1.0) ** 2
    else:
        constraint = lambda_constraint * (ratio.mean() - 1.0) ** 2

    total_loss = -surr_obj + constraint
    return {"loss": total_loss, "surr_loss": -surr_obj, "constraint": constraint}
