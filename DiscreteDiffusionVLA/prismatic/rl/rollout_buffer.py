"""
rollout_buffer.py

Rollout buffer for DFM RL fine-tuning.

Stores (observation, action, reward, done) transitions collected by
running the current DFM-VLA policy in the environment.

After collection, computes Generalised Advantage Estimation (GAE):
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

Algorithm 1 (paper), step 3-4:
    "Collect rollouts D from M using current policy π_k"
    "Estimate advantage A^{π_k} ← Â(D)"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Transition:
    """Single environment step stored in the buffer."""
    # Observation
    input_ids: torch.Tensor          # (B, L) tokenised prompt + masked actions
    attention_mask: torch.Tensor     # (B, L)
    pixel_values: torch.Tensor       # (B, C, H, W) or (B, n_img, C, H, W)
    labels: torch.Tensor             # (B, L) ground-truth label ids
    action_positions_mask: torch.Tensor  # (B, L) True at action token positions

    # Action
    action_token_ids: torch.Tensor   # (B, N_act) predicted clean action tokens
    action_cont: torch.Tensor        # (B, H, D) decoded continuous actions

    # RL signals
    reward: torch.Tensor             # (B,)
    done: torch.Tensor               # (B,) bool
    value: torch.Tensor              # (B,) V(s_t) from value net
    log_prob: torch.Tensor           # (B,) log π_old(a|s)


class RolloutBuffer:
    """
    Fixed-capacity ring buffer for rollout transitions.

    Args:
        capacity:   Max transitions to store (older ones are dropped).
        gamma:      Discount factor γ.
        gae_lambda: GAE λ.
    """

    def __init__(self, capacity: int = 512, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._buf: List[Transition] = []
        self._adv: Optional[torch.Tensor] = None
        self._ret: Optional[torch.Tensor] = None

    def push(self, t: Transition) -> None:
        if len(self._buf) >= self.capacity:
            self._buf.pop(0)
        self._buf.append(t)
        self._adv = self._ret = None  # invalidate cache

    def clear(self) -> None:
        self._buf.clear()
        self._adv = self._ret = None

    def __len__(self) -> int:
        return len(self._buf)

    # ------------------------------------------------------------------

    def compute_advantages(self, last_value: Optional[torch.Tensor] = None) -> None:
        """
        Compute GAE advantages and Monte-Carlo returns in-place.

        Args:
            last_value: Bootstrap V(s_T) for the state after the last step (B,).
        """
        T = len(self._buf)
        if T == 0:
            return

        device = self._buf[0].reward.device
        B = self._buf[0].reward.shape[0]

        advantages = torch.zeros(T, B, device=device)
        returns = torch.zeros(T, B, device=device)

        gae = torch.zeros(B, device=device)
        next_val = last_value if last_value is not None else torch.zeros(B, device=device)

        for t in reversed(range(T)):
            trans = self._buf[t]
            not_done = 1.0 - trans.done.float().to(device)
            delta = trans.reward.to(device) + self.gamma * next_val * not_done - trans.value.to(device)
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae
            returns[t] = gae + trans.value.to(device)
            next_val = trans.value.to(device)

        flat = advantages.view(-1)
        advantages = (advantages - flat.mean()) / (flat.std() + 1e-8)

        self._adv = advantages   # (T, B)
        self._ret = returns      # (T, B)

    def get_batch(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Return all transitions as a flat dict (N = T × B along dim 0).

        Calls compute_advantages() if not already computed.
        """
        if self._adv is None:
            self.compute_advantages()

        def _cat(lst):
            return torch.cat(lst, dim=0)

        input_ids = _cat([t.input_ids for t in self._buf])
        attention_mask = _cat([t.attention_mask for t in self._buf])
        pixel_values = _cat([t.pixel_values for t in self._buf])
        labels = _cat([t.labels for t in self._buf])
        action_positions_mask = _cat([t.action_positions_mask for t in self._buf])
        action_token_ids = _cat([t.action_token_ids for t in self._buf])
        action_cont = _cat([t.action_cont for t in self._buf])
        rewards = _cat([t.reward for t in self._buf])
        log_probs = _cat([t.log_prob for t in self._buf])
        advantages = self._adv.view(-1)   # type: ignore[union-attr]
        returns = self._ret.view(-1)      # type: ignore[union-attr]

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            action_positions_mask=action_positions_mask,
            action_token_ids=action_token_ids,
            action_cont=action_cont,
            rewards=rewards,
            log_probs=log_probs,
            advantages=advantages,
            returns=returns,
        )

        if device is not None:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        return batch
