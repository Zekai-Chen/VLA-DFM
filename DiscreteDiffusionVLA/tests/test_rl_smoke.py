"""
Smoke test for RL fine-tuning trainer (Algorithm 1).

Exercises the full training loop end-to-end with a tiny mock model and
dummy environment, verifying that all 4 Algorithm 1 steps execute without
crashing:
  1. collect_rollouts
  2. update_ratio_network
  3. update_policy
  4. update_value_network
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Mock VLA model — tiny, CPU-friendly, exposes all APIs the trainer calls.
# ---------------------------------------------------------------------------

N_ACTION_BINS = 32
VOCAB_SIZE = 256
HIDDEN_DIM = 64
SEQ_LEN = 80
N_ACT_TOKENS = 56  # 8 * 7 (LIBERO default)
MASK_TOKEN_ID = VOCAB_SIZE - 1
ACTION_BEGIN = VOCAB_SIZE - N_ACTION_BINS - 1
ACTION_END = VOCAB_SIZE - 1


class MockVLA(nn.Module):
    """Minimal mock VLA that satisfies `DFMRLTrainer` API requirements."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=HIDDEN_DIM),
            n_action_bins=N_ACTION_BINS,
            mask_token_id=MASK_TOKEN_ID,
            pad_token_id=0,
            action_vocab_anchor="pad",
        )
        self.mask_token_id = MASK_TOKEN_ID
        self.use_discrete_flow_matching = True  # toggled by _disable_dfm
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.body = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    # --- APIs required by DFMRLTrainer ---

    def _action_vocab_range(self):
        return ACTION_BEGIN, ACTION_END, N_ACTION_BINS

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        inputs_embeds=None,
        output_hidden_states=False,
        **kwargs,
    ):
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            x = self.embed(input_ids)
        else:
            raise ValueError("Need input_ids or inputs_embeds")
        h = torch.tanh(self.body(x))
        logits = self.lm_head(h)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(h,) if output_hidden_states else None,
        )

    def apply_mask_flow_matching(
        self,
        input_ids,
        input_embeddings,
        labels,
        loss_mask_full,
        mask_token_id,
        schedule="cosine",
        time_eps=1e-3,
        t_min=0.0,
        t_max=1.0,
        t_bias_alpha=1.0,
    ):
        B, L = input_ids.shape
        device = input_ids.device
        t = torch.rand(B, device=device).clamp(0.01, 0.99)
        kappa_t = t
        kdot_t = torch.ones_like(t)
        rand = torch.rand(B, L, device=device)
        masked_mask = (rand > kappa_t.view(B, 1)) & loss_mask_full
        masked_input_ids = torch.where(masked_mask, mask_token_id, input_ids)
        ignore = torch.full_like(labels, -100)
        masked_labels = torch.where(masked_mask, labels, ignore)
        masked_emb = self.embed(masked_input_ids.clamp(0, VOCAB_SIZE - 1))
        loss_mask = masked_mask.float()
        return masked_input_ids, masked_emb, masked_labels, loss_mask, kappa_t, kdot_t, t


# ---------------------------------------------------------------------------
# Dummy environment (simplified from rl_finetune.py DummyLiberoEnv)
# ---------------------------------------------------------------------------

class MockEnv:
    def __init__(self, batch_size=2):
        self.B = batch_size
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return self._obs()

    def step(self, action_cont):
        self.step_count += 1
        reward = torch.zeros(self.B)
        done = torch.tensor([self.step_count >= 5] * self.B)
        if done.any():
            reward = (torch.rand(self.B) > 0.5).float()
        info = {"success": reward > 0.5}
        return self._obs(), reward, done, info

    def _obs(self):
        B, L = self.B, SEQ_LEN
        input_ids = torch.randint(0, VOCAB_SIZE, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        pixel_values = torch.rand(B, 3, 32, 32)
        # Action token positions must contain valid action token ids
        action_toks = torch.randint(ACTION_BEGIN, ACTION_END, (B, N_ACT_TOKENS))
        input_ids[:, -N_ACT_TOKENS:] = action_toks
        labels = input_ids.clone()
        labels[:, :-N_ACT_TOKENS] = -100
        action_pos_mask = torch.zeros(B, L, dtype=torch.bool)
        action_pos_mask[:, -N_ACT_TOKENS:] = True
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            action_positions_mask=action_pos_mask,
        )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_rl_trainer_smoke():
    """Full Algorithm 1 loop runs 2 iterations without error."""
    # Ensure LIBERO constants (defaults).
    from prismatic.rl.trainer import DFMRLTrainer, RLFinetuneConfig

    mock_vla = MockVLA()

    cfg = RLFinetuneConfig(
        num_iterations=2,
        rollout_steps=3,
        batch_size=2,
        ppo_epochs=1,
        dfm_epochs=1,
        log_interval=1,
        save_interval=100,  # skip saving during test
        use_wandb=False,
    )

    trainer = DFMRLTrainer(
        vla_model=mock_vla,
        env_fn=lambda: MockEnv(batch_size=2),
        cfg=cfg,
        device="cpu",
    )

    # Run Algorithm 1
    trainer.train()

    print("SMOKE TEST PASSED — Algorithm 1 ran 2 iterations successfully.")


if __name__ == "__main__":
    test_rl_trainer_smoke()
