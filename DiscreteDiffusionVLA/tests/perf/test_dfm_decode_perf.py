import os
import pytest
import torch

from prismatic.discrete_flow.dfm_decode import dfm_decode


def _dummy_tokens_to_logits(cur: torch.LongTensor):
    B, L = cur.shape
    vocab_size = 128
    logits = torch.zeros(B, L, vocab_size)
    logits[..., 1] = 2.0
    hidden = torch.zeros(B, L, 8)
    return logits, hidden


@pytest.mark.perf
def test_dfm_decode_perf_smoke():
    if os.getenv("RUN_PERF") != "1":
        pytest.skip("Set RUN_PERF=1 to enable perf smoke test")

    init_ids = torch.zeros(8, 64, dtype=torch.long)
    final_ids, _, stats = dfm_decode(
        init_ids=init_ids,
        tokens_to_logits=_dummy_tokens_to_logits,
        mask_token_id=0,
        num_steps=8,
        schedule="cosine",
        temperature=1.0,
        adaptive_step=True,
        step_min=1e-4,
        step_max=0.5,
        time_eps=1e-3,
        early_exit=True,
    )

    assert final_ids.shape == init_ids.shape
    assert stats["dfm_nfe_realized"] <= 8
