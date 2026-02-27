import torch

from prismatic.discrete_flow.dfm_decode import dfm_decode


def _dummy_tokens_to_logits(cur: torch.LongTensor):
    # Simple deterministic logits favoring token id 1.
    B, L = cur.shape
    vocab_size = 7
    logits = torch.zeros(B, L, vocab_size)
    logits[..., 1] = 2.0
    logits[..., 2] = 1.0
    hidden = torch.zeros(B, L, 4)
    return logits, hidden


def test_dfm_decode_clamp_invariant():
    init_ids = torch.zeros(1, 4, dtype=torch.long)
    clamp_mask = torch.tensor([[True, False, False, False]])
    clamp_values = torch.tensor([[3, 0, 0, 0]])

    final_ids, _, stats = dfm_decode(
        init_ids=init_ids,
        tokens_to_logits=_dummy_tokens_to_logits,
        mask_token_id=0,
        num_steps=3,
        schedule="linear",
        temperature=1.0,
        adaptive_step=True,
        step_min=1e-4,
        step_max=0.5,
        time_eps=1e-3,
        early_exit=False,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
    )

    assert final_ids.shape == init_ids.shape
    assert final_ids[0, 0].item() == 3
    # Basic stats keys exist
    assert "dfm_nfe_realized" in stats
    assert "dfm_num_changed_tokens" in stats


def test_dfm_decode_stats_bounds():
    init_ids = torch.zeros(2, 6, dtype=torch.long)
    final_ids, _, stats = dfm_decode(
        init_ids=init_ids,
        tokens_to_logits=_dummy_tokens_to_logits,
        mask_token_id=0,
        num_steps=4,
        schedule="cosine",
        temperature=1.0,
        adaptive_step=True,
        step_min=1e-4,
        step_max=0.5,
        time_eps=1e-3,
        early_exit=True,
    )

    assert final_ids.shape == init_ids.shape
    assert stats["dfm_nfe_realized"] <= 4
    assert isinstance(stats["dfm_num_changed_tokens"], list)


def test_dfm_decode_maskgit_mode():
    init_ids = torch.zeros(1, 5, dtype=torch.long)
    final_ids, _, stats = dfm_decode(
        init_ids=init_ids,
        tokens_to_logits=_dummy_tokens_to_logits,
        mask_token_id=0,
        num_steps=3,
        schedule="cosine",
        temperature=1.0,
        adaptive_step=True,
        step_min=1e-4,
        step_max=0.5,
        time_eps=1e-3,
        early_exit=False,
        decode_mode="maskgit",
    )

    assert final_ids.shape == init_ids.shape
    assert "dfm_nfe_realized" in stats
    assert isinstance(stats["dfm_num_changed_tokens"], list)
