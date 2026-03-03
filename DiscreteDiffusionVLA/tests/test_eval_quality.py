import torch

from prismatic.discrete_flow.dfm_decode import dfm_decode
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX


class _DummyTokenizer:
    def __init__(self, vocab_size: int = 32010, pad_token_id: int = 32000) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def batch_decode(self, ids_list):
        return [self.decode(ids) for ids in ids_list]


def _dummy_tokens_to_logits(cur: torch.LongTensor):
    # Deterministic logits favoring token id 1; never sample mask token id 0.
    B, L = cur.shape
    vocab_size = 5
    logits = torch.zeros(B, L, vocab_size)
    logits[..., 0] = -1e9
    logits[..., 1] = 10.0
    hidden = torch.zeros(B, L, 4)
    return logits, hidden


def test_action_masks_partition_action_tokens():
    action_begin = 1000
    action_end = action_begin + ACTION_DIM + 3
    token_ids = torch.tensor([[5] + list(range(action_begin, action_end)) + [7]])

    cur_mask = get_current_action_mask(token_ids, action_begin=action_begin, action_end=action_end)
    next_mask = get_next_actions_mask(token_ids, action_begin=action_begin, action_end=action_end)

    assert cur_mask.shape == token_ids.shape
    assert next_mask.shape == token_ids.shape
    assert cur_mask.dtype == torch.bool
    assert next_mask.dtype == torch.bool
    assert int(cur_mask.sum().item()) == ACTION_DIM
    assert int(next_mask.sum().item()) == (action_end - action_begin - ACTION_DIM)
    assert torch.all(~(cur_mask & next_mask))
    assert torch.all(~cur_mask[(token_ids < action_begin) | (token_ids >= action_end)])


def test_action_masks_legacy_fallback_runs():
    token_ids = torch.tensor([[ACTION_TOKEN_BEGIN_IDX + 1] * (ACTION_DIM + 2)])
    cur_mask = get_current_action_mask(token_ids)
    next_mask = get_next_actions_mask(token_ids)
    assert cur_mask.shape == token_ids.shape
    assert next_mask.shape == token_ids.shape


def test_action_tokenizer_legacy_anchor_uses_constant():
    tok = _DummyTokenizer(vocab_size=32010, pad_token_id=32000)
    action_tokenizer = ActionTokenizer(tok, bins=8, action_vocab_anchor="legacy")
    assert action_tokenizer.action_token_begin_idx == ACTION_TOKEN_BEGIN_IDX
    assert action_tokenizer.action_token_end_idx == ACTION_TOKEN_BEGIN_IDX + 8


def test_dfm_maskgit_resolves_all_masks():
    init_ids = torch.zeros(1, 8, dtype=torch.long)
    final_ids, _, stats = dfm_decode(
        init_ids=init_ids,
        tokens_to_logits=_dummy_tokens_to_logits,
        mask_token_id=0,
        num_steps=4,
        maskgit_num_steps=4,
        schedule="cosine",
        maskgit_schedule="cosine",
        temperature=1.0,
        adaptive_step=True,
        step_min=1e-4,
        step_max=0.5,
        time_eps=1e-3,
        early_exit=False,
        decode_mode="maskgit",
        debug_level=1,
    )

    assert final_ids.shape == init_ids.shape
    assert (final_ids == 0).sum().item() == 0
    assert stats["dfm_mask_frac_final"] == 0.0
    assert stats["dfm_unresolved_final"] == 0
    assert stats["dfm_n_action_positions"] == init_ids.numel()
    assert stats["dfm_n_masked_initial"] == init_ids.numel()
