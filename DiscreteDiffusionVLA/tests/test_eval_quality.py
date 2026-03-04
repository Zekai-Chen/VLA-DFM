import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PROMPT_UTILS = ROOT / "prismatic" / "vla" / "prompt_utils.py"
spec = importlib.util.spec_from_file_location("prompt_utils", _PROMPT_UTILS)
prompt_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prompt_utils)
_build_vla_prompt = prompt_utils.build_vla_prompt
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


def test_legacy_prompt_string():
    prompt = _build_vla_prompt("Pick up the mug", legacy=True)
    assert prompt == "In: What action should the robot take to pick up the mug?\nOut:"
    assert "</s>" not in prompt


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


def test_action_tokenizer_legacy_bins_vocab_anchor():
    tok = _DummyTokenizer(vocab_size=32000, pad_token_id=31990)
    action_tokenizer = ActionTokenizer(tok, bins=8, legacy_bins=True)
    assert action_tokenizer.action_token_end_idx == 32000
    assert action_tokenizer.action_token_begin_idx == 32000 - 9
    assert len(action_tokenizer.bin_centers) == 7
    token_ids = action_tokenizer.encode_actions_to_token_ids(np.array([-1.0, 0.0, 1.0]))
    assert token_ids.min() >= (32000 - 8)
    assert token_ids.max() <= 31999


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
