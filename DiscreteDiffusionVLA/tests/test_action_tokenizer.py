import numpy as np

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.action_vocab import resolve_action_vocab, validate_action_vocab_alignment
from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX


class _DummyTokenizer:
    def __init__(self, vocab_size: int = 32010, pad_token_id: int = 32000) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def batch_decode(self, ids_list):
        return [self.decode(ids) for ids in ids_list]


def test_action_tokenizer_range_pad_anchor():
    tok = _DummyTokenizer(vocab_size=32010, pad_token_id=32000)
    action_tokenizer = ActionTokenizer(tok, bins=8, action_vocab_anchor="pad")
    assert action_tokenizer.action_token_end_idx == 32000
    assert action_tokenizer.action_token_begin_idx == 31992
    assert (action_tokenizer.action_token_end_idx - action_tokenizer.action_token_begin_idx) == action_tokenizer.n_bins


def test_action_tokenizer_decode_bin_centers():
    tok = _DummyTokenizer(vocab_size=32010, pad_token_id=32000)
    action_tokenizer = ActionTokenizer(tok, bins=4, action_vocab_anchor="pad")
    token_ids = np.array([action_tokenizer.action_token_end_idx - i for i in [1, 2, 3, 4]])
    decoded = action_tokenizer.decode_token_ids_to_actions(token_ids)
    assert decoded.shape == (4,)
    assert np.allclose(decoded, action_tokenizer.bin_centers[[0, 1, 2, 3]])


def test_action_tokenizer_no_pad_collision():
    tok = _DummyTokenizer(vocab_size=32010, pad_token_id=32000)
    action_tokenizer = ActionTokenizer(tok, bins=16, action_vocab_anchor="pad")
    assert not (action_tokenizer.action_token_begin_idx <= tok.pad_token_id < action_tokenizer.action_token_end_idx)


def test_action_vocab_alignment_pass():
    tok = _DummyTokenizer(vocab_size=32010, pad_token_id=32000)
    n_bins = 8
    action_range = resolve_action_vocab(tok, n_bins, "pad")
    action_tokenizer = ActionTokenizer(tok, bins=n_bins, action_vocab_anchor="pad")
    assert action_tokenizer.action_token_begin_idx == action_range.begin
    sample_actions = np.zeros((1, 7), dtype=np.float32)
    sample_ids = action_tokenizer.encode_actions_to_token_ids(sample_actions)
    validate_action_vocab_alignment(action_range, sample_ids.tolist())


def test_action_vocab_alignment_fail():
    tok = _DummyTokenizer(vocab_size=32010, pad_token_id=32000)
    action_range = resolve_action_vocab(tok, 8, "pad")
    bad_ids = [action_range.begin - 1, action_range.end]
    try:
        validate_action_vocab_alignment(action_range, bad_ids)
    except ValueError as exc:
        assert "Action token IDs out of range" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-range action token IDs")


def test_action_vocab_legacy_anchor():
    tok = _DummyTokenizer(vocab_size=32010, pad_token_id=32000)
    action_range = resolve_action_vocab(tok, 8, "legacy")
    assert action_range.begin == ACTION_TOKEN_BEGIN_IDX
    assert action_range.end == ACTION_TOKEN_BEGIN_IDX + 8
