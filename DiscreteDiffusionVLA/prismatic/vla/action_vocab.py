"""
action_vocab.py

Helpers for resolving and validating action token ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX


@dataclass(frozen=True)
class ActionVocabRange:
    begin: int
    end: int
    n_bins: int
    anchor: str
    pad_token_id: Optional[int]
    vocab_size: int


def resolve_action_vocab(
    tokenizer,
    n_bins: int,
    anchor: str,
    begin_override: Optional[int] = None,
) -> ActionVocabRange:
    """Resolve the [begin, end) action token range from a tokenizer and anchor."""
    if begin_override is not None:
        begin = int(begin_override)
        end = int(begin + n_bins)
    else:
        if anchor == "legacy":
            begin = int(ACTION_TOKEN_BEGIN_IDX)
            end = int(begin + n_bins)
        elif anchor == "pad":
            if tokenizer.pad_token_id is None:
                raise ValueError("tokenizer.pad_token_id must be set when action_vocab_anchor='pad'")
            end = int(tokenizer.pad_token_id)
        elif anchor == "vocab_size":
            end = int(tokenizer.vocab_size)
        else:
            raise ValueError(f"Unknown action_vocab_anchor: {anchor}")
        if anchor != "legacy":
            begin = int(end - n_bins)

    return ActionVocabRange(
        begin=begin,
        end=end,
        n_bins=int(n_bins),
        anchor=anchor,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        vocab_size=int(tokenizer.vocab_size),
    )


def validate_action_vocab_alignment(
    action_range: ActionVocabRange,
    token_ids: Iterable[int],
) -> None:
    """Validate that token IDs fall within the resolved action token range."""
    if action_range.begin < 0:
        raise ValueError(f"Action vocab begin ({action_range.begin}) is negative.")

    ids = np.asarray(list(token_ids), dtype=np.int64)
    if ids.size == 0:
        return
    min_id = int(ids.min())
    max_id = int(ids.max())
    if min_id < action_range.begin or max_id >= action_range.end:
        raise ValueError(
            "Action token IDs out of range: "
            f"min={min_id} max={max_id} expected=[{action_range.begin}, {action_range.end})"
        )
