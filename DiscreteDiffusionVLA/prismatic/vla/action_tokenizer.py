"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Optional, Union

import numpy as np
try:
    from transformers import PreTrainedTokenizerBase
except ImportError:  # pragma: no cover - optional dependency in tests
    class PreTrainedTokenizerBase:  # type: ignore
        pass

from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX


class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        action_vocab_anchor: str = "pad",
        action_token_end_idx: Optional[int] = None,
        action_token_begin_idx: Optional[int] = None,
        legacy_bins: bool = False,
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer = tokenizer
        self.n_bins = int(bins)
        self.min_action = min_action
        self.max_action = max_action
        self.action_vocab_anchor = action_vocab_anchor
        self.legacy_bins = legacy_bins

        # Legacy mapping: old eval used vocab_size anchoring + n_bins edges (not n_bins+1).
        if legacy_bins:
            self.bins = np.linspace(min_action, max_action, self.n_bins)
            self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
            self.action_vocab_anchor = "legacy"
            self.action_token_end_idx = int(self.tokenizer.vocab_size)
            expected_begin = int(self.action_token_end_idx - (self.n_bins + 1))
            if expected_begin != int(ACTION_TOKEN_BEGIN_IDX):
                raise ValueError(
                    "legacy_bins requires ACTION_TOKEN_BEGIN_IDX alignment. "
                    f"Expected begin={expected_begin} from vocab_size and n_bins, "
                    f"but ACTION_TOKEN_BEGIN_IDX={ACTION_TOKEN_BEGIN_IDX}. "
                    "Update the tokenizer/vocab or constants for legacy DD."
                )
            self.action_token_begin_idx = int(ACTION_TOKEN_BEGIN_IDX)
            return

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        if action_token_begin_idx is not None:
            self.action_token_begin_idx = int(action_token_begin_idx)
            self.action_token_end_idx = int(self.action_token_begin_idx + self.n_bins)
        elif action_token_end_idx is not None:
            self.action_token_end_idx = int(action_token_end_idx)
            self.action_token_begin_idx = int(self.action_token_end_idx - self.n_bins)
        else:
            if action_vocab_anchor == "legacy":
                self.action_token_begin_idx = int(ACTION_TOKEN_BEGIN_IDX)
                self.action_token_end_idx = int(self.action_token_begin_idx + self.n_bins)
                return
            if action_vocab_anchor == "pad":
                if self.tokenizer.pad_token_id is None:
                    raise ValueError("tokenizer.pad_token_id must be set when action_vocab_anchor='pad'")
                self.action_token_end_idx = int(self.tokenizer.pad_token_id)
            elif action_vocab_anchor == "vocab_size":
                self.action_token_end_idx = int(self.tokenizer.vocab_size)
            else:
                raise ValueError(f"Unknown action_vocab_anchor: {action_vocab_anchor}")

            self.action_token_begin_idx = int(self.action_token_end_idx - self.n_bins)

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the action vocabulary range."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.action_token_end_idx - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.action_token_end_idx - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.action_token_end_idx - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    def encode_actions_to_token_ids(self, action: np.ndarray) -> np.ndarray:
        """Discretize continuous actions and return token IDs directly."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        token_ids = self.action_token_end_idx - discretized_action
        return np.asarray(token_ids, dtype=np.int64).reshape(-1)

    @property
    def vocab_size(self) -> int:
        return self.n_bins
