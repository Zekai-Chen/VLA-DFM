import numpy as np
import torch
from types import SimpleNamespace

from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets import RLDSBatchTransform


class _DummyTokenizer:
    def __init__(self, vocab_size: int = 32000, pad_token_id: int = 0) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.last_text = None
        self.last_ids = None

    def __call__(self, text, add_special_tokens=True):
        self.last_text = text
        self.last_ids = list(range(len(text)))
        return SimpleNamespace(input_ids=self.last_ids)

    def decode(self, ids):
        return "x" * len(ids)

    def batch_decode(self, ids_list):
        return ["x" * len(ids) for ids in ids_list]


class _DummyPromptBuilder:
    def __init__(self, model_family: str, system_prompt=None) -> None:
        self.prompt = ""

    def add_turn(self, role: str, message: str) -> str:
        self.prompt += f"{role}:{message}\n"
        return message

    def get_prompt(self) -> str:
        return self.prompt


def _image_transform(_img):
    return torch.zeros((3, 2, 2), dtype=torch.float32)


def test_rlds_batch_transform_legacy_action_string_masking():
    tok = _DummyTokenizer()
    action_tokenizer = ActionTokenizer(tok, bins=8, legacy_bins=True, action_vocab_anchor="vocab_size")
    transform = RLDSBatchTransform(
        action_tokenizer,
        tok,
        image_transform=_image_transform,
        prompt_builder_fn=_DummyPromptBuilder,
        legacy_mode=True,
    )

    actions = np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
    batch = {
        "dataset_name": "dummy",
        "action": actions,
        "observation": {"image_primary": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
        "task": {"language_instruction": b"pick up object"},
    }

    out = transform(batch)
    current_action_string = action_tokenizer(actions[0])
    future_actions_string = "".join(action_tokenizer(actions[1:]))
    action_chunk_string = current_action_string + future_actions_string
    action_chunk_len = len(action_chunk_string)

    assert action_chunk_string in tok.last_text
    assert out["input_ids"].shape[0] == len(tok.last_ids)
    labels = out["labels"].cpu().numpy()
    assert np.all(labels[: -(action_chunk_len + 1)] == IGNORE_INDEX)


def test_legacy_action_mask_fallback():
    token_ids = torch.tensor(
        [[IGNORE_INDEX, IGNORE_INDEX] + [ACTION_TOKEN_BEGIN_IDX + 5] * ACTION_DIM + [0, 0]]
    )
    current_mask = get_current_action_mask(token_ids)
    next_mask = get_next_actions_mask(token_ids)
    assert int(current_mask.sum().item()) == ACTION_DIM
    assert int(next_mask.sum().item()) == 0
