import numpy as np
import pytest
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
    action_tokenizer = ActionTokenizer(tok, bins=256, legacy_bins=True, action_vocab_anchor="legacy")
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


def test_legacy_train_stamps_config():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    finetune_path = root / "vla-scripts" / "finetune.py"
    spec = importlib.util.spec_from_file_location("finetune", finetune_path)
    finetune = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(finetune)
    apply_legacy_tokenization_overrides = finetune.apply_legacy_tokenization_overrides

    cfg = SimpleNamespace(
        legacy_train_mode=True,
        legacy_dfm_mode=False,
        use_discrete_flow_matching=False,
    )
    model_config = SimpleNamespace(
        n_action_bins=256,
        legacy_train_mode=False,
        legacy_eval_mode=False,
        action_vocab_anchor="pad",
        action_token_begin_idx=None,
    )

    class _Proc:
        def __init__(self, vocab_size: int):
            self.tokenizer = SimpleNamespace(vocab_size=vocab_size)

    processor = _Proc(vocab_size=ACTION_TOKEN_BEGIN_IDX + 257)

    apply_legacy_tokenization_overrides(cfg, model_config, processor, legacy_tokenization_mode=True)

    assert model_config.legacy_train_mode is True
    assert model_config.legacy_eval_mode is True
    assert model_config.action_vocab_anchor == "legacy"
    assert model_config.action_token_begin_idx == ACTION_TOKEN_BEGIN_IDX


def test_legacy_dfm_mode_requires_dfm():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    finetune_path = root / "vla-scripts" / "finetune.py"
    spec = importlib.util.spec_from_file_location("finetune", finetune_path)
    finetune = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(finetune)
    resolve_legacy_tokenization_mode = finetune.resolve_legacy_tokenization_mode

    cfg = SimpleNamespace(
        use_discrete_diffusion=False,
        use_discrete_flow_matching=False,
        legacy_train_mode=None,
        legacy_dfm_mode=True,
    )
    with pytest.raises(ValueError):
        resolve_legacy_tokenization_mode(cfg)


def test_legacy_dfm_action_range():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    finetune_path = root / "vla-scripts" / "finetune.py"
    spec = importlib.util.spec_from_file_location("finetune", finetune_path)
    finetune = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(finetune)
    apply_legacy_tokenization_overrides = finetune.apply_legacy_tokenization_overrides

    cfg = SimpleNamespace(
        legacy_train_mode=False,
        legacy_dfm_mode=True,
        use_discrete_flow_matching=True,
    )
    model_config = SimpleNamespace(
        n_action_bins=256,
        legacy_train_mode=False,
        legacy_eval_mode=False,
        action_vocab_anchor="pad",
        action_token_begin_idx=None,
    )

    class _Proc:
        def __init__(self, vocab_size: int, mask_token_id: int):
            self.tokenizer = SimpleNamespace(vocab_size=vocab_size, mask_token_id=mask_token_id)

    processor = _Proc(vocab_size=ACTION_TOKEN_BEGIN_IDX + 257, mask_token_id=32001)

    apply_legacy_tokenization_overrides(cfg, model_config, processor, legacy_tokenization_mode=True)

    assert model_config.legacy_train_mode is True
    assert model_config.legacy_eval_mode is True
    assert model_config.action_vocab_anchor == "legacy"
    assert model_config.action_token_begin_idx == ACTION_TOKEN_BEGIN_IDX


def test_legacy_dfm_model_range_uses_constant():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    root = Path(__file__).resolve().parents[1]
    model_path = root / "prismatic" / "extern" / "hf" / "modeling_prismatic.py"
    spec = importlib.util.spec_from_file_location("modeling_prismatic", model_path)
    modeling_prismatic = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modeling_prismatic)

    dummy = SimpleNamespace(
        config=SimpleNamespace(
            n_action_bins=256,
            legacy_eval_mode=True,
            legacy_train_mode=False,
            use_discrete_flow_matching=True,
        ),
        pad_token_id=32000,
        vocab_size=32000,
        bin_centers=np.zeros(256, dtype=np.float32),
    )

    begin, end, n_bins = modeling_prismatic.OpenVLAForActionPrediction._action_vocab_range(dummy)
    assert begin == ACTION_TOKEN_BEGIN_IDX
    assert end == begin + n_bins
