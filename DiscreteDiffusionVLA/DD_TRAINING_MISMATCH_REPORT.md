# DD Training Mismatch Report (2026-02-27 Parity)

This document summarizes **discrete diffusion (DD) training** mismatches between:
- `codex/generalizedkl-loss-repair` (current)
- `EVAL-libero_object-openvla-2026_02_27-14_38_04` (old reference)

It also notes the **fixes applied** in this codebase to restore 2026-02-27 parity for DD training.

---

## 1) ActionTokenizer binning + action range
**File:** `DiscreteDiffusionVLA/prismatic/vla/action_tokenizer.py`

**Old behavior (2026-02-27):**
- `bins = np.linspace(min_action, max_action, n_bins)`
- `action_token_begin_idx = vocab_size - (n_bins + 1)`
- Token IDs from `vocab_size - discretized_action`

**Current behavior (modern path):**
- `bins = np.linspace(min_action, max_action, n_bins + 1)`
- Action range based on `action_vocab_anchor` (default `pad`) or explicit begin/end
- Token IDs from `action_token_end_idx - discretized_action`

**Fix applied:**
- Legacy DD mode uses old bin edges and vocab-anchored range:
  - `bins = np.linspace(min_action, max_action, n_bins)`
  - `action_token_begin_idx = vocab_size - (n_bins + 1)`
  - `action_token_end_idx = vocab_size`

---

## 2) Training prompt injection: action strings vs token IDs
**File:** `DiscreteDiffusionVLA/prismatic/vla/datasets/datasets.py`

**Old behavior:**
- Convert actions to **strings** via `ActionTokenizer(...)`
- Inject action string directly into GPT turn
- Tokenize once with `add_special_tokens=True`
- No STOP stripping or manual STOP append
- Masking uses **string length** (`len(action_chunk_string)`)

**Current behavior (modern path):**
- Convert actions to **token IDs** via `encode_actions_to_token_ids()`
- GPT turn empty; append action IDs manually
- Strip trailing STOP if present; append STOP after action IDs
- Masking uses **token count** (`len(action_chunk_ids)`)

**Fix applied:**
- Legacy DD mode uses string-based action injection and string-length masking.

---

## 3) Action mask rules (training)
**File:** `DiscreteDiffusionVLA/prismatic/training/train_utils.py`
**File:** `DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

**Old behavior:**
- Mask based on `ACTION_TOKEN_BEGIN_IDX` constant (31743)
- Action tokens identified by `token_ids > ACTION_TOKEN_BEGIN_IDX`

**Current behavior (modern path):**
- Mask based on explicit `action_begin/action_end` from config

**Fix applied:**
- Legacy DD mode always uses legacy masks (no begin/end range).

---

## 4) Model binning inside OpenVLAForActionPrediction
**File:** `DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

**Old behavior:**
- `bins = np.linspace(-1, 1, n_action_bins)`
- `bin_centers` length = `n_bins - 1`

**Current behavior (modern path):**
- `bins = np.linspace(-1, 1, n_action_bins + 1)`
- `bin_centers` length = `n_bins`

**Fix applied:**
- Legacy DD mode uses old bin edges (n_bins).

---

## 5) Action vocab range & validation
**File:** `DiscreteDiffusionVLA/prismatic/extern/hf/modeling_prismatic.py`

**Old behavior:**
- No action vocab validation
- Action range implicit via `ACTION_TOKEN_BEGIN_IDX`

**Current behavior (modern path):**
- `_action_vocab_range()` depends on `action_vocab_anchor`/begin
- `_validate_action_vocab()` can raise when overlapping pad/mask tokens

**Fix applied:**
- Legacy DD mode skips validation and uses legacy action range/masks.

---

## 6) Training script defaults
**File:** `DiscreteDiffusionVLA/vla-scripts/finetune.py`

**Old behavior:**
- `ActionTokenizer(processor.tokenizer)` only
- No anchor/begin overrides
- No auto mask-token insertion

**Current behavior (modern path):**
- ActionTokenizer uses config anchor/begin (default `pad`)
- May auto-add mask token if missing

**Fix applied:**
- DD training defaults to legacy unless explicitly opted out
- Fail-fast if `ACTION_TOKEN_BEGIN_IDX` doesn’t match tokenizer vocab range
- Legacy DD **does not** auto-add mask token (to preserve vocab size)

---

## 7) Config defaults introduced
**File:** `DiscreteDiffusionVLA/prismatic/extern/hf/configuration_prismatic.py`

**Old behavior:**
- No `action_vocab_anchor` or `action_token_begin_idx`

**Current behavior:**
- Defaults `action_vocab_anchor = "pad"`

**Fix applied:**
- Legacy DD ignores these defaults and uses vocab-anchored range

---

# Summary
The DD training regressions are consistent with:
- **Action range shift** (pad-anchored vs vocab-anchored)
- **Bin edge shift** (n_bins vs n_bins+1)
- **Prompt tokenization shift** (string/BPE vs raw IDs)
- **STOP/EOS handling change**
- **Masking logic change** (constant-based vs range-based)

Legacy DD mode restores 2026-02-27 behavior end-to-end while keeping DFM unchanged.
