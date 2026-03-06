# Legacy vs Modern Training Paradigm (DD / DFM) - In Depth

This document explains the differences between **legacy mode** and the **current non-legacy training paradigm** in this codebase. It is written to be implementation-accurate and ties directly to the current code paths under `DiscreteDiffusionVLA`.

Scope:
- Discrete Diffusion (DD) training
- Discrete Flow Matching (DFM) training
- Prompt construction, action tokenization, binning, action range, masks, and config stamping

If you are comparing against the historical branch `EVAL-libero_object-openvla-2026_02_27-14_38_04`, the "legacy" behavior described below is the one we preserved for parity.

---

## 1) High-Level Behavior

Legacy mode is a **prompt-and-string based** action injection pipeline. It uses action **strings** (BPE-decoded tokens) embedded directly in the GPT turn, and relies on **legacy action range constants** and **legacy binning**. It also uses the **legacy action mask** rules that depend on a constant action token start index.

Modern (non-legacy) mode is a **token-ID based** pipeline. It bypasses BPE by writing **action token IDs** directly after the prompt, with explicit STOP/EOS management. It relies on **config-driven action vocab ranges** (anchor + begin index) and **n_bins+1 bin edges**.

Legacy mode is now:
- Default for DD training
- Optional for DFM training via `--legacy_dfm_mode`

Modern mode is:
- Default for DFM training (unless `legacy_dfm_mode=True`)
- Opt-out for DD training (`--legacy_train_mode False`)

---

## 2) Prompt Construction and Action Injection

### Legacy (DD/DFM with legacy tokenization)
Path: `prismatic/vla/datasets/datasets.py` (RLDSBatchTransform, legacy_mode=True)

Behavior:
1. Convert continuous actions to **action strings** via `ActionTokenizer.__call__`.
2. Concatenate current + future actions as strings.
3. Inject those strings directly into the GPT turn.
4. Use base tokenizer to tokenize the entire prompt, **with add_special_tokens=True**.
5. Do **not** strip STOP or manually append STOP.
6. Mask labels using **string length**, not token ID count.

Legacy prompt example:
```
Human: What action should the robot take to <task>?
GPT: <action_string>
```

### Modern (non-legacy)
Path: `prismatic/vla/datasets/datasets.py` (legacy_mode=False)

Behavior:
1. Convert actions into **token IDs** using `encode_actions_to_token_ids`.
2. GPT turn content is an empty string.
3. Tokenize only the prompt. If it ends with STOP, strip it.
4. Append action token IDs directly, then append STOP.
5. Mask labels using **token ID count**.

Modern prompt example:
```
Human: What action should the robot take to <task>?
GPT:
<action_token_ids...>
STOP
```

Key impact:
- **Sequence differs at training time.**
- **Loss mask alignment differs.**
- STOP/EOS placement differs.

---

## 3) Action Tokenization and Binning

### Legacy
Path: `prismatic/vla/action_tokenizer.py` (legacy_bins=True)

Behavior:
- Bins: `np.linspace(min_action, max_action, n_bins)` (n_bins edges)
- Bin centers: `n_bins - 1`
- Action vocab range:
  - `action_token_begin_idx = ACTION_TOKEN_BEGIN_IDX`
  - `action_token_end_idx = tokenizer.vocab_size`
  - Expected: `tokenizer.vocab_size - (n_bins + 1) == ACTION_TOKEN_BEGIN_IDX`
- Action tokens are produced by:
  - `token_id = action_token_end_idx - discretized_action`

### Modern
Path: `prismatic/vla/action_tokenizer.py` (legacy_bins=False)

Behavior:
- Bins: `np.linspace(min_action, max_action, n_bins + 1)` (n_bins+1 edges)
- Bin centers: `n_bins`
- Action vocab range is config-driven:
  - `action_vocab_anchor = pad | vocab_size | legacy`
  - `action_token_begin_idx` can override
- Action tokens are produced by:
  - `token_id = action_token_end_idx - discretized_action`
  - `action_token_end_idx` depends on anchor or begin idx

Key impact:
- **Bin edges differ** (n_bins vs n_bins+1)
- **Action range differs** (constant legacy vs anchor-based)

---

## 4) Action Range Anchoring

### Legacy
Action range is **fixed** by constant:
- `ACTION_TOKEN_BEGIN_IDX`
- `action_token_end_idx = vocab_size`

This is enforced by:
- `action_vocab_anchor = "legacy"`
- `action_token_begin_idx = ACTION_TOKEN_BEGIN_IDX`
- Fail-fast if tokenizer.vocab_size does not align with constant.

### Modern
Action range is **derived** from:
- `action_vocab_anchor` (default: `pad`)
- `action_token_begin_idx` (optional override)
- `pad_token_id` or `vocab_size`

Example:
```
anchor=pad => action_end = pad_token_id
action_begin = action_end - n_bins
```

Key impact:
- Modern training is sensitive to pad/mask token IDs and config drift.
- Legacy is fixed and reproducible across runs if constants match.

---

## 5) Action Masks and Loss Targeting

### Legacy
Path: `prismatic/training/train_utils.py`

Behavior:
- Masks use **legacy fallback** when begin/end not provided.
- Logic depends on `ACTION_TOKEN_BEGIN_IDX`.
- Used when `legacy_tokenization_mode=True`.

### Modern
Behavior:
- Masks are computed with explicit `action_begin`/`action_end`.
- Errors or mismatches occur if begin/end is wrong or missing.

Key impact:
- Legacy masks are robust but constant-based.
- Modern masks are flexible but require correct config.

---

## 6) Model Binning and Action Range (Model Internals)

### Legacy
Path: `prismatic/extern/hf/modeling_prismatic.py`

Behavior:
- `bins = np.linspace(-1, 1, n_action_bins)` (n_bins edges)
- Bin centers length = `n_bins - 1`
- `_action_vocab_range()` in legacy uses:
  - begin = `ACTION_TOKEN_BEGIN_IDX + 1`
  - end = begin + n_bins

### Modern
Behavior:
- `bins = np.linspace(-1, 1, n_action_bins + 1)` (n_bins+1 edges)
- Bin centers length = `n_bins`
- `_action_vocab_range()` uses config anchor + begin idx.

Key impact:
- Legacy DD parity requires legacy binning in model.
- DFM legacy requires careful off-by-one handling for action ranges.

---

## 7) Mask Token Handling

### Legacy DD
- Historically may not require mask token.
- Modern code now stamps mask token if missing only in non-legacy paths.

### Legacy DFM
- **Requires mask token** for DFM decoding.
- Legacy DFM now auto-adds `<mask>` when missing in tokenizer setup.

Key impact:
- DFM legacy needs a valid `mask_token_id`.

---

## 8) Config Stamping (Checkpoint Metadata)

### Legacy
On training, config is explicitly stamped with:
- `legacy_train_mode=True`
- `legacy_eval_mode=True`
- `action_vocab_anchor="legacy"`
- `action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX`

This makes checkpoints **self-describing** and prevents eval drift.

### Modern
Config is stamped based on:
- `action_vocab_anchor` and `action_token_begin_idx` resolved at train time
- DFM explicitly stamps action range (non-legacy)

---

## 9) Evaluation Alignment

### Legacy
Eval auto-enables `legacy_eval_mode` if:
- checkpoint config has `legacy_train_mode=True` or `legacy_eval_mode=True`

For DFM, eval can be forced to legacy if:
- `legacy_eval_mode=True` in checkpoint config

### Modern
Eval uses config-based action range and prompt builder by default.

---

## 10) Summary Table

| Aspect | Legacy Mode | Modern Mode |
|---|---|---|
| Action injection | String-based BPE action tokens embedded in GPT turn | Token-ID injection after prompt |
| STOP/EOS handling | No explicit strip/append | STOP stripped then appended |
| Binning | `n_bins` edges | `n_bins + 1` edges |
| Action range | Fixed constant `ACTION_TOKEN_BEGIN_IDX` | Anchor + begin idx from config |
| Action mask | Legacy constant-based mask | Begin/end range mask |
| Model binning | Legacy bins (n_bins edges) | Modern bins (n_bins+1 edges) |
| Config stamping | Explicit legacy fields stamped | Anchor/begin stamped |
| DFM mask token | Auto-add if missing (legacy DFM only) | Required/validated (modern) |

---

## 11) Practical Implications

1. **Legacy is more stable for DD parity**  
   It exactly matches the old branch behavior and avoids off-by-one drift.

2. **Modern is more flexible but fragile**  
   Action range and binning depend on config correctness and tokenizer state.

3. **DFM legacy requires careful alignment**  
   DFM uses the model's action range logic; off-by-one mismatches are easy to introduce.

---

## 12) Reference Files

- Training entry: `vla-scripts/finetune.py`
- Dataset transform: `prismatic/vla/datasets/datasets.py`
- Tokenizer: `prismatic/vla/action_tokenizer.py`
- Action masks: `prismatic/training/train_utils.py`
- Model internals: `prismatic/extern/hf/modeling_prismatic.py`
- Eval alignment: `experiments/robot/openvla_utils.py`, `experiments/robot/libero/run_libero_eval.py`

