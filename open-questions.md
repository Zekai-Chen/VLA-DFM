# Open Questions: Design-plan.md vs deep-research-report.md vs PLAN.md

## Goals & Scope
- What primary objective are we optimizing (quality, latency, stability, or speed‑quality tradeoff)?
- Are we targeting parity with discrete diffusion results or explicitly aiming to improve tradeoffs?
- Are research-level changes (new tokenization/backbone) explicitly out of scope?

## Modeling & Path
- Should we exploit the report’s claim that with mask-source + convex path, the denoiser can be **time-independent** (no explicit t input)?
- Should we allow probability paths beyond convex mixtures (e.g., add uniform-noise mixtures), or keep convex-only for parity?
- Which coupling do we want in practice: U-coupling vs C-coupling (partial mask of x1)?
- Do we restrict to **mask-only** corruption, or allow non‑mask token replacement noise?
- Do we reuse the current cosine mask schedule or introduce a new \(\kappa(t)\) schedule?

## Training Objective
- Do we stick with **masked CE** only, or also implement the **generalized KL / weighted loss** from the DFM library?
- If we keep masked CE, do we reweight by \(\dot\kappa_t/(1-\kappa_t)\) as the DFM theory suggests?
- Do we compute loss only on masked positions or over all action tokens?

## Time Conditioning
- If time conditioning is used, where should time be injected (patch tokens, special token, attention bias)?
- If not, do we still need a time embedding for inference-only scheduling changes?

## Sampling / CTMC Solver
- Do we implement **Euler velocity update** or **hazard-rate tau-leaping** (MixtureDiscreteEulerSolver)?
- Will we add **adaptive step size** \(h = min(h, (1-\kappa)/\dot\kappa)\) to avoid negative probabilities?
- Should we allow **post-training schedule changes** (\(\kappa_t\)) without retraining, as the report suggests is viable for mask-source?
- Do we implement **NFE skipping** when the state doesn’t change (NFE ≤ N claim)?
- Do we reuse existing `parallel_decode.decode()` or replace it entirely with a CTMC solver?

## Inference Strategy
- Are we fully replacing MaskGIT-style commit/remask with CTMC updates, or offering a hybrid mode?
- Do we need explicit **token clamping** for conditional generation or infilling use cases?
- What default NFE / temperature / schedule values should be exposed and tuned?
- Do we keep the discrete diffusion secondary re-masking heuristics in any hybrid mode?

## Interface & Config
- Do we keep `use_discrete_diffusion` and add `use_discrete_flow_matching`, or rename/replace flags?
- How do we enforce **mutual exclusivity** between DFM and discrete diffusion at runtime?
- Which DFM inference hyperparameters should be first-class config options?

## Data & Tokenization
- Is the action tokenization unchanged (bins + `[MASK]`), or do we adjust vocab size/bins?
- Do we preserve the current action chunk length and stop token behavior?

## Metrics & Evaluation
- What evaluation metrics define “success” (success rate, latency, NFE‑quality curve)?
- Which ablations are required (time conditioning, \(\kappa(t)\) schedule, solver choice)?

## Performance & Compute
- What are expected NFEs vs the current 12‑step MaskGIT loop?
- Where are the hot paths (decoder loop vs CTMC update), and can we keep vectorized decode?

## Compatibility & Rollout
- How do we preserve backward compatibility with existing **discrete diffusion checkpoints and scripts**?
- Do we need a compatibility migration strategy if we change flags or config schemas?

## Risks & Stability
- Could probabilities go negative or require clamping if step sizes are too large?
- Is there a mismatch risk between masked‑CE training and CTMC inference dynamics?
