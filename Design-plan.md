# Design Doc Plan: DFM vs Discrete Diffusion Design Questions

**Summary**
Create `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/design.md` as a concise, question‑driven checklist with brief tradeoffs. The document will enumerate key design decisions for adding discrete flow matching (DFM) alongside discrete diffusion, using details from `/Users/ali/dev/VLA-DFM/deep-research-report.md` to make the questions concrete (e.g., probability path choice, **CTMC hazard/tau‑leaping solver**, step size stability, scheduler changes).

**Scope & Format**
- **Format**: concise bullet list of questions, each with a brief tradeoff note.
- **Location**: `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/design.md`.
- **Audience**: implementers working in this repo who need to decide DFM specifics.

---

## Planned Content Structure

**1) Goals & Non‑Goals**
- What problem does DFM solve vs discrete diffusion (quality, latency, stability)?
- Are we targeting parity with current discrete diffusion results or improving speed/quality tradeoff?
- Non‑goals: Are we avoiding research‑level explorations (e.g., new tokenization or new backbone)?

**2) Modeling & Corruption Path**
- What is the **source distribution** for DFM: all‑`[MASK]` (δm) or uniform?  
  Tradeoff: δm keeps parity with current DD training; uniform may broaden support but changes semantics.
- Which **probability path** \(p_t\) do we use?
  - Convex mixture per token (`MixtureDiscreteProbPath`) vs generalized mixture w/ uniform noise.
  - Tradeoff: convex mixture is simplest and closest to DD; other paths might improve mixing but add complexity.
- What **coupling** do we assume (U‑coupling vs C‑coupling)?
  - Tradeoff: C‑coupling preserves partial structure but changes training samples.

**3) Time Conditioning**
- Do we **explicitly condition on time** \(t\)?
  - Tradeoff: explicit time can improve fit across t but requires model changes; implicit time keeps minimal changes and is closer to existing DD.
- If time‑conditioned, where is time injected?
  - Options: append timestep embedding to patch tokens; add dedicated token; add to attention bias.

**4) Training Objective**
- Which loss: **masked CE** vs **generalized KL** (MixturePathGeneralizedKL)?
  - Tradeoff: masked CE matches current code and is stable; generalized KL is more faithful to DFM theory but more complex.
- How is **mask ratio** scheduled during training?
  - Reuse cosine schedule vs new κ(t) schedule.
- Do we mask **only action tokens** (current behavior) or include other tokens?

**5) Sampling / Inference**
- Which **CTMC solver** for DFM?
  - Euler velocity update vs tau‑leaping (`MixtureDiscreteEulerSolver`).
  - Tradeoff: Euler is simpler; tau‑leaping aligns with DFM library and handles hazard rates.
- How to manage **step size stability**?
  - Fixed NFE vs adaptive `h = min(h, (1-κ)/κdot)`.
- Do we allow **post‑training schedule changes** (κ_t tuning)?
  - Tradeoff: enables inference tuning but may diverge from training distribution.
- How to handle **conditional token clamping** (if needed)?
  - For VLA, likely none; but keep option for infilling/ablations.

**6) Interface & Config**
- New flags: `use_discrete_flow_matching`, plus DFM sampling parameters (num_iter, schedule, temp).
- How to ensure **mutual exclusivity** with `use_discrete_diffusion`?
- How to keep **backward compatibility** with existing checkpoints/scripts?

**7) Data & Tokenization**
- Is action tokenization unchanged (bins + `[MASK]` token)?
- Any change to the **action chunk length** or stop token behavior?

**8) Metrics & Evaluation**
- How do we compare DFM vs DD?
  - Success rate, NFE‑latency tradeoff, quality at equal NFE.
- What ablations are required (time conditioning, κ schedule, solver)?

**9) Performance & Compute**
- Expected NFE and wall‑clock vs current 12‑step MaskGIT loop.
- Where are the **hot paths** (decoder loop vs CTMC update)?
- Can we reuse existing vectorized decode to avoid perf regressions?

**10) Risks & Stability**
- Probabilities going negative if h too large.
- Failure modes of discrete velocity update (need clipping?).
- Divergence between training (masked CE) and inference (CTMC dynamics).

---

## Implementation Steps (for creating the doc)

1. Create `/Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA/design.md`.
2. Add the section headings above.
3. Populate each section with concise question bullets and short tradeoff notes.
4. Cross‑reference DFM details from `/Users/ali/dev/VLA-DFM/deep-research-report.md` where they directly motivate questions (e.g., convex path, κ schedule, solver, adaptive step size, denoiser training).
5. Keep the doc terse and actionable.

---

## Assumptions & Defaults
- The document is **questions + brief tradeoffs**, not a decisions log.
- The initial DFM design is intended to be **as close as possible to current discrete diffusion** to minimize code changes.

---

If you want me to proceed with writing the file, switch me out of Plan mode and I’ll implement it.
