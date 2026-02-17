## Goals & Scope

**What primary objective are we optimizing (quality, latency, stability, or speed-quality tradeoff)?**
Make this a **speed–quality tradeoff** objective with a hard constraint on **no regressions in success rate** on your main benchmarks. Discrete diffusion VLA’s current value prop is strong quality with ~12 refinement steps; DFM’s value prop is getting comparable quality at **lower NFE** (or better stability at the same compute) using CTMC-style updates and safe step sizing. Practically: you optimize for *equal-or-better success rate at fewer steps and predictable stability*.

**Are we targeting parity with discrete diffusion results or explicitly aiming to improve tradeoffs?**
Target **parity first** (same tokenization, same backbone, same chunk length, same evaluation) and only then chase improved NFEs. DFM gives you knobs (schedule, step size, solver, optional corrector) that can improve compute/quality; but you want your first diff to be attributable to “DFM vs diffusion,” not a pile of other changes. Once parity is hit, you tune for lower NFE and/or faster wall-clock.

**Are research-level changes (new tokenization/backbone) explicitly out of scope?**
Yes—treat them as out of scope for the first migration. The VLA paper’s decoder is already designed around **discrete action chunks** and a **[MASK] token**, which maps cleanly to DFM’s “special token source” setup. If you change vocab/binning/backbone now, you lose the ability to isolate whether DFM helped.

---

## Modeling & Path

**Should we exploit the claim that with mask-source + convex path, the denoiser can be time-independent (no explicit t input)?**
Yes, that’s the cleanest swap. With **source = all-[MASK]** and **mixture/convex path** (per-token mixture between source token and target token), the model can learn a **time-independent posterior/denoiser** (p_\theta(x_1 \mid x_t, c)) that doesn’t need (t) explicitly, and you can still vary schedules at inference. This keeps architecture changes minimal (no new time embedding plumbing) and reduces train/infer mismatch risk.

- Ok in 10.1 flow matching is some time equal to diffusion depending on the convension, FM is 0 to 1, and diffusion is 0 to inf. So we should use the 0 to 1 convention for time, so we can reuse the cosine schedule and have a more direct mapping to the DFM paper’s math. Look at page 71 for reparameterization details.

**Should we allow probability paths beyond convex mixtures (e.g., add uniform-noise mixtures), or keep convex-only for parity?**
Keep **convex mixture only** for parity and simplicity. Convex mixture paths are the “default” in the DFM discrete recipe because they produce a simple conditional process (tokens are either source-state or target-state with probability (\kappa(t))). You *can* extend to uniform-token noise (source (p(x_0)) uniform over vocab) or add extra mixture components, but that introduces extra bookkeeping (source distribution must be known/handled everywhere) and usually pushes you toward time-conditioning and heavier ablations.

**Which coupling do we want in practice: U-coupling vs C-coupling (partial mask of x1)?**
For **VLA action generation**, default to **U-coupling / independent pairing** with a *constant* source sample: (x_0 = (m,\dots,m)) where (m=[MASK]). That’s exactly the “text generation with special token source” setup described in the Flow Matching guide: independent pairing with ( \pi(x_0,x_1)=\delta(x_0=m),q(x_1)).
C-coupling (partial mask of (x_1)) is great for **infilling / inpainting** style tasks where (x_0) is “a partially observed version of (x_1).” Use it only if you explicitly want “generate missing sub-actions given fixed sub-actions” as a first-class feature.

**Do we restrict to mask-only corruption, or allow non-mask token replacement noise?**
Restrict to **mask-only** for the first pass. Mask-only aligns with the current VLA training signal (masked CE) and preserves the “easy-first” intuition (uncertain tokens remain masked longer). Non-mask replacement noise is possible (source uniform or random tokens), but it changes the semantics of “corruption” and often increases the burden on the model to recover from arbitrary wrong tokens—usually not what you want for low-NFE robotics decoding.

**Do we reuse the current cosine mask schedule or introduce a new (\kappa(t)) schedule?**
Reuse cosine initially, but implement it as a **(\kappa(t)) scheduler** in DFM terms (so you can swap schedules without touching model code). In mixture-path DFM, (\kappa(t)) is the interpolation coefficient; cosine is a fine first choice because it matches what the VLA inference already uses. After parity, test polynomial schedules (common in DFM code) and expose schedule choice as an inference hyperparameter.

---

## Training Objective

**Do we stick with masked CE only, or also implement generalized KL / weighted loss from the DFM library?**
Implement the **generalized KL (DFM) loss** if you want the “most correct” discrete-flow objective and the cleanest connection to CTMC sampling. If you want the smallest code diff, masked CE is acceptable, but you should treat it as an approximation and validate carefully against solver behavior. A practical compromise: implement generalized KL and keep masked CE behind a flag for ablations.

- We should use the factorized velocity, in the paper, we can train the model on two losses, one is the 7.28 equation and 7.32, so we should likely implement the condition matching loss instead of the generalized KL loss. The conditional matching loss is a weighted CE that has the same form as the original masked CE, but with a time-dependent weight. 

- Check workable code for other dicrete DFM implementations

**If we keep masked CE, do we reweight by (\dot\kappa_t/(1-\kappa_t)) as the DFM theory suggests?**
Yes—if you keep CE, **do the reweighting** (or you’ll overweight late timesteps and undertrain the early “high-mask” regime). In mixture-path DFM, that factor naturally appears in the derived per-token objective; using it makes your CE objective match the dynamics the solver expects. In practice you’ll clamp/extrema-guard the weight to avoid explosions near the endpoints.

- Check workable code for other dicrete DFM implementations

**Do we compute loss only on masked positions or over all action tokens?**
Compute loss **only on the source/masked positions** for parity with the existing discrete diffusion training, and because it’s the most sample-efficient signal (you’re training where the model is uncertain). For generalized KL, you may technically have terms for all positions, but you can implement the factorized form and still restrict to the positions in the “source state” at time (t) (i.e., where (x_t) equals `[MASK]`). If you later introduce non-mask noise, revisit this because “masked positions” no longer captures “corrupted positions.”

- Follow them as close as possible to make sure that we can get positive results. 
---

## Time Conditioning

**If time conditioning is used, where should time be injected (patch tokens, special token, attention bias)?**
If you need time conditioning (because you adopt non-convex paths or conditional couplings), the lowest-risk approach is: **scalar time embedding → added (or FiLM’d) into the action-token embeddings** (not vision/language). That keeps VLM behavior stable and confines “flow-time” effects to the action decoder. A special learned “time token” is also fine, but it can leak into global attention and sometimes destabilize pretrained VLM priors more than a localized additive conditioning.

**If not, do we still need a time embedding for inference-only scheduling changes?**
No. If you go with mask-source + mixture path and a posterior (p_\theta(x_1\mid x_t,c)), you can treat the denoiser as time-independent and still change **(\kappa(t))** and solver step sizes at inference—because time affects the **update rule**, not the model input. The only reason to add time in this regime is if you empirically see that the same (x_t) distribution at different (t) needs different behavior (which is usually a sign you changed the path/coupling).

---

## Sampling / CTMC Solver

**Do we implement Euler velocity update or hazard-rate tau-leaping (MixtureDiscreteEulerSolver)?**
Start with the **MixtureDiscreteEulerSolver-style hazard/tau-leaping update** (the one the DFM library is built around), because it matches the discrete CTMC story and is what the DFM paper recommends as a reference implementation. Euler-velocity updates can be made to work, but they’re easier to get subtly wrong in categorical spaces (normalization issues, negative probabilities). If you need a stepping-stone for parity, you can implement a “discrete Euler” variant that reduces to “sample some positions to update according to rates,” but keep the interface compatible with the library solver.

**Will we add adaptive step size (h = \min(h, (1-\kappa)/\dot\kappa)) to avoid negative probabilities?**
Yes—this is **non-negotiable** if you want stability and you want to push to low NFE. Discrete CTMC updates can produce invalid (negative) intermediate probabilities if you take too-large steps; adaptive step sizing is the standard fix. Implement it with endpoint guards (avoid (t) too close to 0/1) and with a minimum/maximum step to prevent pathological behavior.

**Should we allow post-training schedule changes ((\kappa_t)) without retraining, as viable for mask-source?**
Yes, as a deliberate feature. With a time-independent denoiser and mixture path, the schedule is mostly a **sampling-time knob** controlling how aggressively you move along the path; it doesn’t require retraining in principle. In practice you still validate because changing schedules changes the distribution of intermediate (x_t) you visit, but this is one of the nicest advantages of the mask-source DFM setup.

**Do we implement NFE skipping when the state doesn’t change (NFE ≤ N claim)?**
Yes, implement “no-change early-exit” and (optionally) “skip steps where hazard is tiny.” In discrete settings, it’s common that an update step produces zero token changes once you’re near convergence; paying full NFE there is wasted. The simplest policy is: if an iteration changes 0 tokens (or changes < ε fraction), stop.

**Do we reuse existing `parallel_decode.decode()` or replace it entirely with a CTMC solver?**
Replace the *control logic* with a CTMC solver loop, but reuse the *fast pieces*: batching, masking, logits computation, and vectorized sampling utilities. Conceptually, `parallel_decode.decode()` is MaskGIT-style commit/remask; CTMC sampling is different enough that you’ll fight it if you try to shoehorn it in. The clean approach is a new `dfm_decode()` that calls the same model forward pass but updates tokens according to CTMC rates.

---

## Inference Strategy

**Are we fully replacing MaskGIT-style commit/remask with CTMC updates, or offering a hybrid mode?**
Do a full CTMC replacement for the mainline path, *and* offer a hybrid mode temporarily for debugging/ablation. Hybrid means “CTMC step + optional corrector/remask pass” or “CTMC for N steps then one MaskGIT-style cleanup,” which helps isolate whether failures come from solver dynamics or model quality. Once CTMC is stable and parity is reached, you can delete the hybrid path or keep it behind a debug flag.

**Do we need explicit token clamping for conditional generation or infilling use cases?**
Yes, implement clamping—it’s cheap and very useful. Even if VLA usually conditions on vision/language, clamping lets you do: partial action constraints, debugging, and “infilling” style experiments where some tokens are fixed. The standard rule is: after every CTMC update step, overwrite clamped positions with the user-specified tokens.

**What default NFE / temperature / schedule values should be exposed and tuned?**
Expose: `dfm_num_steps` (NFE), `dfm_step_size` (or step schedule), `dfm_scheduler` ((\kappa(t)) family), `dfm_temperature` (and optional annealing), and `dfm_use_adaptive_step`. Good starting defaults for parity: NFE ≈ 12 (match current), cosine-like schedule, temperature = 1.0 with optional mild decay. Then try NFE ∈ {8, 6, 4} with adaptive steps + temperature anneal to find the new Pareto curve.

**Do we keep the discrete diffusion secondary re-masking heuristics in any hybrid mode?**
Keep it **as an optional corrector**, not as the core algorithm. Secondary re-masking is a heuristic designed for MaskGIT-style refinement; CTMC dynamics already have a built-in notion of incremental stochastic correction through rates. But as a pragmatic bridge, a “corrector” pass can rescue early rollout issues and gives you an immediate apples-to-apples knob: “DFM core + old remask corrector” vs “pure DFM.”

---

## Interface & Config

**Do we keep `use_discrete_diffusion` and add `use_discrete_flow_matching`, or rename/replace flags?**
Keep both flags initially. You need side-by-side A/B runs with identical configs except the decoding method. Renaming/replacing flags is fine later once DFM is the default, but it’s a needless migration headache during bring-up.

**How do we enforce mutual exclusivity between DFM and discrete diffusion at runtime?**
Enforce it in **three places**: (1) config validation right after parsing, (2) model construction (assert you don’t instantiate both decode heads/paths), and (3) inference entrypoint (hard error if both toggled). Don’t silently “prefer DFM” or “prefer diffusion”—that’s how you get confusing results and unreproducible runs. If you want convenience, allow `decoder_type: {diffusion, dfm}` as a single enum later.

**Which DFM inference hyperparameters should be first-class config options?**
First-class: `dfm_num_steps`, `dfm_scheduler_type`, `dfm_step_size` (or `dfm_time_grid`), `dfm_temperature` (+ anneal), `dfm_adaptive_step`, `dfm_corrector` (bool + steps), and `dfm_clamp_mask` support. Second-tier (can default/hide at first): alternate solver variants, divergence-free coeff, and experimental path families. The rule: expose knobs that change the NFE–quality curve and stability; hide everything else until needed.

- Expose as many hyperparms 

---

## Data & Tokenization

**Is the action tokenization unchanged (bins + `[MASK]`), or do we adjust vocab size/bins?**
Keep it unchanged. DFM doesn’t require changing the tokenization; it requires a **known source distribution** and a **path**, and `[MASK]` is already the perfect “special token source” for discrete generation. Change vocab/binning only if you later discover the discretization itself is the bottleneck, not the decoder dynamics.

**Do we preserve the current action chunk length and stop token behavior?**
Preserve it exactly for parity. The VLA paper uses fixed-length action chunks, and the diffusion decoder is designed around that; DFM drops in cleanly if you keep the same length and “all tokens generated in parallel” contract. If you later want variable length / stop tokens, that’s a separate project (and will tangle with coupling/path design).

---

## Metrics & Evaluation

**What evaluation metrics define “success” (success rate, latency, NFE-quality curve)?**
Primary metric: **task success rate** (exactly what the VLA paper reports), because that’s what matters. Secondary: **NFE–quality curve** (success vs steps) and **latency** (wall-clock per rollout). Tertiary: stability metrics like variance across seeds and “catastrophic failure rate” (e.g., success < X% on any task).

**Which ablations are required (time conditioning, (\kappa(t)) schedule, solver choice)?**
Minimum ablations to trust results: (1) time-independent vs time-conditioned model, (2) cosine vs polynomial (\kappa(t)), (3) solver choice (baseline MixtureDiscreteEuler vs a simpler discrete Euler), and (4) adaptive step on/off. Also do a hybrid corrector ablation if you keep remasking around, because it can mask real solver/model issues.

---

## Performance & Compute

**What are expected NFEs vs the current 12-step MaskGIT loop?**
Start by matching 12 for parity, but expect DFM to remain competitive at **~8** steps, and potentially **~4–6** with adaptive step sizing + temperature anneal once the model is trained with a solver-aligned loss. The real gain is not “magically fewer steps,” it’s that the CTMC update can be more decisive per step without the fragile heuristics around commit/remask thresholds. You’ll only know your curve after running the NFE sweep.

**Where are the hot paths (decoder loop vs CTMC update), and can we keep vectorized decode?**
Hot path remains the **model forward pass** (VLM backbone), not the CTMC math. The CTMC update is cheap if you keep it vectorized: compute logits once per step, compute rates/hazards in bulk, sample updates in parallel across positions. Avoid per-token Python loops; implement updates as tensor ops (masking, multinomial/Gumbel, scatter).

---

## Compatibility & Rollout

**How do we preserve backward compatibility with existing discrete diffusion checkpoints and scripts?**
Architecturally, aim for **no checkpoint shape changes**: same model weights, same action head, and DFM only changes the *training corruption* and *sampling loop*. That means old checkpoints can still load and can even be evaluated under a DFM sampler (it may not perform optimally, but it should run). For scripts, keep `use_discrete_diffusion` default behavior intact; DFM is opt-in behind a flag until you flip defaults later.

**Do we need a compatibility migration strategy if we change flags or config schemas?**
Yes, but keep it simple: support old flags for a while and map them to a new enum internally (or vice versa). Add explicit logging like “decoder=DFM” at run start, and include the decoder type in checkpoint metadata so you don’t accidentally compare apples to oranges. If you later remove diffusion, do it in a major version bump with a clear deprecation window.

---

## Risks & Stability

**Could probabilities go negative or require clamping if step sizes are too large?**
Yes—this is a known failure mode in discrete-time approximations of CTMC probability dynamics. That’s exactly why the DFM guidance emphasizes **safe sampling** via adaptive step sizes; if you ignore it, you’ll see NaNs, invalid distributions, or weird “frozen” decoding. Implement adaptive steps first, and still clamp small negatives to 0 as a last-resort safety net (but treat it as a bug indicator, not “normal behavior”).

**Is there a mismatch risk between masked-CE training and CTMC inference dynamics?**
Yes, and it’s one of the biggest practical risks. Masked CE trains “predict the original token at masked positions,” while CTMC sampling assumes the model meaningfully defines rates/velocities along the chosen path; they’re related but not identical. Mitigation: either (a) use the generalized KL objective that the DFM recipe derives for mixture paths, or (b) keep CE but apply the correct (\dot\kappa/(1-\kappa)) weighting and validate with solver-consistent sampling during training-time eval.

