# Discrete Diffusion vs Discrete Flow Matching for VLA Action Decoding

## Executive summary

The discrete diffusion method used in entity["book","Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies","arxiv 2508.20072"] is a *discrete-time* Markov-chain masking process over action tokens. Training collapses the multi-step chain into a single masked-token cross-entropy objective: sample a mask ratio (diffusion “time”), mask that fraction of action-token positions with `[MASK]`, and train a transformer to predict the original tokens at masked indices. citeturn43view2turn24view1 Inference is a MaskGIT-style parallel refinement loop: start fully masked, predict posteriors for masked positions, sample candidate tokens, then *commit* only the top-confidence fraction according to a cosine schedule (adaptive decoding order). A secondary re-masking heuristic (threshold + “residual drop”) revisits uncertain past commitments to reduce error propagation. citeturn24view1turn24view2turn24view3

Discrete Flow Matching in entity["book","Discrete Flow Matching","neurips 2024 paper"] reframes discrete generation as a *continuous-time* Markov chain (CTMC) that follows a predefined probability path \(p_t\) from a source distribution \(p\) (often “all-mask”) to the data distribution \(q\). The model is trained to predict a *posterior / denoiser* \(p_{1|t}(\cdot \mid x_t)\); from that posterior plus a scheduler \(\kappa_t\), you compute a *probability velocity* / generator that drives CTMC sampling. citeturn16view1turn17view0turn20view2turn22view2turn40view0 In the masked-source, convex-path setting (which matches VLA’s “start from `[MASK]`” design), DFM has a useful property: the denoiser can be time-independent, and the paper notes an NFE bound of \(N\) (number of tokens) for Algorithm 1 if you avoid recomputing when the state doesn’t change. citeturn42view2

For replacing discrete diffusion in the VLA paper with DFM, the key practical point is this: **training looks extremely similar** (both are basically “predict \(x_1\) from a partially corrupted \(x_t\)” with cross-entropy), but **sampling is fundamentally different**: VLA’s sampler is a confidence-ranked “commit & re-mask” heuristic, while DFM sampling is a CTMC simulation derived from a probability velocity / generator. citeturn24view1turn17view0turn22view2turn40view0

## Assumptions and problem framing

I assume the action interface is exactly the one described in the VLA paper: continuous controls are discretized into a fixed local vocabulary (256-bin quantile bins for most dimensions plus a discrete gripper token), then packed into a fixed-length action chunk, and a special `[MASK]` token is appended to the action vocabulary. citeturn24view0turn43view2 I also assume the policy is a unified transformer with bidirectional attention over action positions (no causal constraint) and logits only computed at action positions. citeturn24view0

I treat “replace discrete diffusion with DFM” as: keep the same tokenized action-chunk output space and backbone conditioning on vision+language, but swap the generative modeling framework used for the action decoder:
- Replace discrete-time diffusion reverse steps + heuristic adaptive decoding with a DFM probability-path + CTMC solver.
- Optionally replace the masked CE objective in training with a DFM-specific generalized KL loss (or keep CE first and only replace sampling).

The DFM sources used here come from entity["company","Meta","technology company"] (via entity["organization","Meta AI Research","FAIR"]) and coauthors including entity["people","Yaron Lipman","flow matching author"]. citeturn18view0turn34view2

## Discrete diffusion in Discrete Diffusion VLA

### Core logic

**State space.** An action chunk is a length-\(N\) sequence of discrete tokens. The vocabulary is augmented with a special mask token \(M=[MASK]\). citeturn43view2turn24view0

**Forward/noising process (discrete-time Markov chain).** Each diffusion step independently masks each token with probability \(\beta_t\) (transition matrix \(Q_t\)), otherwise leaves it unchanged. This is the entire corruption mechanism—no token substitution other than to \(M\). citeturn43view2

**Reverse/denoising conditionals.** The reverse conditionals \(p(a_{t-1}\mid a_t, c)\) are defined under multimodal context \(c\) (vision+language). The paper writes a Bayes-rule conditional (Eq. 3) and then simplifies it under masking corruption (Eq. 4), where the model predictive distribution drives the reverse kernel. citeturn43view2turn24view2

**Training objective (collapsed, single-step masked-token CE).** Rather than optimize a full diffusion ELBO, they “collapse the multi-step chain” into: sample a mask ratio (diffusion “time”), mask those indices, and compute cross-entropy loss only on masked indices. citeturn43view2turn24view1

**Inference (adaptive decoding + secondary re-masking).** Their decoding loop is explicitly: initialize all action tokens to `[MASK]`, then run a small fixed number of refinement rounds (12 by default). In each round, predict posteriors for masked positions, sample candidate tokens, then keep (commit) the top-scored fraction determined by a cosine schedule; the rest remain or become masked. citeturn24view1turn23view1turn41view2 The “secondary re-masking” layer then re-masks previously committed tokens if they fail either (i) a step-dependent confidence threshold, or (ii) a residual-drop consistency check relative to the step when the token was first committed. citeturn24view2turn24view3

### Training and sampling pseudocode

```pseudo
# Discrete diffusion VLA training (collapsed masked-token CE)
# Inputs: context c = (vision tokens, language tokens, optional proprio), action chunk tokens a (length N)
# Params: transformer θ, action head proj, mask schedule MaskRatioSchedule(·)

for each minibatch:
    a1 = ground_truth_action_tokens()  # x1 in diffusion notation
    γ  = sample_mask_ratio()           # emulate diffusion time (schedule; e.g., linear/cosine)
    S  = sample_mask_set(N, ratio=γ)   # masked indices
    a_t = a1.clone()
    a_t[S] = MASK_TOKEN

    logits = Transformerθ(c, a_t)      # logits at action positions
    loss = CE(logits[S], a1[S])        # hard-label CE on masked indices only
    backprop_and_update(loss)
```
This corresponds to the paper’s “mask ratio → replace by `[MASK]` → CE on masked indices” training pipeline. citeturn43view2turn24view1

```pseudo
# Discrete diffusion VLA sampling (adaptive decoding + secondary re-masking)
# Inputs: context c, steps R (default 12), cosine schedule for keep ratio, temperature schedule
# Outputs: sampled action chunk a

a = [MASK_TOKEN] * N
committed_step = [-1] * N          # first step when token was committed
ref_conf = [0.0] * N               # cached confidence at commit

for r in 1..R:
    logits = Transformerθ(c, a)
    probs = softmax(logits)

    # propose candidates at masked positions
    for i where a[i]==MASK_TOKEN:
        cand[i] = sample_from(probs[i])  # multinomial or Gumbel-Max

    # score masked positions (examples in paper: max prob or confidence gap)
    score[i] = score_fn(probs[i]) for i where a[i]==MASK_TOKEN

    keep_ratio = cosine_keep_ratio(r)    # monotone schedule
    K = top_fraction_indices(score over masked i, fraction=keep_ratio)

    # commit kept candidates, re-mask the rest
    for i in 0..N-1:
        if i in K:
            a[i] = cand[i]
            if committed_step[i]==-1:
                committed_step[i] = r
                ref_conf[i] = max(probs[i])
        else:
            a[i] = MASK_TOKEN

    # secondary re-masking: threshold + residual-drop checks
    for i where a[i]!=MASK_TOKEN:
        if max(probs[i]) < threshold_tau(r):
            a[i] = MASK_TOKEN
        else if (ref_conf[i] - max(probs[i])) > residual_drop_eta(r):
            a[i] = MASK_TOKEN

return a
```
This is a direct computational reading of the paper’s inference description (mask initialization, cosine schedule, confidence ranking, and the two secondary re-masking checks). citeturn24view1turn24view2turn24view3turn41view2

### Computational characteristics

Training is a single forward/backward pass per batch over the unified transformer, since the diffusion chain is collapsed into one masked-token prediction step. citeturn43view2turn24view1

At inference, NFEs equal the number of refinement steps \(T\) (12 by default), since each refinement round is one forward pass predicting posteriors for all currently masked tokens. citeturn41view2 The paper reports latency/throughput for 12 steps and compares to autoregressive decoding and continuous diffusion baselines. citeturn41view2

## Discrete flow matching for discrete sequences

### Core logic

**CTMC and probability velocity.** DFM models discrete sequences as a continuous-time discrete Markov chain over \(t\in[0,1]\). Sampling is defined by updating each coordinate independently using a probability velocity \(u_t^i\), with a small step \(h\), as:
\[
X_{t+h}^i \sim \delta_{X_t^i}(\cdot) + h\,u_t^i(\cdot, X_t)
\]
and \(u_t\) must satisfy normalization/nonnegativity constraints to be a valid PMF update (their Eq. 12–13 framing). citeturn16view1turn17view0 They package this into “Algorithm 1 Flow Matching sampling”. citeturn17view0

**Source distribution and couplings.** For discrete generation they consider source \(p\) as either (i) all-mask sequences \(p=\delta_m\), or (ii) uniform over sequences, and state they focus mainly on (i). citeturn18view0 They also define unconditional coupling (“U-coupling”: \(X_0\sim p\) independent of \(X_1\sim q\)) and a conditional coupling (“C-coupling”: partially mask \(X_1\) to create \(X_0\) that retains some conditioned tokens). citeturn18view0

**Probability paths.** DFM chooses a probability path \(p_t\) that interpolates between \(p\) and \(q\). A key tractable case is a convex mixture *per token* between \(x_0\) and \(x_1\):
\[
p_t(x^i\mid x_0,x_1) = (1-\kappa_t)\,\delta_{x_0}(x^i) + \kappa_t\,\delta_{x_1}(x^i)
\]
with scheduler \(\kappa_t\) increasing from 0 to 1. citeturn15view0turn22view2 The framework also supports more general “mixture of conditional distributions” paths (their Eq. 8) and even adding uniform noise (their Eq. 10). citeturn15view0turn16view0

**Learning target: the probability denoiser/posterior.** For the convex mixture path, the forward-time generating velocity can be written directly in terms of the “probability denoiser” \(p_{1|t}(x^i\mid z)\) and \(\kappa_t\):
\[
u_t^i(x^i, z)=\frac{\dot{\kappa}_t}{1-\kappa_t}\left[p_{1|t}(x^i\mid z)-\delta_z(x^i)\right]
\]
(the paper’s Eq. 24). citeturn22view2 This is exactly the conceptual bridge: **train \(p_{1|t}\), then compute a velocity/generator, then sample**.

**Training objective.** They train the posterior(s) \(\hat w_t\) (commonly just the denoiser \(p_{1|t}\)) using a negative log-likelihood / cross-entropy objective:
\[
L(\theta) = - \sum_{i} \mathbb{E}_{t,(X_0,X_1),X_t}\left[\log p_{1|t}(X_1^i\mid X_t; \theta)\right]
\]
which is their “loss in equation 28” specialization for one posterior. citeturn20view2turn18view0

**Implementation-level view via Meta’s `flow_matching` library.** The official `flow_matching` docs define:
- `MixtureDiscreteProbPath`: a factorized discrete path that stays at \(X_0\) until a random flip time then flips to \(X_1\), governed by a scheduler. citeturn39view0turn35view0
- `MixturePathGeneralizedKL`: a generalized KL loss for discrete FM with an explicit \(\dot\kappa_t/(1-\kappa_t)\) weighting and posterior terms. citeturn38view1turn35view0
- `MixtureDiscreteEulerSolver`: a CTMC simulator for the discrete path, with a tau-leaping-like per-coordinate update that uses posterior sampling and hazard rates. citeturn40view0turn35view0

### Training and sampling pseudocode

Below are two equivalent ways to express DFM sampling: (A) the paper’s “probability velocity Euler update” form, and (B) the library’s CTMC solver form.

```pseudo
# DFM training (posterior / probability denoiser), paper Eq. 28

for each minibatch:
    x1 = data_action_tokens()               # target sample X1 ~ q
    x0 = sample_source_tokens()             # e.g., all MASK if p = δm

    t  = Uniform(0, 1 - ε)                  # often avoid t=1 for stability
    x_t = sample_path(x0, x1, t; kappa)     # per-token mixture via κ_t

    logits = Modelθ(context=c, x=x_t, t=t)  # predicts posterior p_{1|t}(.|x_t,c)
    loss = CrossEntropy(logits, targets=x1) # sum/mean over token positions
    backprop_and_update(loss)
```
This is the DFM paper’s Eq. 28 training loop in standard “denoiser training” form. citeturn20view2turn22view2turn35view0

```pseudo
# DFM sampling (paper Algorithm 1; probability-velocity Euler update)

x = sample_source_tokens()              # e.g., all MASK
h = 1 / nfe
for t in {0, h, 2h, ..., 1-h}:
    # get posterior p_{1|t}(.|x,c)
    probs = softmax(Modelθ(c, x, t))

    # compute velocity u_t^i via Eq. 24 and κ_t
    for each token position i:
        u_i = (kappa_dot(t) / (1 - kappa(t))) * (probs[i] - one_hot(x[i]))
        x[i] ~ Categorical(one_hot(x[i]) + h * u_i)   # requires h small enough

return x  # sample near t=1
```
This matches the paper’s Algorithm 1 update rule \(X^i_{t+h}\sim \delta_{X^i_t}+h u^i_t(\cdot,X)\) and the convex-path velocity formula. citeturn17view0turn22view2turn16view1

```pseudo
# DFM sampling (library-style CTMC tau-leaping; MixtureDiscreteEulerSolver)

x = sample_source_tokens()              # e.g., all MASK
for t in {0, h, 2h, ..., 1-h}:
    probs = softmax(Modelθ(c, x, t))    # posterior p_{1|t}

    for each token position i:
        # sample an auxiliary x1^i from posterior (library step)
        x1_i ~ Categorical(probs[i])

        # define conditional velocity û_t^i based on κ_t and x1_i
        # (optionally add divergence-free correction term)
        u_cond = velocity_conditional(i, x_current=x[i], x1_i, t)

        λ = sum_{v != x[i]} u_cond[v]               # hazard rate
        if Uniform(0,1) <= 1 - exp(-h * λ):
            x[i] ~ Categorical(u_cond normalized over v != x[i])
        else:
            x[i] stays

    # optional: enforce conditioning by clamping certain indices
    x = clamp_conditioned_tokens(x)

return x
```
This is a faithful re-expression of the solver algorithm documented for `MixtureDiscreteEulerSolver` (posterior-sample, compute hazard, flip with probability \(1-e^{-h\lambda}\), then sample the new state). citeturn40view0turn39view0

### Practical stability levers DFM gives you “for free”

**Safe sampling / step-size control.** The DFM paper explicitly warns that large \(h\) can make the per-step PMF \(\delta + h u\) go negative, requiring clamping and accumulating error, and proposes an adaptive step size with a simplified form \(h_{\text{adaptive}}=\min(h, (1-\kappa_t)/\dot\kappa_t)\) for the denoiser parameterization. citeturn42view2 This is exactly the stability bug you’ll hit first if you naïvely port DFM.

**Conditioning in discrete sampling.** For conditional generation, they describe enforcing a mask \(I\) by replacing/clamping conditioned tokens after each update step. citeturn42view2 For VLA, you’ll mostly condition via vision+language inputs (not via partially fixed action tokens), but this mechanism is valuable for ablations like “trajectory infilling” or “plan prefix fixed, remainder generated”.

**Schedule tuning after training.** They describe a “post training scheduler change” procedure and note that for mask modeling (source \(p=\delta_m\)), the denoiser can be time-independent, so the posterior may not be affected by scheduler change. citeturn42view2 In practice: you can often tune \(\kappa_t\) and step size at inference without retraining when you stay in the convex mask-path regime.

## Rigorous comparison of algorithmic differences

### Side-by-side attribute table

| Attribute | Discrete diffusion in the VLA paper | Discrete flow matching (DFM) |
|---|---|---|
| Underlying process | Discrete-time Markov chain (masking diffusion) with per-step transition matrices \(Q_t\). citeturn43view2 | Continuous-time Markov chain (CTMC) driven by a probability velocity / generator. citeturn16view1turn17view0 |
| “Forward” mechanism | Corrupt token \(\to [MASK]\) with prob \(\beta_t\); otherwise unchanged (no other noise). citeturn43view2 | Choose a probability path \(p_t\) from source \(p\) to target \(q\) (convex endpoint mixing; optional uniform-noise mixtures, etc.). citeturn15view0turn16view0turn18view0 |
| What the network predicts | Predictive distribution over tokens at action positions (used in Bayes reverse kernel). citeturn43view2turn24view0 | Posterior / probability denoiser \(p_{1|t}(\cdot\mid x_t)\) (and optionally other posteriors), which is then converted into a velocity. citeturn22view2turn20view2turn39view0 |
| Training objective | Masked cross-entropy only on masked indices (collapsed diffusion training). citeturn43view2turn24view1 | Cross-entropy denoiser training (paper Eq. 28), or generalized KL loss with \(\dot\kappa_t/(1-\kappa_t)\) weighting (library). citeturn20view2turn38view1turn35view0 |
| Time/schedule usage in training | Sample a mask ratio “emulating diffusion time”; no explicit requirement to feed \(t\) if mask ratio is implicit. citeturn24view1turn43view2 | Typically sample \(t\) explicitly and pass it to the model; notebooks and solvers assume the model is called with \((x,t)\). citeturn35view0turn40view0 |
| Sampling procedure | Heuristic parallel refinement: predict posteriors, sample candidates, keep top fraction by confidence (cosine schedule), re-mask others; add secondary re-masking (threshold + residual-drop). citeturn24view1turn24view2turn24view3turn23view1 | Principled CTMC simulation: compute probability velocity (e.g., Eq. 24), then Euler/tau-leaping updates; supports step-size control and corrector/divergence-free terms. citeturn17view0turn22view2turn42view2turn40view0 |
| Error correction mechanism | Only via re-masking + later re-sampling (secondary re-masking). citeturn24view2turn24view3 | Tokens can change directly through the CTMC generator dynamics (flips/jumps), optionally enhanced by corrector sampling. citeturn40view0turn21view0 |
| Conditioning handling | Condition through unified transformer inputs; sampling logic is purely posteriors + confidence ranking. citeturn24view0turn24view1 | Condition through model inputs; additionally supports explicit “token clamping” after each solver step for infilling/prefix constraints. citeturn42view2turn40view0 |
| NFEs / compute at inference | NFE = number of refinement steps \(T\) (12 default) independent of sequence length; paper reports latency and NFE comparisons. citeturn41view2turn24view1 | Depends on solver discretization; with masked source + time-independent denoiser, paper notes NFE can be bounded by \(N\) if you skip recomputation when no token changes. citeturn42view2 |
| Stability knobs | Mostly heuristic: temperature schedule, confidence ranking, re-masking thresholds. citeturn41view0turn24view2turn24view3 | Explicit: step size \(h\), adaptive safe sampling, scheduler \(\kappa_t\), divergence-free/corrector terms. citeturn42view2turn40view0turn38view1 |
| Reported “speed–quality” evidence (in-source) | VLA reports a speed–quality tradeoff sweep over denoising steps and gives concrete latency/throughput for 12 steps on H800. citeturn41view2 | DFM paper emphasizes scheduler/corrector tuning is pivotal and provides “safe sampling” guidance; direct robotics speed numbers are not provided there. citeturn21view0turn42view2 |

### Key algorithmic differences that matter for your replacement

**Diffusion’s reverse kernel vs DFM’s velocity/generator.** In VLA’s discrete diffusion, the conceptual object is a reverse transition \(p(a_{t-1}\mid a_t,c)\) derived from Bayes rule under the masking corruption; the implementation then departs from strict reverse sampling and uses a confidence-ranked commit schedule. citeturn43view2turn24view1 In DFM, the core object is a probability velocity \(u_t\) or equivalent generator for a CTMC that *provably generates the predefined path* when discretized with small \(h\). citeturn16view1turn17view0turn16view3

**Training objectives are closer than they look.** VLA’s training is masked-token CE on masked indices. citeturn43view2turn24view1 DFM’s simplest “denoiser training” loss is also cross-entropy to predict \(X_1\) from \(X_t\) (paper Eq. 28). citeturn20view2turn22view2 The difference is that DFM also offers a generalized KL objective with explicit scheduler-dependent weighting and terms derived from the CTMC formulation. citeturn38view1turn40view0

**Sampling is where you’ll actually feel the swap.** VLA’s sampler is effectively “decode easy tokens early” by confidence and only revises via re-masking. citeturn24view1turn24view2turn23view1 DFM sampling lets tokens jump directly according to learned posteriors converted into a velocity and may thus correct errors without the “mask again” heuristic, at the cost of carefully controlling step size to avoid invalid probabilities. citeturn22view2turn42view2turn40view0

**Scheduler design becomes a first-class interface.** VLA uses a cosine keep-ratio schedule for “how many tokens remain masked” during refinement at inference. citeturn24view1turn23view1 DFM centers a scheduler \(\kappa_t\) and explicitly states scheduler choice is pivotal in experiments. citeturn21view0turn20view2 The `flow_matching` docs / notebooks demonstrate polynomial schedulers (e.g., \(\kappa_t=t^2\)) and expose mechanisms to invert \(\kappa\) or change schedules. citeturn35view0turn39view1turn42view2

## Migration plan for swapping discrete diffusion in VLA to discrete flow matching

### Migration blueprint

**Phase zero: lock a baseline.** Reproduce the published discrete diffusion VLA decoding behavior (12 refinement steps, cosine schedule, secondary remasking) and record: success-rate metrics, token-level masked accuracy, NFE, and latency/throughput. citeturn41view2turn24view1turn37view1 The paper provides reported inference latency/throughput and NFE comparisons you can use as reference targets. citeturn41view2

**Phase one: replace sampling first (lowest risk).** Keep training unchanged (still masked-token CE), but replace the inference loop with a DFM CTMC solver that uses your model’s token posteriors as the denoiser \(p_{1|t}\), then converts to a velocity (Eq. 24) and performs CTMC updates. citeturn22view2turn40view0turn43view2  
Why this is low risk: your model already outputs token posteriors; VLA’s training already matches “predict original token from a masked/noisy input”. citeturn43view2turn24view1

**Phase two: adopt DFM-native training loss.** Replace the masked CE with the generalized KL loss (or the paper’s Eq. 28 denoiser training over all tokens, with consistent path sampling). The official `flow_matching` docs give the explicit loss formula and an example training loop. citeturn38view1turn35view0turn20view2

**Phase three: exploit DFM-only knobs.** Add (a) safe/adaptive step sizing, (b) divergence-free / corrector sampling, and (c) scheduler tuning after training to push quality-speed tradeoffs. citeturn42view2turn40view0turn21view0

### Required model and architecture adjustments

**Time input (recommended, sometimes optional).** Meta’s reference implementations and solvers treat the discrete denoiser model as \(f_\theta(x_t, t)\) (the notebook MLP explicitly embeds \(t\) and concatenates it into the network). citeturn35view0 For a transformer VLA, you have two viable choices:

- **Minimal change (mask convex path, “time-implicit”).** If you stick to masked-source \(p=\delta_m\) and convex mixture paths, DFM proves a time-independence result for the denoiser in that regime. citeturn22view1turn42view2 In practice you can attempt to omit explicit \(t\) conditioning and rely on the masked pattern itself as the “time signal” (similar to the current VLA). citeturn43view2turn24view1  
  Practical downside: you’ll have less control if you later move beyond the convex mask path (e.g., add uniform noise mixtures).

- **Proper DFM interface (recommended).** Add an explicit time embedding input to the action decoder: a scalar \(t\) (or \(\kappa_t\)) embedded via an MLP/sinusoid and injected either (i) as an extra “time token” appended to the action token block, or (ii) as an additive bias to action-token embeddings. This matches the design expectations of DFM solvers and the reference notebook. citeturn35view0turn40view0

**Action head stays mostly identical.** DFM denoiser training and generalized KL loss both operate on posterior probabilities over the vocabulary for each token position. If your current action head already maps hidden states to \(K\)-way logits per action position (256 bins + `[MASK]`), you can reuse it and interpret softmax(logits) as \(p_{1|t}(\cdot\mid x_t,c)\). citeturn24view0turn38view1turn43view2

### Code-level changes you actually need

Below is the minimal set of code changes conceptually; file names will differ across codebases, but the hooks are always the same.

**Replace “mask ratio” corruption with “path sampling”.** Today, VLA training does: sample mask ratio → set those indices to `[MASK]`. citeturn24view1turn43view2 DFM wants: sample \(t\) → sample \(x_t\sim p_t(\cdot\mid x_0,x_1)\). For the convex mask path with \(x_0=\) all-mask, these are almost the same operation, but you’ll want the *same* scheduler \(\kappa_t\) at training and sampling if you plan to use DFM’s theory and solvers. citeturn22view2turn39view0turn35view0

**Replace loss function.**
- Baseline swap (phase one): keep CE-on-masked-index loss, so training matches the original VLA. citeturn43view2turn24view1
- DFM-native (phase two): implement `MixturePathGeneralizedKL` (or equivalent) as your criterion; the official docs provide the explicit per-token loss \(\ell_i(x_1,x_t,t)\) and its dependence on \(\dot\kappa_t/(1-\kappa_t)\). citeturn38view1turn39view0

**Replace inference loop.** Remove “confidence rank → keep fraction → re-mask” and instead implement a CTMC solver:
- Compute posterior probabilities \(p_{1|t}\) from the model.
- Convert to velocity / generator using Eq. 24 (or use the solver’s conditional-velocity construction).
- Update tokens via Euler/tau-leaping.
- Optionally clamp any conditioned indices after each step (rare for VLA, useful for infilling experiments). citeturn22view2turn40view0turn42view2

If you want a known-good reference for code structure, the official discrete notebook demonstrates the exact object graph: `MixtureDiscreteProbPath` + `MixturePathGeneralizedKL` + `MixtureDiscreteEulerSolver`. citeturn35view0turn39view0turn38view1turn40view0

### Training hyperparameters and schedules to try

Grounded in what the papers and official implementation expose as important:

**Schedulers \(\kappa_t\).** DFM explicitly states scheduler choice is pivotal in practice. citeturn21view0turn20view2 The official notebook trains a discrete FM model with a polynomial scheduler (\(\kappa_t=t^2\) in the notebook narrative / `PolynomialConvexScheduler(n=2.0)`). citeturn35view0turn39view1 For your VLA setting, I’d try:

- \(\kappa_t=t\) (linear): simplest baseline.
- \(\kappa_t=t^2\), \(t^3\): pushes more “action” later; often improves stability near \(t=0\) and may match “easy-first” behavior.
- Cosine-shaped \(\kappa_t\): to align with the VLA’s cosine schedule intuition (even if the exact mapping differs). citeturn23view1turn39view1

**Time sampling.** Start with uniform \(t\in[0,1-\varepsilon]\). The official notebook uses a small \(\varepsilon\) and samples \(t\) scaled by \((1-\varepsilon)\). citeturn35view0

**Step count / step size.** VLA found 12 steps a good knee point for speed–quality. citeturn41view2turn23view1 DFM solvers expose step size \(h\) (or NFE). citeturn40view0turn35view0 For parity experiments:
- Fix NFE in {8, 12, 16, 24, 32} and plot success vs latency.
- For stability, implement safe/adaptive step sizes (next bullet).

**Safe sampling (non-negativity).** Implement the adaptive step size rule from the DFM paper (their Eq. 30 for denoiser parameterization). citeturn42view2 If you skip this, you’ll eventually hit negative PMFs when \(\dot\kappa/(1-\kappa)\) spikes near \(t\to 1\).

**Temperature / stochasticity.** VLA’s ablation suggests a linear decay temperature from 1→0 improved success relative to always-argmax or fixed temperature. citeturn41view0turn23view1 DFM solvers also have knobs (e.g., divergence-free coefficient, corrector steps); if you keep DFM sampling stochastic, you may not need extra ad-hoc Gumbel sampling on top.

### Evaluation metrics and ablations for validating the swap

You need to answer two questions: “did we preserve task performance?” and “did we improve the speed/robustness tradeoff?”

**Primary task metrics (match the VLA paper).** Use the same suite-level success rates and environment-level scores the VLA paper reports (e.g., LIBERO success rate; SimplerEnv variants). citeturn43view0turn41view2

**Efficiency metrics.**
- NFEs and wall-clock latency per action chunk, matching the VLA paper’s analysis framing. citeturn41view2
- Throughput (Hz), and a speed–quality curve by sweeping NFE / step size. citeturn41view2turn40view0

**Model-internal diagnostics (fast to compute, great for debugging).**
- Token-level accuracy on masked positions / all positions under the chosen path sampler \(x_t\sim p_t\).
- Average number of token flips per sample during DFM sampling (should drop as \(t\to 1\)).
- Confidence calibration curves: compare predicted token probabilities vs empirical correctness (VLA already leans heavily on confidence). citeturn24view1turn24view2

**Ablation grid that isolates causes.**
- **Sampler-only swap:** same trained checkpoint, compare (A) VLA adaptive decoding vs (B) DFM CTMC solver at matched NFE. citeturn24view1turn40view0
- **Loss swap:** CE (VLA) vs generalized KL (DFM) with same sampler. citeturn43view2turn38view1
- **Scheduler swap:** linear vs polynomial vs cosine-shaped \(\kappa_t\), without retraining if you stay in masked-source/time-independent regime (test the “post training scheduler change” claim empirically). citeturn42view2turn39view1
- **Drop secondary re-masking:** DFM sampling should already allow correction via token flips; test whether VLA’s secondary re-masking is redundant, harmful, or still helpful. citeturn24view2turn40view0
- **Safe-sampling on/off:** confirm that adaptive step sizing improves stability and final performance at larger step sizes. citeturn42view2turn40view0

### Pitfalls and debugging tips

**The first real failure mode is invalid probabilities.** The DFM paper calls out that too-large step size can produce negative probabilities in \(\delta + h u\), requiring clamping and accumulating global error; implement their adaptive step size early. citeturn42view2turn40view0

**Beware the \(t\to 1\) singularity.** Both the DFM loss (via \(\dot\kappa/(1-\kappa)\)) and velocity formula can blow up near \(t=1\). The reference notebook avoids sampling exactly at 1 by using an \(\varepsilon\). citeturn35view0turn38view1turn42view2

**Don’t accidentally change the corruption distribution.** VLA’s corruption is “token → mask with probability, else unchanged.” citeturn43view2 DFM supports other paths (e.g., add uniform noise). citeturn16view0turn18view0 If you mix these, you’re no longer doing an apples-to-apples replacement, and you should expect to retune schedules and maybe add explicit time conditioning.

**Conditioning semantics are different.** VLA conditions on vision+language in the transformer and uses confidence ranking to decide *order* of committing tokens. citeturn24view0turn24view1 DFM has an explicit “clamp conditioned tokens” mechanism meant for infilling/prefix conditioning. citeturn42view2turn40view0 If you introduce action-prefix conditioning experiments, be strict about applying clamps after each solver step.

**Expect different “error correction” behavior.** VLA corrects errors by re-masking committed tokens (secondary re-masking). citeturn24view2turn24view3 DFM corrects by allowing jumps to new token states under the velocity dynamics. citeturn40view0turn22view2 So if you keep VLA’s secondary re-masking on top of DFM sampling, you may double-inject stochasticity or destabilize. Treat “secondary remask” as an ablation, not a default carry-over.

## Primary sources and official resources

The VLA paper’s official implementation is hosted on entity["company","GitHub","code hosting platform"], and the paper links directly to its repository. citeturn23view1turn25view0 The official `flow_matching` library and its documentation (including discrete DFM losses, schedulers, and CTMC solvers) are also on GitHub and have full API docs and runnable notebooks. citeturn31view0turn33view0turn35view0turn38view1turn40view0

```text
Discrete Diffusion VLA paper (HTML): https://arxiv.org/html/2508.20072v3
Discrete Diffusion VLA official repo: https://github.com/Liang-ZX/DiscreteDiffusionVLA/tree/libero

Discrete Flow Matching (NeurIPS 2024 PDF): https://proceedings.neurips.cc/paper_files/paper/2024/file/f0d629a734b56a642701bba7bc8bb3ed-Paper-Conference.pdf
flow_matching library repo: https://github.com/facebookresearch/flow_matching
flow_matching documentation: https://facebookresearch.github.io/flow_matching/
Discrete DFM notebook (official): https://facebookresearch.github.io/flow_matching/notebooks/2d_discrete_flow_matching.html
```

Key primary citations used for method definitions and equations:
- VLA discrete diffusion formalization (forward masking chain, Bayes reverse, masked CE objective). citeturn43view2turn24view1
- VLA inference algorithm (adaptive decoding + secondary re-masking) and default step count. citeturn24view1turn24view2turn24view3turn23view1
- VLA inference efficiency and latency/NFE table. citeturn41view2
- DFM CTMC framework, Algorithm 1, convex path, denoiser-based velocity (Eq. 24), denoiser training (Eq. 28). citeturn17view0turn18view0turn20view2turn22view2turn16view1
- DFM safe sampling, conditioning via clamping, NFE bound for masked modeling, schedule-change guidance. citeturn42view2turn21view0
- Official `flow_matching` API docs for `MixtureDiscreteProbPath`, `MixturePathGeneralizedKL`, and `MixtureDiscreteEulerSolver` (loss formula + solver update). citeturn39view0turn38view1turn40view0turn35view0