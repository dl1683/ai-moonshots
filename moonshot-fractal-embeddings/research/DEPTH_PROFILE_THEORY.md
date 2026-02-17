# First-Principles Theory: Why Representation Quality Follows Bell-Shaped Depth Profiles

## Status: DRAFT v2 — Updated with cross-paradigm experiments (Feb 17, 2026)

## The Empirical Facts (UPDATED)

### Original finding (old transformers only)
logit(Q_norm(x)) follows Gaussian-in-logit across Pythia/OLMo-2/OPT/GPT-2/Cerebras-GPT.
R2=0.78, shape rho=0.84.

### NEW: Cross-paradigm finding (Feb 17, 2026)
The bell shape is ARCHITECTURE-SPECIFIC:
- SSM (Mamba): Strong bell (4/4 datasets, delta=+0.33, peak at ~40-60% depth)
- Transformer (Qwen3): Flat/monotonic profiles (delta~0, no consistent bell)
- Hybrid (Falcon-H1): Follows transformer pattern
- Reasoning (R1-Distill): Declining profiles

### NEW: Residual surgery finding (causal, Feb 17, 2026)
Sweeping residual strength alpha from 0 to 1 on Qwen3-0.6B:
- alpha 0-0.75: ALL profiles decline (no extraction possible)
- alpha = 1.0: Complex profile with late peak
- PHASE TRANSITION at critical alpha* between 0.75 and 1.0
- Beta (curvature) causally controlled: rho=-0.90, p=0.037

**Updated question**: WHY do different architectures produce different depth profiles?
What determines the profile shape, and is there a universal law connecting
architecture → depth profile → capabilities?

---

## The Competition Model

### Setup
- Deep network with L layers, relative depth x = l/L in [0, 1]
- Representation h_l at layer l
- Class separability: S(x) = log(between-class variance / within-class variance)
- Quality: Q_norm(x) = sigma(S(x)) [logistic link]

### Two Competing Forces

**Force 1: Feature Extraction (E)**
Each layer disentangles class-relevant features from raw input structure.
- Strongest in early layers (most structure remaining to extract)
- Diminishes with depth (diminishing returns)
- Model: E(x) = a(1 - x) where a = extraction capacity

**Force 2: Task Specialization (T)**
Each layer compresses representations toward the pre-training objective.
- Weakest in early layers (far from output, weak gradient signal)
- Strongest in late layers (close to output, strong gradient)
- Model: T(x) = bx where b = specialization pressure

### Derivation

The rate of change of separability:

    dS/dx = E(x) - T(x) = a(1 - x) - bx = a - (a + b)x

Integrating from x = 0 with initial separability S_0:

    S(x) = S_0 + ax - (a + b)x^2/2

Completing the square:

    S(x) = [S_0 + a^2/(2(a+b))] - [(a+b)/2] * (x - a/(a+b))^2

This IS our empirical model:

    logit(Q_norm(x)) = b_0 - beta * (x - mu)^2

where:
- b_0 = S_0 + a^2/(2(a+b))    [peak quality; dataset-dependent via S_0]
- beta = (a+b)/2                [curvature; sum of extraction + specialization rates]
- mu = a/(a+b)                  [peak position; ratio of extraction to specialization]

### Physical Meaning of Parameters

| Derived param | Physical meaning | Depends on |
|---------------|-----------------|------------|
| mu = a/(a+b)  | Where extraction balances specialization | Architecture, training |
| beta = (a+b)/2 | Total "force" acting on representations | Model capacity + training |
| b_0           | Maximum achievable separability | Dataset difficulty + model |

### Scale Dependence

**Why does mu depend on log(C/N)?**

The specialization pressure b depends on compute-per-parameter (C/N):
- More training per parameter = more specialization
- b = b_base * phi(C/N) where phi is increasing

The extraction capacity a depends on model size N:
- Larger models extract features more efficiently
- a = a_base * psi(N) where psi is increasing

Peak position: mu = a/(a+b) = psi(N) / [psi(N) + b_base*phi(C/N)/a_base]

For slowly-varying (logarithmic) scaling:
    mu ~ mu_0 + mu_1 * log(C/N)

This matches our empirical mu_1 parameter.

---

## The Deep Insight: DPI vs Learned Structure

The Data Processing Inequality (DPI) says:
    I(X; Y) >= I(h_1; Y) >= I(h_2; Y) >= ... >= I(h_L; Y)

Information can only DECREASE through processing.

But our quality Q(x) INCREASES in early layers. How?

**Resolution**: Quality (kNN accuracy) depends on GEOMETRIC SEPARABILITY, not mutual information. A representation can have high I(h; Y) but terrible kNN accuracy if classes are entangled.

The key insight:
> **Learned transformations trade information for structure.**
> Early layers sacrifice some information but dramatically increase geometric separability.
> Late layers sacrifice both information AND structure for pre-training specialization.

The bell shape is the signature of this tradeoff:
- Information: monotonically decreasing (DPI)
- Separability: first increasing (disentanglement), then decreasing (over-specialization)
- Quality ~ separability, hence bell-shaped

This connects to rate-distortion theory: each layer achieves a point on the rate-distortion curve. The "rate" (information content) decreases, but the "distortion" (for downstream tasks) first decreases then increases.

---

## Testable Predictions

### Prediction 1: Random Networks Are Flat
Random (untrained) networks have a = b = 0 (no learned extraction or specialization).
- PREDICTION: Q(x) ~ Q_chance for all x
- TEST: Run kNN on random model representations

### Prediction 2: Overtrained Models Peak Earlier
As training progresses, b increases (more specialization), so:
    mu = a/(a+b) DECREASES
- PREDICTION: Later training checkpoints should show peak shifting to earlier layers
- TEST: We already see "degradation" in late training. Check if peak actually shifts.

### Prediction 3: Wider Models Have Sharper Peaks
Larger models have larger a AND b, so beta = (a+b)/2 is larger.
- PREDICTION: Depth profiles should be MORE sharply peaked for larger models
- TEST: Compare beta across model sizes within a family

### Prediction 4: Harder Tasks Peak Later
For harder downstream tasks, extraction is harder (effectively smaller a relative to b):
    mu = a/(a+b) is smaller? NO -- harder tasks need MORE extraction, so the network
    needs to go deeper before enough structure is extracted.
- Actually: harder tasks have lower S_0, so the profile is shifted down, not necessarily
  shifted in peak position. Need to think more carefully.
- ALTERNATIVE: Different tasks may probe different features, changing the effective a.

### Prediction 5: Non-Residual Architectures Have Different Curvature
Residual connections maintain gradient flow, making T(x) more uniform across depth.
Without residual connections, T(x) is much stronger near the output.
- PREDICTION: Non-residual networks should show more asymmetric (skewed) profiles
- TEST: Compare architectures with/without residual connections

### Prediction 6: Intermediate Checkpoints Trace the Competition
During training, extraction (a) develops first, specialization (b) grows later.
- PREDICTION: Early in training, the peak is close to x=1 (extraction-dominated)
- As training progresses, peak shifts toward smaller x (specialization grows)
- The TRAJECTORY of the peak through training maps the extraction/specialization dynamics

---

## What Would Make This Nobel-Worthy

### Level 1: Empirical regularity (what we have now)
"Bell-shaped depth profiles are consistent across architectures" -- interesting, not Nobel

### Level 2: First-principles derivation (this document)
"The bell shape follows from competition between extraction and specialization" -- novel, publishable, not Nobel

### Level 3: Quantitative predictions verified (next step)
"The theory makes 6 quantitative predictions, all verified" -- strong, top-venue, not Nobel

### Level 4: Connection to fundamental limits (the goal)
"The extraction/specialization competition is a NECESSARY consequence of information processing
in hierarchical systems. The Gaussian-in-logit form is the UNIQUE fixed point of [some principle].
This implies fundamental efficiency bounds on representation learning."
-- THIS is Nobel territory. We need to show:
  a) The bell shape is not just empirical but mathematically NECESSARY
  b) It implies practical consequences (optimal layer selection, minimum model size)
  c) It connects to a broader theory (information geometry, thermodynamics of learning)
  d) It generalizes beyond transformers to ANY hierarchical information processor

### Level 5: Create a new field
"The Thermodynamics of Representation Quality" -- where the competition model becomes the
foundation for understanding ALL representation learning, the way Boltzmann statistics
underlies thermodynamics.

---

## NEW: Residual Phase Transition Theory (Feb 17, 2026)

### The Signal Propagation Argument

Consider a transformer with L layers and residual connections scaled by alpha.
Each layer l has the form:

    h_{l+1} = alpha * h_l + f_l(h_l)

where f_l is the nonlinear branch (attention + MLP).

**The Jacobian of one layer:**

    J_l = alpha * I + J_{f_l}

where J_{f_l} = d f_l / d h_l is the Jacobian of the nonlinear branch.

**Signal propagation through L layers:**

The end-to-end Jacobian is J_total = prod_{l=1}^{L} J_l.

For the signal to "survive" (not collapse to noise), we need:

    ||J_total|| > epsilon

For the case where each layer's Jacobian has spectral radius rho_l:

    prod_{l=1}^{L} rho_l > epsilon

**Simplifying assumption**: If all layers have similar Jacobian spectral radius rho:

    rho^L > epsilon
    L * log(rho) > log(epsilon)
    rho > epsilon^{1/L}

For the residual layer: rho = alpha + (1-alpha) * rho_f, where rho_f is the
spectral radius of the nonlinear branch.

### Derivation of alpha*

For signal survival: rho > epsilon^{1/L}

    alpha + (1-alpha) * rho_f > epsilon^{1/L}

Solving for alpha:

    alpha > (epsilon^{1/L} - rho_f) / (1 - rho_f)

This is the critical alpha*:

    alpha* = (epsilon^{1/L} - rho_f) / (1 - rho_f)

For large L, using epsilon^{1/L} = exp(log(epsilon)/L) ~ 1 + log(epsilon)/L:

    alpha* ~ (1 + log(epsilon)/L - rho_f) / (1 - rho_f)
           = 1 + log(epsilon) / (L * (1 - rho_f))
           = 1 - |log(epsilon)| / (L * (1 - rho_f))

So: **alpha* = 1 - C/L** where C = |log(epsilon)| / (1 - rho_f).

This gives: **(1 - alpha*) ~ 1/L**

### Comparison with Empirical Fit

Empirically we found: alpha* = 1 - 6.2 * L^(-1.19)

The theory predicts: alpha* = 1 - C * L^(-1.0)

The discrepancy (exponent 1.19 vs 1.0) could arise from:
1. rho_f depends weakly on L (wider models have different nonlinear branch)
2. The "all layers similar" assumption breaks down (early/late layer heterogeneity)
3. The epsilon threshold is not sharp but depends on the evaluation metric
4. Only 3 data points — the exponent 1.19 has large uncertainty

### Predictions from the Theory

**Prediction A**: The per-layer information budget (1-alpha*) scales as ~1/L.
This means deeper models are MORE fragile — each layer has less "room" for
information loss.

**Prediction B**: alpha* depends on rho_f (the nonlinear branch spectral radius).
Models with stronger nonlinear branches (larger rho_f) should have HIGHER alpha*
because the nonlinear contribution is already large and the residual needs to
compensate less. BUT this contradicts our data where Qwen3 (28L, alpha*=0.90)
has higher alpha* than OLMo-2 (16L, alpha*=0.78). This is consistent only if
the depth effect dominates the rho_f effect.

**Prediction C**: At initialization (random weights), rho_f is determined by the
initialization scheme. For models initialized at the "edge of chaos" (rho_f close
to 1 - 1/L), alpha* should be very low — even small residual connections suffice.
After training, rho_f changes and alpha* shifts.

**Prediction D**: The critical exponent gamma (from beta ~ |alpha-alpha*|^gamma)
should equal 1 in the mean-field approximation (linear mixing of residual and
nonlinear branches). Our measured gamma ~ 0.68 suggests non-mean-field corrections,
possibly from inter-layer correlations.

### What This Means for the Manifesto

The scaling alpha* ~ 1 - C/L has a profound implication:

**DEEPER IS NOT FREE.** Each added layer brings the system closer to the
information propagation threshold. This means:

1. There are FUNDAMENTAL EFFICIENCY LIMITS on depth. Beyond a certain depth,
   you must strengthen residual connections or lose information.
2. Architecture design has quantifiable constraints: given L layers, the
   residual strength MUST exceed alpha* or representations collapse.
3. This connects directly to "Intelligence = Geometry": the geometry of
   information flow through depth creates hard constraints that no amount
   of scale can overcome.

---

## Open Questions (Updated)

1. Can we validate the 1/L scaling with the Pythia depth sweep (6 models)?
   - If exponent = 1.0 +/- 0.1, theory confirmed
   - If exponent significantly != 1, what's the correction?

2. Can we MEASURE rho_f directly?
   - Compute Jacobian spectral radius of the nonlinear branch
   - Use this to PREDICT alpha* without any fitting

3. Does the theory predict the PROFILE SHAPE (not just the transition point)?
   - Can we derive the competition model's E(x) and T(x) from the Jacobian?

4. What happens for non-transformer architectures (SSMs, hybrids)?
   - SSMs don't have standard residual connections
   - The theory should explain why SSMs show bell shapes at alpha=1

5. Connection to information bottleneck theory?
   - Tishby's IB predicts compression-then-fitting phases
   - Our theory says: residual connections control the rate of compression
   - Alpha below alpha* = too much compression per layer = information loss
