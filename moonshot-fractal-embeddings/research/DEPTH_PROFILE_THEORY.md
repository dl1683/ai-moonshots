# First-Principles Theory: Why Representation Quality Follows Bell-Shaped Depth Profiles

## Status: DRAFT — seeking derivation, not polishing

## The Empirical Fact
Representation quality Q(x) at relative depth x = l/L follows:
logit(Q_norm(x)) = b_d + alpha*log(C/N) - beta*(x - mu_0 - mu_1*log(C/N))^2

This is a Gaussian-in-logit-space with R2=0.78 across 5+ model families (Pythia, OLMo-2, OPT, GPT-2, Cerebras-GPT) and 5000+ observations.

**Question**: WHY this form? Can we derive it from first principles?

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

## Open Questions

1. Can we make the linear assumption E(x) = a(1-x), T(x) = bx MORE rigorous?
   - Is there a variational principle that selects these linear forms?
   - Or is the Gaussian just a second-order Taylor approximation around the peak?

2. What happens at the boundaries (x=0 and x=1)?
   - Input embedding layer and output projection have special structure
   - The theory should predict deviations at boundaries

3. How does architecture affect the functional forms of E(x) and T(x)?
   - Attention vs MLP contributions
   - Residual connections vs not
   - Different normalization schemes

4. Can we MEASURE a and b independently?
   - a from representation geometry (e.g., intrinsic dimensionality trajectory)
   - b from gradient flow analysis
   - Then PREDICT the depth profile without fitting

5. Connection to information bottleneck theory?
   - Tishby's IB predicts compression-then-fitting phases
   - Our competition model gives a continuous version
   - Are they the same theory in different variables?
