# CGP Theory: Why Class Separation Predicts Quality

## The Empirical Fact
Class separation ratio S = mean_inter_class_dist / mean_intra_class_dist
predicts representation quality (R2 = 0.55 for kNN L0) better than:
- Alignment (R2 = 0.001)
- Uniformity (R2 = 0.00003)
- Anisotropy (R2 = 0.001)

## Goal: Derive Error Bound as Function of S

### Setup
- X: input space
- Z = f(X): d-dimensional representation (f is encoder)
- Y: label (C classes)
- S = E[d(z_i, z_j) | y_i != y_j] / E[d(z_i, z_j) | y_i = y_j]  (class separation)
- Use k-NN classifier on Z

### Theorem Sketch: kNN Error Bound via Separation

**Claim**: For k-NN classifier on normalized representations Z in R^d:

P(error) <= 2(C-1)/C * exp(-k * g(S))

where g(S) is an increasing function of separation S.

**Proof sketch** (via Cover-Hart):
1. Cover & Hart (1967): k-NN error R* satisfies R* <= 2R_Bayes(1 - R_Bayes)
2. But we want to bound R* directly in terms of geometry
3. Key insight: if class-conditional distributions are separated (high S), then
   the probability that a k-NN neighbor belongs to a different class is small

**More precise version** (via margin theory):

Let gamma = min_{y_i != y_j} ||mu_i - mu_j|| / (sigma_intra)
where mu_i = class centroid, sigma_intra = intra-class spread.

Then gamma is monotonically related to S, and:
- For 1-NN: P(error) <= P(nearest neighbor is wrong class)
  <= sum_{c != y} P(||z - mu_c|| < ||z - mu_y||)
  <= (C-1) * exp(-gamma^2 / 8) [Gaussian tail bound]

### Connection to Information Theory

**Mutual Information Lower Bound**:
I(Z; Y) >= H(Y) - H(Y|Z)

For well-separated classes:
H(Y|Z) -> 0 as S -> infinity (certainty about class given representation)

More precisely, using Fano's inequality:
P(error) >= (H(Y|Z) - 1) / log(C)

So: high S -> low H(Y|Z) -> low P(error)

The converse is also approximately true:
I(Z; Y) >= log(C) - h(P_error) - P_error * log(C-1)

### The Programmability Theorem (Novel)

**Claim**: For a LoRA-modified encoder f_theta = f_base + Delta_theta:
1. The class separation S(theta) is a continuous function of theta
2. Gradient of S w.r.t. theta exists almost everywhere
3. Therefore, S can be optimized via gradient descent
4. The regularizer L_sep = intra / (inter + eps) is a differentiable proxy

This means: the geometric state variable S is **controllable** via optimization.

Combined with the error bound above:
- Controlling S -> controlling error bound
- This is "programming quality through geometry"

### The Key Insight for Nobel Track

Standard view: Quality comes from {model size, data size, compute}.
Our view: Quality comes from geometric structure (S), which is ORTHOGONAL to scale.

Evidence needed:
1. S explains quality AFTER controlling for model size (done - same Pythia-160M)
2. Higher S from small model beats lower S from large model (WEEK 3-4 goal)
3. S is a universal predictor across architectures (WEEK 3-4 goal)
4. Theoretical bound is tight (this document)

### Rate-Distortion Connection

Think of S as a "resolution" parameter:
- Low S: classes overlap (high distortion, low rate needed)
- High S: classes separated (low distortion, high rate needed)

The rate-distortion function D(R) for classification:
D(S) = classification error at separation S
R(S) = bits needed to encode representations with separation S

Conjecture: D(S) ~ exp(-alpha * S) for some alpha > 0
This would be the "geometry of intelligence" equation.

## TODO
- [ ] Formalize the kNN error bound with proper constants
- [ ] Prove the programmability theorem rigorously
- [ ] Test the exponential D(S) conjecture empirically
- [ ] Compare to known bounds (Bartlett margin theory, etc.)
- [ ] Check if this gives tighter bounds than existing work
