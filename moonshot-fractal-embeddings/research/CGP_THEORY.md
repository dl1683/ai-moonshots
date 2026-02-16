# CGP Theory: The Geometric Invariant of Intelligence
## Version 2 (post-Codex review, Feb 16 2026)

## Codex Verdict on V1: 2/10 rigor
Key criticisms (all valid):
- Scalar S too weak for universal characterization
- kNN bound misapplied Cover-Hart, missing assumptions, no finite-sample dependence
- Programmability theorem is just "backprop works" in disguise
- Rate-distortion framing is metaphorical, not operational
- Need BOTH upper and lower bounds (necessity + sufficiency)

## The Revolutionary Statement (Codex-proposed target)

> "For a broad class of data-generating processes and downstream tasks,
> there exists a representation-geometric invariant G(P_{Z,Y}) such that
> minimax achievable error and sample complexity are fully characterized
> (up to universal constants) by G; and any model intervention affects
> generalization only through its effect on G."

This is the theorem we need to prove. Everything below works toward this.

## What Should G Be?

### Why Scalar S Fails
S = inter/intra ratio captures ONE aspect of geometry. But:
- Doesn't distinguish balanced vs imbalanced classes
- Ignores multi-scale structure (coarse vs fine categories)
- Doesn't capture distribution shape (Gaussian vs heavy-tailed)
- Not invariant to representation dimension

### The Multi-Scale Geometric Invariant

**Definition**: Given representations Z in R^d with hierarchical labels Y = (Y_0, Y_1, ..., Y_L)
where Y_0 = coarsest, Y_L = finest:

G(P_{Z,Y}) = { S_l(Z, Y_l) : l = 0, 1, ..., L }

where S_l is the class separation at level l:
S_l = E[d(z_i, z_j) | y_l(i) != y_l(j)] / E[d(z_i, z_j) | y_l(i) = y_l(j)]

**But this is still a tuple of scalars.** Need something richer.

### Proposed: The Geometric Spectrum

**Definition (Geometric Spectrum)**: For representations Z with labels Y:

G(Z, Y) = { margin_c(Z) : c in [C] }

where margin_c(Z) = E_{x~P_c}[ min_{c' != c} d(f(x), decision_boundary(c, c')) ]

This is the full margin distribution, not just the average.

**Equivalent formulation**: G is the class-conditional overlap integral:

G_cc'(Z) = integral p(z|y=c) p(z|y=c') dz

for all pairs (c, c'). When G_cc' = 0 for all pairs, perfect separation.

### Even Richer: The Representation Quality Functional

**Definition**: Let phi: R^d x R^d -> R be a kernel. Define:

Q(Z, Y) = trace(Sigma_between) / trace(Sigma_within)

where Sigma_between = (1/C) sum_c (mu_c - mu)(mu_c - mu)^T
      Sigma_within = (1/C) sum_c E[(z - mu_c)(z - mu_c)^T | y=c]

This is the MULTIVARIATE Fisher discriminant ratio (Fukunaga 1990).
It's a scalar, but it captures the FULL covariance structure, not just distances.

**Key property**: Q is invariant to orthogonal transformations of Z.

## Toward the Main Theorem

### Assumptions (need to be stated explicitly)
A1. Class-conditional distributions P(Z|Y=c) are sub-Gaussian with parameter sigma_c
A2. Classes are balanced: P(Y=c) = 1/C for all c
A3. Representations are normalized: ||z|| = 1 (lie on hypersphere S^{d-1})
A4. Dimension d >= C (enough dimensions to separate)

### Upper Bound (Sufficiency)

**Theorem (Upper Bound)**: Under A1-A4, for the k-NN classifier with n training samples:

P(error) <= (C-1) * exp(-n^{1/(d+1)} * Phi(Q)) + O(k/n)

where Phi(Q) is an increasing function of the multivariate Fisher ratio Q.

**Proof approach**:
1. Under sub-Gaussian assumptions, class-conditional densities have exponential tails
2. The probability that a random point falls closer to wrong centroid than right centroid
   is controlled by the minimum eigenvalue of Sigma_between * Sigma_within^{-1}
3. For kNN with n samples, the effective neighborhood radius scales as n^{-1/(d+1)}
4. Combining gives the exponential bound

### Lower Bound (Necessity)

**Theorem (Lower Bound)**: There exist distributions satisfying A1-A4 such that:

P(error) >= (1/2C) * exp(-constant * Q)

for ANY classifier (not just kNN).

**Proof approach**: Minimax over sub-Gaussian distributions.
When Q is small, the class-conditional distributions overlap significantly,
and no classifier can separate them (by Fano's inequality).

### The Mediation Theorem (Novel)

**Theorem (Causal Mediation)**: For any parameterized encoder f_theta:

d(P_error) / d(theta) = [d(P_error)/d(Q)] * [d(Q)/d(theta)]

where the first factor depends ONLY on the geometric invariant Q,
and the second factor depends ONLY on the architecture and training.

**Meaning**: The effect of ANY model change (LoRA, scaling, new objective, etc.)
on generalization is FULLY MEDIATED through its effect on Q.

**What's novel**: This is NOT just "chain rule." The claim is that Q is a SUFFICIENT
statistic for the effect of theta on error. There is no "bypass" path from theta to error
that doesn't go through Q. This is a stronger claim than differentiability.

**How to prove**: Show that P_error is a function of Q only (given fixed data distribution).
This follows from the upper+lower bounds having the same form.

### The Controllability Theorem (Novel)

**Theorem (Geometric Controllability)**: For a rank-r LoRA adapter on a d-dimensional
representation space:

max_{||Delta||_F <= B} |Q(f_base + Delta) - Q(f_base)| >= Omega(r * B / sqrt(d))

**Meaning**: Low-rank interventions can move Q by at least Omega(r*B/sqrt(d)).
This lower-bounds the "controllability" of the geometric state.

**What's novel**: This shows that geometry is not just predictive but CONTROLLABLE
with bounded compute (rank-r LoRA = O(r*d) parameters, small fraction of model).

**Combined with mediation**: Since changing Q changes error, and LoRA changes Q
efficiently, we can program error bounds through geometry.

## The Core Result: Geometry = Intelligence (informally)

Putting it together:

1. Error is characterized by Q (upper + lower bounds)
2. All interventions affect error only through Q (mediation)
3. Q is controllable with low-rank adapters (controllability)

Therefore: **Intelligence (generalization ability) IS geometric structure (Q),
and it is programmable independent of model scale.**

This is "Intelligence = Geometry" as a theorem, not a slogan.

## What's Missing (Honest Assessment)

1. The bounds are not proved yet — only sketched
2. Sub-Gaussian assumption may be too strong for real representations
3. The mediation theorem needs formal causal framework (Pearl's do-calculus?)
4. Controllability bound needs explicit constants
5. Empirical test: does Q predict error BETTER than S? Need to compute Q from data.
6. Multi-scale version: need hierarchical Q for hierarchical tasks
7. Need to show this works across modalities (vision, text, audio)

## Experimental Tests of the Theory

### Test 1: Q vs S as predictor
Compute multivariate Fisher Q for all conditions in Week 2.
If Q predicts better than S (R2 > 0.55), the theory has better explanatory power.

### Test 2: Mediation test
Regress: error ~ Q + objective (should Q absorb all objective effect?)
If coefficient on objective drops to zero when Q is included, mediation holds.

### Test 3: Controllability
Plot Delta_Q vs LoRA rank r. Should scale linearly.

### Test 4: Cross-architecture universality
Same Q -> same error across Pythia, BGE, E5, Gemma?

## Connection to Prior Work
- Fisher discriminant (1936): scalar, single scale, linear
- Wang & Isola (2020): alignment + uniformity, no class separation
- Neural Collapse (Papyan et al., 2020): shows last-layer converges to simplex ETF
  - Our Q generalizes this: NC is when Q -> infinity
  - But NC is about training convergence, we're about generalization
- Representation Learning theory (Arora et al., Tosh et al.): contrastive learning bounds
  - Usually in terms of augmentation overlap, not geometric invariants
- Our contribution: Q as the SUFFICIENT STATISTIC mediating all interventions

## TODO (Priority Order)
1. [ ] Compute Q for Week 2 data and test Q vs S as predictor
2. [ ] Formal mediation test (Q absorbs objective effect)
3. [ ] State and prove upper bound under sub-Gaussian + normalized assumptions
4. [ ] State and prove lower bound via minimax/Fano
5. [ ] Prove mediation theorem (Q is sufficient statistic)
6. [ ] Prove controllability bound
7. [ ] Test cross-architecture universality
8. [ ] Extend to hierarchical/multi-scale Q
