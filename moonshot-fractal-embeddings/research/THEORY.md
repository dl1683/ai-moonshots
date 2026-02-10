# Theory: Limits of Flat Embeddings and Why Fractal Prefix Supervision Works

Last consolidated: February 10, 2026.

Status: This is the canonical theory document for the repository.

This file replaces the following overlapping theory docs:
- `research/THEORETICAL_FOUNDATIONS.md`
- `research/SAMPLE_COMPLEXITY_THEORY.md`
- `research/SCALE_SEPARATED_EMBEDDINGS_THEORY.md`
- `research/GENERALIZATION_THEORY.md`
- `research/SUCCESSIVE_REFINEMENT_THEORY.md`

---

## 1. Goal and Scope

We want one rigorous answer to:
1. What are the limitations of flat or isotropic embeddings for hierarchical tasks?
2. Why should hierarchy-aligned fractal prefix supervision help?
3. Which claims are theorem-level, which are conditional, and which are conjectural?

This document keeps theorem statements, proof sketches, and practical boundary conditions in one place.

---

## 2. Setup and Notation

We use a two-level hierarchy first, then extend to depth `L`.

- Input: `X in X`
- Fine label: `Y1 in {1,...,K1}`
- Coarse label: `Y0 = g(Y1) in {1,...,K0}`
- Conditional hierarchy entropy: `H(Y1|Y0)` (how much residual information remains after coarse label)

Embedding is block-structured:

`z = [z1; z2; ...; zJ]`, with prefix `z<=m = [z1;...;zm]`.

Interpretation:
- Early prefix blocks should carry coarse information.
- Later blocks refine fine distinctions.

Steerability metric used throughout:

`S = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)`.

Large positive `S` means: short prefix is coarse-specialized, full embedding is fine-specialized.

---

## 3. Why Flat/Isotropic Embeddings Are Limited

### 3.1 No semantic rate allocation

A flat embedding trains every prefix toward the same fine label objective.  
This gives no structural pressure for coarse-to-fine specialization.

### 3.2 Access-complexity penalty under coordinate budgets

For a level-`j` decision with only `m` coordinates available:
- In scale-separated embeddings, relevant signal is concentrated in early blocks.
- In isotropic embeddings, signal is spread across all `D` coordinates, so observable signal scales like `m/D`.

This creates an inherent budget disadvantage for isotropic representations in hierarchical queries.

### 3.3 Flat classifier sample complexity scales with leaf count

If there are `C = b^L` leaves:
- Flat multiclass dependence includes `log C = L log b`.
- Hierarchical per-level decomposition depends on local branching (`log b`) rather than full leaf space.

So flat methods pay complexity for all leaves even when only coarse decisions are needed.

---

## 4. Core Theoretical Results

### 4.1 Prefix Sufficiency for Scale-Separated Embeddings

Theorem (prefix sufficiency, noiseless):  
If each block is injective in its own level variable and level `j` information is isolated up to block `j`, then:
1. `L_j` is recoverable from prefix `f_<=j`.
2. `I(L_j; f_>j | f_<=j) = 0`.

Proof sketch:
1. Induct on depth.
2. Each new block resolves the next hierarchy variable given recovered parent prefix.
3. Since `L_j` is a deterministic function of `f_<=j`, conditional mutual information above is zero.

Implication: later blocks are refinement, not required for level-`j` semantics.

### 4.2 Budgeted Decoding: Scale-Separated Upper Bound vs Isotropic Lower Bound

Under Gaussian observation noise `Y = f(L) + Z`, `Z ~ N(0, sigma^2 I)`:

Scale-separated upper bound (nearest-neighbor on prefix):

`P_err(level j) <= (B_j - 1) * exp(-Delta_j^2 / (8 sigma^2))`

where `Delta_j` is minimum class separation at level `j` in the relevant prefix subspace.

Isotropic lower bound (binary reduction + Bretagnolle-Huber type argument):

`P_err >= c * exp(-KL)`, with `KL <= (1+eps) * (m/D) * (Delta_j^2 / (2 sigma^2))`

for a projection budget `m`.

Implication:
- Scale-separated embeddings can achieve low error with `m` near prefix size.
- Isotropic embeddings need `m = Omega(D)` to match comparable error in the same regime.

This yields the access-complexity separation intuition: early-level queries get an `Omega(k/(j+1))`-type advantage when `k` equal-sized blocks are used.

### 4.3 Sample Complexity Separation (Hierarchical vs Flat)

Representative multiclass ERM forms:

- Hierarchical per-level:
  `n_hier = O((d log b + log(L/delta)) / eps^2)`

- Flat leaf classifier:
  `n_flat = O((d log C + log(1/delta)) / eps^2) = O((d L log b) / eps^2)`

With `L > 1`, flat scaling grows with full leaf complexity while hierarchical decomposition tracks local branching.

Complementary information-theoretic lower bound (Fano family):
- For flat `C`-way identification, `n = Omega(log C)` is necessary in the standard construction.

### 4.4 Minimax Lower-Bound Perspective

Assouad-style result form used in this repo:
- For per-level class complexity `d_l` and sufficiently small `n`, there exists a hard distribution with non-trivial excess risk.
- This is minimax (`for all algorithms, exists distribution`), not a universal lower bound over every distribution.

Implication: separation claims are about hardest-case scaling and cannot be interpreted as "every distribution is hard."

### 4.5 Successive-Refinement Interpretation

When hierarchy is deterministic (`Y0 = g(Y1)`), two-stage coding naturally decomposes into:

`H(Y1) = H(Y0) + H(Y1|Y0)`.

Hierarchy-aligned prefix supervision approximates this coding strategy:
- Prefix budget targets coarse semantics.
- Full embedding adds residual fine semantics.

MRL-style uniform supervision across prefixes does not enforce this decomposition and therefore does not guarantee steerability.

### 4.6 End-to-End Generalization Caveats

The theory supports end-to-end claims only under capacity control:
- Bounded backbone complexity.
- Bounded head complexity.
- Surrogate-risk generalization (not direct 0-1 guarantees without conversion).

Critical caution:
- Independent heads per parent configuration can induce an effective complexity term like
  `C_eff = sum_l min(n, b^(l-1))`,
  which can become exponential in depth.
- Shared or restricted heads avoid this explosion.

---

## 5. Practical Predictions from Theory

1. Positive steerability requires non-trivial residual hierarchy (`H(Y1|Y0) > 0`) and a meaningful prefix bottleneck.
2. Inverted supervision should flip steerability sign.
3. There is a Goldilocks region where prefix capacity matches coarse entropy: too small underfits coarse labels, too large leaks fine labels into prefix.
4. Floor and ceiling effects are expected:
- Floor: coarse/fine tasks too hard even at full length.
- Ceiling: coarse task already saturated for all methods.

---

## 6. Empirical Status (Current Repository Snapshot)

Based on the current benchmark/stats artifacts (Feb 8-9, 2026 runs):

1. Eight-dataset scaling trend:
- `rho(H(L1|L0), V5 steer) = 0.743`, `p = 0.0349`
- Product predictor `H(L1|L0) * baseline L1`:
  `rho = 0.905`, `p = 0.0020`

2. Random-effects meta-analysis across 8 datasets:
- pooled Cohen's `d = 1.49`
- `95% CI = [0.69, 2.30]`
- `p = 0.0003`
- heterogeneity `I^2 = 63.1%`

3. Holm-corrected paired steerability tests:
- significant: `TREC`, `DBPedia Classes`, `CLINC`
- non-significant after correction: `Yahoo`, `GoEmotions`, `Newsgroups`, `arXiv`, `WOS`

Primary artifacts:
- `results/paper_statistics_holm.json`
- `results/meta_analysis.json`
- `results/scaling_robustness.json`
- `results/benchmark_bge-small_*.json`

---

## 7. What Is Proven vs Conditional vs Open

### Proven/established in this repo (math + code support)

1. Scale-separated prefix sufficiency concept and its formal consequences.
2. Access-complexity advantage framework under coordinate-budget/noise assumptions.
3. Hierarchical-vs-flat sample complexity separation form (with explicit assumptions).
4. Empirical steerability trend and positive pooled effect across 8 datasets.

### Conditional claims (assumption-sensitive)

1. Tight constants in finite-sample bounds.
2. End-to-end separation with flexible head parameterizations.
3. Exact mapping from optimization dynamics to ideal successive-refinement coding.

### Open questions

1. Sharp threshold characterization for Goldilocks behavior.
2. Fully distribution-free deep nonlinear characterization beyond current assumptions.
3. Multi-level (`L > 2`) optimal block-allocation law with explicit finite-sample guarantees.
4. Cross-modal universality (text/vision/audio/code) under identical hierarchy constraints.

---

## 8. Failure Modes and Limitations

1. Trivial hierarchy (`H(Y1|Y0) approx 0`): little room for steerability.
2. Prefix overcapacity: coarse and fine both fit in prefix; specialization weakens.
3. Prefix undercapacity: coarse signal not reliably encoded.
4. Error propagation in hierarchical inference (teacher-forcing vs deployment gap).
5. Dataset-specific learnability limits can dominate hierarchy effects.
6. Statistical heterogeneity (`I^2` non-trivial) means effect size is not uniform across domains.

---

## 9. Minimal Reference List

Core sources used by this consolidated theory:

1. Equitz and Cover (1991), successive refinement.
2. Rimoldi (1994), achievable rates for refinement.
3. Shalev-Shwartz and Ben-David (2014), multiclass learning bounds.
4. Natarajan (1989), multiclass dimension.
5. Cover and Thomas (2006), Fano inequality.
6. Tsybakov (2009), minimax/testing tools.
7. Bartlett and Mendelson (2002), Rademacher complexity.
8. Maurer (2016), vector-contraction inequality.
9. Kusupati et al. (2022), Matryoshka Representation Learning baseline.
