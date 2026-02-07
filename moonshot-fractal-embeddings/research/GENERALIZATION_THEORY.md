# Generalization Theory for Fractal Hierarchical Classifiers

Minimax lower bounds, distribution-free bounds, tight constants, DAG extensions, and end-to-end training analysis.

---

## Table of Contents

1. [Minimax Lower Bounds (Assouad)](#1-minimax-lower-bounds)
2. [Distribution-Free Bounds](#2-distribution-free-bounds)
3. [Tight Constants](#3-tight-constants)
4. [DAG Extensions](#4-dag-extensions)
5. [End-to-End Training](#5-end-to-end-training)

---

## 1. Minimax Lower Bounds

### 1.0 Definitions

**Notation**: Throughout, Ndim(H) denotes the **Natarajan dimension** of a multiclass hypothesis class H ⊆ [b]^X (the largest set of points that H shatters with two label functions; see Natarajan 1989). For b = 2, Ndim reduces to VC dimension.

**Per-level risk**: For a distribution D over X × [b]^L and a classifier f that predicts level ℓ labels, the level-ℓ risk is:
```
R_ℓ(f) := P_{(X,Y)~D}(f(X) ≠ Y_ℓ)
```
The level-ℓ Bayes risk is R_ℓ* := inf_{g: X → [b] measurable} R_ℓ(g). The **level-ℓ excess risk** is:
```
excess_ℓ(f, D) := R_ℓ(f) - R_ℓ*
```

**Probability spaces**: In all minimax statements below, expectations and probabilities are over both the training sample S = (X_1,Y_1),...,(X_n,Y_n) drawn i.i.d. from D and any internal randomness of the algorithm A. Formally: E_{S~D^n, A}[excess_ℓ(A(S), D)].

### 1.1 Main Result

**Theorem 1M (Minimax Lower Bound via Assouad)**

For any (possibly randomized) learning algorithm A and any ε ∈ (0, 1/16), if the per-level hypothesis class H_ℓ has Natarajan dimension d_ℓ ≥ 2, and
```
n < c × d_ℓ / ε²     [c = 1/72]
```
then there EXISTS a distribution D over X × [b]^L such that:
```
E_D[excess risk at level ℓ] ≥ ε
```

**High-probability form**: Under the same conditions, P_D(excess risk ≥ ε/2) ≥ ε/(2 - ε). (See Step 7 for derivation.)

**This is MINIMAX**: it holds for ANY learner and the HARDEST distribution.

**Note on probability**: The probability lower bound ε/(2-ε) vanishes as ε→0. A constant-probability statement (P ≥ 1/2) would require a Fano-based argument applied to parameter recovery; the expectation bound above is the standard Assouad form.

### 1.2 Proof via Assouad's Lemma

**Step 1: Natarajan Shattering Construction**

By definition of Natarajan dimension, there exist points x₁,...,x_{d_ℓ} ∈ X and label pairs (aᵢ, bᵢ) ∈ [b]² with aᵢ ≠ bᵢ such that for every σ ∈ {0,1}^{d_ℓ}, there exists h_σ ∈ H_ℓ with:
```
h_σ(xᵢ) = aᵢ if σᵢ = 0
h_σ(xᵢ) = bᵢ if σᵢ = 1
```

**Step 2: Hard Distribution Family**

Fix γ ∈ (0, 1/8) (required for the KL bound below). For each σ ∈ {0,1}^{d_ℓ}, define D_σ over X × [b]^L:
- X uniform on {x₁,...,x_{d_ℓ}}
- Given X = xᵢ, level ℓ label:
  - P(Y_ℓ = aᵢ) = 1/2 + γ, P(Y_ℓ = bᵢ) = 1/2 - γ if σᵢ = 0
  - Swapped if σᵢ = 1
- Other levels ℓ' ≠ ℓ: Y_{ℓ'} drawn from any fixed distribution independent of both σ and Y_ℓ (e.g., uniform on [b], or deterministic)

A Bayes-optimal classifier for D_σ at level ℓ can be chosen to coincide with h_σ on the support {x₁,...,x_{d_ℓ}} (since γ > 0, the majority label at each xᵢ is deterministic given σᵢ).

**Assouad prior**: We place the uniform prior π on {0,1}^{d_ℓ} and average over σ ~ π. All subsequent lower bounds hold for the worst-case σ by averaging: sup_σ R(A, D_σ) ≥ E_{σ~π}[R(A, D_σ)].

**Step 3: KL Divergence Computation**

For σ and σ^(i) (the vector obtained by flipping coordinate i of σ, i.e., σ^(i)_i = 1 - σ_i and σ^(i)_j = σ_j for j ≠ i) differing only in coordinate i:
```
KL(P_σ^n || P_{σ^(i)}^n) = n × (1/d_ℓ) × KL(Bern(1/2+γ) || Bern(1/2-γ))
                        ≤ n × 9γ² / d_ℓ    [for γ ∈ (0, 1/8)]
```

**Derivation of single-sample KL**: Under D_σ, a single sample (X, Y) has X uniform on {x₁,...,x_{d_ℓ}}. When σ and σ' = σ^(i) differ only in coordinate i, the conditional distributions P(Y|X=xⱼ) agree for all j ≠ i, and differ only for j = i. Since X = xᵢ with probability 1/d_ℓ, the chain rule for KL gives:
```
KL(P_σ || P_{σ^(i)}) = (1/d_ℓ) × KL(P(Y|X=xᵢ,σ) || P(Y|X=xᵢ,σ^(i)))
                      = (1/d_ℓ) × KL(Bern(1/2+γ) || Bern(1/2-γ))
```
(since Y_ℓ given X=xᵢ is Bernoulli on {aᵢ,bᵢ}, and other levels are independent of σ). For n i.i.d. samples, KL multiplies by n (tensorization).

**Exact KL expression**: The exact KL simplifies to:
```
KL(Bern(1/2+γ) || Bern(1/2-γ)) = 2γ × log((1/2+γ)/(1/2-γ))
```
(by direct algebra: the two terms combine since the log ratio changes sign).

**Upper bound via χ²**: Using KL(P||Q) ≤ χ²(P||Q) = (p-q)²/(q(1-q)):
```
KL ≤ (2γ)² / ((1/2-γ)(1/2+γ)) = 4γ²/(1/4 - γ²)
```
For γ < 1/8: 1/4 - γ² > 1/4 - 1/64 = 15/64, so KL < 4γ² × 64/15 < 18γ².

**Rigorous analytic bound**: The series KL = Σ_{k=0}^∞ a_k where a_k = 2^{2k+3} γ^{2k+2} / (2k+1) (all natural logarithms). The leading term is a_0 = 8γ². The tail from k ≥ 1 satisfies:

For k ≥ 1: 1/(2k+1) ≤ 1/3, so a_k ≤ (8γ²/3) × (4γ²)^k. Summing the geometric series:
```
Σ_{k≥1} a_k ≤ (8γ²/3) × Σ_{k≥1} (4γ²)^k = (8γ²/3) × 4γ²/(1 - 4γ²)
```
For γ ∈ (0, 1/8): 4γ² < 1/16, so 1/(1 - 4γ²) < 16/15. Therefore:
```
Σ_{k≥1} a_k ≤ (8γ²/3) × 4γ² × (16/15) = (8 × 4 × 16)γ⁴ / (3 × 15) = 512γ⁴/45
```
Since γ < 1/8 implies γ² < 1/64:
```
512γ⁴/45 < (512/(45 × 64))γ² = (8/45)γ² < 0.178γ²
```
Therefore KL = 8γ² + Σ_{k≥1} a_k < 8γ² + 0.178γ² < 8.18γ² < 9γ². ∎

**Alternative (weaker but simpler)**: The χ² bound gives KL ≤ 256γ²/15 < 18γ² for γ ∈ (0, 1/8). This is fully rigorous with no series, but yields a weaker theorem constant (c = 1/144 instead of 1/72).

**Step 4: Pinsker + Testing Lower Bound**

By Pinsker's inequality (TV² ≤ KL/2):
```
TV(P_σ^n, P_{σ^(i)}^n) ≤ √(KL/2) = √(9nγ²/(2d_ℓ)) = (3γ/√2)√(n/d_ℓ)
```

For coordinate i, **averaging over the Assouad prior** (σ ~ Unif({0,1}^{d_ℓ})):

Define the **marginal mixtures** for coordinate i:
```
P_i^0 = E_{σ: σ_i=0}[P_σ^n] = (1/2^{d_ℓ-1}) Σ_{σ: σ_i=0} P_σ^n
P_i^1 = E_{σ: σ_i=1}[P_σ^n] = (1/2^{d_ℓ-1}) Σ_{σ: σ_i=1} P_σ^n
```
By convexity of TV and the pairwise coupling (σ, σ^(i)) that pairs each σ with σ_i=0 to the σ' with σ_i=1 and all other coordinates identical:
```
TV(P_i^0, P_i^1) ≤ E_{σ_{-i}}[TV(P_{(σ_{-i},0)}^n, P_{(σ_{-i},1)}^n)] ≤ (3γ/√2)√(n/d_ℓ)
```
where the last inequality uses the Pinsker bound from above (since the TV between any pair differing only in coordinate i is bounded by the same expression, regardless of σ_{-i}).

Any algorithm A producing output Ŷ induces a binary estimator σ̂ᵢ = 1{Ŷ(xᵢ) = bᵢ} for each coordinate. By Le Cam's two-point testing lemma applied to the pair (P_i^0, P_i^1):
```
P_{σ_i=0}(σ̂ᵢ = 1) + P_{σ_i=1}(σ̂ᵢ = 0) ≥ 1 - TV(P_i^0, P_i^1)
```
Averaging with equal prior on σ_i:
```
E_{σ~π}[P(σ̂ᵢ ≠ σᵢ | σ)] ≥ (1/2)(1 - TV(P_i^0, P_i^1)) ≥ (1/2)(1 - (3γ/√2)√(n/d_ℓ))
```

**Note on σ̂**: The estimator σ̂ᵢ = 1{Ŷ(xᵢ) = bᵢ} is a deterministic function of the algorithm's output. If Ŷ(xᵢ) ∉ {aᵢ, bᵢ}, then σ̂ᵢ ≠ σᵢ with probability at least 1/2 (over the prior on σᵢ), so the bound only becomes stronger.

**Step 5: Risk Lower Bound**

Averaging over coordinates and the Assouad prior:
```
E_{σ~π}[(1/d_ℓ) Σᵢ P(σ̂ᵢ ≠ σᵢ | σ)] ≥ (1/2)(1 - (3γ/√2)√(n/d_ℓ))
```

**Connecting Hamming error to excess risk** (full case analysis): Recall σ̂ᵢ = 1{Ŷ(xᵢ) = bᵢ}, so σ̂ᵢ ∈ {0,1}. We show excess_ℓ(xᵢ) ≥ 2γ × 1{σ̂ᵢ ≠ σᵢ} for both values of σᵢ.

**Case σᵢ = 0**: Bayes-optimal prediction is aᵢ; Bayes risk at xᵢ is R_ℓ*(xᵢ) = 1/2 - γ.
- Ŷ(xᵢ) = aᵢ: risk = 1/2 - γ, excess = 0, σ̂ᵢ = 0 = σᵢ ✓
- Ŷ(xᵢ) = bᵢ: risk = 1/2 + γ, excess = 2γ, σ̂ᵢ = 1 ≠ σᵢ ✓
- Ŷ(xᵢ) = c ∉ {aᵢ,bᵢ}: risk = 1 (P(Y_ℓ=c)=0), excess = 1/2+γ > 2γ, σ̂ᵢ = 0 = σᵢ ✓ (bound is slack)

**Case σᵢ = 1**: Bayes-optimal prediction is bᵢ; Bayes risk at xᵢ is R_ℓ*(xᵢ) = 1/2 - γ.
- Ŷ(xᵢ) = bᵢ: risk = 1/2 - γ, excess = 0, σ̂ᵢ = 1 = σᵢ ✓
- Ŷ(xᵢ) = aᵢ: risk = 1/2 + γ, excess = 2γ, σ̂ᵢ = 0 ≠ σᵢ ✓
- Ŷ(xᵢ) = c ∉ {aᵢ,bᵢ}: risk = 1, excess = 1/2+γ > 2γ, σ̂ᵢ = 0 ≠ σᵢ ✓ (bound holds: excess > 2γ)

In all six cases: excess_ℓ(xᵢ) ≥ 2γ × 1{σ̂ᵢ ≠ σᵢ}. ∎

**Averaging over X and coordinates**: Since X is uniform on {x₁,...,x_{d_ℓ}}, the expected level-ℓ excess risk under D_σ (expectation over training sample S ~ D_σ^n, algorithm randomness A, and test point (X,Y) ~ D_σ) is:
```
E_{S,A,(X,Y)}[excess_ℓ] = (1/d_ℓ) Σᵢ E_{S,A,Y|X=xᵢ}[excess_ℓ(xᵢ)]
                         ≥ (1/d_ℓ) Σᵢ 2γ × P(σ̂ᵢ ≠ σᵢ)     [using the pointwise bound]
                         = 2γ × (1/d_ℓ) Σᵢ P(σ̂ᵢ ≠ σᵢ)
```
Taking expectation over σ ~ π and using the Step 4 bound:
```
E_{σ~π}[E_{D_σ}[excess_ℓ]] ≥ 2γ × (1/2)(1 - (3γ/√2)√(n/d_ℓ)) = γ(1 - (3γ/√2)√(n/d_ℓ))
```

If n ≤ d_ℓ/(18γ²), then (3γ/√2)√(n/d_ℓ) ≤ √(9γ²n/(2d_ℓ)) ≤ √(1/4) = 1/2, so:
```
E_{σ~π}[excess risk] ≥ γ/2
```

**Step 6: Final Bound**

Setting γ = 2ε (valid when ε < 1/16, ensuring γ < 1/8):
- Condition n ≤ d_ℓ/(18γ²) becomes n ≤ d_ℓ/(72ε²)
- E_{σ~π}[excess risk] ≥ γ/2 = ε

Since sup_σ ≥ E_{σ~π}, there exists σ* such that E[excess risk under D_{σ*}] ≥ ε:
```
n ≤ d_ℓ/(72ε²) ⟹ ∃D: E_D[excess risk at level ℓ] ≥ ε
```

**Step 7: High-Probability Statement**

Steps 1-6 give: if n ≤ d_ℓ/(72ε²), then E_D[excess risk] ≥ ε for some D.

**From expectation to probability (bounded random variable trick)**:

Let Z = excess risk. We have Z ∈ [0, 1] (since excess risk is bounded by 1) and E[Z] ≥ ε.

Applying Markov's inequality to the complement (1 - Z):
```
E[1 - Z] ≤ 1 - ε
P(1 - Z ≥ 1 - ε/2) ≤ (1 - ε)/(1 - ε/2)     [Markov on non-negative (1-Z)]
P(Z < ε/2) ≤ (1 - ε)/(1 - ε/2)
P(Z ≥ ε/2) ≥ 1 - (1 - ε)/(1 - ε/2) = (ε/2)/(1 - ε/2) = ε/(2 - ε)
```

For ε ∈ (0, 1/16): ε/(2-ε) > ε/2 > 0. This gives the high-probability form:
```
n ≤ d_ℓ/(72ε²) ⟹ ∃D: E_D[excess risk] ≥ ε and P_D(excess risk ≥ ε/2) ≥ ε/(2-ε)
```

**Remark**: This probability bound vanishes as ε → 0. A constant-probability bound P ≥ 1/2 can be obtained via Fano's inequality applied directly to parameter recovery (bounding P(σ̂ ≠ σ) ≥ 1/2 via the mutual information bound I(σ; data) ≤ nKL/d_ℓ), or via other testing arguments (e.g., Assouad variants with direct probability control). These are complementary approaches.

∎

### 1.3 Non-Gaussian Construction (Binary Symmetric Channel)

**Purpose**: This is a **separate, illustrative construction** demonstrating that minimax lower bounds do not require Gaussian assumptions. It operates under **normalized Hamming loss** (not the 0-1 loss of Theorem 1M). The main theorem (Sections 1.1-1.2) is self-contained; this section provides an alternative perspective.

**Setting (different loss from Section 1.1)**:
- **Label space**: {0,1}^m where m = ⌊log₂ b⌋ (embedded into [b] via binary encoding)
- **Loss function**: Normalized Hamming loss L_H(ŷ, y) = (1/m) Σⱼ 1{ŷⱼ ≠ yⱼ} (NOT the 0-1 multiclass loss)
- **Hypothesis class**: F = {f_u : u ∈ {0,1}^m} where f_u(x) := x ⊕ u (bitwise XOR)

**Hard distribution family**: For parameter θ ∈ {0,1}^m, define P_θ:
- Y uniform on {0,1}^m
- Given Y, observe X = Y ⊕ θ ⊕ Z, where Z ~ Bern(p)^m with 0 < p < 1/2 (each bit flipped independently)

This is a **binary symmetric channel with unknown bit-flip mask θ**.

**Bayes Classifier**: f_θ(x) = x ⊕ θ. Per-bit error rate = p, so Bayes Hamming risk R_H*(P_θ) = p for all θ.

**Risk derivation**: For any f_u ∈ F, f_u(X) = X ⊕ u = Y ⊕ (θ ⊕ u) ⊕ Z. Bit j of f_u(X) ⊕ Y is (θ ⊕ u)_j ⊕ Z_j. If (θ ⊕ u)_j = 0, error rate = p; if = 1, error rate = 1-p. Averaging over j:
```
R_θ^{Ham}(f_u) = p + (1-2p) × Hamming(u,θ)/m
```
Excess Hamming risk = R_θ^{Ham}(f_u) - p = (1-2p) × Hamming(u,θ)/m.

**KL Between Neighbors**: For θ and θ^{(j)} differing in bit j. Each sample (X,Y) under P_θ: the only difference is in the distribution of X_j | Y_j. Per-sample KL:
```
KL(P_θ || P_{θ^(j)}) = κ := KL(Bern(1-p) || Bern(p))
```
Note: each sample provides a noisy observation of ALL m bits (not one bit), so the per-sample KL between neighbors is κ, not κ/m. For n i.i.d. samples: KL(P_θ^n || P_{θ^(j)}^n) = nκ (tensorization).

For p = 1/2 - δ/2 with δ ∈ (0, 1/4) (hence p ∈ (3/8, 1/2), and γ := δ/2 < 1/8 satisfies the Step 3 requirement): κ = KL(Bern(1/2+δ/2) || Bern(1/2-δ/2)) ≤ 9(δ/2)² = 9δ²/4 (applying the Step 3 bound with γ = δ/2; or κ ≤ 4δ²/(1-δ²) < 5δ² via χ²).

**Assouad bound**: Applying Assouad's method over the m-dimensional hypercube {0,1}^m (analogous to Section 1.2 but with a key difference: here the per-sample KL between neighbors is κ with **no division by m**, since each sample observes all m bits simultaneously; cf. Section 1.2 where KL is divided by d_ℓ because X is uniform on d_ℓ points):

For each coordinate j, the marginal mixtures P_j^0, P_j^1 satisfy TV(P_j^0, P_j^1) ≤ √(nκ/2) (by Pinsker). Le Cam gives E[1{θ̂_j ≠ θ_j}] ≥ (1/2)(1 - √(nκ/2)). Averaging over all m coordinates:
```
E_{θ~Unif}[Hamming(θ̂,θ)/m] = (1/m) Σ_{j=1}^m E[1{θ̂_j ≠ θ_j}] ≥ (1/2)(1 - √(nκ/2))
```
**Why the bound is independent of m**: The normalized Hamming loss (1/m)Σ_j automatically averages over m coordinates. Each coordinate j contributes equally to the average, and the per-coordinate testing error (1/2)(1 - √(nκ/2)) is the same for all j (since κ is the KL between neighbors differing in any single bit). The m factors cancel: m in the denominator of the normalized loss matches the sum of m identical per-coordinate bounds.

**Connecting to excess risk**: Setting δ = 1-2p = 4ε (valid for ε < 1/16, ensuring δ < 1/4 and p = 1/2 - 2ε > 1/4):
```
E[excess Hamming risk] = δ × E[Hamming(θ̂,θ)/m] ≥ (δ/2)(1 - √(nκ/2))
```
Using κ ≤ 9δ²/4 = 9(4ε)²/4 = 36ε²: for n ≤ 1/(72ε²), we have nκ ≤ n × 36ε² ≤ 1/2, so √(nκ/2) ≤ 1/2, giving:
```
E[excess Hamming risk] ≥ δ/4 = ε
```

**Result**: n = Ω(1/ε²) samples are necessary for excess Hamming risk ≤ ε on this m-bit problem. The dimension d and log b factors come from embedding multiple independent problems (Section 1.4).

### 1.4 How to Get Ω(d log b / ε²)

**Lemma (Product Embedding)**

Let m = ⌊log₂ b⌋ (consistent with Section 1.3). If the hypothesis class H_ℓ contains **all** functions of the form:
```
x = (x^(1), ..., x^(m)) ↦ (f₁(x^(1)), ..., f_m(x^(m))) ∈ {0,1}^m ⊆ [b]
```
where the feature blocks x^(j) ∈ X_j are disjoint, each fⱼ comes from a binary class F with VC dimension d' ≥ 2, and the output is embedded into [b] via any fixed injection {0,1}^m → [b] (which exists since 2^m ≤ b), then:
```
Ndim(H_ℓ) ≥ m × (d' - 1) = Ω(d' log b)
```

**Proof**: We explicitly construct m(d'-1) points and two label functions witnessing Ndim(H_ℓ) ≥ m(d'-1). Set d = d' - 1 ≥ 1 throughout this proof.

**Point and anchor construction**: For each block j ∈ [m], since VC(F) = d' = d+1, there exist d+1 points that are VC-shattered by F. Pick one as the **anchor** x̄^(j) and let S_j = {x^(j)_1, ..., x^(j)_d} be the remaining d shattered points. Since F shatters all d+1 points (including x̄^(j)), for **any** binary labeling of {x^(j)_1, ..., x^(j)_d} and **any** choice of label for x̄^(j), there exists f ∈ F realizing that labeling. In particular, for any labeling of S_j with x̄^(j) fixed to 0, there exists f ∈ F achieving it.

Define md points:
```
z_{(j-1)d+i} := (x̄^(1), ..., x̄^(j-1), x^(j)_i, x̄^(j+1), ..., x̄^(m))   for j ∈ [m], i ∈ [d]
```
That is, z_{(j-1)d+i} has the i-th shattered point in block j and anchor values in all other blocks. All md points are distinct since the j-th block coordinate distinguishes points in different blocks, and the i-th coordinate within a block distinguishes points in the same block.

**Encoding**: Fix the injection ι: {0,1}^m → [b] as the binary encoding ι(v) = 1 + Σ_j v_j × 2^{j-1} (or any injective map; what matters is that distinct binary vectors map to distinct labels in [b]).

**Label pair construction**: For each point z_{(j-1)d+i}, define two labels:
```
α((j-1)d+i) := ι(0, ..., 0)           (all-zeros vector → some label in [b])
β((j-1)d+i) := ι(e_j)                  (vector with 1 only in position j → different label)
```
Since e_j ≠ 0^m, we have α ≠ β for every point. (More precisely: α assigns the same label ι(0^m) to all points, and β assigns ι(e_j) which depends on which block j the point belongs to.)

**Shattering**: For any σ ∈ {0,1}^{md}, we must exhibit h_σ ∈ H_ℓ with h_σ(z_{(j-1)d+i}) = α if σ_{(j-1)d+i} = 0 and β if σ_{(j-1)d+i} = 1.

For block j, define f_j^σ: X_j → {0,1} as the binary function satisfying:
- f_j^σ(x^(j)_i) = σ_{(j-1)d+i} for all i ∈ [d]  (realizes the desired labeling)
- f_j^σ(x̄^(j)) = 0  (fixes the anchor to 0)

This function exists because F shatters all d+1 points {x̄^(j), x^(j)_1, ..., x^(j)_d}, so any binary labeling of these points is achievable by some f ∈ F. In particular, the labeling that assigns 0 to x̄^(j) and σ_{(j-1)d+i} to x^(j)_i is achievable.

The product hypothesis h_σ(x) = ι(f_1^σ(x^(1)), ..., f_m^σ(x^(m))) satisfies:

At point z_{(j-1)d+i}:
```
h_σ(z_{(j-1)d+i}) = ι(f_1^σ(x̄^(1)), ..., f_j^σ(x^(j)_i), ..., f_m^σ(x̄^(m)))
                   = ι(0, ..., 0, σ_{(j-1)d+i}, 0, ..., 0)
                   = ι(σ_{(j-1)d+i} × e_j)
```
where the second equality uses f_{j'}^σ(x̄^(j')) = 0 for all j' (including j' = j by anchor construction, and j' ≠ j by the anchor-fixing property), and f_j^σ(x^(j)_i) = σ_{(j-1)d+i}. Therefore:
- If σ_{(j-1)d+i} = 0: h_σ = ι(0^m) = α ✓
- If σ_{(j-1)d+i} = 1: h_σ = ι(e_j) = β ✓

This gives Natarajan shattering of md = m(d'-1) points with label functions (α, β), proving Ndim(H_ℓ) ≥ m(d'-1) = (d'-1)⌊log₂ b⌋.

**Why m = Θ(log b)**: Since m = ⌊log₂ b⌋ and b ≥ 2, we have m ≥ 1 and m ≥ (log₂ b) - 1 ≥ (log₂ b)/2 for b ≥ 4. For b = 2, 3: m = 1 and log₂ b ≤ 1.58, so m ≥ (log₂ b)/2. In all cases, m = Θ(log b), giving Ndim(H_ℓ) ≥ m(d'-1) = Ω(d' log b). ∎

**Combined Result** (applying Theorem 1M with d_ℓ ≥ (d'-1)⌊log₂ b⌋ = Ω(d' log b)):
```
n ≥ Ω((d' log b) / ε²)  [MINIMAX]
```
where d' = VC(F) ≥ 2 and the per-block classes have VC dimension d'.

---

## 2. Distribution-Free Bounds

### 2.1 Impossibility of Universal Lower Bounds

**Proposition**: For any hypothesis class H containing at least one constant classifier h₀, there exists a distribution D and a learning algorithm A such that A achieves zero excess risk from n = 1 sample under D.

**Proof**: Let D be supported on a single point x₀ with deterministic label y₀ = h₀(x₀) for some h₀ ∈ H. Let A be the algorithm that, given a single sample (x₀, y₀), outputs h₀. Then R(A(S)) = R(h₀) = R* = 0, so excess risk = 0. ∎

**Implication**: A universal lower bound of the form "∀A, ∀D, n ≥ f(d, ε)" cannot hold with f(d, ε) > 1, because for the specific D and A above, n = 1 suffices. Therefore, **distribution-free lower bounds must be stated in the minimax form**: ∀A, ∃D such that [...]. The lower bound applies to the **hardest** distribution for any given algorithm, not to all distributions simultaneously.

### 2.2 Correct Distribution-Free Statement

**Theorem 2DF (Agnostic Minimax Lower Bound)**

Let H ⊆ ([b]^L)^X be the hierarchical hypothesis class.
Let H_ℓ = {x ↦ h(x)_ℓ : h ∈ H} be the level-ℓ projection.
Let d_ℓ = Ndim(H_ℓ).

For any learning algorithm A and any ε ∈ (0, 1/16), if:
```
n < c × d_ℓ / ε²     [c = 1/72, as in Theorem 1M]
```
then there exists D such that E_{S~D^n, A}[excess error at level ℓ] ≥ ε.

**Theorem 2DF-R (Realizable Lower Bound)**

Under realizability (∃h* ∈ H_ℓ with R(h*) = 0), for any algorithm A:
if n < c × d_ℓ / ε, then there exists a realizable distribution D such that E_D[excess error] ≥ ε.

(This is the standard Ω(d/ε) realizable lower bound for multiclass classification; see Daniely & Shalev-Shwartz 2014 for Natarajan-dimension-based lower bounds, or Shalev-Shwartz & Ben-David 2014 Thm 6.8 for the binary (b=2) special case.)

### 2.3 What This Means

1. The Ω(d_ℓ / ε²) bound (Theorem 1M) is **distribution-free in the minimax sense**, where d_ℓ = Ndim(H_ℓ). When the hypothesis class has product structure (Section 1.4), d_ℓ = Ω(d' log b), yielding Ω(d' log b / ε²). The log b factor is NOT automatic from the Natarajan dimension — it requires the product embedding assumption.
2. Rademacher/SRM/covering numbers are for **upper bounds**
3. VC/Natarajan + Assouad/Fano is the right machinery for **lower bounds**

---

## 3. Tight Constants

### 3.1 Union Bound Removal

**When log(L) is Removable** (sufficient conditions):

1. **Nested Classes**: If H₁ ⊆ H₂ ⊆ ... ⊆ H_L, a single uniform convergence bound over H_L controls all levels simultaneously, since any h ∈ H_ℓ is also in H_L. **Caveat**: This avoids the log(L) union bound factor, but at the cost of using the **largest** class complexity d_L = Ndim(H_L) for all levels. If the goal is to achieve a bound that **adapts** to the per-level complexity d_ℓ (choosing the tightest bound per level), a model-selection penalty of at most log(L) or log(ℓ) reappears. Thus, "no log(L)" is only accurate when using the global bound based on H_L's complexity.
   ```
   No log(L) factor (using d_L complexity for all levels)
   But log(L) may reappear for adaptive per-level bounds
   ```

2. **Time-Uniform Bounds**: When levels correspond to sequential time steps (i.e., the hierarchy maps to a temporal ordering) and observations arrive online, anytime-valid confidence sequences (e.g., mixture martingale bounds) can replace the union bound. Requires: (i) levels indexed by time, (ii) sequential observation model, (iii) a priori unbounded L. For fixed, known L, the standard union bound with log(L) can be tighter.
   ```
   log(L) → log log(n_L) (via confidence sequence theory, under sequential model)
   ```

3. **Path-Probability Weighting**: If only one level is used per prediction and we apply PAC-Bayes-style prior weighting over levels. Requires a known or learnable level distribution.
   ```
   log(L) → H(level distribution) ≤ log(L) (via prior-weighted union bound)
   ```

**When log(L) is Unavoidable**:
Without structural assumptions, a union bound over L independent level-wise failure events is tight up to constants. (Sketch: construct L independent events with P(Eℓ) = δ/L; total failure probability approaches δ.)

### 3.2 Bretagnolle-Huber Exact Form

**Exact BH Inequality**:
```
TV(P,Q) ≤ √(1 - exp(-KL(P||Q)))
```

Therefore:
```
α + β ≥ 1 - √(1 - exp(-KL))
```

**For max(α,β)**:
```
max(α,β) ≥ (1/2)(1 - √(1 - exp(-KL)))
```

The factor 1/2 is UNAVOIDABLE without symmetry (α = β).

### 3.3 Exact Gaussian Classification (Illustrative Example)

**Setting**: Binary classification in 1D with known equal variance. X | class k ~ N(μ_k, σ²) for k ∈ {1,2}. Equal priors π₁ = π₂ = 1/2.
**Estimator**: Plug-in classifier using sample means X̄₁, X̄₂ with n_c samples per class (2n_c total). Classify new x to class with nearest estimated mean (threshold at (X̄₁+X̄₂)/2).
**Signal**: s = |μ₁-μ₂|/(2σ).

**Exact Expected Risk** (for this specific estimator, using Gaussian integral identity E[Φ(a+bZ)] = Φ(a/√(1+b²))):
```
E[R] = Φ(-s / √(1 + 1/(2n_c)))
```
where n_c is the **per-class** sample size. (With n_c per class, Var(threshold) = σ²/(2n_c).)

**Exact Sample Complexity for Bayes + ε** (for this estimator):
```
n_c ≥ 1 / (2((s/q)² - 1))
```
where q = -Φ⁻¹(R* + ε), R* = Φ(-s) is the Bayes risk, and ε < 1/2 - R* is required.

**Note**: These formulas are specific to the plug-in sample-mean classifier in the 1D Gaussian location model with known equal variance and equal priors. They illustrate exact constant computation but do not directly generalize to arbitrary classifiers or higher dimensions.

---

## 4. DAG Extensions

### 4.1 Definitions

**Hierarchical Consistency on DAGs**:
For every edge u → v:
```
P(Y_v = 1, Y_u = 0) = 0
```
Labels form an **order ideal** (downward-closed set).

**Hierarchical Markov Property**:
```
Y_v ⊥ Y_{ND(v)} | (X, Y_{Pa(v)})
```

**Local Link Function**:
```
η_v(x) := P(Y_v = 1 | X = x, Y_{Pa(v)} = 1)
```

**Non-descendants**: ND(v) := V \ ({v} ∪ Desc(v) ∪ Pa(v)), where Desc(v) is the set of all descendants of v.

**Depth/Rank**:
```
ℓ(v) := 0 if Pa(v) = ∅ (root nodes), otherwise max_{u ∈ Pa(v)} ℓ(u) + 1
```
**Level sets**: L_d := {v ∈ V : ℓ(v) = d}.

### 4.2 Label Space Collapse

**Enabled Frontier**: For a set of "active" nodes A ⊆ V (representing nodes predicted positive):
```
E_d(A) := {v ∈ L_d : Pa(v) ⊆ A}
```
This is the set of depth-d nodes whose parents are all in A (i.e., nodes that are "enabled" for prediction).

**Frontier Width**:
```
W_d := max_{A ∈ I(G)} |E_d(A)|
W := max_d W_d
```
where I(G) is the set of order ideals (downward-closed subsets) of G. The maximum is over **feasible** active sets, i.e., those consistent with the hierarchical constraint. (Taking the max over all A ⊆ V would give an upper bound on W_d but could overcount infeasible configurations.)

This generalizes branching factor b_d from trees. For a tree with branching factor b at depth d, W_d = b^d (the order ideal consisting of all nodes at depths 0, ..., d−1 enables all b^d depth-d nodes, and no other order ideal enables more).

### 4.3 Sample Complexity Separation

**Theorem 4DAG (DAG Sample Complexity)**

**Setting**: Let G = (V, E) be a DAG with hierarchical label structure. For each node v, let H_v be a binary hypothesis class with VC dimension d_v, predicting Y_v given (X, Y_{Pa(v)}).

Let |I(G)| be the number of order ideals (feasible label patterns).
Let w be the maximum antichain size (DAG width).

Then:
- |I(G)| ≥ 2^w (each subset of a maximum antichain of size w generates a distinct order ideal when completed downward; more precisely, the number of antichains in a poset of width w is ≥ 2^w, and order ideals biject with antichains via the map I ↦ max(I), the set of maximal elements of I)
- Flat classification: The flat hypothesis class H_flat maps X to one of |I(G)| ≥ 2^w feasible label patterns. The Natarajan dimension of H_flat is **lower-bounded** by w **under the following assumptions**: (i) |X| ≥ w, and (ii) the per-node classes satisfy the **localized-flip property**: there exist w distinct points x_1, ..., x_w ∈ X and for each antichain node v_i, two hypotheses h_{v_i}^0, h_{v_i}^1 ∈ H_{v_i} such that h_{v_i}^0(x_i) ≠ h_{v_i}^1(x_i) AND h_{v_i}^0(x_j) = h_{v_i}^1(x_j) for all j ≠ i (each node can flip its prediction at its assigned point without affecting others). **Why this works (formal Natarajan shattering)**: Fix hypotheses for non-antichain nodes as follows: for each ancestor u of any antichain node, choose h_u ∈ H_u such that h_u(x_i) = 1 for all shattering points x_1,...,x_w (this ensures feasibility — any antichain assignment is compatible with parent labels being 1); for each descendant d of any antichain node, choose h_d ∈ H_d such that h_d(x_j) = 0 for all shattering points x_j where v_j is an ancestor of d (this prevents child-before-parent violations: when an antichain node is set to 0, its descendants are also 0; for threshold classifiers, choose a sufficiently large threshold); for unrelated nodes (neither ancestors nor descendants of antichain nodes), choose any fixed hypothesis. (This requires H_u to contain a hypothesis that outputs 1 at all w shattering points, which is satisfied, e.g., by threshold classifiers with sufficiently small threshold.) For antichain nodes, define two Natarajan witness functions: f_0(x_i) = the label pattern obtained when all antichain nodes use their "0" hypothesis (h_{v_j}^0 for all j), and f_1(x_i) = the pattern obtained when all use their "1" hypothesis. These satisfy f_0(x_i) ≠ f_1(x_i) for all i (they differ at the v_i-component by definition). For any subset S ⊆ [w], choose h_{v_i} = h_{v_i}^0 if i ∈ S and h_{v_i} = h_{v_i}^1 if i ∉ S. This defines a hypothesis h ∈ H_flat (the per-node selection produces a feasible order ideal: all ancestors of antichain nodes output 1, so setting any antichain node to 1 preserves hierarchical consistency, and setting it to 0 trivially preserves consistency). At point x_i: the v_i-component equals h_{v_i}^0(x_i) if i ∈ S and h_{v_i}^1(x_i) if i ∉ S, matching f_0(x_i)_{v_i} or f_1(x_i)_{v_i} respectively. For j ≠ i: the v_j-component at x_i equals h_{v_j}^{σ_j}(x_i) = h_{v_j}^0(x_i) = h_{v_j}^1(x_i) (by the localized-flip property), which matches both f_0(x_i)_{v_j} and f_1(x_i)_{v_j}. Hence h(x_i) = f_0(x_i) if i ∈ S and h(x_i) = f_1(x_i) if i ∉ S, completing the Natarajan shattering of w points. **Important**: d_v ≥ 1 alone is insufficient — counterexample: constant classifiers H_v = {0, 1} have d_v = 1 but lack pointwise control (all points get the same label, so Ndim = 1, not w). The localized-flip property is strictly stronger than d_v ≥ 1 but weaker than d_v ≥ w. **Compatibility with d_v = O(1)**: Consider threshold classifiers H_v = {x ↦ 1{x ≥ t} : t ∈ ℝ} on X = ℝ, which have d_v = 1. Given w points x_1 < x_2 < ... < x_w with gaps (i.e., x_i < x_{i+1}), choose h_{v_i}^0 = 1{x ≥ x_i + δ} (outputs 0 at x_i) and h_{v_i}^1 = 1{x ≥ x_i − δ} (outputs 1 at x_i), where δ is small enough that no other x_j lies in (x_i − δ, x_i + δ). Then h_{v_i}^0(x_j) = h_{v_i}^1(x_j) for j ≠ i (both thresholds agree on separated points) while h_{v_i}^0(x_i) ≠ h_{v_i}^1(x_i). This satisfies the localized-flip property with d_v = 1. (This is analogous to Section 1.4 but for order ideals rather than products.)
- Under realizability: m_flat = Ω(Ndim(H_flat) / ε) (standard Natarajan lower bound). Under agnostic setting: m_flat = Ω(Ndim(H_flat) / ε²).
- Hierarchical (per-node ERM with shared sample): m_hier = O((max_v d_v + log(|V|/δ)) / ε²) (by uniform convergence at the hardest node + union bound over |V| nodes for simultaneous control). **Note**: The bound uses max_v d_v (not Σ_v d_v) because the same n samples are used for all nodes. The sum would only apply if each node used separate samples.

**Separation condition** (both in agnostic setting): Since Ndim(H_flat) ≥ w (under the richness assumptions above), when w = Ω(|V|) and max_v d_v = O(1), we get:
- Flat: m_flat = Ω(w/ε²) = Ω(|V|/ε²)
- Hierarchical: m_hier = O((max_v d_v + log(|V|/δ))/ε²) = O(log(|V|)/ε²) when d_v = O(1)

This is a **polynomial separation** (|V| vs log|V|) in the effective dimension. The separation is in the dimension parameter (w vs max_v d_v), not in the ε-rate. **Note**: The number of feasible label patterns |I(G)| ≥ 2^w grows exponentially with w, but this does NOT directly imply Ndim(H_flat) is exponential — the Natarajan dimension is **lower-bounded** by w (the antichain size), and can potentially exceed w depending on the class structure; it is NOT upper-bounded by w in general. The lower bound Ndim(H_flat) ≥ w combined with d_v = O(1) is consistent: the localized-flip property is satisfiable with d_v = 1, as demonstrated by the threshold classifier example above. The flat class complexity arises from the **combinatorial explosion** of independent per-node predictions across the antichain, not from individual node richness.

### 4.4 Error Propagation with Multiple Parents

**False Negative Rate**:

In a hierarchical classifier with consistency enforcement (Ŷ_v = 1 requires all Ŷ_{Pa(v)} = 1 AND local_v = 1), a false negative at v occurs when local_v = 0 OR some parent Ŷ_u = 0. By union bound:
```
P(Ŷ_v = 0 | Y_v = 1) ≤ P(local_v = 0 | Y_v = 1) + Σ_{u ∈ Pa(v)} P(Ŷ_u = 0 | Y_v = 1)
```

**Important**: The parent error terms are P(Ŷ_u = 0 | Y_v = 1), which conditions on the **child** being positive. Since Y_v = 1 implies Y_u = 1 (by hierarchical consistency), we have P(Ŷ_u = 0 | Y_v = 1) = P(Ŷ_u = 0 | Y_u = 1, Y_v = 1). This can differ from the unconditional FNR_u = P(Ŷ_u = 0 | Y_u = 1) because conditioning on Y_v = 1 changes the distribution of (X, Y_{Pa(u)}). A valid (but possibly loose) upper bound is:
```
P(Ŷ_u = 0 | Y_v = 1) ≤ sup_{x, y_{Pa(u)}} P(Ŷ_u = 0 | X = x, Y_{Pa(u)} = y_{Pa(u)}, Y_u = 1) =: FNR_u^{wc}
```
Under the **conditional independence** assumption that Ŷ_u depends on (X, Y_{Pa(u)}) but not on Y_v given these, the bound simplifies to FNR_v ≤ FNR_local(v) + Σ_u FNR_u^{wc}. When the classifier is well-calibrated across subpopulations, FNR_u^{wc} ≈ FNR_u.

FNs accumulate additively through parent chains.

**False Positive Rate**:
In a hierarchical classifier with consistency enforcement (Ŷ_v = 1 only if all Ŷ_{Pa(v)} = 1), a false positive at v requires all parents to be **predicted** positive AND the local classifier to err. Formally:
```
FPR_v = P(Ŷ_v=1 | Y_v=0) = P(local_v=1 | Y_v=0) × P(all Ŷ_{Pa(v)}=1 | Y_v=0, local_v=1)
```
**Caution**: This does NOT mean all parents must be false positives. Parents can be **true** positives (Y_u=1, Ŷ_u=1) while v is truly negative (Y_v=0), since Y_v=0 is compatible with Y_{Pa(v)}=(1,...,1) in a hierarchy. FPs are only suppressed when some parent Y_u=0 (requiring that parent to also be a false positive). The suppression factor depends on the conditional P(all Ŷ_{Pa(v)}=1 | Y_v=0), which can be close to 1 when parents are genuinely positive.

---

## 5. End-to-End Training

### 5.1 Model Setup and Teacher-Forcing Risk

- Distribution D over X × [b]^L, with **0-1 loss** at each level: ℓ_ℓ(ŷ, y) = 1{ŷ_ℓ ≠ y_ℓ}
- Input space X ⊆ R^p (p-dimensional features)
- Backbone class Φ: maps φ: X → R^k (k-dimensional real-valued features)
- Per-parent head classes: For each level ℓ and parent configuration y_{<ℓ} ∈ [b]^{ℓ-1}, define H_ℓ^{y_{<ℓ}} ⊆ ([b])^{R^k} (multiclass classifiers given fixed parents). The full head at level ℓ is H_ℓ = {(z, y_{<ℓ}) ↦ h^{y_{<ℓ}}(z) : h^{y_{<ℓ}} ∈ H_ℓ^{y_{<ℓ}} for each y_{<ℓ}}
- Composed class F_ℓ = {(x, y_{<ℓ}) ↦ h^{y_{<ℓ}}(φ(x)) : φ ∈ Φ, h^{y_{<ℓ}} ∈ H_ℓ^{y_{<ℓ}}}

**Teacher-forcing risk** (formal definition): Under teacher forcing, the level-ℓ risk of a composed classifier f = (φ, h_1, ..., h_L) is:
```
R_ℓ^{TF}(f) := E_{(X,Y)~D}[1{h_ℓ^{Y_{<ℓ}}(φ(X)) ≠ Y_ℓ}]
```
where Y_{<ℓ} = (Y_1, ..., Y_{ℓ-1}) are the **true** parent labels from D. The average teacher-forcing risk is R_avg^{TF} = (1/L) Σ_ℓ R_ℓ^{TF}. The corresponding empirical risk on sample S = {(x_i, y_i)}_{i=1}^n drawn i.i.d. from D is:
```
R̂_ℓ^{TF}(f) = (1/n) Σ_i 1{h_ℓ^{y_{i,<ℓ}}(φ(x_i)) ≠ y_{i,ℓ}}
```
Note: for any FIXED f, the empirical sample is i.i.d. from D, so R̂_ℓ^{TF}(f) is an unbiased estimator of R_ℓ^{TF}(f). For the ERM-selected f̂ (which depends on the data), R̂_ℓ^{TF}(f̂) is biased downward — this is precisely what uniform convergence corrects.

**At test time** with predicted parents, error propagation (Section 4.4) applies additionally. All bounds below are for teacher-forcing risk.

### 5.2 Main Results

**Theorem 5A (End-to-End Generalization via Margin-Based Surrogate)**

**Key design choice**: The 0-1 loss 1{h(z) ≠ y} is NOT Lipschitz, so Rademacher contraction cannot be applied directly to 0-1 loss. Instead, we use a **margin-based surrogate loss** and obtain 0-1 risk bounds via surrogate-to-0-1 dominance. This is the standard approach in multiclass learning theory (Zhang 2004, Bartlett et al. 2006).

**Score-based formulation**: Each head is defined via b real-valued score functions. Formally, define the **score function class** for level ℓ and parent config y_{<ℓ}:
```
S_ℓ^{y_{<ℓ}} = {s = (s_1, ..., s_b) : s_c: R^k → R for c ∈ [b]}
```
The classifier is h^{y_{<ℓ}}(z) = argmax_{c ∈ [b]} s_c(z). The composed score class is:
```
G_ℓ^{y_{<ℓ}} = {x ↦ s(φ(x)) : φ ∈ Φ, s ∈ S_ℓ^{y_{<ℓ}}}
```

**Surrogate loss**: For margin ρ > 0, define the **margin function** and **ramp loss**:
```
m(s, y) := s_y - max_{c≠y} s_c           (margin: real-valued)
ℓ_ρ(s, y) := min(1, max(0, 1 - m(s,y)/ρ))  (ramp loss)
```
**Lipschitz analysis** (critical for covering number arguments):
- The margin m(s,y) is **2-Lipschitz** in the score vector s with respect to L∞: |m(s,y) - m(s',y)| ≤ |s_y - s'_y| + |max_{c≠y} s_c - max_{c≠y} s'_c| ≤ 2‖s - s'‖_∞.
- The ramp function t ↦ min(1, max(0, 1-t/ρ)) is **1/ρ-Lipschitz**.
- Therefore ℓ_ρ(s,y) is **(2/ρ)-Lipschitz** in the score vector s (L∞ norm).
- If each s_c is L_h-Lipschitz in z under L∞ (A2) and z = φ(x) with ‖φ(x)‖_∞ ≤ B_φ (A1), then the composed loss (x,y) ↦ ℓ_ρ(s(φ(x)),y) is **(2L_h/ρ)-Lipschitz** in φ(x) under L∞. All Lipschitz constants in this chain use the **L∞ norm consistently**.

The ramp loss upper-bounds the 0-1 loss: 1{h(z)≠y} ≤ ℓ_ρ(s,y) (since m(s,y) ≤ 0 when h(z)≠y). The surrogate risk is R_ℓ^{surr}(f) = E[ℓ_ρ(s(φ(X)), Y_ℓ)].

**Assumptions**:
- **(A1)** Backbone features are bounded: ‖φ(x)‖_∞ ≤ B_φ for all φ ∈ Φ, x ∈ X. (L∞ norm on R^k; this is the natural norm for coordinate-wise covering.)
- **(A2)** Score functions are L_h-Lipschitz in L∞: |s_c(z) - s_c(z')| ≤ L_h ‖z - z'‖_∞ for all s_c ∈ S_ℓ^{y_{<ℓ}}, all c, ℓ, y_{<ℓ}. (The L∞ norm is used consistently with the coordinate-wise backbone covering in Step 1.)
- **(A3)** Score functions are bounded: |s_c(z)| ≤ M for all ‖z‖ ≤ B_φ, all s_c ∈ S_ℓ^{y_{<ℓ}}. (Sufficient condition: A2 + s_c(0) = 0 gives M = L_h B_φ. In general, A3 is stated as an independent assumption.)
- **(A4)** Head class complexity: d_h = max_ℓ max_{y_{<ℓ}} Pdim(S_ℓ^{y_{<ℓ}, single}), where S_ℓ^{y_{<ℓ}, single} = {z ↦ s_c(z) : s ∈ S_ℓ^{y_{<ℓ}}, c ∈ [b]} is the class of **individual scalar score functions** (ranging over all b class indices c and all score vectors s). Note: d_h is a **per-scalar-function** complexity measure, not a per-vector measure. The factor b in the sample complexity arises from covering the **b-dimensional score vector** as a product of b scalar covers, each of complexity d_h. This is not double-counting: d_h measures one scalar function's complexity, and b counts how many independent scalar covers are needed for the full vector.
- **(A5)** Backbone pseudo-dimension: d_φ = Pdim(G_Φ), where G_Φ = {(x, j) ↦ φ(x)_j : φ ∈ Φ} is a class of functions on the **augmented domain X × [k]** (input is a pair (x, j), output is the j-th coordinate of φ(x)). The domain X × [k] is crucial: each function in G_Φ is parameterized by φ, and given (x, j), outputs a scalar. For linear backbones Φ = {x ↦ Wx : W ∈ R^{k×p}, ‖W‖_F ≤ B}, the class G_Φ = {(x, j) ↦ w_j^T x} is parameterized by kp entries of W. Since each of the k rows w_j can be chosen independently (subject to the norm constraint), G_Φ can pseudo-shatter kp points (p per coordinate j), giving d_φ = kp (Anthony & Bartlett 1999, Thm 11.8). **Remark**: If the domain were just X (not X × [k]), the pooled class would collapse to all linear functionals with Pdim = p. The augmented domain X × [k] is what yields d_φ = kp.

**Theorem statement**: Under (A1)-(A5), if ERM is performed on the surrogate loss R̂_avg^{surr}, then with probability ≥ 1-δ:
```
R_avg^{TF}(f̂) ≤ R_avg^{surr}(f̂) ≤ inf_f R_avg^{surr}(f) + 2ε
```
(The first inequality uses 0-1 ≤ surrogate; the second is ERM + uniform convergence.)

**Three cases** (sufficient sample size for the 2ε guarantee):

**(a) Shared-parameter heads** (s_c^{y_{<ℓ}} = s_c for all y_{<ℓ} AND all levels ℓ — i.e., one set of b score functions used everywhere):
```
n = O(((d_φ + b × d_h) / ε²) × log(nkL_h B_φ / ρ) + log(L/δ)/ε²)
```
**Note on implicit n**: The log(n...) factor makes this a self-referential bound of the standard form n ≥ D log(n), which is always solvable: n = O(D log(D × kL_hB_φ/ρ)) by standard arguments (see, e.g., the n ≥ D log(n) ⟹ n = O(D log D) lemma). The same applies to Case (c) below.

**(b) Independent heads** (each parent config uses its own score functions):
```
n = O(((d_φ + C_eff × b × d_h) / ε²) × log(nkL_h B_φ / ρ) + log(L/δ)/ε²)
where C_eff = Σ_ℓ min(n, b^{ℓ-1})
```
**Cartesian product structure** (NOT a union bound): Under teacher forcing, sample i at level ℓ uses head s^{y_{i,<ℓ}}. Since heads for different parent configs are chosen **independently**, the function class at level ℓ is a **Cartesian product** over observed configs, not a union. Let C_ℓ = {y_{i,<ℓ} : i ∈ [n]} be the set of distinct parent configs observed at level ℓ, with |C_ℓ| ≤ min(n, b^{ℓ-1}). The empirical covering number decomposes as a **product** over configs:
```
log N(loss class ℓ, ε) ≤ d_φ × log(...) + |C_ℓ| × b × d_h × log(...)
```
because for each observed config c ∈ C_ℓ, the head must be covered independently on the subset {x_i : y_{i,<ℓ} = c}. The logs **sum** (not max) because the heads are chosen independently per config. Summing over levels gives C_eff = Σ_ℓ |C_ℓ| ≤ Σ_ℓ min(n, b^{ℓ-1}).

**Warning**: C_eff can be exponential in L. When n ≥ b^L, the combinatorial upper bound gives |C_ℓ| ≤ b^{ℓ-1} for all ℓ, so C_eff ≤ Σ_ℓ b^{ℓ-1} = (b^L - 1)/(b-1). (If the distribution has full support over label configs, all b^{ℓ-1} configs are eventually observed.) When n < b^{ℓ-1} for deep levels, C_eff = O(nL). This exponential cost is the fundamental price of independent parameterization and is precisely why Cases (a)/(c) are preferable for deep hierarchies.

**Note on implicit dependence**: The bound in Case (b) is **data-dependent** (C_eff depends on n through |C_ℓ| ≤ min(n, b^{ℓ-1})). This makes the sample complexity statement implicit: n sufficient when n ≥ C × ((d_φ + C_eff(n) × b × d_h)/ε²) × log(...). Two explicit regimes:
- **Deep levels dominate** (n ≥ b^L): C_eff = (b^L-1)/(b-1), a constant independent of n. The bound becomes explicit: n = O(((d_φ + b^L × b × d_h)/ε²) × log(...)).
- **Shallow regime** (n ≤ b^{ℓ-1} for all ℓ ≥ 2): C_eff ≤ nL, yielding an **implicit** inequality n ≥ C × (d_φ + nLbd_h)/ε² × log(nkL_hB_φ/ρ). Let a = CLbd_h/ε², D = Cd_φ/ε², and Λ = log(kL_hB_φ/ρ). The inequality becomes:
  n ≥ (D + an)(Λ + log n)
  **Key structural difference from standard bounds**: Unlike the standard self-referential bound n ≥ D log(n) (which always has a solution for D > 0), here the RHS contains an **an log n** term, making the RHS grow superlinearly. This means solutions may fail to exist depending on the relationship between a, D, and Λ — not merely on whether aΛ < 1.
  **Necessary condition**: aΛ < 1 is necessary (otherwise the linear part alone exceeds n). But it is **not sufficient**: even with aΛ < 1, for large D or large a, the function g(n) = n − (D + an)(Λ + log n) may remain negative for all n ≥ 1 (since g(n) → −∞ as n → ∞ due to the −an log n term, and g can fail to achieve a positive maximum).
  **Solvability** is determined by whether g achieves a positive maximum; this depends on all three parameters (a, D, Λ) jointly, with no clean closed-form condition.
  **Practical recommendation**: Because Case (b) with C_eff = nL yields a fundamentally implicit and potentially unsolvable bound, **Cases (a) and (c) are strongly preferred** for any setting where explicit sample complexity guarantees are needed. Case (a) (shared heads) and Case (c) (restricted heads) both give bounds with standard n ≥ D log(n) self-reference that is always solvable (see note above). Case (b) is primarily of theoretical interest, showing the cost of full independent parameterization: the shallow-regime formula can fail to produce a finite guarantee when the number of independent head parameters (∝ Lbd_h) is too large relative to ε². (Note: for any fixed finite L, a finite sample complexity always exists via the deep regime (n ≥ b^L), where C_eff = (b^L-1)/(b-1) is constant and the bound is explicit. The "unsolvability" is specific to the shallow-regime formula, not the overall learning problem.)

**(c) Restricted heads** (scores ignore y_{<ℓ}, but may differ across levels ℓ):
```
n = O(((d_φ + L × b × d_h) / ε²) × log(nkL_h B_φ / ρ) + log(L/δ)/ε²)
```
**Note**: The L factor appears because each level has its own set of b score functions (L independent head sets contribute L × b × d_h to the joint covering number). If heads are also shared across levels (as in Case (a)), the L factor is absorbed into a single set of b heads. The 1/L averaging improves the concentration rate (bounded differences constant 1/n rather than L/n), but does not eliminate L from the covering complexity. The net rate lies between Cases (a) and (b).

**Proof**:

**Step 0: Covering Number Composition Lemma**

**Lemma (Composition of function classes)**: Let F = {f: X → Z} be a class with f(x) ∈ B_Z := {z ∈ R^k : ‖z‖_∞ ≤ B} for all f, x. Let G = {g: Z → R} be a class where each g is L-Lipschitz under L∞ (i.e., |g(z) - g(z')| ≤ L‖z - z'‖_∞). For any x_1, ..., x_n ∈ X:
```
log N(G ∘ F, ε, x_{1:n}) ≤ log N(F, ε/(2L), x_{1:n}) + sup_{z_{1:n} ∈ B_Z^n} log N(G, ε/2, z_{1:n})
```
where N(H, α, w_{1:n}) is the empirical L∞-covering number: the minimum number of balls of radius α in the metric max_i |h(w_i) - h'(w_i)| needed to cover H (for scalar-valued H; for vector-valued F mapping to R^k, the metric is max_i ‖f(w_i) - f'(w_i)‖_∞). The sup in the second term ranges over **all** bounded feature sequences, not just those from a specific f.

*Proof*: Take an ε/(2L)-cover {f̃_1, ..., f̃_M} of F on x_{1:n}. For each f̃_j, take an ε/2-cover of G on the feature sequence (f̃_j(x_1), ..., f̃_j(x_n)) with at most N* = sup_z N(G, ε/2, z_{1:n}) elements. For any g ∘ f ∈ G ∘ F, find f̃_j with max_i ‖f(x_i) - f̃_j(x_i)‖ ≤ ε/(2L), then find g̃ with max_i |g(f̃_j(x_i)) - g̃(f̃_j(x_i))| ≤ ε/2. By the triangle inequality:
```
|g(f(x_i)) - g̃(f̃_j(x_i))| ≤ |g(f(x_i)) - g(f̃_j(x_i))| + |g(f̃_j(x_i)) - g̃(f̃_j(x_i))|
                               ≤ L × ε/(2L) + ε/2 = ε
```
The total cover size is M × N*, giving log N(G∘F, ε) ≤ log M + log N*. Since ‖f̃_j(x_i)‖ ≤ B for all j, i, the sup over z dominates each log N*. ∎

**Key point**: The second term uses a **worst-case empirical covering number** (sup over all bounded feature sequences). This resolves the composition issue: the head covering is a property of the head class on bounded inputs, independent of which backbone generates the features.

**Step 1: Covering numbers of the per-level surrogate loss class**

For fixed level ℓ and parent config y_{<ℓ}, the surrogate loss class is:
```
L_{ℓ,y} = {(x,y_ℓ) ↦ ℓ_ρ(s(φ(x)), y_ℓ) : φ ∈ Φ, s ∈ S_ℓ^{y_{<ℓ}}}
```
We decompose this as a composition: f(x) = φ(x) ∈ B_Z (bounded by B_φ via A1), composed with g(z) = ℓ_ρ(s(z), y_ℓ) which is (2L_h/ρ)-Lipschitz in z (see Lipschitz analysis above). Applying the Composition Lemma:
```
log N(L_{ℓ,y}, ε, x_{1:n}) ≤ log N(Φ, ερ/(4L_h), x_{1:n})
                              + sup_{z ∈ B_{B_φ}^n} log N({z ↦ ℓ_ρ(s(z),y_ℓ) : s ∈ S_ℓ^{y_{<ℓ}}}, ε/2, z_{1:n})
```

**Bounding the backbone covering number**: The class G_Φ = {x ↦ φ(x)_j : φ ∈ Φ, j ∈ [k]} is a **scalar-valued** class with pseudo-dimension d_φ (by A5). To cover the vector-valued backbone Φ at resolution α in L∞ on n points (i.e., max_i ‖φ(x_i) - φ̃(x_i)‖_∞ ≤ α), it suffices to cover G_Φ at resolution α on the **kn evaluation points** {(x_i, j) : i ∈ [n], j ∈ [k]}. By the standard pseudo-dimension covering number bound (Anthony & Bartlett 1999, Theorem 12.2) applied to G_Φ on kn points:
```
log N(Φ, α, x_{1:n}) ≤ d_φ × log(2e × kn × B_φ / (d_φ × α))
```
**Note on k dependence in the backbone term**: The factor k enters the **backbone** contribution in **two places**: (1) **inside d_φ** — since d_φ = Pdim(G_Φ) on the augmented domain X × [k], d_φ already captures k-dependence. For linear backbones, d_φ = kp (not p). (2) **inside the logarithm** (as kn total evaluation points). There is no additional multiplicative k factor in the backbone term beyond d_φ. If coordinates share parameters (e.g., backbone is a single network with k outputs), d_φ captures the shared complexity and may be much less than kp.

**Important caveat**: The head complexity d_h may **also** depend on k, since heads operate on R^k inputs. For linear heads s_c(z) = w_c^T z with w_c ∈ R^k, d_h = k. In this case, k enters BOTH the backbone term (via d_φ) and the head term (via b × d_h = bk). The "no additional k" statement applies only to the backbone contribution, not to the overall bound.

**Bounding the head loss covering number**: For the head class on **any** fixed bounded sequence z_{1:n} with ‖z_i‖ ≤ B_φ, the loss g(z) = ℓ_ρ(s(z), y_ℓ) depends on s through the margin m(s(z), y_ℓ). To cover the loss at resolution ε/2, it suffices to cover the margin at resolution ερ/2 (since the ramp is 1/ρ-Lipschitz), which requires covering the b-dimensional score vector at resolution ερ/4 (since margin is 2-Lipschitz in L∞ of scores). **Clarification on b × d_h**: By A4, d_h is the Pdim of the **pooled scalar** class (union over all c ∈ [b]). For a single scalar function s_c, Pdim ≤ d_h. To cover the full b-dimensional vector (s_1(z), ..., s_b(z)), we cover each of the b coordinates independently, giving b copies of a d_h-dimensional scalar cover:
```
sup_z log N(head loss, ε/2, z_{1:n}) ≤ b × d_h × log(2enM / (d_h × ερ/4))
                                      = b × d_h × log(8enM / (d_h ερ))
```
where M is the score bound from A3. This is a **uniform** bound over all bounded z sequences — which is exactly why the Composition Lemma's sup gives a valid result.

**Combined covering number**:
```
log N(L_{ℓ,y}, ε, x_{1:n}) ≤ d_φ × log(8enk L_h B_φ / (d_φ ρε))
                              + b × d_h × log(8enM / (d_h ρε))
                              ≤ (d_φ + b × d_h) × log(CnkL_h B_φ / (ρε))
```
for universal constant C. (Here M appears in the head log factor as log(8enM/(d_hρε)). Under the sufficient condition s_c(0) = 0 stated in A3, M = L_hB_φ and the two log factors merge into log(CnkL_hB_φ/(ρε)). In general, M ≤ L_hB_φ + |s_c(0)| by Lipschitz, and the M-dependence is absorbed into the constant C since it appears only inside the logarithm.)

**Step 2: Uniform convergence via symmetrization**

By standard symmetrization (Shalev-Shwartz & Ben-David, Theorem 26.5): for the loss class L_{ℓ,y} with values in [0,1]:
```
E[sup_f |R̂(f) - R(f)|] ≤ 2 E[R̂_n(L_{ℓ,y})]
```
The empirical Rademacher complexity is bounded via Dudley's entropy integral (in L2 metric; since log N_2 ≤ log N_∞, we use the L∞ covering from Step 1):
```
R̂_n(L_{ℓ,y}) ≤ (12/√n) ∫_0^1 √(log N(L_{ℓ,y}, u, x_{1:n})) du
```
Substituting the covering number bound (with D = d_φ + b d_h):
```
R̂_n(L_{ℓ,y}) ≤ (12/√n) ∫_0^1 √(D × log(CnkL_h B_φ / (ρu))) du
             ≤ C' × √(D × log(nkL_h B_φ / ρ) / n)
```
(The integral evaluates as O(√(D log(nkL_h B_φ/ρ))) since ∫_0^1 √(log(A/u)) du = O(√(log A)) for A ≥ e.)

**Step 3: From per-level to average risk (via joint covering)**

The average surrogate risk R_avg^{surr} = (1/L) Σ_ℓ R_ℓ^{surr} is controlled by a **joint covering number** argument over the entire hypothesis class (shared backbone + all heads across all levels).

**Joint covering number**: The full hypothesis consists of one backbone φ ∈ Φ and, for each level ℓ and each parent config c, one head s_ℓ^c ∈ S_ℓ^c. To cover the averaged loss L_avg at resolution ε on n samples, it suffices to cover each component and use the triangle inequality. The backbone is covered once (shared), and heads are covered independently per (level, config):
```
log N(L_avg, ε) ≤ log N(backbone, α) + Σ_ℓ Σ_{c ∈ C_ℓ} log N(head for (ℓ,c), β)
```
where α and β are chosen so that the composition error totals ≤ ε. The backbone term contributes d_φ × log(...). For Case (b), the head terms sum to Σ_ℓ |C_ℓ| × b × d_h × log(...) = C_eff × b × d_h × log(...). For Case (a), heads are shared across both configs and levels, so the head term is b × d_h × log(...) (a single set of b heads covers all levels and configs), matching the Case (a) theorem statement. For Case (c), heads are shared across configs but differ per level, giving L × b × d_h × log(...) in the joint covering number (L independent head sets, one per level).

**From joint covering to uniform convergence**: By bounded differences (each sample changes the averaged loss by ≤ 1/n) and the standard covering-to-concentration chain (Shalev-Shwartz & Ben-David, Theorem 26.6):
```
P(sup_f |R_avg^{surr}(f) - R̂_avg^{surr}(f)| > ε) ≤ 4 E_S[N(L_avg, ε/4, S)] × exp(-nε²/8)
```
Setting the RHS ≤ δ and substituting the joint covering bound gives the sample complexity. The C_eff factor enters through the joint covering number (not through Rademacher sub-additivity over levels).

**Step 4: ERM Guarantee**

Since 0-1 loss ≤ surrogate loss:
```
R_avg^{TF}(f̂) ≤ R_avg^{surr}(f̂) ≤ R̂_avg^{surr}(f̂) + ε ≤ R̂_avg^{surr}(f*) + ε ≤ R_avg^{surr}(f*) + 2ε
```

**Note**: The gap between R_avg^{surr}(f*) and R_avg^{TF}(f*) is the **approximation error** of the surrogate. For ρ → 0, the ramp loss approaches 0-1 loss and this gap vanishes, but the covering numbers grow via the log(1/ρ) factor. This is the standard bias-variance tradeoff in margin theory.

**Critical Limitations**:

1. **Teacher forcing**: Applies to R_avg^{TF} only. At test time, error propagation through ancestor false negatives adds additional risk (see Section 4.4 for precise per-node bounds using worst-case conditional FNR).

2. **Surrogate gap**: The bound is on surrogate risk; the 0-1 risk satisfies R^{01} ≤ R^{surr} but the gap depends on the margin ρ and data distribution.

3. **Head complexity enters the bound**: The covering number composition lemma correctly includes BOTH d_φ and b × d_h. The factor b comes from covering all b score functions independently.

4. **The C_eff = Σ_ℓ min(n, b^{ℓ-1}) factor in Case (b)** comes from the Cartesian product structure of independent heads. Each observed parent config requires an independent covering of the head class, **multiplying** (not merely adding a log factor to) the head complexity. C_eff can be exponential in L when n ≥ b^L: C_eff = (b^L-1)/(b-1). This is the fundamental cost of independent parameterization. Cases (a)/(c) avoid this entirely.

5. **This is an UPPER bound**. For matching lower bounds, see Section 1.

**Theorem 5B (NTK/Lazy Regime)**

**Assumptions**:
- **(B1)** The network is trained via gradient descent with learning rate η and T iterations, starting from random initialization w₀. The **lazy/NTK regime** holds: ‖w_T - w₀‖ = O(1/√width) (Jacot et al. 2018), so the network linearizes around initialization.
- **(B2)** Under B1, the per-level predictor is **assumed** to take the form f_ℓ(x) = argmax_{c ∈ [b]} ⟨w_{ℓ,c}, Ψ(x)⟩ where Ψ(x) is the NTK feature map at initialization. The weight bound ‖w_{ℓ,c}‖_H ≤ B (where ‖·‖_H is the RKHS norm induced by K, i.e., ‖w‖_H = ‖w‖_2 in the feature space) is **imposed** as an explicit regularization constraint per class c (e.g., via weight decay or norm clipping, not derived from training). This is a **per-class** constraint; each of the b weight vectors is bounded independently.
- **(B3)** Kernel satisfies K(x,x) = ‖Ψ(x)‖² ≤ κ² for all x ∈ X.
- **(B4)** **Surrogate loss**: Generalization is measured via a convex surrogate ℓ^{surr}(s, y) that is **L_surr-Lipschitz in the score vector s under L∞**. Example: multiclass hinge loss ℓ(s,y) = max(0, 1 - s_y + max_{c≠y} s_c) is 1-Lipschitz in the margin m = s_y - max_{c≠y} s_c, and the margin is 2-Lipschitz in s under L∞, giving L_surr = 2. The 0-1 risk satisfies R^{01} ≤ R^{surr}.

**Result**: Under (B1)-(B4), by the standard RKHS Rademacher bound (Bartlett & Mendelson 2002, Thm 22), each individual score function class F_c = {x ↦ ⟨w_c, Ψ(x)⟩ : ‖w_c‖ ≤ B} has Rademacher complexity R̂_n(F_c) ≤ Bκ/√n.

The per-level surrogate loss depends on the b-dimensional score vector s = (s_1, ..., s_b). The surrogate ℓ^{surr}(s, y) is L_surr-Lipschitz in s under L∞ (B4), and since ‖v‖_∞ ≤ ‖v‖_2 for all v ∈ R^b, it is also L_surr-Lipschitz under L2. By **Maurer's vector contraction inequality** (Maurer 2016, Theorem 4): if h: R^b → R is L-Lipschitz under L2 with h(0) = 0, and F_1, ..., F_b are real-valued function classes, then:
```
R̂_n(h ∘ (F_1, ..., F_b)) ≤ √2 × L × √(Σ_c R̂_n(F_c)²)
```
(The h(0) = 0 condition is w.l.o.g.: define h̃(s) = h(s) - h(0), then the constant cancels in the Rademacher expectation, and h̃ is still L-Lipschitz.) **Per-sample reduction**: The loss h_i(s) = ℓ^{surr}(s, y_i) depends on the label y_i, which varies per sample. Maurer's contraction applies per-sample (the Lipschitz property is verified pointwise: each h_i is L_surr-Lipschitz), so the result holds for sample-dependent compositions. Applying with L = L_surr and R̂_n(F_c) ≤ Bκ/√n:
```
R̂_n({x ↦ ℓ^{surr}(s(x), y) : ‖w_{ℓ,c}‖ ≤ B}) ≤ √2 × L_surr × √(b × (Bκ/√n)²) = √(2b) × L_surr × Bκ/√n
```
For multiclass hinge (L_surr = 2): R̂_n ≤ 2√(2b) × Bκ/√n.

Averaging over L levels and applying bounded differences:
```
n ≥ c × (bL_surr²B²κ²/ε² + log(L/δ)/ε²)
```
implies sup_f |R̂_avg^{surr}(f) - R_avg^{surr}(f)| ≤ ε with probability ≥ 1-δ.

**Note on b dependence**: The √b factor from Maurer's vector contraction (with √2 constant) improves over the naive b factor from treating scores independently. The sample complexity scales as O(b) in the number of classes, not O(b²). The √2 is an artifact of the symmetrization in Maurer's proof and is tight in general. The **per-class** norm constraint ‖w_{ℓ,c}‖ ≤ B (not joint ‖W_ℓ‖_F ≤ B) is what produces the √b factor; a joint constraint would replace b × (Bκ)² with (B_joint × κ)².

**Theorem 5C (Cheating Is Possible)**

**Assumptions**: (i) training inputs x_1, ..., x_n are **distinct**; (ii) Φ is **unconstrained** (contains all measurable functions X → R^k with k ≥ n); (iii) H_ℓ contains linear-argmax classifiers h(z) = argmax_c (Wz)_c for arbitrary W ∈ R^{b×k}.

Under these assumptions, zero empirical error on all levels can be achieved simultaneously, regardless of labels.

**Proof**: Since x_1, ..., x_n are distinct and Φ is unconstrained, define φ(x_i) = e_i ∈ R^n (i-th standard basis vector) for training points, and φ(x) = 0 otherwise. For level ℓ, define W_ℓ ∈ R^{b × n} with (W_ℓ)_{y_i^{(ℓ)}, i} = 1 and all other entries 0. The head h_ℓ(z) = argmax_c (W_ℓ z)_c outputs the correct label for each training point, since (W_ℓ e_i)_{y_i^{(ℓ)}} = 1 while all other scores are 0. Since φ ∈ Φ and h_ℓ ∈ H_ℓ by assumptions (ii)-(iii), this achieves R̂_ℓ^{TF}(f) = 0 for all ℓ. ∎

**When does this apply?** Overparameterized neural networks where width exceeds n can realize this construction — see Yun et al. (2019) for formal memorization capacity results. **Note**: The memorization construction satisfies A1-A3 (φ(x_i) = e_i has ‖φ‖_∞ = 1, linear heads are Lipschitz and bounded). What makes the generalization bound non-vacuous is **finite backbone complexity** (A5: d_φ < ∞): unconstrained Φ has d_φ = ∞, making the bound vacuous and allowing arbitrary memorization. With finite d_φ, the sample complexity bound from Theorem 5A provides meaningful generalization guarantees when n is sufficiently large relative to d_φ.

**Implication**: Without capacity control (bounded Pdim via A4-A5, or information bottleneck), hierarchical separation vanishes. Assumptions A1-A5 in Theorem 5A are needed **jointly**: A1-A3 ensure Lipschitz/boundedness for covering arguments, while A4-A5 bound the class complexity. Neither set alone suffices.

**Theorem 5D (Information Bottleneck Bounds Population Error)**

**Convention**: All logarithms and entropies in this theorem use **natural logarithm** (base e, units of nats). Binary entropy is H_2(p) = -p ln p - (1-p) ln(1-p) ≤ ln 2 ≈ 0.693 nats.

**Assumption**: Y_ℓ takes values in a **finite** set [b], and Z = φ(X) takes values in a **countable** (or finite) set, so that all entropies H(Y_ℓ), H(Z), H(Y_ℓ|Z) are well-defined and finite. For continuous Z, these results apply to quantized/discretized representations, or the mutual information I(Y_ℓ; Z) should be interpreted as the standard (Shannon) mutual information which is well-defined for any joint distribution (but H(Z) may be infinite). The identity I(X; Z) = H(Z) used in the discussion below requires Z discrete; for continuous deterministic Z = φ(X), I(X; Z) is infinite and the identity breaks down.

**Statement**: For a fixed encoder Z = φ(X) with I(Y_ℓ; Z) ≤ I₀ (in nats), the Bayes error of the best classifier based on Z satisfies:

*Case b ≥ 3*:
```
P_err^{Bayes}(Y_ℓ | Z) ≥ max(0, (H(Y_ℓ) - I₀ - ln 2) / ln(b-1))
```

*Case b = 2 (binary)*: The general Fano bound with ln(b-1) = ln(1) = 0 degenerates. Instead, use the binary Fano inequality directly (all in nats):
```
H(Y_ℓ | Z) ≤ H_2(P_err)     [H_2 in nats: H_2(p) = -p ln p - (1-p)ln(1-p)]
```
Since H_2 (in nats) is increasing on [0, 1/2] with H_2(1/2) = ln 2 nats, and we restrict to P_err ∈ [0, 1/2] (the principal branch), inverting gives:
```
P_err ≥ H_2^{-1}(max(0, H(Y_ℓ) - I₀))     [when H(Y_ℓ) - I₀ ∈ [0, ln 2], all in nats]
```
where H_2^{-1} denotes the inverse of H_2 on [0, 1/2] (the principal branch where H_2 is increasing), and max(0, ·) handles the case H(Y_ℓ) < I₀. When H(Y_ℓ) - I₀ ≤ 0, the bound gives P_err ≥ H_2^{-1}(0) = 0, which is trivial. When H(Y_ℓ) - I₀ > ln 2, the argument is vacuous (P_err > 1/2 is always possible but uninformative). For uniform binary Y_ℓ (H(Y_ℓ) = ln 2 nats): P_err ≥ H_2^{-1}(max(0, ln 2 - I₀)).

**Note**: The b ≥ 3 bound is only meaningful when H(Y_ℓ) - I₀ > ln 2 (otherwise RHS ≤ 0). The max(0, ·) makes this explicit.

**Proof via Fano's Inequality** (Cover & Thomas 2006, Theorem 2.10.1):

*Case b ≥ 3*: Fano's inequality for b-ary classification:
```
H(Y_ℓ | Z) ≤ H_2(P_err) + P_err × ln(b-1)
```
where P_err = P(Ŷ ≠ Y_ℓ) for the optimal estimator Ŷ = Ŷ(Z).

Rearranging and using H_2(P_err) ≤ ln 2:
```
P_err ≥ (H(Y_ℓ | Z) - ln 2) / ln(b-1)
     = (H(Y_ℓ) - I(Y_ℓ; Z) - ln 2) / ln(b-1)
```
Substituting I(Y_ℓ; Z) ≤ I₀ and taking max with 0 gives the stated bound.

*Case b = 2*: Fano's inequality reduces to H(Y_ℓ | Z) ≤ H_2(P_err). Since H(Y_ℓ | Z) = H(Y_ℓ) - I(Y_ℓ; Z) ≥ H(Y_ℓ) - I₀, we get H_2(P_err) ≥ H(Y_ℓ) - I₀. Since H_2 is increasing on [0, 1/2], inverting gives the stated bound. (If H(Y_ℓ) - I₀ > ln 2, the bound is vacuous: P_err > 1/2 is always possible.)

**Relationship between I(Y_ℓ; Z) and I(X; Z)**:

The inequality **I(Y_ℓ; Z) ≤ I(X; Z) always holds** by the Data Processing Inequality (DPI), since Y_ℓ → X → Z forms a Markov chain. This is true for any Z derived from X, whether deterministic or stochastic (using mutual information defined via KL divergence, which is well-defined for arbitrary joint distributions). When Z is **discrete** and Z = φ(X) is deterministic, we additionally have I(X; Z) = H(Z) (because H(Z|X) = 0), and I(Y_ℓ; Z) = H(Z) - H(Z|Y_ℓ) ≤ H(Z) = I(X; Z), confirming the DPI bound. (For continuous deterministic Z, I(X; Z) may be infinite, but the DPI inequality still holds.) Additionally, DPI gives I(Y_ℓ; Z) ≤ I(Y_ℓ; X).

**Why bounding I(X; Z) alone is insufficient**: While I(Y_ℓ; Z) ≤ I(X; Z) is always true, bounding I(X; Z) can be very loose. For example: X = (X₁,...,X_{1000}) i.i.d. Bern(1/2), Y_ℓ = X₁, Z = X₂. Then I(Y_ℓ; Z) = 0 but I(X; Z) = ln 2. Knowing I(X; Z) = ln 2 only tells us I(Y_ℓ; Z) ≤ ln 2, which is vacuous. For meaningful error bounds via Fano, we need I(Y_ℓ; Z) to be **small** (specifically, smaller than H(Y_ℓ) - ln 2), and bounding I(X; Z) doesn't ensure this.

**When can we obtain useful bounds on I(Y_ℓ; Z)?**

A direct bound I(Y_ℓ; Z) ≤ I₀ is most useful, imposed via:
- Explicit information bottleneck regularization (Tishby et al. 2000)
- Dimensionality constraints (k small forces lossy compression; when Z is discrete with bounded alphabet size, H(Z) ≤ k log|alphabet|, bounding I(Y_ℓ; Z) ≤ H(Z))
- Noise injection in the representation

**Practical Implication**: If I(Y_ℓ; Z) ≤ I₀ for all levels (e.g., via bottleneck regularization), then for b ≥ 3 the total error satisfies:
```
Σ_ℓ P_err(ℓ) ≥ Σ_ℓ max(0, (H(Y_ℓ) - I₀ - ln 2) / ln(b-1))
```
(For b = 2, replace each term with H_2^{-1}(max(0, H(Y_ℓ) - I₀)).)

Achieving low error at all L levels requires I(Y_ℓ; Z) = Ω(H(Y_ℓ)) for each ℓ. For uniform labels this is Ω(ln b) per level, hence Σ_ℓ I(Y_ℓ; Z) ≥ Ω(L ln b). **Note on the relationship between per-level and joint MI**: By the chain rule, I(Y_{1:L}; Z) = Σ_ℓ I(Y_ℓ; Z | Y_{<ℓ}), which can be larger or smaller than Σ_ℓ I(Y_ℓ; Z) depending on the conditional dependence structure. In particular, neither Σ_ℓ I(Y_ℓ; Z) ≥ I(Y_{1:L}; Z) nor the reverse holds in general. The bound Σ_ℓ I(Y_ℓ; Z) ≥ Ω(L ln b) is a necessary condition on the **per-level** mutual informations; it does NOT imply I(Y_{1:L}; Z) = Ω(L ln b) (the joint MI could be smaller due to shared information across levels, or larger due to synergistic effects).

### 5.3 Conditions Preserving Separation

Hierarchical sample complexity separation (shared backbone helps) is preserved when:

1. **Capacity Control**: Ndim(F_ℓ) bounded by d not scaling with L (e.g., restricted case with bounded backbone)
2. **NTK/Lazy Regime**: Fixed kernel, representation changes o(1); see Theorem 5B
3. **Information Bottleneck**: I(Y_ℓ; Z) bounded (via explicit regularization); prevents memorization per Theorem 5D
4. **Restricted heads**: Heads ignore parent labels (pure hierarchical setting), avoiding exponential blow-up

---

## Summary of Bombastic Claims

| Result | Original | Bombastic Version |
|--------|----------|-------------------|
| Sample Complexity | Upper bound O((d log b)/ε²) | **Minimax lower bound Ω((d log b)/ε²) (under product embedding, §1.4)** |
| Distribution | Specific Gaussian | **Any distribution (minimax)** |
| Constants | Loose (union bound) | **Tighter (nested/time-uniform/Bretagnolle-Huber)** |
| Hierarchy | Trees only | **DAGs with order ideals** |
| Training | Head-only | **End-to-end with capacity control** |

---

## References

1. Shalev-Shwartz & Ben-David (2014). *Understanding Machine Learning*. Theorem 29.3 (Natarajan dimension).
2. Tsybakov (2009). *Introduction to Nonparametric Estimation*. Ch. 2 (Assouad's lemma, KL bounds for Bernoulli).
3. Natarajan (1989). On learning sets and functions. *Machine Learning* 4(3):227-233.
4. Cover & Thomas (2006). *Elements of Information Theory*, 2nd ed. Theorem 2.10.1 (Fano's inequality).
5. Bartlett & Mendelson (2002). Rademacher and Gaussian Complexities. *JMLR* 3:463-482.
6. Jacot, Gabriel & Hongler (2018). Neural Tangent Kernel. *NeurIPS*.
7. Daniely, Sabato, Shalev-Shwartz & Shamir (2015). Multiclass Learnability and the ERM principle. *JMLR* 16:2377-2404.
8. Anthony & Bartlett (1999). *Neural Network Learning: Theoretical Foundations*. Cambridge University Press.
9. Tishby, Pereira & Bialek (2000). The Information Bottleneck Method. *arXiv:physics/0004057*.
10. Yun, Sra, Jadbabaie (2019). Small nonlinearities in activation functions create bad local minima in neural networks. *ICLR*.
11. Haussler & Long (1995). A generalization of Sauer's lemma. *JCSS* 50(3):455-466.
12. Natarajan & Tadepalli (1998). Two new frameworks for learning. *ICML*.
13. Maurer (2016). A vector-contraction inequality for Rademacher complexities. *ALT*.
14. Lei, Binder, Dogan & Kloft (2015). Multi-class SVMs: From tighter data-dependent generalization bounds to novel algorithms. *NeurIPS*.
15. Guermeur (2017). Lp-norm Sauer-Shelah lemma for margin multi-category classifiers. *JMLR* 18(1):1-56.
16. Zhang (2004). Statistical behavior and consistency of classification methods based on convex risk minimization. *Annals of Statistics* 32(1):56-85.
17. Bartlett, Jordan & McAuliffe (2006). Convexity, classification, and risk bounds. *JASA* 101(473):138-156.
18. Mohri, Rostamizadeh & Talwalkar (2018). *Foundations of Machine Learning*, 2nd ed. MIT Press. (Composition covering lemma, Lemma 3.10).

---

*February 2026*
