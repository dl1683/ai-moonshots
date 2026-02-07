# Sample Complexity Theory for Fractal Classifiers

Sample complexity separation, information-theoretic lower bounds (Fano), and scale-separated embedding theory.

---

## Table of Contents

1. [Sample Complexity Separation Theorem](#1-sample-complexity-separation-theorem)
2. [Information-Theoretic Lower Bound (Fano)](#2-information-theoretic-lower-bound-fano)
3. [Scale-Separated Embeddings](#3-scale-separated-embeddings)
4. [References](#4-references)

---

## 1. Sample Complexity Separation Theorem

### 1.1 Theorem Statement

**Theorem 1 (Sample-Complexity Separation for Fractal vs. Flat Classification)**

Let b ≥ 2 and L ≥ 1. The label space is a rooted b-ary tree of depth L; each leaf label is identified with a sequence Y = (Y₁, ..., Y_L) ∈ [b]^L, so the number of leaves is C = b^L. Let X be the instance space and D a distribution over X × [b]^L.

For each level ℓ ∈ {1, ..., L}, let H_ℓ be a hypothesis class of functions h_ℓ: X × [b]^{ℓ-1} → [b] (predict the ℓth child given the true parent prefix), with Natarajan dimension at most d.

**For the flat classifier comparison**, let H_flat ⊆ [C]^X be a hypothesis class of functions h_flat: X → [C] with **Natarajan dimension at most d**. This ensures a fair comparison where both classifier types use hypothesis classes of the same complexity.

Assume **head-only training**: the backbone is fixed, and each level head h_ℓ is trained independently by ERM on the induced training set {(x_i, y_{i,1:ℓ-1}), y_{i,ℓ}}_{i=1}^n.

Define the **conditional level-ℓ risk**:
```
R_ℓ(h_ℓ) = Pr_{(X,Y)~D}[h_ℓ(X, Y_{1:ℓ-1}) ≠ Y_ℓ]
```

**Then:**

1. **Fractal (conditional-parent) upper bound.** There exists a universal constant c > 0 such that if
   ```
   n ≥ c × (d log b + log(L/δ)) / ε²
   ```
   then with probability at least 1-δ, all levels satisfy
   ```
   R_ℓ(ĥ_ℓ) ≤ R*_ℓ + ε  for every ℓ = 1, ..., L
   ```

   **Sample accounting:** The same n samples {(x_i, y_i)}_{i=1}^n are used for all L levels. Each level extracts different label information (y_{i,ℓ}) from the same (x_i, y_i) pair. Total sample complexity: **n = O((d log b + log(L/δ)) / ε²)**.

2. **Flat upper bound.** There exists a universal constant c' > 0 such that if
   ```
   n ≥ c' × (d log C + log(1/δ)) / ε²
   ```
   then with probability at least 1-δ,
   ```
   R_flat(ĥ_flat) ≤ R*_flat + ε
   ```
   Since C = b^L, this is n = O((d × L × log b) / ε²)

### 1.2 Proof

**Step 1: Multiclass ERM Sample-Complexity Tool**

We use the multiclass fundamental theorem (Shalev-Shwartz & Ben-David, Theorem 29.3):

For any hypothesis class H ⊆ [k]^X with Natarajan dimension d, ERM is agnostic PAC-learnable with sample complexity:
```
m(ε, δ) = O((d log k + log(1/δ)) / ε²)
```

More precisely, there exist constants C₁, C₂ > 0 such that:
```
C₁ × (d + log(1/δ)) / ε² ≤ m(ε, δ) ≤ C₂ × (d log k + log(1/δ)) / ε²
```

**Step 2: Fractal (Conditional-Parent) Upper Bound**

Fix any level ℓ. The mapping (X, Y) ↦ (X, Y_{1:ℓ-1}, Y_ℓ) is deterministic, so an i.i.d. sample {(x_i, y_i)}_{i=1}^n ~ D^n induces an i.i.d. sample for the level-ℓ problem.

The label set size at this level is k = b. By the tool in Step 1, for any δ_ℓ > 0, if
```
n ≥ c × (d log b + log(1/δ_ℓ)) / ε²
```
then with probability at least 1 - δ_ℓ:
```
R_ℓ(ĥ_ℓ) ≤ R*_ℓ + ε
```

Set δ_ℓ = δ/L and apply the union bound over ℓ = 1, ..., L. Then with probability at least 1 - δ, the inequality holds simultaneously for all levels. ∎

**Step 3: Flat Upper Bound**

For the flat classifier, the label set size is k = C = b^L. Apply Theorem 29.3 directly to H_flat (Natarajan dimension ≤ d):

If n ≥ c' × (d log C + log(1/δ)) / ε², then with probability at least 1 - δ:
```
R_flat(ĥ_flat) ≤ R*_flat + ε
```

Since log C = L log b, the flat sample complexity is O((d × L × log b) / ε²). ∎

### 1.3 Corollary (Leaf Error Conversion)

Let the composed fractal predictor output a leaf by sequentially applying ĥ₁, ..., ĥ_L. Define E_ℓ = "the first mistake occurs at level ℓ." These events are disjoint, and:
```
Pr(leaf error) = Σ_{ℓ=1}^L Pr(E_ℓ) ≤ Σ_{ℓ=1}^L R_ℓ(ĥ_ℓ)
```

Therefore, to achieve leaf error ≤ ε, it suffices to enforce R_ℓ(ĥ_ℓ) ≤ ε/L for all ℓ. Using the same multiclass bound with ε/L yields:
```
n = O(L² × (d log b + log(L/δ)) / ε²)
```

---

## 2. Information-Theoretic Lower Bound (Fano)

### 2.1 Theorem Statement

**Theorem 2 (Information-Theoretic Lower Bound for Flat Classifiers)**

For any integer C ≥ 2, there exists an explicit family of C Gaussian distributions {P_θ}_{θ=1}^C such that for any estimator θ̂(X^n) based on n i.i.d. samples from the unknown P_θ with θ uniform on [C], the probability of error satisfies:
```
Pr(θ̂ ≠ θ) ≥ 1 - (n × δ²/σ² + log 2) / log C
```

In particular, if δ/σ is a fixed constant, then to achieve success probability > 1/2, it is necessary that:
```
n = Ω(log C)
```

### 2.2 Explicit Construction

Let d = C. Let e₁, ..., e_C be the standard basis of ℝ^C. Fix constants σ² > 0 and δ > 0. For each θ ∈ [C], define:
```
P_θ = N(μ_θ, σ²I_d)  with  μ_θ = δ × e_θ
```

This is a fully explicit family of C Gaussians in ℝ^C.

### 2.3 Proof

**Step 1: KL Divergences are Exact and Constant**

The density of N(μ, σ²I_d) is:
```
p_μ(x) = (2πσ²)^{-d/2} × exp(-||x - μ||² / (2σ²))
```

For Gaussians with common covariance σ²I_d:
```
KL(N(μ_i, σ²I_d) || N(μ_j, σ²I_d)) = ||μ_i - μ_j||² / (2σ²)
```

For i ≠ j:
```
||μ_i - μ_j||² = ||δe_i - δe_j||² = 2δ²
```

Hence:
```
KL(P_i || P_j) = 2δ² / (2σ²) = δ²/σ²  for i ≠ j
KL(P_i || P_i) = 0
```

For n i.i.d. samples, P_i^n is the n-fold product, so:
```
KL(P_i^n || P_j^n) = n × KL(P_i || P_j) = n × δ²/σ²
```

**Step 2: Mutual Information Upper Bound**

Let θ be uniform on [C] and let P̄^n = (1/C) × Σ_{j=1}^C P_j^n. The mutual information is exactly:
```
I(θ; X^n) = (1/C) × Σ_{i=1}^C KL(P_i^n || P̄^n)
```

We upper bound it by the average pairwise KL. Using convexity of -log:
```
-log((1/C) × Σ_{j=1}^C p_j^n) ≤ (1/C) × Σ_{j=1}^C (-log p_j^n)
```

So:
```
KL(P_i^n || P̄^n) ≤ (1/C) × Σ_{j=1}^C KL(P_i^n || P_j^n)
```

Averaging over i yields:
```
I(θ; X^n) ≤ (1/C²) × Σ_{i=1}^C Σ_{j=1}^C KL(P_i^n || P_j^n)
```

Plugging in the exact KL values:
```
I(θ; X^n) ≤ (1/C²) × C(C-1) × n × δ²/σ² = (1 - 1/C) × n × δ²/σ² ≤ n × δ²/σ²
```

**Step 3: Apply Fano's Inequality**

Fano for M = C says:
```
Pr_err ≥ 1 - (I(θ; X^n) + log 2) / log C
```

Using the bound on I:
```
Pr_err ≥ 1 - (n × δ²/σ² + log 2) / log C
```

Therefore, to achieve Pr_err < 1/2, it is necessary that:
```
n × δ²/σ² + log 2 ≥ (1/2) × log C
```
i.e.,
```
n ≥ (σ²/δ²) × ((1/2) × log C - log 2)
```

For any fixed signal-to-noise ratio δ/σ = Θ(1), this is **n = Ω(log C)**. ∎

### 2.4 Hierarchical Extension (b-ary Tree)

Assume a balanced b-ary tree of depth L with leaf labels θ ∈ [b]^L. Then C = b^L.

**Construction:** For each level ℓ ∈ {1, ..., L}, define b Gaussian components in a fresh block ℝ^b:
```
Q_k^{(ℓ)} = N(δ × e_k, σ²I_b),  k ∈ [b]
```

For a leaf label θ = (i₁, ..., i_L) ∈ [b]^L, define:
```
P_θ = N(μ_θ, σ²I_{Lb}),  μ_θ = (δe_{i₁}, ..., δe_{i_L}) ∈ (ℝ^b)^L
```

**Per-Level Lower Bound:** Fix a level ℓ. The block-ℓ marginal is one of the b Gaussians {Q_k^{(ℓ)}}_{k=1}^b. Any estimator θ̂ induces an estimator î_ℓ of the level-ℓ index.

If θ̂ ≠ θ, then at least one level is wrong, hence:
```
Pr(î_ℓ ≠ i_ℓ) ≤ Pr(θ̂ ≠ θ)
```

Each level is exactly the flat M = b problem. Applying Fano with M = b implies:
```
Pr(î_ℓ ≠ i_ℓ) ≥ 1 - (n × δ²/σ² + log 2) / log b
```

Therefore, to have Pr(î_ℓ ≠ i_ℓ) < 1/2:
```
n ≥ (σ²/δ²) × ((1/2) × log b - log 2) = Ω(log b)
```

**Conclusion:** Hierarchical b-ary structure reduces per-level complexity from Ω(log C) to **Ω(log b)**. ∎

---

## 3. Scale-Separated Embeddings

### 3.1 Setup and Definitions

Let D = k × d₀. Coordinates are partitioned into k disjoint blocks B₀, ..., B_{k-1}, each of size d₀. For any subset S ⊂ [D], let P_S denote coordinate projection.

**Definition 3.1 (Scale-Separated Embedding)**

A scale-separated embedding for a k-level hierarchy is:
```
f(L) = (f₀(L₀), f₁(L₀, L₁), ..., f_{k-1}(L₀, ..., L_{k-1}))
```
with f_ℓ supported only on block B_ℓ. Define the level-j prefix embedding:
```
F_j(L) = (f₀(L₀), ..., f_j(L₀,...,L_j)) ∈ ℝ^{(j+1)d₀}
```

**Level-j Separation:** For each level j, define:
```
Δ_j := min{||F_j(L) - F_j(L')||₂ : L₀...L_{j-1} = L'₀...L'_{j-1}, L_j ≠ L'_j}
```

This is the minimum Euclidean distance between prefix embeddings of labels that share the same parent but differ at level j.

**Definition 3.2 (ε-Isotropy)**

Fix m. A vector v ∈ ℝ^D is **ε-isotropic at scale m** if for every subset S with |S| = m:
```
||P_S v||₂² ≤ (1+ε) × (m/D) × ||v||₂²
```

An embedding g is ε-isotropic at level j if every difference vector v = g(L) - g(L') for label pairs that differ first at level j is ε-isotropic at scale m.

**Observation Model:** Y = f(L) + Z where Z ~ N(0, σ²I_D)

**Access Model:** A decoder chooses a subset S ⊂ [D] with |S| = m and only observes Y_S = P_S(Y).

### 3.2 Supporting Lemmas

*Note: Lemmas 1 and 5 are directly used in Theorems B and C respectively. Lemmas 2-4 provide chi-squared and Beta concentration bounds that support extensions to non-Gaussian noise models and refined coordinate counting arguments.*

**Lemma 1 (Gaussian Tail Bound).** If G ~ N(0,1) and t ≥ 0, then:
```
P(G ≥ t) ≤ exp(-t²/2)
```

**Proof.** For any λ > 0:
```
P(G ≥ t) = P(exp(λG) ≥ exp(λt)) ≤ exp(-λt) × E[exp(λG)] = exp(λ²/2 - λt)
```
Minimizing over λ > 0 gives λ = t, hence P(G ≥ t) ≤ exp(-t²/2). ∎

---

**Lemma 2 (Chi-Squared Tail Bounds).** Let X ~ χ²_r. For t > 0:
```
P(X - r ≥ t) ≤ exp(-t²/(4(r+t)))
P(r - X ≥ t) ≤ exp(-t²/(4r))
```

In particular, for 0 < δ ≤ 1:
```
P(|X - r| ≥ δr) ≤ 2 × exp(-δ²r/8)
```

**Proof.** The mgf is M_X(λ) = (1-2λ)^{-r/2} for λ < 1/2. For the upper tail, for λ > 0:
```
P(X - r ≥ t) ≤ exp(-λ(r+t)) × (1-2λ)^{-r/2}
```
Minimizing the exponent over λ yields λ = t/(2(r+t)), giving the stated bound. The lower tail follows similarly with λ < 0. ∎

---

**Lemma 3 (Beta Distribution From Chi-Square).** Let U ~ χ²_m and V ~ χ²_{D-m} be independent. Then:
```
R = U/(U+V) ~ Beta(m/2, (D-m)/2)
```

**Proof.** The joint density of (U,V) is the product of chi-square densities. With change of variables T = U+V, R = U/(U+V), the Jacobian is |∂(U,V)/∂(R,T)| = T. Integrating out T yields the Beta density in R. ∎

---

**Lemma 4 (Beta Concentration From Chi-Square Bounds).** Let R ~ Beta(m/2, (D-m)/2) and μ = m/D. For 0 < ε ≤ 1:
```
P(|R/μ - 1| ≥ ε) ≤ 2exp(-ε²m/128) + 2exp(-ε²(D-m)/128)
```

**Proof.** Let U, V be as in Lemma 3 so R = U/(U+V). Fix δ = ε/4. If |U-m| ≤ δm and |V-(D-m)| ≤ δ(D-m), write U = m(1+u), V = (D-m)(1+v) with |u|, |v| ≤ δ. Then:
```
R = m(1+u) / (m(1+u) + (D-m)(1+v)) = μ × (1+u) / (1 + μu + (1-μ)v)
```

Let w = μu + (1-μ)v. Then |w| ≤ δ and:
```
|R/μ - 1| = |u - w| / (1 + w) ≤ (|u| + |w|) / (1 - δ) ≤ 2δ / (1 - δ) ≤ 4δ = ε
```
for δ ≤ 1/2. Apply Lemma 2 with δ = ε/4 to obtain the bound. ∎

---

**Lemma 5 (Bretagnolle-Huber Inequality).** Let P, Q be distributions, and let A be any event. Then:
```
P(Aᶜ) + Q(A) ≥ (1/2) × exp(-KL(P||Q))
```

**Proof.** Let p, q denote densities, and BC = ∫√(pq) (Bhattacharyya coefficient). By Cauchy-Schwarz:
```
∫|p-q| = ∫|√p - √q| × (√p + √q) ≤ (∫(√p - √q)²)^{1/2} × (∫(√p + √q)²)^{1/2}
```

Compute:
```
∫(√p - √q)² = 2(1 - BC),  ∫(√p + √q)² = 2(1 + BC)
```

Hence: TV(P,Q) = (1/2)∫|p-q| ≤ √(1 - BC²)

By monotonicity of Rényi divergence, KL(P||Q) ≥ D_{1/2}(P||Q) = -2 log BC. Therefore BC ≥ exp(-KL/2), hence:
```
TV(P,Q) ≤ √(1 - exp(-KL))
```

For any event A: P(Aᶜ) + Q(A) ≥ 1 - TV(P,Q). Using 1 - √(1-x) ≥ x/2 for 0 ≤ x ≤ 1:
```
P(Aᶜ) + Q(A) ≥ (1/2) × exp(-KL(P||Q))
```
∎

---

### 3.3 Theorem B: Scale-Separated Upper Bound

**Theorem B (Scale-Separated Upper Bound).** Let m = (j+1)d₀ and S = B₀ ∪ ... ∪ B_j (coordinate blocks). Let b_j denote the **branching factor** (number of children) at level j. Then there exists a decoder using only Y_S such that the **multiclass error** (probability of predicting any wrong class) satisfies:
```
P_err(L_j | L_{<j}) ≤ (b_j - 1) × exp(-Δ_j² / (8σ²))
```

*Note: B_j denotes coordinate block j; b_j denotes branching factor at level j. These are distinct.*

**Proof:**

**Step 1: Setup.** Fix L_{<j} = (L₀, ..., L_{j-1}). The decoder must distinguish among b_j possible values of L_j (multiclass problem with b_j classes).

**Step 2: Codeword structure.** For each possible L_j value a ∈ {0, ..., b_j - 1}, define:
```
μ_a = f_{≤j}(L_{<j}, a) ∈ ℝ^m,  where m = (j+1)d₀
```

The decoder observes Y_{≤j} = μ_{L_j} + Z_{≤j}, where Z_{≤j} ~ N(0, σ²I_m).

**Step 3: Nearest-neighbor decoder.** The decoder outputs:
```
L̂_j = argmin_{a ∈ {0,...,b_j-1}} ||Y_{≤j} - μ_a||²
```

**Step 4: Pairwise error analysis.** For any two distinct classes a ≠ b:
```
P(||Y - μ_b||² < ||Y - μ_a||² | L_j = a)
```

Since Y = μ_a + Z:
```
= P(||Z - (μ_b - μ_a)||² < ||Z||²)
= P(||μ_a - μ_b||² - 2⟨μ_b - μ_a, Z⟩ < 0)
= P(⟨μ_b - μ_a, Z⟩ > ||μ_a - μ_b||²/2)
```

**Step 5: Gaussian tail bound.** The random variable ⟨μ_b - μ_a, Z⟩ is Gaussian with:
- Mean: 0
- Variance: ||μ_b - μ_a||² × σ²

Let d_{ab} = ||μ_a - μ_b|| ≥ Δ_j. Then:
```
P(⟨μ_b - μ_a, Z⟩ > d_{ab}²/2) = P(N(0, d_{ab}²σ²) > d_{ab}²/2)
                                = P(N(0,1) > d_{ab}/(2σ))
                                ≤ exp(-d_{ab}²/(8σ²))     [Gaussian tail: P(N(0,1) > t) ≤ exp(-t²/2)]
                                ≤ exp(-Δ_j²/(8σ²))
```

**Step 6: Union bound.** The decoder errs if any incorrect class c ≠ L_j has smaller distance. Summing over all b_j - 1 incorrect classes:
```
P_err(L_j = a | L_{<j}) ≤ Σ_{c ≠ a} P(prefer c over a) ≤ (b_j - 1) × exp(-Δ_j²/(8σ²))
```

**Step 7: Worst case.** Taking the maximum over all true classes a:
```
P_err(L_j | L_{<j}) = max_a P_err(L_j = a | L_{<j}) ≤ (b_j - 1) × exp(-Δ_j²/(8σ²))
```
∎

### 3.4 Theorem C: Isotropic Lower Bound

**Theorem C (Isotropic Lower Bound).** Let g be a (level-j, ε)-isotropic embedding with ||g(L) - g(L')||² = Δ̃_j² for pairs differing only at L_j. For any decoder observing only m coordinates, the **worst-case binary error** (max of Type I and Type II) satisfies:
```
P_err = max(α, β) ≥ (1/4) × exp(-(1+ε) × (m/D) × Δ̃_j² / (2σ²))
```

*Note: The constant 1/4 follows directly from Lemma 5 (Bretagnolle-Huber): α+β ≥ (1/2)exp(-KL), hence max(α,β) ≥ (1/4)exp(-KL).*

**Proof:**

**Step 1: Binary subproblem.** Consider any two labels L, L' differing only at level j.

**Step 2: Observed distributions.** The decoder observes P_S(Y) ∈ ℝ^m, where:
- Under H₀ (L true): P_S(Y) ~ N(P_S(g(L)), σ²I_m)
- Under H₁ (L' true): P_S(Y) ~ N(P_S(g(L')), σ²I_m)

**Step 3: KL divergence.** For Gaussians with same covariance:
```
KL(H₀ || H₁) = ||P_S(g(L)) - P_S(g(L'))||² / (2σ²) = ||P_S(g(L) - g(L'))||² / (2σ²)
```

**Step 4: Apply isotropy.** By ε-isotropy:
```
||P_S(g(L) - g(L'))||² ≤ (1+ε) × (m/D) × ||g(L) - g(L')||² = (1+ε) × (m/D) × Δ̃_j²
```

Therefore:
```
KL ≤ (1+ε) × (m/D) × Δ̃_j² / (2σ²)
```

**Step 5: Apply Lemma 5 (Bretagnolle-Huber).** By Lemma 5, for any test (acceptance region A):
```
α + β = P_0(Aᶜ) + P_1(A) ≥ (1/2) × exp(-KL(P_0||P_1))
```
where α = P(decide H₁ | H₀) and β = P(decide H₀ | H₁).

**Step 6: Conclude.** For worst-case error (max of Type I and Type II):
```
P_err = max(α, β) ≥ (α + β)/2 ≥ (1/4) × exp(-KL)
```

Substituting the KL bound:
```
P_err = max(α, β) ≥ (1/4) × exp(-(1+ε) × (m/D) × Δ̃_j² / (2σ²))
```
∎

### 3.4 Main Corollary: Access Complexity Ratio

**Corollary.** For level j in a k-level hierarchy with uniform block sizes d_j = d₀:

| Embedding Type | Coordinates Needed for Error ε |
|---------------|-------------------------------|
| Scale-separated | m_f = (j+1) × d₀ |
| Isotropic | m_g = Ω(D) = Ω(k × d₀) |

**Access Complexity Ratio:** m_g / m_f = **Ω(k / (j+1))**

For the coarsest level (j=0), the ratio is **Ω(k)**.

---

## 4. References

1. Shalev-Shwartz, S. & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press. Theorem 29.3.

2. Natarajan, B.K. (1989). On learning sets and functions. *Machine Learning*, 4(1), 67-97.

3. Tsybakov, A. (2009). *Introduction to Nonparametric Estimation*. Springer. Lemma 2.6 (Bretagnolle-Huber inequality).

4. Cover, T. & Thomas, J. (2006). *Elements of Information Theory*, 2nd ed. Wiley. (Fano's inequality, KL divergence for Gaussians)

5. Laurent, B. & Massart, P. (2000). Adaptive estimation of a quadratic functional by model selection. *Annals of Statistics*, 28(5), 1302-1338. (Chi-squared concentration)

6. Vershynin, R. (2018). *High-Dimensional Probability*. Cambridge University Press. (Gaussian concentration, sub-Gaussian theory)

---

*February 2026*
