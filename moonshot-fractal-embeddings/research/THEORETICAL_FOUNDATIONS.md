# Theoretical Foundations: The Hierarchical Sufficiency Principle

## The Central Claim

**"Optimal representations are nested filtrations of sufficient statistics aligned to semantic hierarchy."**

This is not an engineering trick. This is a candidate fundamental law of representation learning, with roots in information theory, statistical physics, and differential geometry.

## The Gap Nobody Has Filled

No existing paper proves the full chain:

> Given a hierarchical semantic source with depth d, the rate-distortion optimal embedding at resolution k is the k-th prefix of the full embedding, and this prefix is a sufficient statistic for level-k classification.

The pieces exist. Nobody assembled them.

---

## 1. SUCCESSIVE REFINEMENT (Information Theory)

### Equitz & Cover 1991
- **Result**: A source is *successively refinable* iff optimal multi-rate descriptions form a Markov chain: X -> Q_fine -> Q_coarse
- **Connection**: Our V5 prefixes ARE this Markov chain. Prefix j=1 is Q_coarse, prefix j=4 is Q_fine.

### No 2019 — Universality of Log-Loss
- **Result**: Under logarithmic loss (cross-entropy), ANY discrete memoryless source is successively refinable
- **Connection**: Since we train with cross-entropy/contrastive losses, ANY semantic source we encounter is guaranteed to be successively refinable. Our loss function choice is provably universal.

### Charvin et al. 2023 — Successive Refinement of IB
- **Result**: Derives conditions for successive refinability within Information Bottleneck framework. Introduces "soft" measure of refinement loss.
- **Connection**: Our steerability metric IS a soft successive refinement measure — it quantifies how much information optimality is preserved when truncating the embedding.

## 2. RENORMALIZATION GROUP (Physics)

### Kline & Palmer 2021 — IB = RG Equivalence
- **Result**: FORMAL EQUIVALENCE between Information Bottleneck and non-perturbative RG. IB coarse-graining = RG coarse-graining with soft cutoffs. Semigroup structure: successive IB transforms compose consistently.
- **Connection**: Our fractal embedding structure implements RG flow in embedding space. Prefix truncation = coarse-graining. Adding dimensions = inverse RG (adding short-range detail).

### Peraza Coppola et al. 2025 — RG for DNNs
- **Result**: RG framework for analyzing self-similarity in learning curves. Identifies scaling intervals, classifies perturbations as relevant/irrelevant.
- **Connection**: Self-similarity of learning curves = self-similarity of our fractal embeddings. Inverted hierarchies hurt = irrelevant structure injected. True hierarchies help = relevant structure preserved.

### Geiger et al. 2023 — Separation of Scales in CNNs
- **Result**: Deep CNNs exhibit separation of scales — layers couple only through second-order statistics. Enables thermodynamic description.
- **Connection**: Scale separation is the PHYSICAL MECHANISM that makes nested representations natural. Each layer's representation is approximately a sufficient statistic for the next.

## 3. FIBER BUNDLES (Geometry)

### Liu 2025 — Fiber Bundle Networks
- **Result**: Reformulates classification as geometric optimization on fiber bundles. Categories = base space, features = fibers.
- **Connection**: Our semantic hierarchy IS the base space. Coarse categories at the base, fine categories as fibers. Prefix picks the base position; suffix moves in the fiber.

### Bundle Networks (OpenReview)
- **Result**: Fiber bundles model many-to-one maps via local trivialization
- **Connection**: The many-to-one map from fine instances to coarse categories is what our hierarchy encodes. Local trivialization = prefix dimensions for coarse classification are locally independent of remaining fine-grained dimensions.

## 4. RATE-DISTORTION (Embeddings)

### Theoretical Limitations of Embedding-Based Retrieval (2025)
- **Result**: Embedding dimension fundamentally limits distinct top-k subsets
- **Connection**: Rate-distortion lower bound for embeddings. Our fractal structure ALLOCATES this finite capacity across hierarchy levels — rate-distortion optimal if source is successively refinable (guaranteed by No 2019).

## 5. MULTI-SCALE REPRESENTATIONS (Deep Learning)

### Westphal et al. 2025 — Generalized IB (Synergy)
- **Result**: Synergistic functions achieve superior generalization. IB optimization simultaneously optimizes GIB.
- **Connection**: Our hierarchical structure creates SYNERGY between coarse prefix and fine suffix. Explains why steerability is strongest in deep hierarchies (CLINC 10->150): more synergy between scales.

### Mallat 2012 — Wavelet Scattering
- **Result**: Cascaded wavelet transforms produce provably sufficient statistics for classification at each invariance level
- **Connection**: Scattering transform IS a filtration of sufficient statistics. Our V5 learns the SEMANTIC analogue of what Mallat proved for GEOMETRIC hierarchies.

---

## THE GRAND UNIFICATION

| Component | Supporting Theory | Key Papers |
|-----------|------------------|------------|
| "Nested" | Successive refinement: optimal multi-rate compression IS nested | Equitz-Cover 1991, No 2019 |
| "Filtration" | RG flow produces filtrations; IB = RG under Gaussianity | Kline-Palmer 2021, Geiger 2023 |
| "Sufficient statistics" | IB discovers sufficient stats; scattering transforms are sufficient | Charvin 2023, Mallat 2012 |
| "Aligned to hierarchy" | Fiber bundles formalize category-fibered representations | Liu 2025, Bundle Networks |
| "Optimal" | Log-loss universality guarantees successive refinability | No 2019 |

---

## THE DECISIVE EXPERIMENT: Predict-Before-Train

To prove a LAW (not just an observation):

1. From raw data only, estimate hierarchy complexity profile H_l (entropy added per depth level)
2. PREDICT optimal prefix breakpoints k_l from H_l, before any model training
3. Train different architectures (bge-small, Qwen3, different dims)
4. Test whether measured steerability and task-optimal breakpoints collapse to predicted k_l
5. Verify universality across domains (text, vision, code, biology)

If prediction holds across architectures/tasks → LAW, not trick.

## THE NECESSITY THEOREM (What We Need to Prove)

**Claim**: For hierarchical classification tasks with depth d > 1, nested representations achieve strictly better sample complexity than flat representations of the same total dimension.

**Proof sketch** (to be formalized):
1. Hierarchical task decomposes into d conditional classification problems
2. At level l, only dimensions 1..k_l are relevant (coarse sufficient statistic)
3. Flat representation uses ALL d dimensions for level l → wastes capacity on irrelevant fine-grained features
4. Nested representation uses only k_l dimensions → effective sample size scales as n/k_l instead of n/D
5. By Fano's inequality, this gives strictly better minimax risk for coarse classification

This would be the THEORETICAL contribution. Combined with the predict-before-train experiment, it's a complete story.

## EMPIRICAL SCALING LAW (Validated Feb 2026)

### The Discovery

**V5 steerability is a monotonic function of H(L1|L0), predictable from data alone.**

| Dataset | H(L1|L0) | V5 Steer | MRL Steer | V5-MRL Gap |
|---------|----------|----------|-----------|------------|
| Yahoo | 1.229 | -0.004* | +0.006 | -0.010 |
| Newsgroups | 1.882 | +0.022 | +0.009 | +0.013 |
| TREC | 2.211 | +0.045 | +0.003 | +0.041 |
| CLINC | 3.903 | +0.053 | -0.010 | +0.063 |

\*Yahoo: 1 seed only; 3-seed mean ~+0.011 from prior run.

**Statistical Results:**
- Spearman rho = 1.0 (perfect rank correlation), exact permutation p = 0.042
- V5-MRL Gap: also rho = 1.0, p = 0.042
- MRL: rho = -0.8, p = 0.20 (NOT significant — hierarchy-agnostic training cannot capture this)
- Linear fit: Steer = 0.019 * H(L1|L0) - 0.016, R^2 = 0.75

### The Hierarchical Rate-Allocation Law (Codex GPT-5.3 formulation)

S_V5(H, j1) = kappa * [H - C(j1)]+

Where:
- H = H(L1|L0) = conditional entropy of hierarchy (data property)
- C(j1) = effective fine-detail capacity of short prefix (architecture property)
- kappa = steerability per bit (universal conversion rate)
- [x]+ = max(0, x) (hinge — below capacity threshold, no steerability)

**Falsifiable predictions:**
1. Increasing short-prefix width raises C(j1), shifting elbow H_c to the right
2. At matched H, different datasets should collapse to similar S (sufficiency)
3. MRL should not show comparable positive slope vs H (confirmed: rho = -0.8)

### Causal Validation Plan (In Progress)

**Synthetic Hierarchy Experiment:**
- Base: CLINC 150 fine classes (same text content across all conditions)
- Vary: K0 coarse groups = {2, 3, 5, 10, 15, 25, 50, 75}
- This DIRECTLY MANIPULATES H(L1|L0) while holding text fixed
- Expected: 8 data points tracing out the hinge law
- If confirmed: causal + observational evidence combined = overwhelming

**Capacity Sweep (tests C(j1) prediction):**
- Inverted V5 with prefix dim = {32d, 64d, 128d}
- Increasing prefix capacity should shift the threshold H_c
- If confirmed: the law has two independent testable components

### Path to Universal Constant

Codex analysis suggests the slope (0.019) is not a universal constant but a
regime-specific effective slope. To derive from first principles:
1. R_1 = I(L1; Z_j1 | L0) = mutual info of fine labels given short prefix
2. Steer ~ H(L1|L0) - R_1 (unresolved conditional entropy)
3. Convert info gain to accuracy gain via Fano's inequality
4. Upper bound on slope: ~1/log2(K1 - 1) (dataset-dependent)

After capacity normalization, the slope may become architecture-invariant.
This would be the "universal constant" — the efficiency with which
any hierarchical representation converts entropy into steerability.

---

## WHAT MAKES THIS TURING-LEVEL

1. **Necessity theorem**: hierarchical tasks REQUIRE nested representations for optimal scaling
2. **Universal empirical law**: one or few exponents governing semantic scale allocation across domains
3. **Operational control**: scale-targeted interventions that reliably alter coarse/fine semantics (we already have this via causal ablations!)
4. **Physics connection**: embedding space has RG flow structure, and our training is the first to explicitly align with it

---

## Competitive Landscape (Updated)

| Work | What They Do | What They Don't Do |
|------|-------------|-------------------|
| MRL (Kusupati 2022) | Variable-length embeddings | No hierarchy alignment, no steerability |
| SMRL (EMNLP 2025) | Fix MRL gradient variance | Still not steerable |
| CSR (ICML 2025) | Sparse coding alternative | Different paradigm (sparsity vs nesting) |
| HEAL (ACL 2025) | External label alignment | Post-hoc constraint, not structural |
| Scattering (Mallat) | Geometric multi-scale | Only geometric, not learned semantic |
| Hyperbolic embeddings | Geometric hierarchy | Single-scale in radial coordinate |
| FiberNet (Liu 2025) | Fiber bundle classification | No prefix/scale structure |
| Nested Diffusion (CVPR 2025) | Multi-scale generation | Generation, not retrieval |

**Nobody** does: hierarchy-aligned prefix supervision with proven successive refinement properties and causal mechanism verification. We are alone in this space.
