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
