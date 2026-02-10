# Fractal Embeddings: Theory, Narrative & Writing Guide

## The Story Arc

### Opening Hook
Modern embedding models like those using Matryoshka Representation Learning (MRL) let you truncate embeddings to save compute -- but truncation only changes *fidelity* (how much information you keep), not *granularity* (how coarse or fine the meaning is). This is a missed opportunity: real-world semantics are hierarchical. "What is the capital of France?" is simultaneously a LOCATION question and a CITY question. What if truncation could steer between these levels?

### The Insight
Instead of training all prefix lengths on the same fine-grained labels (as MRL does), train short prefixes on coarse labels and full embeddings on fine labels. This is **hierarchy-aligned prefix supervision** -- a simple one-line change to MRL training that creates a fundamentally new capability.

### Why It Works (Theory)
This connects to **successive refinement** from information theory (Equitz & Cover 1991, Rimoldi 1994). Hierarchical sources are naturally successively refinable: the optimal multi-resolution code first encodes coarse information, then the refinement. V5 training approximates this optimal code. MRL performs single-resolution coding at each rate, losing the nested structure.

### The Evidence (Causal, Not Just Correlational)
Four controlled ablations prove causality:
1. **Invert** the alignment: steerability flips sign (short prefixes now specialize for FINE)
2. **Remove** prefix supervision: steerability collapses to zero
3. **UHMT** (hierarchy-aware but not aligned): near-zero steerability
4. **MRL** control: near-zero everywhere

The UHMT result is the killer argument: even if you train on both coarse and fine labels at every prefix length, you get no steerability. It's not about *knowing* the hierarchy -- it's about *aligning* prefix lengths with hierarchy levels.

### The Scaling Story
Steerability isn't one-size-fits-all. It depends on the interaction of:
- **Hierarchy depth** H(L1|L0): how much refinement information exists
- **Model capacity**: whether the pretrained model can already resolve fine distinctions

The product of these two factors predicts steerability with rho = 0.90 (p = 0.002). This explains why WOS (deepest hierarchy but lowest learnability) shows less steerability than CLINC (moderate hierarchy but high learnability).

A synthetic experiment with controlled hierarchies reveals a **Goldilocks optimum**: steerability peaks when coarse task complexity matches prefix capacity. Too few coarse classes = spare capacity leaks fine info. Too many = prefix can't distinguish them. This is exactly what successive refinement theory predicts.

---

## Theoretical Framework

### Connection to Information Theory

The key theoretical insight is that V5 training implements an approximation to the **successive refinement** coding scheme from rate-distortion theory.

**Background:** Equitz & Cover (1991) and Rimoldi (1994) showed that for certain source classes, one can achieve optimal compression at multiple rates simultaneously using a nested code: first describe the source at a coarse rate R1, then refine with rate R2.

**Our contribution:** We show that:
1. Hierarchical semantic sources (where coarse = deterministic function of fine) are naturally successively refinable
2. V5 training maps this to embedding space: block 1 encodes Y0 (coarse), blocks 2-J encode Y1|Y0 (refinement)
3. MRL's flat supervision breaks this structure -- it performs single-resolution coding at each prefix length

### Theorem 1: Hierarchy-Successive-Refinement

**Setup:** Let (X, Y0, Y1) be a hierarchical source with Y0 = g(Y1). Encoder produces z in R^d with prefix z_{<=m}. Let C(d') denote effective capacity of d'-dimensional embedding.

**Statement (informal):** Assume C(d/J) >= H(L0) and C(d/J) < H(L1).
- Under V5 supervision: I(z_{<=1}; L0) > I(z_{<=1}; L1|L0) (coarse-prioritized prefix)
- Under MRL: no specialization, I(z_{<=1}; L0) ~ I(z; L0)

**Intuition:** V5's prefix loss depends only on L0, so the optimal prefix maximizes I(z_{<=1}; L0). Since C(d/J) < H(L1) but >= H(L0), the prefix allocates capacity preferentially to coarse.

### Theorem 2: Goldilocks Capacity-Demand Matching

**Statement:** With fixed H(Y1) and varying K0:
- When H(L0) < C(d/J): spare capacity leaks L1|L0 info, reducing S
- When H(L0) > C(d/J): Fano's inequality degrades coarse classification, reducing S
- Peak at H*(L0) ~ C(d/J)
- Taylor expansion: S ~ S* - alpha*(H(L0) - H*)^2

This matches the empirical quadratic fit with R^2 = 0.964.

### Testable Prediction
Doubling prefix dimension from 64 to 128 should shift the Goldilocks peak rightward (to higher K0), as C(d/J) increases. This is verifiable via a capacity sweep ablation.

---

## Narrative Framing Options

### Option 1: "Semantic Zoom" (Visual Metaphor)
Like Google Maps zoom: zoom out to see countries (coarse), zoom in to see streets (fine). Fractal embeddings let you do this with a single embedding -- just truncate for "zoomed out", extend for "zoomed in."

### Option 2: "One Model to Rule Them All" (Practical Framing)
Instead of deploying separate coarse and fine encoders (doubling cost), fractal embeddings give you both in a single model. The 64d prefix actually beats a dedicated 256d coarse encoder.

### Option 3: "Information Theory Meets Deep Learning" (Academic Framing)
We show that a 30-year-old result in rate-distortion theory (successive refinement) has a direct, practical application in modern embedding systems. The theory predicts exactly when and why hierarchy-aligned training works.

### Option 4: "Alignment, Not Awareness" (Causal Framing)
The UHMT control is the most surprising result. Giving every prefix access to both hierarchy levels should help, right? Wrong. Only *aligning* specific prefix lengths with specific hierarchy levels creates steerability. This is a fundamental insight about how neural networks organize information.

---

## Key Numbers for Headlines

- **8/8 datasets** show V5 > MRL steerability (p = 0.004 sign test)
- **Pooled effect size d = 1.49** (p = 0.0003, meta-analysis)
- **10x retrieval ramp** (V5 6.3pp vs MRL 0.6pp)
- **3.7x faster** coarse queries (64d vs 256d HNSW)
- **Zero inference overhead** -- same model, same parameters as MRL
- **rho = 0.90** predictability of steerability from hierarchy structure

## Potential Weaknesses to Address

1. **Small absolute steerability on shallow hierarchies** -- This is predicted by theory (low H(L1|L0) = little to separate). The product predictor explains this.

2. **CLINC L1 k-NN accuracy drops** -- The 384d->256d projection trades raw k-NN accuracy for prefix structure. Classification head accuracy is >95%. This is a deliberate trade-off.

3. **No HEAL/CSR/SMRL head-to-head** -- These methods address orthogonal goals. HEAL optimizes global embeddings without prefix steerability; CSR targets sparsity; SMRL improves MRL fidelity. None offers prefix-level steering.

4. **Ablations on 2 datasets** -- Main results span 8 datasets. Ablation concentration on CLINC+TREC reflects compute constraints but both high-H and moderate-H settings are covered.

5. **Head-only training limits** -- WOS shows a floor effect at 336 fine classes. Backbone fine-tuning might help but wasn't necessary for the core claim.
