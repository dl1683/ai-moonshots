# Launch Materials: Truncation Is Not Compression

## Paper Title
**Truncation Is Not Compression: Hierarchy-Aligned Embeddings via Successive Refinement**

## Concept Brand
**Semantic Zoom**

## One-Liner
> "Truncation without alignment is amputation, not multi-resolution."

---

## Tweet Thread (7 tweets)

### 1/7 (Hook)
Most "compressed embeddings" are not compressed. They are amputated.
If your 256-d vector is just a chopped 1536-d vector, you lose structure, retrieval quality, and control.
New paper: *Truncation Is Not Compression*.

### 2/7 (The Problem)
[IMAGE: 2x2 grid]
Cols = Full vector vs Truncated vector.
Rows = Standard MRL vs Hierarchy-aligned.
Top-right (MRL truncated): same category, worse quality.
Bottom-right (Ours truncated): coarser category, appropriate granularity = semantic zoom.

### 3/7 (The Insight)
Key insight: embeddings are multi-resolution only when each prefix is explicitly trained to carry the right level of meaning. Otherwise truncation is just lossy cropping.

### 4/7 (The Theory)
We proved it mathematically: steerability = H(L1|L0) - prefix_leakage - residual.
Any method achieving semantic zoom MUST keep fine details out of short prefixes.
Flat MRL violates this. No amount of parameters can fix it.

### 5/7 (The Evidence)
[IMAGE: forest plot]
9 datasets, 3 model families, meta-analytic d=1.49 (p<0.0001), 9/9 sign test.
4 causal ablations: invert alignment -> reverses. Remove it -> collapses to zero.
Zero inference cost. Same model. Just change the training loss.

### 6/7 (The Prediction)
[IMAGE: scaling law figure]
We derived a scaling law predicting WHEN this works best.
Pre-registered our predictions. Ran the experiments.
4.3% prediction error. Science, not curve-fitting.

### 7/7 (CTA)
Paper: [arxiv link]
Code: https://github.com/dl1683/ai-moonshots
Demo: [HF Spaces link]

Built with 1 GPU by 1 researcher. Sometimes ideas beat scale.
"Truncation without alignment is amputation, not multi-resolution."

---

## Blog Post Outline

**Title:** "Why Truncating Embeddings Is Semantically Incoherent (And How to Fix It)"

### Structure:
1. **Hook** (200 words): "You're probably truncating embeddings right now. Here's why it doesn't do what you think."
2. **The Problem** (300 words): MRL gives you smaller embeddings, but they encode the same thing at lower quality. Real semantics are hierarchical. "quantum entanglement" IS "physics" IS "science."
3. **The One-Paragraph Insight** (100 words): Train short = coarse, long = fine. Done. Zero cost.
4. **The Visual Proof** (2x2 grid + zoom demo screenshots)
5. **Why This Has to Be True** (200 words): Successive refinement theorem. No math -- just the intuition. "The prefix can't serve two masters."
6. **The Results** (300 words): Forest plot, scaling law, causal ablations. Focus on hardest-to-argue numbers.
7. **What This Means for You** (200 words): If you use OpenAI embeddings, Pinecone, Weaviate -- here's what changes. Truncation features in these products are leaving performance on the table.
8. **Code + Demo links**

---

## Target Amplifiers

### Tier 1 (Direct stakeholders)
- Nils Reimers (@nils_reimers) - sentence-transformers creator
- Tom Aarsen (@tomaborsen) - sentence-transformers maintainer at HF
- Aditya Kusupati - MRL first author (UW/Google)

### Tier 2 (Large audience, topic-adjacent)
- Lilian Weng (@lilianweng) - OpenAI, info bottleneck blog
- Sasha Rush (@srush_nlp) - embedding invertibility paper
- Sebastian Raschka (@rasaborsen) - ML newsletter

### Tier 3 (Community amplifiers)
- r/MachineLearning
- HackerNews
- DAIR.AI ML Papers of the Week
- Pinecone / Weaviate / Qdrant communities

---

## Key Visual Assets Needed
1. **2x2 grid** (MRL vs V5, truncated vs full) - THE viral image
2. **Forest plot** (meta-analysis, all 9 datasets pointing same direction)
3. **Semantic zoom animation/GIF** (query results at 64d -> 128d -> 192d -> 256d)
4. **Scaling law figure** (product predictor with pre-registered predictions)
5. **Hugging Face Space demo** (interactive slider)
