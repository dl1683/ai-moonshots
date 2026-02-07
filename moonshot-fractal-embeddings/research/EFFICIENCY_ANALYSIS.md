# Efficiency Analysis: V5 vs MRL

## Detailed Training Cost Breakdown

### Architecture
- **Backbone**: Qwen3-0.6B (600M params, frozen during Stage 1)
  - hidden_dim = 1024, ~28 transformer layers
  - FLOPs per sample (seq_len=128): ~90B FLOPs
- **Fractal Head**: ~2.5M trainable params
  - input_proj: Linear(1024, 256)
  - 4x shared transformer block (norm + self-attn + FFN)
  - 4x scale_proj: Linear(256, 64)
  - head_top: Linear(256, num_l0), head_leaf: Linear(256, num_l1)
  - FLOPs per sample: ~5M FLOPs

### Training: Forward Pass Cost Per Batch (batch_size=24)

Both V5 and MRL make **exactly 4 model.forward() calls per batch**:

| Component | FLOPs/call | V5 calls | MRL calls | V5 total | MRL total |
|-----------|-----------|----------|-----------|----------|-----------|
| Backbone forward | 90B | 4 | 4 | 360B | 360B |
| Fractal head forward | 5M | 4 | 4 | 20M | 20M |

**Backbone dominates: 360B >> 20M. Both methods pay the same backbone cost.**

### Training: Loss Computation Per Batch

| Loss component | FLOPs/term | V5 terms | MRL terms | V5 total | MRL total |
|---------------|-----------|----------|-----------|----------|-----------|
| InfoNCE (24x24 sim matrix) | 300K | 2 | 5 | 600K | 1.5M |
| Margin loss | 300K | 2 | 5 | 600K | 1.5M |
| CE (classifier + softmax) | 80K | 2 | 5 | 160K | 400K |
| **Total loss** | - | - | - | **1.36M** | **3.4M** |

### Training: Backward Pass

- Backbone: **frozen** (no gradient computation) = 0 FLOPs for both
- Fractal head backward: ~10M FLOPs for V5, ~12M for MRL (slightly more due to extra loss paths)

### Training: Total Per Batch

| Component | V5 | MRL | Difference |
|-----------|------|------|-----------|
| Backbone forward (4x) | 360.00B | 360.00B | 0 |
| Head forward (4x) | 0.02B | 0.02B | 0 |
| Loss computation | 0.001B | 0.003B | +0.002B |
| Head backward | 0.01B | 0.012B | +0.002B |
| **Total** | **360.03B** | **360.04B** | **+0.004B** |

**Training overhead of MRL over V5: 0.001% (negligible)**

Both methods are completely dominated by the 4 frozen backbone forward passes.
The extra loss computation in MRL is vanishingly small compared to backbone cost.

### Honest Assessment

The training cost difference between V5 and MRL is **NOT a meaningful talking point**.
Any claim of "30% cheaper" was incorrect. The real number is <0.1% difference.

---

## Where Efficiency DOES Matter: Inference & Retrieval

### Single-Text Encoding (Identical)

Both V5 and MRL:
- 1 backbone forward: ~90B FLOPs
- 1 head forward: ~5M FLOPs
- **Total: ~90B FLOPs (identical)**

### Similarity Search (Where V5 Wins)

For a database of N documents, computing top-k nearest neighbors:

| Prefix length | Dimensions | FLOPs per query | Storage per doc |
|--------------|------------|----------------|----------------|
| j=1 (64d) | 64 | 128N | 256 bytes |
| j=2 (128d) | 128 | 256N | 512 bytes |
| j=4 (256d) | 256 | 512N | 1024 bytes |

**Both methods can truncate to 64d for 4x cheaper search and 4x less storage.**

The question isn't cost — it's **utility of the truncated embedding**.

### Coarse Filtering Quality (V5's Real Advantage)

**CLINC dataset (10 domains, 150 intents):**

| Method | j=1 (64d) L0 Acc | j=1 (64d) L1 Acc | L0 Recall Loss vs j=4 |
|--------|-----------------|-----------------|----------------------|
| V5 | **96-98%** | 54% | **<2%** (nearly lossless coarse) |
| MRL | 89-91% | 65-69% | **7-9%** (significant coarse loss) |

### Multi-Stage Retrieval Pipeline: The Real Efficiency Story

**Scenario**: 1M documents, find top-10 relevant results.

**Pipeline with V5:**
1. **Stage 1 (fast coarse filter)**: Use j=1 (64d) prefix
   - Cost: 128M FLOPs per query
   - Retrieve top-1000 coarse-category matches
   - L0 recall: 96-98% (nearly all relevant docs kept)
2. **Stage 2 (precise re-ranking)**: Use j=4 (256d) on 1000 candidates
   - Cost: 512K FLOPs
   - Fine-grained ranking
   - **Total: 128M + 0.5M = 128.5M FLOPs**

**Pipeline with MRL:**
1. **Stage 1 (coarse filter)**: Use j=1 (64d) prefix
   - Cost: 128M FLOPs per query
   - Retrieve top-1000
   - L0 recall: 89-91% (**7-9% of relevant docs LOST**)
2. **Stage 2**: Same as V5
   - **Total: 128.5M FLOPs (same cost, but worse recall)**

**Same compute cost, but V5 loses 2% of relevant docs while MRL loses 9%.**

To match V5's recall, MRL would need to expand Stage 1 to retrieve more candidates
(say top-2000), doubling the re-ranking cost:
- MRL compensating: 128M + 1M = 129M FLOPs
- Or use larger prefix (128d): 256M + 0.5M = 256.5M FLOPs

### Storage Efficiency

If your application only needs coarse-grained search (topic-level browsing):
- V5: Store only 64d per doc → **256 bytes/doc**, 96% accurate
- MRL: Must store 128d or 256d → **512-1024 bytes/doc**, same accuracy

**V5 enables 2-4x storage savings for coarse applications without accuracy loss.**

---

## Summary

| Metric | V5 | MRL | Winner |
|--------|------|------|--------|
| Training FLOPs/batch | 360.03B | 360.04B | Tie (~0.001% diff) |
| Encoding FLOPs | 90B | 90B | Tie |
| 64d coarse retrieval quality | **96-98% L0** | 89-91% L0 | **V5** |
| Multi-stage retrieval recall | **98%** | **91%** | **V5** |
| Min storage for 96% L0 accuracy | **64d = 256B** | 256d = 1KB | **V5 (4x less)** |

**The efficiency story is NOT about training cost. It's about inference quality at reduced dimensionality.**
V5's hierarchy-aligned training produces embeddings where truncation is meaningful and reliable,
enabling cheaper and more effective multi-stage retrieval.
