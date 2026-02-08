# Competitive Landscape (Feb 2026)

## Closest Competitors

### HEAL (ICLR 2025)
- Uses hierarchical contrastive losses with external label hierarchies
- Requires matrix factorization preprocessing
- Does NOT offer steerability via dimensional truncation
- Our advantage: intrinsic hierarchy from prefix structure, no external preprocessing

### CSR/CSRv2 (ICML 2025, ICLR 2026)
- Adaptive dimensionality via sparsity (not prefix truncation)
- Purely compression/efficiency — no hierarchical structure
- Orthogonal and potentially complementary

### SMRL (EMNLP 2025)
- Improves MRL training stability (sequential, reduced gradient variance)
- No hierarchy awareness — still fidelity-only truncation
- Our V5 progressive prefix supervision is fundamentally different

### M3 Multimodal (ICLR 2025)
- Matryoshka nesting applied to visual tokens in VLMs
- Coarse-to-fine granularity but in discrete token space
- Validates our intuition across modalities

## Key Papers to Address

### Random Dimension Removal (Aug 2025, arXiv:2508.17744)
- Shows 50% random removal barely hurts flat performance
- MUST preempt: random removal preserves flat but destroys steerability
- This is actually our strongest argument for WHY structured allocation matters

### Theoretical Limits of Embedding Retrieval (DeepMind, Aug 2025, arXiv:2508.21038)
- Proves expressiveness bounded by dimensionality
- Motivates multi-resolution access to same embedding
- Strengthens our case for fractal embeddings

## Our Unique Position
| Feature | CSR | SMRL | HEAL | M3 | Fractal V5 |
|---------|-----|------|------|-----|------------|
| Adaptive dims | Sparse | Prefix | No | Tokens | Prefix |
| Hierarchical | No | No | External | Implicit | Intrinsic |
| Steerable | No | No | No | Partial | Yes |
| Causal proof | No | No | No | No | Yes |
| Scaling law | No | No | No | No | Yes |
| Theory | No | No | No | No | Successive refinement |

## No one combines prefix steerability + intrinsic hierarchy + causal proof + scaling law.
