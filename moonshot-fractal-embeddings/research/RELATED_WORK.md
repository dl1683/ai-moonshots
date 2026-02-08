# Related Work Analysis (Feb 2026)

## Competitive Positioning: Nobody is doing what we're doing

Three separate threads exist but are NOT unified:
1. MRL extensions (compression/efficiency)
2. Hierarchy-aware embeddings (geometric)
3. Controllable/steerable embeddings (interpretability)

Our V5 = first to unify: hierarchy-aligned prefix supervision with truncation-based steerability.

## Direct MRL Extensions (no hierarchy awareness)

| Paper | Venue | Key Idea | Steerability? |
|-------|-------|----------|---------------|
| SMRL | EMNLP 2025 | Sequential training, reduce gradient variance | No |
| CSR | ICML 2025 (Oral) | Sparse coding instead of prefix truncation | No |
| CSRv2 | ICLR 2026 | Fix dead neurons in ultra-sparse regime | No |
| Starbucks (2D-MRL) | SIGIR 2025 | Layer + dimension axes | No |
| Matryoshka-Adaptor | EMNLP 2024 | Post-hoc MRL for APIs | No |
| TMRL | ArXiv Jan 2026 | Temporal subspace within MRL | Partial (time only) |

**TMRL is notable**: validates MRL subspaces CAN carry specific semantic meaning. Same thesis as ours.

## Hierarchy-Aware Embeddings (different mechanism)

| Paper | Venue | Key Idea | Prefix Steerability? |
|-------|-------|----------|---------------------|
| HEAL | ICLR 2025 Workshop | Depth-wise contrastive loss with hierarchy | No (full embedding) |
| HiTs | NeurIPS 2024 | Hyperbolic Poincare embeddings for hierarchy | No |
| AMA | NeurIPS 2025 | Multi-scale affinity alignment | No |

**HEAL is closest competitor** but: workshop paper, no prefix steerability, requires labels at test time.

## Key Supporting Papers

### Takeshita et al. (EMNLP 2025 People's Choice)
"Randomly removing 50% of embedding dimensions has minimal impact"
- Standard embeddings waste dimensions due to uniform distribution
- Our V5 = constructive solution: organize dimensions hierarchically
- **Cite prominently in introduction**

### Superposition Scaling Laws (NeurIPS 2025 Oral, Best Paper Runner-up)
Liu, Liu, Gore (MIT): "Superposition yields robust neural scaling"
- Models pack more features than dimensions via superposition
- Our hierarchy-aligned prefix supervision = *organizing* this superposition
- Early dims for coarse features, later dims for fine features = reduced interference
- **Cite for theoretical motivation**

### Scaling Laws for Embedding Dimension (ArXiv Feb 2026)
Killingback et al.: Power law for dimension vs retrieval performance
- OOD performance can DEGRADE with larger dimensions
- Supports our argument: *what information goes where* matters more than raw dimension
- **Cite in scaling law section**

## Papers Validating Our Direction

- **TMRL**: MRL subspaces CAN carry semantic meaning (temporal)
- **MetaEmbed (Meta, Sep 2025)**: Multi-vector granularity training validates the direction
- **iBERT**: Sense-level steerability via sparse decomposition (different mechanism, same goal)

## Positioning Statement

"The field has independently discovered that (a) MRL prefixes contain wasteful redundancy [Takeshita 2025], (b) embedding dimensions compete via superposition interference [Liu 2025], and (c) MRL subspaces can be semantically loaded [TMRL 2026]. Our hierarchy-aligned prefix supervision unifies these insights: by aligning each prefix level with a hierarchy level, we simultaneously reduce redundancy, organize superposition, and create semantic steerability — all without sacrificing accuracy."
