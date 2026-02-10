# Future Research Directions

**Last Updated**: February 10, 2026

---

## Completed Work

### Empirical Validation
- V5 architecture validated on Yahoo Answers (5 seeds, +5.36% L0, +6.47% L1)
- Hierarchy randomization control proves causality (95% CI excludes zero)
- 20 Newsgroups real-world benchmark (p=0.0232)
- Scaling law validated (depths 2-5, 4 baselines)
- Boundary conditions identified (depths 5-7, requires learnable per-level signal)

### Theoretical Foundation
- **THEORY.md**: Canonical theory doc (minimax/sample-complexity/access-complexity/successive-refinement summary)

---

## Priority 1: Additional Real-World Benchmarks

Candidates:
- **CIFAR-100**: 20 superclasses → 100 fine classes, natural 2-level hierarchy
- **iNaturalist**: Deep biological taxonomy (kingdom → species), 5000+ classes
- **Product taxonomies**: Amazon/e-commerce hierarchies, 3-6 levels
- **arXiv papers**: Subject → sub-subject, scientific hierarchy

### Why This Matters
Current benchmarks use 2-level hierarchies with <100 classes. Deep taxonomies (3-6+ levels, 1000+ classes) are where the theoretical advantage is largest (O(log b) vs O(log C) gap grows with depth).

---

## Priority 2: Publication

### Workshop Paper (Short-term)
- Target: NeurIPS/ICML workshop on structured prediction or hierarchical learning
- Content: V5 results + hierarchy randomization + scaling law + key theorems

### Full Paper (Medium-term)
- Target: COLT/ALT for theory, or ICML/NeurIPS main for empirical+theory
- Content: Complete theoretical framework (generalization + sample complexity theory) + comprehensive empirical validation

---

## Priority 3: Extensions

### Hierarchical Reasoning via Fractal Latent Space
**Status**: Consolidated in `research/IDEAS.md`

Core idea: If LLM hidden states had fractal/hierarchical structure, the model might exhibit better hierarchical reasoning (planning, analogy, multi-step).

### Hyperbolic Fractal Embeddings
**Status**: Consolidated in `research/IDEAS.md`

Core idea: Combine fractal prefix structure with hyperbolic geometry for natural tree-metric embeddings.

### Trajectory Embeddings
**Status**: Consolidated in `research/IDEAS.md`

Core idea: Meaning as dynamics rather than points — embeddings that capture how concepts evolve across scales.

---

## Not Pursuing (Lessons Learned)

- **E2E fine-tuning on small datasets**: Overfitting collapse, NaN gradients (V1-V4 lesson)
- **Vector quantization variants**: Worse than baseline (fractal_vq experiment)
- **Shallow hierarchies (<3 levels) with <100 classes**: Insufficient room for hierarchical advantage
