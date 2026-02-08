# Fractal Embeddings: Paper Outline (v2 — Codex-Reviewed)

**Title**: Fractal Embeddings: Hierarchy-Aligned Prefix Supervision for Steerable Semantic Granularity

**Target**: NeurIPS 2026 (deadline: late May) or EMNLP 2026 (ARR: August)

**Codex Readiness**: 6.5/10 current → 8.5/10 with cross-backbone causal replication

**Framing**: Method paper first (how to build steerable embeddings), discovery second (scaling law)

---

## Central Claim: The Fractal Embedding Principle

For prefix-truncated embeddings, hierarchy-aligned supervision determines steerability direction, and steerability magnitude increases with hierarchy refinement entropy.

Compactly:
- **Alignment**: short→L0, full→L1 ⟹ S > 0
- **Inversion**: short→L1, full→L0 ⟹ S < 0
- **Scaling**: S ≈ α H(L1|L0) + β, α > 0

---

## Section Structure (with figure/table assignments)

### 1. Introduction (1 page)
**Figure 1 (Teaser)**: V5 vs MRL on CLINC — same accuracy, huge steerability gap.

- Embedding spaces map text to vectors, but real-world semantics are *hierarchical*
- MRL (Kusupati et al. 2022) enables multi-resolution embeddings but treats all scales equally
- We show that *aligning* prefix supervision with hierarchy creates "steerable" embeddings
- State the Fractal Embedding Principle

### 2. Problem Setup and Definitions (0.5 pages)
**Table 1**: Datasets, hierarchy stats, H(L1|L0), train/test sizes, metric definitions.

| Dataset | n_L0 | n_L1 | Branch | H(L1|L0) | Status |
|---------|-------|-------|--------|----------|--------|
| Yahoo | 4 | 10 | 2.5 | 1.23 | DONE |
| GoEmotions | 4 | 28 | 7.0 | 1.88 | PENDING |
| Newsgroups | 6 | 20 | 3.3 | 1.88 | DONE |
| TREC | 6 | 50 | 8.3 | 2.21 | DONE |
| arXiv | 20 | 123 | 6.2 | 2.62 | PENDING |
| DBPedia Cls | 9 | 70 | 7.8 | 3.17 | PENDING |
| CLINC | 10 | 150 | 15.0 | 3.90 | DONE |
| WOS | 10 | 336 | 33.6 | 5.05 | PENDING |

### 3. Method: Progressive Prefix Supervision (V5) (1 page)
**Figure 2**: Architecture/training schematic (V5 vs MRL). [LaTeX/tikz]

- V5: j=1 (64d) trained with L0, j=4 (256d) trained with L1
- MRL: ALL prefix lengths trained with L1
- Block dropout, head-only training, frozen backbone
- Key insight: maps information structure of hierarchy onto dimensional structure of embedding

### 4. Main Results: Accuracy Parity + Steerability Gains (1.5 pages)
**Table 2**: Classification parity (V5 vs MRL, p-values).
**Figure 3**: Cross-dataset steerability forest plot with CIs/effect sizes.

Current results:
| Dataset | V5 Steer | MRL Steer | Gap | n seeds |
|---------|----------|-----------|-----|---------|
| Yahoo | +0.011 | +0.004 | +0.007 | 3 |
| Newsgroups | +0.022 | +0.009 | +0.013 | 3 |
| TREC | +0.045 | +0.003 | +0.041 | 3 |
| CLINC | +0.053 | -0.010 | +0.063 | 5/1 |

### 5. Causal Identification (1.5 pages)
**Figure 4**: CLINC ablations (normal/inverted/no-prefix) with sign reversal.
**Table 3**: Full ablation statistics (means, SD, p-values, effect sizes).

Results (CLINC, 5 seeds):
- V5 (aligned): S = +0.053 ± 0.003
- Inverted: S = -0.018 ± 0.004 (p < 0.000001 vs V5)
- No-prefix: S = +0.009 ± 0.005 (p < 0.000001 vs V5)

### 6. Fractal Law: Steerability vs Hierarchy Depth (1.5 pages)
**Figure 5**: S vs H(L1|L0) real datasets with fit + Spearman.
**Figure 6**: Synthetic hierarchy causal curve (same text, varied K0/entropy).

Current observational:
- Spearman ρ = 1.0 (p = 0.042), R² = 0.79

Synthetic (RUNNING):
- K0=2 (H=6.225): V5=+0.134, MRL=-0.010
- K0=3 (H=5.640): V5=+0.150, MRL=+0.008
- Remaining 6 conditions in progress

### 7. Generality and Limits (1 page)
**Table 4**: Cross-model replication (bge-small vs Qwen3-0.6B).

- Architecture invariance (TODO: run Qwen3 experiments)
- Failure cases: ceiling effects, shallow hierarchies
- LOO prediction test
- Boundary conditions: K0=2 (binary L0) shows capacity threshold

### 8. Related Work (0.5 pages)
- MRL (NeurIPS 2022): our baseline — no steerability
- SMEC/SMRL (EMNLP 2025): Sequential MRL — no steerability
- CSR (ICML 2025 oral): Sparse coding — different problem
- HEAL (ACL workshop): External label alignment
- Hyperbolic/Poincare: geometric hierarchy — different paradigm

### 9. Limitations, Ethics, and Conclusion (0.5 pages)

---

## Figures (6 total)

| # | Description | Status |
|---|-------------|--------|
| 1 | V5 vs MRL teaser (CLINC prefix curves) | DONE |
| 2 | Method diagram (V5 vs MRL training) | TODO (LaTeX) |
| 3 | Cross-dataset steerability forest plot | DONE |
| 4 | Causal ablation bar/point plot | DONE |
| 5 | Scaling law scatter S vs H(L1|L0) | DONE |
| 6 | Synthetic hierarchy causal curve | RUNNING |

## Tables (4 total)

| # | Description | Status |
|---|-------------|--------|
| 1 | Dataset stats and hierarchy profiles | DONE |
| 2 | Classification accuracy parity | DONE |
| 3 | Causal ablation full statistics | DONE |
| 4 | Cross-model replication | TODO |

---

## Critical Path to NeurIPS Submission

1. [RUNNING] Synthetic hierarchy experiment → Figure 6
2. [TODO] Cross-model replication (Qwen3-0.6B on CLINC+TREC) → Table 4
3. [TODO] 4 new dataset benchmarks (GoEmotions, arXiv, DBPedia Cls, WOS) → Strengthen Fig 5
4. [TODO] LOO prediction test with n=8 → Section 7
5. [TODO] LaTeX paper draft
6. [TODO] Codex review of final draft

**Killer experiment** (Codex recommendation): Cross-backbone synthetic hierarchy replication on Qwen3-0.6B. If it works → "general principle" not "single-model effect."
