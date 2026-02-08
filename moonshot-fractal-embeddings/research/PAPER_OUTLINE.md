# Fractal Embeddings: Paper Outline (v2 — Codex-Reviewed)

**Title**: Fractal Embeddings: Hierarchy-Aligned Prefix Supervision for Steerable Semantic Granularity

**Target**: NeurIPS 2026 (deadline: late May) or EMNLP 2026 (ARR: August)

**Codex Readiness**: 6.5/10 current → 8.5/10 with cross-backbone causal replication

**Framing**: Method paper first (how to build steerable embeddings), discovery second (scaling law)

---

## Abstract (Codex-drafted, 240 words)

Modern embedding models support dimensional truncation, but truncation typically changes fidelity rather than semantic level, leaving no mechanism to *steer* between coarse and fine meaning at inference time. We introduce **Fractal Embeddings**, a hierarchy-aligned prefix supervision scheme that trains short prefixes (64d) on coarse labels (L0) and full embeddings (256d) on fine labels (L1), using a frozen backbone with learned heads only. Against a matched Matryoshka Representation Learning (MRL; NeurIPS 2022) baseline trained on L1 at all prefix lengths, our V5 method preserves full-resolution performance (no loss at j=4) while inducing robust steerability: truncated prefixes encode coarse semantics, whereas full vectors recover fine semantics. Causal ablations identify alignment as the driver of this effect: aligned supervision yields S=+0.053, inverted alignment flips direction (S=-0.018), and removing prefix-specific supervision collapses the signal (S=+0.009); p<10^-6 across 5 seeds, 3 conditions, and CLINC. In a synthetic hierarchy with fixed text and varied coarse partitions, steerability scales with H(L0) and exhibits a Goldilocks optimum at ~12-16 coarse classes (quadratic R²=0.964). Across four real datasets, steerability is perfectly rank-ordered by H(L1|L0) (Spearman rho=1.0, p=0.042); MRL remains near zero in all synthetic, real, and ablation settings. These results establish hierarchy alignment as a principled control knob for semantic granularity, enabling a single embedding to support coarse-to-fine behavior without sacrificing fine-grained accuracy.

---

## Central Claim: The Fractal Embedding Principle (v2 — updated by synthetic experiment)

For prefix-truncated embeddings:
1. **Alignment determines direction**: short→L0, full→L1 ⟹ S > 0; inverted ⟹ S < 0
2. **Magnitude scales with prefix task demand**: S ~ H_control = H(L0), with capacity saturation
3. **MRL is non-steerable**: S_MRL ≈ 0 regardless of hierarchy structure

Compactly:
- **Direction**: alignment → S > 0, inversion → S < 0 (causal ablation)
- **Mechanism**: prefix is a routing bottleneck; S ~ H(L0) when total entropy is fixed (synthetic)
- **Observational proxy**: S ~ H(L1|L0) in natural datasets (because H(L0) and H(L1|L0) covary)
- **Saturation**: S plateaus when H(L0) exceeds prefix capacity C(j1)

**Key insight from synthetic experiment**: The observational S ~ H(L1|L0) correlation was confounded by H(L0) co-movement. The synthetic experiment (constant total entropy, varied K0) reveals S ~ H(L0), not H(L1|L0). The prefix task demand is the true driver.

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

### 6. Steerability Scaling: Mechanism and Prediction (1.5 pages)
**Figure 5**: S vs H(L1|L0) real datasets with fit + Spearman (observational).
**Figure 6**: Synthetic hierarchy causal curve (S vs K0 / H(L0), same text).
**Figure 7** (NEW): Entropy allocation plot — S vs H(L0)/H_total with real + synthetic.

Observational (real datasets):
- Spearman ρ = 1.0 (p = 0.042), R² = 0.79 for S vs H(L1|L0)
- But H(L0) and H(L1|L0) covary positively in natural datasets

Synthetic (ALL 8 CONDITIONS COMPLETE):
- Inverted-U shape: S rises K0=2→15, peaks at K0=10-15, then declines K0=25→75
- Peak: K0=15, V5=+0.278 (H_L0=3.91 bits)
- S vs H(L0): rho=+0.55 (non-monotonic due to inverted-U, but rising half perfect)
- S vs H(L1|L0): rho=-0.55 (NEGATIVE — breaks observational confound)
- MRL consistently near zero across all 8 conditions

**"Goldilocks Hierarchy" framing (Codex):**
- Rising side: more coarse classes → richer prefix codebook → more steerability
- Falling side: coarse routing exceeds effective prefix capacity → errors hurt
- Peak at H*(L0) ≈ 3.6-4.0 bits ≈ K0=12-16 effective coarse states (matches 64d prefix)
- Fit: hinge model (primary, mechanistic) + quadratic (robustness check)
- Natural datasets operate near the optimum (CLINC H(L0)=3.32)

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
