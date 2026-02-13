# Claim-to-Evidence Checklist

## Status: Updated Feb 12, 2026

Every claim in the abstract and main text mapped to supporting evidence.

---

## Abstract Claims

| # | Claim | Evidence | Status |
|---|-------|----------|--------|
| 1 | MRL has no mechanism to steer between coarse/fine | By construction: MRL trains all prefixes on same L1 loss | PROVED |
| 2 | Short prefixes learn coarse, full learn fine | Table 1 (teaser), Table 3 (retrieval ramp) | VERIFIED |
| 3 | Four ablations: alignment causally drives steerability | Table 4 (ablation): inverted reverses, no-prefix collapses, UHMT collapses | VERIFIED (p<0.001, d>=6.1 CLINC) |
| 4 | Meta-analysis pooled d=1.49, p=0.0003 | meta_analysis.py output, Table 2 | VERIFIED |
| 5 | 8/8 sign test p=0.004 | compute_paper_stats.py, binomial test | VERIFIED |
| 6 | CLINC S=+0.150+/-0.028 vs MRL +0.007+/-0.016 | benchmark_bge-small_clinc.json, Table 2 | VERIFIED |
| 7 | Goldilocks quadratic R^2=0.964 | Fig 6 (synthetic), scaling_robustness.py | VERIFIED |
| 8 | Product predictor rho=0.90, p=0.002 | scaling_robustness.py, Fig 5 | VERIFIED |
| 9 | Backbone control: 4.5x params -> zero steerability | backbone_finetune_control.json, Table 7, Fig backbone | VERIFIED |
| 10 | d=11.6, 7.4, 4.4 on CLINC, DBPedia, TREC | backbone_finetune_control.json | VERIFIED |
| 11 | Three encoder families (architecture invariance) | Table 5 (cross-model): BGE, E5, Qwen3 | VERIFIED |
| 12 | Successive refinement theory, converse bound | Section 9 (theory), 5 theorems | PROVED |

## Main Text Claims

| # | Claim | Evidence | Status |
|---|-------|----------|--------|
| 13 | MRL retrieval ramp +0.6pp vs V5 +6.3pp (10x) | retrieval_benchmark_clinc.json, Table 3 | VERIFIED |
| 14 | Workload-adaptive Pareto: dominates at >=35% coarse | Fig 11 (pareto), appendix | VERIFIED |
| 15 | Single model replaces dual-encoder | Three-seed comparison, Section 5.5 | VERIFIED |
| 16 | Three-level hierarchy: d=10.9 | three_level results, Table 8 | VERIFIED |
| 17 | Million-scale retrieval: prefix truncation works at 4.6M | BEIR results (Table BEIR), Fig BEIR | VERIFIED |
| 18 | SVD rotation +38% vs truncation at 64d | beir_contrastive_projection.json | VERIFIED |
| 19 | Hierarchy-aligned projection: 94-97% of SVD quality | beir_contrastive_projection.json | VERIFIED |
| 20 | Classification projection collapses (-88%) | beir_v5_mrl_retrieval.json | VERIFIED |
| 21 | Capacity sweep: rho=1.000, r=0.989, p=0.011 | capacity_sweep results | VERIFIED |
| 22 | Hierarchy noise: 85% retained at 10% corruption | noise_robustness results | VERIFIED |

## Theory Claims

| # | Claim | Evidence | Status |
|---|-------|----------|--------|
| 23 | Theorem 1: Hierarchy -> prefix allocates MI to coarse | Formal proof in Section 9 | PROVED |
| 24 | Theorem 2: Goldilocks from capacity-demand matching | Formal proof + synthetic validation R^2=0.964 | PROVED + VERIFIED |
| 25 | Converse bound: flat supervision cannot achieve high S | Formal proof (Theorem 4) | PROVED |
| 26 | Product scaling law derivable from theory | Corollary 2, empirical rho=0.90 | PROVED + VERIFIED |
| 27 | Log-loss universality connection (No 2019) | Citation + formal connection | CITED + PROVED |

## Deep Hierarchy (IN PROGRESS)

| # | Claim | Evidence | Status |
|---|-------|----------|--------|
| 28 | V5 steerability on HUPD patents (natural hierarchy) | 4/5 V5 seeds: +0.052, +0.044, +0.030, +0.070 | IN PROGRESS |
| 29 | MRL near-zero on HUPD | 4/5 MRL seeds: +0.020, -0.026, -0.004, -0.008 | IN PROGRESS |
| 30 | Deep hierarchy generalisation (HWV Wikipedia) | Not yet started | PENDING |

## Gaps / Future Work

- [ ] Deep hierarchy experiments complete (ETA: ~2 hours)
- [ ] Vision replication (CIFAR-100 with image hierarchies) - stretch goal
- [ ] Cross-domain transfer (train on one hierarchy, test on another) - future work
- [ ] Larger models (>1B params) - resource constrained

---

## Pre-submission Checks

- [x] All p-values verified against compute_paper_stats.py output
- [x] All effect sizes verified against meta_analysis.py output
- [x] Cross-references checked (all \ref point to existing \label)
- [x] SMRL/SMEC naming fixed to SMEC throughout
- [x] Missing citations added (20 Newsgroups, DBPedia)
- [x] Grammar/spelling consistency (British throughout)
- [ ] PDF compilation clean (no warnings)
- [ ] Deep hierarchy results added to paper (when complete)
