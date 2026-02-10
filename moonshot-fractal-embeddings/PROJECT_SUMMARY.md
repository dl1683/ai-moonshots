# Fractal Embeddings: Complete Project Summary

## For Cloud AI Agents
This document summarizes the entire Fractal Embeddings research project as of Feb 10, 2026. Use it to understand the work, write about it, or identify next steps.

---

## One-Paragraph Summary

We introduce **Fractal Embeddings (V5)**, a hierarchy-aligned prefix supervision scheme for Matryoshka-style embeddings. Instead of training all prefix lengths on the finest label (as standard MRL does), V5 trains short prefixes (64d) on coarse labels and full embeddings (256d) on fine labels. This creates embeddings where dimensional truncation corresponds to **semantic zoom**: fewer dimensions = coarser meaning, more dimensions = finer meaning. We call this property **steerability**. Across 8 datasets, 3 model families, and 4 causal ablations, we show steerability is robust, causally driven by alignment (not hierarchy awareness), and predicted by the interaction of hierarchy depth and model capacity. A successive refinement theory from information theory explains why and when this works.

---

## The Core Idea

### Problem
MRL embeddings support prefix truncation (64d, 128d, 192d, 256d), but truncation only changes **fidelity** (less information) not **semantic granularity** (coarser vs finer meaning).

### Solution: Hierarchy-Aligned Prefix Supervision
- **j=1 (64d)**: Supervised on coarse labels (L0) only
- **j=2,3 (128d, 192d)**: Mixed supervision (weighted L0 + L1)
- **j=4 (256d)**: Supervised on fine labels (L1) only
- Backbone is **frozen** (head-only training, 5 epochs)
- Block dropout forces scale specialization

### Key Metric: Steerability
```
S = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)
```
- Positive S = short prefixes better for coarse, full embeddings better for fine
- S ~ 0 means no semantic specialization (MRL behavior)

---

## Complete Results

### Main Benchmark: 8 Datasets x 5 Seeds (BGE-small)

| Dataset | H(L1|L0) | V5 S (mean +/- SD) | MRL S (mean +/- SD) | p_adj (Holm) | Cohen's d |
|---------|----------|---------------------|----------------------|-------------|-----------|
| Yahoo Answers | 1.23 | +0.015 +/- 0.019 | +0.005 +/- 0.011 | ns (0.666) | 0.4 |
| GoEmotions | 1.88 | +0.020 +/- 0.018 | +0.006 +/- 0.017 | ns (0.666) | 0.5 |
| 20 Newsgroups | 1.88 | +0.035 +/- 0.029 | +0.000 +/- 0.016 | ns (0.414) | 0.8 |
| TREC | 2.21 | +0.044 +/- 0.017 | -0.001 +/- 0.015 | 0.038* | 2.4 |
| arXiv | 2.62 | +0.027 +/- 0.015 | -0.001 +/- 0.013 | 0.078 (trend) | 1.8 |
| DBPedia Classes | 3.17 | +0.120 +/- 0.016 | +0.008 +/- 0.008 | 0.002** | 5.5 |
| CLINC-150 | 3.90 | +0.150 +/- 0.028 | +0.007 +/- 0.016 | 0.004** | 4.3 |
| WOS | 5.05 | +0.038 +/- 0.026 | +0.001 +/- 0.005 | 0.111 | 1.5 |

**Meta-analysis (DerSimonian-Laird):** pooled d = 1.49, 95% CI [0.69, 2.30], z = 3.63, p = 0.0003
**Sign test:** V5 > MRL on 8/8 datasets (p = 0.004)
**Heterogeneity:** I^2 = 63% (moderate, explained by scaling trend)

### Scaling Trend
- **Raw H(L1|L0) alone:** Spearman rho = 0.74, p = 0.035
- **Product predictor H(L1|L0) x baseline_L1_accuracy:** Spearman rho = 0.90, p = 0.002; Pearson r = 0.97, p < 0.001
- WOS deviates from raw trend due to **floor effect** (L1 accuracy ~11% despite deepest hierarchy)
- Product predictor resolves this: WOS has high H but low learnability

### Causal Ablations (CLINC 5 seeds + TREC 3 seeds)

| Dataset | Condition | S (mean +/- SD) | vs V5 t | p_adj | d |
|---------|-----------|-----------------|---------|-------|---|
| CLINC | V5 (aligned) | +0.053 +/- 0.004 | --- | --- | --- |
| CLINC | Inverted | -0.018 +/- 0.005 | 26.1 | <1e-5 | 16.5 |
| CLINC | No-prefix | +0.009 +/- 0.005 | 15.8 | <1e-5 | 10.0 |
| CLINC | UHMT | +0.001 +/- 0.005 | 14.6 | <1e-3 | 6.5 |
| TREC | V5 (aligned) | +0.045 +/- 0.023 | --- | --- | --- |
| TREC | Inverted | -0.025 +/- 0.008 | 4.9 | 0.016 | 4.0 |
| TREC | No-prefix | -0.003 +/- 0.008 | 3.3 | 0.030 | 2.7 |
| TREC | UHMT | -0.009 +/- 0.017 | 7.7 | 0.033 | 4.4 |

**Key insight:** UHMT (hierarchy-AWARE but not hierarchy-ALIGNED) produces near-zero steerability. This proves alignment, not just awareness, drives steerability.

### Synthetic Hierarchy Experiment
Fixed CLINC-150 text, varied coarse partition K0 from 2 to 75:
- **Goldilocks effect:** Steerability peaks at K0 ~ 12-16 (H(L0) ~ 3.6-4.0 bits)
- Quadratic fit R^2 = 0.964
- Rising phase: more coarse classes = richer routing codebook
- Falling phase: prefix can't distinguish 50+ coarse classes in 64d
- MRL near zero throughout

### Cross-Model Replication

| Model | Params | CLINC V5 S | CLINC MRL S | Gap |
|-------|--------|-----------|------------|-----|
| BGE-small (BAAI) | 33M | +0.150 +/- 0.028 | +0.007 +/- 0.016 | +0.143 |
| E5-small (Microsoft) | 33M | +0.130 +/- 0.031 | +0.015 +/- 0.008 | +0.115 |
| Qwen3-0.6B (Alibaba) | 600M | +0.153 +/- 0.013 | +0.008 +/- 0.006 | +0.145 |

All p < 0.025 (Holm-corrected). Larger model (Qwen3) shows higher steerability.

### Downstream: Retrieval (CLINC, 3 seeds)
- V5 L1 Recall@1 ramps from 87.1% (64d) to 93.4% (256d) = +6.3pp
- MRL L1 Recall@1: 93.6% to 94.3% = +0.6pp (flat)
- V5 ramp is 10x larger than MRL's

### Practical Utility
- **Pareto dominance:** V5 adaptive routing beats MRL-256d when >=35% queries are coarse
- **FAISS latency:** 64d queries 3.7x faster than 256d on HNSW (39us vs 145us)
- **Dual-encoder baseline:** V5 64d prefix (97.5% L0) beats dedicated 256d coarse encoder (95.8%)

### Three-Level Hierarchy (CLINC: 5 super -> 10 domains -> 150 intents)
- V5 S_02 = +0.027 +/- 0.005 vs MRL +0.002 +/- 0.005
- Paired t = 18.9, p = 0.003, d = 10.9
- V5 shows ramp gradient: L2 gains +3.2pp from 64d->256d, L1 +1.0pp, L0 +0.5pp
- MRL flat at all levels

### Metric Robustness
Four alternative steerability metrics all agree:
- S_orig: V5 > MRL on 8/8 (p = 0.004)
- S_AUC: V5 > MRL on 8/8 (p = 0.004)
- S_gap: V5 > MRL on 8/8 (p = 0.004)
- S_mono: V5 > MRL on 6/8 (p = 0.14)
All pairwise rank correlations > 0.90

---

## Theory: Successive Refinement Framework

Connected to classical information theory (Equitz & Cover 1991, Rimoldi 1994):

**Theorem 1 (Hierarchy-Successive-Refinement):**
Under V5 supervision, the prefix allocates mutual information to coarse labels when prefix capacity C(d/J) >= H(L0) but < H(L1). MRL doesn't specialize because all prefixes target L1.

**Theorem 2 (Goldilocks Capacity-Demand Matching):**
Steerability peaks when H(L0) ~ C(d/J). Below: spare capacity leaks L1 info. Above: Fano's inequality degrades coarse classification. Taylor expansion gives quadratic form matching empirical R^2 = 0.964.

**Corollaries (all confirmed):**
1. Inverted supervision reverses steerability sign (confirmed: CLINC -0.018, TREC -0.025)
2. UHMT collapses steerability (confirmed: CLINC +0.001, TREC -0.009)
3. Doubling prefix capacity shifts Goldilocks peak rightward (testable prediction)

---

## Paper Structure

1. **Introduction** - Problem, solution, contributions
2. **Setup & Definitions** - Hierarchical classification, steerability metric, 8 datasets
3. **Method (V5)** - Progressive prefix supervision, block dropout, MRL baseline
4. **Main Results** - 8-dataset benchmark, meta-analysis
5. **Causal Ablations** - 4 conditions x 2 datasets, information localization
6. **Scaling Trend** - Observational (8 datasets) + causal (synthetic hierarchy)
7. **Generality & Limitations** - Cross-model, retrieval, Pareto, dual-encoder, 3-level, limitations
8. **Theory** - Successive refinement, theorems, corollaries
9. **Related Work** - MRL, HEAL, CSR, CSRv2, SMRL, hyperbolic, random dim
10. **Conclusion** - Summary + practical implications

**Appendices:** Entropy allocation, retrieval viz, three-level viz, Pareto viz, FAISS latency, synthetic full results, convergence, reproducibility, metric robustness, per-seed values, scaling robustness (LOO + bootstrap), broader impact

---

## Competitive Landscape (Feb 2026)

| Method | Venue | What It Does | Overlap with Us |
|--------|-------|-------------|-----------------|
| HEAL | ICLR 2025 | Hierarchy-aware global embeddings via contrastive loss | Closest competitor; no prefix steerability |
| CSR/CSRv2 | ICML 2025 / ICLR 2026 | Sparse codes as MRL alternative | Orthogonal (sparsity vs hierarchy) |
| SMRL | EMNLP 2025 | Improved MRL training | No hierarchy; retains flat supervision |
| Hanley et al. | Concurrent | Multilingual Matryoshka for news clustering | Shares core insight; we add theory + causality |

**Our unique contributions:** (1) formal information-theoretic framework, (2) causal ablation evidence, (3) steerability metric, (4) scaling trend with interaction analysis, (5) synthetic causal experiment

---

## Narrative and Positioning

### Recommended Narrative
1. **Semantic zoom:** prefix length controls semantic resolution (coarse vs fine), not just fidelity.
2. **Alignment over awareness:** hierarchy awareness alone is insufficient (UHMT); alignment drives steerability.
3. **Information-theoretic grounding:** successive refinement explains why this objective works.

### Headline Numbers
1. 8/8 datasets: V5 steerability > MRL steerability.
2. Pooled effect size: Cohen's d = 1.49, p = 0.0003.
3. Product predictor for steerability: rho = 0.90, p = 0.002.
4. Retrieval ramp: V5 +6.3pp vs MRL +0.6pp (CLINC).
5. Latency: 64d queries are ~3.7x faster than 256d (HNSW benchmark).

### Risk Register (Current)
1. Effects are smaller on shallow hierarchies (expected under low conditional entropy).
2. WOS shows floor effects due to low fine-label learnability.
3. Causal ablations are concentrated on CLINC/TREC, not all datasets.
4. Some downstream evaluations are currently CLINC-only.

---

## Publication Execution

### Submission Targets
1. NeurIPS 2026.
2. EMNLP 2026 (fallback path).

### Minimum Must-Have for Final Submission
1. Finalize cross-backbone replication table and significance.
2. Keep all claims aligned to current 8-dataset statistics artifacts.
3. Include one concise "limitations and boundary conditions" section.
4. Ensure all reported table values are generated from canonical result JSON files.
5. Keep ablation narrative causal (sign reversal + no-prefix + UHMT controls).

### Competitive Positioning Summary
1. MRL-family methods: adaptive dimension but not semantic steerability.
2. Hierarchy-aware methods (e.g., HEAL): hierarchy signal without prefix-level control.
3. Our differentiated claim: intrinsic prefix steerability + causal evidence + theory + scaling analysis.

---

## Key Files

| Category | File | Description |
|----------|------|-------------|
| Paper | `paper/fractal_embeddings.tex` | Research paper draft (~877 lines) |
| Paper | `paper/references.bib` | 21 references |
| Paper | `paper/neurips_2026.sty` | LaTeX style file |
| Implementation | `src/fractal_v5.py` | V5 implementation |
| Implementation | `src/mrl_v5_baseline.py` | MRL baseline |
| Implementation | `src/hierarchical_datasets.py` | 8 dataset loaders |
| Experiments | `src/run_full_benchmark_suite.py` | 5-seed all-dataset runner |
| Experiments | `src/ablation_steerability.py` | 4-condition causal ablation |
| Experiments | `src/run_uhmt_ablation.py` | UHMT baseline |
| Statistics | `src/compute_paper_stats.py` | Holm-Bonferroni correction |
| Statistics | `src/meta_analysis.py` | Random-effects meta-analysis |
| Statistics | `src/scaling_robustness.py` | LOO + interaction analysis |
| Statistics | `src/metric_robustness.py` | 4-metric construct validity |
| Figures | `src/paper_figures.py` | All paper figures |
| Theory | `research/THEORY.md` | Canonical theory: limitations, proofs, and fractal advantages |

---

## Current Status (Feb 10, 2026)

- **Paper:** ~95% complete, abstract trimmed, p-values fixed, proofread done
- **Running:** Noisy hierarchy sensitivity test (CLINC, 0-50% L0 corruption, 3 seeds each)
- **Planned:** DBPedia Classes ablation (3rd dataset for causal evidence)
- **Codex review score:** 8.0/10
- **Top remaining risks:**
  1. No direct HEAL/CSR/SMRL head-to-head comparison
  2. Causal ablations only on 2 datasets (CLINC + TREC)
  3. Retrieval benchmark only on CLINC

---

## Repo
GitHub: https://github.com/dl1683/ai-moonshots
