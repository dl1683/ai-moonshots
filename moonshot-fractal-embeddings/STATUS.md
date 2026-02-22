# CTI Universal Law - Research Status

**As of: February 22, 2026**

## The Discovery

**CTI Universal Law (canonical form):**

    logit(q_norm) = A_renorm(K) * kappa_nearest * sqrt(d_eff_formula) + C

Where:
- `q_norm = (acc - 1/K) / (1 - 1/K)` — normalized 1-NN accuracy
- `kappa_nearest = delta_min / (sigma_W_global * sqrt(d))` — nearest-class separation signal-to-noise ratio
- `d_eff_formula = tr(Sigma_W) / sigma_centroid_dir^2` — effective dimensionality (anisotropy ratio)
- `A_renorm(K=20) = 1.0535` — pre-registered universal constant (Theorem 15)

**Observed alpha from LOAO-12 NLP architectures:** alpha = 1.477, CV = 2.3% (PASS)

---

## Current Codex Score: Nobel 4.7/10, Turing 6.4/10, Fields 3.0/10

---

## Key Validated Results

| Experiment | Result | Status |
|---|---|---|
| LOAO 7 architectures | alpha=1.536+/-0.067, CV=4.4% PASS | DONE |
| LOAO 12 NLP architectures | alpha=1.477, CV=2.3% PASS | DONE |
| Frozen do-intervention pythia-160m/dbpedia | alpha=1.601, r=0.974 PASS | DONE |
| Per-class formula (3 seeds, K=20) | partial r(d_eff|kappa)=0.493 p=0.0001 | DONE |
| NLP orthogonal causal factorial | Arm A r=0.899, Arm B r=0.450 (1-layer) | DONE |
| Margin phase diagram (K=3) | r(margin, B_j2_r)=-0.988 p<0.0001 PASS | DONE |
| ViT cross-modality LOAO | R2=0.96, A_ViT=0.63 (structural universality) | DONE |
| ViT orthogonal factorial | Arm A r=0.943, Arm B r=0.945 (2-layer), Arm C FAIL (dense regime) | DONE |
| Rank-spectrum factorial | r=0.459 p<0.0001 (directional pass, pre-reg threshold 0.85 not met) | DONE |
| Checkpoint phase diagram | B_j2_r geometrically driven (step512 B=0.831) | DONE |
| DBpedia NC-loss head intervention (Session 37) | FAIL (specificity): shuffled_nc=nc_full, pythia-410m ceiling | DONE |
| Multi-arch frozen do-intervention (Session 37) | PARTIAL: r>0.87 all 5 models (direction PASS), alpha=0.701 avg (quantitative FAIL) | DONE |

## Key Theoretical Results (Theorems 1-16)

Canonical theory document: `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md`

- Th.13: Factor model derivation (CV=4.25%)
- Th.14/15: A_renorm universal = sqrt(4/pi) for K large
- Th.16: d_eff_gram is wrong (224x larger than d_eff_formula for CIFAR CE)

## Open Problems

1. **Formal regime-aware law:** `logit(q_i) = A*sqrt(d_eff)*[kappa_j1 + lambda*sum w_r*kappa_jr]`
   - NLP: lambda->0 (sparse); ViT CIFAR-10: lambda>0 (dense)
   - Need to derive w_r = exp(-A_local * delta_kappa_r * sqrt(d_eff)) with A_local approx 1.75 (universal?)

2. **A_ViT/A_NLP = 1.67 derivation:** via margin thermodynamics (A prop 1/T_eff)

3. **External replication:** highest leverage for Nobel score jump

4. **Practical utility demo:** predict class difficulty from geometry alone

5. **Renormalized Universality Theorem verification (next):**
   - Theorem: A/sqrt(d_eff_formula) = sqrt(4/pi) = 1.128 universally
   - NLP check: 1.477/sqrt(1.71) = 1.129 (1.477 back-calculated — need direct d_eff measurement)
   - ViT next: predict d_eff_ViT = A_ViT^2/(4/pi) = 7.5^2/1.273 = 44, measure actual d_eff_formula
   - Prediction: A/sqrt(measured_d_eff) = 1.128 for both modalities

## Failed Causal Interventions (logged for paper)

All failed: joint CE+triplet, two-stage CE+centroid-triplet, anti-triplet, dist_ratio regularizer,
NC-loss at 60 epochs, cross-task do-intervention, DBpedia NC-loss head (specificity failure, shuffled==nc),
multi-arch frozen do-intervention (direction confirmed but alpha ~0.47x predicted — K-1 competitive geometry issue).
ONLY PASS: pythia-160m/dbpedia frozen do-intervention (isolated pair, alpha=1.601 r=0.974).

## Active Paper

`paper/cti_universal_law.tex` — targeting COLM 2026

## Key Source Files

| File | Purpose |
|---|---|
| `src/cti_kappa_nearest_universal.py` | Main LOAO universality test |
| `src/cti_extend_loao.py` | Extended LOAO (12 archs) |
| `src/cti_vit_loao.py` | ViT cross-modality validation |
| `src/cti_checkpoint_sweep.py` | Checkpoint phase diagram |
| `src/cti_orthogonal_factorial.py` | NLP causal factorial |
| `src/cti_vit_orthogonal_factorial.py` | ViT causal factorial |
| `src/cti_rank_spectrum_factorial.py` | Rank-spectrum test |
| `src/cti_do_intervention_text.py` | Frozen do-intervention |
| `src/cti_profile_likelihood.py` | Profile likelihood CI |
| `src/cti_two_step_analysis.py` | Two-step causal test |
| `src/cti_fit_universal_law.py` | Law fitting utilities |
| `src/cti_held_out_universality.py` | Held-out architecture test |
| `src/cti_dbpedia_nc_intervention.py` | NC-loss head intervention (frozen backbone) |
| `src/cti_do_intervention_multi_arch.py` | Multi-arch frozen do-intervention (5 models) |
| `src/hierarchical_datasets.py` | Dataset loading utility |
| `src/multi_model_pipeline.py` | Multi-model embedding pipeline |
| `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md` | Master theory document |
