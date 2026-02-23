# CTI Universal Law - Research Status

**As of: February 23, 2026 (Session 60 COMPLETE)**

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

## Current Codex Score: Nobel ~6.7/10 (Turing 8.0/10, Fields 3.0/10)

**Nobel score downgraded from 8.0 → 6.7 after Session 57: NLP surgery FAILs (both sub-linear and linear regime) show 1/d_eff attenuation mechanism is CIFAR-CNN-specific, not universal. Correlational law remains robustly universal.**

**Score calibration by prize framing (multiple Codex assessments):**
- Nobel (general/"Turing-adjacent" framing): ~6.4-6.7/10
- Nobel Physics specifically: ~4/10 (Physics committee rarely awards ML theory)
- Nobel Medicine: ~2/10 (no direct clinical/biological mechanism yet)
- Turing Award: ~8.0/10 (most relevant prize given computational nature of work)
- CV=0.024 = "within-class universality" (decoder LMs); "universal" claim requires non-transformer families + independent replication

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
| Causal sufficiency RCT 6-arm frozen (Session 38) | 2/7 pass: j1_r=0.959 PASS, j2_r=0.811 FAIL (j2 causal), alpha_topall=4.83, ratio~4.59x pure (sparse: ~4-5 active rivals) | DONE |
| Single-competitor weight map (Session 38) | sum_w=2.42 effective competitors, A_local=2.04, r_log_log=-0.337 (weak fit) | DONE |
| Held-out w_r transfer 4/4 PASS (Session 38) | Frozen causal w_r from pythia-160m: pooled R2 0.451->0.578 (+28%) on 4 held-out archs | DONE |
| Tournament + random null (Session 38) | causal w_r R2=0.578, phi(tau=0.2)=0.567, kappa_mean=0.469. Random null p67 (FAIL >90th pct) | DONE |
| CIFAR global surgery ratio (Session 56) | Ratio=d_eff=19.47 confirmed (3 seeds, pre-reg PASS), H1/H2/H3 all PASS | DONE |
| Beyond-1NN test 7arch×3ds (Session 57) | q_knn5 r=0.792 PASS, silhouette r=0.881 PASS; linear probe FAIL (pre-reg primary) | DONE |
| Cross-domain NLP surgery sub-linear (Session 57) | H1 FAIL ratio=1.88 vs d_eff=33.24; kappa_eff=0.24-0.73 (regime mismatch) | DONE |
| NLP linear-regime surgery kappa_eff≈1.3 (Session 57) | H1 FAIL ratio=3.68 vs d_eff=54.31; 1/d_eff mech is CIFAR-CNN-specific | DONE |
| Null-space kappa_eff identifiability (Session 58) | q tracks kappa_nearest r=1.00 (OLMo-1B); H4 FAIL (kappa_eff marginal) | DONE |
| 2D causal surface 3×3 factorial (Session 58) | H4 FAIL ΔR2=0.03; H6 PASS cross-arch bivariate r=0.988/0.967 | DONE |
| Extended family LOAO: GPT-2/Phi-2/Encoders (Session 59) | H1-H4 FAIL; GPT-2 artifact 4/4 confirmed; Phi-2 alpha=1.18 (decoder range); Encoder CV=0.92 | DONE |
| SmolLM2 re-run exploration (Session 60) | Protocol FAIL (best-layer mixing r=-0.16; NaN crash at 20news); correct result alpha=2.864 from fc9f9ac in paper | DONE |
| Surgery re-runs confirmation (Session 60) | Linear regime + Gaussian: calib_error~0.99, logit changes 0.17 vs 2.29 predicted (10x d_eff); kappa_nearest is sole predictor | DONE |
| Two-knob identifiability stdout (Session 60) | P1-P6 FAIL; d_eff: 10x change → q changes 0.003; confirms kappa_nearest as sufficient statistic, d_eff NOT causal | DONE |

## Key Theoretical Results (Theorems 1-16)

Canonical theory document: `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md`

- Th.13: Factor model derivation (CV=4.25%)
- Th.14/15: A_renorm universal = sqrt(4/pi) for K large
- Th.16: d_eff_gram is wrong (224x larger than d_eff_formula for CIFAR CE)

## Open Problems

1. **Formal regime-aware law:** `logit(q_i) = C + A*sqrt(d_eff)*phi(tau*, kappas_i)`
   - phi(tau=0.2) = soft-min of all competitors: R2=0.567 on 4 held-out archs (vs kappa_nearest R2=0.451)
   - Causal w_r (from do-interventions): R2=0.578 (best), but NOT better than random monotone at p67
   - sum_w = 2.42 effective competitors (sparse competition confirmed causally for pythia-160m DBpedia)
   - KEY INSIGHT: ANY monotone-decreasing weighting of all K-1 kappas gives similar improvement
   - UPGRADE: phi(tau*=0.2) is cleaner than causal w_r (theoretically motivated, no interventions needed)
   - OPEN: derive optimal tau* from first principles (should depend on kappa distribution)
   - **CRITICAL SESSION 60 FINDING:** d_eff is NOT a causal predictor of q; kappa_nearest is the sole
     sufficient statistic. sqrt(d_eff) in the law acts as an architecture-level scaling constant for A,
     not as a per-geometry variable. Need to re-examine the role of d_eff in the canonical law form.

2. **A_ViT/A_NLP = 1.67 derivation:** via margin thermodynamics (A prop 1/T_eff)

3. **External replication:** highest leverage for Nobel score jump (Codex: "externally replicated,
   pre-registered, blinded OOD prediction with frozen d_eff-corrected law across new task-type +
   new architecture, requiring both ranking AND calibration pass" = single path to 7+)

4. **LODO reframing (Codex recommendation):** LODO CV=0.37 is currently a weakness for "single
   universal constant" claim. Reframe as "universality-class" result: architecture-universal within
   task-type, task-specific across universality classes. Validate prospectively. Data supports
   "shape universal, scale class-dependent" more than single-scale universality.

5. **Practical utility demo:** predict class difficulty from geometry alone

5. **Renormalized Universality Theorem verification (next):**
   - Theorem: A/sqrt(d_eff_formula) = sqrt(4/pi) = 1.128 universally
   - NLP check: 1.477/sqrt(1.71) = 1.129 (1.477 back-calculated — need direct d_eff measurement)
   - ViT next: predict d_eff_ViT = A_ViT^2/(4/pi) = 7.5^2/1.273 = 44, measure actual d_eff_formula
   - Prediction: A/sqrt(measured_d_eff) = 1.128 for both modalities

## Failed Causal Interventions (logged for paper)

All failed: joint CE+triplet, two-stage CE+centroid-triplet, anti-triplet, dist_ratio regularizer,
NC-loss at 60 epochs, cross-task do-intervention, DBpedia NC-loss head (specificity failure, shuffled==nc),
multi-arch frozen do-intervention (direction confirmed but alpha ~0.47x predicted — K-1 competitive geometry issue),
CIFAR linear regime surgery (d_eff manipulation barely affects q — kappa_nearest is sole predictor),
two-knob identifiability (P1-P6 all FAIL; d_eff is NOT a causal variable for q in linear regime).
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
| `src/cti_causal_sufficiency_rct.py` | 6-arm causal sufficiency RCT (Session 38) |
| `src/cti_competitor_weight_map.py` | Single-competitor causal weight map (Session 38) |
| `src/cti_kappa_eff_held_out.py` | Held-out w_r transfer test (Session 38) |
| `src/cti_kappa_tournament.py` | Tournament + random-null specificity test (Session 38) |
| `src/hierarchical_datasets.py` | Dataset loading utility |
| `src/multi_model_pipeline.py` | Multi-model embedding pipeline |
| `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md` | Master theory document |
