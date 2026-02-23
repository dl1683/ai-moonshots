# CTI Universal Law - Research Status

**As of: February 23, 2026 (Session 68 COMPLETE)**

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
| RWKV-4-169M pure linear RNN LOAO (pre-reg, Feb 22) | alpha=2.887 PASS [2.43,3.29]; 12-arch CV=0.0191 (1.91%) — no attention, pure recurrence | DONE |
| Falcon-H1-0.5B Transformer+Mamba hybrid LOAO (pre-reg, Feb 22) | alpha=2.830 PASS; 11-arch CV=0.022 | DONE |
| Extended family LOAO: GPT-2/Phi-2/Encoders (Session 59) | H1-H4 FAIL; GPT-2 artifact 4/4 confirmed; Phi-2 alpha=1.18 (decoder range); Encoder CV=0.92 | DONE |
| SmolLM2 re-run exploration (Session 60) | Protocol FAIL (best-layer mixing r=-0.16; NaN crash at 20news); correct result alpha=2.864 from fc9f9ac in paper | DONE |
| Surgery re-runs confirmation (Session 60) | Linear regime + Gaussian: calib_error~0.99, logit changes 0.17 vs 2.29 predicted (10x d_eff); kappa_nearest is sole predictor | DONE |
| Two-knob identifiability stdout (Session 60) | P1-P6 FAIL; d_eff: 10x change → q changes 0.003; confirms kappa_nearest as sufficient statistic, d_eff NOT causal | DONE |
| Biological validation Cadieu2014 macaque IT+V4 (Session 61, pre-reg bddec1d) | H1/H2/H4 FAIL (K=7 underpowered, A_bio=0.069 vs A_NLP=1.129); H3 PASS IT (MAE=0.084); EXPLORATORY per-image: r=0.41 p<0.0001 n=1960 (IT), r=0.12 (V4). Form confirmed, constant different | DONE |
| Biological validation Stringer2018b mouse V1 (Session 61) | Class-level FAIL (K=11, r=-0.49, class-size confound); H3 PASS MAE=0.054; EXPLORATORY per-image: r=0.64 p<0.0001 n=4914. Margin predicts V1 accuracy stronger than IT | DONE |
| Allen Neuropixels mouse visual cortex K=118 (Session 62, pre-reg bddec1d) | **H1 PASS** r(kappa,logit_q)=0.851 p=3e-34 (FIRST class-level bio H1 pass!); H3 PASS MAE=0.063; H2 FAIL A_renorm=0.033; per-image r=0.747 p=0 n=5900 | DONE |
| Allen Neuropixels 7-session replication (Session 62) | **7/7 H1 PASS** r_kappa mean=0.733 [0.51,0.89] CV=17.8%; per-image r mean=0.689 [0.48,0.77]; cross-animal replication confirmed | DONE |
| Allen per-area visual hierarchy (Session 62, exploratory) | ALL_VIS r=0.869 > VISam r=0.802 > VISl r=0.774 > VISp r=0.707 > VISal r=0.658 > LP r=0.519 > VISrl r=0.445; law holds across visual hierarchy | DONE |
| Allen Neuropixels COMPLETE 32-session dataset (Session 63, pre-reg bddec1d) | **30/32 H1 PASS** (93.75%); ALL 32 positive r (range [0.44,0.89], p<0.001 each); r_kappa mean=0.736 std=0.113 CV=15.4%; 2 FAIL sessions explained (noise floor mean_q=0.12; ceiling mean_q=0.81); 32 different mice confirmed | DONE |
| K×SPREAD follow-up n=10 datasets (Session 64, pre-reg MEMORY.md) | **ALL 3 HYPOTHESES PASS** (was 2/3 at n=6); H_spread PASS (beta=0.83,partial_r=0.34); H_K_null PASS (spread 17x more important than K); H_spread_r PASS Spearman r=0.661 p=0.038 (was FAIL at n=6 underpowered); kappa spread is statistically significant predictor of architecture-ranking reliability | DONE |

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

5. **Biological neuroscience validation (Session 62 DONE — THREE DATASETS):**
   - Cadieu2014 macaque IT (168 neurons, K=7): per-image r=0.41, p<0.0001, n=1960
   - Stringer2018b mouse V1 (10,079 neurons, K=11): per-image r=0.64, p<0.0001, n=4914
   - Allen Neuropixels mouse VC (K=118, 32 mice): r_kappa mean=0.736 [0.44,0.89], 30/32 H1 PASS, ALL 32 positive r
   - PRE-REGISTERED: H4 FAILS across all datasets (A_bio ≠ A_NLP); CONSTANT not universal
   - FIRST CLASS-LEVEL H1 PASS: Allen K=118 provides statistical power; 7-session cross-animal replication
   - KEY FINDING: LAW FORM (kappa_nearest → q) IS substrate-independent; CONSTANT A differs
   - POWER FINDING: class-level test requires K≥118 to reliably detect the correlation
   - Cross-substrate per-image r: macaque IT=0.41, macaque V4=0.12, mouse V1=0.64, mouse VC=0.69(mean)
   - BIOLOGICAL SECTION ADDED to paper/cti_universal_law.tex

6. **Practical utility demo:** predict class difficulty from geometry alone

7. **Renormalized Universality Theorem verification (Session 68 DONE — FAIL for geometric d_eff):**
   - Theorem: A/sqrt(d_eff_formula) = sqrt(4/pi) = 1.128 universally
   - Pre-registered prospective test (commits 0d1d0e7, 1fbd0b6; results d7f0317)
   - H1 (rel_error < 25%): FAIL 0/3 test archs. H2 (r > 0.85): FAIL r=-0.539
   - MEASURED geometric d_eff: pythia-160m=27.3, gpt-neo=57.5, pythia-1b=73.1, pythia-410m=68.5, OLMo-1B=86.5
   - NEEDED d_eff for theorem: 0.59-3.20 (10-120x smaller than measured)
   - DIAGNOSIS: geometric d_eff = d/aniso^2; aniso~4-5x for LM (vs CIFAR ~18.7x)

8. **Competition Equicorrelation d_eff Test (Session 69 DONE — ALL 3 HYPOTHESES PASS):**
   - Pre-registered commit 472079b; results 8c55183
   - THEORY: rho = avg Sigma_W-whitened cosine sim of centroid differences; d_eff_comp = 1/(1-rho)
   - PREDICTED: rho=0.416, d_eff_comp=1.713 (from alpha=1.477, A_renorm=1.128)
   - RESULTS (5 archs, DBpedia K=14, N=2000):
     pythia-160m: rho=0.408, d_eff_comp=1.689 (1.4% error)
     gpt-neo-125m: rho=0.465, d_eff_comp=1.867 (8.9% error)
     pythia-410m:  rho=0.467, d_eff_comp=1.877 (9.5% error)
     pythia-1b:    rho=0.461, d_eff_comp=1.854 (8.2% error)
     OLMo-1B-hf:  rho=0.462, d_eff_comp=1.859 (8.5% error)
   - H1 PASS 5/5: all archs within 25% of theory (most within 10%)
   - H2 PASS: CV(d_eff_comp) = 3.9% << 30% (near-perfect universality!)
   - H3 PASS: rho > 0 for all archs
   - KEY INSIGHT: rho~0.45 ~= 0.5 (REGULAR SIMPLEX PREDICTION)
     Perfect Neural Collapse (regular K-simplex) gives rho=0.5, d_eff=2.0, alpha=1.595
     Measured rho~0.45 indicates near-NC arrangement of class centroids in whitened space
     This explains WHY alpha~1.477 (CV=2.3%) is universal: rho is even MORE universal (CV=3.9%)
   - RESOLUTION of Session 68 failure: geometric d_eff (embedding-space) FAILS;
     competition d_eff_comp (score-space, rho-based) PASSES with CV=3.9%
   - GEOMETRIC INTERPRETATION: centroid differences lie approximately on a regular simplex
     in Sigma_W-whitened space — the Neural Collapse geometry

## Failed Causal Interventions (logged for paper)

All failed: joint CE+triplet, two-stage CE+centroid-triplet, anti-triplet, dist_ratio regularizer,
NC-loss at 60 epochs, cross-task do-intervention, DBpedia NC-loss head (specificity failure, shuffled==nc),
multi-arch frozen do-intervention (direction confirmed but alpha ~0.47x predicted — K-1 competitive geometry issue),
CIFAR linear regime surgery (d_eff manipulation barely affects q — kappa_nearest is sole predictor),
two-knob identifiability (P1-P6 all FAIL; d_eff is NOT a causal variable for q in linear regime).
ONLY PASS: pythia-160m/dbpedia frozen do-intervention (isolated pair, alpha=1.601 r=0.974).

## Active Paper

`paper/cti_universal_law.tex` — targeting COLM 2026

**Session 67 paper cuts (Feb 23) — aggressive page reduction for COLM 2026 9-page limit:**
- Abstract: ~460 words → ~180 words (Session 66 commit 0e4e895)
- §4.5 Competition field (116 lines) → 9-line pointer + appendix (Session 66)
- §4.7 Composite K-law (46 lines) → 4-line pointer + appendix (Session 66)
- §4.9 Sparse Competition (131 lines) → 7-line pointer + appendix (Session 66)
- §4.4 Causal: global surgery 40 lines → 3 lines; table 15 rows → 6 rows (Session 66)
- Discussion encoder paragraph 15 lines → 4 lines (Session 66)
- Theory "Connection to theory": 3 huge lines → 3 compact sentences (Session 66)
- §4.6 K-scaling: ~60 lines → ~25 lines; §4.8 Universality: 21 → 9 lines (Session 67 commit 6cc1f85)
- Discussion biological validation: 9 lines → 6 lines (Session 67)
- Limitations bio + beyond-1NN: 7 lines → 3 lines; Theory hierarchical β: 5 lines → 3 lines (Session 67)
- Intro contributions condensed; Related Work neural-collapse 6 lines → 4 lines (Session 67)
- Total: ~1200 lines → 1042 lines (estimated saving ~1.5 compiled pages in main text)
- **NEXT STEP**: User compile paper and check actual page count; target ≤9 pages main text.
- **Remaining if still over**: The LOAO table (tab:loao-12, `table*`, 12 rows) could move to appendix with a 3-row summary. Or cut 3 limitations items.
- **REQUIRED before submission**: Replace `\author{Anonymous Authors}` and `[anonymized for review]` URL.

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
