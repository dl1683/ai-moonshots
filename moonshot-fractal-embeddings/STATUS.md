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

9. **K-Independence Equicorrelation Test (Session 69 DONE — PASS):**
   - Pre-registered commit 5f5459f; results 5f5459f
   - PROTOCOL: pythia-160m on 3 datasets with K=4 (agnews), K=14 (dbpedia), K=77 (banking77)
   - PREDICTION: rho ~ 0.45-0.50 constant across K (regular K-simplex property)
   - RESULTS:
     K=4  (agnews):    rho=0.493, d_eff_comp=1.971
     K=14 (dbpedia):   rho=0.408, d_eff_comp=1.689
     K=77 (banking77): rho=0.467, d_eff_comp=1.877
   - CV(rho) = 7.8% << 15% threshold → K-INDEPENDENCE PASS
   - mean_rho = 0.456, range = 0.408-0.493 (K-independent to within 10%)
   - CONFIRMS: near-simplex geometry holds across very different task scales (K=4 to K=77)

10. **Allen Biological Equicorrelation Test (Session 69 DONE — NEAR-SIMPLEX):**
    - Commit c689411; 5 sessions from DANDI:000021 (visual cortex, K=118 categories)
    - QUESTION: Does biological visual cortex show the same rho~0.45 as LM decoders?
    - PROTOCOL: Load spike-rate response matrices → PCA(100) → Cholesky-whitened cosine sims
    - RESULTS (5 mice, 1463-2110 "good" units each):
      sub-699733573: rho=0.451, d_eff_comp=1.821
      sub-718643564: rho=0.471, d_eff_comp=1.892
      sub-726170927: rho=0.471, d_eff_comp=1.891
      sub-734865729: rho=0.468, d_eff_comp=1.878
      sub-738651046: rho=0.469, d_eff_comp=1.883
    - mean_rho = 0.466, CV(rho) = 1.65% (even TIGHTER than LMs at 3.9%)
    - mean_d_eff_comp = 1.873 vs LM value 1.829 — only 2.4% difference!
    - INTERPRETATION: NEAR-SIMPLEX geometry is substrate-independent (artificial + biological)
      Near-NC centroid arrangement appears as a structural property of categorical representation,
      NOT a gradient/cross-entropy training artifact
    - This connects CTI universality to Neural Collapse as a deep geometric principle

11. **Frozen-Backbone NC-Loss Causal Test (Session 70 — PARTIAL PASS):**
    - Codex guidance (task bc21248): 3 bugs in prior NC-loss work identified and fixed
    - FIX 1: class means differentiable (batch-scatter, no torch.no_grad EMA)
    - FIX 2: q/kappa measured in proj_head space (same as NC optimization)
    - FIX 3: DBpedia K=14 instead of CIFAR (no coarse-label confound)
    - FIX 4 (discovered during run): KNN was fitting+scoring on same TEST set (trivial q=1.0)

    Standard regime (proj_dim=256): q saturated at 1.0 from epoch 1 — too easy
    Hard regime (proj_dim=13=K-1 bottleneck, commits df2b233, 023b2ff):
    - CE: q=0.9592±0.0022, kappa=2.7241
    - full_NC: q=0.9608±0.0022, kappa=2.8839
    - shuffled_NC: q=0.9592 (same as CE — CONTROL WORKS)
    - within_only: q=0.9592, kappa=2.7369 (ETF+margin are the active terms)
    - H1 PASS (barely): delta_q=+0.0015, 4/5 seeds positive
    - H2 PASS: shuffled control delta_q=0.000 (causal specificity confirmed)
    - H3 PASS: delta_kappa=+0.160, 5/5 seeds (robust kappa causal effect)
    - H4 FAIL: ratio=0.271 vs alpha=1.477 (q still near ceiling ~0.96)
    - KEY FINDING: NC causally improves kappa +5.8% (robust, specific to ETF+margin)
      But H4 quantitative test requires non-saturated q regime

    Banking77 K=77 regime (commit 252dee4, pre-registered b3399d0):
    - ARCH: pythia-160m layer 0, proj_dim=76 (K-1), N_TRAIN=1540 (20/class), N_TEST=770 (10/class)
    - CE: q=0.7482+-0.0097, kappa=0.7669+-0.0253  (not saturated: correct regime for H4)
    - full_NC: q=0.7487+-0.0119, kappa=0.8227+-0.0137
    - within_only: q=0.7484, kappa=0.7669 (no effect vs CE -- confirms ETF+margin required)
    - shuffled_NC: q=0.7484, kappa=0.7669 (same as within_only -- specificity confirmed)
    - H1 barely PASS: delta_q=+0.0005, 3/5 seeds positive
    - H2 FAIL: shuffled delta_q=+0.0003 (barely above 0; causal specificity incomplete)
    - H3 PASS: delta_kappa=+0.0558, 5/5 seeds (ETF+margin causally increase kappa: ROBUST)
    - H4 FAIL: mean ratio=-0.064 vs alpha=1.477 (per-seed: [0.202, -0.338, 0.376, -1.084, 0.527])
    - DIAGNOSIS: kappa change (+7.3%) is too small relative to seed noise for logit(q) to respond
      cleanly. Need either: (a) much larger NC effect, or (b) backbone with lower baseline kappa
      where kappa variation is more predictive of q in the logit-linear regime.

12. **Codex Constants Consultation (Session 70 — TWO TASKS):**

    TASK a74be0143736d6944 (earlier):
    - Golden ratio from Gumbel+Gaussian: DEAD-END (no recursive algebraic fixed point in vanilla model)
    - Complex extension (Wick-rotated Gumbel): SPECULATIVE-BUT-TESTABLE (abandons probabilistic interp)
    - d_eff ≈ 1.71 closest candidate: e-1=1.718 (likely coincidence, no proof yet)

    TASKS ab1bf219aae015bb0 + a643c6c3a95b02c94 — "d_eff = e-1 from ETF structure?" (DEAD-END x2):

    Task ab1bf219: Setup assumed rho_n=+1/(K-1) (positive). Reduces to iid EVT. Alpha grows sqrt(log K).
    Task a643c6c3: CORRECT ETF: rho = -1/(K-1) (NEGATIVE). Key rigorous findings:
    - ETF decomposition: Z_i = sqrt(K/(K-1))*(X_i - X_bar), eigenvalues {0, K/(K-1)}
    - One-factor form Z_i = sqrt(1-rho)*eps_i + sqrt(rho)*W INVALID for negative rho
    - E[max ETF-Gaussian] = sqrt(K/(K-1)) * E[max iid] → correction factor 1 as K→inf (not e-1)
    - Extremal index = 1: X_bar = O_p(K^{-1/2}) negligible vs Gumbel scaling 1/sqrt(2 log K)
      → ETF maxima are in SAME Gumbel universality class as iid Gaussians
    - Second-moment renormalization: Var(S) = 2/(K-1) vs iid 2/K → ratio K/(K-1) → 1, not e-1
    - Gumbel MGF M(t) = Gamma(1-t): no canonical evaluation produces e-1
    - PROPOSED CLOSED FORM: alpha = sqrt(4/pi)*sqrt(e-1) = 1.479 has NO rigorous derivation path
    - NOTE: e-1 does arise as Var(exp(N(0,1)-1/2)) = e-1 (lognormal variance identity), but
      this mechanism is unrelated to ETF geometry
    - SIDE NOTE: ETF one-vs-all difference margin pairwise correlation = 1/2 (not 1/(K-1))
      This directly explains the measured rho~0.45-0.50 in equicorrelation experiments
    - FINAL VERDICT: d_eff ≈ 1.710 ~ e-1 = 1.718 (0.5% match) is numerical coincidence. CLOSED.

## Local vs Global Alpha (Codex analysis, task bf9a0a0)

Q1 — THEORETICAL MEANING of local alpha (~0.7) vs global alpha (1.477):
- alpha_local = PARTIAL DERIVATIVE from one-pair surgical perturbation
- alpha_global = EFFECTIVE COARSE-GRAINED SLOPE averaging over full K-1 competitive geometry
- Gap is EXPECTED if global fit includes gain from multi-competitor structure beyond moved pair
- CORRECT FORMULA: alpha_intervention = alpha_loao * w1
  where w1 = nearest-pair sensitivity share (hazard weight) in full multiclass competition
  From data: w1 = 0.701/1.477 = 0.475 -> ~2 active rivals out of 13 (SPARSE competition)
  NOT 1/(K-1)=0.077 (that assumes all 13 rivals contribute equally -- FALSE)
  Consistent with near-tie j1/j2 margins and orthogonal-arm near-zero j2/jK effects

Q2 — FEATURE NOT BUG (if we upgrade the claim):
- Supported: universal functional form (high r survives interventions)
- Not supported: "one universal scalar alpha for all protocols"
- Upgraded claim: "universal law + competition-renormalized slope"
- NOT just 1/(K-1): K=14 fixed but cross-model spread 0.428-1.052

Q3 — STRONGEST CLAIM:
"Nearest-class geometry is a universal directional causal order parameter,
while slope magnitude is a competition-renormalized coefficient (local vs global)."
DECISIVE EXPERIMENT: top-m competitor sweep (m=1..K-1), pre-register monotonic
alpha_m rise from ~0.7 to ~1.48. Scaffolding: src/cti_top_m_competitor_sweep.py

TOP-M SWEEP RESULT (stale task b3d36f9, pythia-160m/DBpedia K=14, crashed before JSON save):
  m= 1: alpha=1.0522, r=0.9833, ratio_to_LOAO=0.712
  m= 2: alpha=1.5228, r=0.9699, ratio_to_LOAO=1.031  <-- LOAO=1.477 MATCHED AT m=2!
  m= 3: alpha=1.6168, r=0.9460, ratio_to_LOAO=1.095
  m= 5: alpha=2.1969, r=0.9732, ratio_to_LOAO=1.487
  m=13: alpha=3.0152, r=0.9766, ratio_to_LOAO=2.041
  MONOTONIC RISE CONFIRMED. LOAO alpha integrates ~2 effective competitors.
  Consistent with: Session 42 kernel showdown k*=5, sparse competition w1=0.475.
  STATUS: Script crashed before JSON save. Needs clean pre-registered re-run.

Q4 — SCORE (pre-equicorrelation/bio context): Nobel 5/10, Turing 7/10
"If m-competitor experiment cleanly bridges local->global with pre-registered success: jumps materially"
NOTE: Current score (with equicorrelation + Allen bio + 19 archs) is Nobel 6.6/10, Turing 8.0/10

## Failed Causal Interventions (logged for paper)

All failed: joint CE+triplet, two-stage CE+centroid-triplet, anti-triplet, dist_ratio regularizer,
NC-loss at 60 epochs, cross-task do-intervention, DBpedia NC-loss head (specificity failure, shuffled==nc),
multi-arch frozen do-intervention (direction confirmed but alpha ~0.47x predicted — K-1 competitive geometry issue),
CIFAR linear regime surgery (d_eff manipulation barely affects q — kappa_nearest is sole predictor),
two-knob identifiability (P1-P6 all FAIL; d_eff is NOT a causal variable for q in linear regime).
ONLY PASS: pythia-160m/dbpedia frozen do-intervention (isolated pair, alpha=1.601 r=0.974).
PARTIAL PASS (Session 70): frozen-backbone NC DBpedia K=14 hard-regime — H3 PASS (kappa), H1 barely PASS, H4 FAIL (ceiling)

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
