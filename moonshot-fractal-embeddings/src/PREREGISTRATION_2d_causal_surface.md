# Pre-Registration: 2D Causal Surface (Bivariate kappa × kappa_eff Law)

**Registered:** 2026-02-23 (after identifiability test result at commit a1458cf)

## Motivation

The identifiability test (commit 6938940) established:
1. Null-space scaling: kappa_eff invariant, kappa changes → q tracks kappa (H5: r=1.00 for OLMo-1b)
2. Global surgery (commit 1abdef6): kappa invariant (tr preserved), kappa_eff changes → q tracks kappa_eff

These two pre-registered interventions show BOTH kappa (total SNR) and kappa_eff (directional SNR)
independently cause q. This motivates the bivariate law: logit(q) = alpha1*kappa + alpha2*kappa_eff + C.

The current one-parameter LOAO law (logit(q) = alpha*kappa + C) works because within each dataset,
d_eff ≈ constant across architectures, making kappa and kappa_eff nearly collinear. The 2D
causal surface experiment tests the full bivariate law across a factorial manipulation grid.

## Method: Orthogonal 2D Factorial

**Key insight**: Two interventions are orthogonal:
- Null-space scaling (s_null): scales within-class residuals in null(P_B) by s_null.
  Changes kappa (via tr(Sigma_W)), INVARIANT kappa_eff (= delta_min/sigma_cd).
- Compensated signal-space scaling (s_signal): scales within-class residuals in span(P_B)
  by s_signal, compensates null-space to PRESERVE tr(Sigma_W).
  Changes kappa_eff (via sigma_cd), INVARIANT kappa.

Combined: apply null by s_null AND signal by s_signal (where s_null is set to BOTH
achieve the desired kappa level AND compensate for signal scaling):

Given target (a, b) = (kappa_new/kappa_base, kappa_eff_new/kappa_eff_base):
- s_signal = 1/b  (since kappa_eff_new = kappa_eff_base/s_signal → s_signal = 1/b)
- s_null^2 = (a^2 * trW - trW_sig/b^2) / trW_null

## Pre-Registered Design

**Dataset**: 20newsgroups, K=20, N=5000 (from causal_v2_embs_*.npz)
**Architectures**: olmo-1b, qwen3-0.6b (deberta excluded: trW_null≈0, test vacuous)
**Train/test split**: 4000/1000, stratified (seed=42, same as identifiability test)

**2D Grid (target fractions):**
- (a, b) ∈ {0.5, 1.0, 2.0} × {0.5, 1.0, 2.0}
- 9 grid points per architecture (minus infeasible: a=0.5, b=0.5 infeasible for olmo-1b)
- Feasibility: requires s_null^2 >= 0.01 (otherwise skip)

## Pre-Registered Hypotheses

**H1 (IMPLEMENTATION CHECK)**: At (a=1.0, b=1.0) (baseline), kappa and kappa_eff within
1% of baseline values (verifies orthogonal construction).

**H2 (kappa_eff invariance in null-only row)**: At (a=0.5, 1.0, 2.0) with b=1.0 (null-only):
kappa_eff varies < 2% (theoretical invariance confirmed empirically).

**H3 (kappa invariance in signal-only column)**: At (a=1.0) with b=0.5, 1.0, 2.0 (signal+compensation):
kappa varies < 2% (compensation works).

**H4 (PRIMARY -- BIVARIATE FIT)**: Fit logit(q) = alpha1*kappa + alpha2*kappa_eff + C
on all feasible grid points per architecture. Report R² and compare to:
- Univariate kappa alone: R²_kappa
- Univariate kappa_eff alone: R²_kappa_eff
- Bivariate: R²_bivariate
H4 PASS: R²_bivariate > max(R²_kappa, R²_kappa_eff) + 0.05 (meaningful improvement).

**H5 (SIGN CHECK)**: alpha1 > 0 AND alpha2 > 0 (both components help, consistent with
kappa ↑ → q ↑ and kappa_eff ↑ → q ↑).

**H6 (LOAO TRANSFER)**: Fit bivariate model on olmo-1b, predict qwen3-0.6b (and vice versa).
R²_transfer >= 0.90 on held-out architecture. Tests whether the bivariate coefficients transfer.

## Evaluation

OVERALL PASS: H4 + H5 on at least 1 architecture, H6 on cross-arch transfer.

If PASS: "The complete geometry law is logit(q) = alpha1*kappa + alpha2*kappa_eff + C;
both total SNR (kappa) and directional SNR (kappa_eff) independently cause q,
validated by orthogonal causal interventions." Nobel score expected: ~7.5-8.0/10.

If FAIL: bivariate does not improve over univariate; the two mechanisms act redundantly
rather than independently in the regression context (even if both causal).

## Script

`src/cti_2d_causal_surface.py` -- to be written AFTER this pre-registration is committed.
