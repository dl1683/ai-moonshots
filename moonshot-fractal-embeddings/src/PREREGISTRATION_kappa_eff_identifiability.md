# Pre-Registration: kappa_eff Identifiability via Null-Space Scaling

**Registered:** 2026-02-23 (after NLP linear-regime surgery FAIL at commit 5ccf526)

## Context / Motivation

Both NLP surgery tests failed H1: the 1/d_eff attenuation ratio is CIFAR-CNN-specific.
The root cause hypothesis: in NLP, the nearest-centroid direction spans a LARGER fraction of
the signal subspace than in trained CNNs, making single vs. global surgery non-negligible.

**Deeper question**: Is kappa_eff = kappa * sqrt(d_eff) = delta_min / sigma_centroid_dir
the TRUE primitive variable, or is kappa alone sufficient?

**Key mathematical insight**: Null-space scaling of within-class residuals EXACTLY preserves
kappa_eff while changing kappa and d_eff independently.

### Why null-space scaling decouples kappa from d_eff

Let P_B = K-1 principal directions of centered centroids (signal subspace).
Let x_c = x - mu_k(x) = within-class residual.
Apply: x_c' = x_c_signal + s * x_c_null, where x_c_signal = P_B P_B^T x_c

Then (with e_1 = Delta_hat in signal subspace):
- tr(Sigma_W) CHANGES: tr(Sigma_W_sig) + s^2 * tr(Sigma_W_null) [increases for s>1]
- sigma_centroid_dir = e_1^T Sigma_W e_1 is INVARIANT (e_1 orthogonal to null space)
- Therefore kappa_eff = delta_min / sigma_centroid_dir is EXACTLY INVARIANT to s
- kappa = delta_min / sqrt(tr(Sigma_W)) DECREASES for s > 1 (more null variance)
- d_eff = tr(Sigma_W) / sigma_centroid_dir^2 INCREASES for s > 1

This creates a genuine dual-knob: kappa and d_eff move in OPPOSITE DIRECTIONS
while kappa_eff stays fixed. If q tracks kappa_eff (stays constant), the theory is validated.

## Pre-Registered Design

**Dataset**: 20newsgroups, K=20, N=5000
**Architectures**: All available pre-saved embeddings from causal_v2_embs_*.npz:
- deberta-base: (5000, 768)
- olmo-1b: (5000, 2048)
- qwen3-0.6b: (5000, 1024)
**Train/test split**: 4000/1000, stratified (seed=42)
**Null-space scaling factors**: s in [0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

## Implementation

1. Load pre-saved embeddings X (5000 x d), labels y
2. Stratified split: 4000 train, 1000 test
3. Compute class centroids from train set
4. Compute P_B = top K-1 right singular vectors of (centroids - grand_mean)
5. For each s:
   a. Decompose train/test within-class residuals into signal + null
   b. Scale null component: x_c' = x_c_signal + s * x_c_null
   c. Reconstruct: x' = mu_k + x_c' (centroids preserved)
   d. Compute kappa, d_eff, kappa_eff from scaled train embeddings
   e. Compute q_1nn (1-NN accuracy, normalized) from scaled embeddings

## Pre-Registered Hypotheses

**H1 (IMPLEMENTATION CHECK)**: kappa_eff varies by < 1% across all s values.
This is guaranteed by construction; failure indicates a coding bug.

**H2 (kappa SENSITIVITY)**: kappa at s=3.0 < 0.7 * kappa at s=0.5.
Confirms that null-space scaling genuinely changes kappa (not a no-op).

**H3 (d_eff SENSITIVITY)**: d_eff at s=3.0 > 1.5 * d_eff at s=0.5.
Confirms that null-space scaling genuinely changes d_eff.

**H4 (q INVARIANCE -- PRIMARY)**: q (1-NN, normalized) varies by < 5% in absolute
across all s values (i.e., max(q) - min(q) < 0.05).
This is the KEY Nobel-level prediction: kappa_eff is the sufficient statistic for q.

**H5 (SUFFICIENCY COMPARISON)**: Spearman r(|kappa_eff - kappa_eff_base|, |q - q_base|) < 0.10
AND Spearman r(|kappa - kappa_base|, |q - q_base|) > 0.30, per architecture.
This shows q does NOT track kappa alone but does follow kappa_eff (which is constant).

## Evaluation

OVERALL PASS (PRIMARY): H1 + H4 across at least 2 of 3 architectures.
OVERALL PASS (SECONDARY): H2 + H3 also pass (confirming the manipulation worked).

If H4 PASSES: kappa_eff = delta_min / sigma_centroid_dir is the fundamental variable.
"1-NN accuracy is determined solely by the directional SNR of the nearest class boundary."
This converts the NLP surgery FAIL into a mechanistic WIN.

If H4 FAILS: q changes even when kappa_eff is constant, meaning kappa_eff is NOT sufficient.
This would require either higher-moment corrections or a revised theory.

## Script

`src/cti_kappa_eff_identifiability.py` -- to be written AFTER this pre-registration is committed.
