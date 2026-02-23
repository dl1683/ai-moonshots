# Pre-Registration: NLP Linear-Regime Global Surgery Test

**Registered:** 2026-02-23 (after observing cross-domain surgery FAIL at commit 8ea7288)

## Context / Motivation

Cross-domain surgery (commit 8ea7288) on frozen pre-trained NLP embeddings FAILED H1:
ratio ≈ 2 instead of d_eff ≈ 33-43.

**Diagnosed root cause**: kappa_eff = 0.24-0.73 << 1.0.
The CIFAR surgery (commit 59faa5d) had kappa_eff ≈ 1.0 (linear regime), where the CTI
formula logit(q) = A*kappa*sqrt(d_eff) + C is valid.

**Hypothesis**: The 1/d_eff attenuation ratio ≈ d_eff holds ONLY when kappa_eff ≈ 1.0 (linear regime).
If NLP embeddings with kappa_eff ≈ 1.0 are used, the ratio should recover ≈ d_eff.

## Regime Selection Protocol (Pre-Registered)

From the beyond-1NN test results, pythia-1b on 20newsgroups Layer 16 has kappa=0.2694.
If d_eff ≈ 14 (the value where kappa*sqrt(d_eff) = 1.0), this is near the linear regime.

**Layer selection rule (pre-registered)**:
- Extract embeddings at ALL sampled layers for pythia-160m, pythia-410m, pythia-1b
  on 20newsgroups (same layers as beyond-1NN test: [3,6,9,12] for 160m, etc.)
- For each model, select the layer where |kappa_eff - 1.0| is minimized
  where kappa_eff = kappa_nearest * sqrt(d_eff_formula) from measured geometry
- If no layer has kappa_eff ∈ [0.7, 1.5], mark that architecture as "not in valid range"
- Use only architectures with a valid linear-regime layer

**Models (in order of preference, by expected kappa_eff proximity to 1.0)**:
1. pythia-1b: Layer 16 expected (from beyond-1NN log kappa=0.2694)
2. OLMo-1B: Layer 16 (kappa=0.2489)
3. pythia-410m: Layer 12 (kappa=0.1490) — likely sub-linear still
4. pythia-160m: Layer 12 (kappa=0.0876) — likely sub-linear

## Surgery Protocol

Same as PREREGISTRATION_crossdomain_surgery.md:
- ARM 1 (Single): Scale Delta_hat direction by 1/sqrt(r)
- ARM 2 (Global): Scale all K-1 signal-subspace directions by 1/sqrt(r)
- Surgery levels: r in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
- A_RENORM_K20 = 1.0535 (K=20, 20newsgroups)
- N=5000 samples, 80/20 stratified split

## Pre-Registered Success Criteria

Same H1/H2/H3 as CIFAR surgery:

**H1 (PRIMARY):** median ratio ∈ [d_eff/3, 3*d_eff] for valid (arch, r) pairs
**H2 (SECONDARY):** direction correct ≥ 3/4 valid pairs
**H3 (TERTIARY):** kappa invariance < 0.5%

**SEPARATE (PRE-REGISTERED) comparison:**
- Cross-regime comparison: report median ratio for sub-linear regime (commit 8ea7288)
  vs linear-regime (this test) architectures separately
- Pre-registered hypothesis: linear-regime ratio will be closer to d_eff than sub-linear

## What This Proves (If H1 Passes)

The 1/d_eff attenuation is REGIME-SPECIFIC but DOMAIN-AGNOSTIC:
- It holds for ANY embedding (vision OR NLP) when kappa_eff ≈ 1.0
- It fails when kappa_eff << 1.0 (sub-linear regime)
This is a clean mechanistic prediction: the CTI formula is the right predictor, and the
mechanism requires being near the linear-regime operating point.

## Script

`src/cti_nlp_linear_regime_surgery.py` — to be written AFTER this pre-registration is committed.
