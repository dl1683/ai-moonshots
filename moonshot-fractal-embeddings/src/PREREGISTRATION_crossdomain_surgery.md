# Pre-Registration: Cross-Domain Global Surgery Ratio Test (NLP)

**Registered:** 2026-02-23 (before any data collection for this test)

## Motivation

The CIFAR-100 global surgery test (commit 59faa5d) confirmed the 1/d_eff attenuation
hypothesis in the vision domain (K=20, ResNet50, ratio/d_eff ∈ [0.91,1.15] at r=1.5).

This test asks: does the same mechanism hold in the NLP domain?
If YES: the 1/d_eff attenuation is domain-agnostic — it is a property of embedding
geometry (the ratio of signal-subspace variance to total within-class variance), not
a vision-specific artifact.

## Hypothesis

**1/d_eff Attenuation Mechanism (cross-domain):**
- Single-direction surgery changes sigma_centroid_dir in ONE dimension
- Global surgery (all K-1 signal directions) produces the FULL predicted effect
- Ratio: delta_logit_global / delta_logit_single ≈ d_eff_base (independent of domain)

## Pre-Saved Embeddings

Using pre-existing embeddings from results/causal_v2_embs_*.npz
(last-layer representations, eval mode, no random augmentation):

| Architecture      | d    | File                              |
|-------------------|------|-----------------------------------|
| deberta-base      | 768  | causal_v2_embs_deberta-base_20newsgroups.npz  |
| olmo-1b           | 2048 | causal_v2_embs_olmo-1b_20newsgroups.npz       |
| qwen3-0.6b        | 1024 | causal_v2_embs_qwen3-0.6b_20newsgroups.npz    |

- Dataset: 20newsgroups, K=20 (same K as CIFAR surgery)
- N=5000 per model (same order as CIFAR surgery, 4000 train / 1000 test stratified)
- Same K=20, so A_RENORM_K20 = 1.0535 applies

## Surgery Protocol

**ARM 1 (Single):** Same as cti_global_vs_single_surgery.py
- Scale nearest centroid pair direction Delta_hat by 1/sqrt(r)
- Preserve tr(Sigma_W) via scale_perp

**ARM 2 (Global):** Same as cti_global_vs_single_surgery.py
- Scale ALL K-1 signal-subspace dimensions by 1/sqrt(r)
- Preserve tr(Sigma_W) via scale_null
- Valid range: r >= r_min_valid = tr(W_sig)/tr(W)

**Surgery levels:** r in [0.5, 0.7, 1.0 (baseline), 1.5, 2.0, 3.0]
**Architectures:** deberta-base, olmo-1b, qwen3-0.6b (3 "seeds" across NLP architectures)
**Train/test split:** 80/20 stratified (same random_state=seed_idx)

## Pre-Registered Success Criteria

**PRIMARY (H1):** ratio(r) = delta_logit_global(r) / delta_logit_single(r)
- H1-PASS: median ratio across valid (r, arch) pairs in [d_eff_base/3, d_eff_base*3]
- d_eff_base = measured from loaded embeddings per architecture
- Exclude invalid r values where global surgery has scale_null^2 < 0

**SECONDARY (H2):** Global surgery direction
- H2-PASS: delta_logit_global > 0 for r > 1 AND < 0 for r < 1 for >=3/4 valid pairs

**TERTIARY (H3):** Kappa invariance for both surgeries (valid records only)
- H3-PASS: |kappa_new - kappa_base| / kappa_base < 0.5%

## What This Proves (If Passes)

The 1/d_eff attenuation is domain-agnostic: it holds for NLP (text LM) embeddings just
as for vision (CIFAR-100 ResNet) embeddings. Combined with the CIFAR result, this
establishes the mechanism as a universal property of the CTI law geometry, not a
vision-specific artifact.

## Preregistered Reference to CIFAR Result

At r=1.5, CIFAR surgery (commit 59faa5d):
- ratio/d_eff ∈ [0.91, 1.15] across 3 seeds
- Median ratio = 18.74, d_eff = 24.52

NLP surgery should show: median ratio ∈ [d_eff_base/3, 3*d_eff_base]

## Script

`src/cti_crossdomain_surgery.py` — to be written AFTER this pre-registration is committed.
