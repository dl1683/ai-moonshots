# NC-Loss Experiment Pre-Registration
**Date: 2026-02-22 (BEFORE results arrive)**
**Status: LOCKED — do not modify after experiment completes**

## Background

The universal law logit(q) = alpha * kappa_nearest + C_task (alpha=1.549, CV=4.4% across 7
architecture families) predicts that kappa_nearest is a causal driver of normalized 1-NN
classification quality q. Previous causal tests (triplet training, dist_ratio regularizer)
failed due to gradient conflict or insufficient sensitivity.

NC-loss (Neural Collapse loss) provides a theoretically motivated approach:
- L_within: forces within-class tightness (reduces sigma_W)
- L_ETF: forces inter-class means toward ETF geometry (maximizes kappa_nearest for K classes)
- L_margin: soft margin to prevent collapse
Combined: L = L_CE + lambda * (L_within + 0.5*L_ETF + 0.5*L_margin)

The ETF geometry is the UNIQUE arrangement that MAXIMIZES min_centroid_distance for K means
on a unit sphere, directly targeting kappa_nearest.

## Experiment Design

**Dataset**: CIFAR-100 coarse (K=20 superclasses, 50K train, 10K test)
**Architecture**: CIFAR-native ResNet18 (3x3 first conv, no maxpool, 512-dim features)
**Projection head**: 512 -> 256 (L2-normalized, EMA class means, momentum=0.95)
**Lambda schedule**: 0 for epochs 1-40, linear ramp to 0.15 for epochs 41-120, constant 0.15 for 121-200

**Arms**:
- CE: Baseline cross-entropy only (lambda=0)
- NC: CE + NC-loss with true labels
- shuffled_NC: CE + NC-loss with permuted labels (control — ETF on wrong classes)

**Quick pilot**: 2 arms (CE, NC), 3 seeds, 60 epochs, checkpoints [0,25,40,60]
**Full RCT**: 3 arms (CE, NC, shuffled_NC), 5 seeds, 200 epochs, checkpoints [0,40,80,120,160,200]

## Pre-registered Predictions

### Minimum Threshold (Quick Pilot, 60-epoch, sign test)
- **P1**: delta_q = mean_q_NC - mean_q_CE > 0 (SIGN TEST across 3 seeds)
- **P2**: delta_kappa = mean_kappa_NC - mean_kappa_CE > 0

### Main Threshold (Full RCT, 200-epoch)
- **P3**: delta_q >= 0.02 (effect size threshold)
  - Evidence for: NC geometry directly increases 1-NN classification quality
  - Minimum detectable effect = 0.02 normalized accuracy improvement
- **P4**: delta_kappa > 0 (causal mechanism confirmed)
- **P5**: shuffled_NC arm DOES NOT improve q (q_shuffled_NC <= q_CE + 0.01)
  - This confirms ETF structure requires correct class assignments
- **P6**: delta_logit(q) / delta_kappa approximately in range [1.0, 3.0]
  - LOAO alpha = 1.549, but NC-loss training operates differently from population regression
  - Loose test: the q improvement should be consistent with kappa change * law

### Mechanistic Predictions (from theory)
- **P7**: NC-loss should reduce L_within (within-class variance decreases)
- **P8**: NC-loss should reduce L_ETF (class means approach ETF geometry)
- **P9**: kappa_nearest increases monotonically with lambda (warming up)

## Falsifying Predictions

The following outcomes would FALSIFY the causal chain:

| Outcome | Interpretation |
|---------|----------------|
| delta_q < 0 (NC hurts) | NC geometry HURTS classification |
| delta_q > 0 but delta_kappa < 0 | q improves via different mechanism, not kappa |
| shuffled_NC ALSO improves q | ETF structure alone helps, regardless of correct classes |
| kappa up but q NOT up (delta_q < 0) | Law is correlational only, NOT causal |

## Decision Rules

1. If P1 PASS (sign test): SUPPORT for causal hypothesis, proceed to full RCT
2. If P1 FAIL: NC-loss approach does not work, consider alternative interventions
3. If P3 PASS (delta_q >= 0.02): STRONG causal evidence, write up for paper
4. If P3 FAIL but P1 PASS: Partial support, analyze why effect is small
5. If P5 FAIL (shuffled improves): ETF structure helps regardless of labels, revise theory

## Expected Baseline Values (from epoch 40 checkpoint, seed=0, CE arm)
- q_CE at epoch 40 = 0.4944
- kappa_CE at epoch 40 = 0.4797
- (These will not match final because training continues to epoch 60/200)

## Theoretical Justification

**Why NC-loss should increase kappa_nearest:**
1. L_ETF pushes class means toward equiangular tight frame (ETF) geometry
2. ETF maximizes the minimum pairwise distance for K means on unit sphere
3. kappa_nearest = min_centroid_dist / (sigma_W * sqrt(d))
4. L_within decreases sigma_W simultaneously
5. Both effects increase kappa_nearest

**Why kappa_nearest increase should improve q:**
logit(q) = 1.549 * kappa_nearest + C_task [empirical law]
If kappa increases by delta_kappa, predicted delta_logit(q) = 1.549 * delta_kappa
For kappa going from 0.5 to 0.6 (delta=0.1):
predicted delta_q ≈ 0.1 * 1.549 * q*(1-q) / 1 ≈ 0.1 * 1.549 * 0.25 ≈ 0.039

**Why shuffled_NC should NOT work:**
- Shuffled labels create random ETF structure unrelated to class semantics
- CE loss trains for the correct semantics
- The two objectives conflict, preventing meaningful ETF formation

## Pre-registration Timestamp

This document was written at approximately 2026-02-22 16:15 EST, before any NC-loss
results were observed (experiment started at 15:45, only saw epoch 40 CE baseline: q=0.494).

**First checkpoint results will appear at approximately:**
- Quick pilot: 17:41 EST (epoch 60 for 3 seeds)
- Full RCT: tomorrow morning (epoch 200 for 5 seeds)
