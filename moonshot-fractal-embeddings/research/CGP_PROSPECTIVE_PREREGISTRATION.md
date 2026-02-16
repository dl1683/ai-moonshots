# CGP Prospective Causal-Sufficiency Challenge
## Pre-Registration (Locked Before Seeing Week 3 Full Results)

## Date: Feb 16, 2026

## Overview

We predict that the composite geometric invariant G = kappa * C * d * Q / (C-1)
is causally sufficient for predicting kNN classification accuracy. Specifically:

**The same G value should produce the same kNN accuracy regardless of how G
was achieved (which model, which lambda_sep, which dataset).**

This is the strongest test of our universality claim. If G is truly a universal
predictor, then we can:
1. Fit a curve f(G) -> kNN_accuracy from EXISTING data
2. Pre-register predictions for NOVEL conditions
3. Verify those predictions on held-out conditions

## Phase 1: Fit the Universal Curve

Using ALL data from Week 2 (Pythia-160M) and Week 3 (bge-small, e5-small, MiniLM),
fit the relationship:

  kNN_l1 = a * (1 - exp(-b * G)) + c

where a, b, c are free parameters. This saturating exponential is motivated by
Theorem 1: P_NC(error) <= (C-1) * exp(-G/4), so quality should approach 1.0
as G -> infinity.

Alternative functional forms to test:
- Linear: kNN_l1 = a * G + b
- Log: kNN_l1 = a * log(G + 1) + b
- Power: kNN_l1 = a * G^b + c

Select the form with best held-out R2 using leave-one-model-out cross-validation.

## Phase 2: Pre-Registered Predictions

### Held-out Architecture: bge-base-en-v1.5
- This is a LARGER model (110M params, 768-dim) not used in any prior experiment
- Train with lambda_sep = [0.1, 0.3, 1.0], seeds [42, 123, 456]
- Evaluate on: clinc, dbpedia_classes, agnews, trec

### Held-out Dataset: banking77
- 77 intents, 13 intent categories
- NOT used in any prior CGP experiment
- Evaluate with all Week 3 models at all lambda_sep values

### Pre-Registered Predictions
For each novel condition, we:
1. Compute G from the embeddings
2. Predict kNN_l1 = f(G) using the fitted curve
3. Measure actual kNN_l1

### Success Criteria (Locked)

**Primary:**
- Prediction RMSE < 0.10 on held-out conditions (generous for first test)
- Prediction R2 > 0.3 (G explains >30% of variance in novel conditions)

**Secondary (Nobel-track):**
- Adding model indicator to f(G) does NOT significantly improve R2 (p > 0.05)
  (This tests FULL MEDIATION: G absorbs all model-specific effects)
- Residual variance is consistent with seed-level noise (Levene's test)

**Tertiary (dream):**
- Same G achieved via different paths gives kNN_l1 within seed-level SD
  Test: for pairs of conditions with |G_i - G_j| < epsilon, test |kNN_i - kNN_j|

## Phase 3: Analysis Protocol

1. Fit f(G) on training data (Week 2 + Week 3, excluding held-out)
2. Compute G for each held-out condition
3. Predict kNN_l1 for each held-out condition
4. Compute RMSE, R2, and mediation statistics
5. Report all results regardless of outcome

## Potential Failure Modes

1. G is model-specific: different architectures need different curves
   - Would require adding model-specific parameters (kills universality)
2. G is dataset-specific: different datasets need different curves
   - Would require dataset-specific normalization
3. kappa dominates: G is mostly driven by centroid regularity, not Fisher Q
   - Would suggest the composite invariant is overcomplicated
4. Training damage swamps signal: LoRA fine-tuning destroys pretrained
   representations so badly that geometry is no longer predictive
   - Would need a different intervention mechanism

## Prior Art Differentiation

This experiment is designed to be NOVEL relative to:
- He & Su (PNAS 2023): described class separation process, no prospective prediction
- NeurIPS 2024 geometric complexity: causal but no prospective prediction, different metric
- Scaling Collapse (ICML 2025): universal curves for loss, not representation quality
- Layer by Layer (ICML 2025): measurement framework, no causal prediction

CGP is the first to pre-register and prospectively test a geometric invariant
for predicting representation quality across architectures.
