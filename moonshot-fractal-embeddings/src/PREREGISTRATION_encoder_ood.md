# Pre-Registration: Encoder OOD Blind Test

**Registered:** 2026-02-23 (before any data collection on RoBERTa-base)

## Test Description

Extend the encoder universality finding to a new encoder architecture (RoBERTa-base)
on a new dataset (banking77), using alpha estimated from the 3 existing encoder calibrations.

## Architecture Under Test (NEW, not in calibration)
- `FacebookAI/roberta-base` (12 layers, 768-dim, RoBERTa architecture)
- NOT in the 3-arch encoder calibration set {bert-base-uncased, deberta-base, bge-base-v1.5}

## Dataset Under Test (NEW for encoder models)
- banking77 (K=77) -- was only in decoder OOD tests, never in encoder calibration

## Fixed Parameters (from 3-arch encoder calibration on dbpedia/agnews/20newsgroups)
- alpha_encoder: estimated from LOAO over 3 encoders = ~7.0-7.1 (will be computed fresh)
- beta: -0.267 (encoder fit)
- C_d_banking77: estimated from decoder partial caches (6 architectures)

## Success Criteria (PRE-REGISTERED)

**Primary:**
1. Independently-fitted alpha for RoBERTa-base ∈ [5.0, 10.0] (encoder interval)
   - Based on: BERT=7.14, DeBERTa=7.05, BGE=9.0 (mean=7.73, ±2σ=[5.6, 9.9])
   - Threshold [5.0, 10.0] gives conservative coverage
2. Within-RoBERTa r(kappa_nearest, logit_q) on banking77 ≥ 0.90
   (tests functional form, not absolute level)

**Secondary (exploratory):**
- Blind prediction using (alpha_encoder, beta_encoder, C_d_banking77_decoder) yields r ≥ 0.80

## Experimental Design

```python
# Step 1: Compute encoder LOAO parameters
# Fit on 3 encoder caches (bert, deberta, bge) x 3 datasets x ~4 layers
# Get alpha_enc, beta_enc

# Step 2: Generate RoBERTa-base cache
# Mean-pool at proportional layers [3, 6, 9, 12] on banking77 K=77
# Compute kappa_nearest, q_norm at each layer

# Step 3: Check alpha
# Fit alpha independently on RoBERTa-base's 4 points
# alpha_roberta ∈ [5.0, 10.0] ?

# Step 4: Check functional form
# r(kappa_nearest_by_layer, logit_q_by_layer) for RoBERTa on banking77 ≥ 0.90?

# Step 5: Blind prediction
# C_d from decoder partial caches, alpha_enc from encoder LOAO
# r(pred, obs) for 4 RoBERTa-banking77 points ≥ 0.80?
```

## Commit Hash
This pre-registration will be committed to git BEFORE running any experiments.
"""
