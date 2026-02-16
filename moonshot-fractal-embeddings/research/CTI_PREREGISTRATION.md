# Pre-Registered Hypotheses for CTI Holdout Validation

## Date: 2026-02-15
## Status: LOCKED before any holdout analysis

## Exploratory Phase Summary
- 11 fit models x 8 fit datasets = 88 curves analyzed
- Power law wins only 9.1% (AIC), piecewise sigmoid wins 67%
- Non-monotonicity increases with model size within families
- Final layer is optimal in only 55.7% of curves
- Cross-dataset invariance mean rho = 0.655

## Pre-Registered Hypotheses

### H1: Non-Monotonicity Scaling Law
**Claim**: Within architecture families, larger models exhibit stronger non-monotonicity in their depth-quality curves.

**Operationalization**: For each holdout model, compute Spearman correlation (rho) between layer index and kNN L1 accuracy across all test datasets. Compare against same-family fit models.

**Prediction**:
- gte-large (encoder, large) will have rho < 0.7 (matching pattern: larger = more non-monotonic)
- qwen3-0.6b (decoder-to-encoder) will have rho < 0.8
- bge-m3 (trilingual, large) will have rho < 0.75

**Success criterion**: At least 2/3 holdout model predictions correct within +/- 0.1 of predicted rho.

### H2: Final Layer Suboptimality
**Claim**: The final layer is NOT the optimal extraction layer for a substantial fraction of model-dataset pairs.

**Operationalization**: For each holdout model x dataset, identify the layer with highest kNN L1 accuracy. Check if it's the final layer.

**Prediction**:
- Final layer is optimal in <= 65% of holdout curves
- At least 2/4 holdout models will have final-layer-optimal rate < 60%

**Success criterion**: Final-layer-optimal rate on holdout <= 65%.

### H3: Piecewise Sigmoid Best Out-of-Sample
**Claim**: Piecewise sigmoid provides the best out-of-sample predictive accuracy for depth-quality curves.

**Operationalization**: For each curve, fit all 5 forms on even-indexed layers, predict odd-indexed layers. Compare RMSE.

**Prediction**:
- Piecewise sigmoid will have lowest mean RMSE on held-out layers
- Will beat power law by at least 30% RMSE reduction

**Success criterion**: Piecewise sigmoid wins out-of-sample in >= 40% of curves (accounting for parsimony preference, may lose to simpler forms on sparse curves).

## Holdout Split (from ATLAS_SPLIT_LOCK.md)

### Holdout Models (DO NOT ANALYZE UNTIL PREDICTIONS FILED)
- bge-m3 (trilingual encoder)
- nomic-v2 (MoE encoder) -- note: may need special handling for output_hidden_states
- qwen3-0.6b (decoder-to-encoder)
- gte-large (large encoder)

### Holdout Datasets
- agnews, amazon, dbpedia

## Additional Pre-Registered Analyses

### A1: Cross-Dataset Invariance on Holdout
- Predict: holdout datasets will show similar invariance (rho > 0.5) to fit datasets
- If invariance holds on unseen datasets, the "model fingerprint" claim is strengthened

### A2: Peak Layer Prediction
- From fit data, extract average relative peak position per model family
- Predict peak layer for holdout models: peak_layer = round(relative_peak * n_layers)
- Success: prediction within +/- 1 layer for 12-layer models, +/- 2 for 24-layer models

## Commitment
These predictions are filed BEFORE any holdout data is examined.
The holdout sweep will be run only after this document is committed to git.
