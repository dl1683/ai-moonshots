# Pre-Registration: Expanded Holdout H8+ (n>=60 Cross-Axis Blind Replication)

**Registered:** 2026-02-24 (commit to be made BEFORE any new holdout computation)
**Previous holdout result:** r=0.957, MAE=0.057, n=23 (commit 650f27e)
**Codex recommendation:** Expand to n>=60 with more families and datasets

## Motivation

The initial holdout test (n=23) showed strong prospective generalization (r=0.957). Codex
scored 7/10 Nobel/Turing but required n>=60 with distributional checks across families,
sizes, and dataset domains to be convincing.

## Design: Factorized Family x Size x Dataset

### Holdout Models (8 total, ALL unseen during CTI law training)

| Model | Family | Params | Layers | Status |
|-------|--------|--------|--------|--------|
| roberta-base | encoder | 125M | 12 | Existing holdout |
| distilbert-base-uncased | encoder | 66M | 6 | Existing holdout |
| albert-base-v2 | encoder (shared) | 12M | 12 | **NEW** |
| opt-125m | decoder/OPT | 125M | 12 | Existing holdout |
| pythia-2.8b | decoder/Pythia | 2.8B | 32 | Existing holdout |
| stablelm-3b-4e1t | decoder/StableLM | 3B | 32 | Existing holdout |
| gemma-3-1b | decoder/Gemma | 1B | 26 | Existing holdout |
| bloom-560m | decoder/BLOOM | 560M | 24 | **NEW** |

### Holdout Datasets (8 total)

| Dataset | K | Domain |
|---------|---|--------|
| agnews | 4 | News topics |
| dbpedia | 14 | Wikipedia categories |
| 20newsgroups | 20 | Usenet discussions |
| go_emotions | 28 | Emotion classification |
| banking77 | 77 | Banking intents |
| amazon_massive | 60 | Voice assistant intents |
| yahoo | 10 | Yahoo Answers topics |
| emotion | 6 | Emotion detection |

### Target: 8 models x 8 datasets = 64 predictions

## Frozen Parameters (from 20-model training set, commit 650f27e)

### Family-specific alpha (slope)
- encoder: 5.0163
- decoder: 1.9066
- hybrid: 1.6983
- ssm: 1.4910

### C_d (dataset difficulty intercepts)
- 20newsgroups: -1.706
- agnews: -0.785
- dbpedia: -0.242
- go_emotions: -2.992
- amazon_massive: -0.973
- banking77: -1.199
- emotion: -2.479
- yahoo: -2.166

### C_f (family quality intercepts)
- encoder: -1.130
- decoder: 0.195
- hybrid: 0.083
- ssm: 0.188

### Gamma (model size term, included but expected weak)
- gamma: 0.105
- gamma_intercept: -0.624

## Prediction Formula

For each holdout (model, dataset) pair:
```
logit(q_pred) = alpha_f * kappa_best_layer + C_d + C_f + gamma * log(params_M) + gamma_intercept
q_pred = sigmoid(logit(q_pred))
```

Where kappa_best_layer = max kappa across 4 proportional-depth layers.

## Pre-Registered Success Criteria

### Primary (must pass all):
1. **H8a**: Pearson r(logit_actual, logit_pred) >= 0.85 across all n predictions
2. **H8b**: Mean absolute error (MAE) on q_norm scale <= 0.10
3. **H8c**: At least 80% of predictions within absolute error < 0.15

### Secondary (informative, not gated):
4. **H8d**: Per-family Pearson r >= 0.70 (encoder subset, decoder subset separately)
5. **H8e**: No systematic bias: mean signed error in [-0.05, +0.05]
6. **H8f**: Outlier rate: <= 10% of predictions with error > 0.20

### Exclusion Rules (pre-registered):
- Exclude predictions where q_actual <= 0.01 or q_actual >= 0.99 (floor/ceiling)
- If fewer than 48 valid predictions remain after exclusions, test is UNDERPOWERED (report but don't claim pass/fail)

## Baseline Comparisons (required)

1. **Dataset-mean baseline**: predict q = mean(q_training_models) for that dataset
2. **Family-mean baseline**: predict q = mean(q_training_models_same_family) for that dataset
3. **Random baseline**: permuted predictions (n=1000 permutations)

## Analysis Plan

1. Generate kappa cache for all (holdout_model, dataset) pairs
2. Apply frozen formula to produce predictions
3. Compare predictions to actual q values
4. Report all pre-registered metrics
5. Present to Codex for review
