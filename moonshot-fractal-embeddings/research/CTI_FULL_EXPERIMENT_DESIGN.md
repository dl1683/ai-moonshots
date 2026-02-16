# CTI Full Experiment Design (Draft)

## Status: DRAFT - Pending pilot results and Codex review

## Critical Design Decision: Fast Metric

The pilot uses V5-head steerability (~8 min/point). At 5,400 points = 30 days.
**Solution**: Use kNN accuracy on hierarchical labels as primary metric (~10 sec/point).
- kNN is architecture-agnostic (stronger universality claim)
- kNN has no training randomness (only data sampling)
- kNN directly measures representation quality at each layer
- Validate correlation between kNN and V5-steerability on pilot subset

### Metrics per evaluation point
- `Q_L0(l)`: kNN accuracy for L0 (coarse) labels using layer `l` representations
- `Q_L1(l)`: kNN accuracy for L1 (fine) labels using layer `l` representations
- `D_L1(l)`: Distortion = `1 - Q_L1(l) / Q_L1(best_layer)`
- `HSR(l)`: Hierarchical Separation Ratio = `Q_L0(l) / Q_L1(l)` (how much easier coarse vs fine)

### Primary metric for power law fitting
`D_L1(C)` where `C = layer / total_layers`

---

## Model Selection (15 total)

### Family 1: Encoder Transformers (BGE/E5)
| Role | Model | Params | Layers | Hidden Dim |
|------|-------|--------|--------|------------|
| Fit | bge-small-en-v1.5 | 33M | 12 | 384 |
| Fit | bge-base-en-v1.5 | 109M | 12 | 768 |
| Fit | bge-large-en-v1.5 | 335M | 24 | 1024 |

### Family 2: Decoder Transformers (Pythia)
| Role | Model | Params | Layers | Hidden Dim |
|------|-------|--------|--------|------------|
| Fit | Pythia-160M | 160M | 12 | 768 |
| Fit | Pythia-410M | 410M | 24 | 1024 |
| Fit | Pythia-2.8B | 2.8B | 32 | 2560 |

### Family 3: SSM (Mamba)
| Role | Model | Params | Layers | Hidden Dim |
|------|-------|--------|--------|------------|
| Fit | Mamba-130M | 130M | 24 | 768 |
| Fit | Mamba-370M | 370M | 48 | 1024 |
| Fit | Mamba-1.4B | 1.4B | 48 | 2048 |

### Family 4: Hybrid (Falcon-H1)
| Role | Model | Params | Layers | Hidden Dim |
|------|-------|--------|--------|------------|
| Fit | Falcon-H1-0.5B | 500M | 24 | 1024 |
| Fit | Falcon-H1-1.5B | 1.5B | 24 | 1536 |
| Fit | Falcon-H1-3B | 3B | 32 | 2048 |

### Holdout Models (for prediction)
| Role | Model | Params | Layers | Hidden Dim |
|------|-------|--------|--------|------------|
| Holdout | e5-base-v2 | 109M | 12 | 768 |
| Holdout | Qwen3-0.6B | 600M | 28 | 1024 |
| Holdout | Granite-4.0-Micro-H | 350M | varies | varies |

---

## Datasets (10 total)

### Fit Datasets (8)
1. CLINC (10 L0 -> 150 L1) - intent classification
2. DBPedia_Classes (9 L0 -> 70 L1) - Wikipedia topics
3. TREC (6 L0 -> 50 L1) - question types
4. Yahoo (10 L0 -> 10 L1) - Q&A topics
5. 20Newsgroups (6 L0 -> 20 L1) - news
6. GoEmotions (8 L0 -> 28 L1) - emotions
7. ArXiv (6 L0 -> varies L1) - paper topics
8. HWV-L0L2 (varies) - hierarchical word vectors

### Holdout Datasets (2)
9. WOS (7 L0 -> varies L1) - web of science
10. AGNews (4 L0 -> 4 L1) - news classification

---

## Evaluation Protocol

### Per model:
1. Load model, extract ALL hidden layer representations for each dataset
2. For each layer l = 1, ..., L:
   - Extract hidden states for all samples
   - Mean-pool over tokens (for decoders, use last token)
   - L2-normalize
   - kNN (k=5) accuracy for L0 and L1 labels
3. Compute D_L1(l) = 1 - Q_L1(l) / max(Q_L1)
4. Compute C(l) = l / L (relative compute)

### Curve fitting:
- Power law: D(C) = D_inf + k * C^(-alpha)
- Exponential: D(C) = D_inf + k * exp(-beta * C)
- Log: D(C) = D_inf + k / log(1 + gamma * C)

### Critical consideration: Mid-layer peak
"Layer by Layer" (ICML 2025) showed quality often peaks mid-network.
If quality is non-monotonic:
- **Option A**: Only fit layers 1 to L* (up to peak) - scaling of improvement phase
- **Option B**: Two-phase model with transition at C*
- **Option C**: Modified power law with peak: D(C) = D_inf + k*C^(-a) + g*C^b
- Determine which option best describes the data empirically

---

## Compute Budget

### kNN evaluation (primary):
- Points: 15 models x ~20 layers avg x 10 datasets = 3,000 evaluations
- Time per eval: ~10s (extract reps) + ~5s (kNN) = 15s
- Total: 3,000 x 15s = 12.5 hours

### V5 steerability (validation subset):
- Points: 3 models x 6 layers x 3 datasets x 3 seeds = 162 evaluations
- Time per eval: ~8 min
- Total: 162 x 8 min = 21.6 hours

### Grand total: ~34 hours (1.5 days)

---

## Pre-registered Predictions (before running holdout models)

After fitting on 12 fit models x 8 fit datasets:
1. **Global alpha interval**: 95% CI for pooled alpha
2. **Holdout curve prediction**: For each holdout model, predict D at each layer
3. **Rank ordering**: At C=0.5, predict rank ordering of models by D_L1

---

## Success Criteria (from CTI_SUCCESS_CRITERIA.md)

- Global alpha 95% CI half-width <= 0.03
- Holdout median MAPE <= 5%
- Bayes factor for power law vs best rival >= 30
- Compute controller: >= 25% FLOPs reduction at same quality (<= 0.5pt drop)

---

## Implementation Plan

### Phase 1: Fast kNN sweep (12 hours)
1. Create `cti_knn_sweep.py` - extract layer representations + kNN eval
2. Run all 12 fit models on all 8 fit datasets
3. Fit power law curves, check monotonicity

### Phase 2: Analysis and prediction (2 hours)
1. Fit global and per-model power laws
2. Generate pre-registered predictions for holdout
3. Save predictions before running holdout

### Phase 3: Holdout validation (4 hours)
1. Run 3 holdout models on all 10 datasets
2. Compare predictions to actuals
3. Compute MAPE, Bayes factors

### Phase 4: V5 steerability cross-validation (22 hours)
1. Train V5 heads on subset to validate kNN-steerability correlation
2. Confirm that kNN D(C) power law implies steerability D(C) power law
