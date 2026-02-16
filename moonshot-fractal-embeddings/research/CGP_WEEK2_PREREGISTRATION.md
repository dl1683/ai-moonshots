# CGP Week 2: Pre-Registered Analysis Plan
## Locked BEFORE seeing full results (Feb 16, 2026)

## Primary Hypotheses

### H1: Monotonic Dose-Response
- lambda_sep increase -> monotonic increase in class separation (L1)
- Test: Jonckheere-Terpstra trend test across lambda_sep levels
- Pass: p < 0.05 for contrastive objective on BOTH datasets

### H2: Quality Improvement
- Pooled effect of lambda_sep > 0 vs lambda_sep = 0 has CI excluding zero
- Test: Mixed-effects model with random seed intercepts
- Pass: 95% CI for lambda_sep coefficient excludes zero

### H3: Class Separation Mediates
- Dose-response Spearman(class_sep, knn_quality) > 0.5 pooled
- Test: Spearman correlation across all non-baseline conditions
- Pass: rho > 0.5, p < 0.01

### H4: Uniformity is NOT the mechanism
- lambda_uni conditions show no improvement (or degradation) vs lambda_uni=0
- Test: Paired comparison at matched lambda_sep
- Pass: No significant improvement from uniformity

## Decision Gates

### GREEN (proceed to Week 3-4: cross-architecture)
- H1 pass AND H2 pass AND H3 pass
- Interpretation: Class separation is a programmable control knob

### YELLOW (repeat with modifications)
- H1 or H2 pass but not both
- Interpretation: Signal exists but noisy; need more seeds or different sweep

### RED (pivot direction)
- H1 fail AND H2 fail
- Interpretation: Class separation is not causally controllable this way

## Week 3 Actions by Outcome

### If GREEN:
1. Replicate on held-out model (bge-base instead of Pythia-160M)
2. Replicate on held-out dataset (at least 2 new datasets)
3. Add linear probe metric alongside kNN (address metric monoculture)
4. Begin theory: WHY does class separation predict quality? Information-theoretic bound?

### If YELLOW:
1. Increase seeds to 5-10
2. Try continuous lambda_sep sweep (not just 4 points)
3. Try different regularizer formulations (triplet-based, prototype-based)

### If RED:
1. Pivot to direct geometry intervention (manipulate spectral properties)
2. Or pivot to training dynamics (geometry trajectory matters, not endpoint)

## Statistical Methods
- Mixed-effects: statsmodels MixedLM or scipy
- Trend test: Jonckheere-Terpstra (scipy.stats or manual)
- Effect sizes: Cohen's d with Hedges' correction
- Multiple testing: Holm-Bonferroni across datasets
- Bootstrap CIs: 10000 resamples
