# CTI Pivot: Architecture-Specific Scaling Laws

## Date: 2026-02-15
## Designed by: Codex (GPT-5.3, xhigh reasoning) based on empirical data

## Key Empirical Finding
Universal power law D(C) = D_inf + k*C^(-alpha) is FALSIFIED.
Instead: architecture-specific functional forms with cross-dataset invariance.

## New Direction: Option D (Codex recommendation)

**Title**: "Architecture-Specific Scaling Laws for Representation Quality"

**Core claims**:
1. Power law is falsified as universal law
2. Representation quality vs depth follows architecture-specific scaling laws
3. Curve shape is a model-intrinsic invariant across datasets
4. Encoders show regime changes (phase-like), decoders monotonic saturation

## Priority Experiment Stack

### 1. Large Atlas (IN PROGRESS)
- 14 fit models x 8 fit datasets = 112 sweeps
- 4 holdout models x 3 holdout datasets for validation
- Fit: power law, exponential, sigmoid, linear, piecewise-sigmoid
- Use AIC/BIC + out-of-sample error

### 2. Universality Class Discovery
- Cluster normalized curves
- Test if architecture/pretraining objective predicts class
- Hold out model families for validation
- Target: >80% class prediction on unseen families

### 3. Cross-Metric Invariance
- Extend beyond kNN: linear probe, retrieval, STS
- If shape survives metric change, claim is stronger

### 4. Mechanistic Phase Evidence
- Per-layer: CKA, anisotropy, effective rank, intrinsic dim
- Look for abrupt changes at dip/peak transitions
- Causal: skip/reweight critical layers, test curve shifts

### 5. Practical Layer Selector
- Build selector from partial curves or calibration
- Benchmark vs final-layer default
- Target: +2-4 points or compute savings

## Publishability Targets (NeurIPS/ICML)
- Power law wins near-zero across large benchmark
- Architecture-specific forms win out-of-sample
- Class prediction >80% on unseen families
- Peak-layer prediction error ~1 layer for 12-layer models
- Practical selector beats final-layer by 2-4 points

## Connection to Manifesto
"Intelligence = Geometry" means the SHAPE of the quality curve reveals structural properties.
Different architectures have different geometric signatures in how they process information.
Practical payoff: skip unnecessary compute by knowing optimal extraction layer.
