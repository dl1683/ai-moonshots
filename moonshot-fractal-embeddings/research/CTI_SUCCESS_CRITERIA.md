# CTI (Compute Thermodynamics of Intelligence) — Success Criteria

## Designed by Codex (GPT-5.3, xhigh reasoning), Feb 15, 2026

### Definitions

- **C**: cumulative FLOPs to produce representation at layer `l`
- **D**: normalized distortion: `D = (S_full - S_l) / (S_full - S_chance + 1e-6)`
- **Fit**: `D(C) = D_inf + k * C^(-alpha)`

---

## Pilot Pass/Fail Criteria (3 models x 6 layers)

| Criterion | Pass threshold | Falsification threshold |
|---|---|---|
| Monotonicity | Spearman(C,D) <= -0.9 on >= 80% curves | > 30% curves violate monotonicity |
| Fit quality | Median adjusted R^2 >= 0.93, 25th pct >= 0.85 | Median R^2 < 0.85 |
| Exponent stability | Pairwise model abs(alpha_i-alpha_j) <= 0.10; pooled 95% CI width <= 0.10 | Any pair > 0.20 or pooled CI includes 0 |
| Predictive power | Fit first 4 layers, predict last 2: median MAPE <= 7% and abs error <= 0.02 D | MAPE > 12% |
| Rival laws | Power law beats exponential/log law by dAIC >= 6 on >= 70% curves | Rival wins by dAIC >= 10 on > 50% curves |
| Parameter sanity | 0 <= D_inf <= 0.25, 0.15 <= alpha <= 1.2, k > 0 | Frequent nonphysical params |

### Decision Rule

- **Green**: pass >= 5/6 and no falsification trigger → expand to full experiment
- **Yellow**: pass 4/6 → expand pilot before claims
- **Red**: pass <= 3/6 or any hard falsification trigger → pivot to phase transitions

---

## Full Paper Experiment Set (if pilot is Green)

- **Models**: 15 total (4 families x 3 sizes = 12 fit + 3 holdout)
- **Compute points per model**: 12
- **Tasks**: 10 (mixed retrieval/classification/reasoning)
- **Seeds**: 3
- **Total evaluated points**: 5,400
- **Pre-registered predictions before full sweep**:
  - Global alpha interval
  - Holdout-curve prediction error bound
  - Rank ordering at fixed compute budget

### Minimum Publishable Outcomes

- Global alpha 95% CI half-width <= 0.03
- Holdout median MAPE <= 5%
- Bayes factor for power law vs best rival >= 30
- Compute controller yields >= 25% FLOPs reduction at same task score (<= 0.5pt drop)

---

## Fallback: Phase Transition Theory

If CTI fails (Red), pivot to renormalization/phase transitions:
- Piecewise power law beats single power law with dBIC > 10 on >= 75% curves
- At least 2 stable universality classes with within-class exponent std < 0.08
- Critical compute C* predicted from low-compute region, recovered within +/-10%
- One intervention shifts C* in predicted direction within +/-15%

---

## 9/10 Nobel-Track Paper

- Core claim: falsifiable geometric-thermodynamic law predicts intelligence distortion from compute across architectures
- Scope: 40+ models, 4 modalities/domains, 3 independent labs
- Unified exponent: alpha = 0.47 +/- 0.02
- 95% held-out curves predicted within +/-4% distortion
- Controller cuts energy/FLOPs by 30-40% at equal quality
- Pre-registered predictions, documented failures, public falsification protocol
