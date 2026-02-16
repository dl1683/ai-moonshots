# Literature Synthesis: Scaling Laws Gap (Feb 15, 2026)

## THE GAP WE FILL

| Axis | Has a law? | Who? |
|------|-----------|------|
| Width scaling | YES (L~1/m) | Superposition (NeurIPS 2025 Best Paper Runner-up) |
| Training compute | YES (Kaplan/Chinchilla) | Scaling Collapse (ICML 2025 Oral) |
| **Depth-representation quality** | **NO** | **US (proposed)** |

## Key Prior Work

### "Layer by Layer" (ICML 2025 Oral, LeCun co-author)
- Intermediate layers beat final by up to 16%
- Autoregressive: mid-layer entropy valley at 40-60% depth
- Bidirectional (BERT-style): flatter curves
- **They describe the phenomenon. We quantify and predict it.**

### Renormalizable Spectral-Shell Dynamics (Dec 2025)
- Derives scaling laws from RG in function space
- Unifies NTK and feature learning
- **THREAT**: Covers "RG explains scaling" territory
- **BUT**: Works on training time, not layer depth
- **Our positioning**: SPATIAL complement to their TEMPORAL work

### 1/sqrt(depth) collapse (Dec 2025)
- Feature learning collapses in first internal layer at large depth
- This is WHY quality is non-monotonic
- Depth-aware LR correction can fix it
- **Relevant**: Different architectures have different collapse patterns

## Strategic Positioning

**Do NOT compete with spectral-shell dynamics on scaling law derivation.**
Instead: CTI = spatial (layer-wise) complement to temporal (training-time) scaling.

The spectral-shell paper explains how loss decreases during training.
We explain how representation quality evolves through network depth.
These are orthogonal axes of the same phenomenon.

## Novel Claims Available to Us

1. Architecture-specific functional forms for D(depth)
2. Cross-dataset invariance of quality curve shape (model-intrinsic fingerprint)
3. Two universality classes: encoder (non-monotonic) vs decoder (monotonic)
4. Predicting optimal extraction layer from architecture parameters
5. Practical compute savings from optimal layer selection

## Falsification Risks

- If curve shapes are NOT reproducible across datasets → claim 2 fails
- If architecture doesn't predict universality class → claim 3 fails
- If optimal layer can't be predicted better than "use final" → claim 5 fails
- If spectral-shell dynamics gets extended to layer-wise → we lose novelty
