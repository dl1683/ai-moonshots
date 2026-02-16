# CTI Research Landscape (Feb 2026)

## Key Finding: CTI is NOVEL but window is narrowing

Nobody has published D(C) = D_inf + k*C^(-alpha) for layer-wise representation quality.
The specific combination of (layer-wise compute, representation distortion, cross-architecture universality) is unclaimed.

## Closest Prior Work

1. **"Scaling Collapse" (ICML 2025 Oral)** — Universal compute curves, BUT for training loss not representations. Reviewers WILL compare us to this.
2. **"Layer by Layer" (ICML 2025 Oral)** — Representation quality across 32 tasks, mid-layer beats final layer. CRITICAL: quality is often NON-MONOTONIC with depth.
3. **"Superposition Yields Robust Neural Scaling" (NeurIPS 2025 Best Paper Runner-up)** — L ~ 1/m from geometric superposition. About width not depth.

## High Threats

- Scaling Collapse/Supercollapse — already claims "universal" compute-quality curves
- Renormalizable Spectral-Shell Dynamics (Dec 2025) — derives scaling exponents from RG methods
- Feature Learning Dynamics (Dec 2025) — 1/sqrt(depth) collapse at large depth

## CRITICAL RISK: Mid-Layer Peak

"Layer by Layer" shows representation quality is often NON-MONOTONIC with depth.
A pure power law assumes monotonic improvement. CTI may need:
- Modified form: D(C) = D_inf + k*C^(-alpha) + g*C^(beta) with optimal C*
- Two-phase model with transition at optimal depth
- Or the steerability metric may behave differently (possible)

## What Makes CTI Novel (5 things)

1. Layer-wise depth as independent variable (not total training FLOPs)
2. Representation quality/distortion, not loss
3. Cross-architecture universality at representation level
4. Thermodynamic interpretation via rate-distortion theory
5. Practical compute controller (predict optimal layer for any budget)

## Strategic Risk

Window is open but narrowing. Groups at Google (Pennington), Harvard (Pehlevan), MIT (Gore) have the theoretical machinery to reach similar conclusions.
