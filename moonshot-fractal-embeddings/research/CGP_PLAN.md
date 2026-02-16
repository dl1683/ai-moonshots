# Causal Geometry Programming (CGP)
## Date: 2026-02-15
## Source: Codex strategic design (xhigh reasoning)

## The Vision

Move from "objective affects geometry" (our finding) to
"we can PROGRAM geometry and thereby program intelligence."

If validated, this introduces a new control paradigm:
**intelligence as a programmable geometric state**, with scale
as a secondary resource, not the primary law.

## Evidence So Far

1. Atlas: 88 depth-quality curves showing non-monotonicity, piecewise sigmoid
2. Geometry-quality duality: sign reverses between encoder/decoder
3. Crossover: Embedding-Gemma (decoder arch, embedding obj) shows encoder-like correlations
4. Key insight: Training objective determines geometry-quality relationship

## 3-Month Plan (Single RTX 5090)

### Weeks 1-4: Find Causal Geometric State Variables
- Controlled fine-tunes: 3 model families x 4 objectives at matched FLOPs
- Build low-dimensional geometry state z (spectral slope, anisotropy, curvature, trajectory features)
- Fit structural model: Objective -> z -> Quality
- Test invariance across architectures
- GATE: z predicts quality out-of-family better than size/arch baselines

### Weeks 5-8: Intervene on Geometry
- Add geometry-targeting regularizers/schedules
- Force specific z trajectories, keep architecture/data fixed
- Intervention tests: same model+data+FLOPs, different geometry trajectories
- Quantify mediated effect size
- GATE: intervention moves quality in predicted direction in most settings

### Weeks 9-12: Build the Objective Compiler
- Controller: task profile -> objective mixture/schedule -> target geometry
- Fixed-compute wins on held-out tasks
- HEADLINE: smaller compiled model matches/beats larger standard-objective baseline
- Deliverable: preprint + open-source toolkit

## Day-90 Success Bar (Non-negotiable)
1. Causal evidence geometry mediates objective -> quality (not just correlation)
2. One reproducible "geometry beats scale" result at fixed compute
3. Reusable compiler that outputs objective schedules from geometry targets

## Week 1 Concrete Tasks
1. Build multi-objective LoRA fine-tuning pipeline for Pythia-160M
2. Objectives: contrastive, MLM, classification, original LM
3. Match FLOPs across objectives
4. Measure per-layer geometry (reuse existing pipeline)
5. Measure downstream quality (reuse kNN pipeline)
6. Build structural model: Objective -> z -> Quality
