# Exploratory Ideas

Status: speculative concepts, not part of the validated core claims.
Last updated: February 10, 2026.

This file consolidates prior exploratory notes into one place so the repo stays navigable.

---

## 1) Hierarchical Reasoning in LLMs

Hypothesis:
- If latent states are explicitly multi-scale (coarse to fine), LLM reasoning may improve on planning, analogy, and compositional generalization.

Implementation directions:
1. Fractal token embeddings.
2. Fractal hidden-state projections.
3. Scale-aware attention heads.
4. Hierarchical chain-of-thought templates.

Testable outcomes:
1. Better high-level planning.
2. Better multi-hop reasoning with lower token cost.
3. Clearer abstraction probes across scales.

---

## 2) Hyperbolic Fractal Embeddings

Hypothesis:
- Combining hierarchy-aligned fractal supervision with hyperbolic geometry could better fit tree-like semantic structure.

Core idea:
1. Keep prefix-aligned supervision.
2. Replace Euclidean operations in the head with hyperbolic operations (Poincare model).
3. Use radial structure to encode abstraction depth.

Expected benefit:
- Lower distortion for deep taxonomies and better coarse-to-fine separation in strongly hierarchical datasets.

Key risk:
- Numerical instability and optimization difficulty in hyperbolic training.

---

## 3) Trajectory Embeddings (Meaning as Dynamics)

Hypothesis:
- Representing meaning as a trajectory through a learned semantic field (rather than a single point) may capture ambiguity resolution and compositional flow.

Core idea:
1. Encode text to an initial state.
2. Evolve through learned dynamics (ODE-style).
3. Use checkpoints/final attractor as multi-scale representation.

Expected benefit:
- Better handling of context-driven disambiguation and semantic composition.

Key risk:
- High compute overhead and unclear gains over strong point-embedding baselines.

---

## 4) Prioritization

Recommended order:
1. LLM hierarchical reasoning probes (lowest integration risk).
2. Hyperbolic fractal head prototype (medium risk, strong hierarchy relevance).
3. Trajectory embeddings (highest risk/cost).

---

## 5) Exit Criteria for Promotion to Core Research

An idea should move from exploratory to core only if it has:
1. A reproducible benchmark win over current V5/MRL baselines.
2. A clear causal mechanism test (not just correlation).
3. A concise theory argument tied to `research/THEORY.md`.

