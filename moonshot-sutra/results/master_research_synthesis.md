# Master Research Synthesis for v0.5.4 Design
## 15 Cross-Domain Research Agents + Codex Deep Think + Chrome Probe Results
## Generated 2026-03-20

This document compiles ALL findings from 15 cross-domain research sweeps, one Codex deep analysis, and Chrome v0.5.4 probe results. It is the input for the MASTER Codex session that will design v0.5.4.

### NEW DOMAINS (Batch 2, added per user request)

**3J. DYNAMICAL SYSTEMS + CONTROL THEORY** (agent completed)
Key findings for Sutra:
1. **Contraction analysis**: If each processing module is contracting (spectral norm < 1), composition is guaranteed to converge. Sutra's recurrent loop SHOULD be a contracting system.
2. **Reservoir computing**: A fixed random dynamical system + linear readout achieves universal function approximation. Edge of chaos = maximum computational capacity. Memory-nonlinearity tradeoff: total info capacity = N (reservoir dimension).
3. **Bifurcation as compute control**: Saddle-node creates/destroys memory states. Hopf creates oscillatory modes. Pitchfork implements symmetry-breaking decisions. Parameter mu reconfigures computation qualitatively.
4. **Kuramoto synchronization**: Phase transition at critical coupling K_c. Below: incoherence. Above: partial synchronization. Chimera states: synchronized + incoherent regions coexist.
5. **Delay DDEs**: Time delay creates infinite-dimensional dynamics from finite parameters. Delay reservoir: 1 node + 1 delay = N virtual nodes. Extreme parameter efficiency.
6. **Lyapunov stability**: Prove convergence WITHOUT solving equations. Contraction + Lyapunov = modular stability guarantees that compose.

**3K. CODING THEORY + SIGNAL PROCESSING** (agent completed)
Key findings for Sutra:
1. **BP on LDPC**: Local iterative message passing achieves near-Shannon-limit. Proves sparse, local, iterative computation matches global brute-force. THE mathematical proof Sutra needs.
2. **Compressed sensing**: Natural dimensionality = sparsity s, not ambient dimension n. Width should be O(s log(n/s)). Random projections preserve all sparse structure.
3. **Wavelets/MRA**: Nested subspaces V_0 ⊂ V_1 ⊂ ... are the optimal multiresolution decomposition. O(n) via recursive filterbanks. Directly maps to Sutra's multi-scale processing.
4. **Rate-distortion**: Autoencoders ARE rate-distortion optimizers. beta-VAE traces the R(D) curve. Shape-gain factorization (LayerNorm!) is optimal for Gaussian-like distributions.
5. **Epsilon-machines**: The minimal optimal predictor has C_mu bits of state. Transformers use fixed-width hidden state regardless of input complexity. ADAPTIVE state complexity is optimal.
6. **Spatial coupling**: Suboptimal local processing elevated to OPTIMAL global processing through proper coupling. Boundary conditions seed correct inference that propagates inward. Skip connections ARE spatial coupling.
7. **Expander graphs**: Every subset of variables must have diverse connections. Random sparse attention should work as well as dense attention if expansion property holds.

**3L. CATEGORY THEORY (DEEP)** (agent completed)
Key organizational principles for Sutra:
1. **Yoneda**: Objects ARE their relationships (not internal structure). Test system by its input-output behavior.
2. **Adjunctions**: Optimal translation between structure levels. Compression (left adjoint) and reconstruction (right adjoint) are optimally paired. Lossy compression = unit is not isomorphism.
3. **Monads**: Sequential effectful computation. Kleisli composition = how to chain computations that produce decorated outputs. Probability monad = Markov kernels.
4. **Sheaves**: Local-to-global assembly. Consistent local data CAN be assembled globally. Cohomology measures obstruction to assembly. Sensor networks, distributed systems.
5. **Galois connections**: Abstraction and concretization are adjoint. Closed sets form complete lattices. Abstract interpretation IS a Galois connection.

**3M. THERMODYNAMICS + INFO GEOMETRY (DEEP)** (agent completed)
Key findings:
1. **Free energy principle (full derivation)**: F = KL(q||p_posterior) + (-ln P(data)). Minimizing F = Bayesian inference + surprise minimization. Perception, action, learning ALL from one principle.
2. **Fluctuation theorems**: Dissipated work = k_BT * KL(forward || reverse). Slower training = less entropy = more efficient learning. This is WHY warmup + cosine decay works.
3. **Fisher metric is UNIQUE** (Chentsov's theorem). Natural gradient = steepest descent in distribution space. Adam approximates diagonal Fisher.
4. **Wasserstein geometry**: Global metric on distributions (unlike local Fisher). Displacement interpolation = physically moving mass along optimal paths. Many PDEs are gradient flows in Wasserstein space (JKO scheme).
5. **RG c-theorem**: C decreases monotonically under coarse-graining. Information loss is monotone across scales. Each layer IS an RG step.
6. **Spin glass ultrametricity**: Loss landscape organized as hierarchical tree. TAP equations for mean-field analysis. Glass-to-order transition during training.

**3N. NCA + SOMS + COLLECTIVE (DEEP)** (agent completed)
Key principles with equations:
1. **NCA**: ~8000 params govern morphogenesis. Pool-based training creates attractor for target pattern. Regeneration = expanded basin of attraction. Hidden channels carry positional gradient information.
2. **Lenia**: Continuous CA with Gaussian growth function. PDE: dA/dt = G(K*A). Homeostatic attractor via Goldilocks zone.
3. **SOM**: Competitive learning + neighborhood function. Large-to-small sigma annealing = coarse-to-fine. Neural Gas: rank-based neighborhoods adapt to data.
4. **ACO**: P_ij = tau^alpha * eta^beta / sum. Stigmergy: indirect coordination through environment. Shortest paths emerge from deposit + evaporation balance.
5. **Physarum**: dD/dt = |Q| - D. Positive feedback (flow amplifies conductivity) + decay. Provably optimal for linear flow problems. Bonifaci proof: converges to shortest path.
6. **Reaction-diffusion**: Turing instability when D_v >> D_u. Lambda = 2pi/omega_max sets characteristic wavelength. Local excitation + global inhibition = universal pattern formation.
7. **PSO**: v_i = omega*v_i + phi_1*r_1*(p_i-x_i) + phi_2*r_2*(g-x_i). Stochastic oscillation around weighted average of personal/social best.

---

## PART 1: CODEX DEEP THINK (5 Novel Ideas)

Codex identified the key pattern: **at 67M scale, Sutra accepts simple shared state and coarse bias; rejects high-entropy learned control.**

### Idea 1: Error Scratchpad (PRIORITY 1)
- Write prediction error (delta = LN(mu_t - sg(mu_{t-1}))) to scratchpad, not raw state
- Sparse, low-bandwidth, directly aligned with what late recurrence fixes
- Kill: <2% BPT gain OR <10% late-step improvement

### Idea 2: Sparse Discrete Broadcast Packets
- Replace smeared write-summary with m winner packets per step
- FSQ-style discrete codes, broadcast to all tokens next step
- Removes blur from current scratchpad averaging
- Kill: code usage collapses <30% OR BPT <1.5%

### Idea 3: Pheromone Router (PRIORITY 2)
- Decaying scalar trace over past positions: s_j <- rho*s_j + deposit_j
- Retrieval score: qk + alpha*s_j
- Deposit: high write-stage occupancy, large recurrent delta
- Kill: top-k recall not +5pts AND BPT <1%

### Idea 4: Entropy-Driven Clonal Expansion
- Select top p% highest-entropy positions, give them extra route-write micro-step
- Deterministic selection, not learned
- Adaptive compute without halting network
- Kill: under matched FLOPs, entropy-selected positions don't beat random by >=10%

### Idea 5: Depth-Drop Bootstrap (PRIORITY 3)
- Random truncate recurrence depth k in {2..8}
- KL to stop-grad full-depth teacher on same batch
- Makes every step a viable stopping point
- Kill: early-late gap not reduced >=25% by step 300

---

## PART 2: CHROME PROBE RESULTS (dim=128, 300 steps each)

### v0.5.4 Probes (vs v0.5.3 baseline)

| Probe | Test BPT | Delta | Late Step Value | Verdict |
|-------|----------|-------|----------------|---------|
| **v0.5.3 Baseline** | **7.317** | -- | 0.016 (step 5->6 drop) | -- |
| Error Scratchpad | 7.343 | -0.4% | 0.037 (2.2x better) | KILL (early noise) |
| Pheromone Router | 7.327 | -0.1% | 0.035 (2.2x better) | MARGINAL |
| Depth-Drop Bootstrap | (running) | -- | -- | -- |

**KEY PATTERN**: Both Error Scratchpad and Pheromone Router make late recurrence steps
2.2x more valuable, but hurt early steps. Root cause: both mechanisms need time to accumulate
meaningful signals (error deltas need calibrated predictions; pheromone traces need accumulated
usefulness history). At step 0-1 they inject noise.

**PROPOSED FIX**: Delay novel mechanisms until recurrent step 3. Steps 0-2 use standard
processing. Steps 3+ benefit from error/pheromone signals. This should preserve the 2.2x
late-step improvement while eliminating early-step degradation.

### Grokfast Probe (vs base v0.5 = 7.265 BPT)

| Config | Test BPT | Delta | Verdict |
|--------|----------|-------|---------|
| **Grokfast(a=0.95, l=2.0)** | **6.468** | **+11.0%** | **STRONG PASS** |
| Grokfast(a=0.98, l=5.0) | NaN | -- | KILL (diverged) |
| Grokfast(a=0.99, l=10.0) | (running) | -- | Likely KILL |

BEST Chrome result this session. Benefit GROWS with training steps.
Pure training trick, 5 lines of code. Add to production immediately.

### Peri-LN Probe (vs base v0.5 = 7.182 BPT)

| Config | Test BPT | Delta | Verdict |
|--------|----------|-------|---------|
| **Peri-LN** | **6.995** | **+2.6%** | **PASS** |

Zero-cost normalization. FEWER params. Already in OLMo2/Gemma2/Gemma3.

### Complete Chrome Summary

| Probe | BPT | vs Base | Verdict |
|-------|-----|---------|---------|
| **Grokfast(a=0.95,l=2.0)** | **6.468** | **+11.0%** | **PASS** |
| **Peri-LN** | **6.995** | **+2.6%** | **PASS** |
| Error Scratchpad | 7.343 | -0.4% | KILL |
| Pheromone Router | 7.327 | -0.1% | MARGINAL |
| Depth-Drop Bootstrap | NaN | -- | KILL |

PATTERN: Training tricks + normalization WIN. Novel architecture mechanisms FAIL at 67M.

---

## PART 3: CROSS-DOMAIN RESEARCH (11 Domains)

### 3A. BIOLOGY / NEUROSCIENCE

**Top mechanisms (ranked by implementability):**

1. **Dendritic Computation (Quadratic Neurons)** — NeurIPS 2024
   - Replace y = sigma(w^T x + b) with y = sigma(x^T A x + w^T x + b)
   - Low-rank A (rank k=4): adds only 512 params at dim=128
   - More expressive per-neuron = fewer params needed
   - Paper: "Dendritic Integration Inspired ANNs" (NeurIPS 2024)
   - Nature Communications Jan 2025: matches/outperforms with FEWER params

2. **Neuromodulation (Gain Modulation)** — 5 lines of code
   - Global gain scalar changes processing mode: output = sigma(g * W @ x)
   - One set of weights, MULTIPLE behaviors via tiny gain signal
   - eLife 2024: gain modulation IS the mechanism for task switching in humans
   - Maps to Sutra Stage 6 (Compute Control)

3. **Predictive Coding** — Error-only propagation
   - Each layer generates top-down prediction, only ERROR propagates upward
   - Inherently compressive (only "surprise" moves through network)
   - PLOS Comp Bio 2025: PC EMERGES from energy efficiency constraints
   - PyTorch implementations: Torch2PC, PRECO

4. **Cerebellar Learning** — Fast error prediction
   - Auxiliary network predicts FUTURE error from current state
   - Main model receives predicted error, corrects BEFORE seeing actual error
   - Nature Communications 2023: accelerates learning, reduces oscillatory failures
   - Could make 300 steps equivalent to 1000 steps

5. **Grid Cell Positional Encoding** — Zero additional params
   - Multi-scale Fourier features with bio-optimal scale ratios
   - Optimal ratio: e^(1/d) where d = dimensionality
   - GridPE paper (2024): matches/beats SOTA positional encodings
   - Free efficiency from better math

6. **Hebbian Feature Layer** — Unsupervised front-end
   - Soft WTA + BCM rule: delta_W = eta * z * (z - theta) * x^T
   - Online adaptation during inference, no backprop
   - Maps to Sutra Stage 1 (Segmentation/Compression)

7. **Hippocampal Replay** — Compressed experience replay
   - Surprise-weighted generative replay for continual learning
   - Nature Communications Feb 2025: CH-HNN with dual representations

### 3B. SMALL MODEL ML MECHANISMS (2024-2025)

**Highest impact:**

1. **nGPT Hypersphere Normalization** — 4-20x fewer training steps!
   - ALL vectors constrained to unit norm on hypersphere
   - Matrix-vector products become cosine similarities in [-1,1]
   - Weight decay unnecessary. Replaces normalization layers entirely
   - Paper: arXiv 2410.01131 (Loshchilov et al., Oct 2024)
   - DIRECTLY embodies "Intelligence = Geometry"

2. **Gated Attention** — NeurIPS 2025 Best Paper
   - Sigmoid gate AFTER attention output, before output projection
   - >0.2 perplexity drop. Eliminates attention sink problem
   - Already deployed in Qwen3-Next-80B
   - Paper: arXiv 2505.06708

3. **Differential Attention** — ICLR 2025
   - Attention = DIFFERENCE of two softmax maps (noise cancellation)
   - Outperforms at all scales, reduces hallucination
   - Paper: arXiv 2410.05258

4. **Value Residual Learning** — 16% fewer params same quality
   - Propagate first-layer values to all subsequent layers
   - Addresses information loss in deep networks
   - Paper: arXiv 2410.17897

5. **Multi-Token Prediction** — Meta, adopted by DeepSeek-V3/Qwen-3
   - Predict 2-4 tokens ahead simultaneously, shared trunk
   - +12% HumanEval, +17% MBPP, 3x faster inference
   - Paper: arXiv 2404.19737

6. **minGRU** — "Were RNNs All We Needed?"
   - Stripped GRU: single update gate, parallel scan training
   - 175x faster than sequential. ~10 lines of PyTorch
   - Matches Mamba/Transformers
   - Paper: arXiv 2410.01201

7. **Data Mixing Laws** — ICLR 2025
   - Optimize data domain mixture mathematically on cheap small runs
   - Same as 48% more training steps
   - Paper: arXiv 2403.16952

8. **Peri-LN** — ICML 2025
   - LayerNorm before AND after each sublayer
   - Fixes training stability. Zero extra params
   - Already in OLMo2, Gemma2, Gemma3

### 3C. QUANTUM PHYSICS

**Key principles for architecture design:**

1. **Information lives in GEOMETRY** (amplitudes, phases, inner products, subspaces)
2. **Nature computes in richer space than it reports** (amplitude space vs probability space)
3. **Tensor network geometry must match correlation geometry** — wrong topology = exponential cost, right topology = polynomial cost
4. **MERA = discrete anti-de Sitter space** — hierarchical multi-scale architecture IS holographic
5. **Min-cut principle** — information flow through any structured network is determined by minimum cut
6. **Depth = robustness** — deeper representations are more protected (holographic error correction)
7. **Scale is a spatial dimension** — RG flow depth direction = extra dimension in AdS/CFT

**Implementable mechanisms:**

1. **Wave Network** (arXiv 2411.02674) — Complex vectors, magnitude = semantics, phase = relations
   - 2.4M params achieves 91.66% AG News (comparable to 100M BERT at 94.64%)
   - 77% memory reduction, 86% training time reduction

2. **Unitary State Transitions** — Cayley transform parameterization
   - Perfect gradient flow: ||Wh|| = ||h|| guaranteed
   - Solves vanishing/exploding gradient for recurrence

3. **Interference-Based Routing** — Complex-valued gates where amplitudes interfere
   - 10x more parameter-efficient than classical routers
   - Paper: arXiv 2512.22296

### 3D. CATEGORY THEORY + ALGEBRA

**Highest practical impact:**

1. **Tropical Attention** — NeurIPS 2025
   - Max-plus semiring replaces softmax
   - 3-9x faster inference, ~20% fewer params
   - Superior OOD generalization (train size 8 -> test size 1024)
   - First to handle NP-hard reasoning tasks
   - Code: github.com/Baran-phys/Tropical-Attention

2. **Sheaf Neural Networks** — SOTA on heterophilic graphs
   - Learned restriction maps for inter-component communication
   - Information flow mediated by learned linear maps, not scalar weights
   - Multiple SOTA results: 88.11% Texas, 89.80% Wisconsin

3. **Automata Theory Constraints** — ICLR 2025 Oral
   - Linear RNNs with only positive eigenvalues CANNOT solve parity
   - Negative eigenvalues unlock state tracking
   - Non-triangular matrices needed for counting modulo 3
   - Sutra MUST ensure right algebraic properties in transition kernel

4. **Formal Language Expressiveness** — ICLR 2025
   - ALL tested architectures limited to regular languages + simple DCFLs
   - Sutra opportunity: provably exceed transformer expressiveness

**Meta-principles from category theory:**
- Objects defined by relationships, not internal structure (Yoneda)
- Adjunctions are the fundamental organizational pattern
- Composition is the primitive operation
- Logic and geometry are dual (topoi)

### 3E. PHYSICS-INSPIRED MECHANISMS

1. **Wave-PDE Nets** — PRICAI 2025 Oral
   - Each layer simulates wave equation with trainable velocity c(x) + damping
   - Symplectic spectral solver: O(n log n) via FFTs
   - MATCHES Transformer performance, 30% faster, 25% less memory
   - Proven universal approximator
   - Paper: arXiv 2510.04304

2. **Grokfast** — 50x grokking acceleration, 5 lines of code
   - Low-pass filter on gradient time series
   - Amplifies slow generalization-inducing modes
   - Paper: arXiv 2405.20233, GitHub available

3. **Symmetry Breaking for Specialization** — npj AI Nov 2025
   - Augmenting input dims with constants improves performance
   - ReMoE: ReLU routing (fully differentiable, ICLR 2025)
   - Connect to Sutra's stage specialization

4. **Thermodynamic Bounds** — Linear operations: zero energy cost (reversible), nonlinear activations: unavoidable cost
   - Sutra should minimize activation state transitions

### 3F. THERMODYNAMICS + INFORMATION GEOMETRY

1. **Fisher Information Metric** — Natural geometry on statistical manifolds
   - Cramer-Rao bound: optimal estimation
   - Natural gradient descent exploits this geometry

2. **Renormalization Group** — Scale-invariant feature learning (AAAI 2025)
   - Each layer = coarse-graining operation
   - RG flow equations for network parameters
   - Scaling law universality at large data

3. **Free Energy Principle (Friston)** — Unifies perception, action, learning
   - Variational free energy minimization
   - Connects to predictive coding

### 3G. NCA / SOMs / MORPHOGENESIS

**Core principles:**

1. **Shared local rule + hidden channels + many iterations = global patterns** (NCA)
   - Single tiny rule for any pattern size = maximum compression
   - Hidden channels carry positional/gradient information

2. **Coarse-to-fine annealing** (SOM)
   - Large neighborhood first (global ordering) -> small neighborhood (local refinement)
   - Direct analogy to multi-scale processing

3. **Local excitation / global inhibition** (Turing patterns)
   - Short-range positive feedback + long-range negative feedback
   - The RATIO of ranges determines pattern scale

4. **Grow structure to match complexity** (Growing Neural Gas)
   - Allocate computational resources proportional to local complexity

### 3H. COLLECTIVE INTELLIGENCE / SWARM

1. **NCA Pre-Pre-Training for LLMs** — arXiv 2603.10055, March 2026
   - 164M NCA tokens give 6% LM improvement, 1.6x faster convergence
   - OUTPERFORMS 1.6B tokens of CommonCrawl
   - Teaches in-context learning before language

2. **Global Workspace Theory** — ICLR 2022
   - Validates scratchpad as correct pattern (what biology uses for conscious integration)
   - LINEAR complexity (vs quadratic for attention)
   - Sutra independently converged on this mechanism

3. **Stigmergy** — Indirect coordination through shared environment
   - Mathematical explanation of WHY scratchpad works
   - O(n) communication achieving global coherence

4. **Physarum Routing** — Provably optimal for linear flow problems
   - Flow-conductivity feedback: tubes that carry more flow get thicker
   - Ideal model for Stage 4 routing

5. **Edge of Chaos** — Optimal NN performance at order-chaos transition
   - Maximizes information transmission, dynamical range, storage capacity
   - Mean field theory gives exact initialization conditions

### 3I. INFORMATION THEORY / COMPRESSION

1. **Grokfast Gradient Filtering** — 5 lines, 50x acceleration
2. **Compression Ratio as Eval Metric** — r=-0.95 with benchmark scores (COLM 2024)
3. **Differentiable MDL Objective** — arXiv 2509.22445 (ICLR 2026)
4. **Batch-Entropy Regularization** — Quantifies info flow, enables deep training without skip connections
5. **LZ Penalty for Generation** — Eliminates degenerate repetition

---

## PART 4: CONVERGENT THEMES ACROSS ALL DOMAINS

The 11 domains converge on these design principles:

### Theme 1: GEOMETRY IS THE RIGHT ABSTRACTION
- Quantum: information lives in geometry (amplitudes, phases)
- nGPT: hypersphere constraint = 4-20x speedup
- Category theory: objects ARE their relationships
- Wave-PDE: wave propagation IS information routing
- **Implication for Sutra**: constrain representations to geometric manifolds

### Theme 2: SHARED SIMPLE RULES + ITERATIONS > COMPLEX SINGLE-PASS
- NCA: one tiny rule + iterations = arbitrary complexity
- Sutra's recurrent loop is already this pattern
- minGRU: stripped-down gating matches complex architectures
- **Implication**: don't add complexity to the rule; add quality to the shared state

### Theme 3: ERROR/SURPRISE DRIVES LEARNING
- Predictive coding: only prediction error propagates
- Cerebellar learning: predicted error accelerates convergence
- Codex idea 1: error scratchpad >> state scratchpad
- Grokfast: separate slow (generalization) from fast (memorization)
- **Implication**: communication channels should carry surprise, not raw state

### Theme 4: MULTI-SCALE ORGANIZATION
- MERA/holography: scale = extra dimension
- Reaction-diffusion: differential rates create multi-scale patterns
- SOM annealing: coarse-to-fine
- RG: each layer is coarse-graining
- **Implication**: Sutra stages should explicitly operate at different temporal/spatial scales

### Theme 5: TOPOLOGY OF COMPUTATION MATTERS
- Tensor networks: wrong topology = exponential cost
- Automata theory: algebraic structure of transitions determines expressiveness
- Sheaf networks: learned maps between components > scalar weights
- Tropical algebra: max-plus semiring > softmax for reasoning
- **Implication**: the graph structure of stage transitions is the architecture's most important feature

---

## PART 5: RECOMMENDED v0.5.4 DESIGN CANDIDATES

Based on cross-domain convergence, here are the TOP candidates for Codex to evaluate:

### TIER 1: Immediate (validated, <50 lines, high expected impact)
1. **Grokfast gradient filtering** — 5 lines, 50x acceleration
2. **Error scratchpad** — Write delta not raw state (Chrome probe running)
3. **Pheromone router** — Decaying trace for routing (Chrome probe running)
4. **Peri-LN** — Normalize before AND after each sublayer

### TIER 2: High Priority (validated elsewhere, need Chrome validation)
5. **Wave-PDE routing** — Replace LocalRouter attention with wave equation FFT solver
6. **Dendritic neurons in StageBank** — Quadratic x^T A x + w^T x + b
7. **Depth-drop bootstrap** — Random truncation + KL to teacher (Chrome probe running)
8. **Neuromodulation gain** — Stage 6 compute control via learned gain scalar

### TIER 3: Architectural (require careful integration)
9. **nGPT hypersphere normalization** — Constrain all vectors to unit norm
10. **Unitary state transitions** — Replace transition kernel with unitary-parameterized version
11. **Tropical routing** — Max-plus semiring in router instead of softmax
12. **NCA pre-pre-training** — Train on NCA data before language data

### TIER 4: Training improvements (orthogonal to architecture)
13. **Multi-token prediction** — Auxiliary heads predicting 2-4 tokens ahead
14. **Data mixing optimization** — Mathematical optimization of data proportions
15. **Batch-entropy regularization** — Auxiliary loss quantifying information flow
