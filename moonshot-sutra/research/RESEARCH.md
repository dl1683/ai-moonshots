# Sutra Research Log

## Chrome Cycle 1: Eval Set Design (2026-03-19)

### Theory: What Makes a Discriminating Eval?

From Item Response Theory (IRT), each question has discrimination `a_i` and difficulty `b_i`. The information function `I_i(theta) = a_i^2 * P_i(theta) * Q_i(theta)` peaks when question difficulty matches model ability. A good eval is a spectrum analyzer across the ability range.

### Key Design Decision: Zero Fact Lookup

Current benchmarks (MMLU, ARC, HellaSwag) primarily test knowledge retrieval. This favors models trained on more data, not models with better architecture. For Sutra, we need to test the REASONING ENGINE, not the knowledge database.

**Principle**: If a model knows WHAT to look up and HOW to combine information, fact lookup becomes a trivial tool call. The hard part is the thinking process.

### Taxonomy (500 questions, 7 categories)

| Category | Count | Tests |
|----------|-------|-------|
| Strategic Reasoning | 100 | Planning, trade-offs, game theory, resource allocation |
| Synthesis & Combination | 100 | Cross-domain connection, creative problem solving |
| Critical Analysis | 80 | Flaw detection, assumption identification, evidence evaluation |
| Instruction Following | 80 | Precision under complex interacting constraints |
| Drafting Under Constraints | 60 | Generation quality with multiple simultaneous requirements |
| Code & Algorithmic Thinking | 50 | Algorithm design, debugging reasoning, optimization |
| Meta-Cognition | 30 | Self-awareness, knowledge gap identification, calibrated uncertainty |

Difficulty distribution within each: 20% easy, 30% medium, 30% hard, 20% extreme.

### Scoring Framework

Three modes:
1. **exact_match** — one correct answer (math, logic, some code)
2. **constraint_check** — binary per-constraint, score = fraction met (instruction following)
3. **rubric** — multi-dimensional 0-3 scoring (drafting, synthesis, analysis)

Automated scoring: exact_match (~40%) + constraint_check (~30%) + LLM-as-judge with explicit rubrics (~30%).

### Codex Review (2026-03-19): 7/10 prompt bank, 4/10 benchmark-ready

Key issues: scorer doesn't implement rubric scoring, exact_match answers ambiguous (SR001/SR003), difficulty cliff-edges, rubric inconsistencies (SR041), category overlap (drafting vs IF), meta-cognition saturates. Coverage gaps: formal proof, adversarial ambiguity, long dependency chains, counterfactual updating, calibrated estimation, multi-turn self-correction. Fix agent running.

### Dead Ends

*(None yet)*

---

## Chrome Cycle 2: Architecture Research Sweep (2026-03-19)

### Internet Research Findings

#### 1. Mamba-3 (ICLR 2026) — The New SSM Frontier
- **4% better than transformers** on LM benchmarks, **7x faster** at long sequences
- Three innovations: exponential-trapezoidal discretization (2nd order accurate), complex-valued SSMs (solve synthetic reasoning transformers couldn't), MIMO formulation (4x more parallel ops)
- Achieves Mamba-2 perplexity with **half the state size**
- At 1.5B: +1.8pp average downstream accuracy over Gated DeltaNet
- **Key insight**: complex SSMs recover state tracking — addresses a fundamental SSM weakness

#### 2. Hybrid Architectures Dominate Production (2025-2026)
- Jamba: Transformer+Mamba+MoE, 1:7 attention:Mamba ratio
- Nemotron-H: 3x faster than comparable Transformers, matches/exceeds on MMLU/GSM8K/HumanEval
- RWKV-X: linear complexity training, constant-time inference, near-perfect 64K passkey retrieval
- Bamba: 2x throughput over comparable Transformers
- **Pattern**: attention for complex reasoning, SSM for efficiency. Ratio matters.

#### 3. Small Model Architecture: Depth > Width
- At 70M params, architecture choice (LLaMA3/Qwen3/Gemma3) matters <1%
- **Depth-width ratio is the true determinant**, not specific architecture
- Hidden >=512 OR 32+ layers required — "dead zone" at 16-48 layers with hidden <512
- 32 layers is "Goldilocks" for small models — beats 12-layer designs
- Canon layers (depthwise causal convolutions): +0.13% params for +1-2% factuality
- **Diffusion LLMs**: 3.8x throughput, -1.33% accuracy, +1.67% TruthfulQA factuality

#### 4. Test-Time Compute Scaling — Small Model Superpower
- "Compute-optimal" test-time scaling: 4x more efficient than best-of-N
- Recurrent depth approach (3.5B params, 800B tokens): iterate a recurrent block to arbitrary depth at test time
- By late 2025: 10-20B models with test-time compute ≈ frontier model reasoning
- **This is huge for Sutra**: a small model that can dynamically scale inference compute could punch way above its weight class

#### 5. Compression = Intelligence (Theoretical)
- Bridging Kolmogorov Complexity and Deep Learning (arXiv 2509.22445): proves asymptotically optimal description length objectives exist for Transformers
- Compression Represents Intelligence Linearly: linear relationship between compression ability and benchmark performance
- Grokking = compression phase transition (complexity dynamics paper)
- MDL principle: best model = best compressor. Training = learning to compress.
- **This validates Sutra's thesis**: if we build a better compressor, we get a more intelligent model

#### 6. Biological Inspiration
- Neuromorphic LLM on Intel Loihi 2: same accuracy as GPU, **half the energy**
- Spiking neural networks: ~69% activation sparsity, orders of magnitude energy savings
- SpikeLLM: first spiking LLM at 7-70B scale
- SpikeRWKV: spiking + RWKV hybrid
- **Key lesson**: event-driven, sparse computation is massively more efficient. Can we build this into the architecture?

#### 7. Why Small Models Fail at Reasoning
- Clear capability boundary at ~3B parameters
- Sub-2B struggle with autonomous architectural decisions
- Root cause: limited capacity for multi-step compositional reasoning
- **Fixes that work**: teacher distillation, synthetic reasoning data, Chain-of-Reasoning (blend NL + code + symbols), Logic-RL, test-time compute scaling
- **Key insight**: the bottleneck is in TRAINING, not just architecture. Better training recipes help as much as better architectures.

### DISCARDED: Industry-Survey Hypotheses (H1-H5 v1)

These were incremental combinations of existing architectures (hybrid SSM, compression-native, neuromorphic, deep narrow, standard+training). Discarded because they copy, not derive. Kept as context only.

---

## Chrome Cycle 3: First-Principles Derivation (2026-03-19)

### What IS Intelligence? (Derived, not assumed)

Five irreducible operations any intelligent text system must perform:
1. **Compress** — predict next symbols (Shannon: prediction = compression)
2. **Compose** — combine known concepts into novel structures (algebraic)
3. **Abstract** — collapse equivalent forms into shared representations (quotient space)
4. **Reason** — chain inference steps of variable depth (iterated function composition)
5. **Select** — choose from exponential space of continuations (search/optimization)

### The Five Axioms of Current AI (Questioned)

| Axiom | What if violated? | Implication |
|-------|------------------|-------------|
| Representations must be real-valued vectors | Complex, hyperbolic, p-adic, distributional | Better geometric match to data structure |
| Processing depth is fixed | Variable/adaptive depth | Compute proportional to problem difficulty |
| Learning = gradient descent on loss | Evolution, energy minimization, MDL, program synthesis | Different optimization landscape, possibly better |
| More parameters = more capability | Structure > size | Exponentially more efficient representations |
| Process tokens sequentially | Operate on concepts of variable granularity | Computation at the right grain |

### Core Hypotheses (First-Principles)

**H1: Compression Machine** — MDL-trained model will be more intelligent per parameter than CE-trained
- Kill: MDL model has >5% worse perplexity than CE at same params

**H2: Variable-Depth Reasoning** — Dynamic computation depth → >20% reasoning gain at same FLOPs
- Kill: <5% improvement on 5+ step reasoning

**H3: Hierarchical Representation Space** — Hyperbolic/structured geometry → >10% compositional generalization gain
- Kill: <3% improvement OR >30% training divergence rate

**H4: Energy-Based Reasoning** — Global energy minimization → more coherent multi-step reasoning
- Kill: >10x slower AND <3% more accurate

**H5: Concept-Level Computation** — Variable-granularity units → >1.5x inference efficiency at equal quality
- Kill: >10% worse perplexity

### Experiment Batch 1 v1: REJECTED BY CODEX (5/10)

Codex found: probes confounded (adaptive depth also changes capacity), wrong operationalizations
(WordNet tests taxonomy not abstraction), missing controls (no matched-compute comparisons),
missing critical probes (compression↔capability, working memory). See results/codex_batch1_review.md.

### Experiment Batch 1 v2: "The Primitives" (Redesigned per Codex)

**Priority 1 — MUST RUN (Codex-approved top 3 + 2 missing):**

**Probe A: Compression ↔ Capability Correlation** (CORE THESIS TEST)
- Question: Does better compression ACTUALLY predict better reasoning at fixed params?
- Design: Train 6 tiny models (10M params each) with different objectives: CE, CE+L2, CE+dropout,
  label smoothing, MDL-approx (variational bits-back), and a deliberately BAD compressor (noisy labels).
  Measure held-out bits-per-byte AND accuracy on 50 synthetic reasoning tasks.
  Plot compression vs capability. If r > 0.8, thesis confirmed.
- Controls: Same architecture, same data, same params, same training steps. Only objective varies.
- Kill: If r(compression, capability) < 0.5, compression ≠ intelligence at this scale.
- Time: ~3 hours (6 models × 30min each)

**Probe B: Variable Depth with Matched Compute** (Redesigned from v1 Probe 2)
- Question: Does adaptive depth help reasoning when compute is matched?
- Design: Three 20M-param models on synthetic multi-step arithmetic:
  (A) Fixed 12 layers, (B) Fixed 24 layers (same width, more params for fairness note),
  (C) Shared-weight recurrent block, 1-24 iterations, adaptive halting.
  Matched-FLOPs comparison: give C the same total FLOPs as A per input.
- Controls: Random-halting baseline (same FLOPs distribution but random depth per input).
- Measurements: Accuracy by reasoning depth (1,2,3,5,8,10-step), FLOPs per correct answer.
- Kill: If adaptive depth < fixed-24 on hard tasks OR < fixed-12 at matched FLOPs.
- Time: ~4 hours

**Probe C: Working Memory / State Tracking** (NEW — Codex-recommended)
- Question: What is the minimal mechanism for variable binding and state tracking?
- Design: Three 10M-param models on synthetic state-tracking tasks
  (variable assignment, pointer chasing, stack operations):
  (A) Standard transformer (attention as implicit memory),
  (B) Explicit external memory (neural Turing machine style key-value store),
  (C) Recurrent state (hidden state carried forward, no attention).
  Test on tasks requiring 1, 5, 10, 20 variable bindings.
- Controls: Scrambled-variable baseline (same tasks, variables shuffled — tests memorization vs tracking).
- Kill: If all three architectures fail at >5 bindings, the problem is training not architecture.
- Time: ~3 hours

**Probe D: Energy-Based vs Autoregressive with Matched Inference Compute** (Redesigned from v1 Probe 4)
- Question: Does global optimization produce more coherent reasoning than left-to-right generation?
- Design: Three 20M-param models on synthetic reasoning (logic chains where answer depends on late context):
  (A) Standard autoregressive,
  (B) AR + verifier/reranker (generate N, score, pick best — same test-time compute as C),
  (C) Discrete diffusion (iterative denoising — matched test-time FLOPs to B).
- Controls: Matched inference FLOPs across B and C. A gets 1x, B and C get 5x.
- Kill: If AR+verifier matches or beats diffusion at same compute, energy-based adds nothing over search.
- Time: ~6 hours

**Probe E: Concept-Level vs Token-Level with Matched Sequence Budget** (Redesigned from v1 Probe 5)
- Question: Does variable-granularity representation improve quality-per-latency?
- Design: Four 20M-param models on English text:
  (A) BPE tokenizer (standard, ~4 chars/token),
  (B) Byte-level (no tokenizer, raw bytes),
  (C) Unigram tokenizer (different segmentation algorithm),
  (D) Oracle word/morpheme boundaries (cheating baseline — upper bound).
  All trained to same number of characters seen (not tokens).
- Controls: Matched character budget, not token budget. Report quality-per-latency, not just perplexity.
- Kill: If BPE dominates all alternatives on quality-per-latency, tokenization isn't the bottleneck.
- Time: ~4 hours

**Total: 5 probes, ~20 hours, all 10-20M params. Results determine which primitives combine into Sutra.**

**Status**: Probe A running. Probes B-E in implementation.

---

## Chrome Cycle 4: Cross-Domain Intelligence Patterns (2026-03-19)

### Universal Patterns Across Intelligent Systems

Surveyed: brains, immune systems, ant colonies, slime molds, gene regulatory networks, plant/mycorrhizal networks, thermodynamics of computation, category theory, power laws. The same structural patterns keep recurring:

#### Pattern 1: Prediction = Compression = Intelligence (UNIVERSAL)
- **Brain (Friston)**: Free energy principle — the brain minimizes surprise (prediction error). Perception IS prediction. Learning IS model compression. This is mathematically identical to variational inference.
- **Immune system**: Antibodies are compressed models of pathogen structure. Affinity maturation IS compression — refining the model until it captures the essential features with minimum complexity.
- **Gene regulatory networks**: Boolean networks converge to attractors — stable compressed states that represent cell phenotypes. The attractor IS the compressed representation of a developmental program.
- **Thermodynamics (Landauer)**: Erasing one bit costs kT*ln(2) energy minimum. Intelligence has a physical cost. The most efficient intelligence is the one that compresses the most with the least erasure.
- **INSIGHT FOR SUTRA**: Compression isn't just A property of intelligence — it IS intelligence. Every intelligent system we observe in nature is fundamentally a compression engine. Our architecture should have compression as its PRIMARY operation, not a byproduct.

#### Pattern 2: No Central Controller (UNIVERSAL)
- **Ant colonies**: Shortest paths emerge from local pheromone rules. No ant knows the global optimal path. Intelligence emerges from local interactions + environmental modification (stigmergy).
- **Immune system**: No central controller decides which antibody to produce. Clonal selection + hypermutation = distributed search with local feedback.
- **Slime mold (Physarum)**: Solves TSP in linear time, no neurons. Flow dynamics in a tube network. Each tube expands/contracts based on LOCAL nutrient flow.
- **Plant/mycorrhizal networks**: Scale-free topology, small-world properties. Each root segment makes semi-autonomous decisions. Network-level intelligence from local computation.
- **Gene regulatory networks**: Boolean attractor dynamics. No master gene. Cell fate is determined by the NETWORK topology, not any individual node.
- **INSIGHT FOR SUTRA**: Current models have a fundamentally centralized architecture — everything passes through a global attention mechanism or hidden state. What if intelligence emerged from LOCAL interactions between components? Not "one big model" but "many small interacting agents." This is fundamentally different from any existing architecture.

#### Pattern 3: Adaptive Resource Allocation (UNIVERSAL)
- **Brain**: Predictive coding allocates precision (computational resources) to surprising signals. Expected inputs get minimal processing. Surprises get maximum attention.
- **Immune system**: Clonal expansion — cells that recognize a threat multiply 10,000x. Resources flow to where they're needed.
- **Ant colonies**: Pheromone concentration builds on successful paths, evaporates from failed ones. Resources (ants) naturally flow to productive activities.
- **Slime mold**: Tubes on productive paths thicken (more flow). Tubes on dead-end paths thin and retract.
- **INSIGHT FOR SUTRA**: Every efficient intelligent system in nature allocates compute dynamically. Current models give equal compute to every token. This is provably wasteful. Our architecture MUST allocate more computation to harder/more informative tokens and less to predictable ones.

#### Pattern 4: Multi-Scale Structure (UNIVERSAL)
- **Fractals/power laws**: The same patterns appear at every scale in nature. Coastlines, galaxies, vascular networks, neural networks, social networks — all scale-free.
- **Gene regulatory networks**: Operate at molecular scale, cell scale, tissue scale, organism scale — same Boolean logic at each level.
- **Mycorrhizal networks**: Local root decisions, tree-level strategies, forest-level resource sharing — intelligence at every scale.
- **Category theory**: Functors map between scales while preserving structure. Compositionality IS the mathematical expression of multi-scale invariance.
- **INSIGHT FOR SUTRA**: Representations should be multi-scale by construction. Not "one embedding per token" but "representations at character, word, phrase, sentence, paragraph, document level" — with the SAME operations applying at each scale (self-similarity). This is exactly what fractals are: the same structure at every scale.

#### Pattern 5: Learning Without Backpropagation (UNIVERSAL)
- **Immune system**: Clonal selection + somatic hypermutation = evolutionary search, not gradient descent.
- **Gene regulatory networks**: Attractor dynamics = convergence to stable states without gradient computation.
- **Ant colonies**: Reinforcement via pheromone = environmental modification, not weight updates.
- **Slime mold**: Flow dynamics converge to optimal networks via physical forces, not learning algorithms.
- **INSIGHT FOR SUTRA**: Gradient descent is ONE way to learn, not THE way. What if parts of our system learned through evolution (architecture search), parts through energy minimization (attractor dynamics), and parts through gradient descent? Hybrid learning strategies might find solutions that gradient descent alone cannot.

### Revised Hypothesis: The Stigmergic Compression Engine

Based on cross-domain patterns, I now think the most promising Sutra direction is:

**A system of many small, locally-interacting compression agents that:**
1. Each compress their local input (Pattern 1)
2. Communicate through a shared medium, not central attention (Pattern 2)
3. Allocate more agents/iterations to surprising inputs (Pattern 3)
4. Operate at multiple scales simultaneously (Pattern 4)
5. Use hybrid learning: gradient descent for local compression + evolutionary search for agent topology (Pattern 5)

This is NOT a transformer. NOT an SSM. NOT a neural network in the traditional sense. It's more like a colony of compression agents — a "Physarum of text" where:
- Each agent receives a local window of text
- Agents compress their window and deposit "pheromone" (compressed features) on a shared medium
- Other agents read nearby pheromone to build larger-scale representations
- The system naturally routes computation to surprising/hard passages
- The shared medium has multi-scale structure (like a fractal)

**Key theoretical prediction**: If this works, it would achieve O(n) scaling (like SSMs) with dynamic-depth reasoning (like adaptive transformers) and compositional generalization (from the multi-scale structure) — the best of all worlds from first principles.

**This needs probing IMMEDIATELY.** Can a system of locally-interacting agents learn to model text at all? This is the foundational question before any optimization.

---

## Probe Results (as they come in)

### Probe E: Tokenization Analysis — COMPLETED

**Result**: Whitespace tokenization (BPB=1.48) beats BPE (1.65), morpheme (1.58), and all character/byte methods (4.27). Compression range 2.88x across methods.

**Key insight**: Natural word boundaries are already excellent compression units for this corpus. BPE's learned merges don't add much over word-level splitting. Entropy-optimal greedy character-level segmentation FAILS (4.27 BPB, same as raw bytes) — because it can't capture word-level structure from character-level decisions.

**Implication for Sutra**: Word/phrase-level computation is more efficient than subword. H5 (concept-level) has directional support — but the real test is model quality, not just tokenizer compression. A model that processes WORDS rather than subword tokens could be inherently more efficient.

**Caveat**: This is a synthetic corpus with limited vocabulary. Real English with rare words, names, code, etc. would shift the balance toward BPE/subword for handling OOV.

---

## Deep Theory Session: Rethinking Every Assumption (2026-03-19)

### The Tokenization Problem — What's Actually Wrong?

Every current tokenizer makes the same assumption: segmentation is fixed BEFORE the model sees the data. The model has no say in how input is chunked. This forces the model to waste capacity undoing bad segmentation decisions.

**Insight from neuroscience (predictive coding)**: The brain doesn't process fixed chunks. It processes at the granularity that minimizes prediction error — coarse for predictable, fine for surprising. A truly intelligent tokenizer would be:
1. Part of the model itself (learned, not predetermined)
2. Dynamic (different segmentation for different contexts)
3. Adaptive (resolution proportional to information content)

**Radical direction**: Treat text as a continuous 1D signal, not discrete tokens. Model learns its own "sampling rate" adaptively. Like the cochlea for audio — frequency decomposition, not fixed windows.

### In-Generation Verification — Three Biological Models

**1. Immune System Model (Generate and Test)**: Generate many candidate continuations, select best via internal coherence function. In stigmergic framework: each agent generates a candidate, medium amplifies high-quality ones, suppresses garbage. Natural selection in real-time.

**2. Energy Landscape Model (Roll Downhill)**: Define energy function over complete sequences. Generation = finding minimum energy configuration. Like protein folding — don't assemble left-to-right, explore landscape and settle. Energy function could be SIMPLER than autoregressive model because it evaluates static coherence, not sequential dependence.

**3. Predictive Coding Model (Error-Driven Refinement)**: Hierarchical model predicts top-down, errors propagate bottom-up. High level: "express [concept]." Mid level: "[sentence structure]." Low level: "[specific words]." Verification is BUILT IN — every level checks the one below against the plan.

### Multi-Scale Processing — Why Single-Scale Is Provably Suboptimal

Language has at least 6 scales: character, word, phrase, sentence, paragraph, document. Current models are single-scale (one token at a time). The visual cortex processes V1→V2→V4→IT simultaneously with bidirectional flow. A model that processes ALL scales simultaneously with both up (abstraction) and down (prediction/verification) flow should be fundamentally more efficient.

This is fractal computation: the SAME compression operation at every scale, with cross-scale error signals.

### Draft Architecture Concept: Sutra v0.1

```
CONTINUOUS INPUT (byte stream)
    |
ADAPTIVE SEGMENTER (learned, dynamic granularity — resolution ~ surprise)
    |
MULTI-SCALE REPRESENTATION
    +-- Fine (char/subword)
    +-- Medium (word/phrase)
    +-- Coarse (sentence/paragraph)
    |
STIGMERGIC PROCESSING (at each scale independently)
    - Local compression agents (shared weights within scale)
    - Shared medium for agent communication
    - No global attention — O(n) not O(n^2)
    - Multiple message-passing rounds
    |
CROSS-SCALE INTERACTION (predictive coding)
    - Fine -> Coarse (abstraction / error propagation up)
    - Coarse -> Fine (prediction / verification down)
    |
ENERGY-BASED SELECTION (for generation)
    - Multiple candidate outputs scored by coherence energy
    - Best candidate selected (immune-system-like)
    |
ADAPTIVE DEPTH
    - Easy: 2-3 processing rounds
    - Hard: 10-20+ rounds
    - Halting learned per-input
    |
OUTPUT
```

**Why each component (theoretical justification):**
- Adaptive segmenter: fixed tokenization wastes capacity (Probe E: word boundaries beat BPE)
- Multi-scale: language IS multi-scale; single-scale provably suboptimal for hierarchical data
- Stigmergic: O(n) local interaction matches biological efficiency; global attention unnecessary (Pattern 2)
- Cross-scale predictive coding: verification built into the architecture (Friston free energy)
- Energy-based: left-to-right commit provably suboptimal for global coherence
- Adaptive depth: reasoning difficulty varies by orders of magnitude (Pattern 3)

**STATUS: THEORETICAL DRAFT. Needs Codex review + empirical probes before committing.**

Key uncertainties to resolve:
1. Can stigmergic processing at any scale match transformer quality? (Probe F tests this)
2. Does adaptive segmentation actually help or just add complexity?
3. Can cross-scale interaction be trained stably with gradient descent?
4. Is energy-based generation practical for text at reasonable speed?
5. How many parameters does this need to be competitive?

### Theoretical Deep Dive: Long-Range Dependencies Without Global Attention

THE hardest challenge for local-only processing. "The cat that the dog that the rat bit chased ran away" requires matching token 1 with token 11.

**Solution: Multi-scale hierarchy converts long-range to short-range.**

At scale 0 (tokens): dependency spans 11 positions.
At scale 1 (words): dependency spans ~5 positions.
At scale 2 (phrases): dependency spans ~2 positions (NP, relative clause, VP).

Each scale only needs LOCAL interaction (window ~5-8). Long-range becomes short-range at higher scales.

**Formal receptive field analysis:**
- Compression ratio per scale: r (e.g., r=4 for token→word, r=8 for word→phrase)
- With S scales and window size w at each scale:
- Effective receptive field = w * r^S
- With w=8, r=4, S=3: receptive field = 8 * 64 = 512 tokens
- With w=8, r=4, S=4: receptive field = 8 * 256 = 2048 tokens
- EXPONENTIAL growth with scales, LINEAR compute per scale

This is EXACTLY the advantage of wavelet transforms over Fourier: local processing with multi-resolution gives O(n) computation with O(n log n) effective connectivity. Transformers use O(n²) to achieve the same connectivity.

**Why this might be strictly BETTER than attention:**
- Attention gives every position equal access to every other position — most of which is wasted (attention maps are very sparse in practice)
- Multi-scale hierarchy gives STRUCTURED connectivity: strong at short range, weaker but present at long range — matching the actual dependency structure of language
- Brain uses exactly this: local dense connections + hierarchical long-range routing

**Trainability:** Standard backprop works. Gradient flows: output → fine scale → medium (via cross-scale connection) → coarse scale → back down. Same as any multi-scale CNN (U-Net, FPN). Well-understood.

**Key insight for Codex discussion:** The stigmergic medium at each scale acts as a "routing table" — agents at the fine scale don't need to see far, they just read the medium which contains COMPRESSED information from far away, deposited by agents that were closer to the source. Information propagates at the speed of the medium, not the speed of local agents. Like how pheromone trails carry information about distant food sources to ants that have never been there.

### GNN Perspective: Why Stigmergic = Local GNN

Transformers are GNNs on COMPLETE graphs (every token connected to every other).
Stigmergic model is a GNN on a LOCAL 1D lattice (each agent connected to neighbors only).
Multi-scale adds HIERARCHICAL edges (coarse-scale connections = long-range shortcuts).

This is EXACTLY a small-world network: mostly local connections + a few long-range shortcuts.
Small-world networks are known to have O(log n) average path length with O(n) edges.
Complete graphs have O(1) path length but O(n²) edges.
The question is whether O(log n) path length is sufficient for language modeling.

**Prediction**: For most text, O(log n) is MORE than sufficient. Only pathological nested structures (garden-path sentences, deeply nested recursion) require the full O(1) path length that attention provides. And those are rare in practice.

### Immune System Insight: V(D)J Compositional Agents

The immune system achieves 10^11 unique antibodies from ~300 gene segments via V(D)J recombination.
Small library of PARTS combined combinatorially → exponential coverage from linear parameters.

**Direct application to Sutra:**

Instead of fixed agents, use COMPOSITIONAL agents assembled from a small library:
- V segments (10): input encoding modules (how to read from medium)
- D segments (10): processing modules (compression strategies)
- J segments (10): output modules (how to write to medium)
- 10 × 10 × 10 = 1000 agent configurations from only 30 modules

Agent configuration SELECTED per-input (like antibody selection for antigens). This is MoE at the
module level — exponentially more combinations than standard MoE with linearly many parameters.

**Additional immune principles:**
- Affinity maturation → fine-tuning: small random perturbations to selected agent improve fit
- Clonal expansion → adaptive compute: well-matching agents get amplified in the medium
- Negative selection → safety: agents that match "self" (training distribution) too specifically are suppressed to prevent overfitting

**Potential upgrade to Sutra v0.2:** Replace fixed shared-weight agents with V(D)J compositional agents.
Test: does compositional assembly outperform shared-weight or independent-weight agents at matched params?
This could be its own probe (Probe G).

### Statistical Mechanics of Training: Phase Transitions and Grokking

Research findings (2025-2026):
- Grokking is a FIRST-ORDER phase transition (discontinuous jump from memorization to generalization)
- Critical exponents exist: exact analytic expressions for grokking probability and time distribution
- Singular Learning Theory (SLT) explains: properly regularized networks exhibit sharp complexity phase
  transition where complexity rises during memorization, then FALLS as network discovers simpler patterns
- Complexity dynamics: the LOCAL LEARNING COEFFICIENT (LLC) correlates linearly with compressibility

**Connection to Sutra thesis**: If compression = intelligence, and grokking = compression phase transition,
then Sutra's architecture should be designed to MAXIMIZE grokking likelihood. This means:
1. Strong regularization (MDL-style, not just L2) to push toward simpler representations
2. Architecture that supports sharp phase transitions (discrete state changes, not just smooth gradients)
3. Training schedule that encourages exploration before exploitation (high temp → low temp, simulated annealing)

**Testable prediction**: A model trained with MDL-style objectives should grok FASTER than one trained
with standard CE, because MDL explicitly penalizes complexity. This is an extension of Probe A.

### Adaptive Segmenter Prototype Results

Gumbel-Softmax segmenter confirmed:
- Differentiable: gradients flow through discrete boundary decisions
- Temperature controls sharpness: T=5 (soft) → T=0.1 (hard)
- 66K parameters — negligible overhead
- Trainable end-to-end with rest of model

BUT: Probe E analysis shows tokenization may not be the highest-value component.
Whitespace (1.48 BPB) already beats theoretical word-level entropy (1.82 BPB).
The real gains come from PROCESSING, not SPLITTING.

**Decision**: Adaptive segmenter is a nice-to-have, not critical path. Focus probes on
processing architecture (stigmergic, variable depth, working memory) first.

---

## Codex Hard Challenge of Sutra v0.1 — 4/10 (2026-03-19)

### The Core Criticism (ACCEPTED)
"You are spending the entire complexity budget on control machinery before you have shown
a base language-modeling primitive that is competitive."

Five components stacked = five hard problems at once. None individually proven. This is overengineering.

### Valid Criticisms (ACCEPTED, updating design):
1. **Over-squashing**: local message passing compresses exponentially many signals into fixed-width
   states. This IS the GNN over-squashing problem. Hierarchy helps but doesn't eliminate it.
2. **Multi-scale isn't free**: language isn't a clean tree. Bad coarse summary destroys fine-scale info.
3. **Predictive coding ≈ backprop**: either standard backprop through cross-scale links (which is what
   we'd actually do) or a slower approximation. Not a new learning principle.
4. **Energy-based generation has no clear advantage NOW**: AR+reranking at matched compute may match it.
5. **"Provably suboptimal" claims are asserted, not derived**: need actual proofs.

### Invalid / Debatable:
1. "Content-addressable retrieval needs global attention" — SSMs (Mamba) achieve competitive perplexity
   with NO content-addressable retrieval. The question is HOW MUCH global attention is needed, not WHETHER.
2. "No theorem that language admits scale-separable factorization" — true, but empirical evidence
   (MEGABYTE, HM-RNN) shows multi-scale helps even without a theorem.

### The 10/10 Criterion (Codex):
ONE simple core mechanism that:
- Beats a matched 10M-100M transformer on perplexity
- Beats it on long-range/state-tracking probes
- Preserves O(n) scaling
- Works with ordinary training
- Does NOT need hidden global attention

### REDESIGNED MVP: Sutra v0.2-MVP

Based on Codex feedback, strip to the MINIMUM that tests the core hypothesis:

```
BYTE INPUT
    |
FIXED-WINDOW CHUNKING (no adaptive segmentation)
    |
LOCAL COMPRESSOR (shared weights, processes one chunk)
    |
CHUNK-LEVEL RECURRENT MESSAGE PASSING
    - Each chunk reads from neighboring chunk summaries
    - Multiple rounds of message passing
    - PLUS: tiny global scratchpad (8-16 memory tokens)
    |
AUTOREGRESSIVE PREDICTION (standard next-byte loss)
    |
OPTIONAL: ADAPTIVE NUMBER OF MESSAGE-PASSING ROUNDS
```

**What changed from v0.1:**
- REMOVED: multi-scale (test two-scale only: bytes + chunks)
- REMOVED: energy-based generation (standard AR)
- REMOVED: adaptive segmentation (fixed windows)
- REMOVED: cross-scale predictive coding (standard backprop)
- ADDED: tiny global scratchpad (Codex's compromise for long-range)
- SIMPLIFIED: one mechanism to test — local compression + message passing

**This is essentially MEGABYTE but with message-passing between chunks instead of a global
patch-level transformer.** The hypothesis: message passing (O(n)) can replace the global
transformer (O(n²)) at the chunk level with minimal quality loss.

**Probe F tests exactly this.** If Probe F shows stigmergic ≈ transformer, we have the core.
If Probe F fails, we know local-only doesn't work and need the global scratchpad.

### Immediate Action Plan (based on ALL feedback so far):

1. WAIT for Probe A (compression↔capability) and Probe F (stigmergic) results
2. If Probe F fails: implement v0.2-MVP WITH global scratchpad
3. If Probe F succeeds: implement v0.2-MVP WITHOUT scratchpad (pure local)
4. Run Probes B (depth) and C (memory) to inform depth and memory components
5. Build v0.2-MVP at 10-50M params
6. If v0.2-MVP matches transformer baseline on perplexity: SCALE UP
7. If v0.2-MVP fails: analyze WHY, iterate

**The goal is ONE simple mechanism that works, not five coupled mechanisms that might.**

### Core Philosophical Insight: Structure-Matching (2026-03-19)

**Key realization**: Biology is optimized by billions of years of evolution under HARD physical
constraints. Every biological intelligent system has an architecture that MIRRORS the structure
of the problems it solves:
- Brain hierarchy mirrors perceptual hierarchy
- Immune system combinatorial diversity mirrors pathogen diversity
- Ant colony pheromone trails mirror spatial foraging structure

Current AI does the OPPOSITE: builds generic architectures and hopes they discover domain
structure during training. This is like building a car before studying roads.

**Sutra's real question**: What is the STRUCTURE of language and reasoning, and what architecture
naturally MIRRORS that structure? Not "what's the best generic architecture" but "what shape
should the computation be to match the shape of the problem?"

**Language structure (what we know):**
1. Hierarchical: characters < words < phrases < sentences < paragraphs < documents
2. Compositional: meaning of whole = function of meaning of parts + structure
3. Sequential with long-range: mostly local dependencies, occasional long-range
4. Variable complexity: some tokens trivial to predict, others require deep reasoning
5. Multi-modal internally: code ≠ prose ≠ dialogue ≠ argument (different structures)

**Architecture should match:**
1. Multi-scale processing (matches hierarchy)
2. Compositional operations (matches compositionality)
3. Mostly local + sparse long-range (matches dependency structure)
4. Variable depth (matches complexity variation)
5. Content-dependent routing (matches internal multi-modality)

**This is not a new observation but it's the RIGHT framing.** We're not looking for "the best
attention replacement." We're looking for "the computation that mirrors language structure."

### Theoretical Insight: Phase Synchronization as O(n) Long-Range Communication

The over-squashing problem (Codex critique): local message passing compresses exponentially many
distant signals into fixed-width states. The brain faces the same constraint (fixed-width neurons).

Brain's solutions: (1) hierarchical routing, (2) thalamic gating, (3) oscillatory synchronization.

**Option 3 is novel for AI:** Gamma oscillations create "virtual wires" between distant neurons
through phase locking. No physical connection needed.

**Application to Sutra:** Each chunk representation is COMPLEX-VALUED (magnitude + phase).
Chunks that need to communicate synchronize their phases. Information flows preferentially
between phase-aligned chunks regardless of distance.

Mathematically: attention weight w_ij = |cos(phase_i - phase_j)| (phase alignment).
Phase is learned/evolved during message passing. Chunks with similar content develop similar
phases automatically. This is content-based routing WITHOUT quadratic attention.

Complexity: O(n) if implemented as:
1. Each chunk computes its phase from content (local operation)
2. A global phase signal is broadcast (O(n) computation: just an average or FFT)
3. Each chunk reads the global signal and adjusts (local operation)

This is DIFFERENT from both attention (O(n²) pairwise) and scratchpad (fixed slots).
It's O(n) broadcast-based routing. Like how radio works: everyone transmits on different
frequencies, and you tune to the channel you want.

**Testable prediction:** Complex-valued representations with phase-based routing should
outperform real-valued + scratchpad for long-range dependencies at matched parameters.
This could be Probe H.

NOTE: Mamba-3 already showed gains from complex-valued SSMs. This might be WHY.

**UPDATE: KILLED BY CODEX (2/10 as attention replacement, 5/10 as hybrid auxiliary).**
Phase sync IS linear attention with rank-2 feature map: cos(θ_i-θ_j) = cos(θ_i)cos(θ_j) + sin(θ_i)sin(θ_j).
Not novel. Creates aliasing when multiple bindings interfere. Low capacity: O(r*d_v) vs O(n*d_v) for attention.
Language needs sparse, exact retrieval (copy names, match brackets, bind variables) — global sketch superposes.
**Keep as potential hybrid bias. Not the core mechanism.**

### Decision Tree: What We Build Based on Probe Results

```
Probe F (stigmergic)?
├── ratio < 1.5: Local works → v0.2 WITHOUT scratchpad → add phase sync → scale up
└── ratio > 1.5: Local insufficient → v0.2 WITH scratchpad → if still fails: hybrid

Probe A (compression)?
├── r > 0.5: MDL-style training, architecture maximizes compression
└── r < 0.5: Standard CE, focus on architecture not objective

Probe B (depth)?
├── adaptive >> fixed on hard tasks: include PonderNet-style halting
└── adaptive ≈ fixed: use fixed depth (simpler)

Probe C (memory)?
├── external memory best: Sutra needs explicit key-value store
├── transformer best: attention IS the memory mechanism
└── all fail >5 vars: training problem, need curriculum
```

### Meta-Question: Why Do All Efficient Attention Replacements Become Hybrids?

Longformer, BigBird, Performer, Mamba, RWKV, Hyena, RetNet — all tried O(n).
None REPLACED transformers. All became hybrids. Why?

Answer: attention isn't just computation — it's CONTENT-ADDRESSABLE MEMORY.
Every O(n) replacement loses content-addressing. The real question:
what is the MINIMUM mechanism for content-addressable routing?

Phase sync = broadcast-based content routing (O(n)).
Sufficient for: semantic similarity connections (noun-verb, topic coherence).
Insufficient for: specific binding (pronoun resolution, variable tracking).

Probe C will tell us if specific binding requires attention or can be done otherwise.

### Prototype Results Summary (CPU experiments, 2026-03-19)

| Prototype | Result | Signal |
|-----------|--------|--------|
| Phase sync | Phases self-organize (diff=1.18, >>0.5) | POSITIVE |
| V(D)J routing | Routes specialize strongly (KL=11.95) | POSITIVE |
| Phase+V(D)J combined | Routes specialize by token, perf ≈ baseline on random data | NEUTRAL |
| Gumbel segmenter | Differentiable, gradients flow | POSITIVE (but deprioritized) |

**Pattern**: Mechanisms WORK (routes specialize, phases organize) but don't show
performance gains on trivial data. Need structured data where routing MATTERS.
This is expected — you don't need content-dependent processing for random digits.
The real test is on language with genuine structure (the v0.2-MVP test).

### MQAR Retrieval Test (CPU, 2026-03-19)

5 KV pairs, 2 queries, 100 epochs, 5K train samples:
- Transformer (1.2M params): 2% accuracy — WEAK signal, beginning to learn at epoch 60
- GRU (314K params): 0% accuracy — ZERO signal, completely flat

**Key finding**: Attention provides a weak but REAL signal for retrieval that recurrence lacks.
Neither model solved MQAR in this budget, but the transformer is directionally learning.

**Implication for Sutra**: Pure recurrence (stigmergic message passing without ANY retrieval
mechanism) may fundamentally lack the ability to do content-addressable lookup. This supports
Codex's recommendation for a scratchpad or sparse attention.

**Caveat**: Models are tiny, training short. MQAR with 5 KV pairs SHOULD be solvable by both
with enough training. The question is whether recurrence EVER catches up. Need longer run.

**Action**: Run MQAR with 500 epochs and compare. If transformer solves it and GRU doesn't
even after 500 epochs, attention IS fundamentally required for retrieval.

### Paper Deep Dives: MEGABYTE + PonderNet (2026-03-19)

**MEGABYTE design parameters for Sutra:**
- Patch size 8 for text (robust 48-768 for other modalities)
- Local model: 12-24 layers, dim 768-1024 (processes within patch)
- Global model: 24-32 layers, dim 1024-2560 (processes across patches)
- BOTH levels critical: removing either degrades BPB by ~0.6 (from 0.687)
- Per-patch FFN = P× larger FFN for same FLOPs (efficiency win)
- PG-19: 0.908 BPB vs transformer 1.057 (14% improvement)
- End-of-patch degradation exists (strided inference helps at 2x cost)

**PonderNet design parameters for adaptive depth:**
- Halting: geometric distribution, λ_n = sigmoid(linear(h_n))
- Loss: L_recon + 0.01 * KL(p_halt || geometric(λ_p))
- λ_p ∈ [0.1, 0.9] robust (0.1 = expect ~10 steps is safe default)
- bAbI: 6.1x fewer steps than Universal Transformer
- Extrapolation: trained on 1-48 elements, works on 49-96
- Key: prediction from ACTUAL halting step, not weighted average

**Integration into Sutra v0.2-MVP:**
1. Patch size = 8 bytes (MEGABYTE validated)
2. Local model within patch (small MLP/transformer, 2-4 layers)
3. Global model between patches: THIS is where our innovation goes
   - MEGABYTE uses global transformer (O(n²/P²))
   - Sutra tests message passing (O(n/P)) vs small scratchpad
4. Add PonderNet halting to message passing rounds (adaptive depth)
5. Train on 60MB real corpus (code + wiki + prose)

### Updated Probe Priority

Given the cross-domain insights, I'm adding a new probe that's potentially more important than any existing one:

**Probe F: Stigmergic Text Modeling** (NEW — highest priority)
- Question: Can a system of locally-interacting agents (no global attention) learn to model text?
- Design: 10M params. Instead of one model, create N=16 small "agents" each with ~600K params.
  Each agent sees a 32-token window. Agents write compressed features to a shared 1D "medium."
  Agents read from their local neighborhood on the medium. Multiple passes (like message passing).
  Compare perplexity against a single 10M-param transformer on same data.
- Controls: Random medium (agents write but read random positions), single-agent baseline.
- Kill: If stigmergic model perplexity is >2x the transformer baseline, local interaction is insufficient.
- Time: ~4 hours
