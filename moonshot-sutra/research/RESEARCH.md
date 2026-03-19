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

**Status**: READY TO IMPLEMENT AND RUN.
