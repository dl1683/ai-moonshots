# Sutra Scratchpad — Raw Thinking, Ideas, Debates

This is the working document for unvalidated ideas, cross-domain intuitions,
and Codex back-and-forth. Nothing here is confirmed — it's the whiteboard.

Ideas graduate to RESEARCH.md only after Codex validation + empirical signal.

---

## Active Ideas (need Codex debate)

### Idea 1: Boosting Applied to Processing Depth

Each message passing round focuses on what PREVIOUS rounds got wrong.
Round 1: easy local patterns. Round 2: fix Round 1's errors. Round 3: residual.

Implementation: feed the prediction error from round N as explicit input to round N+1.
This IS predictive coding but derived from boosting — concrete and implementable.

**Questions for Codex**:
- Is this actually different from residual connections? (residual = implicit boosting)
- Would explicit error signals help or just add noise?
- Has anyone done "boosted message passing" in GNN literature?
- What's the simplest experiment to test this?

### Idea 2: Episodic Memory via Activation Caching

During training, cache the medium states for each input. During inference, retrieve
similar cached states via sparse retrieval. Neural KNN — the model retrieves its
own past computations for similar inputs.

Connects to: neuroscience episodic memory, RAG but internal, prototype networks.

**Questions for Codex**:
- How big would the cache need to be? Memory cost?
- Is this just a learned nearest-neighbor classifier in disguise?
- How does this interact with gradient-based learning? Does the cache need updating?
- What's the simplest experiment?

### Idea 3: Sparse Attention as "Appropriate Attention"

Insight from XGBoost: trees ignore uninformative features naturally. In language,
most tokens are uninformative for predicting any given target. Sparse retrieval
(k=4-32) isn't "degraded attention" — it's APPROPRIATE attention that filters noise.

**Questions for Codex**:
- Is there an information-theoretic argument for optimal k?
- Does mutual information between tokens follow a power law? (few high-MI pairs, many near-zero)
- If so, what k captures 90% of the mutual information?
- Could k be LEARNED per-input (easy inputs k=2, hard inputs k=32)?

### Idea 4: Multi-Round Message Passing as Implicit Ensemble

Each round sees the data differently (medium state changed). Multiple rounds =
implicit ensemble where each round contributes a different perspective.

Could add explicit diversity: dropout/noise on medium between rounds = Monte Carlo
message passing. Multiple stochastic passes averaged for final prediction.

**Questions for Codex**:
- Is there theory on when implicit ensembles match explicit ones?
- Would noise injection on the medium hurt optimization or help generalization?
- Connection to dropout training — is this just dropout with extra steps?

### Idea 5: PonderNet Halting as Native Calibration

When model halts after few rounds → "confident" (easy). Many rounds → "uncertain" (hard).
The halting distribution IS a confidence signal built into the architecture.

Train to ABSTAIN on hard questions rather than hallucinate. Use halting depth as
calibration signal. Native to Sutra — transformers need post-hoc calibration.

**Questions for Codex**:
- Does halting depth actually CORRELATE with prediction accuracy?
- Could we enforce this with an auxiliary loss (accuracy should increase with depth)?
- Is Probe B's collapse (always 1 step) evidence against this whole idea?

### Idea 6: Bias-Variance for Architecture Design

Sutra's strong local+sparse bias matches ~80% of language (why it wins on structured
tasks). The ~20% needing global reasoning causes sequential reasoning loss.

Need enough flexibility for the 20% without losing efficiency of the 80% bias.
Options: k=16-32, small attention layer every N rounds, hybrid routing.

**Questions for Codex**:
- Can we MEASURE what % of a real corpus is "local" vs "global"?
- Our dependency analysis showed 50% within 42 words — is that the right measure?
- What's the minimum global mechanism that handles the 20%?

### Idea 7: Self-Growing Architecture (Embryonic Development)

Patches start identical, specialize during training based on position and neighbor
signals. Like brain development — Broca's area specializes for language, V1 for vision.

**Questions for Codex**:
- Is there precedent for architectures that develop specialization during training?
- How is this different from just having different weights per layer?
- Could we use a regularization loss that ENCOURAGES specialization?

---

## Parked Ideas (interesting but not actionable now)

- DNA error correction → quantization-native (no signal at toy scale)
- Market consensus → adaptive depth (conceptually nice, no clear implementation advantage)
- Music motifs → compositional primitives (supports V(D)J but deferred)
- Ecological succession → training curriculum (nice metaphor, standard curriculum learning)

---

## Codex Debate Log

### Round 1 (2026-03-19)

**Codex verdict**: Focus on Ideas 3+6 (sparse k sweep + minimum global mechanism).
These are directly on Sutra's core thesis.

**Promoted to test**:
- Idea 3 (Priority 5): Sweep k={2,4,8,16,32,adaptive} on MQAR + BPB + long-range
- Idea 6 (Priority 5): Three variants — local-only vs local+sparse vs local+scratchpad
- Idea 5 (Priority 4): PonderNet calibration after min_rounds fix

**Killed/Parked**:
- Idea 1: "just iterative refinement, error signal destabilizes" → PARKED
- Idea 2: "cache contradicts edge thesis, ~1GB for 1M states" → PARKED
- Idea 4: "shared-weight rounds not diverse enough for real ensemble" → KILLED
- Idea 7: "highest upside but worst use of current compute" → PARKED

**Key Codex insight**: "Prove language = mostly local + tiny sparse global.
Identify the minimum global mechanism that closes the long-range gap."

**For Round 2**: Experimental designs below.

---

## Round 2 Proposal: Two Priority-5 Experiments

### Experiment A: Sparse-k Sweep (Idea 3)

**Question**: What k captures most of the performance of dense attention under
Sutra's message-passing backbone?

**Design**: Same v0.3 architecture, same data (2M bytes real corpus), same epochs.
Vary ONLY k in sparse retrieval: {0 (no retrieval), 2, 4, 8, 16, 32, N (full attention)}.
Also test adaptive-k: router outputs k per-patch via Gumbel-softmax.

**Measurements**: BPB on test set, MQAR accuracy (5 KV pairs), training time.
**Controls**: k=0 (message passing only) and k=N (full attention) are the bounds.
**Kill**: If k=4 already gets >90% of k=N performance, larger k adds nothing.
**Interesting**: If adaptive-k learns different k for different patches.

**Estimated compute**: 8 variants × ~30 min each = ~4 hours GPU.
Run after 10M test completes.

### Experiment B: Minimum Global Mechanism (Idea 6)

**Question**: What is the cheapest global mechanism that closes the long-range gap?

**Design**: v0.3 backbone, same data, same epochs. Three variants:
1. LOCAL ONLY: message passing, no sparse retrieval, no scratchpad
2. LOCAL + SPARSE: message passing + top-k retrieval (k=8)
3. LOCAL + SCRATCHPAD: message passing + 8-16 global memory tokens
4. LOCAL + SPARSE + SCRATCHPAD: everything
5. FULL ATTENTION baseline (transformer)

**Measurements**: BPB, MQAR, structured reasoning accuracy, long-range copy task.
**Controls**: Matched params across all variants.
**Kill**: If LOCAL+SPARSE matches FULL ATTENTION within 10%, scratchpad is unnecessary.
**Interesting**: Where does each variant fail? Which task type needs which mechanism?

**Estimated compute**: 5 variants × ~30 min each = ~2.5 hours GPU.

---

## Codex Round 2 Feedback (Integrated)

**Execution order**: A first (find right k), then B (with chosen k).

**Experiment A fixes**:
- Fixed k ONLY first. Adaptive-k is a separate follow-up.
- Match FLOPs not just epochs (k=N does more compute per step)
- Add tasks: MQAR difficulty sweep, passkey/needle, selective copy
- Add: 3 seeds, random-retrieval control, retrieved-distance histograms
- 2M pilot → rerun top 2-3 k values on 10M+ if differences are small

**Experiment B fixes**:
- Add variant: periodic global refresh (non-content-addressable) — tests if
  SPARSITY specifically matters or just "any global signal"
- Match params TWO ways: strict (shrink width) AND matched FLOPs
- Add tasks: passkey/needle, bracket matching, state tracking
- 3 seeds, 2 scales minimum for publishable
- Target claim: "sparse k=X recovers Y% of dense attention at Z% compute"

**STATUS**: Designs finalized. Run A pilot immediately after 10M completes.

---

## Deep Think: What Can Fungi Teach Us About Intelligence? (2026-03-19)

### The Biological Facts

Mycorrhizal networks and slime molds do things that SHOULD be impossible:

1. **Physarum stores memory in tube diameter.** When it finds food, the tubes leading
   to that food THICKEN. When a path is unproductive, tubes THIN and retract. The
   GEOMETRY of the network IS the memory. No separate memory storage — the network
   topology itself encodes what the organism has learned.

2. **Fungi solve optimization problems through flow dynamics.** Nutrient transport
   through the network follows local conservation laws at each junction. But these
   local laws COUPLE behavior across the whole network, achieving GLOBAL optimization
   through purely LOCAL rules. February 2025 research showed fungi use TRAVELLING
   WAVES for resource transport — pulses of nutrients that propagate through the network.

3. **Mycorrhizal networks have "hub and spoke" topology.** Not uniform — some nodes
   (hub trees, "mother trees") have many connections, most have few. Scale-free network.
   This topology allows EXACTLY SOLVABLE flow solutions. The network self-organizes
   to create this topology through growth and pruning.

4. **Slime molds learn WITHOUT neurons.** Physarum shows habituation (learns to ignore
   repeated harmless stimuli), retains learning through 1-month dormancy, and stores
   information in THREE ways simultaneously:
   - Chemical (absorbed substances = "circulating memory")
   - Electrical oscillations (frequency changes = short/long-term memory)
   - Network morphology (tube diameters = structural memory)

5. **Fungi make DECISIONS about trade.** Mycorrhizal fungi modulate which trees get
   resources based on the tree's contribution to the network. Trees that provide more
   carbon get more nutrients. This is a market-like mechanism without any central broker.

### What This Means for Sutra (Deep Original Thinking)

**Concept A: Memory IN the Network, Not Separate FROM It**

Current AI: model weights store knowledge, activations are temporary. Memory and
computation are SEPARATE — weights persist, activations are discarded after each forward pass.

Fungi: the network IS the memory. Tube diameters encode past experience. The "weights"
(tube widths) are continuously modified by the "activations" (nutrient flow).

**For Sutra**: What if the shared medium in our message passing wasn't just a temporary
communication buffer that resets each forward pass, but a PERSISTENT state that accumulates
across inputs? The medium state would encode what the model has "learned" from recent
context — like a continuously updated working memory that's part of the architecture,
not separate from it.

This is different from KV-cache (which stores past attention outputs) because the medium
would be PROCESSED between rounds, not just stored. It's a living, evolving memory that
gets refined through message passing, like how fungal tube diameters get refined through
nutrient flow.

**Implementation**: Between batches, carry forward a fraction of the medium state (exponential
moving average). Each batch starts from a "warm" medium, not zeros. Over time, the medium
develops persistent structure that reflects the training data distribution.

**Concept B: Geometry as Information (Tube Diameter = Weight)**

Fungi don't store information in a separate data structure. The PHYSICAL SHAPE of the
network encodes information. Thick tubes = important paths. Thin tubes = unimportant.

**For Sutra**: What if the sparse retrieval pattern itself was informative? Instead of
computing top-k from scratch each time, maintain a PERSISTENT routing table that records
which patches typically retrieve from which others. Over time, this table reflects the
statistical structure of the data — like how fungal networks grow thick connections
between frequently-exchanging nodes.

This is essentially a LEARNED sparse attention pattern that evolves during training.
Not random sparsity, not fixed sparsity, but GROWN sparsity that reflects the data.

**Concept C: Travelling Waves for Information Propagation**

February 2025 research: fungi use TRAVELLING WAVES — pulses that propagate through
the network — for nutrient transport. Not steady-state flow, but PULSES.

**For Sutra**: What if message passing used wave-like propagation instead of uniform
rounds? Each "wave" of information propagates outward from its source, carrying a
specific signal. Multiple waves from different sources can overlap and interact.

This is different from standard message passing (which updates ALL nodes simultaneously
each round). Wave propagation is SEQUENTIAL and DIRECTED — information travels FROM
sources TO destinations, like how sound waves propagate from a speaker to a listener.

**Implementation**: Instead of uniform rounds, process patches in order of their
"surprise" (prediction error). High-surprise patches generate waves first, which
propagate outward through the medium. Low-surprise patches receive waves but generate
little. This naturally allocates compute to where it's needed.

**Concept D: Multi-Modal Memory (Chemical + Electrical + Structural)**

Physarum stores information THREE ways simultaneously: chemical, electrical, structural.
Each modality has different timescales: electrical = fast (seconds), chemical = medium
(minutes), structural = slow (hours/days).

**For Sutra**: What if we had three memory timescales?
- FAST: the current hidden state (resets each input) — like electrical oscillations
- MEDIUM: a running average of recent medium states (persists across a few inputs) —
  like chemical circulation
- SLOW: the learned weights themselves (persist across all inputs) — like tube structure

This is essentially: working memory (fast) + episodic buffer (medium) + long-term memory (slow).
Current transformers only have fast + slow. The MEDIUM timescale is missing.

The medium timescale could capture things like: "we've been processing Python code for
the last 50 inputs, so prime the medium for Python patterns." This is context-dependent
priming — exactly what the brain does when you're in a specific environment.

**Concept E: Market-Based Resource Allocation (Fungal Trading)**

Mycorrhizal fungi trade resources based on contribution. Trees that give more carbon
get more nutrients. This creates INCENTIVES for cooperation — a market without a broker.

**For Sutra**: The message passing medium could implement a TOKEN ECONOMY where patches
"pay" for retrieval with the quality of their own contributions. Patches that produce
good summaries (low prediction error) get priority retrieval access. Patches that produce
garbage get deprioritized.

This would naturally allocate retrieval bandwidth to the most competent patches,
like how markets allocate capital to the most productive companies.

### Priority Assessment

These concepts range from immediately testable to deeply speculative:

| Concept | Testability | Novelty | Priority |
|---------|------------|---------|----------|
| A: Persistent medium | HIGH (simple EMA) | MEDIUM | **Test soon** |
| B: Grown sparsity | MEDIUM (need routing table) | HIGH | Test after k-sweep |
| C: Wave propagation | LOW (complex implementation) | HIGH | Defer |
| D: Multi-timescale memory | MEDIUM (add EMA buffer) | MEDIUM | **Test soon** |
| E: Token economy | LOW (complex incentive design) | HIGH | Defer |

**For Codex Round 3**: Challenge A and D — these are testable and could be added to
the v0.3 architecture with minimal changes.

### Codex Round 3 Verdict

A (persistent medium): "shared leaky hidden state" — not novel, gradient vanishes, batch-order
dependent. PARKED.

D (multi-timescale): "A plus better framing" — fast weights exist, EMA buffer is blurry priming.
PARKED.

**SURPRISE: B (grown sparsity) is the BEST BET.** Changes routing, compute, and inductive bias
in measurable way. More novel than A/D, more practical than C/E.

**Grown sparsity = routing table evolves like fungal tubes:**
- Connections that carry useful info → thicken (higher retrieval priority)
- Connections that carry garbage → atrophy (pruned from routing table)
- Over training, the sparse pattern GROWS to reflect true text dependency structure
- At inference: near-free retrieval (just look up the routing table)
- The routing pattern IS the model's learned understanding of language structure

This connects to: geometry AS information (fungi), attention pattern AS representation
(the connections are more important than the nodes), and structure matching (the
grown routing pattern mirrors the data's dependency structure).

**Implementation sketch**:
- Maintain a soft routing matrix R of size (max_patches × max_patches)
- Each forward pass: R[i,j] += alpha * (was patch j useful when patch i retrieved it?)
- R[i,j] *= decay (tubes atrophy without reinforcement)
- Sparse retrieval uses top-k from R[i,:] instead of computing scores from scratch
- R is a LEARNED persistent structure, not a temporary computation

**Status**: Queue for Experiment C (after A and B from Round 2).

**"Was retrieval useful?" signal — SOLVED via gradient magnitude:**
After backprop, gradient magnitude through each retrieval connection tells us
how much that connection contributed to loss reduction. Accumulate into routing
table: high gradient → thicken (increase priority), low gradient → thin (atrophy).
This IS how fungal tubes work: high nutrient flow → thicken, low flow → atrophy.
Gradient flow = nutrient flow. Implementation: register backward hook on retrieval
connections, accumulate abs(grad) into routing table with EMA decay.

---

## Deep Think: Edges > Nodes (2026-03-19)

### The Observation

In graph theory, social networks, neural networks, mycorrhizal networks — the EDGE
structure often contains more information than the node features.

"The bank by the river was muddy." Understanding isn't in "bank" (ambiguous) — it's
in the CONNECTION between "bank" and "river" that disambiguates. The relationship IS
the meaning. The edge IS the understanding.

### Current AI Treats Edges as Disposable

In transformers:
- Node features (embeddings, hidden states) = primary, persisted, analyzed
- Edge features (attention weights) = computed on the fly, immediately discarded
- Nobody saves attention patterns. Nobody treats them as the model's output.

This is backwards if edges carry more information than nodes.

### What If We Flipped This?

**Edge-primary architecture**: Instead of updating NODE representations and computing
edges temporarily, update EDGE representations and derive nodes from them.

Concrete version for Sutra:
- The message passing medium stores EDGE weights between patches (not patch features)
- Each patch's representation is DERIVED from its edges: "I am the node connected to
  patches 3, 7, and 12 with weights 0.8, 0.3, 0.6"
- Message passing updates the EDGES, not the nodes
- Prediction uses the edge structure to compose node features on-the-fly

This is actually how GNNs can work — edge-centric message passing where edges are the
primary objects and nodes are just aggregation points.

### Practical Benefits (not just theory)

1. **Interpretability for free**: The edge structure IS the model's understanding of
   the text. You can literally READ it: "the model thinks position 5 is strongly
   connected to position 23" = "the model thinks these words are related."

2. **Compression**: Edges are SPARSE. A 256-token sequence has 256 nodes but only
   ~256×k = 1024 meaningful edges (with k=4). Storing 1024 edges is cheaper than
   storing 256 d-dimensional vectors.

3. **Compositionality**: Edges compose naturally. If A→B and B→C, there's an implicit
   A→C path. This is how transitive reasoning works — you don't need to represent it
   explicitly, it emerges from the edge structure.

4. **Grown sparsity is natural**: In edge-primary, the routing table IS the model's
   state. Growing and pruning edges IS learning. No separate mechanism needed.

5. **Transfer**: The edge PATTERNS (not specific edges) transfer across inputs.
   "Adjective always connects to nearest noun" is an edge pattern, not a node pattern.
   Learning edge patterns = learning linguistic structure directly.

### Connection to Graph Neural Networks

This isn't totally new — edge-conditioned GNNs exist. But applying edge-primary
processing to LANGUAGE (where the graph changes per input) would be novel.

The key difference from attention: attention computes edges FROM nodes (Q×K^T).
Edge-primary STARTS with edges and derives nodes. The causal arrow is reversed.

### Is This Practical or Just Elegant?

PRACTICAL test: take our v0.3 architecture, but instead of storing patch features
in the medium, store EDGE WEIGHTS between patches. Each patch's feature is computed
by aggregating its incoming edge weights × neighbor features. Compare BPB.

If edge-primary matches or beats node-primary at same params, it validates the
principle AND gives us interpretability AND enables grown sparsity naturally.

### Things That Don't Need to Be Revolutionary to Be Useful

While the edge-primary idea is conceptually interesting, there are SIMPLER wins:

1. **Just increasing k from 4 to 16** could close the sequential reasoning gap.
   No fancy concepts needed. Just more retrieval bandwidth.

2. **Adding a simple GRU layer** after message passing could give sequential
   processing capability. Old tech, proven, straightforward.

3. **Better training data** (synthetic reasoning chains from local LLMs) would
   help more than any architectural change at this scale.

4. **Gradient accumulation** to increase effective batch size would stabilize training.

5. **Learning rate warmup** was missing from some of our experiments — adding it
   consistently would improve all results.

Not everything needs to be a paradigm shift. Sometimes you need to tighten the
bolts before reinventing the engine.

---

## Critical: Parameter Mismatch in Experiments (2026-03-19)

Audit: Sutra v0.3 has 5.5x FEWER params than transformer at same dim=256:
- Sutra (dim=256): 1.2M params
- Transformer (8 layers): 6.5M params

Previous "wins" partly from natural regularization (smaller → less overfitting).
BUT: matched-param test (112K vs 100K) still showed 34% Sutra win → architecture IS better.

**Fix**: Sutra dim=608 ≈ 6.2M params matches transformer. Run matched-param at 10M scale.
Current test still valuable: if 1.2M Sutra ≈ 6.5M transformer, that's 5.5x efficiency.

---

## Quick Ideas (Not Full Concepts, Just Notes)

**Patch size controls TWO things simultaneously:**
1. Number of message passing opportunities (more patches = more rounds of info exchange)
2. Bottleneck width (more patches = more summaries = wider bottleneck = less compression)

These are CONFOUNDED. Patch=4 beating patch=16 could be either or both.
Disentangle with: same patch size, vary summary dimension. If larger summary
helps → bottleneck matters. If not → it's the rounds.

---

## Alternative Architecture Concepts (Wild Exploration)

### Graph Language Model
Instead of processing SEQUENCES, learn the dependency GRAPH and process it.
Model simultaneously discovers text structure AND processes it.
The graph IS the understanding. Prototype running on CPU.

### Text as Physics Field
Treat text as a continuous field, not discrete tokens. Apply field theory:
Fourier analysis, Green's functions, renormalization group.
Connects to CTI universal law (alpha = critical exponent?).
Diffusion LLMs are a primitive version of this.

### VSA-Based Message Passing (Structured Composition)
Vector Symbolic Architectures: bind = elementwise multiply, bundle = add, unbind = inverse.
Can represent ANY compositional structure as a single vector, decomposable.
What if message passing used VSA operations instead of MLPs?
Each message = structured composition, not black-box transformation.
"dog(agent) + chase(action) + cat(patient)" = one decomposable vector.
NOVEL: nobody has used VSA as message-passing primitive for language modeling.
Connects to category theory (2025 paper: VSA ops = right Kan extensions).

### GRU Inside Patches — VALIDATED
GRU patch beats MLP patch by 8.7% on sequential arithmetic (0.367 vs 0.402).
Adding recurrence WITHIN patches gives sequential processing capability.
For v0.4: swap MLP patch processor for GRU. Closes sequential reasoning gap
while keeping message passing + sparse retrieval for inter-patch.

---

## Sutra v0.4 Architecture (All Validated Components Combined)

IF the fair ablation validates v0.3 core, upgrade to v0.4:

```
BYTE INPUT
    |
4-BYTE PATCHES (validated: all patch sizes beat transformer)
    |
GRU PATCH PROCESSOR (validated: 8.7% better than MLP on sequential)
    |
PATCH SUMMARIES (mean pool)
    |
GROWN SPARSE RETRIEVAL (Codex top pick, gradient-driven routing table)
    + Local message passing (validated: 28-46% structural advantage)
    + Routing table evolves: high-gradient connections thicken, low atrophy
    |
ADAPTIVE DEPTH with min_rounds=2 (PonderNet, collapse fixed)
    |
BROADCAST BACK + PREDICT (standard CE loss)
```

**Novel contributions in v0.4:**
1. Grown sparsity from fungal biology (routing table = fungal tube network)
2. GRU+message passing hybrid (recurrence within patches, message passing between)
3. Gradient-driven routing evolution (gradient flow = nutrient flow)

**Standard components (not novel but validated):**
4. Byte-level patching (from MEGABYTE)
5. PonderNet adaptive depth (from DeepMind)
6. Sparse top-k retrieval (from efficient attention literature)

### Information-Weighted Loss (Focal Loss for Language)
Weight loss by per-position information content: surprising bytes get higher
weight, predictable bytes get lower. Like MP3 removes unhearable frequencies.
Implementation: focal loss — weight = (CE_per_position)^gamma where gamma>0
makes the model focus on what it finds HARDEST to predict.
This is adaptive resource allocation applied to the LOSS, not compute.
Simple to implement, zero architecture change, potentially big impact.

### THEOREM DRAFT: Information-Optimal Compute Allocation

**Statement**: For text source X with MI profile I(d) ∝ d^(-alpha) for d > 1:
- Optimal local window d_0 = argmin_d {cumulative_MI(1..d) >= 0.75 * total_MI}
- Optimal sparse k ≈ integral_{d_0}^{inf} I(d) dd / mean(I(d) for d > d_0)
- Architecture matching this allocation is information-optimal

**From measurement**: I(d) ≈ 1.12 * d^(-1.2). d_0 ≈ 5 bytes. k ≈ 4-8.
Predicts EXACTLY our v0.4 configuration.

**Testable predictions**:
1. Code (more structure) should have steeper alpha → smaller d_0, larger k
2. Stories (more narrative) should have shallower alpha → larger d_0, smaller k
3. Optimal architecture config should CHANGE with the text domain
4. Architecture that matches MI profile should beat one that doesn't

**QUANTIFIED (2026-03-19):**
- Fit: I(d) = 0.637 * d^(-0.547), R^2 = 0.900
- d_0 = 5 bytes (75% cumulative MI)
- Predicted optimal k = 5 (matches v0.4 k=4-8)
- Per-domain alphas: mixed=0.757, stories=0.689, code=0.565
- Code has SLOWEST decay = most long-range deps (as predicted)

**Codex R4 verdict: 5/10 as written, 7/10 if reframed properly.**
NOT novel — Li (1989), Ebeling (1994) already measured this.
Single power law DOESN'T FIT — tail is much heavier (broken power law or stretched exp).
My prediction direction was WRONG (code shallower, not steeper as I claimed).
DO NOT CLAIM "theorem predicts v0.4 exactly." That's overclaim.

REFRAME AS: "Text shows local MI core + weak long-range tail. Domain-dependent shape.
Architectures matching measured dependency geometry perform better at fixed FLOPs."
This is credible and 7/10 if backed by ablation data.

### Theoretical Explanation: Why GRU+MsgPass Works

Hypothesis: language has TWO kinds of dependency:
- TEMPORAL: within-word character ordering (sequential, local)
- SPATIAL: between-word semantic relationships (relational, potentially distant)

GRU processes temporal. Message passing processes spatial.
Attention treats EVERYTHING as spatial. RNN treats everything as temporal.
GRU+MsgPass decomposes into the right mechanism for each dependency type.

Information-theoretic formalization:
I(x_i; x_j) = I_temporal(x_i; x_j) + I_spatial(x_i; x_j)
Using separate mechanisms for each component = more parameter-efficient.
Like how FFT decomposes a signal into frequency components processed separately.

This is WHY v0.4 beats transformer: it processes language structure with
mechanism-structure alignment, not a one-size-fits-all approach.

### What Makes v0.4 GENUINELY Novel vs Existing Hybrids

Existing (Jamba, Falcon-H1): interleave attention + SSM at SAME scale.

v0.4: different mechanisms at DIFFERENT SCALES:
- Level 1 (bytes): GRU sequential processing within patches
- Level 2 (words/phrases): message passing between patches
- Level 3 (sentences/documents): sparse content-addressable retrieval

This IS structure-matching: the architecture mirrors language's hierarchical
structure. No existing hybrid does multi-scale mechanism selection.

---

## Strategic Scenario Planning: What Happens After Production Training?

### Scenario A: Sutra WINS (BPB < Pythia-410M ~0.92)
If 475M Sutra beats 410M Pythia on BPB with 1.6B tokens (vs Pythia's 300B tokens):
- **Narrative**: "187x less training data, fewer params, better BPB. Architecture matters."
- **Next**: Scale to 1B-4B immediately. Download more data (RedPajama, FineWeb).
- **Paper**: Submit to COLM/NeurIPS with architecture + MI analysis + ablations.
- **Risk**: BPB doesn't translate to downstream task performance. Need to also test generation.

### Scenario B: Sutra COMPETITIVE (BPB within 10% of Pythia)
If 475M Sutra gets ~1.0 BPB (within 10% of Pythia's ~0.92):
- **Narrative**: "Matches Pythia with 187x less data. With equal data, would win."
- **Next**: Train longer (use all MiniPile, maybe download more data). The architecture
  works, it just needs more data to shine.
- **Improvement**: Add mixed precision (2x speed), larger batch, longer training.
- **Risk**: "Matches with less data" is weaker claim than "beats outright."

### Scenario C: Sutra BEHIND (BPB 1.2-2.0, significantly worse)
If 475M Sutra can't get below 1.2 BPB:
- **Diagnosis**: Architecture doesn't scale well from toy to production.
- **Possible causes**:
  1. Message passing too slow to propagate info in long sequences
  2. Sparse retrieval k=16 insufficient at this scale
  3. GRU patch processing bottlenecks sequential computation
  4. Byte-level modeling needs much more data than token-level
- **Action**: Add more attention layers (hybrid), increase k, switch to token-level.
- **Risk**: May need fundamental architecture changes, not just tuning.

### Scenario D: Training FAILS (diverges, OOM, extremely slow)
If training crashes or is impractically slow:
- **Diagnosis**: Architecture doesn't scale to 475M on GPU.
- **Action**: Scale down to 170M (dim=3072), fix numerical issues, add gradient scaling.
- **Quick fix**: Mixed precision + gradient accumulation + smaller batch.

### Improvements to Prepare Regardless of Outcome

1. **Mixed precision training script** (AMP/bf16) — 2x speed for next run
2. **lm-eval custom model wrapper** — for standard benchmark comparison
3. **Token-level variant** — Sutra with BPE tokenizer as alternative to byte-level
4. **More training data** — download RedPajama-sample or FineWeb-Edu subset
5. **Grown sparsity** — implement gradient-driven routing table for v0.5

### Questions for Codex

1. What's the MOST LIKELY outcome given our toy-scale results?
2. If byte-level BPB doesn't translate to downstream performance, what's plan B?
3. Should we have a token-level Sutra ready as a fallback?
4. What's the minimum training compute to be competitive with Pythia-410M?
5. How do we handle the "less data" narrative honestly? (Our 1.6B tokens vs 300B)

---

## SYSTEMATIC AI PIPELINE BREAKDOWN

Every model, from GPT-4 to a 100K param toy, goes through the SAME abstract stages.
Improving ANY stage improves the whole. Understanding each stage lets us target
the weakest link instead of randomly trying things.

### Stage 1: INPUT REPRESENTATION
**What it does**: Convert raw text into numbers the model can process.
**Current approaches**: BPE (GPT/Llama), SentencePiece (T5), byte-level (ByT5/MEGABYTE)
**Sutra v0.4**: Raw bytes (vocab=256)
**What works**: BPE balances vocab size vs sequence length well
**What fails**: Fixed tokenization loses information at boundaries ("New" + "York")
**Theoretical best**: Learned, adaptive tokenization (model decides its own chunking)
**Our opportunity**: Token-level branch with BPE, OR keep bytes with our patch structure
  as a LEARNED tokenizer that groups bytes into concepts

### Stage 2: INITIAL ENCODING
**What it does**: Map input IDs to rich feature vectors (embeddings + position)
**Current approaches**: Learned embeddings + RoPE/ALiBi/sinusoidal position encoding
**Sutra v0.4**: Learned byte embeddings + patch-level GRU
**What works**: RoPE (rotation) gives good length generalization
**What fails**: Absolute position embeddings don't generalize to new lengths
**Our opportunity**: The GRU within patches gives us RELATIVE position encoding for
  free — GRU state naturally tracks position within each patch. For between-patch
  position, we could add RoPE to the message passing.

### Stage 3: LOCAL FEATURE EXTRACTION
**What it does**: Build features from nearby tokens (n-gram patterns, local syntax)
**Current approaches**: First few attention layers, convolutions, local attention
**Sutra v0.4**: GRU within patches (processes 4 bytes sequentially)
**What works**: Convolutions are fast and effective for local patterns
**What fails**: Fixed receptive field can't adapt to content
**Our opportunity**: GRU gives adaptive local processing. Could ADD convolutions
  (Canon layers: +1-2% factuality) as a cheap complement.

### Stage 4: CONTEXT INTEGRATION
**What it does**: Combine information across positions (the "understanding" step)
**Current approaches**: Self-attention (O(n^2)), SSMs (O(n)), linear attention
**Sutra v0.4**: Message passing between patches (O(n) with window)
**What works**: Full attention for complex reasoning, SSMs for efficiency
**What fails**: Attention is too expensive for long contexts, SSMs lose info
**Our opportunity**: Message passing IS attention-lite with structural bias.
  The two-regime MI finding says ~75% of info is local (message passing handles)
  and ~25% is sparse long-range (Stage 5 handles).
  KEY QUESTION: is our message passing doing enough? Or do we need some
  attention here too (hybrid like Jamba's 1:7 ratio)?

### Stage 5: LONG-RANGE RETRIEVAL
**What it does**: Find and use information from distant context
**Current approaches**: Full attention, sparse attention, KV-cache, RAG
**Sutra v0.4**: Sparse top-k retrieval (k=16, content-addressable)
**What works**: Full attention for exact retrieval, RAG for external knowledge
**What fails**: Full attention too expensive, RAG adds latency
**Our opportunity**: Sparse retrieval IS the right mechanism per MI analysis.
  GROWN SPARSITY could make it even better (routing table evolves).
  Could also add 8-16 GLOBAL MEMORY TOKENS as scratchpad (Codex recommended).

### Stage 6: DEPTH / REASONING
**What it does**: Iterative refinement for complex reasoning
**Current approaches**: More transformer layers, chain-of-thought, test-time compute
**Sutra v0.4**: PonderNet adaptive halting (1-8 message passing rounds)
**What works**: More layers = more reasoning depth. CoT = explicit reasoning steps.
**What fails**: Fixed depth wastes compute on easy inputs.
**Our opportunity**: PonderNet gives ADAPTIVE depth (per-input). But it collapsed
  in Probe B. min_rounds=2 fix helps. Could also explore: different KINDS of
  processing per round (first rounds = syntax, later = semantics, like compilers).

### Stage 7: OUTPUT GENERATION
**What it does**: Convert internal representations to text output
**Current approaches**: Linear head + softmax (autoregressive), diffusion, energy-based
**Sutra v0.4**: Linear head + standard autoregressive sampling
**What works**: AR generation is simple and effective
**What fails**: Left-to-right commit prevents look-ahead planning
**Our opportunity**: For now, standard AR is fine. Later: could add verifier/reranker
  (generate multiple, score, pick best) for better reasoning quality.

---

---

## !!! SUTRA VISION: STAGE-SUPERPOSITION STATE MACHINE !!!

The 7 stages are NOT a linear pipeline. They form a STATE GRAPH where:
- Multiple stages ACTIVE SIMULTANEOUSLY on different positions
- Stages LOOP BACK (Stage 7 verify → Stage 4 reroute → Stage 5 update → retry 7)
- Each position in SUPERPOSITION of stages (80% constructing, 15% routing, 5% verifying)
- Like quantum superposition: model state IS a mixture, not one stage at a time
- Like spiking neural networks: asynchronous, no global clock, waves not layers
- EACH PATCH at its OWN stage based on content/difficulty

This is FUNDAMENTALLY DIFFERENT from all existing architectures.
Transformers/SSMs/hybrids process everything through same stage simultaneously.
Sutra: heterogeneous wavefront through stages at different rates per position.

For EACH stage: derive optimal mechanism FROM MATH. Not from existing models.
v0.5 attention refresh = copying Jamba. MUST derive novel Stage 4 mechanism instead.

---

### CODEX R1 CORRECTION: Better 7-Stage Decomposition

Our stages mixed mechanisms with capabilities. Codex proposed:

1. **Segmentation / Compression**: raw stream → patches/tokens
2. **State Init / Addressing**: embeddings + position (underleveraged by us)
3. **Local State Construction**: within-patch GRU, convs (our strength)
4. **Communication / Routing**: msg passing + retrieval + scratchpad (4+5 merged)
5. **State Update / Memory Write**: MLP, residual, memory retain/overwrite
6. **Compute Control / Deliberation**: depth, halting, adaptive compute
7. **Readout / Decode / Verify**: logits, sampling, reranking, verification

KEY INSIGHT: **Reasoning = repeated (4+5+6), not a standalone box.**
Memory needs READ and WRITE, not just retrieval.
BOTTLENECK: Stage 4+5 interface (local diffusion + global routing).
"Extra depth cannot recover missing evidence" — fix 4+5 FIRST.

### PIPELINE DEBATE QUEUE (3-4 rounds to converge)

**R2 (running)**: Per-stage best-in-class, Sutra gaps, concrete fixes.
**R3 (queued)**: Cross-stage interactions — how do stages COMPOSE?
  - Does improving Stage 1 (tokenization) reduce pressure on Stage 4 (communication)?
  - Are there stages where improvement in one AMPLIFIES another?
  - What's the optimal ORDER of improvement?
**R4 (queued)**: Final convergence — one unified framework.
  - Merge all rounds into a single definitive pipeline document
  - Each stage: definition, best approach, Sutra approach, gap, fix, priority
  - Signed off by Codex as the guide for all future research

### CROSS-STAGE AMPLIFICATION ANALYSIS (for R3)

**Key insight: earlier stages amplify ALL downstream stages.**

Improving Stage 1 (Segmentation/tokenization):
→ Stage 3: GRU processes semantic units not characters → better features
→ Stage 4: fewer patches → less communication needed → cheaper
→ Stage 5: richer per-position state → better memory
→ Stage 6: shorter sequences → fewer rounds needed
→ AMPLIFICATION: HIGH (4 downstream stages improved)

Improving Stage 4 (Communication/routing):
→ Stage 5: better info = better updates
→ Stage 6: more evidence available = better halting decisions
→ AMPLIFICATION: MEDIUM (2 downstream stages)

Improving Stage 6 (Compute control/depth):
→ Nothing downstream to amplify
→ AMPLIFICATION: LOW (terminal stage)
→ Codex: "Extra depth cannot recover missing evidence"

**CONCLUSION: Fix stages UPSTREAM first for maximum leverage.**
Priority order: 1 (tokenization) → 2 (addressing) → 4 (communication) → 6 (depth)
NOT: 6 → 4 → 1

**This is the strongest argument for token-level branch:**
Fixing Stage 1 amplifies EVERYTHING downstream.

### THEORETICAL OPTIMAL PER STAGE (for R4 convergence)

| Stage | Theoretical Optimal | Current Best | Sutra v0.4 | Gap |
|-------|-------------------|-------------|-----------|-----|
| 1. Segmentation | MDL-optimal learned | BPE (corpus stats) | Raw bytes | HIGH |
| 2. Addressing | Hierarchical + relative | RoPE | GRU implicit | MEDIUM |
| 3. Local | Tiny attention within unit | Conv+attention | GRU | LOW |
| 4. Communication | Exact routing at min cost | Full attention O(n²) | MsgPass+sparse O(n) | MEDIUM |
| 5. State Update | Bayesian updating | Gated MLP+residual | MLP+residual | LOW |
| 6. Compute Control | Exact difficulty-matched | Fixed depth | PonderNet | MEDIUM |
| 7. Readout | Quality-optimal selection | AR+reranking | Standard AR | LOW |

**Biggest gaps**: Stage 1 (bytes vs optimal), Stage 4 (local-only vs full routing).
**Sutra's strengths**: Stage 3 (GRU is good for local), Stage 5 (standard works).

**Biological parallel**: every stage has a biological "optimal" too:
- Brain: retina (1), topographic maps (2), V1 (3), cortical connectivity (4),
  synaptic plasticity (5), attention allocation (6), motor output (7)
- All evolved over 500M years to be near-optimal for their data distribution.
- Our architecture is days old. The gap is expected.

### WHERE IS SUTRA WEAKEST?

Looking at this breakdown, Stage 4 (Context Integration) is the riskiest.
Our message passing with window=4 means each round propagates info by 4 positions.
With 6 rounds: effective receptive field = 4^6... no wait, it's additive not
multiplicative. 6 rounds × window 4 = reach of ~24 patches = ~96 bytes.
For 512 byte sequences, that's only 19% of the sequence per round of local msg.

The sparse retrieval (Stage 5) compensates, but k=16 out of 128 patches is
only 12.5% of positions. Combined: ~19% local + ~12.5% retrieval = ~30% of
positions are reachable per forward pass.

A transformer reaches 100% of positions. We reach ~30%. Is 30% enough?
The MI analysis says 75% local + 25% sparse ≈ we should reach ~100% of the
information. But the IMPLEMENTATION reaches fewer POSITIONS than ideal.

**Fix**: Increase window to 8-16 (doubles local reach) OR add a single
global attention layer every N rounds (like Jamba's hybrid approach).

### Evolutionary Routing Search
Hybrid training: gradient descent for model, evolution for routing patterns.
Routing table mutations tested each generation, best kept. Escapes local optima.
Finer-grained than NAS — searches routing within fixed architecture.
Continuous during training, not separate preprocessing.

### Learned Compression (No Tokenizer)
Replace tokenization with learned byte-to-concept compression.
Multi-scale convolutions + importance scoring + adaptive pooling.
The model learns its own tokenization end-to-end. Prototype running.

---

**Regularization IS compression**: L1/L2/dropout force simpler representations.
Sutra's patch structure provides STRUCTURAL regularization — info must flow through
summary bottleneck. This is why smaller Sutra (1.2M) beats larger transformer (6.5M)
on small data — the bottleneck prevents overfitting naturally.

---

## FRESH EXPLORATION #1: What if we approached this as a CODING problem?

### The Premise

Code execution is a SOLVED problem. Computers run programs perfectly.
Language understanding is an UNSOLVED problem. Models hallucinate constantly.

What if we designed a model that THINKS in something closer to code?
Not symbolic AI (too rigid) but a model whose INTERNAL representation
is more program-like than vector-like.

### What Makes Programs Different From Vectors?

1. Programs are COMPOSITIONAL (functions call functions)
2. Programs have EXPLICIT control flow (if/else, loops)
3. Programs maintain EXPLICIT state (variables, memory)
4. Programs are VERIFIABLE (you can trace execution)
5. Programs are INTERPRETABLE (you can read them)

Current neural nets have NONE of these properties in their internals.
Sutra's stages TRY to add some (explicit state, verification) but
the internal representations are still opaque vectors.

### What if the Representation Itself Was Structured?

Instead of h_i ∈ R^d (opaque vector), what if:

```
h_i = {
    type: "noun_phrase",           # Categorical type
    content: R^128,                 # Dense content vector
    bindings: {agent: ptr_to_5},   # Explicit variable bindings
    confidence: 0.85,               # Calibrated uncertainty
    needs: ["verb", "complement"],  # What this position still needs
}
```

A STRUCTURED representation that carries typed, inspectable information.
Not pure symbolic (the content is still a dense vector) but HYBRID:
symbolic structure + neural content.

### Why This Might Work

1. Variable binding becomes EXPLICIT (not implicit in hidden states)
2. The model can CHECK its own state ("do I have a verb? no → route for one")
3. Type information helps routing ("nouns route to verbs, not to other nouns")
4. Confidence is built-in (not just Kalman variance but explicit)
5. The "needs" field IS the demand signal for routing

### Why This Might NOT Work

1. Structured representations are hard to optimize with gradient descent
2. Discrete types require Gumbel-Softmax or REINFORCE
3. Pointer-based bindings are non-differentiable
4. The representation space is much more complex to navigate
5. Current models work FINE with opaque vectors at scale

### Connection to Neuro-Symbolic AI

This is in the neuro-symbolic family but different:
- Standard neuro-symbolic: neural perception → symbolic reasoning
- Our proposal: the REPRESENTATION is hybrid, not the processing pipeline
- Each position carries BOTH symbolic structure AND neural content
- Processing is still neural (differentiable) but on structured states

---

## FRESH EXPLORATION #2: What if intelligence is COMPRESSION applied recursively?

### The Core Claim (Stronger Than Before)

Not just "compression = intelligence" but "RECURSIVE compression = intelligence."

A single compression step: raw text → features (what transformers do).
RECURSIVE compression: raw → local features → patterns of features →
patterns of patterns → ... → the simplest possible description.

Each level of recursion discovers HIGHER-ORDER structure.
Level 0: character frequencies (trivial)
Level 1: word patterns (basic language model)
Level 2: phrase structures (syntactic understanding)
Level 3: argument patterns (reasoning capability)
Level N: increasingly abstract reasoning primitives

### The Mathematical Structure

Iterated function system: f applied recursively.
h_{level+1} = compress(h_{level})
where compress removes redundancy while preserving predictive info.

This is EXACTLY what deep networks do layer by layer.
But current networks use FIXED compression (same attention/MLP at every level).
What if the COMPRESSION FUNCTION itself changed at each level?

Level 0 compression: character → word (needs local patterns)
Level 1 compression: word → phrase (needs syntax)
Level 2 compression: phrase → meaning (needs semantics)
Level 3 compression: meaning → argument (needs logic)

Different information types at each level → different compression optimal.

### Connection to Renormalization Group (Physics)

In statistical mechanics, renormalization group (RG) is EXACTLY recursive
compression: zoom out one step, integrate out fine-grained details, get
a coarser description. Repeat. At each scale, the EFFECTIVE THEORY changes.

Language RG:
- Scale 0: bytes. Effective theory: character n-grams.
- Scale 1: words. Effective theory: syntax + morphology.
- Scale 2: sentences. Effective theory: semantics + pragmatics.
- Scale 3: paragraphs. Effective theory: discourse + argumentation.

Each scale has its own "effective theory" — the compression function that
works best at that scale. This IS the multi-grain processing we proposed
for Stage 3, but now framed as a PHYSICAL PRINCIPLE.

### What This Means for Architecture

Instead of "7 stages in a pipeline," think of it as:
"Multiple RG scales, each with its own effective processor."

The model doesn't process in stages — it simultaneously maintains
representations at MULTIPLE compression levels, and information flows
both UP (abstraction) and DOWN (prediction/verification) between levels.

This is U-Net for language: encoder compresses, decoder expands,
skip connections bridge scales. But with DIFFERENT operations at each scale.

### Why This Might Be THE Core Idea

Codex said: "there's probably one real core idea, not six."
What if recursive compression IS that one idea?

It naturally explains:
- Why local processing matters (Level 0-1 compression)
- Why routing matters (cross-level information flow)
- Why depth matters (more recursion = higher-level abstraction)
- Why uncertainty tracks quality (well-compressed = confident)
- Why the MI has two regimes (local compression is fast, global is slow)

Everything else (routing mechanism, memory, halting) is IMPLEMENTATION
of recursive compression. The theory IS the compression.

---

## FRESH EXPLORATION #3: What if we built a DIFFERENTIABLE DATABASE?

### The Premise

Language models store knowledge in weights. Databases store knowledge in tables.
Weights: fast access, poor precision, non-updatable at inference.
Tables: precise, updatable, slow sequential access.

What if the model's memory was structured like a DATABASE?
Keys, values, indexes — but all differentiable and trainable.

### Structure

```
Memory = {
    table_1: {keys: Tensor[M, d_key], values: Tensor[M, d_val]},  # facts
    table_2: {keys: ..., values: ...},                               # rules
    index: learned hash function for O(1) lookup
}
```

At each step, the model can:
- QUERY: look up a key → get a value (like SELECT in SQL)
- INSERT: add a new key-value pair (like INSERT)
- UPDATE: modify an existing value (like UPDATE)
- DELETE: remove a pair (like DELETE)

### Why This Is Different From KV-Cache / RAG

KV-cache: stores past activations, read-only at inference.
RAG: external retrieval from a document store.
Our database: PART OF THE MODEL, differentiable, trained end-to-end,
supports all CRUD operations, and the INDEX is learned.

The database IS the model's long-term memory. The GRU/routing is the
model's working memory. Two memory systems, like hippocampus (episodic)
and neocortex (semantic) in the brain.

### Connection to Stage 5 (Memory Write)

Stage 5 IS the database write operation. But currently it's just a
gated vector update. A structured database gives:
- NAMED entries (not just vectors)
- SELECTIVE access (query for specific info, not attend to everything)
- PERSISTENT storage (doesn't decay through layers)
- UPDATABLE at inference (can learn from context)

---

## FRESH EXPLORATION #4: What if position doesn't matter?

### The Radical Question

Current models encode position EVERYWHERE. But does a model NEED to
know that "cat" is at position 5? Or does it only need to know:
- "cat" is the subject of "sat"
- "cat" comes before "sat"
- "cat" is near "the"

These are RELATIONAL facts, not positional ones.

### Position-Free Architecture

What if instead of position embeddings, we encoded RELATIONSHIPS?
Each token gets features based on its RELATIONS to other tokens,
not its absolute or relative position.

Implementation: start with no position info. Let the model DISCOVER
that position matters (if it does) through the routing mechanism.
The routing table (from grown sparsity) IS the relational structure.
Position emerges from relationships, not the other way around.

### Why This Might Work

1. Language IS relational. "The cat sat" works regardless of position 5 or 500.
2. Removes the length generalization problem entirely (no position to extrapolate).
3. The model only learns what ACTUALLY matters for prediction.
4. Simpler — fewer inductive biases, more room for the model to discover structure.

### Why This Probably Won't Work

1. Word ORDER matters in most languages ("dog bites man" ≠ "man bites dog")
2. Without ANY positional signal, the model can't distinguish permutations
3. GRU within patches already provides implicit position (sequential processing)
4. Every successful model uses position encoding — there's a reason

### The Compromise

Use position encoding ONLY within patches (GRU provides this naturally).
Between patches: NO position encoding. Let routing discover which patches
are related, not assume nearby patches are more related.

This IS what our v0.4 does for the message passing (window-based = implicit
position) but NOT for the sparse retrieval (which uses RoPE or top-k scores).
What if sparse retrieval was PURELY content-based, no position bias at all?

---

## FRESH EXPLORATION #5: What if the model was a SIMULATOR, not a predictor?

### The Premise

Current models predict P(next_token | context). They're PREDICTORS.
But understanding language requires SIMULATING: building a mental model
of the situation described and running it forward.

"John picked up the ball. He threw it to Mary."
A predictor: P("Mary" | "He threw it to") = high (pattern matching).
A simulator: builds a WORLD MODEL with {John: has ball → throws → Mary: catches}.
The prediction comes FROM the simulation, not from token statistics.

### What's Different

A predictor uses CORRELATIONS: "after 'threw it to', names are likely."
A simulator uses CAUSATION: "John had the ball, threw it, so Mary has it now."

Correlations break on novel situations. Causation generalizes.
This is why LLMs hallucinate: they predict plausible-sounding tokens
without maintaining a consistent world model.

### Architecture Implications

The model's hidden state IS the world model. At each token:
1. UPDATE the world model (new info from the token)
2. RUN the world model forward (what follows from current state?)
3. PREDICT from the simulation (what token describes the next state?)

This is NOT a new idea (world models, JEPA, etc.) but the specific
framing for LANGUAGE is: the hidden state must be a SIMULATOR STATE,
not just a feature vector. It tracks entities, relations, and dynamics.

### Connection to Our Work

- Stage 5 (memory write) = world model update
- Stage 4 (routing) = finding which entities/relations need updating
- Stage 3 (local) = parsing input into entity/relation changes
- Stage 7 (verify) = checking simulation consistency

The 7 stages COULD be reinterpreted as a simulation loop.
But the key insight is: the REPRESENTATION should be a world state
(entities + relations + dynamics), not an opaque vector.

### Why This Connects to Exploration #1 (Structured Representations)

Exploration #1 proposed: type + content + bindings + confidence + needs.
That IS a world model state: entities (typed), relationships (bindings),
uncertainty about the model (confidence), what's missing (needs).

Maybe #1 and #5 are the SAME idea from different angles:
structured representations = world model states.

---

## FRESH EXPLORATION #6: What if we used ENERGY FUNCTIONS instead of next-token prediction?

### The Premise

Next-token prediction forces LEFT-TO-RIGHT generation.
But understanding isn't left-to-right. You often need the END
of a sentence to understand the BEGINNING.

"The bank by the river..." — "bank" is ambiguous until "river" resolves it.

Energy-based models define E(entire_sequence) and find low-energy configs.
Generation = finding the sequence with lowest energy.
Understanding = checking that the observed sequence has low energy.

### What's Different From Autoregressive

AR: commit to each token, can't go back. One-directional.
Energy: evaluate the WHOLE sequence at once. Bidirectional.

This means: the model can use FUTURE context to understand PAST tokens.
"The bank" at position 1 gets resolved when "river" at position 8 is processed.

### Architecture: Iterative Refinement

Start with noisy/random sequence. Each step: reduce energy everywhere.
After enough steps: converge to a coherent, low-energy sequence.

This IS diffusion for language (MDLM, SEDD, etc.) but the framing
is different: not "denoise" but "find the minimum energy configuration."

### Connection to Our Stages

Energy minimization IS our inner loop (Stages 3-5 repeated):
- Stage 3: local energy reduction (fix local inconsistencies)
- Stage 4: global energy reduction (fix long-range inconsistencies)
- Stage 5: update state to reflect lower energy

The "energy" IS the prediction loss at each position.
Iterative refinement = multiple rounds of message passing.
Convergence to low energy = our model processing until confident.

### The Problem

Energy-based text generation is SLOW and currently worse than AR.
But energy-based UNDERSTANDING (scoring existing text) might be
valuable: use energy as a VERIFIER (Stage 7) even if generation is AR.

Score: is this response LOW ENERGY (coherent, consistent)?
If not: regenerate or edit.

---

## CODEX TRIAGE RESULTS

| Exploration | Novelty | Feasibility | Potential | Verdict |
|------------|---------|-------------|-----------|---------|
| #1 Code-like reps | 3 | 2 | 3 | Later auxiliary, not core |
| #2 Recursive compression | 4 | 4 | **5** | **CORE BET** |
| #3 Differentiable DB | 2 | 3 | 3 | Subsystem only |
| #4 Position-free | 3 | 1 | 1 | **KILL** |
| #5 World simulator | (not triaged yet) | | | Connects to #1 |
| #6 Energy functions | (not triaged yet) | | | Verifier role |

**Codex missing angle**: Iterative constraint satisfaction / belief propagation.
Language understanding = finding globally consistent state under local constraints.
This IS message passing in graphical models. Connects to everything naturally.

## DEEP DIVE: Recursive Compression (The Core Bet)

If recursive compression IS the one core idea, what does the architecture look like?

### The Renormalization Group Architecture

```
Level 0: raw bytes                    [N positions, d dims]
    ↓ compress_0 (character → word)
Level 1: word features                [N/4 positions, d dims]
    ↓ compress_1 (word → phrase)
Level 2: phrase features              [N/16 positions, d dims]
    ↓ compress_2 (phrase → meaning)
Level 3: meaning features             [N/64 positions, d dims]

Cross-level information flow (both UP and DOWN):
Level 0 ←→ Level 1 ←→ Level 2 ←→ Level 3
```

Each level has its OWN effective processor (different compressions at different scales).
Information flows UP (abstraction) and DOWN (prediction/verification).

### This IS Sutra v0.4 But Formalized as RG

v0.4 has:
- Level 0: bytes → patches (our 4-byte chunking = compress_0)
- Level 1: patch processing (GRU = compress_1 within patch)
- Level 1→2: message passing between patches (cross-patch = compress_2)
- Only 2 levels! Missing levels 2→3 and deeper.

The v0.4 architecture is a SHALLOW renormalization — only 2 levels of
compression. A deeper architecture would add more levels:
- Level 2: groups of patches (sentences/paragraphs)
- Level 3: groups of groups (sections/documents)

### Why More Levels Might Help

The MI two-regime finding: local alpha=0.94, global alpha=0.26.
The TRANSITION at d~10 is where Level 0→1 compression happens.
But there might be ANOTHER transition at d~100 (Level 1→2)
and another at d~1000 (Level 2→3).

If we measured MI at even longer ranges, we might find a THIRD regime
with an even slower decay — the paragraph/discourse level.

### The Experiment

Measure MI at very long ranges (d=1000-10000) on MiniPile.
If there's a third regime transition, it validates the multi-level RG.
If MI is flat beyond d~500, two levels are enough.

---

## FRESH EXPLORATION #7: Belief Propagation / Constraint Satisfaction (Codex-suggested)

### The Premise

Language understanding = finding a globally consistent interpretation
that satisfies many LOCAL constraints simultaneously.

"The cat that the dog chased ran away."
Constraints:
- "cat" is subject of "ran" (syntactic)
- "dog" is subject of "chased" (syntactic)
- "cat" is object of "chased" (semantic)
- "chased" happened before "ran" (temporal)
- all referents are animate (world knowledge)

Understanding = finding the assignment of roles/relations that satisfies ALL constraints.
This is CONSTRAINT SATISFACTION, not next-token prediction.

### Connection to Graphical Models

In probabilistic graphical models, belief propagation (BP) finds the
most probable assignment of variables given observed evidence and
factor constraints. The BP algorithm:

1. Each variable sends BELIEFS to its neighbors
2. Each factor sends MESSAGES based on constraint compatibility
3. Iterate until convergence
4. Read off the most probable assignment

This IS message passing! Our Stage 4 IS belief propagation!
But we never framed it that way.

### What BP Gives Us That Generic Message Passing Doesn't

BP messages have a SPECIFIC meaning: "given my constraint, I believe
variable X should be in state S with probability P."

Generic message passing: "here's some information from my neighborhood."
BP message passing: "based on what I know, you should be X because of constraint C."

The difference: BP messages are STRUCTURED (carry beliefs, not just features).
And BP has a CONVERGENCE guarantee (on trees; approximate on loopy graphs).

### The BP-Sutra Architecture

Instead of generic message passing between patches:

```
Each patch maintains BELIEFS: distribution over possible interpretations.
Each pair of connected patches shares FACTORS: compatibility constraints.

Message from patch j to patch i:
  m_{j→i} = Σ_{x_j} factor(x_i, x_j) · belief_j(x_j) / m_{i→j}

Updated belief at patch i:
  belief_i(x_i) ∝ Π_j m_{j→i}(x_i) · local_evidence_i(x_i)
```

The beliefs are DISTRIBUTIONS, not point estimates.
The messages carry STRUCTURED information about compatibility.
Convergence can be measured: when beliefs stop changing, the model is "done."

### Why This Might Be The Core Idea

It unifies:
- Message passing (BP IS message passing, but structured)
- Uncertainty (beliefs ARE probability distributions)
- Convergence/halting (stop when beliefs converge)
- Routing (send messages along factor graph edges, not everywhere)
- Verification (check global consistency of converged beliefs)

AND it's derived from mathematics (probability theory, not biology).

### The Challenge

Standard BP is for DISCRETE variables. Language positions have
CONTINUOUS high-dimensional representations. Need CONTINUOUS BP.

This EXISTS: Gaussian BP, expectation propagation, variational message passing.
These are well-studied in the probabilistic programming community.

### Connection to Kalman

Kalman filter IS Gaussian belief propagation on a chain graph!
Our Kalman state updates (mean, variance) ARE BP messages for a
linear-Gaussian model. So the Kalman idea from the stage framework
IS a special case of BP.

This means: BP is the GENERAL version of what we were already doing
with Kalman. Kalman = BP on a chain. Full BP = BP on the full factor graph.

---

## CONVERGENCE: Multi-Scale Belief Propagation (Explorations #2 + #7 Merged)

### The Synthesis

Recursive compression (#2) = WHAT (compress at multiple scales).
Belief propagation (#7) = HOW (structured messages with convergence).

Combined: **at each scale, BP finds the consistent interpretation.
Moving to a coarser scale compresses the beliefs from the finer scale.**

### The Architecture (One Unified Principle)

```
Scale 0 (bytes):    [N positions]
  BP at scale 0: find consistent character-level interpretation
  Beliefs_0 converge → compress into:

Scale 1 (words):    [N/4 positions]
  BP at scale 1: find consistent word-level interpretation
  Uses compressed beliefs from Scale 0 as evidence
  Beliefs_1 converge → compress into:

Scale 2 (phrases):  [N/16 positions]
  BP at scale 2: find consistent phrase-level interpretation
  Uses compressed beliefs from Scale 1 as evidence
  Beliefs_2 converge → compress into:

Scale 3 (meaning):  [N/64 positions]
  BP at scale 3: find consistent semantic interpretation
```

Information flows BOTH directions:
- UP: beliefs compress into coarser scales (abstraction)
- DOWN: coarser beliefs constrain finer scales (prediction/verification)

### Why This Is THE Core Idea

ONE mechanism (Gaussian BP) at EVERY scale.
Different scales handle different MI regimes:
- Scale 0-1: local MI regime (alpha=0.94, fast convergence)
- Scale 1-2: transition regime
- Scale 2-3: global MI regime (alpha=0.26, slow convergence)

The BP convergence time at each scale MATCHES the MI decay:
- Local BP: fast (few iterations needed, strong local constraints)
- Global BP: slow (many iterations, weak long-range constraints)

Adaptive depth EMERGES: positions converge at different rates.
Local positions: BP converges fast → done early.
Complex positions: BP converges slow → needs more iterations.
No PonderNet needed — convergence IS the halting criterion.

### Comparison to Existing Architectures

- Transformer: flat, all-to-all, no scale hierarchy, no beliefs
- Mamba: sequential, single scale, no beliefs
- Jamba: hybrid but same-scale, no beliefs
- U-Net: multi-scale but no BP, different domain (images)
- Factor graph NN: BP-inspired but single scale, not for language

**Multi-scale BP for language modeling: genuinely novel.**

### The Experiment to Test This

Simplest version: 2-scale BP (bytes + words).
- Scale 0: local GRU (captures byte-level beliefs)
- Scale 1: message passing with BELIEF messages (not just features)
- Messages: (mean, precision) pairs, combined via precision-weighted averaging
- Convergence: stop when beliefs stop changing (norm of update < threshold)

Compare against: v0.4 (same compute, no belief structure).
If belief structure helps convergence and prediction: THE idea validated.
If not: BP adds overhead without benefit. Kill it.

---

## FRESH EXPLORATION #8: What can MUSIC theory teach us about sequence modeling?

### Not "music generation" but "music THEORY as mathematical framework"

Music theory has spent centuries analyzing HOW sequences create meaning:
- HARMONY: simultaneous compatible notes (= compatible features at same position)
- COUNTERPOINT: independent voices that complement each other (= multi-stream processing)  
- RHYTHM: hierarchical temporal structure (= multi-scale timing)
- TENSION/RESOLUTION: building expectation then satisfying it (= prediction then confirmation)
- MODULATION: shifting key/context smoothly (= topic transitions)

These are all SEQUENCE PROCESSING concepts, formalized over centuries.

### Counterpoint as Multi-Stream Architecture

In counterpoint, multiple INDEPENDENT melodic lines run simultaneously.
Each line follows its OWN rules, but they must HARMONIZE at intersection points.
The beauty emerges from the INTERACTION, not from any single line.

For Sutra: what if we processed MULTIPLE INDEPENDENT streams simultaneously?
- Stream 1: syntax processing (like a bass line — structural foundation)
- Stream 2: semantic processing (like a melody — the "meaning")
- Stream 3: pragmatic processing (like harmony — context/relationship)
- Stream 4: factual checking (like rhythm — grounding beats)

Each stream follows its own processing rules. They INTERSECT at key points
(like our Stage 4 routing) to ensure global consistency.

This is the 4-stream output-first idea but from a MUSIC perspective.
Music theory gives us formal rules for HOW streams should interact
(harmonic rules, voice-leading rules, resolution rules).

### Tension-Resolution as Prediction Model

Music creates TENSION (dissonance, unexpected notes) and then RESOLVES it.
Language does the same: "The cat that the dog chased..." creates syntactic
tension (nested clause) that resolves with "...ran away."

What if the model explicitly tracked TENSION (unresolved elements)?
Each position carries a "tension score" — how many unresolved dependencies
point to/from this position. The model actively seeks RESOLUTION
(routing to resolve open dependencies).

This connects to: the "needs" field in structured representations (#1),
the "demand" vector in supply/demand routing, and the variance in Kalman.
All are measures of "unresolved tension" that drives processing.

---

## FRESH EXPLORATION #9: TOPOLOGICAL perspective on representations

### What if we analyzed representations as TOPOLOGICAL spaces?

Current analysis: representations are VECTORS in R^d. Distances, angles, clusters.

Topological analysis: representations have SHAPE — holes, loops, connected
components. The shape persists even as the space is deformed.

### Persistent Homology of Language Representations

Topological data analysis (TDA) uses persistent homology to find the
SHAPE of data: which features are "holes" (things that are consistently
NOT represented) vs "clusters" (things that group together).

For language: the TOPOLOGY of the representation space might reveal:
- Connected components = major semantic categories
- 1-cycles (loops) = circular reference chains (pronouns → referents → pronouns)
- Higher homology = complex relational structures

If we DESIGNED the representation space to have the RIGHT topology
(matching the topology of language structure), features might be
more robust and generalizable.

### Topological Regularization

Add a loss term that encourages the representation space to have
the "right" topological structure:
- Encourage distinct clusters for distinct concepts (0-homology)
- Discourage spurious loops in the representation (1-homology)
- Match the persistence diagram of representations to the persistence
  diagram of the known language structure

This is VERY speculative but connects to: the geometry = intelligence thesis,
fractal embeddings (our parent project), and the idea that STRUCTURE in
the representation is more important than SCALE.

---

## FRESH EXPLORATION #10: What if we treated ATTENTION as the bug, not the feature?

### The Contrarian View

Everyone is trying to KEEP attention but make it cheaper.
Mamba, RWKV, linear attention — all try to approximate attention.

What if attention is the WRONG MECHANISM ENTIRELY?

Attention says: "every position should look at every other position
and decide what's relevant." This is SEARCH — brute force content search.

What if the right approach is NOT search but STRUCTURE?
Instead of searching for relevant positions, BUILD the structure
that DEFINES which positions are relevant.

### The Analogy: Libraries vs Google

Google (attention): search everything, rank by relevance, return top-k.
Library (structure): organized by Dewey Decimal, you KNOW where to look.

Google is powerful but expensive. Libraries are efficient because the
STRUCTURE tells you where information lives.

What if the model built an internal "library" — a structured index
that organizes information by TYPE and RELATION, so routing is O(1)
lookup, not O(n) search?

This IS the grown sparsity idea (routing table) but pushed further:
the routing table isn't just "who connects to whom" but a full
STRUCTURAL INDEX organized by content type and relation.

### Connection to Other Explorations

- Grown sparsity (#4 from original): the routing table IS the index
- Belief propagation (#7): the factor graph IS the structure
- Recursive compression (#2): the compression hierarchy IS the organization

All these explorations keep pointing to the same thing:
**STRUCTURE over SEARCH.** Build the right structure, and you don't
need to search. The structure IS the understanding.

---

## META-PATTERN: Structure Over Search

Across 10 explorations, one principle keeps emerging:

**Don't SEARCH for relevant information (attention = brute force search).
BUILD the STRUCTURE that defines relevance (our approach = organized index).**

Library vs Google. Dewey Decimal vs keyword search.
The structure IS the understanding. Building it IS the intelligence.

This IS "Intelligence = Geometry" taken to its logical conclusion:
not just better embedding spaces but the COMPUTATION STRUCTURE itself
should mirror the PROBLEM STRUCTURE.

Transformers: flat search over all positions (Google).
Sutra: multi-scale structured processing where structure = understanding (Library).

---

## PRACTICAL: Minimum Viable Multi-Scale BP (When Compute Frees)

### The 100-Line Test

Two scales: bytes (fine) and words/patches (coarse).
BP at each scale: precision-weighted message passing.
Cross-scale: compress fine→coarse (abstraction), coarse→fine (prediction).
Compare: does BP structure beat generic message passing at matched params?

```
class MinimalMSBP(nn.Module):
    # Scale 0: byte-level GRU (local beliefs)
    # Scale 1: patch-level BP (global beliefs)
    # Cross-scale: compress up, predict down
    # Halting: when beliefs converge (norm of update < threshold)
```

This test isolates THE core claim: structured (BP) messages > generic messages.
If YES: multi-scale BP is the architecture direction.
If NO: structure doesn't help, go back to generic routing.

### What We'll Measure
1. BPB on test set (quality)
2. Convergence speed (how many rounds to stabilize)
3. Belief quality (do beliefs correlate with correctness?)
4. Cross-scale benefit (does scale 1 actually help scale 0?)

---

## FRESH EXPLORATION #11: What if the TRAINING OBJECTIVE is wrong?

### Everyone Uses Cross-Entropy. Is That Optimal?

Cross-entropy: minimize -log P(correct_token | context). Treats every
token equally. Treats every error equally.

But: predicting "the" after "in" is TRIVIAL and teaches nothing.
Predicting the answer to "What is 17*23?" is HARD and teaches everything.

### Alternative Objectives (From First Principles)

**A. Focal Loss for Language**
Weight = (CE)^gamma: hard tokens get exponentially more weight.
Already proven for object detection (class imbalance). For language:
the "class imbalance" is between easy (function words) and hard
(content words, reasoning, facts) predictions.

**B. Contrastive Loss**
Instead of "predict the right token," ask "distinguish the right token
from wrong ones." Forces the model to understand WHY the right answer
is right, not just that it's likely.

**C. Compression Objective (MDL)**
Minimize BOTH prediction error AND model complexity.
L = CE + lambda * description_length(model).
Forces the model to find SIMPLER patterns that predict as well.

**D. Multi-Objective: Different Losses Per Stage**
Stage 3 loss: local prediction quality (can local features predict next byte?)
Stage 4 loss: routing quality (does routing improve prediction vs no routing?)
Stage 5 loss: retention quality (does memory improve prediction vs fresh start?)

Each stage has its OWN loss that measures its SPECIFIC contribution.
The total loss is the sum. But now each stage is optimized for its purpose.

**E. Self-Supervised Consistency**
Instead of predicting the next token, predict whether TWO different
processing of the same text agree. If the model routes differently on
two passes of the same input, something is wrong.

This encourages CONSISTENT routing — the model should process the same
text the same way regardless of initialization/noise.

### Which Objective Best Serves "Structure Over Search"?

If the core insight is that STRUCTURE is more valuable than SEARCH,
then the objective should REWARD structure and PENALIZE brute-force
memorization.

MDL (option C) does exactly this: simpler models that predict well
are PREFERRED over complex models that predict equally well.
A model that builds good structure (simple) beats a model that
memorizes (complex). The objective ENCODES our thesis.

### This Connects to Grokking

Grokking = phase transition from memorization to generalization.
MDL objective = explicit pressure toward the generalizing solution.
Prediction: MDL-trained models grok FASTER than CE-trained models.

---

## FRESH EXPLORATION #12: What can THERMODYNAMICS teach us about training dynamics?

### Free Energy Minimization as Training

In thermodynamics: systems evolve to minimize free energy F = E - TS.
E = internal energy (how well the system fits data).
T = temperature (how much randomness/exploration).
S = entropy (how disordered the state is).

Training a neural net IS free energy minimization:
- E = loss (prediction error)
- T = learning rate (controls exploration vs exploitation)
- S = weight entropy (model complexity)

Low T (low LR): system freezes into local minimum. Exploits.
High T (high LR): system explores broadly. Doesn't converge.
Annealing (decrease T over time): explore first, then settle.

### Phase Transitions in Training

At critical temperature T_c: the system undergoes phase transition.
Above T_c: disordered (memorization, random weights).
Below T_c: ordered (generalization, structured weights).
AT T_c: maximum fluctuations, maximum learning rate.

Grokking IS a phase transition. The system stays disordered (memorizing)
until it crosses T_c, then rapidly crystallizes into the generalizing solution.

### Design Implication: Temperature-Aware Architecture

What if different parts of the model had different TEMPERATURES?
- Stage 3 (local): low temperature (stable, well-understood patterns)
- Stage 4 (routing): medium temperature (exploring connections)
- Stage 5 (memory): high temperature initially, cooling as confidence grows
- Stage 7 (verify): very low temperature (need reliable checking)

The Kalman variance IS a form of per-position temperature.
High variance = high temperature = exploring.
Low variance = low temperature = settled.

### Simulated Annealing for Architecture Search

Instead of gradient descent for routing: simulated annealing.
High temperature: try random routings. Accept bad ones sometimes.
Low temperature: only accept improvements. Converge to optimal.

The grown sparsity routing table could use annealing:
- Early training: high temperature, routing table changes rapidly
- Late training: low temperature, routing table stabilizes
- The routing table CRYSTALLIZES during training, like a physical crystal

---

## FRESH EXPLORATION #13: What if we built DUAL representations?

### The Premise: Primal and Dual Spaces

In optimization, every problem has a DUAL. The dual provides:
- A different VIEW of the same problem
- BOUNDS on the optimal solution
- Sometimes EASIER to solve

What if each position maintained TWO representations?
- PRIMAL: what this position IS (content, features)
- DUAL: what this position NEEDS (constraints, requirements)

The primal is the standard hidden state.
The dual is the demand vector from supply/demand routing.

### Why Dual Representations Help

Processing alternates between primal and dual updates:
1. Given current primal → compute what's needed (dual update)
2. Given current dual → find what satisfies it (primal update)

This is EXACTLY how primal-dual optimization works:
alternating between improving the solution and tightening the constraints.

### Connection to Belief Propagation

In BP, messages FROM variables and messages TO variables are the
primal/dual pair. Variable beliefs = primal. Factor messages = dual.
BP alternates between updating both.

So: primal-dual representations = BP in disguise.
Another road leading to the same destination.

### Practical Implementation

State at position i: (h_primal_i, h_dual_i) ∈ R^(2d)
Primal update: h_primal ← route(h_primal, h_dual_neighbors)
Dual update: h_dual ← constraints(h_primal, h_primal_neighbors)

This doubles the state size (like Kalman mean+variance)
but the dual explicitly represents WHAT'S MISSING — exactly
the information that routing (Stage 4) needs.

---

## CONVERGENCE MAP: Where Do All 13 Explorations Point?

### Three Distinct Clusters Emerged

**Cluster A: Structured Processing (The Core)**
#2 (recursive compression), #7 (belief propagation), #10 (structure over search),
#13 (primal-dual)
→ ALL point to: process through STRUCTURE, not through brute-force search.
→ The computation should BUILD and REFINE a structure (factor graph, hierarchy,
   routing table) rather than search through all possibilities.

**Cluster B: Adaptive Compute (The Mechanism)**
#5 (world simulator), #6 (energy functions), #12 (thermodynamics)
→ ALL point to: the amount of processing should ADAPT to difficulty.
→ Convergence, energy minimization, and temperature all provide natural
   adaptive compute without explicit halting mechanisms.

**Cluster C: Richer Representations (The Foundation)**
#1 (code-like), #3 (differentiable DB), #8 (music counterpoint), #9 (topology)
→ ALL point to: the REPRESENTATION should carry more structure than a flat vector.
→ Types, relations, beliefs, streams — some form of structured state.

**Killed:**
#4 (position-free): language needs order. Dead.
#11 (training objectives): useful engineering, not core architecture.

### The Unified Architecture (All Three Clusters)

IF we combine the best from each cluster:
- FROM A: multi-scale hierarchy where structure IS understanding
- FROM B: convergence-driven adaptive compute (Kalman variance or BP residual)
- FROM C: state carries (mean, uncertainty) not just flat vector

This IS the Kalman Machine (Combo 2) or hybrid MSBP, depending on framing.

### The Honest Assessment

After 13 explorations and extensive Codex debate:
- Novelty: MODERATE. No single mechanism is brand-new.
- Combination novelty: HIGH. This specific integration hasn't been done.
- Chance of beating transformers: LOW-MEDIUM (Codex consistently rates 4-5/10).
- The VALUE is in the FRAMEWORK, not any single mechanism.

The real question is whether structured-processing architectures can
match brute-force-search architectures (transformers) on quality.
That's an EMPIRICAL question. Theory can't answer it. Experiments can.

---

## PRACTICAL: Next Training Run Improvements (based on current run observations)

### What We're Seeing
- Loss oscillating 0.54-0.57, BPB ~0.78
- LR still warming up (hasn't peaked at 3e-4 yet)
- 5K tok/s throughput
- 10.4GB VRAM (of 24GB available)

### What To Change For Next Run

1. **Mixed precision (bf16)**: Would nearly 2x throughput. Currently fp32.
   Save ~5GB VRAM, enable larger batch size.

2. **Larger batch**: Current effective batch=32. With bf16 + freed VRAM,
   could do effective batch=128. Larger batch = more stable gradients.

3. **Token-level input**: BPE instead of bytes. 4x shorter sequences
   at same information density. More compute-efficient.

4. **Gradient accumulation with torch.compile**: compile the forward pass
   for additional speedup on RTX 5090.

5. **More aggressive LR schedule**: Current warmup is 2000 steps (13 min of
   training). Could do 500 steps and spend more time at peak LR.

6. **Mixed data**: Add TinyStories to MiniPile for better narrative quality.

7. **Checkpoint EVERY 2K steps** not 10K (current: 65 min between saves).

### The FASTEST Path to Competitive

Based on Codex (Combo 5 Pragmatist):
Token-level Sutra v0.4 + bf16 + larger batch + GRUCell write + RoPE.
Train on MiniPile + TinyStories (~6GB).
This should be 2-4x faster AND produce better quality from tokenization alone.

Prepare this script NOW so it launches immediately after current run + eval.
