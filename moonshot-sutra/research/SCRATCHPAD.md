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

**Regularization IS compression**: L1/L2/dropout force simpler representations.
Sutra's patch structure provides STRUCTURAL regularization — info must flow through
summary bottleneck. This is why smaller Sutra (1.2M) beats larger transformer (6.5M)
on small data — the bottleneck prevents overfitting naturally.
