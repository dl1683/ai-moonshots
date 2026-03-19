# Sutra Stage Analysis: Deep Theoretical Design

## The Meta-Architecture: Stage-Superposition State Machine

Before analyzing individual stages, we must define HOW stages compose.

### The State Vector

At any point during processing, each position i has a STATE VECTOR:

```
S_i = (stage_distribution, features, confidence, routing_plan)
```

Where:
- `stage_distribution`: probability over stages [p1, p2, p3, p4, p5, p6, p7]
  e.g., [0, 0, 0.2, 0.6, 0.15, 0.05, 0] = "mostly routing, partially constructing"
- `features`: the actual representation (dim-dimensional vector)
- `confidence`: scalar — how "done" is this position? Controls readout.
- `routing_plan`: which other positions this one needs to communicate with

The stage distribution is NOT externally imposed — it EMERGES from the features
themselves. A position that has integrated all needed context naturally shifts
toward Stage 6/7. A position still missing evidence stays in Stage 4/5.

### The Transition Dynamics

Each processing step, for each position:
1. Compute current stage distribution from features (a learned classifier)
2. Apply the MIXTURE of stage operations weighted by the distribution
3. Update features, confidence, and routing plan
4. Positions with confidence > threshold emit output (Stage 7)
5. Positions whose Stage 7 verify FAILS loop back (confidence drops)

This is a CONTINUOUS dynamical system, not discrete stages.
The "stages" are ATTRACTORS in the state space — the system flows through
them, spending more time near the attractors that match the current need.

### Mathematical Formulation

Let h_i^t be the feature vector of position i at step t.
Let σ_i^t = softmax(W_stage · h_i^t) be the stage distribution.

The update rule:

```
h_i^{t+1} = Σ_s σ_i^t[s] · F_s(h_i^t, context_i^t)
```

Where F_s is the stage-specific operation and context_i^t depends on the stage:
- F_1: segmentation/compression transform
- F_3: local GRU/conv applied to within-patch neighbors
- F_4: message passing from medium + sparse retrieval
- F_5: gated memory write/update
- F_6: halting/depth decision
- F_7: readout projection + verification

Since σ is soft, the position does a WEIGHTED MIXTURE of all operations.
Early in processing: σ peaks at Stage 3 (construction).
Later: σ shifts to Stage 4-5 (routing/memory).
Final: σ shifts to Stage 7 (readout).

The shift happens NATURALLY through the features — no external scheduler.

---

## STAGE 1: Segmentation / Compression

### The Problem
Convert a raw byte stream into processing units that preserve predictive
structure while minimizing unnecessary sequence length.

### Why This Matters
Every downstream stage operates on UNITS. If units are wrong (splitting "New York"
into "New" + "York"), every downstream stage wastes capacity fixing the split.
The unit boundaries determine the GRAIN of all subsequent computation.

### Current Approaches and Their Flaws

**BPE (Byte-Pair Encoding)**:
- Learns merge rules from corpus statistics
- Flaws: fixed at training time, can't adapt to context, splits rare words badly,
  different tokenizers for different languages, "the" always = 1 token regardless
  of whether it matters

**Byte-level (our v0.4)**:
- No tokenization at all — raw bytes
- Flaws: 4-8x longer sequences, each byte carries little semantic info,
  the model must learn character-level patterns that BPE handles for free

**BLT (Meta, 2024)**:
- Dynamic byte patching based on local entropy
- Better: adapts patch boundaries to content
- Still: not learned end-to-end with the model

### First-Principles Derivation

From information theory: the optimal segmentation maximizes the mutual
information between each segment and the prediction target, subject to
a constraint on the number of segments (compression).

Formally: find segmentation boundaries b_1, ..., b_K that maximize:

```
Σ_k I(segment_k; target) subject to K ≤ budget
```

This is a rate-distortion problem. The optimal solution places boundaries
where the PREDICTIVE INFORMATION changes most — at concept boundaries.

Between "the" and "cat": low information change → DON'T segment here,
keep as one unit if possible.
Between "cat" and "sat": high information change → DO segment.
Between "New" and "York": low change → keep together.

### Novel Mechanism: Entropy-Gradient Segmentation

Instead of fixed patches OR BPE, use the MODEL'S OWN prediction entropy
to decide where to place boundaries:

1. Compute per-byte prediction entropy: H_i = -Σ p(x_i|context) log p(x_i|context)
2. Place boundaries where entropy SPIKES (= surprisal = concept boundary)
3. Boundaries are soft (Gumbel-Softmax) so gradients flow
4. The model learns its own optimal segmentation end-to-end

This is fundamentally different from BPE (corpus statistics) and BLT (fixed
entropy threshold). The segmentation is LEARNED, ADAPTIVE, and OPTIMIZED
for the downstream prediction task — not for compression alone.

### How This Interacts with Stage-Superposition

In a stage-superposition system, segmentation doesn't happen "first."
A position might be partially segmented (deciding its boundary) while
simultaneously starting local construction. The boundary decision itself
depends on downstream processing — creating a beneficial feedback loop.

Early in processing: boundaries are soft/uncertain.
As the model processes: boundaries sharpen based on what helps prediction.
This is like how the brain's perceptual grouping is influenced by top-down
expectations — you "hear" word boundaries partly because you expect words.

---

## STAGE 2: State Initialization / Addressing

### The Problem
Each unit needs to know: (a) WHAT it is (content), and (b) WHERE it is
(position relative to other units). Without good addressing, communication
(Stage 4) can't target the right positions.

### Why This Matters
The addressing scheme determines the GEOMETRY of the model's internal space.
If positions can't tell how far apart they are, long-range dependencies are
invisible. If positions can't distinguish content, retrieval returns noise.

### First-Principles Derivation

From a geometric perspective: the ideal addressing scheme embeds positions
into a metric space where DISTANCE correlates with RELEVANCE.

For language:
- Adjacent bytes: almost always relevant → distance should be small
- Same-sentence tokens: often relevant → moderate distance
- Cross-paragraph tokens: rarely relevant → large distance
- EXCEPT: specific patterns (pronouns → referents, variables → definitions)
  where relevance is HIGH despite large positional distance

This means the ideal metric space is NOT Euclidean (where distance = position gap).
It should be a SEMANTIC metric where distance reflects relevance, not position.

### Novel Mechanism: Content-Modulated Positional Encoding

Standard: position embedding = f(index)
RoPE: rotation = g(relative_distance)

Proposed: position embedding = f(index, CONTENT)

The positional encoding of a pronoun "it" should POINT TOWARD its referent,
not just encode "I am at position 47." The addressing is DYNAMIC — the
position embedding changes as the model processes, pulling related positions
closer in the embedding space.

Implementation:
```
pos_i = RoPE(i) + α · attention_to_content(h_i, global_summary)
```

The α term shifts the position embedding based on content, creating a
LEARNED metric space where semantically related positions are closer.

### How This Interacts with Stage-Superposition

In superposition, addressing is continuously refined. Early: pure positional.
Later: content-modulated positional. This means Stage 2 isn't just initialization
— it's an ongoing process that sharpens as the model understands the input.

---

## STAGE 3: Local State Construction

### The Problem
Build rich features from nearby units before any long-range communication.
This is where character patterns become word features, word features become
phrase features.

### Why This Matters
The QUALITY of local features determines what can be communicated in Stage 4.
If local construction produces poor features, no amount of routing fixes the problem.
"Garbage in, garbage out" at the patch level.

### First-Principles Derivation

The optimal local processor captures ALL within-unit dependencies while being
maximally INFORMATIVE about the prediction target.

From the MI perspective: maximize I(local_features; target | context)
subject to dim(local_features) ≤ budget.

This is the information bottleneck (Tishby 2000): compress the input while
preserving information relevant to prediction. The GRU naturally does this —
its gating mechanism decides what to retain and what to discard.

### What Makes GRU Good Here

The GRU's update gate z and reset gate r implement a form of information
bottleneck: z decides how much of the past to keep (compression), r decides
how much of the past to use for the new candidate (relevance selection).

For within-patch processing (4 bytes), this is near-optimal because:
1. The sequence is very short (4 steps) — no vanishing gradient issue
2. The gating handles the variable-information-per-byte problem naturally
3. Sequential processing is correct here (byte order matters within words)

### Novel Enhancement: Multi-Grain Local Construction

Instead of processing at ONE granularity (4 bytes), process at MULTIPLE:
- Grain 1: individual bytes (character-level features)
- Grain 2: byte pairs (bigram features)
- Grain 3: full patch (word/morpheme features)

Combine via learned weighted sum. This is the SAME principle as the
two-regime MI finding: different information lives at different scales.

Implementation: parallel 1D convolutions with kernel sizes [1, 2, 4]
plus the GRU, all feeding into a gated combination.

### How This Interacts with Stage-Superposition

In superposition, some positions may need DEEPER local construction
(rare words, technical terms) while others are done quickly ("the", "a").
The stage distribution naturally handles this: complex positions stay
in Stage 3 longer (higher σ[3]) before transitioning to Stage 4.

---

## STAGE 4: Communication / Routing

### The Problem
Move information between positions so that distant relevant evidence
enters each position's working state. THIS IS THE CRITICAL STAGE.

### Why This Matters
"Wrong answers are born when evidence isn't routed." — Codex R2.
If position 50 needs information from position 5, and the communication
mechanism can't deliver it, no amount of processing at position 50 helps.

### The Fundamental Tradeoff

Full attention: EVERY position can access EVERY other position.
Cost: O(n²). Quality: maximum. But most access is wasted (sparse attention
maps in practice show >90% of weights are near-zero).

Local only: each position accesses nearby positions.
Cost: O(n). Quality: misses long-range. Our MI analysis shows ~25% of
predictive information is beyond local reach.

### Why "Just Add Attention" Is Wrong for Sutra

Adding attention layers copies Jamba/Falcon-H1. It works but:
1. It's not derived from first principles
2. It doesn't exploit our MI analysis (two-regime structure)
3. It treats all positions equally (uniform O(n²)) when only ~25% of
   information is long-range

### First-Principles Derivation: What SHOULD Stage 4 Be?

From the two-regime MI finding:
- Local regime (d<10, alpha=0.94): fast decay, handled by message passing
- Global regime (d>10, alpha=0.26): slow decay, needs targeted routing

The OPTIMAL Stage 4 mechanism should:
1. Do DENSE local communication (message passing, cheap, handles 75% of MI)
2. Do SPARSE targeted global communication (handles 25% of MI)
3. The sparsity pattern should be CONTENT-DEPENDENT (not fixed, not random)
4. The routing should IMPROVE over processing rounds (grown sparsity)

### Novel Mechanism: Information-Gradient Routing

Instead of attention (compute all pairwise scores) or fixed sparse (predetermined):

**Route information along INFORMATION GRADIENTS.**

Like how heat flows from hot to cold, information should flow from where
it IS to where it's NEEDED. Each position broadcasts: "I have information
about X." Each position also broadcasts: "I need information about Y."
Information flows when supply matches demand.

Implementation:
1. Each position computes a SUPPLY vector: s_i = what info I have
2. Each position computes a DEMAND vector: d_i = what info I need
3. Routing weight w_{ij} = similarity(s_j, d_i) — supply j matches demand i
4. Only route when w_{ij} > threshold (sparse, O(n·k_effective))
5. The threshold adapts per round (starts high = only strong matches, lowers as needed)

This is NOT attention (which computes ALL pairwise). This is MATCHMAKING
— supply and demand find each other through the medium, like a market.

The supply/demand vectors live in a LOW-DIMENSIONAL space (e.g., 16-32 dims)
while the actual features are high-dimensional (e.g., 5120 dims). This means
the routing decision is cheap (16-dim dot product) but the routed information
is rich (5120-dim features).

### How This Differs from Existing Approaches

- vs Attention: O(n·k) not O(n²). Content-dependent sparsity. Separate routing from content.
- vs Sparse attention: learned routing, not top-k of scores. Supply/demand framework.
- vs Message passing: goes beyond local. Targeted, not diffusive.
- vs Mixture of Experts: routes INFORMATION, not computation. Every position participates.

### How This Interacts with Stage-Superposition

Positions in different stages have different supply/demand profiles:
- Stage 3 positions: low supply (still constructing), high demand (need context)
- Stage 5 positions: high supply (rich features), moderate demand (filling gaps)
- Stage 7 positions: high supply (ready to share), low demand (already confident)

The routing naturally connects Stage 5 suppliers with Stage 3 demanders.
Information flows from "done" positions to "still processing" positions.
This is emergent load-balancing — no explicit scheduler needed.

---

## STAGE 5: State Update / Memory Write

### The Problem
After receiving new information from Stage 4, each position must decide:
what to KEEP, what to OVERWRITE, what to FORGET.

### Why This Matters
Without selective memory, each communication round overwrites the previous
state. The model can't build up understanding over multiple rounds — it just
sees the latest message, not the accumulated context.

### First-Principles Derivation

From Bayesian inference: each new piece of evidence should UPDATE the
current belief, not REPLACE it. The update should be proportional to
the RELIABILITY of the new evidence and the UNCERTAINTY of the current state.

```
h_new = (1 - gate) · h_old + gate · evidence
gate = σ(W · [h_old, evidence, uncertainty])
```

High uncertainty + reliable evidence → large gate (accept new info).
Low uncertainty + unreliable evidence → small gate (keep current belief).

This is EXACTLY what GRU gates do — but we should make the gating
EXPLICITLY dependent on uncertainty (confidence from Stage 6).

### Novel Enhancement: Bayesian-Inspired Gated Write

Standard GRU: gate = σ(W · [h, x])
Proposed: gate = σ(W · [h, x, CONFIDENCE(h)])

Where CONFIDENCE(h) is the model's own estimate of how certain it is
about position h. Positions with high confidence resist updates (like
strong prior beliefs). Positions with low confidence accept updates
readily (like uninformative priors).

This naturally prevents the "good state destroyed by noisy update" problem
that plagues deep message-passing networks (over-smoothing).

### How This Interacts with Stage-Superposition

Memory write is ALWAYS active (it's part of every processing step).
But the INTENSITY of writing varies:
- Positions in Stage 3: mostly writing from local construction (self-generated)
- Positions in Stage 4/5: writing from external communication (received info)
- Positions in Stage 7: mostly reading (verifying, not updating)

---

## STAGE 6: Compute Control / Deliberation

### The Problem
Some inputs need 2 processing rounds. Some need 20. Allocate compute
proportional to difficulty without wasting resources on easy inputs.

### First-Principles Derivation

The optimal compute allocator minimizes TOTAL compute subject to a
quality constraint. This is a scheduling problem:

```
minimize Σ_i rounds_i  subject to  quality_i ≥ threshold ∀i
```

From our MI analysis: easy tokens (high local MI, low global MI) need
few rounds. Hard tokens (high global MI) need many rounds.

The PonderNet geometric prior is one solution. But the stage-superposition
framework offers a BETTER one: the compute control IS the stage distribution.

Positions that are "done" (high σ[7]) stop consuming compute naturally.
Positions that need more work (high σ[3-5]) keep processing.
No explicit halting mechanism needed — the stage distribution IS the control.

### How This Emerges from Stage-Superposition

In a standard architecture, compute control is a bolt-on mechanism (PonderNet).
In stage-superposition, it's INTRINSIC. The confidence signal that determines
stage distribution also determines compute allocation. They're the same thing.

High confidence → σ shifts to Stage 7 → position stops updating → no compute.
Low confidence → σ stays in Stage 3-5 → position keeps processing → more compute.

This is more elegant AND more expressive than PonderNet.

---

## STAGE 7: Readout / Decode / Verify

### The Problem
Convert internal state to output tokens. Optionally verify correctness
and loop back if verification fails.

### The Decode-Verify Loop

```
7a: Readout — project features to vocabulary logits
7b: Sample/select — choose output token(s)
7c: Verify — check: is this output consistent with the full context?
7d: If verification fails → reduce confidence → Stage 4 (reroute)
    If verification passes → emit output → move to next position
```

This loop is what makes Stage 7 NOT terminal. Bad outputs get CAUGHT
and sent back for more processing. Good outputs proceed.

### How This Creates Self-Correction

In autoregressive generation: the model generates, verifies, and if the
verification score is too low, it can:
1. Increase depth (more Stage 4-5 rounds) before trying again
2. Route to different context (different Stage 4 targets)
3. Express uncertainty (abstain rather than hallucinate)

This is built-in self-correction WITHOUT an external verifier model.
The verification IS the model checking its own confidence.

---

## PROPOSED COMPLETE SYSTEM

Putting it all together:

```
Input bytes
    ↓
[Stage 1] Entropy-gradient segmentation (soft, learned boundaries)
    ↓
[Stage 2] Content-modulated positional encoding (dynamic, refining)
    ↓
INNER LOOP (each position at its own stage):
    ↓
[Stage 3] Multi-grain local construction (conv + GRU, parallel grains)
    ↓
[Stage 4] Information-gradient routing (supply/demand matchmaking)
    ↓
[Stage 5] Bayesian-inspired gated write (confidence-weighted updates)
    ↓
[Stage 6] Implicit via stage distribution (no explicit controller)
    ↓
CHECK: is any position confident enough for Stage 7?
    ↓
[Stage 7a] Readout + [Stage 7b] Decode + [Stage 7c] Verify
    ↓
If verify fails: confidence drops → back to inner loop
If verify passes: emit output
```

All stages happen SIMULTANEOUSLY for different positions.
The "loop" is really a continuous flow through the state graph.
No global clock, no synchronous layers, no fixed depth.

### Why This Should Be Better

1. **No wasted compute**: easy positions race through, hard ones get more time
2. **Self-correcting**: verify catches mistakes before they're emitted
3. **Naturally calibrated**: confidence IS the stage distribution
4. **Information-efficient**: supply/demand routing sends info WHERE it's needed
5. **Interpretable**: can read the stage distribution to understand what the model is doing
6. **Biologically plausible**: matches cortical processing (asynchronous, multi-scale, recurrent)

---

## ALTERNATIVE ARCHITECTURES (Same Vision, Different Implementations)

### Architecture A: "The Wavefront" (Pure Stage-Superposition)

The design above. Each position carries a stage distribution.
Soft mixture of all stage operations per position per step.

**Pros**: Maximally expressive. Continuous. Elegant math.
**Cons**: GPU batching nightmare. Every position does different work.
May collapse to uniform stage distribution (everything converges).

### Architecture B: "The Pipeline" (Hard Stage Assignment)

Instead of soft superposition, HARD-ASSIGN each position to one stage.
Positions advance through stages explicitly. At each step:
- Positions in Stage 3 → all run GRU together (batched efficiently)
- Positions in Stage 4 → all run routing together (batched efficiently)
- Positions in Stage 7 → all run readout together

A position advances when its "promotion score" exceeds a threshold.
Like an assembly line: each station processes a different batch of items.

**Pros**: GPU-friendly (batch by stage). Clear, debuggable. No collapse risk.
**Cons**: Less expressive (hard boundaries). Positions can get "stuck" at a stage.
Needs explicit promotion mechanism.

**Implementation**:
```
for each step:
    groups = group_positions_by_stage(stage_assignments)
    for stage_id, positions in groups:
        features[positions] = F_stage(features[positions], context)
    stage_assignments = update_stages(features, promotion_scores)
```

### Architecture C: "The Wave Pool" (Fixed Stages, Variable Frequency)

All positions go through ALL stages in fixed order (like a transformer).
BUT: the NUMBER OF ROUNDS at each stage varies per position.
Easy positions: 1 round of Stage 4. Hard positions: 8 rounds.

This is MEGABYTE-like but with per-position adaptive depth AT EACH STAGE.
The stage order is fixed; the TIME SPENT at each stage is variable.

**Pros**: Standard batching works. Stage order is well-defined.
Compatible with existing GPU kernels.
**Cons**: Less flexible than A or B. Still sequential at the stage level.

**Implementation**: PonderNet-style halting per stage, not per model.

### Architecture D: "The Market" (Decentralized Stigmergic)

No explicit stages at all. Each position is an AGENT in a market.
Agents have: state (features), inventory (what info they have),
needs (what info they want), budget (remaining compute).

Each step:
1. All agents POST their needs to the medium (O(n) broadcast)
2. All agents CHECK the medium for matching supply (O(n) scan)
3. Matched pairs TRADE information (O(k) per agent)
4. Each agent UPDATES its state based on received info
5. Agents with empty needs and full confidence → output

The "stages" EMERGE from agent behavior:
- Early: agents need everything → lots of trading (= Stage 4)
- Middle: agents have info, refining → selective trading (= Stage 5)
- Late: agents are done → outputting (= Stage 7)

**Pros**: Truly decentralized. No explicit stages. Most biologically faithful.
Elegantly simple at the concept level. Grown sparsity is natural.
**Cons**: Hard to train (reward signal is sparse). May not converge.
Market dynamics can oscillate. Very novel = very risky.

### Architecture E: "The Hybrid Wave" (Best of A + C)

Fixed stage ORDER but soft stage MIXING within each step.
Each step runs stages 3→4→5 in order, but the INTENSITY of each
stage operation is modulated by the position's stage distribution.

```
for each step:
    h = σ[3] * F_3(h) + (1-σ[3]) * h  # Soft local construction
    h = σ[4] * F_4(h) + (1-σ[4]) * h  # Soft routing
    h = σ[5] * F_5(h) + (1-σ[5]) * h  # Soft memory write
```

Positions that are "past" Stage 3 have σ[3] ≈ 0 (skip it).
Positions that need more routing have σ[4] ≈ 1 (full routing).

**Pros**: GPU-friendly (fixed order, batched). Soft modulation gives
expressiveness. No collapse risk (bounded by stage order).
Compatible with standard training.
**Cons**: Less flexible than A (fixed order). Stage mixing is simple.
The intensity modulation might not capture complex dependencies.

### COMPARISON TABLE

| Architecture | GPU-Friendly | Expressiveness | Novelty | Risk | Trainability |
|-------------|-------------|---------------|---------|------|-------------|
| A: Wavefront | LOW | HIGHEST | HIGH | HIGH | UNKNOWN |
| B: Pipeline | HIGH | MEDIUM | MEDIUM | MEDIUM | GOOD |
| C: Wave Pool | HIGHEST | LOW | LOW | LOW | BEST |
| D: Market | LOW | HIGH | HIGHEST | HIGHEST | UNKNOWN |
| E: Hybrid Wave | HIGH | MEDIUM-HIGH | MEDIUM | LOW | GOOD |

### RECOMMENDATION

**Start with E (Hybrid Wave).** It's GPU-friendly, trainable, and captures
the stage-superposition vision in a practical form. The intensity modulation
gives per-position adaptive processing without the batching nightmare.

**If E works**: graduate to A (full Wavefront) for maximum expressiveness.
**If E fails**: fall back to C (Wave Pool) which is just per-stage PonderNet.
**D (Market)** is the long-term moonshot — save for when the basic idea is proven.

### Key Unknowns

1. Can the stage distribution be LEARNED stably with gradient descent?
2. Does supply/demand routing converge or oscillate?
3. How do we handle BATCHING if each position is at a different stage?
   (GPU parallelism requires uniform computation per position)
4. Is soft stage mixing expressive enough, or do we need hard routing?
5. What's the training objective for the stage classifier?
