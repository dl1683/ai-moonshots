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

---

## DEEP DIVE: Information-Gradient Routing (The Core Novel Mechanism)

### Why This Is THE Most Important Component

Stage 4 determines whether the right evidence reaches the right position.
If we solve Stage 4 with a genuinely novel mechanism (not attention, not
message passing, not sparse attention), we have a REAL contribution.

### Mathematical Formalization

Each position i has:
- SUPPLY vector: s_i ∈ R^d_route (what information this position offers)
- DEMAND vector: d_i ∈ R^d_route (what information this position needs)
- VALUE vector: v_i ∈ R^d_model (the actual information to transmit)

Routing weight from j to i:

```
w_{ij} = max(0, <s_j, d_i> / √d_route)  [ReLU, not softmax]
```

ReLU instead of softmax because:
1. Not all positions NEED external info (some demands are satisfied → w=0)
2. A position can receive from ZERO sources (no forced normalization)
3. Sparsity is NATURAL (most supply/demand pairs don't match)

The received information at position i:

```
received_i = Σ_j w_{ij} · v_j / (Σ_j w_{ij} + ε)
```

Normalized by total weight received (not total positions).

### How This Differs From Attention — Mathematically

**Standard attention:**
```
Q = W_Q · h,  K = W_K · h,  V = W_V · h
attn_{ij} = softmax_j(Q_i · K_j / √d)
output_i = Σ_j attn_{ij} · V_j
```

**Information-gradient routing:**
```
S = W_S · h,  D = W_D · h,  V = W_V · h
route_{ij} = ReLU(S_j · D_i / √d_route)
output_i = Σ_j route_{ij} · V_j / (Σ_j route_{ij} + ε)
```

The differences:
1. **ReLU vs softmax**: routing can be ZERO (no forced attention to something)
2. **Supply vs Key**: supply is what you HAVE, key is what you ARE.
   Supply is a FUNCTION of the content — "I have information about cats."
   Key is the content itself — "I am the word cat."
3. **Demand vs Query**: demand is what you NEED, query is what you WANT.
   Demand reflects the GAP — "I need a subject for my verb."
   Query reflects the current state — "I am looking for something."
4. **Asymmetry**: supply/demand is fundamentally asymmetric. Position j may
   supply to i without i supplying to j. Attention is also asymmetric but
   doesn't frame it as supply/demand.

### The Deeper Difference: ROUTING DIMENSIONS ≠ CONTENT DIMENSIONS

This is the KEY insight. In attention, Q and K live in the SAME space as V
(or a projection of it). Routing decisions and content are entangled.

In information-gradient routing, supply/demand live in a SEPARATE low-dimensional
routing space (d_route << d_model). This means:
- Routing is CHEAP (d_route × d_route operations, e.g., 32×32)
- Content is RICH (d_model × d_model, e.g., 5120×5120)
- The routing can be computed FIRST, then only matched pairs exchange content

This is like a postal system: the ADDRESS (routing) is small, the PACKAGE
(content) is large. You don't need to examine every package to route mail —
you just read the address.

### Computational Complexity

**Naive**: O(n² · d_route) for all pairwise supply/demand scores.
Still quadratic, but with d_route << d_model, the constant is much smaller.

**Efficient**: If supply/demand vectors are sparse or structured:
- LSH on supply/demand vectors → O(n · k_bucket) per position
- Product quantization → O(n · sqrt(n)) approximate routing
- For Sutra: broadcast supply to medium, each position scans medium → O(n · d_route)

**With the medium**:
1. All positions WRITE supply to shared medium: O(n)
2. Medium aggregates supplies (e.g., by clustering): O(n)
3. Each position READS from medium to find matching supplies: O(n · k)
4. Total: O(n · k) where k = number of matches per position

This IS O(n) if k is constant (which it is — each position needs ~k supplies).

### Connection to Biological Systems

**Brain**: Neurotransmitter signaling IS supply/demand. Dopamine neurons
SUPPLY a signal. Target neurons have RECEPTORS (demand). Connection
strength depends on receptor-ligand match, not pairwise computation.

**Immune**: Antigen-presenting cells SUPPLY pathogen fragments on MHC molecules.
T-cells DEMAND matching fragments via their TCR. Connection = recognition.
This is literally supply/demand matchmaking in biology.

**Market**: Sellers supply goods, buyers demand them. Price discovery happens
through the medium (the market), not pairwise negotiation between all
seller-buyer pairs.

### Training Signal

The supply/demand vectors are learned end-to-end through the language modeling loss.
When a routing decision helps prediction (correct info reaches the right place),
the gradient reinforces the supply/demand alignment. When routing fails,
the gradient adjusts supply/demand to route differently next time.

No auxiliary loss needed — the language modeling loss ITSELF trains the routing.

### Potential Failure Modes

1. **Routing collapse**: all positions learn the same supply/demand → no differentiation.
   Fix: diversity regularization on supply vectors (like V(D)J diversity).

2. **Routing oscillation**: supply/demand keeps changing each round → no stable routing.
   Fix: momentum on supply/demand (EMA of previous rounds).

3. **Dead routing**: some positions never match anything → isolated.
   Fix: guaranteed minimum connectivity (always route to k nearest neighbors as fallback).

4. **Gradient sparsity**: ReLU routing means zero gradient for non-matched pairs.
   Fix: use smooth approximation of ReLU (softplus) or straight-through estimator.

### Proposed Experiments (When Compute Is Available)

1. **Supply/demand vs attention**: matched params, same model, swap ONLY Stage 4.
   Compare BPB, MQAR accuracy, and routing patterns.

2. **Routing dimension sweep**: d_route = {8, 16, 32, 64, d_model}.
   Does separating routing from content actually help?

3. **Sparsity analysis**: measure how sparse the routing matrix is in practice.
   If >90% sparse, the O(n·k) claim holds.

4. **Interpretability**: visualize supply/demand vectors — do they correspond
   to semantic roles? (e.g., "I have a subject" / "I need a subject")

---

## CODEX HARD CHALLENGE: Critical Flaws Identified

**Novelty 7/10. Practicality 3/10. Chance 4/10 (modest) / 2/10 (robust).**

### Flaw 1: Stage Collapse
softmax(W·h) router will collapse to uniform or single-stage domination.
Nothing prevents all positions from converging to same stage distribution.
**Fix needed**: load balancing, capacity constraints, monotonic advancement,
routing noise, or explicit supervision.

### Flaw 2: Supply/Demand ≈ Attention Without Global Constraints
Without column constraints (supply budgets) and competition between queries
for scarce supply, information-gradient routing IS attention with different names.
**What makes it genuinely new**: ENTROPIC OPTIMAL TRANSPORT formulation.
Route = argmax Σ R_ij · match_ij - ε·H(R) subject to supply/demand constraints.
This IS different from attention because it has BUDGETS — a source can't be
overused by many queries simultaneously.

### Flaw 3: Bayesian Write ≈ GRU Unless Uncertainty Is Real
Adding "confidence" as just another input to a gate = GRU with extra feature.
**What makes it genuinely different**: track (mean, variance) as the state.
Updates become PRECISION-WEIGHTED Kalman updates, not just gated averages.
High variance state + low variance evidence → large update (correct Bayes).

### Flaw 4: GPU Batching Kills Free Per-Token Assignment
**Fix**: monotonic constrained routing. Tokens only STAY or ADVANCE one stage
per step. Chunk-level batching (whole chunks at same stage). This gives
coarse-grained routing that hardware can handle.

### Flaw 5: Underconstraint (BIGGEST)
End-to-end gradient descent doesn't care about staged interpretation.
Will find degenerate solutions that ignore the structure.
**Fix**: combine structural constraints (monotonic advancement) with
explicit stage objectives (e.g., Stage 3 loss = local prediction quality,
Stage 5 loss = state retention accuracy).

### REVISED ARCHITECTURE: Constrained Stage-Superposition

Based on Codex challenge, the viable version is Architecture B+E hybrid:

```
MONOTONIC STAGE ADVANCEMENT with SOFT INTENSITY:

Each position starts at Stage 1. Each step, it can:
- STAY at current stage (if not ready to advance)
- ADVANCE to next stage (if promotion criterion met)
- NEVER go backward (monotonic constraint)

Within current stage, INTENSITY is soft (how much of that stage's
operation to apply). But the STAGE ITSELF advances monotonically.

Load balancing: cap on how many positions can be at each stage.
Chunk-level: whole patches advance together (GPU-friendly).
```

This addresses ALL five flaws:
1. No collapse: monotonic advancement prevents staying in one stage
2. Supply/demand with budgets: optimal transport formulation
3. Real uncertainty: (mean, variance) state with Kalman updates
4. GPU-friendly: chunk-level advancement, batch by stage
5. Constrained: monotonic + per-stage losses fight degenerate solutions

### Key Unknowns (Updated)

1. Does monotonic advancement lose expressiveness vs free movement?
2. Can optimal transport routing be computed efficiently? (Sinkhorn: O(n²) iterative)
3. Does tracking (mean, variance) double the state size (2x memory)?
4. How granular should chunks be for stage advancement? (patch-level? sentence-level?)
5. What per-stage auxiliary losses work best?
6. Does separating routing dims from content dims ACTUALLY help at scale?

---

## DEEP DIVE: Entropic Optimal Transport as Routing

### Why This IS Different From Attention

Standard attention: R_ij = softmax_j(q_i · k_j / √d). Each row normalized
independently. No constraint on how much any key is used. Popular keys
accessed by ALL queries simultaneously → attention sinks.

Optimal transport: find R that maximizes total match quality UNDER CONSTRAINTS:

```
max_R Σ_ij R_ij · match(s_j, d_i) - ε · Σ_ij R_ij log R_ij
s.t.  Σ_j R_ij = demand_i    (each query gets exactly its demand)
      Σ_i R_ij ≤ supply_j    (each source gives at most its supply)
      R_ij ≥ 0
```

### The Key Differences (Mathematical)

1. **Column constraints**: source j has LIMITED supply. Can't serve everyone.
   In attention: no limit, popular keys attract unlimited attention.

2. **Global coupling**: routing of query i affects all other queries
   (through shared supply constraints). In attention: queries are independent.

3. **Balanced routing**: prevents attention sinks. Information is DISTRIBUTED
   across sources, not concentrated in a few popular ones.

4. **Information-theoretic**: entropy term maximizes info transfer under capacity.

### Sinkhorn Algorithm (Efficient Computation)

Solve via alternating row/column normalization:

```
K = exp(match_matrix / ε)  # Kernel matrix
for _ in range(n_iter):     # ~10-20 iterations
    u = demand / (K @ v)    # Row normalization
    v = supply / (K^T @ u)  # Column normalization
R = diag(u) @ K @ diag(v)   # Optimal transport matrix
```

Complexity: O(n² × n_iter). For n=128 patches: 128² × 20 = 327K ops.
Comparable to attention but with GLOBAL optimality guarantees.

### Why This Matters for Sutra

At the PATCH level (n=128, not token n=512), the cost is manageable.
And the quality improvement from balanced, constrained routing could be
significant — preventing the failure modes that make pure message passing
insufficient (over-squashing, attention sinks, information concentration).

This IS mathematically derived, NOT copied from existing architectures.
Optimal transport theory predates modern ML. Applying it to neural routing
is principled and novel for language modeling.

---

## DEEP DIVE: Kalman State Updates (Stage 5)

### Why Track Uncertainty

Standard: h ∈ R^d (point estimate). No uncertainty.
Proposed: (μ, log_σ²) ∈ R^(2d). Mean + variance.

Kalman update combines prior belief with new evidence OPTIMALLY:

```
K = σ²_prior / (σ²_prior + σ²_evidence)   # Kalman gain
μ_new = μ_prior + K · (evidence - μ_prior)  # Updated mean
σ²_new = (1 - K) · σ²_prior                 # Updated (reduced) variance
```

### Why This Prevents Over-Smoothing

As a position accumulates evidence: σ² decreases → K decreases →
harder to overwrite. Information-theoretic STIFFNESS.

This is the PRINCIPLED solution to the GNN over-smoothing problem.
Positions that have gathered enough evidence RESIST further updates.
Positions that lack evidence WELCOME updates. No tunable hyperparameter
— the behavior emerges from Bayesian inference.

### Practical Implementation

State: (μ, log_σ²) where log_σ² prevents negative variance.

When receiving message m with estimated noise σ²_msg:
```
K = sigmoid(W_K · [log_σ², m, σ²_msg])  # Learned Kalman gain
μ_new = μ + K * (m - μ)                  # Precision-weighted update
log_σ²_new = log_σ² + log(1 - K + ε)    # Variance reduction
```

The learned K approximates the true Kalman gain but adapts to
non-Gaussian distributions. The variance reduction is monotonic
— each update DECREASES uncertainty (as it should).

### Connection to Stage 6 and 7

**Stage 6 (Compute Control)**: σ² IS the confidence signal.
High variance = uncertain = need more processing.
Low variance = confident = can advance to output.
No separate halting mechanism needed — variance IS the controller.

**Stage 7 (Verify/Abstain)**: σ² determines abstention.
If σ² > threshold at readout time → abstain ("I'm not sure").
This gives NATIVE calibration — the model KNOWS what it doesn't know.

### Cost

2x state size (mean + variance). 2x memory. ~1.5x compute
(extra ops for variance update). Worthwhile if it eliminates
over-smoothing AND gives free calibration AND enables adaptive depth.

---

## DEEP DIVE: Entropy-Gradient Segmentation (Stage 1)

The optimal boundary placement maximizes I(segment; future) - lambda*|segments|.
Approximation: boundary where model's OWN prediction entropy spikes.
Connection to superposition: boundaries REFINE during processing (dynamic segments).

## DEEP DIVE: Content-Modulated Addressing (Stage 2)

Multi-level: RoPE(position) + LEARNED_ROLE(content) + CONTEXT_SHIFT(h, global).
Dynamic addressing: positions shift in semantic space during processing.
Connects to OT routing: match score = content_match + address_proximity.
This naturally implements two-regime MI: local favored, content overrides distance.

---

## THE COMPLETE SUTRA NEXT-GEN SYSTEM (All Stages Unified)

Combining all deep dives:

1. **Segmentation**: Entropy-gradient boundaries, soft, end-to-end learned, dynamic
2. **Addressing**: Multi-level (position + role + context-shift), feeds into routing
3. **Local Construction**: Multi-grain (conv + GRU), parallel kernel sizes
4. **Communication**: Optimal transport routing with supply/demand budgets,
   separate routing dims (d_route << d_model), Sinkhorn solver
5. **State Update**: Kalman (mean, variance), precision-weighted, anti-over-smoothing
6. **Compute Control**: IMPLICIT via variance — high variance = keep processing
7. **Readout/Verify**: Variance-based abstention + decode-verify loop back to Stage 4

Execution: Constrained Stage-Superposition (monotonic advancement, chunk-level batching)
Architecture E (Hybrid Wave): fixed stage ORDER, soft intensity MODULATION

Novel mechanisms: OT routing (from transport theory), Kalman states (from Bayes),
entropy segmentation (from info theory), supply/demand (from market theory).
ALL derived from first principles. NONE copied from existing architectures.

---

## CRITICAL DESIGN PRINCIPLE: System Coherence

Every stage's design must consider: does this HELP or HURT the other stages?

### The Coherence Test

For any proposed mechanism at Stage N, ask:
1. Does Stage N's OUTPUT format match what Stage N+1 NEEDS as input?
2. Does Stage N PRESERVE information that later stages require?
3. Does Stage N CREATE signals that other stages can USE?
4. Do the inductive biases of Stage N ALIGN with those of other stages?

Example of INCOHERENCE (from OT routing test):
- Stage 3 (chunk pool) averaged features, destroying key-value associations
- Stage 4 (OT routing) needed key-value associations to route correctly
- Result: Stage 3's bias (averaging) fought Stage 4's need (specificity)

Example of COHERENCE (from Kalman + compute control):
- Stage 5 (Kalman) tracks variance as part of state update
- Stage 6 (compute control) uses variance to decide depth
- Stage 7 (verify) uses variance to decide abstention
- Result: ONE signal serves THREE stages. They lift each other.

### Design Rule: Every Stage's Output IS the Next Stage's Input

The representation flowing between stages must be designed HOLISTICALLY:
- After Stage 3 (local): features should preserve BOTH local patterns AND
  information needed for routing (keys, queries, content types)
- After Stage 4 (routing): received info should be tagged with SOURCE
  identity so Stage 5 can gate appropriately
- After Stage 5 (write): updated state should include calibrated uncertainty
  so Stage 6 can allocate compute correctly

This means: design the INTER-STAGE INTERFACE FIRST, then design each stage
to satisfy that interface. Like designing API contracts before implementations.

---

## ARCHITECTURE EXPLORATION: Playing With Combinations

### The Variables We Can Mix

For each stage, there are multiple mechanism OPTIONS. The space of possible
architectures is the PRODUCT of choices across stages. Let's enumerate the
most promising options per stage, then explore interesting combinations.

**Stage 1 Options:**
- A1a: Fixed byte patches (current v0.4)
- A1b: BPE token patches (token-level branch)
- A1c: Entropy-gradient adaptive patches (novel)
- A1d: Learned Gumbel-Softmax boundaries (end-to-end)

**Stage 2 Options:**
- A2a: Learned embeddings only (current)
- A2b: + RoPE on inter-patch interactions
- A2c: + Content-modulated dynamic addressing
- A2d: Hierarchical multi-level (byte + patch + section)

**Stage 3 Options:**
- A3a: GRU within patches (current v0.4)
- A3b: Conv + GRU (multi-grain)
- A3c: Tiny within-patch attention (2-head, 4 tokens)
- A3d: Mamba-style selective SSM within patch

**Stage 4 Options:**
- A4a: Local message passing + sparse top-k (current v0.4)
- A4b: Optimal transport routing with supply/demand budgets
- A4c: Information-gradient supply/demand (without OT constraint)
- A4d: Grown sparsity (routing table evolves via gradient magnitude)
- A4e: Hybrid: local msg pass + periodic OT global refresh

**Stage 5 Options:**
- A5a: Additive merge + MLP (current v0.4)
- A5b: GRUCell gated write
- A5c: Kalman (mean, variance) with precision-weighted updates
- A5d: DeltaNet-style explicit erase/write gates

**Stage 6 Options:**
- A6a: PonderNet with min_rounds (current v0.4)
- A6b: Implicit via Kalman variance (if using A5c)
- A6c: Sequence-level controller {2,4,8} passes
- A6d: No explicit control (fixed depth)

**Stage 7 Options:**
- A7a: Linear head + AR decoding (current)
- A7b: + Variance-based abstention (if using A5c)
- A7c: + Verifier head + reranking N=4
- A7d: Decode-verify loop (verify fails → back to Stage 4)

### Candidate Architecture Combinations

#### COMBO 1: "The Conservative" (Lowest Risk)
A1b + A2b + A3a + A4a + A5b + A6d + A7a
BPE tokens, RoPE, GRU patches, current msg+sparse, GRUCell write, fixed depth, standard AR.
This is basically v0.4 with tokens instead of bytes + two small fixes.

**Coherence analysis**: HIGH. Each change is small and independent.
BPE helps Stage 3 (GRU processes meaningful tokens). RoPE helps Stage 4
(sparse retrieval has distance info). GRUCell helps Stage 5 (selective update).
No component fights another.

**Expected outcome**: Modest improvement over v0.4. ~10-20% better BPB from
tokenization alone. Not revolutionary but safe.

**Experiments needed**: 1 run. Compare against v0.4 byte-level.

---

#### COMBO 2: "The Kalman Machine" (Novel + Coherent)
A1b + A2c + A3b + A4a + A5c + A6b + A7b
BPE tokens, content-modulated addressing, conv+GRU, current routing,
Kalman states, implicit variance control, variance-based abstention.

**Coherence analysis**: VERY HIGH. The Kalman variance is the UNIFYING SIGNAL:
- Stage 5 produces variance → Stage 6 uses it for depth → Stage 7 uses it for abstention
- Content-modulated addressing (Stage 2) uses the state uncertainty to adjust positions
- Three stages share ONE signal. Maximum coherence.

**Expected outcome**: Better calibration, natural adaptive depth, honest abstention.
May not improve raw BPB much but should improve RESPONSE QUALITY on hard tasks.

**Experiments needed**:
1. BPB comparison against Combo 1
2. Calibration test: does variance actually correlate with correctness?
3. Abstention test: does variance-thresholded output improve accuracy?

---

#### COMBO 3: "The Transport Network" (Most Novel)
A1c + A2d + A3b + A4b + A5c + A6b + A7d
Adaptive entropy patches, hierarchical addressing, conv+GRU, OT routing,
Kalman states, implicit control, decode-verify loop.

**Coherence analysis**: MEDIUM-HIGH. The OT routing needs good features
from Stage 3 (the lesson from our negative result). Conv+GRU provides
richer features than mean-pooling. Adaptive patches (Stage 1) match the
entropy structure. Kalman states flow through control and verification.

BUT: adaptive patches (A1c) creates variable-length chunks which complicates
Stage 3 (GRU expects fixed-length). Need padding or variable-length GRU.

**Expected outcome**: If it works, the most defensible novelty claim.
OT routing is mathematically derived. Kalman states are principled.
Adaptive segmentation matches MI analysis. Everything from first principles.

**Risk**: HIGH. Three novel components (entropy patches, OT routing, Kalman)
interacting. Any one failure cascades. The OT prototype already showed weakness.

**Experiments needed**:
1. Fix OT routing on retrieval task (don't mean-pool, use raw features)
2. Test adaptive segmentation alone (does it find real word boundaries?)
3. Test Kalman alone (does variance track correctness?)
4. Then combine the winners.

---

#### COMBO 4: "The Biological" (Cross-Domain Inspired)
A1d + A2c + A3a + A4d + A5c + A6b + A7d
Learned Gumbel boundaries, content-modulated addressing, GRU patches,
GROWN SPARSITY routing, Kalman states, implicit control, verify loop.

**Coherence analysis**: HIGH. The grown sparsity (routing table evolves like
fungal tubes) naturally produces routing patterns that Stage 5 can use.
The Kalman variance tells the routing WHERE to invest effort. Learned
boundaries adapt to the grown routing patterns.

This is the most BIOLOGICALLY faithful: grown connections (fungi),
Bayesian updating (brain), adaptive segmentation (retina), verification
(immune system double-check).

**Expected outcome**: Potentially the most parameter-efficient if grown
sparsity discovers good routing patterns. The routing table becomes a
learned structure that encodes text dependency geometry.

**Risk**: MEDIUM-HIGH. Grown sparsity needs many training steps to develop
meaningful structure. Early training may be poor because routing table
starts random. Need curriculum or warm-start.

**Experiments needed**:
1. Does grown sparsity develop meaningful patterns? (Measure KL between
   routing table at epoch 1 vs epoch 10 — should show structure)
2. Does gradient-driven routing match MI-measured dependency structure?
3. Compare Combo 4 routing patterns against learned attention patterns.

---

#### COMBO 5: "The Pragmatist" (Ship Fastest)
A1b + A2b + A3a + A4a + A5b + A6a + A7c
BPE tokens, RoPE, GRU patches, current msg+sparse, GRUCell write,
PonderNet halting, verifier reranking.

This is v0.4 with: tokens, RoPE, gated write, verifier head. Four small
improvements, each well-understood, no novel risk.

**Coherence analysis**: HIGH. All changes are additive, none conflicts.
The verifier head (Stage 7) is independent of other stages.

**Expected outcome**: Best BPB of any combination (tokenization helps most).
Verifier improves response quality. Not novel but PRACTICAL.

**Experiments needed**: 1 run. This should be the FIRST thing we train
after v0.4 production finishes. It's the highest-probability win.

---

### COMPARISON MATRIX

| Combo | Novelty | Coherence | Risk | Ship Speed | Expected Win |
|-------|---------|-----------|------|------------|-------------|
| 1: Conservative | LOW | HIGH | LOW | FAST | Modest (~20% BPB) |
| 2: Kalman Machine | MEDIUM | VERY HIGH | MEDIUM | MEDIUM | Better calibration |
| 3: Transport Network | HIGH | MEDIUM-HIGH | HIGH | SLOW | Most defensible if works |
| 4: Biological | HIGH | HIGH | MEDIUM-HIGH | SLOW | Most efficient if works |
| 5: Pragmatist | LOW | HIGH | LOW | FASTEST | Best BPB + response quality |

### RECOMMENDED SEQUENCE

1. **FIRST**: Ship Combo 5 (Pragmatist) — fastest path to competitive model
2. **SECOND**: Test Combo 2 (Kalman Machine) — adds novelty without high risk
3. **THIRD**: Test individual novel components from Combo 3/4 in isolation
4. **LONG-TERM**: Build toward Combo 3 or 4 as the full vision

This way we ALWAYS have a working competitive model while exploring novelty.
