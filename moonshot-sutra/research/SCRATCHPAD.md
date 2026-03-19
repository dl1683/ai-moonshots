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

*(Debates will be appended here as they happen)*
