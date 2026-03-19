# CLAUDE.md — Moonshot Sutra (सूत्र)

**This is a sub-project of [AI Moonshots](../CLAUDE.md). Always read the parent CLAUDE.md and the global `~/.claude/CLAUDE.md` for universal workflow rules, Codex review gates, and engineering discipline.**

---

## Project Identity

Sutra is a from-scratch edge AI model. The thesis: **compression = intelligence**. Just as Panini compressed all of Sanskrit into ~4,000 sutras, we compress intelligence into minimal parameters by starting from better mathematics — not by shrinking existing architectures.

Every existing small model inherits the assumptions of the large model paradigm. Sutra questions those assumptions from the ground up.

---

## Hard Rules

### No Inherited Paradigms
- **No pre-trained weights.** Everything trains from scratch.
- **No copying existing tokenizers or architectures** without a mathematical argument for why that specific design is optimal for our constraints. "Everyone uses it" is not a justification.
- **Every architecture decision requires a theoretical argument before implementation.** No "let's try it and see" without a hypothesis for why it should work.
- **No assumption that the answer is a neural network.** The search space includes ALL of ML (kernel methods, probabilistic models, Bayesian networks, reservoir computing, evolutionary systems, etc.), ALL biological intelligence (brains, immune systems, collective intelligence, gene networks), ALL mathematical frameworks (information theory, category theory, algebraic topology, statistical mechanics, optimal transport, dynamical systems), and any other system that processes information efficiently. Modern deep learning is one option, not the default.

### Mathematical First
- Design choices flow from math, not intuition or convention.
- If a component exists in the model, there must be a reason grounded in information theory, geometry, optimization theory, or physics — not "transformers do it this way."
- When in doubt, derive. Don't copy.

### Benchmarking is Mandatory
- Every capability claim must be benchmarked against existing edge models in the same parameter class.
- Primary baselines: Phi-4 (5.6B), Qwen3-4B, Gemma-3-1B, Molmo-1B, Ministral-3B, SmolLM series.
- We don't get to claim "best" without numbers against these.

---

## Definition of Done

Sutra is "done" when ALL of the following are true:
1. **Benchmark sweep**: Competitive or superior on major intelligence benchmarks (MMLU, ARC, HellaSwag, GSM8K, HumanEval, WinoGrande, TruthfulQA, etc.) against models in the same parameter class
2. **Demo-able**: A live, callable model that can be tested interactively
3. **Custom hard eval**: A suite of supremely difficult queries we design ourselves, tested head-to-head against local LLMs
4. **The David story**: The narrative is "this should be much weaker but it actually outperforms" — efficiency through better math, not more scale

---

## Design Constraints

### Parameter / Capacity Budget
- Target: **4B parameters or below** — but parameter count may not even be the relevant metric if the architecture is not a neural network
- The real constraint is: trainable from scratch on a single RTX 5090 (24GB VRAM), deployable on edge hardware
- If the design transcends neural networks, define the appropriate capacity metric and justify the budget

### Hardware
- **Training**: single NVIDIA RTX 5090 (24GB VRAM)
- **Inference target**: phones, laptops, embedded devices
- Design for the constraint, don't fight it

### Compute / Time
- Not a constraint. Run as much as needed. The architecture itself should make training fast — that's the point.

### Quantization-Native
- Designed for low precision from day one — not quantized after the fact
- Architecture choices must be robust to INT4/INT8 from the start

### Text Only
- **Text only.** No multimodal. No images, audio, video, or structured data.
- Multimodal is a future project when we have the resources. Not in scope for Sutra.

---

## Design Process

This moonshot uses the **Chrome workflow aggressively** — deep theoretical reasoning with empirical probes as a first-class reasoning tool. Theory and experiments feed each other in alternating cycles. We don't just think then build — we think, probe, learn, revise, probe again. Every theoretical claim gets tested against reality early and often.

Inspired by Chrome from Dr. Stone: the self-taught scientist who builds understanding through disciplined experimentation grounded in theory. No pure armchair derivation. No blind trial-and-error. Theory proposes, experiment disposes, theory refines.

### Exploration = Building
There are no separate "research" and "implementation" phases. Research drives data, data drives research. They are coupled. Mix long shots with close-to-home approaches. Codex decides when the design is sound.

### Pivot Criteria
The data tells us. Codex reviews artifacts and decides whether to continue or pivot. No approach is sacred — if the probes say it's dead, it's dead.

### Autonomy — ELITE SCIENTIST MODE
Fully autonomous operation. User is monitoring and will provide feedback directly.

**NEVER STOP WORKING. NOT FOR ONE SECOND. ZERO IDLE TIME.**
Every message must contain productive work — experiments, code, theory, research.
If you catch yourself writing status updates without doing work, START DOING WORK.
When experiments run in background:
- Develop alternative theories, challenge assumptions, sketch new architectures
- Research cross-domain insights (biology, physics, math — NOT just ML)
- Discuss ideas with Codex, iterate on theory
- Prototype components, run analysis, generate data
- Update task list with next steps discovered during theory work
- Log ALL discussions, theorems, insights in research/RESEARCH.md

**Think like an elite scientist:** when the lab equipment is running, you're at the whiteboard
deriving the next experiment, reading papers, challenging your own assumptions, arguing with
colleagues (Codex). Compute time = theory time. ZERO idle time.

**Task list is LIVE:** Automatically create tasks for new ideas, new probes, new theories as
they emerge during research. The task list drives autonomous momentum — never run out of
things to do.

### THE SUTRA VISION: Stage-Superposition State Machine

The 7 processing stages are NOT a linear pipeline. They form a STATE GRAPH:
- Multiple stages ACTIVE SIMULTANEOUSLY on different positions
- Stages LOOP BACK (verify → reroute → retry)
- Each position in SUPERPOSITION of stages (like quantum states)
- Each patch progresses at its OWN RATE (like spiking neural networks)
- Stage 7 (verify) can loop to Stage 4 (reroute) — decode-verify is iterative

The 7 stages (converged through 4 Codex debate rounds):
1. Segmentation/Compression  2. State Init/Addressing
3. Local Construction  4. Communication/Routing
5. State Update/Memory Write  6. Compute Control  7. Readout/Decode/Verify

For EACH stage: derive the optimal mechanism FROM MATH, not from existing models.
DO NOT copy (adding attention = copying Jamba). DERIVE something better.

### Data Strategy
- Open datasets + any files available on this laptop
- Complete freedom — no constraints on data sources
- Data pipeline is part of the architecture design, not an afterthought

---

## File Management (HARD RULES — ZERO EXCEPTIONS)

These rules prevent the repo from exploding during autonomous work.

### Maximum File Inventory
| Purpose | File(s) | Location |
|---------|---------|----------|
| Constitution | `CLAUDE.md` | root |
| Public face | `README.md` | root |
| All research, theory, Chrome cycle results, literature | `research/RESEARCH.md` | single rolling doc |
| Experiment ledger (machine-readable) | `experiments/ledger.jsonl` | one JSONL file |
| Experiment summary (human-readable) | `experiments/EXPERIMENTS.md` | one summary |
| Source code | `code/*.py` (minimal files) | one canonical runner + modules |
| Results | `results/*.json` | structured JSON only |
| Figures | `results/figures/*.png` | publication-quality only |
| Custom eval suite | `eval/` | eval queries + scoring |

### Anti-Entropy Rules
- **Never create a new file when you can edit an existing one.**
- **Every commit must leave the repo cleaner or equal — never dirtier.**
- **Delete outdated content aggressively** — old probe results, superseded theory sections, dead code.
- **No duplicate information across files** — one canonical location per fact.
- **Consolidation pass after every major Chrome cycle** — merge, prune, compress.
- **No `v2.py`, `new.py`, `temp.py`, `old_*.py`** — if it's superseded, delete it.
- **No script sprawl** — one canonical execution path, config-driven variation.
- **Results are structured JSON** — not loose text files, not notebooks, not CSVs.
- **Figures are pruned** — only keep what's publication-quality or actively needed for Codex review.

### What Gets Deleted
- Any file not referenced by CLAUDE.md, RESEARCH.md, EXPERIMENTS.md, or active code
- Superseded theory sections (replaced in RESEARCH.md, not archived)
- Probe results older than the current design iteration (summarized in ledger, raw deleted)
- Dead code paths, commented-out blocks, TODO placeholders

---

## Artifact Strategy

### What We Store and Why
| Artifact | Format | Retention | Purpose |
|----------|--------|-----------|---------|
| Chrome cycle findings | Section in `research/RESEARCH.md` | Consolidated after each cycle | Theory evolution |
| Probe results (success) | `results/*.json` | Keep if validates current theory | Positive evidence |
| Probe results (failure) | `results/*.json` | Keep if invalidates a hypothesis | Negative evidence (equally valuable) |
| Failed approaches | Section in `research/RESEARCH.md` | Permanent — never delete | Prevents revisiting dead ends, informs pivots |
| Experiment runs | Entry in `experiments/ledger.jsonl` | Permanent (append-only) | Reproducibility — includes failures with status="FAIL" |
| Code | `code/*.py` | Keep only current working version | Implementation |
| Codex review output | Summarized into `research/RESEARCH.md` | Output files deleted after ingestion | Design decisions |
| Figures | `results/figures/*.png` | Only publication-quality | Communication |

### Failure Tracking (MANDATORY)
Negative results are first-class data in the Chrome workflow. Every failed probe must be:
1. **Logged in `experiments/ledger.jsonl`** with `status: "FAIL"` and `notes` explaining why it failed
2. **Summarized in `research/RESEARCH.md`** under a "Dead Ends" section with: what was tried, what happened, why it failed, what we learned
3. **Referenced in Sutra memory** (`Failed Approaches` section) if it's a major directional dead end
4. **Never deleted** — failed approaches prevent future sessions from repeating mistakes
5. **Reviewed by Codex** — failures often reveal more than successes about the underlying structure

### Codex Review Artifacts
Every Codex review produces an output file. After Claude reads it:
1. Key decisions/findings get added to `research/RESEARCH.md`
2. Action items get executed
3. **The output file is deleted** — RESEARCH.md is the single source of truth

---

## Codex Prompt Template (MANDATORY)

**Every single Codex invocation MUST begin with this preamble. No exceptions.**

```
MANDATORY FIRST STEP: Read CLAUDE.md in this repository root. Every rule there is binding.

CONTEXT YOU MUST KNOW:
- We are in a CHROME WORKFLOW: theory + experiments alternate. Currently in THEORY-HEAVY
  mode because a 475M-param production model is training on GPU in the background.
  We cannot do heavy compute experiments right now — only theoretical analysis and
  lightweight CPU prototyping.
- The architecture vision is a STAGE-SUPERPOSITION STATE MACHINE with 7 stages that
  form a state graph (not a pipeline). See research/STAGE_ANALYSIS.md for full details.
- DO NOT recommend copying existing architectures (e.g., "just add attention layers").
  Every mechanism must be DERIVED from first principles (math, info theory, biology).
- CRITICAL: This is an INTEGRATED SYSTEM. All stages must COMPLEMENT each other,
  not compete. Each stage's output must be exactly what the next stage needs.
  Components must lift each other exponentially, not drag each other down with
  conflicting biases. Evaluate proposals for SYSTEM COHERENCE, not just per-stage quality.
- File management: NEVER create new files unless necessary. Edit existing. Delete outdated.
- All findings → research/RESEARCH.md. Raw ideas → research/SCRATCHPAD.md.

After reading CLAUDE.md, explore the repo structure to understand current state.

ACTUAL TASK:
[task goes here]
```

This template is non-negotiable. If a Codex prompt doesn't start with this, it's a bug.

---

## Resources Available

### From AI Moonshots Parent Project
- **CTI Universal Law**: `../moonshot-cti-universal-law/` — universal law of representation quality, could directly inform architecture
- **Fractal Embeddings**: `../moonshot-fractal-embeddings/` — multi-scale geometric structure in embeddings
- **Self-Constructing Intelligence**: findings on evolutionary emergence from random init
- **Latent Space Reasoning**: random tokens unlocking hidden capabilities
- **Model Directory**: `../models/MODEL_DIRECTORY.md` — all available models for comparison

### Hardware
- NVIDIA RTX 5090 (24GB VRAM)
- Full laptop compute available

### Data
- Open datasets (HuggingFace, academic, etc.)
- Any files on this machine
- No restrictions
