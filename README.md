# AI Moonshots

**Research experiments that question the foundations of AI — not incremental improvements, but paradigm shifts.**

> *"AI should be like electricity. It should be like vaccines. It should be cheap, ubiquitous, and useful to the poorest person on the street, not just the richest corporation in the cloud."*

---

## The Premise

The current AI paradigm assumes **Intelligence = Scale** — more parameters, more data, more compute, more money. We believe **Intelligence = Geometry** — better mathematical structure, better information allocation, better representations.

You don't need a data center to be intelligent. You just need better geometry.

Every moonshot here questions a foundational assumption. When an assumption breaks, the results reshape how we think about the problem. All experiments run on a **single NVIDIA RTX 5090 laptop** (24GB VRAM). No cluster required.

---

## Moonshots

### Flagship

| Moonshot | What It Questions | Status | Repo |
|----------|------------------|--------|------|
| **CTI Universal Law** | *Can we derive a universal law governing learned representations from first principles?* | COLM 2026 submission ready | [moonshot-cti-universal-law](https://github.com/dl1683/moonshot-cti-universal-law) |
| **Latent Space Reasoning** | *Can random untrained tokens unlock hidden reasoning capabilities in small models?* | NeurIPS paper ready | [Latent-Space-Reasoning](https://github.com/dl1683/Latent-Space-Reasoning) |
| **Fractal Embeddings** | *Are flat embedding spaces fundamentally wrong for hierarchical data?* | Validated (5-seed, causal) | [moonshot-fractal-embeddings](https://github.com/dl1683/moonshot-fractal-embeddings) |
| **Sutra (सूत्र)** | *Can a byte-level model built from first principles beat scaled-down architectures?* | Stage 0 complete, Stage 1 active | [moonshot-sutra](https://github.com/dl1683/moonshot-sutra) |

### [CTI Universal Law](https://github.com/dl1683/moonshot-cti-universal-law) — *The Compute Thermodynamics of Intelligence*

A first-principles derivation (via Extreme Value Theory and Gumbel statistics) of a universal law governing how representation quality determines task performance — validated across 19 architectures, 10 datasets, 4 modality families, and biological neural systems. 28-page paper targeting COLM 2026.

**The law:** `logit(q) = alpha * kappa_nearest + C` where q is normalized accuracy and kappa is nearest-class signal-to-noise ratio.

**Key results:**
- **Universal functional form** holds across transformers, SSMs, CNNs, hybrids, and mouse visual cortex
- **Cross-architecture**: LOAO across 12 NLP architectures — alpha=1.477, CV=2.3%, R²=0.955
- **Holdout validation**: 11 holdout models × 8 datasets — r=0.879, MAE=0.077
- **Leave-one-family-out**: all 4 architecture families pass (r >= 0.84)
- **Biological generalization**: 30/32 mouse V1 Neuropixels sessions pass; multi-area (5 cortical areas) all ≥87%
- **Equicorrelation**: rho ~ 0.45 is constant across all families AND biological V1 (CV=1.65%)
- **Causal**: confusion-matrix prediction r=0.842, frozen do-interventions, orthogonal factorial design
- **Cross-modal**: ViT-Large R²=0.964, ResNet-50, alpha varies by family (decoders 1.48, ViT 0.63, CNN 4.4)
- **Practical**: cross-model architecture ranking rho=0.833 (p=0.005)

### [Latent Space Reasoning](https://github.com/dl1683/Latent-Space-Reasoning) — *Hidden Capabilities, Unlocked by Noise*

Prepending random embedding-scale tokens to a small language model's input dramatically improves reasoning — with zero training, zero fine-tuning, and zero parameter changes. This is a new axis of improvement orthogonal to scaling, fine-tuning, prompting, RAG, and sampling.

**The finding:** Small models aren't failing because they lack capability — they're stuck in suboptimal output policies. Two random tokens are enough to break the default.

**Key results:**
- **Qwen3-4B arithmetic**: 32% → 51.6% (+19.6pp) with just 2 random tokens
- **Direction-agnostic**: random noise matches optimized projections (p=1.0) — the direction doesn't matter
- **Non-monotonic dose-response**: 2 tokens optimal; more tokens degrade (sweet spot, not "more is better")
- **100% oracle coverage**: 10 random directions collectively solve every task
- **Attention sink rescue**: model collapse (14 words) → complete diagnostic plan (650+ words)
- **Cross-model validated**: positive on 3/4 models (Qwen3-4B, Qwen3-8B, phi-2); mechanism splits into convergence aid vs computation aid depending on model capacity
- **Chain-of-thought mediates**: disable thinking, effect vanishes entirely — perturbation works *through* reasoning

### [Fractal Embeddings](https://github.com/dl1683/moonshot-fractal-embeddings) — *When Structure Matters*

Multi-scale embeddings where hierarchy-aligned prefix supervision forces different embedding scales to specialize for different hierarchy levels. Proves that correct geometric structure *causally* improves representations — wrong structure hurts. Includes V5 architecture with block dropout and head-only training.

**Key results:**
- **+5.36% coarse, +6.47% fine-grained** accuracy on Yahoo Answers (5-seed validation)
- **+0.70%** on 20 Newsgroups (p=0.0232, statistically significant)
- **Correct hierarchy helps, wrong hierarchy hurts** — 95% CI excludes zero (causal proof)
- **Scaling validated**: consistent 1-2.5% gains across hierarchy depths 2-5
- **Theoretical foundation**: minimax generalization bounds, sample complexity separations, access complexity advantage Ω(k/(j+1)) vs isotropic embeddings

---

### [Sutra (सूत्र)](https://github.com/dl1683/moonshot-sutra) — *Compression Is Intelligence*

A from-scratch byte-level language model that operates directly on raw bytes — no tokenizer, no vocabulary, no inherited assumptions. Derived from first principles: if compression is intelligence, then better compression (better mathematics) beats brute-force scale. Features the **Ekalavya Protocol** — multi-teacher cross-architecture knowledge distillation at the byte level, where any model (LLM, encoder, vision, STEM) becomes a teacher regardless of its tokenizer or architecture.

**Key results:**
- **Sutra-Dyad-153M** learns English from raw bytes on a single laptop GPU — BPB 6.5 → 2.19 in 3K steps
- **Byte-level operation** eliminates tokenizer mismatch — enables cross-architecture KD that token models can't do
- **Dual-scale architecture**: 12-layer global transformer + local byte decoder, designed for variable-rate processing
- **Stage 1 in development**: adaptive patching, cross-attention, byte-residual bypass (194M params)
- **80GB** pre-processed byte training data, streaming pipeline

### In Development

| Moonshot | What It Questions | Description |
|----------|------------------|-------------|
| **Fractal Mind** | *Can shared-weight blocks with learned halting achieve unbounded reasoning depth?* | Reached 10x iteration efficiency with no accuracy loss on Copy/Reverse/Sort/LM tasks |
| **Self-Constructing Intelligence** | *Can intelligence build itself from minimal seeds without training?* | Evolutionary emergence from random initialization — XOR, AND, OR solved through pure evolution |
| **Tokenization Limits** | *What are the information-theoretic limits of tokenization?* | Empirical Fano bounds and collision mining across 7 model families |
| **Complex Fractal Adapters** | *Can complex-valued adapters with fractal weight sharing compress models?* | Structured factorization + evolutionary topology search for Qwen3-0.6B |
| **Open Intelligence Infrastructure** | *What would a transparent, transaction-first intelligence substrate look like?* | Causal hypergraph world models with budget-aware inference and audit trails |
| **Deterministic Knowledge Structure** | *Can we build an AI-native fact database with semantic equivalence and deterministic querying?* | Version history, provenance tracking, multi-agent sync |

---

## The Vision

These moonshots are connected. They build toward five paradigm-shifting questions:

1. **Can we derive intelligence from first principles?** (CTI, Fractal Embeddings, Sutra)
2. **Can intelligence construct itself?** (Self-Construction, Fractal Mind)
3. **Are small models more capable than we think?** (Latent Space Reasoning, Sutra)
4. **Can compression replace scale?** (Sutra, Tokenization Limits)
5. **What are the fundamental limits of neural networks?** (CTI, Fractal Mind)
6. **Can we make intelligence a public utility?** (Open Intelligence Infrastructure, Sutra)

See the [model directory](models/MODEL_DIRECTORY.md) for the full set of models used across experiments.

---

## Hardware

All experiments run on a single NVIDIA RTX 5090 laptop (24GB VRAM). If the theory is right, you don't need a supercomputer.

## License

Each moonshot repository contains its own license.
