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
| **CTI Universal Law** | *Can we derive a universal law governing learned representations from first principles?* | Paper submitted | [moonshot-cti-universal-law](https://github.com/dl1683/moonshot-cti-universal-law) |
| **Fractal Embeddings** | *Are flat embedding spaces fundamentally wrong for hierarchical data?* | Validated | [moonshot-fractal-embeddings](https://github.com/dl1683/moonshot-fractal-embeddings) |

### CTI Universal Law — *The Compute Thermodynamics of Intelligence*

A first-principles derivation (via Extreme Value Theory and Gumbel statistics) of a universal law governing how representation quality determines task performance across architectures, modalities, and even biological neural systems. Pre-registered, reproducible, validated on 19+ architectures and 10 datasets.

**Key results:**
- Universal law `logit(q) = alpha * kappa + C` holds across transformers, SSMs, CNNs, and hybrids
- Equicorrelation `rho ~ 0.45` is constant across all architecture families AND mouse visual cortex (V1)
- First-principles EVT/Gumbel derivation — not curve fitting, but a derived physical law
- Leave-one-model-family-out validation: all 4 families pass `r >= 0.84`

### Fractal Embeddings — *When Structure Matters*

Multi-scale embeddings where different prefix lengths specialize for different hierarchy levels. Proves that correct geometric structure causally improves representations — wrong structure hurts.

**Key results:**
- **+5.36% coarse, +6.47% fine-grained** accuracy on Yahoo Answers (5-seed validation)
- **Correct hierarchy helps, wrong hierarchy hurts** — 95% CI excludes zero (causal proof)
- Full theoretical foundation: minimax bounds, sample complexity separations

---

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

1. **Can we derive intelligence from first principles?** (CTI, Fractal Embeddings)
2. **Can intelligence construct itself?** (Self-Construction, Fractal Mind)
3. **Can we compute meaning exactly?** (Tokenization Limits, Knowledge Structure)
4. **What are the fundamental limits of neural networks?** (CTI, Fractal Mind)
5. **Can we make intelligence a public utility?** (Open Intelligence Infrastructure)

See the [model directory](models/MODEL_DIRECTORY.md) for the full set of models used across experiments.

---

## Hardware

All experiments run on a single NVIDIA RTX 5090 laptop (24GB VRAM). If the theory is right, you don't need a supercomputer.

## License

Each moonshot repository contains its own license.
