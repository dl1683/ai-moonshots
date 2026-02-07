# AI Moonshots

**Research experiments that question the foundations of AI — not incremental improvements, but paradigm shifts.**

---

## The Premise

Current AI research optimizes within established paradigms. We're questioning the paradigms themselves:

- What if flat embedding spaces are fundamentally wrong for hierarchical data?
- What if intelligence is about *structure*, not learned weights?
- What if we can *derive* optimal architectures from first principles?
- What if reasoning chains are performance, not computation?

Each moonshot targets a foundational assumption. When an assumption breaks, the results reshape how we think about the problem.

---

## Moonshots

### [Fractal Embeddings](moonshot-fractal-embeddings/) — *When Structure Matters*

**Status: Validated**

Traditional embeddings force coarse and fine-grained semantics to compete for the same dimensions. Fractal embeddings introduce multi-scale structure where different scales specialize for different levels of a hierarchy.

**Key results:**
- **+5.36% coarse, +6.47% fine-grained** accuracy on Yahoo Answers (5-seed validation)
- **Correct hierarchy helps, wrong hierarchy hurts** — 95% CI excludes zero (causal proof)
- **Consistent improvement at depths 2-5**, with lower variance than flat baselines
- Statistically significant on 20 Newsgroups real-world benchmark (p=0.023)

Includes full theoretical foundation: minimax lower bounds via Assouad's lemma, sample complexity separation proofs, and information-theoretic analysis of scale-separated embeddings.

**[Read more →](moonshot-fractal-embeddings/)**

---

### More moonshots coming soon.

---

## Hardware

All experiments run on a single NVIDIA RTX 5090 (24GB VRAM). No cluster required.

## License

Each moonshot directory contains its own license. See individual moonshot READMEs for details.
