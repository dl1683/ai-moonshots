# Pre-Registration: Extended Family LOAO (Decoder + Encoder Separation)

**Registered:** 2026-02-23 (after session 58 2D causal surface at commit 7158256)

## Motivation

The paper's current 12-architecture LOAO uses decoder-only LMs and hybrids (alpha_mean=1.477,
CV=2.3%). The comprehensive 19-architecture single-intercept test shows CV=1.75% but mixes
decoder/encoder/SSM families. A natural open question: do encoder-only architectures (BERT,
DeBERTa, ELECTRA) follow the same CTI law with a DIFFERENT slope (as the paper reports
encoder alpha~7.7), and is that encoder slope ITSELF universal within the encoder family?

This pre-registration tests:
1. Whether the decoder-family 12-arch LOAO is robust when extended to include GPT-2 and Phi-2
   (two classic decoder architectures from different lineages: OpenAI 2019, Microsoft 2023)
2. Whether the encoder-family slope (deberta-base, electra-small, bert-base-uncased) is tight
   within the encoder family, yielding a separate universal encoder constant

## Pre-Registered Hypotheses

**H1 (GPT-2 decoder LOAO stability):** The LOAO alpha for gpt2 (fitting on 12 archs,
predicting gpt2) falls within [1.35, 1.65] (5-sigma envelope from 1.477 ± 0.034).
NOTE: GPT-2's last layer may show kappa-q degradation (known causal LM behavior); the
LOAO will be evaluated using all 4 proportional-depth layers. If last-layer inclusion
causes negative alpha, this will be documented as a "last-layer causal LM artifact."

**H2 (Phi-2 decoder LOAO stability):** The LOAO alpha for phi2 (fitting on 12 archs,
predicting phi2) falls within [1.35, 1.65].

**H3 (Encoder-family within-family CV):** For 3 encoder architectures (deberta-base,
electra-small, bert-base-uncased), the per-dataset LOAO alpha CV is < 0.50 (encoders
are expected to have different alpha from decoders, but tight within the encoder family).

**H4 (Encoder vs decoder slope separation):** The encoder family alpha_mean is
significantly different from decoder alpha=1.477 (at least 2x larger).

## Method

1. Load kappa_near_cache files for all architectures (datasets: 20newsgroups, agnews,
   dbpedia, go_emotions -- the same 4 datasets used in the 12-arch analysis)
2. Run per-dataset-intercept LOAO:
   - For gpt2 and phi2: fit on 12 original archs, predict each one
   - For encoder group: run within-encoder LOAO (3 archs)
3. Compute alpha and CV for each group

## Prediction Framework

Expected outcomes:
- gpt2: alpha may be negative or unstable (last-layer degeneration in causal LM)
- phi2: alpha ≈ 1.2-1.5 (decoder fine-tuned for instructions, may deviate slightly)
- deberta-base: alpha ≈ 5-10 (encoder, much higher per-class separation)
- electra-small: alpha ≈ 5-10 (encoder discriminative model)
- bert-base-uncased: alpha ≈ 2-7 (classic bidirectional encoder)
- encoder CV: depends on architectural similarity within family

## Results File

`results/cti_extended_family_loao.json`

## Script

`src/cti_extended_family_loao.py` -- written AFTER this pre-registration is committed.
