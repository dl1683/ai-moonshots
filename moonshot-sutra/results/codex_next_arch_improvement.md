The single next change worth testing is a **tiny causal global scratchpad**: `8-16` shared memory slots that every token can read from, and that Stage 5 can write to on each recurrent step.

Why this one:
The current model in [sutra_v05_ssm.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py) has strong local syntax and token-to-token routing, but it has **no explicit sequence-level state**. Stage 4 routes between token states, and Stage 5 writes back into token states; nothing stores a slowly changing discourse summary. That is exactly the setup that gives you grammatical text with weak coherence: the model can predict locally, but it keeps having to re-infer topic, entities, and “what is going on” from scattered token states.

Coherence specifically needs a **medium-timescale memory**. Your own notes already point there in [research/RESEARCH.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md) and [research/SCRATCHPAD.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/SCRATCHPAD.md): fast state + slow weights is not enough; discourse needs a shared working memory. More halting will not create that object. Better routing alone just moves information around with nowhere stable to condense it.

Concrete test:
- Add `S=8` scratchpad slots `M ∈ R^{S×D}`.
- Each recurrent step:
  - tokens read from `M` with a cheap top-k or low-rank read
  - Stage 5 writes back to `M` with a gated EMA-style update
  - inject the readout into the existing router/write path
- Keep everything else the same: switching kernel, recurrence, router, BayesianWrite.

CPU prototype:
- `dim=128`, `seq_len=128`, `max_steps=4`
- A/B: baseline v0.5.2 vs `+scratchpad`
- Evaluate on MiniPile/TinyStories shard plus one cheap entity-tracking task
- Success signal: better BPT and visibly better name/pronoun/topic consistency in greedy generation

If I had to place one 67M bet for **coherence**, it’s this: **make Stage 5 write to a tiny shared discourse memory, not only back into per-token state**.

One caveat: this should improve coherence more than factual knowledge. At `67M`, factual knowledge is still mostly scale/data-limited. But a scratchpad is the cleanest architectural fix for “it speaks English but loses the plot.”