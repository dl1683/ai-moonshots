The pattern in `research/RESEARCH.md` is clear: at `dim=128 / 300 steps`, wins come from making the right behavior easier to discover, not from adding a smarter controller. So I’d stay close to the current `v0.5.3` shape in [scratchpad.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/scratchpad.py) and [sutra_v05_ssm.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py), and try these three:

1. **Multi-timescale scratchpad, but with fixed decays**
Change the scratchpad from one EMA timescale to a small bank of fixed half-lives, e.g. 2 fast slots, 4 medium slots, 2 slow slots with hard-coded decays like `0.5 / 0.8 / 0.95`. No learned time constants, no halting, same read/write mechanism.
This adds a strong discourse prior: some memory should track recent clause state, some should track topic/entity identity across the sequence. It directly targets **coherence** and entity persistence.
Why it should work small-scale: scratchpad already wins; this is just a stricter bias on what memory is for, without adding a new optimization problem.

2. **Deterministic dyadic routing taps**
Augment or partially replace learned global top-k retrieval with fixed causal skips at lags `{1, 2, 4, 8, 16, ...}`. Each token always gets messages from those exponentially spaced past positions, then mixes them with the existing local window.
This gives the model a built-in multiscale path, so long dependencies become reachable immediately instead of requiring it to learn content-based retrieval geometry in 300 steps. It targets **reasoning** and medium-range coherence.
Why it should work small-scale: it is a routing skeleton, not a new learned control system. Very similar to why scratchpad helps: useful communication channels exist from step 1.

3. **Pointer-copy output head**
Keep the tied softmax, but add a tiny copy branch that can place probability mass on tokens seen in the causal context, using existing router attention or a lightweight extra copy attention. Final output is a gate between vocab logits and copy distribution.
This adds a direct inductive bias for exact surface-form reuse: names, numbers, variables, delimiters, repeated entities. It targets **factual consistency** in-context, which is the kind of “factual” small models can improve fastest.
Why it should work small-scale: copying is much easier than compressing exact token identity into hidden state and reconstructing it later. No auxiliary loss, just cleaner CE gradients.

If I had to rank them for `128/300`, I’d do: `1) multi-timescale scratchpad`, `2) pointer-copy head`, `3) dyadic routing taps`.

The common theme is: fixed structure over learned control. That matches every success/failure you logged.