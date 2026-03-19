Grounded in [SCRATCHPAD](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/SCRATCHPAD.md#L137>), the existing `k=4` MQAR signal in [RESEARCH](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/RESEARCH.md#L698>), and the current `v0.3` thesis in [RESEARCH](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/RESEARCH.md#L817>).

**Experiment A**
- Clean for fixed `k`; not clean once `adaptive-k` is mixed in. The router adds params, Gumbel temperature choices, and optimization noise. Run fixed `k` first, adaptive second.
- Equal epochs is a confound. Compare at equal tokens and also equal training FLOPs or wall-clock; `k=N` may win just by spending more compute.
- `2M` bytes is enough for coarse effects (`k=0` vs `4` vs `N`), but probably underpowered for subtle `4/8/16/32` differences on BPB. Use `2M` as a pilot; rerun the top 2-3 `k` values on `10M+` if gaps are small.
- Yes, add task types: MQAR difficulty sweep, passkey/needle retrieval, and selective copy or state tracking. Missing controls: 3 seeds, fixed context length, random-retrieval control, and retrieved-distance histograms.

**Experiment B**
- The 5 variants are close, but one baseline is missing: periodic tiny global refresh over chunk summaries every `N` rounds. That tests whether content-addressable sparsity is actually necessary.
- Match fairness two ways: strict matched params by shrinking width/depth when scratchpad adds weights, and matched FLOPs/latency because scratchpad mainly changes bandwidth/compute. Report both.
- MQAR is the right first long-range test because it isolates true retrieval, but it is not enough. Add passkey/needle with variable offsets and one compositional long-range task: selective copy, bracket matching, or state tracking.
- Publishable means more than “one variant worked”: 3 seeds, matched baselines, at least 2 scales or context lengths, real-text BPB plus synthetic retrieval, and a crisp efficiency claim like “tiny sparse/global mechanism recovers X% of dense attention at Y% compute.”

Run `A` before `B`. `B` hardcodes `k=8`, which is arbitrary until `A` identifies the right sparse regime. Best sequence: `A` as a pilot on `2M`, rerun top `k` values on larger data if close, then `B` with the chosen `k`.