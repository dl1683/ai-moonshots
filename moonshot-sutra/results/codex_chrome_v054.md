**Recommendation**

Test `P1` first for v0.5.4. If the objective is better generation, my order is `P1 > P0 > P2 > P3`.

The reason is empirical, not aesthetic: the biggest current generation-oriented win is better shared state (`scratchpad +10.2%`), while production data says deep recurrence is still valuable and the current halting signal is not calibrated, so the bottleneck looks more like state tracking/routing than compute stopping ([research/RESEARCH.md:1820](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1820), [research/RESEARCH.md:1767](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1767), [research/RESEARCH.md:1783](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1783)). Current `v0.5.3` is exactly `switching kernel + scratchpad + gain clamp + max_steps=8` ([code/launch_v053.py:51](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v053.py#L51)).

**Exact CPU Experiment**

Run this first batch at `dim=128`, `300` optimizer steps:

- Shared config: `ff=256`, `seq_len=64`, `batch=8`, `max_steps=8`, `window=2`, `k_retrieval=4`, `scratchpad_slots=8`, `lr=8e-4`, AdamW `wd=0.01`, seeds `42` and `43`.
- Data: first `512` sequences from `data/minipile_tokens.pt` for train, next `128` for held-out.
- Arm A: baseline `v0.5.3-small`.
- Arm B: `P1a` complex embeddings only. Reinterpret `128` real dims as `64` complex dims and apply `exp(i*pi*l/L)`.
- Arm C: `P1b` Arm B plus Hermitian routing in the global router only. Keep the local window branch unchanged.
- Arm D: `P0-soft` as a follow-up, not the first run if CPU is tight. Use `alpha_t = sigmoid(a(mu_t)*t + b(mu_t))` and `pi_{t+1} = (1-alpha_t)pi_t + alpha_t(pi_t @ K(mu_t))`, but keep the hard cap at `8`.

Evaluate at steps `100/200/300` with:
- Held-out BPT.
- Generation panel: `12` fixed prompts, greedy decode `64` tokens, scored `0-3` for coherence, topic retention, and entity consistency.
- Reasoning panel: `20` exact-match easy/medium items from `eval/sutra_eval_500.jsonl`, greedy decode `48` tokens.
- Mechanism sanity: stage entropy, Route/Write/Verify occupancy, reroute rate.

Promotion rule: pick the best generation composite first; BPT is only a veto. I’d use `0.5*coherence + 0.3*topic + 0.2*reasoning`, and reject any arm with BPT worse than baseline by `>3%`.

**Hypotheses / Kill Criteria**

- `P1`: Hypothesis: phase-aware representations improve long-range binding, so topic maintenance and reasoning exact-match rise without needing more steps. Kill if generation composite is not `>=10%` better than baseline by step `300`, or if stage entropy collapses `>15%`.
- `P0`: Hypothesis: CfC-style dwell times make hard tokens linger in `Route/Write/Verify` and easy tokens move cleanly, improving generation quality inside the same 8-step budget. Kill if the learned gate saturates to nearly the same schedule for all content, or if generation is flat while average effective depth falls.
- `P2`: Hypothesis: expand-then-sparsify gives more addressable states and better scratchpad/routing keys. Kill if active density misses the target band (`~5-15%`) or outputs become keywordish/repetitive with no generation gain.
- `P3`: Hypothesis: search finds a better stage mix than hand design. Kill if a small search budget cannot beat the best manual arm. Do not use this as the first v0.5.4 test.

**Direct Answers**

CfC time constants should not replace `max_steps=8` yet. They should sit under a hard cap first. Right now the model still needs deep recurrence, and your stop signal is not trustworthy enough to remove the cap.

FlyVec-style expand-then-sparsify should not directly replace `StageBank`. It is better as a Stage-2 addressing/state-code layer, or as the key space for routing/scratchpad. `StageBank` is a recurrent processor; FlyVec is a sparse code constructor.

Yes, you can steal Comply’s power, but only the representation trick. Their “1 layer beats BERT” result is telling you that sparse high-dimensional phase-aware coding is strong; it is not telling you to collapse Sutra into a one-layer LM. Use that power in embeddings, routing keys, verifier templates, and scratchpad addressing.

NCP sparsity maps well onto your stage graph, but not as literal `90%` macro-edge sparsity. With only 7 stages, `90%` would over-prune. The right mapping is: `1-2` sensory, `3` intermediate, `4-5-6` recurrent command core, `7` motor/verify with loopback to `4`. Your current graph is already sparse, but the NCP lesson is to concentrate recurrence in `4/5/6`, not to make every stage equally sparse ([code/sutra_v05_ssm.py:27](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L27)).

If you want, I can turn this into a concrete `v0.5.4` ablation matrix and patch plan against the current code surface.