v0.5 SSM is not competitive with the field yet. The hard numbers say “promising early learner, still materially behind mature small LMs.” I used the logged v0.5 run at [results/v05_log.txt](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/v05_log.txt), the recorded eval in [research/RESEARCH.md#L1599](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1599), and the Pythia baseline in [results/clean_benchmark_pythia.json](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/clean_benchmark_pythia.json). For the budget forecast, I fit the observed post-warmup loss curve from steps 1000-5000: `train_loss_nats ≈ 11.56 - 0.784 ln(step)`. At step 5000 that agrees closely with the eval point.

1. Normalized for tokens seen: yes, the early learning efficiency is strong; no, the current quality is not competitive.
   - v0.5: `7.0663` BPT at `163.84M` seen tokens.
   - Pythia-70m: `3.5592` BPT at `300B` seen tokens.
   - Absolute gap today: `+3.5071` BPT. v0.5 is at `1.99x` Pythia’s BPT.
   - If you normalize by average BPT reduction from a uniform-vocab baseline (`log2(50257) ≈ 15.62`), v0.5 has improved by `8.55` BPT in `0.16384B` tokens, or about `52.2 BPT / Btoken`.
   - Pythia improved by `12.06` BPT in `300B` tokens, or about `0.0402 BPT / Btoken`.
   - On that crude token-efficiency metric, v0.5 is about `1300x` more data-efficient in the early regime.
   - Bottom line: the slope is good; the absolute position is still bad.

2. Training budget to match Pythia-70m, based on the curve:
   - Target is `3.5592` BPT, which is `2.467` nats/token.
   - Solving the fitted curve gives about `109k` steps.
   - At `32768` tokens/step, that is about `3.57B` seen tokens.
   - Relative to the `1.697B`-token corpus, that is about `2.10` epochs.

3. Gap between “best possible at 1.7B tokens” and “competitive with Pythia”:
   - One corpus pass is about `51,779` steps.
   - The fitted curve projects v0.5 to about `4.40` BPT at `1.7B` seen tokens.
   - Gap to Pythia at that point: `4.40 - 3.559 ≈ 0.84` BPT.
   - That is about `23.7%` higher cross-entropy than Pythia.
   - In budget terms, one epoch is not enough; you need about `1.87B` more seen tokens beyond the 1.7B-token corpus pass.

4. Fundamentally limited, or just undertrained?
   - Mostly undertrained.
   - The architecture is not obviously dead: on controlled probes it beats matched transformers by `11.8%` to `32.6%` mean BPB in [results/matched_param_scaling.json](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/matched_param_scaling.json), and by `71%` to `74%` on the two-scale synthetic source in [research/RESEARCH.md#L1425](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1425).
   - But on natural-language LM, the current implementation still projects to miss Pythia after a full 1.7B-token pass.
   - So the right read is: not fundamentally broken, but undertrained and still carrying real architecture liabilities on natural text.

5. What changes the competitive picture most?
   - For reaching Pythia-70m specifically: more data. The current 67M model projects to parity at about `3.57B` seen tokens without any architecture change.
   - For moving beyond that into SmolLM2-class territory: bigger model plus more data. `67M` vs `360M` params and `3.6B` vs `4T` tokens is still a massive deficit.
   - Architecture improvement is third, based on current measured gains. The cleanest measured upgrade in-repo is the switching kernel at `+4.1%` BPT in [research/RESEARCH.md#L1575](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1575). That is useful, but far smaller than the `49.6%` BPT reduction needed to go from `7.07` to `3.56`.

6. Three specific weaknesses v0.5 has that Pythia doesn’t:
   - It lacks exact token-level global causal attention. v0.5 routes through local/restricted message passing and compressed summaries, so long-range token dependencies are a harder problem than in Pythia’s full causal attention stack. See [code/sutra_v05_ssm.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py).
   - Its claimed sparse compute is not real yet. The reaudit shows stage MLPs still run on the full tensor whenever any token selects that stage, so the advertised Top2 sparsity does not produce honest compute savings. See [results/codex_v05_reaudit.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v05_reaudit.md) and [code/sutra_v05_ssm.py#L105](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L105).
   - Its control loop is incomplete/unstable. The current forward path explicitly says “No verify/reroute yet,” and the reaudit flags the verify/emission path as not training-safe. Pythia has no comparable self-routing subsystem to destabilize optimization. See [code/sutra_v05_ssm.py#L270](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L270) and [results/codex_v05_reaudit.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v05_reaudit.md).

Net: v0.5 looks like a fast early learner, not a competitive small LM yet. If the current curve holds, it can plausibly enter Pythia-70m territory at roughly `3.6B` seen tokens, but it is nowhere near SmolLM2-class competitiveness today.