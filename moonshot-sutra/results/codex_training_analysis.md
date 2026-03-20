Using [CLAUDE.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/CLAUDE.md), [v05_log.txt](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/v05_log.txt), and [v05_metrics.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/v05_metrics.json), I’d fit the post-warmup curve only, because steps 100-1000 are dominated by LR warmup.

1. Post-warmup training loss is well fit by:
   `L_train(s) ~= 3.98 + 2.20 * (s / 1000)^(-0.52)` for `s >= 1000`.
   This implies an asymptotic floor near `3.98` nats. It is an inference from the logged points, not a reported metric.
   Quantitatively:
   `L(10k) ~= 4.64`, `L(50k) ~= 4.27`, `L(100k) ~= 4.18`.
   Plateau depends on definition:
   visible flattening: `~8k-10k` steps,
   gain < `0.02` nats per 1k steps: `~14k` steps,
   gain < `0.01` nats per 1k steps: `~23k` steps.
   True asymptote is not reached by `100k`; even there the model is still `~0.20` nats above the fitted floor.

2. Calibrating test BPT from step 5000:
   train bits/token at 5000 = `4.9309 / ln 2 ~= 7.11`,
   measured test BPT = `7.0663`,
   gap = `-0.048` bits/token.
   Holding that gap fixed gives:
   `10k: ~6.65 BPT`
   `50k: ~6.10 BPT`
   `100k: ~5.98 BPT`
   Reasonable uncertainty bands are about `+-0.10`, `+-0.15`, `+-0.20` BPT respectively.

3. “Good enough” needs two thresholds:
   for non-gibberish, mostly coherent text: around `15k-25k` steps, corresponding to test BPT roughly `6.5 -> 6.3`;
   for actual reasoning-task usefulness: probably never in this 67M regime.
   Quantitative anchor: official SmolLM2-135M, after `2T` pretraining tokens, reports only `1.4` GSM8K (5-shot) and `31.5` MMLU-cloze. A 67M model is unlikely to become genuinely reasoning-capable just by extending this same run.

4. Learning-rate comparison:
   your run: peak LR `3e-4`, 1% warmup to step `1000`, then effectively flat through `5200`.
   Pythia nearest sizes:
   `70M -> 1e-3`
   `160M -> 6e-4`
   with 1% warmup and cosine decay.
   So v0.5 is running at `0.30x` the Pythia-70M LR and `0.50x` the Pythia-160M LR despite being closer to 70M scale.
   SmolLM2/Smol-family uses WSD-like schedules with a long stable phase; that matches your schedule shape better than Pythia’s full cosine. But Hugging Face’s published smol training guidance explicitly shows `1e-4` under-converging and `2e-4`/`3e-4` converging much faster on larger ablations, so your curve looks conservative, not aggressive.

5. Anomalies:
   no instability, no divergence, no spike behavior;
   post-warmup curve is an unusually clean power law;
   largest local reversals are only about `0.02` nats, which is normal minibatch noise;
   the only oddity is test BPT being lower than train bits by `0.048` bits (`~0.7%`). That is small enough to attribute to an easier eval slice or train-loss noise, not leakage by itself.
   The real anomaly relative to published small-model recipes is under-aggressive LR.

6. Single hyperparameter change:
   raise peak LR from `3e-4` to `6e-4`.
   That is the highest-probability improvement.
   It moves you in line with Pythia-160M and still below Pythia-70M. Given the absence of spikes or instability, the current run looks LR-limited rather than capacity-limited at this stage.

Sources:
[Pythia repo](https://github.com/EleutherAI/pythia)
[Pythia model card](https://huggingface.co/EleutherAI/pythia-160m)
[SmolLM pretraining docs](https://github.com/huggingface/smollm/blob/main/text/pretraining/README.md)
[SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
[Smol training playbook excerpt](https://gist.github.com/unclecode/e5da5fb6a1d37022b089e243e0d9e00e)