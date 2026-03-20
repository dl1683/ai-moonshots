Stop `v0.5` and switch to `v0.5.2`, but do it at a clean checkpoint, not by killing unsaved progress.

If you can force-save now, switch now. If not, let the current run reach the next `15K` checkpoint and warm-start from that. The trainer only saves every `5K` steps in [sutra_v05_train.py:41](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_train.py#L41), [sutra_v05_train.py:260](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_train.py#L260), [sutra_v05_train.py:267](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_train.py#L267). I would not continue the current `v0.5` all the way to `100K`.

Reason: if this were only `6e-4 -> 8e-4`, I’d say keep running. The repo already contains that counterargument in [review_step5000_chrome.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/review_step5000_chrome.md). But `v0.5.2` is not just LR. It bundles the switching kernel, the root-cause stability fix, and the higher LR; the repo’s synthesis projects about `5.63` BPT at `100K` vs `5.98` for current `v0.5`, roughly `0.35` BPT better, and explicitly recommends the new config in [codex_v052_synthesis.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v052_synthesis.md). With `94%+` transfer, the first `12K` steps are not really sunk cost. The remaining `88K` steps are the expensive part, so that is where you want the better dynamics.

Your current generation quality is expected. At `10K`, the repo itself logs “scientific language, better coherence” but still weak generation in [RESEARCH.md#L1795](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1795). The training analysis says “non-gibberish, mostly coherent text” should emerge around `15K-25K`, and “actual reasoning-task usefulness” is probably not happening in this `67M` regime in [codex_training_analysis.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_training_analysis.md). So yes: grammar without facts, reasoning, or story ability at `~12K` is normal.

Concrete call:
1. Do not keep current `v0.5` as the main `100K` run.
2. Save at `15K` or force-save now.
3. Warm-start `v0.5.2` from the latest checkpoint.
4. Reset the run budget and treat `v0.5.2` as the production trajectory.

“Actually useful” depends on the bar:
- For cleaner, locally coherent text: likely `15K-25K`.
- For factual recall, sustained reasoning, or real story generation: probably not from this `67M` run alone, even by `100K`.