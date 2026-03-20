The next 3 experiments should be about measurement, data, and scale, not new architecture. Your own log already says the architecture is real and core components matter ([RESEARCH.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1756)). The weak link is that the current reasoning eval is not yet a reliable scaling gate: the repo itself says it is only “4/10 benchmark-ready” ([RESEARCH.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L38)), the rubric scorer is still a placeholder ([score.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/eval/score.py#L126)), and the tail of the 500Q set includes ultra-hard synthesis prompts that are well above 70M-200M scale ([sutra_eval_500.jsonl](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/eval/sutra_eval_500.jsonl#L485), [sutra_eval_500.jsonl](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/eval/sutra_eval_500.jsonl#L500)).

**Next 3 Chrome experiments**
1. `Eval floor calibration`
Run `Pythia-70m`, `Pythia-160m`, `Pythia-410m`, `SmolLM2-135M`, and `SmolLM2-360M` on a stratified `easy+medium exact-match / constraint` subset of your eval, and treat the full 500Q only as a secondary qualitative check.
Hypothesis: the full 500Q is floor-saturated below ~1B, but a 50-100 question exact-match subset will separate 70M < 160M < 360M/410M.
Kill criterion: if that subset does not rank models monotonically or the spread from 70M to 360M is under 5 absolute points, the eval is too noisy/hard to use as a scaling gate.

2. `67M data bottleneck test`
Keep v0.5.2 fixed at 67M and compare matched-token runs on `current mix` vs `FineWeb-Edu` vs `FineWeb-Edu + math/code oversample`.
Hypothesis: at this size, better data will move reasoning more than another small architecture tweak, and should beat the current mix on both BPT and the calibrated reasoning subset.
Kill criterion: if the best FineWeb-Edu recipe gives under 3 absolute points on the calibrated subset and under 5% relative BPT gain, data quality alone is not enough and scaling becomes the main lever.

3. `Minimal scale ladder`
Using the winning data recipe, train `67M -> 135M/160M -> 200M` with the same tokenizer, optimizer family, and evals.
Hypothesis: the first real reasoning lift for Sutra should show up on the calibrated subset somewhere between 135M and 200M if the architecture scales; if 200M only improves perplexity and not reasoning, bigger runs are premature.
Kill criterion: if 200M beats 67M on BPT but improves the calibrated reasoning subset by less than 5 absolute points, do not jump to 400M+ on the same recipe.

**Scale expectation**
If by “reasoning emerges” you mean GSM8K-style math reasoning in a base model, 70M-160M is still basically floor. Public model cards put `Pythia-70m` at `0.3%` GSM8K (5-shot), `Pythia-160m-deduped` at `0.23%`, `SmolLM2-135M` at `1.4%`, `SmolLM2-360M` at `3.2%`, and `SmolLM2-1.7B` at `31.0%`. So the honest answer is: weak signals appear by `~135M-360M`, but robust emergence is more like `~0.5B-1B+`, unless the recipe is unusually strong. For Sutra, I would expect:
- No meaningful movement on the full 500Q before `~1B` or post-training.
- First movement on a sane exact-match reasoning subset at `~150M-300M`.
- GSM8K-like results worth caring about only once you have both `200M+` and the better `FineWeb-Edu + math/code` recipe.

Sources: [Pythia-70m model card](https://huggingface.co/EleutherAI/pythia-70m), [Pythia-160m-deduped model card](https://huggingface.co/EleutherAI/pythia-160m-deduped/blob/refs%2Fpr%2F4/README.md), [SmolLM2-360M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), [SmolLM2-1.7B model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B).