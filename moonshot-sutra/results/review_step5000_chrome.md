The 3 CPU experiments I’d run now are:

1. **Checkpoint step-budget sweep**
Run the 5K checkpoint at forced recurrent depths `1..8` on a fixed held-out shard and measure BPT plus per-token win/loss. This tells you whether `max_steps=8` is still earning its keep at production scale.
Decision rule: if step 5 is within `<=1%` of step 8 BPT, adaptive depth / lower max-steps becomes the next highest-ROI change.

2. **Mechanism attribution ablation**
Evaluate the same checkpoint with `router off`, `BayesianWrite off/frozen`, and `stage transitions collapsed` to see what actually buys the 6.88 BPT.
Decision rule: any ablation costing `>5%` BPT is core; anything costing `<1-2%` is probably scaffolding and should not dominate v0.5.2 work.

3. **Halting-signal calibration**
Log per-token `lambda`, stage entropy, and stage mass at intermediate steps, then correlate them with final correctness and with “did extra steps help?” This directly tests whether the earlier lambda/entropy-halting story survives at 67M scale.
Decision rule: if AUROC is `>=0.65` or correlation is `>=0.25`, prioritize adaptive halting next; otherwise deprioritize it.

**Recommendation:** let this run continue at `6e-4`.

Why:
- Your best production-scale proxy already says `6e-4` and `8e-4` are basically tied: `9.57` vs `9.56` BPT. That is a `0.01` BPT difference, effectively noise.
- The current run is already your best clean result: `7.0663 -> 6.8805` at 5K, a real `2.6%` gain over the old `3e-4` run.
- NaN tax is tiny. `4 / 5000` skipped steps is only `0.08%`. Extrapolated over the remaining `95K` steps, that’s about `76` skipped steps. Even using the worse early-window rate (`4 / 1747 = 0.23%`), it’s only about `219` steps. That is not enough to justify an LR change midstream.

So:
- **Do not restart from scratch.**
- **Do not switch this run to `8e-4`.**
- If you’re willing to interrupt once, the only intervention I’d consider is **resume from the 5K checkpoint with the clamp but keep base LR at `6e-4`**. Otherwise just let it finish and use `clamp + 8e-4` on the next fresh run.