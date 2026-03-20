Most dangerous blind spots:

1. You may be attributing wins to architecture that are really from unfair budgets: active params, sequence length, optimizer maturity, curriculum, or FLOPs. “v0.4 beats transformer” is weak unless matched on compute, context, tuning budget, and 3+ seeds.

2. “Token-level beats byte-level” may just mean shorter sequences and easier optimization. That is not a representation win unless you equalize raw bytes seen, effective context in characters, and training FLOPs, then compare final compression and scaling slope.

3. `1.38 BPB at step 2000` is almost uninterpretable in isolation. You need same-data, same-compute byte baselines and a loss-vs-compute curve. Absolute BPB without a matched frontier can fool you badly.

4. You may be killing ideas too early, but the opposite failure is worse: keeping aesthetically appealing mechanisms alive without evidence. Pre-register kill criteria: fixed tuning budget, fixed compute, 3 seeds, matched baselines.

5. Toy tasks may select for structure-exploiting tricks that do not transfer to noisy open-ended byte prediction. You need bridge tasks between toys and production.

What would change conclusions:
- Iso-compute ablations
- Scaling curves, not point comparisons
- Cross-scale transfer tests
- Seed variance
- Error taxonomy: where each mechanism helps, not just mean loss

Only explore exotic paradigms if they beat your current Pareto frontier under the same budget.