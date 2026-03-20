**A) Next 3 architectural modifications, ranked**

1. **Make “done” positions absorbing via uncertainty-gated freezing**
   The 30% hurt result says solved positions are still being perturbed. Add a monotonic `frozen/emitted` mask: once entropy or variance drops below threshold, that position stops receiving writes and reroutes. Let frozen positions remain as message sources, not sinks. This should improve loss and cut compute immediately.

2. **Replace scalar confidence with a true uncertainty state: `(mu, log_var)` or `(mu, lambda)`**
   This is the missing glue for Stages 5, 6, and 7. Write should reduce uncertainty monotonically; control should halt based on uncertainty; verify should abstain/reroute based on the same signal. Right now Stage 6 is dead because it has no native state variable to read. This is the highest-upside coherence fix.

3. **Turn the transition kernel into a switching dynamical system**
   Use 2-4 learned “strategy modes” with content-conditioned mixing, not one universal kernel. Example modes: local-fast, route-heavy, verify-heavy. Prose and code are already separating; give the model a cleaner basis for that separation. This should amplify the novel effect and make trajectories more interpretable.

**B) What the 30% hurt positions mean, and the fix**

They imply the recurrence is **not contractive on solved states**. The architecture has dynamics, but not stable attractors. Once a token is effectively solved, later route/write steps can overshoot, over-smooth, or inject irrelevant evidence.

The specific fix is: **irreversible early freeze**.
A position that is low-entropy or passes verify should:
- stop being updated,
- stop being rerouted,
- optionally remain readable by other positions.

That turns “correct enough” into an absorbing state instead of a temporary state that later steps can destroy.

**C) Stage 6 never activates: force, remove, or wait?**

**Do not force it as a standalone stage, and do not remove it yet.**
Reinterpret it first.

Stage 6 should be a **derived control field** from uncertainty, not a separate learned behavior competing with 4/5/7. If you add variance/precision state and Stage 6 still has zero occupancy after that, then remove it as an explicit node and keep compute control continuous.

So: **wait on removal, but change its definition next.**

**D) How to amplify content-dependent transitions**

1. **Give transitions better sufficient statistics**
   Feed the kernel not just hidden state, but local entropy profile, segment-boundary signal, retrieval gain, and verify-failure residual. Those are the variables that actually distinguish prose from code.

2. **Train on sharper regime contrast**
   Mix domains where optimal trajectories are maximally different:
   plain prose, code, tables/JSON, bracket languages, long-range copy tasks, local-noise denoising. The more distinct the dependency geometry, the more distinct the learned trajectories.

3. **Add a trajectory-specialization objective**
   Reward transition-path differentiation when it improves prediction. Concretely: encourage high mutual information between content regime and stage path, while regularizing against trivial domain-label memorization. You want different trajectories because the dependency structure differs, not because the topic differs.

4. **Use strategy-mode kernels**
   A small mixture over transition operators will likely strengthen prose/code separation much more than one dense kernel can.

**E) Three CPU experiments to run now**

1. **Replay-based adaptive freezing on saved traces**
   Take existing multi-step trajectories and simulate halting at steps 2/3/4 using entropy thresholds.
   Hypothesis: you recover most of the 1-4 step benefit, remove many hurt positions, and save 30-50% recurrent compute.

2. **Tiny switching-kernel ablation on mixed synthetic tasks**
   Compare one transition kernel vs 2-4 strategy kernels on a toy corpus mixing local tasks and long-range routing tasks.
   Hypothesis: mixed kernels increase path divergence and improve performance specifically on heterogeneous batches.

3. **Kalman-write vs additive-write on tiny CPU models**
   Same parameter budget, same tasks, but Stage 5 uses either residual/additive update or precision-weighted update.
   Hypothesis: uncertainty-aware write reduces over-processing, improves calibration, and gives a real Stage 6 signal.

**F) The deeper mathematical insight**

The missing math is a **compute-distortion or free-energy law for recurrence**.

Right now the model has a state graph, but no principle saying each extra step must reduce a scalar objective. That is why some positions improve and some degrade. You need a per-position Lyapunov quantity like:

`F = predictive uncertainty + verify inconsistency + lambda * compute cost`

Each recurrent step should decrease `F`, and halting should occur when expected decrease falls below compute cost.

If you formulate v0.5 as **successive refinement with optimal stopping**, a lot snaps into place:
- Stage path = refinement policy
- Halting = marginal rate-distortion test
- Verify = consistency check on residual distortion
- Content dependence = different inputs have different compute-distortion curves

That would make v0.5 fundamentally better, because it gives the recurrence a law, not just a mechanism.