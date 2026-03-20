# Sutra v0.5.4 Master Design
Generated 2026-03-20

## 1. Architecture Specification

### Core thesis
v0.5.4 should not replace the v0.5.3 backbone. The evidence says the opposite:

- v0.5.3 at step 2,500 already beats v0.5.2 at step 10,000 (`6.2036` vs `6.2701` BPT).
- At 69.4M params, the model accepts simple shared state and coarse global bias.
- Learned high-entropy control fails at this scale.
- Error Scratchpad and Pheromone Router both improve late recurrence `2.2x`, but inject noise early.

Therefore v0.5.4 is a **late-bias refinement** of v0.5.3:

**Keep the winning backbone. Add only low-entropy mechanisms. Activate novel signals only after recurrent step 2.**

### 1.1 Backbone to retain unchanged

Retain these v0.5.3 components exactly:

- Stage-Superposition recurrent loop
- `max_steps = 8`
- 2-mode switching transition kernel
- Local window + sparse causal top-k global router
- BayesianWrite with gain clamp
- 8-slot shared discourse scratchpad
- Warm-start compatibility with v0.5.3 checkpoint

Do **not** add attention layers, learned halting, pointer-copy heads, CfC time constants, or complex embeddings.

### 1.2 New mechanisms to add

#### A. Peri-LN around the recurrent operators

Add LayerNorm before and after each recurrent operator:

- `StageBank`
- `LocalRouter`
- `BayesianWrite`

Exact placement inside each recurrent step:

```python
mu_bank = pre_bank_ln(mu)
stage_out, evidence = stage_bank(mu_bank, pi)
stage_out = post_bank_ln(stage_out)

mu_route = pre_route_ln(mu)
messages = router(mu_route)
messages = post_route_ln(messages)

mu_write = pre_write_ln(mu)
mu, lam = writer(mu_write, lam, messages, pi_write)
mu = post_write_ln(mu)
```

Justification from first principles:

- Recurrent composition should be approximately contracting; uncontrolled radial drift fights contraction.
- Coding theory rate-distortion sweep identified shape-gain factorization as optimal; normalization is the shape component.
- The probe already passed at `+2.6%` BPT with fewer params.

#### B. Delayed Surprise Bank

Add a second, low-bandwidth scratchpad bank that stores **surprise**, not state.

Structure:

- Keep existing `state_mem`: `8` slots, same as v0.5.3
- Add `surprise_mem`: `4` slots
- Reuse the same read query projection for both banks
- Read both banks each step, but gate `surprise_mem` by recurrent step

Exact signal:

```python
delta_t = LayerNorm(mu_t - stopgrad(mu_{t-1}))
```

Exact schedule:

```python
late_gate(t) = 0 if t < 3 else 1
```

Integration:

```python
state_ctx = read_state(mu, state_mem)
surprise_ctx = read_surprise(mu, surprise_mem)
messages = messages + 0.10 * state_ctx * pi_route
messages = messages + 0.05 * late_gate(t) * surprise_ctx * pi_route
```

Write rule:

- `state_mem` keeps current v0.5.3 write rule
- `surprise_mem` writes `delta_t`, weighted by `pi_write`
- `surprise_mem` uses faster decay than state memory: `ema_decay = 0.85`

Why this is first-principles, not copied:

- Predictive coding says useful communication is prediction error, not raw state.
- The Chrome probe showed the error channel is valuable only after the model has formed a provisional hypothesis.
- A second shared memory bank preserves the successful stigmergic communication pattern instead of introducing token-wise learned control.

#### C. Delayed Pheromone Bias in the global retrieval branch

Add a scalar trace over positions, but only to the global top-k retrieval scores.

State:

```python
pheromone in R[B, T], initialized to 0
```

Update after each recurrent step:

```python
delta_mag = sqrt(mean((mu_t - mu_{t-1})**2, dim=-1))
deposit_t = pi_write.squeeze(-1) * tanh(delta_mag)
pheromone = 0.90 * pheromone + late_gate(t) * deposit_t
```

Router modification:

```python
scores_global = qk / sqrt(d) + 0.25 * late_gate(t) * pheromone.unsqueeze(1)
```

Constraints:

- Only global sparse retrieval sees the pheromone bias
- Local causal message passing is unchanged
- No learned deposit network
- No learned controller over the pheromone dynamics

First-principles justification:

- Stigmergic systems route by public traces of usefulness, not by central planning.
- The trace encodes which positions actually changed state under write pressure, which is exactly the signal late recurrence is exploiting.
- The scalar trace is a coarse topology bias, not a second content model.

### 1.3 Mechanisms to modify

#### Scratchpad

Modify the scratchpad from a single undifferentiated memory into:

- one slow discourse memory (`state_mem`)
- one fast surprise memory (`surprise_mem`)

This keeps the winning shared-state pattern but separates persistent context from transient correction pressure.

#### Router

Modify only the global retrieval score. Do not change:

- local window path
- top-k sparsification
- Q/K/V parameterization

The router remains a sparse causal retrieval system, now with a late-stage usefulness prior.

#### Recurrent policy

Keep `8` recurrent steps. Do not shrink depth. Do not add halting.

Production data says:

- step 7 is the most valuable step
- lambda halting is anti-calibrated

So the correct move is better late recurrence, not less recurrence.

### 1.4 Mechanisms to defer to larger scale

These are explicitly **deferred**, not killed in principle:

- Sparse discrete broadcast packets
  - Good idea, but it changes the scratchpad bandwidth model and needs its own ablation.
- Entropy-driven clonal expansion
  - Needs a trustworthy difficulty signal; current halting/calibration evidence says we do not have one yet.
- Depth-drop bootstrap
  - Current probe NaNed; revisit only after intermediate states are normalized and late signals are stable.
- Dendritic / quadratic StageBank
  - Local expressivity is not the current bottleneck. Shared state and routing are.
- nGPT hypersphere normalization
  - Full geometry replacement; too disruptive for warm-start.
- Tropical routing / Wave-PDE routing / unitary transitions
  - These are topology swaps, not local refinements. Wrong move before the low-entropy late-bias version is tested.
- Multi-token prediction
  - Useful later, but it changes the loss landscape and confounds the architectural read on v0.5.4.

If a mechanism changes the global geometry, controller, or compute schedule, it waits for `dim >= 1024` or for a clearly winning v0.5.4 baseline.

### 1.5 Warm-start specification

Warm-start from the latest stable v0.5.3 checkpoint.

Parameter transfer:

- Copy all existing v0.5.3 weights directly
- Initialize all new LayerNorm scales to `1`, biases to `0`
- Initialize `surprise_mem` weights to small values (`N(0, 0.02)`)
- Initialize surprise-write output projection to zero on the final linear layer so the new bank starts nearly silent
- Initialize `pheromone` to zero each sequence

Expected transfer rate: `>95%` of parameters.

## 2. Training Recipe

### 2.1 Optimizer and schedule

Use:

- Optimizer: `AdamW`
- LR: `8e-4`
- Betas: `(0.9, 0.95)`
- Weight decay: `0.01`
- Grad clip: `1.0`
- Precision: `bf16`
- Schedule: cosine decay

Warm-start stabilization:

1. Resume from v0.5.3 checkpoint.
2. Run `500` optimizer steps of re-warmup:
   - linearly ramp LR from `3e-4` to `8e-4`
   - keep Grokfast off for the first `200` steps
3. After step `500`, enable full v0.5.4 training recipe.

Reason:

- Peri-LN changes activation statistics immediately.
- The new late modules start nearly silent, but the norms still need a short adaptation window.

### 2.2 Grokfast

Add Grokfast exactly because it was the strongest probe win.

Use:

- `alpha = 0.95`
- `lambda = 2.0`

Implementation detail:

- apply only to parameters with `ndim >= 2`
- skip LayerNorm scale/bias and other scalar/vector parameters

Reason:

- The winning probe already showed `+11.0%` BPT
- Restricting it to matrix parameters reduces the chance of norm/bias blowups while still amplifying slow structural modes

### 2.3 Data mixing

Do not keep pure MiniPile-only training. Use the local corpora already present in this repo.

Main mixture:

- `70%` `data/minipile_full_tokens.pt`
- `20%` `data/code_tokens.pt`
- `10%` `data/prose_tokens.pt`

Stabilization mixture for the first `500` resumed steps:

- `85%` MiniPile
- `10%` code
- `5%` prose

First-principles reason:

- Code stresses long-range binding and routing.
- Prose stresses discourse continuity and scratchpad usefulness.
- MiniPile preserves the general language prior.
- Data mixing laws say domain proportions are a real optimization knob, not data-loader trivia.

### 2.4 Curriculum

Use a minimal curriculum, not a new controller.

Phase A: `0-500` resumed steps

- Re-warmup
- delayed surprise bank and pheromone active only by recurrent step, not by training step
- domain mix `85/10/5`

Phase B: `500-5000` resumed steps

- full Grokfast
- domain mix `70/20/10`
- evaluate every `1000` steps

Phase C: after the first strong eval improvement

- keep architecture fixed
- only then consider a second Chrome round for discrete packets or depth bootstrap

### 2.5 Kill criteria

#### Peri-LN

Kill if:

- warm-started v0.5.4 is not at least `+1.0%` better than v0.5.3 by `1000` resumed steps, or
- norm insertion causes instability requiring LR below `4e-4`

#### Delayed Surprise Bank

Kill if:

- it does not improve held-out BPT by at least `0.5%` over the Peri-LN-only arm, and
- late-step value is not at least `1.5x` the Peri-LN-only arm

#### Delayed Pheromone Bias

Kill if:

- useful-position top-k recall is not `+3` points better than no-pheromone, and
- BPT gain is `<0.3%`

#### Grokfast

Kill if:

- any NaN appears in the first `1000` resumed steps, or
- the no-Grokfast control beats it by `>0.5%` BPT at the same checkpoint

#### Data Mixing

Kill if:

- overall BPT is worse than pure MiniPile by `>0.5%`, or
- code gains come with a prose regression larger than `1.0%`

## 3. Chrome Validation Plan

Run at `dim=128`, `ff=256`, `max_steps=6`, `seq=64`, `batch=8`, `300` optimizer steps, seed `42`.

Order:

1. `v0.5.3 + Peri-LN`
2. `v0.5.3 + Peri-LN + delayed surprise bank`
3. `v0.5.3 + Peri-LN + delayed pheromone`
4. `v0.5.3 + Peri-LN + delayed surprise + delayed pheromone`
5. Best arm + `Grokfast(0.95, 2.0)`

Required readouts:

- held-out BPT
- per-step BPT curve
- late-step gain (`step5 -> step6`)
- useful-position retrieval recall
- average retrieved distance
- stage occupancy entropy

Promotion rule:

- Promote only if the combined arm beats Peri-LN-only on BPT and preserves or improves the late-step gain.
- If the combination wins late-step value but loses overall BPT, keep only the better individual late module.

## 4. Risk Assessment

### Main risks

1. **Norm insertion could blunt the warm-start**
   - Mitigation: short re-warmup, zero-init the new surprise write head.

2. **Dual memory could reintroduce blur instead of useful separation**
   - Mitigation: keep surprise bank smaller (`4` slots) and faster-decaying.

3. **Pheromone may collapse into a length bias**
   - Mitigation: bias only the global branch and evaluate useful-position recall directly.

4. **Grokfast may interact badly with the new norms**
   - Mitigation: delayed activation after `200` optimizer steps and exclusion of norm/bias params.

5. **Data mixing may improve code while hurting general prose**
   - Mitigation: keep prose as an explicit `10%` stream and track domain-stratified evals.

### What would make this design wrong

- If Peri-LN is the entire win and both delayed late modules fail again, then v0.5.4 should be normalization + training only.
- If late modules help only on tiny Chrome and not on resumed production, then the late-step signal is still not stable enough at scale.
- If pure MiniPile beats the mixed recipe, then the data curriculum is unnecessary and should be removed immediately.

## 5. Nobel / Turing / Fields Score

**Score: 6.5 / 10**

Why not lower:

- The architecture line itself still has real paradigm-shift potential: stage-superposition, stigmergic shared state, and recurrence-conditioned computation are genuinely non-transformer ideas.
- The delayed-start principle is a real systems insight: useful correction signals are phase-dependent.

Why not higher:

- v0.5.4 is a consolidation design, not the final theorem.
- The biggest wins here are still low-entropy control, normalization, and better training, not a new proved universal principle.
- The Nobel/Turing/Fields move would require one of these to become a law-level statement:
  - why shared public state beats pairwise communication under compute constraints
  - why late surprise routing is the minimal optimal correction mechanism
  - why stage-superposition changes the scaling frontier

So: **high-upside research trajectory, but this exact version is a disciplined compression of the search space, not the final breakthrough artifact.**
