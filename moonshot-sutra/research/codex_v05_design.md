**Verdict**

No. Right now the repo is building a strong hybrid recurrent LM, not the stage-superposition machine described in [research/STAGE_ANALYSIS.md](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/STAGE_ANALYSIS.md). [`code/sutra_v04.py`](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v04.py) is a fixed sequence of patch GRU → message passing → sparse retrieval → broadcast. The existing [`code/sutra_v05.py`](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05.py) is still that same template with routing upgrades, not a state graph. So the user’s criticism is correct.

What v0.5 should be is an actual **Stage-Superposition State Machine**: per-position stage probabilities, explicit stage-transition dynamics, intrinsic compute control, and a real verify→reroute loop.

**v0.5 Design**

Let each active position carry

```text
z_i^t = (mu_i^t, lambda_i^t, a_i^t, pi_i^t, emitted_i^t)
mu_i^t      in R^d        feature mean
lambda_i^t  in R_+^d      precision = confidence
a_i^t       in R^r        dynamic address
pi_i^t      in Delta^7    stage probability vector
```

Use an explicit stage graph, not a classifier-only shortcut. Allowed edges:

```text
1->{1,2,3}
2->{2,3,4}
3->{3,4,5}
4->{4,5,6,7}
5->{4,5,6,7}
6->{4,6,7}
7->{4,7}
```

For each step:

```text
K_i^t = MaskedRowSoftmax(G(z_i^t))              # 7x7 transition kernel, content-dependent
y_{i,s}^t = F_s(z^t)_i                          # proposal from stage s
e_{i,s}^t = u_s^T y_{i,s}^t                     # evidence that stage s is useful now

pi_i^{t+1} = Top2Project( ((pi_i^t K_i^t) ⊙ exp(e_i^t / tau)) / Z_i^t )
```

That is the core change: **processing is driven by transitions on the stage graph**.

Stage 4 routing should be a budgeted transport problem:

```text
P^t = argmax_P  <S D^T + B(a), P> - eps H(P)
s.t. P >= 0, causal, row/col budgets, sparse top-k support
```

`S` is supply, `D` is demand. This gives routed message

```text
r_i^t = sum_j P_ij^t V mu_j^t
```

Stage 5 is Bayesian evidence accumulation, not residual add:

```text
m_i^t = Wm[mu_i^t ; r_i^t]
kappa_i^t = softplus(Wk[mu_i^t ; r_i^t])        # evidence gain >= 0

lambda_i^{t+1} = lambda_i^t + pi_i^{t+1}[5] * kappa_i^t
mu_i^{t+1} = (lambda_i^t ⊙ mu_i^t + pi_i^{t+1}[5] * kappa_i^t ⊙ m_i^t) / lambda_i^{t+1}
```

Stage 6 is intrinsic:

```text
continue_i^t = 1[
    (1 - pi_i^{t+1}[7]) +
    beta * mean(1 / lambda_i^{t+1}) +
    gamma * max(0, theta_v - v_i^t)
    > epsilon
]
```

Stage 7 is real decode→verify→reroute:

```text
logits_i^t = Wo mu_i^{t+1}
yhat_i^t = argmax(logits_i^t)
v_i^t = sigmoid(Vf[mu_i^{t+1} ; r_i^t ; emb(yhat_i^t)])

if v_i^t < theta_v:
    pi_i^{t+1}[4] += alpha
    pi_i^{t+1}[7] -= alpha
    demand_i^{t+1} += Werr[emb(yhat_i^t) ; mu_i^{t+1}]
else:
    emitted_i = 1
```

That loopback is not simulated. Failed readout literally changes next-step routing demand.

**Forward Pass Pseudocode**

```python
def forward(bytes):
    seg = soft_segment(bytes)                     # Stage 1
    state = init_state(seg)                       # Stage 2

    for t in range(T_max):
        active = ~state.emitted
        if active.sum() == 0:
            break

        K = transition_kernel(state[active])
        proposals = stage_bank(state, active)    # F1..F7
        pi = update_stage_probs(state.pi[active], K, proposals)
        pi = top2_project(pi)                    # keep real superposition, bounded cost

        state = refine_segments_and_addresses(state, proposals, pi)
        routed = transport_route(state, pi)      # Stage 4
        state = bayes_write(state, routed, pi)   # Stage 5

        cand = pi[:, 7] > read_threshold
        logits = readout(state[cand])
        verify = verifier(state[cand], logits)

        fail = cand & (verify < theta_v)
        ok = cand & ~fail

        state.emitted[ok] = True
        state = reroute_failed_positions(state, fail, logits)

        state.active = compute_continue_mask(state, pi, verify) & ~state.emitted

    return final_logits_from_emitted_or_last_state(state)
```

**Training**

Use four losses:

```text
L = L_next_token
  + λc L_compute
  + λv L_verify
  + λr L_reroute
  + λs L_stage_entropy
```

Where:

- `L_next_token`: standard teacher-forced CE on final accepted readout.
- `L_compute = mean_t,i continue_i^t`: penalize wasted rounds.
- `L_verify`: BCE forcing verifier to predict whether the current Stage-7 proposal is actually correct.
- `L_reroute`: after a failed verify, the routing plan at `t+1` must differ from `t`; use a margin on `JS(P_i^{t+1}, P_i^t)`.
- `L_stage_entropy`: target entropy near `log 2`, so positions stay in sparse superposition instead of collapsing to one stage or smearing over seven.

Curriculum:

1. `0-20%` of tokens: only Stages 1-3-5-7 active. Learn segmentation, addressing, local construction, basic write/readout.
2. `20-60%`: open Stage 4 and train routing on prose+code+synthetic long-range tasks.
3. `60-85%`: enable Stage 6 compute penalty and Stage 7 verifier.
4. `85-100%`: full loopback training, sample extra from positions that fail verification.

For a single RTX 5090, keep it feasible by:
- shared weights across recurrent steps
- `Top2Project(pi)` so each position executes at most two stages per round
- BF16 + checkpointing
- max 10-12 rounds, target average 4-5
- early freezing of emitted positions

**Parameters at `d=768`**

For a practical tied-weight version with shared recurrent stage bank and an `8k` segment codebook:

- Core recurrent state machine: about `28M`
- Segment codebook: about `6.3M`
- Total tied: about `34M`
- Total untied decoder: about `40M`

That is small enough for `1.7B` tokens on one 24GB card and large enough to test the idea honestly.

**Why This Is Actually Different**

It is not “another LM with some routing tricks” because:

- The computational state is **stage occupancy**, not layer index.
- Depth is **per-position and content-dependent**, not global.
- Verification is **inside** generation and can causally force rerouting before emission.
- Confidence is a **state variable** (`lambda`), reused by write, control, and verify.
- Communication is a **budgeted transport problem**, not blanket all-to-all mixing.

There are neighbors in the literature, but the integrated object here is different: a language model whose primitive is **state-graph evolution**, not stacked layers.

**Potential**

If this works cleanly and beats strong baselines, the upside is high.

- Nobel: `1/10`
- Turing: `8/10`
- Fields: `2/10`

Bluntly: the relevant ceiling is Turing-level, not Nobel/Fields. Today it is still a `3/10` idea in evidence terms. If it works, it jumps because it would replace “depth as layers” with “computation as per-position state evolution,” which is a real architectural shift.