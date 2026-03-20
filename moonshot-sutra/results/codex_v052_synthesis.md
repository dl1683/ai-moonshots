**v0.5.2**
Optimal `v0.5.2` is **`v0.5` core kept intact**, plus **three changes only**: **2-mode switching transition kernel**, **BayesianWrite gain clamp at `10.0`**, and **peak LR raised to `8e-4`**. Do **not** carry over the `v0.5.1` halting stack. Today’s production-scale evidence says the architecture itself is working, deep recurrence is essential, and lambda-based halting is not calibrated yet. The relevant findings are in [research/RESEARCH.md#L1756](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1756), [research/RESEARCH.md#L1767](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1767), [research/RESEARCH.md#L1783](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1783), [research/RESEARCH.md#L1580](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1580), [research/RESEARCH.md#L1734](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1734), [research/RESEARCH.md#L1676](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1676), and the current baseline in [code/sutra_v05_train.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_train.py).

Use this exact config:

```yaml
version: v0.5.2

model:
  base: SutraV05
  vocab_size: 50257
  dim: 768
  ff_dim: 1536
  seq_len: 512
  max_steps: 8
  window: 4
  k_retrieval: 8
  topk_active_stages: 2
  tied_embeddings: true

transition_kernel:
  type: switching
  n_modes: 2
  hidden: 256
  gate_hidden: 64
  gate_inputs: mu_only
  stage_graph_mask: true

router:
  enabled: true

bayesian_write:
  enabled: true
  gain_clamp_max: 10.0

halting:
  enabled: false

inter_step_loss:
  enabled: false

explicit_verify_head:
  enabled: false

training:
  optimizer: AdamW
  lr_peak: 8.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps: 1000
  lr_schedule: cosine
  max_train_steps: 100000
  batch_size: 8
  grad_accum: 8
  effective_batch: 64
  tokens_per_step: 32768
  grad_clip_norm: 1.0
  precision: bf16
  eval_every: 5000
  save_every: 5000
  log_every: 100
  finite_guard: true
```

**Change from v0.5**
- Replace single transition kernel with **2-mode switching kernel**.
- Keep `max_steps=8`.
- Raise LR from current `6e-4` to **`8e-4`**.
- Clamp BayesianWrite effective gain at **`10.0`**.
- Restart as a **fresh 100K run**.

**Do not change**
- Do not remove router, transitions, or recurrence.
- Do not reduce to `max_steps=5` or `6`.
- Do not add lambda halting, adaptive freezing, or the full `v0.5.1` halting/inter-step stack yet.
- Do not scale to `dim=1024` yet.
- Keep `dim=768`, `ff_dim=1536`, `seq_len=512`, `window=4`, `k_retrieval=8`, bf16, AdamW, cosine, tied embeddings.

**Expected improvement**
- Best estimate: **`~5.63` BPT at 100K**.
- That is about **`0.35` BPT better** than the current `v0.5` trajectory (`~5.98`), or roughly **6% lower BPT**.
- It still leaves you about **58% worse than Pythia-70M**, so this is an optimization/de-risking release, not the scaling release.

**768 vs 1024**
Stay at **`dim=768`** for `v0.5.2`.

Reason: today’s wins are from **optimization and transition structure**, not width. `768` is the validated production scale, the model is not overfitting, and the next clean experiment is to realize the cheap proven gains first. `1024` adds cost and instability before the current architecture has finished proving out at the now-correct settings.