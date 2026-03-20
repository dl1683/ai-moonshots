v0.5.1 should integrate three things now: a 4-mode switching transition kernel, train-time lambda-driven soft halting/freezing, and a halt-aligned intermediate-step loss. That is the highest-impact path for text generation because it makes easy positions stop getting damaged, preserves extra depth for genuinely hard tokens, and gives the recurrent loop a usable stopping law. I would also wire the existing verifier/reroute path into the loop now, because the hooks already exist and are currently dead in [sutra_v05_ssm.py](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L153) and [sutra_v05_ssm.py](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L266).

Use `max_steps=6`, not `8` and not `4`. The probe result says most useful refinement is done by step 4, but a hard cap of 4 would clip the long-tail cases that matter for coherent generation. Six keeps two recovery steps for hard tokens while removing the most harmful tail. With halting integrated correctly, the target should be effective average steps around `3.5-4.2`.

Yes, add intermediate-step loss. The exact weighting I’d use is:
`stop_mass_t = survival_{t-1} * halt_t`, `survival_t = survival_{t-1} * (1 - halt_t)`, `stop_mass_last = survival_{K-1}`.
Then train with `L_total = L_final + 0.25 * L_intermediate`, where
`L_intermediate = sum_t w_t * CE_t / sum_t w_t`,
`w_t = 0.15 / K + 0.85 * stop_mass_t`.
That keeps final generation quality dominant, but forces steps 1-4 to become real candidate readouts instead of untrained internal churn.

For differentiable lambda halting, use a hazard model:
`halt_t = sigmoid(H([mu_t, log(mean(lambda_t)), verify_t]))`.
Freeze only sinks, not sources: future writes, residuals, reroutes, and routing reception are multiplied by `alive_t`; routed messages are still computed from all positions, so halted tokens remain readable context. That matches the Chrome result that freezing must be trained end-to-end, not added post hoc.

Param budget: keep `dim=768` and `ff_dim=1536`. Add only:
- switching-kernel gate, 4 modes, hidden 64: `+49,800`
- halting head, hidden 64: `+49,409`

Current total is `66,954,809`. New total is `67,054,018` params, so v0.5.1 stays effectively a `67.1M` model. That is the right budget. Do not spend budget on width yet; dynamic compute control is a better use of parameters for generation quality.

Exact code changes in [sutra_v05_ssm.py](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py):

Replace the transition kernel at [sutra_v05_ssm.py:53](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L53):
```python
class SwitchingTransitionKernel(nn.Module):
    """Content-dependent transition kernel with a small set of strategy modes."""

    def __init__(self, dim, hidden=256, n_modes=4, gate_hidden=64):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, N_STAGES * N_STAGES),
        )
        self.mode_gate = nn.Sequential(
            nn.Linear(dim + 2, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, n_modes),
        )
        self.mode_logits = nn.Parameter(torch.zeros(n_modes, N_STAGES, N_STAGES))

    def forward(self, h, lam, verify_score):
        B, N, D = h.shape
        base = self.base(h).view(B, N, N_STAGES, N_STAGES)
        lam_stat = torch.log(lam.mean(dim=-1, keepdim=True).clamp(min=1e-6))
        gate_in = torch.cat([h, lam_stat, verify_score], dim=-1)
        mode_mix = F.softmax(self.mode_gate(gate_in), dim=-1)
        mode = torch.einsum("bnm,mij->bnij", mode_mix, self.mode_logits)
        raw = base + mode
        mask = STAGE_GRAPH.to(h.device).unsqueeze(0).unsqueeze(0)
        raw = raw.masked_fill(mask == 0, float("-inf"))
        return F.softmax(raw, dim=-1)
```

Insert this new halting head after the verifier at [sutra_v05_ssm.py:184](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L184):
```python
class HaltingHead(nn.Module):
    """Differentiable per-position halting hazard from state precision + verify."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, mu, lam, verify_score):
        lam_stat = torch.log(lam.mean(dim=-1, keepdim=True).clamp(min=1e-6))
        return torch.sigmoid(self.net(torch.cat([mu, lam_stat, verify_score], dim=-1)))
```

Change `SutraV05.__init__` at [sutra_v05_ssm.py:236](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L236):
```python
def __init__(self, vocab_size=50257, dim=768, ff_dim=1536,
             max_steps=6, window=4, k_retrieval=8,
             read_threshold=0.3, verify_threshold=0.5,
             reroute_alpha=0.3, n_transition_modes=4,
             halt_hidden=64, halt_floor=0.15):
    super().__init__()
    self.dim = dim
    self.vocab_size = vocab_size
    self.max_steps = max_steps
    self.read_threshold = read_threshold
    self.verify_threshold = verify_threshold
    self.reroute_alpha = reroute_alpha
    self.halt_floor = halt_floor

    self.emb = nn.Embedding(vocab_size, dim)
    self.pos_emb = nn.Embedding(2048, dim)

    self.init_mu = nn.Linear(dim, dim)
    self.init_lam = nn.Linear(dim, dim)

    self.transition = SwitchingTransitionKernel(dim, n_modes=n_transition_modes)
    self.stage_bank = StageBank(dim, ff_dim)
    self.router = LocalRouter(dim, window=window, k=k_retrieval)
    self.writer = BayesianWrite(dim)
    self.verifier = Verifier(dim, vocab_size)
    self.halter = HaltingHead(dim, hidden=halt_hidden)

    self.ln = nn.LayerNorm(dim)
```

Replace `forward` at [sutra_v05_ssm.py:266](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L266):
```python
def forward(self, x, targets=None):
    B, T = x.shape
    device = x.device

    h = self.emb(x) + self.pos_emb(torch.arange(T, device=device))
    mu = self.init_mu(h)
    lam = F.softplus(self.init_lam(h)) + 0.1

    pi = torch.zeros(B, T, N_STAGES, device=device, dtype=h.dtype)
    pi[:, :, 2] = 1.0

    alive = torch.ones(B, T, 1, device=device, dtype=h.dtype)
    prev_verify = torch.zeros(B, T, 1, device=device, dtype=h.dtype)
    prev_reroute = torch.zeros(B, T, self.dim, device=device, dtype=h.dtype)

    final_logits = torch.zeros(B, T, self.vocab_size, device=device, dtype=h.dtype)
    expected_steps = torch.zeros(B, T, 1, device=device, dtype=h.dtype)
    inter_num = torch.zeros((), device=device, dtype=h.dtype)
    inter_den = torch.zeros((), device=device, dtype=h.dtype)

    for t in range(self.max_steps):
        expected_steps = expected_steps + alive

        K = self.transition(mu, lam, prev_verify)
        pi_evolved = torch.bmm(
            pi.view(B * T, 1, N_STAGES),
            K.view(B * T, N_STAGES, N_STAGES)
        ).view(B, T, N_STAGES)

        stage_out, evidence = self.stage_bank(mu, pi)
        pi_new = pi_evolved * F.softmax(evidence / 0.5, dim=-1)
        pi_new = pi_new / pi_new.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        pi_active = top2_project(pi_new)
        frozen_pi = torch.zeros_like(pi_active)
        frozen_pi[:, :, 6] = 1.0
        pi = alive * pi_active + (1.0 - alive) * frozen_pi

        route_gate = alive * pi[:, :, 3:4]
        messages = self.router(mu)
        messages = route_gate * (
            messages + self.reroute_alpha * (1.0 - prev_verify) * prev_reroute
        )

        mu, lam = self.writer(mu, lam, messages, alive * pi[:, :, 4:5])
        mu = mu + alive * stage_out * 0.1

        normed = self.ln(mu)
        logits_t = F.linear(normed, self.emb.weight) / math.sqrt(self.dim)

        pred = logits_t.argmax(dim=-1)
        pred_emb = self.emb(pred)
        verify_score, reroute_signal = self.verifier(mu, pred_emb)

        if t == self.max_steps - 1:
            halt_mass = alive
        else:
            halt_mass = alive * self.halter(mu, lam, verify_score)

        final_logits = final_logits + halt_mass * logits_t

        if targets is not None:
            ce_t = F.cross_entropy(
                logits_t.view(B * T, self.vocab_size),
                targets.view(B * T),
                reduction="none",
            ).view(B, T, 1)
            w_t = (self.halt_floor / self.max_steps) + (1.0 - self.halt_floor) * halt_mass
            inter_num = inter_num + (w_t * ce_t).sum()
            inter_den = inter_den + w_t.sum()

        alive = alive * (1.0 - halt_mass)
        prev_verify = verify_score
        prev_reroute = reroute_signal

    aux = {
        "compute_cost": expected_steps.mean(),
        "avg_steps": expected_steps.mean(),
        "mean_lambda": lam.mean(),
    }

    if targets is not None:
        final_loss = F.cross_entropy(
            final_logits.view(B * T, self.vocab_size),
            targets.view(B * T),
        )
        aux["intermediate_loss"] = inter_num / inter_den.clamp(min=1e-6)
        aux["loss"] = final_loss + 0.25 * aux["intermediate_loss"]

    return final_logits, aux
```

Also change the smoke-test defaults at [sutra_v05_ssm.py:328](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L328) and [sutra_v05_ssm.py:346](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L346) from `max_steps=8` to `max_steps=6`.

I could not apply the patch here because the workspace is read-only.