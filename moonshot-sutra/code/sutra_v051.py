"""Sutra v0.5.1: Stage-Superposition with Switching Kernel + Lambda Halting.

Upgrades from v0.5 (validated by Chrome probes + Codex architecture review):
1. 4-mode switching transition kernel (+4.1% BPT at 0.2% param cost)
2. Differentiable lambda-based halting (stops 30% of positions hurt by over-processing)
3. Inter-step loss (forces all steps to be valid readouts, enables adaptive depth)
4. Verify + reroute wired back in (prev_verify feeds into kernel + halting)
5. max_steps=6 (91% benefit in 4, 2 recovery steps for hard tokens)

67.1M params at dim=768. Use LR=1e-3 (validated: +15.2% over 3e-4).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stage graph
STAGE_GRAPH = torch.tensor([
    [1,1,1,0,0,0,0],[0,1,1,1,0,0,0],[0,0,1,1,1,0,0],
    [0,0,0,1,1,1,1],[0,0,0,1,1,1,1],[0,0,0,1,0,1,1],[0,0,0,1,0,0,1],
], dtype=torch.float32)
N_STAGES = 7


def top2_project(pi):
    top2_vals, top2_idx = pi.topk(2, dim=-1)
    result = torch.zeros_like(pi)
    result.scatter_(-1, top2_idx, top2_vals)
    return result / result.sum(dim=-1, keepdim=True).clamp(min=1e-8)


class SwitchingTransitionKernel(nn.Module):
    """4-mode content-dependent transition kernel."""
    def __init__(self, dim, hidden=256, n_modes=4, gate_hidden=64):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, N_STAGES*N_STAGES))
        self.mode_gate = nn.Sequential(nn.Linear(dim+2, gate_hidden), nn.SiLU(), nn.Linear(gate_hidden, n_modes))
        self.mode_logits = nn.Parameter(torch.zeros(n_modes, N_STAGES, N_STAGES))

    def forward(self, h, lam, verify_score):
        B, N, D = h.shape
        base = self.base(h).view(B, N, N_STAGES, N_STAGES)
        lam_stat = torch.log(lam.mean(dim=-1, keepdim=True).clamp(min=1e-6))
        gate_in = torch.cat([h, lam_stat, verify_score], dim=-1)
        mode_mix = F.softmax(self.mode_gate(gate_in), dim=-1)
        mode = torch.einsum('bnm,mij->bnij', mode_mix, self.mode_logits)
        raw = base + mode
        mask = STAGE_GRAPH.to(h.device).unsqueeze(0).unsqueeze(0)
        return F.softmax(raw.masked_fill(mask == 0, float('-inf')), dim=-1)


class StageBank(nn.Module):
    """7 stage-specific MLPs, only active stages computed."""
    def __init__(self, dim, ff_dim=None):
        super().__init__()
        ff = ff_dim or dim * 2
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, ff), nn.SiLU(), nn.Linear(ff, dim))
            for _ in range(N_STAGES)
        ])
        self.evidence = nn.Linear(dim, N_STAGES)

    def forward(self, h, pi):
        B, N, D = h.shape
        active_mask = pi > 0
        weighted = torch.zeros(B, N, D, device=h.device, dtype=h.dtype)
        for s, stage_fn in enumerate(self.stages):
            if active_mask[:, :, s].any():
                weighted = weighted + pi[:, :, s:s+1] * stage_fn(h)
        return weighted, self.evidence(weighted)


class BayesianWrite(nn.Module):
    """Precision-weighted state updates."""
    def __init__(self, dim):
        super().__init__()
        self.msg_proj = nn.Linear(dim * 2, dim)
        self.gain_proj = nn.Linear(dim * 2, dim)

    def forward(self, mu, lam, message, pi_write):
        combined = torch.cat([mu, message], dim=-1)
        m = self.msg_proj(combined)
        kappa = F.softplus(self.gain_proj(combined))
        effective_gain = (pi_write * kappa).clamp(max=10.0)  # prevent NaN from unbounded precision
        lam_new = lam + effective_gain
        mu_new = (lam * mu + effective_gain * m) / lam_new.clamp(min=1e-6)
        return mu_new, lam_new


class LocalRouter(nn.Module):
    """Causal local + sparse global routing."""
    def __init__(self, dim, window=4, k=8):
        super().__init__()
        self.window, self.k = window, k
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.msg_net = nn.Sequential(nn.Linear(dim*2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.out_proj = nn.Linear(dim*2, dim)

    def forward(self, mu):
        B, N, D = mu.shape
        padded = F.pad(mu, (0,0, self.window,0))
        neighbors = torch.stack([padded[:, self.window-w:self.window-w+N, :] for w in range(1, self.window+1)], dim=2)
        self_exp = mu.unsqueeze(2).expand_as(neighbors)
        local_msgs = self.msg_net(torch.cat([self_exp, neighbors], dim=-1)).mean(dim=2)
        q, k, v = self.q_proj(mu), self.k_proj(mu), self.v_proj(mu)
        scores = torch.bmm(q, k.transpose(1,2)) / math.sqrt(D)
        scores = scores + torch.triu(torch.ones(N,N,device=mu.device)*float('-inf'), diagonal=1)
        if N > self.k:
            topk_v, topk_i = scores.topk(self.k, dim=-1)
            sparse = torch.full_like(scores, float('-inf'))
            sparse.scatter_(2, topk_i, topk_v)
            attn = F.softmax(sparse, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)
        return self.out_proj(torch.cat([local_msgs, torch.bmm(attn, v)], dim=-1))


class Verifier(nn.Module):
    """Verify readout quality, produce reroute signal."""
    def __init__(self, dim):
        super().__init__()
        self.verify_net = nn.Sequential(nn.Linear(dim*2, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.reroute_proj = nn.Linear(dim, dim)

    def forward(self, mu, pred_emb):
        v = torch.sigmoid(self.verify_net(torch.cat([mu, pred_emb], dim=-1)))
        return v, self.reroute_proj(mu - pred_emb)


class HaltingHead(nn.Module):
    """Differentiable halting from precision + verify score."""
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.net[-1].bias, -2.0)  # start conservative (low halt prob)

    def forward(self, mu, lam, verify_score):
        lam_stat = torch.log(lam.mean(dim=-1, keepdim=True).clamp(min=1e-6))
        return torch.sigmoid(self.net(torch.cat([mu, lam_stat, verify_score], dim=-1)))


class SutraV051(nn.Module):
    """v0.5.1: Stage-Superposition with switching kernel, lambda halting, verify loop."""

    def __init__(self, vocab_size=50257, dim=768, ff_dim=1536,
                 max_steps=6, window=4, k_retrieval=8, reroute_alpha=0.3):
        super().__init__()
        self.dim, self.vocab_size, self.max_steps = dim, vocab_size, max_steps
        self.reroute_alpha = reroute_alpha
        self.halt_floor = 0.15

        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(2048, dim)
        self.init_mu = nn.Linear(dim, dim)
        self.init_lam = nn.Linear(dim, dim)

        self.transition = SwitchingTransitionKernel(dim)
        self.stage_bank = StageBank(dim, ff_dim)
        self.router = LocalRouter(dim, window=window, k=k_retrieval)
        self.writer = BayesianWrite(dim)
        self.verifier = Verifier(dim)
        self.halter = HaltingHead(dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, targets=None):
        B, T = x.shape
        device = x.device

        h = self.emb(x) + self.pos_emb(torch.arange(T, device=device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=device, dtype=h.dtype)
        pi[:, :, 2] = 1.0  # start at Stage 3

        alive = torch.ones(B, T, 1, device=device, dtype=h.dtype)
        prev_verify = torch.zeros(B, T, 1, device=device, dtype=h.dtype)
        prev_reroute = torch.zeros(B, T, self.dim, device=device, dtype=h.dtype)
        final_logits = torch.zeros(B, T, self.vocab_size, device=device, dtype=h.dtype)
        expected_steps = torch.zeros(B, T, 1, device=device, dtype=h.dtype)
        inter_num = torch.zeros((), device=device, dtype=h.dtype)
        inter_den = torch.zeros((), device=device, dtype=h.dtype)

        for t in range(self.max_steps):
            expected_steps = expected_steps + alive

            # Stage transitions (switching kernel uses precision + verify)
            K = self.transition(mu, lam, prev_verify)
            pi_evolved = torch.bmm(
                pi.view(B*T, 1, N_STAGES), K.view(B*T, N_STAGES, N_STAGES)
            ).view(B, T, N_STAGES)

            # Stage bank
            stage_out, evidence = self.stage_bank(mu, pi)
            pi_new = pi_evolved * F.softmax(evidence / 0.5, dim=-1)
            pi = top2_project(pi_new / pi_new.sum(-1, keepdim=True).clamp(min=1e-8))

            # Route (with reroute signal from previous verify failure)
            route_gate = alive * pi[:, :, 3:4]
            messages = self.router(mu)
            messages = route_gate * (messages + self.reroute_alpha * (1.0 - prev_verify) * prev_reroute)

            # Bayesian write
            mu, lam = self.writer(mu, lam, messages, alive * pi[:, :, 4:5])
            mu = mu + alive * stage_out * 0.1

            # Readout + verify
            logits_t = F.linear(self.ln(mu), self.emb.weight) / math.sqrt(self.dim)
            pred_emb = self.emb(logits_t.argmax(dim=-1))
            verify_score, reroute_signal = self.verifier(mu, pred_emb)

            # Halting (lambda-based, differentiable)
            halt_mass = alive * self.halter(mu, lam, verify_score) if t < self.max_steps - 1 else alive
            final_logits = final_logits + halt_mass * logits_t

            # Inter-step loss
            if targets is not None:
                ce_t = F.cross_entropy(
                    logits_t.view(B*T, self.vocab_size), targets.view(B*T), reduction='none'
                ).view(B, T, 1)
                w_t = (self.halt_floor / self.max_steps) + (1.0 - self.halt_floor) * halt_mass
                inter_num = inter_num + (w_t * ce_t).sum()
                inter_den = inter_den + w_t.sum()

            alive = alive * (1.0 - halt_mass)
            prev_verify = verify_score
            prev_reroute = reroute_signal

        aux = {
            "compute_cost": expected_steps.mean(),
            "avg_steps": expected_steps.mean().item(),
            "mean_lambda": lam.mean().item(),
        }
        if targets is not None:
            final_loss = F.cross_entropy(final_logits.view(B*T, self.vocab_size), targets.view(B*T))
            aux["intermediate_loss"] = inter_num / inter_den.clamp(min=1e-6)
            aux["loss"] = final_loss + 0.25 * aux["intermediate_loss"]

        return final_logits, aux

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = SutraV051(vocab_size=50257, dim=64, ff_dim=128, max_steps=4, window=2, k_retrieval=4)
    print(f"v0.5.1 params: {m.count_params():,}")
    x = torch.randint(0, 50257, (2, 32))
    y = torch.randint(0, 50257, (2, 32))
    logits, aux = m(x, targets=y)
    print(f"Forward OK: loss={aux['loss'].item():.4f}, avg_steps={aux['avg_steps']:.1f}")
    aux["loss"].backward()
    print(f"Backward OK, no NaN: {not any(torch.isnan(p.grad).any() for p in m.parameters() if p.grad is not None)}")

    # Causality
    xa = torch.randint(0, 50257, (1, 16)); xb = xa.clone(); xb[0,10] = (xb[0,10]+1) % 50257
    with torch.no_grad(): la, _ = m(xa); lb, _ = m(xb)
    print(f"Causality: diff={((la[0,:10]-lb[0,:10]).abs().max()):.6f}")

    mf = SutraV051(vocab_size=50257, dim=768, ff_dim=1536, max_steps=6)
    print(f"Full: {mf.count_params():,} ({mf.count_params()/1e6:.1f}M)")
