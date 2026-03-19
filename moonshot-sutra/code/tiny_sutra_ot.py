"""TinySutra: Core prototype for OT routing + Kalman states + monotonic advancement.

From Codex R2 challenge. Tests ONLY the core claim:
- Chunked OT routing (supply/demand with Sinkhorn)
- Uncertainty-weighted Kalman state updates
- Monotonic stage advancement

Test task: chunked source/query retrieval with scarce sources and distractors.
Baseline: same model with standard attention instead of Sinkhorn.
Success: higher accuracy AND lower source overuse.

Credit: Core architecture from Codex (GPT-5.2) R2 review.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn(scores, supply, demand, eps=0.3, iters=8):
    """Entropic optimal transport via Sinkhorn iterations."""
    logK = scores / eps
    u = torch.zeros_like(demand)
    v = torch.zeros_like(supply)
    for _ in range(iters):
        u = torch.log(demand + 1e-8) - torch.logsumexp(logK + v[:, None, :], dim=-1)
        v = torch.log(supply + 1e-8) - torch.logsumexp(logK.transpose(1, 2) + u[:, None, :], dim=-1)
    return torch.exp(logK + u[..., None] + v[:, None, :])


class TinySutra(nn.Module):
    """Minimal Sutra: OT routing + Kalman + monotonic advancement."""

    def __init__(self, vocab=256, d_model=64, d_route=16, chunk=4, steps=3, n_classes=10):
        super().__init__()
        self.chunk = chunk
        self.steps = steps
        self.emb = nn.Embedding(vocab, d_model)
        self.local = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.to_state = nn.Linear(d_model, 2 * d_model)
        self.to_route = nn.Linear(d_model, 2 * d_route)
        self.to_value = nn.Linear(d_model, d_model)
        self.to_obs = nn.Linear(d_model, 2 * d_model)
        self.to_supply = nn.Linear(d_model, 1)
        self.to_demand = nn.Linear(d_model, 1)
        self.to_advance = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, n_classes)

    def chunk_pool(self, x):
        B, T, D = x.shape
        M = T // self.chunk
        x = x[:, :M * self.chunk].view(B, M, self.chunk, D)
        return x.mean(dim=2)

    def forward(self, ids):
        h = self.emb(ids)
        h = self.chunk_pool(h)
        h = self.local(h.transpose(1, 2)).transpose(1, 2)

        mu, logvar = self.to_state(h).chunk(2, dim=-1)
        logvar = logvar.clamp(-4, 4)
        done = torch.zeros(mu.size(0), mu.size(1), device=mu.device)

        for _ in range(self.steps):
            unresolved = 1.0 - done

            s, d = self.to_route(mu).chunk(2, dim=-1)
            scores = torch.einsum("bid,bjd->bij", d, s) / math.sqrt(s.size(-1))

            supply = F.softplus(self.to_supply(mu).squeeze(-1)) * (0.5 + done) + 1e-4
            demand = F.softplus(self.to_demand(mu).squeeze(-1)) * unresolved + 1e-4

            total = torch.minimum(supply.sum(-1, keepdim=True), demand.sum(-1, keepdim=True))
            supply = total * supply / supply.sum(-1, keepdim=True)
            demand = total * demand / demand.sum(-1, keepdim=True)

            R = sinkhorn(scores, supply, demand)
            value = self.to_value(mu)
            evidence = R @ value

            obs_mu, obs_logvar = self.to_obs(evidence).chunk(2, dim=-1)
            obs_logvar = obs_logvar.clamp(-4, 4)

            p_state = torch.exp(-logvar)
            p_obs = torch.exp(-obs_logvar)
            mu = (p_state * mu + p_obs * obs_mu) / (p_state + p_obs + 1e-8)
            logvar = -torch.log(p_state + p_obs + 1e-8)

            advance = torch.sigmoid(self.to_advance(mu).squeeze(-1))
            done = done + (1.0 - done) * advance

        logits = self.head(mu.mean(dim=1))
        aux = {
            "done_mean": done.mean(),
            "var_mean": logvar.exp().mean(),
            "transport_entropy": -(R.clamp_min(1e-8) * R.clamp_min(1e-8).log()).sum((-1, -2)).mean(),
        }
        return logits, aux


class TinyAttentionBaseline(nn.Module):
    """Same architecture but with standard attention instead of OT routing."""

    def __init__(self, vocab=256, d_model=64, chunk=4, steps=3, n_classes=10):
        super().__init__()
        self.chunk = chunk
        self.steps = steps
        self.emb = nn.Embedding(vocab, d_model)
        self.local = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.update = nn.GRUCell(d_model, d_model)
        self.head = nn.Linear(d_model, n_classes)

    def chunk_pool(self, x):
        B, T, D = x.shape
        M = T // self.chunk
        return x[:, :M * self.chunk].view(B, M, self.chunk, D).mean(dim=2)

    def forward(self, ids):
        h = self.emb(ids)
        h = self.chunk_pool(h)
        h = self.local(h.transpose(1, 2)).transpose(1, 2)

        for _ in range(self.steps):
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            attn = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.size(-1)), dim=-1)
            msg = attn @ v
            B, M, D = h.shape
            h = self.update(msg.reshape(B * M, D), h.reshape(B * M, D)).reshape(B, M, D)

        return self.head(h.mean(dim=1)), {}


if __name__ == "__main__":
    # Quick verification
    model = TinySutra(vocab=256, d_model=64, d_route=16, chunk=4, steps=3, n_classes=10)
    baseline = TinyAttentionBaseline(vocab=256, d_model=64, chunk=4, steps=3, n_classes=10)

    x = torch.randint(0, 256, (4, 32))

    logits, aux = model(x)
    print(f"TinySutra: {logits.shape}, done={aux['done_mean']:.3f}, var={aux['var_mean']:.4f}")

    logits_b, _ = baseline(x)
    print(f"Baseline:  {logits_b.shape}")

    print(f"\nTinySutra params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Baseline params:  {sum(p.numel() for p in baseline.parameters()):,}")
    print(f"\nReady for retrieval task comparison.")
