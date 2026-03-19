"""TinyMSBP: Multi-Scale Belief Propagation prototype.

From Codex MSBP challenge. Tests: does BP-structured message passing
beat generic message passing at matched FLOPs?

Two scales: fine (patches) and coarse (groups of patches).
Gaussian BP: precision-weighted updates with damping.
Cross-scale: compress up, constrain down.

Credit: Prototype structure from Codex (GPT-5.2) MSBP review.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMSBP(nn.Module):
    def __init__(self, vocab=256, d=64, patch=4, rounds=3):
        super().__init__()
        self.patch, self.rounds = patch, rounds
        self.emb = nn.Embedding(vocab, d)
        self.init_belief = nn.Linear(d, 2 * d)   # eta0, log_lambda0
        self.msg = nn.Linear(d, 2 * d)            # diagonal GaBP surrogate
        self.readout = nn.Linear(d, vocab)

    def bp(self, mu, eta0, lam0, coarse=None):
        if coarse is not None:
            coarse = coarse.repeat_interleave(2, dim=1)[:, :mu.size(1)]
            eta0 = eta0 + lam0 * coarse
        for _ in range(self.rounds):
            left = F.pad(mu[:, :-1], (0, 0, 1, 0))      # causal neighbor only
            a, log_q = self.msg(left).chunk(2, dim=-1)
            q = F.softplus(log_q) + 1e-3
            lam = lam0 + a.square() / q
            eta = eta0 + a * left / q
            mu_new = eta / lam
            if (mu_new - mu).pow(2).mean() < 1e-4:
                break
            mu = 0.5 * mu + 0.5 * mu_new               # damping
        return mu, lam

    def forward(self, x):
        B, T = x.shape
        P, N = self.patch, (T + self.patch - 1) // self.patch
        x = F.pad(x, (0, N * P - T))
        h = self.emb(x).view(B, N, P, -1).mean(2)      # patch summaries

        eta0, log_lam0 = self.init_belief(h).chunk(2, dim=-1)
        lam0 = F.softplus(log_lam0) + 1e-3
        mu0 = eta0 / lam0

        mu_f, lam_f = self.bp(mu0, eta0, lam0)         # fine scale
        mu_c = F.avg_pool1d(mu_f.transpose(1, 2), 2, ceil_mode=True).transpose(1, 2)
        eta_c = F.avg_pool1d((lam_f * mu_f).transpose(1, 2), 2, ceil_mode=True).transpose(1, 2)
        lam_c = F.avg_pool1d(lam_f.transpose(1, 2), 2, ceil_mode=True).transpose(1, 2)
        mu_c, _ = self.bp(mu_c, eta_c, lam_c)          # coarse scale
        mu_f, _ = self.bp(mu_f, eta0, lam0, coarse=mu_c)  # downward constraint

        tok = mu_f.unsqueeze(2).expand(-1, -1, P, -1).reshape(B, N * P, -1)[:, :T]
        return self.readout(tok)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class GenericMPBaseline(nn.Module):
    """Generic message passing baseline (same as v0.4 but simpler)."""
    def __init__(self, vocab=256, d=64, patch=4, rounds=3):
        super().__init__()
        self.patch, self.rounds = patch, rounds
        self.emb = nn.Embedding(vocab, d)
        self.msg_net = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.update_net = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)
        self.readout = nn.Linear(d, vocab)

    def forward(self, x):
        B, T = x.shape
        P, N = self.patch, (T + self.patch - 1) // self.patch
        x = F.pad(x, (0, N * P - T))
        h = self.emb(x).view(B, N, P, -1).mean(2)

        for _ in range(self.rounds):
            left = F.pad(h[:, :-1], (0, 0, 1, 0))
            self_exp = h
            msgs = self.msg_net(torch.cat([self_exp, left], dim=-1))
            h = self.ln(h + self.update_net(torch.cat([h, msgs], dim=-1)))

        tok = h.unsqueeze(2).expand(-1, -1, P, -1).reshape(B, N * P, -1)[:, :T]
        return self.readout(tok)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    msbp = TinyMSBP(vocab=256, d=64, patch=4, rounds=3)
    baseline = GenericMPBaseline(vocab=256, d=64, patch=4, rounds=3)

    x = torch.randint(0, 256, (4, 64))
    out_bp = msbp(x)
    out_mp = baseline(x)
    print(f"MSBP: {out_bp.shape}, params={msbp.count_params():,}")
    print(f"MP:   {out_mp.shape}, params={baseline.count_params():,}")
    print("Both working. Ready for comparison test.")
