"""Chrome: Peri-LN (Pre+Post LayerNorm) for Sutra v0.5.

Peri-LN (arXiv 2502.02732, ICML 2025) applies LayerNorm BEFORE and AFTER
each sublayer. Zero new parameters. Already deployed in OLMo2, Gemma2, Gemma3.
Fixes training instability at small scale where loss spikes are disproportionately destructive.

Test: compare standard Pre-LN (current Sutra) vs Peri-LN wrapper.

Usage: python code/chrome_peri_ln.py
"""

import sys, math, time, json
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sutra_v05_ssm import (SutraV05, N_STAGES, STAGE_GRAPH, top2_project,
                            StageBank, BayesianWrite, LocalRouter, Verifier,
                            StageTransitionKernel)

SEQ, BATCH, VOCAB = 64, 8, 50257
DIM, FF_DIM = 128, 256
TRAIN_STEPS = 300
EVAL_SEQS = 64

print("=" * 60)
print("CHROME: Peri-LN (Pre+Post LayerNorm) on Sutra v0.5")
print("=" * 60)

tokens = torch.load(REPO / "data" / "minipile_tokens.pt", weights_only=True)
N_SEQ = min(512, (tokens.numel() - 1) // (SEQ + 1))
data_x = torch.stack([tokens[i*(SEQ+1):i*(SEQ+1)+SEQ] for i in range(N_SEQ)])
data_y = torch.stack([tokens[i*(SEQ+1)+1:i*(SEQ+1)+SEQ+1] for i in range(N_SEQ)])
print(f"Data: {N_SEQ} sequences\n")


class SutraV05PeriLN(nn.Module):
    """v0.5 with Peri-LN: LayerNorm before AND after each stage operation.

    Current Sutra uses only a final LayerNorm. Peri-LN adds normalization
    around the stage bank, router, and writer — stabilizing residual variance.
    """
    def __init__(self):
        super().__init__()
        self.dim = DIM
        self.max_steps = 3
        self.emb = nn.Embedding(VOCAB, DIM)
        self.pos_emb = nn.Embedding(2048, DIM)
        self.init_mu = nn.Linear(DIM, DIM)
        self.init_lam = nn.Linear(DIM, DIM)
        self.transition = StageTransitionKernel(DIM, hidden=DIM*2)
        self.stage_bank = StageBank(DIM, FF_DIM)
        self.router = LocalRouter(DIM, window=2, k=4)
        self.writer = BayesianWrite(DIM)

        # Peri-LN: pre+post norms for each sublayer
        self.pre_bank_ln = nn.LayerNorm(DIM)
        self.post_bank_ln = nn.LayerNorm(DIM)
        self.pre_route_ln = nn.LayerNorm(DIM)
        self.post_route_ln = nn.LayerNorm(DIM)
        self.pre_write_ln = nn.LayerNorm(DIM)
        self.post_write_ln = nn.LayerNorm(DIM)

        self.ln = nn.LayerNorm(DIM)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=x.device)
        pi[:, :, 2] = 1.0

        for t in range(self.max_steps):
            K = self.transition(mu)
            pi_ev = torch.bmm(pi.view(B*T,1,N_STAGES), K.view(B*T,N_STAGES,N_STAGES)).view(B,T,N_STAGES)

            # Peri-LN around stage bank
            mu_normed = self.pre_bank_ln(mu)
            so, ev = self.stage_bank(mu_normed, pi)
            so = self.post_bank_ln(so)

            pi_new = pi_ev * F.softmax(ev / 0.5, dim=-1)
            pi_new = pi_new / pi_new.sum(-1, keepdim=True).clamp(min=1e-8)
            pi = top2_project(pi_new)

            # Peri-LN around router
            mu_normed = self.pre_route_ln(mu)
            messages = self.router(mu_normed) * pi[:,:,3:4]
            messages = self.post_route_ln(messages)

            # Peri-LN around writer
            mu_normed = self.pre_write_ln(mu)
            mu, lam = self.writer(mu_normed, lam, messages, pi[:,:,4:5])
            mu = self.post_write_ln(mu)

            mu = mu + so * 0.1

        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(DIM)
        return logits, {"compute_cost": torch.tensor(0.0), "avg_steps": self.max_steps}

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_eval(name, model):
    torch.manual_seed(42)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    t0 = time.time()
    log = []

    print(f"\n  {name}: {model.count_params():,} params")

    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, N_SEQ, (BATCH,))
        x, y = data_x[idx], data_y[idx]
        logits, _ = model(x)
        Tc = min(logits.size(1), y.size(1))
        ce = F.cross_entropy(logits[:,:Tc].reshape(-1, VOCAB), y[:,:Tc].reshape(-1))
        if torch.isnan(ce):
            opt.zero_grad()
            continue
        ce.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        bpt = ce.item() / math.log(2)
        log.append(bpt)
        if (step+1) % 50 == 0:
            avg = sum(log[-50:])/len(log[-50:])
            print(f"    Step {step+1}: BPT={bpt:.3f} avg50={avg:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits, _ = model(data_x[:EVAL_SEQS])
        Tc = min(logits.size(1), data_y[:EVAL_SEQS].size(1))
        test_bpt = F.cross_entropy(
            logits[:,:Tc].reshape(-1, VOCAB),
            data_y[:EVAL_SEQS,:Tc].reshape(-1)
        ).item() / math.log(2)

    elapsed = time.time() - t0
    return {"name": name, "test_bpt": round(test_bpt, 4),
            "time": round(elapsed, 1), "last50": round(sum(log[-50:])/len(log[-50:]), 4),
            "params": model.count_params()}


# Baseline: standard Pre-LN (Sutra v0.5)
torch.manual_seed(42)
baseline = SutraV05(vocab_size=VOCAB, dim=DIM, ff_dim=FF_DIM, max_steps=3, window=2, k_retrieval=4)
r_base = train_eval("Pre-LN Baseline", baseline)
print(f"  -> {r_base['test_bpt']:.4f} BPT\n")

# Peri-LN
torch.manual_seed(42)
peri = SutraV05PeriLN()
r_peri = train_eval("Peri-LN", peri)
print(f"  -> {r_peri['test_bpt']:.4f} BPT\n")

# Results
delta = r_base["test_bpt"] - r_peri["test_bpt"]
pct = delta / r_base["test_bpt"] * 100
print("=" * 60)
print(f"Baseline: {r_base['test_bpt']:.4f} BPT ({r_base['params']:,} params)")
print(f"Peri-LN:  {r_peri['test_bpt']:.4f} BPT ({r_peri['params']:,} params)")
print(f"Delta: {delta:+.4f} ({pct:+.1f}%)")
extra_params = r_peri["params"] - r_base["params"]
print(f"Extra params: {extra_params:,} ({extra_params/r_base['params']*100:.1f}% overhead)")
verdict = "PASS" if pct > 0.5 else ("MARGINAL" if pct > 0 else "KILL")
print(f"Verdict: {verdict}")

json.dump({"baseline": r_base, "peri_ln": r_peri, "delta_pct": round(pct, 2), "verdict": verdict},
          open(REPO / "results" / "chrome_peri_ln.json", "w"), indent=2)
