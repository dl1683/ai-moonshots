"""Launch v0.5.4: v0.5.3 + Peri-LN + Delayed Pheromone.

v0.5.3 improvements (retained):
  - 2-mode switching transition kernel (+4.1%)
  - BayesianWrite gain clamp(max=10.0) (eliminates NaN)
  - 8-slot scratchpad memory (+10.2% BPT)

v0.5.4 additions (Chrome-validated):
  - Peri-LN: LayerNorm before+after StageBank, Router, Writer (+5.9% BPT)
  - Delayed Pheromone: scalar position trace in global retrieval (best late-step 2.6x)
  - Training: Grokfast(alpha=0.95, lambda=2.0) applied in trainer (+13.6% combined)

Surprise Bank was KILLED (hurts every arm it touches in ablation).

Usage:
    python code/launch_v054.py --warmstart results/checkpoints_v053/step_5000.pt
"""

import sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sutra_v05_ssm import (SutraV05, N_STAGES, STAGE_GRAPH, top2_project,
                            StageBank, BayesianWrite, LocalRouter, Verifier)
from scratchpad import Scratchpad


class SwitchingKernel2(nn.Module):
    """2-mode content-dependent transition kernel (from v0.5.2)."""
    def __init__(self, dim, hidden=256, gate_hidden=64):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, N_STAGES * N_STAGES))
        self.mode_gate = nn.Sequential(nn.Linear(dim, gate_hidden), nn.SiLU(),
                                        nn.Linear(gate_hidden, 2))
        self.mode_bias = nn.Parameter(torch.zeros(2, N_STAGES, N_STAGES))

    def forward(self, h):
        B, N, D = h.shape
        base = self.base(h).view(B, N, N_STAGES, N_STAGES)
        mix = F.softmax(self.mode_gate(h.mean(dim=1)), dim=-1)
        mode = torch.einsum('bm,mij->bij', mix, self.mode_bias).unsqueeze(1)
        raw = base + mode
        mask = STAGE_GRAPH.to(h.device).unsqueeze(0).unsqueeze(0)
        return F.softmax(raw.masked_fill(mask == 0, float('-inf')), dim=-1)


class SutraV054(nn.Module):
    """v0.5.4: Stage-Superposition + Switching Kernel + Scratchpad + Peri-LN + Pheromone.

    Changes from v0.5.3:
    1. Peri-LN: LayerNorm before AND after StageBank, Router, Writer
       - Stabilizes recurrent composition (contraction analysis)
       - Shape-gain factorization (rate-distortion optimal)
       - Chrome-validated: +5.9% BPT

    2. Delayed Pheromone: scalar trace over positions in global retrieval
       - Stigmergic routing by public traces of usefulness
       - Activates only at recurrent step 3+ (delayed start principle)
       - Chrome-validated: best late-step value (2.6x baseline)
    """

    def __init__(self, vocab_size=50257, dim=768, ff_dim=1536,
                 max_steps=8, window=4, k_retrieval=8, n_scratch_slots=8,
                 pheromone_rho=0.90, pheromone_alpha=0.25):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.pheromone_rho = pheromone_rho
        self.pheromone_alpha = pheromone_alpha

        # Embeddings (tied with output)
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(2048, dim)

        # State initialization
        self.init_mu = nn.Linear(dim, dim)
        self.init_lam = nn.Linear(dim, dim)

        # Core modules (from v0.5.3)
        self.transition = SwitchingKernel2(dim)
        self.stage_bank = StageBank(dim, ff_dim)
        self.router = LocalRouter(dim, window=window, k=k_retrieval)
        self.writer = BayesianWrite(dim)
        self.verifier = Verifier(dim, vocab_size)
        self.scratchpad = Scratchpad(dim, n_slots=n_scratch_slots)

        # v0.5.4: Peri-LN (pre+post normalization for each recurrent operator)
        self.pre_bank_ln = nn.LayerNorm(dim)
        self.post_bank_ln = nn.LayerNorm(dim)
        self.pre_route_ln = nn.LayerNorm(dim)
        self.post_route_ln = nn.LayerNorm(dim)
        self.pre_write_ln = nn.LayerNorm(dim)
        self.post_write_ln = nn.LayerNorm(dim)

        # Output
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        B, T = x.shape
        device = x.device

        # Stage 1-2: Embed and initialize
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=device, dtype=h.dtype)
        pi[:, :, 2] = 1.0

        # Initialize scratchpad memory
        mem = self.scratchpad.init_memory(B, device)

        # v0.5.4: Pheromone trace (per-position scalar)
        pheromone = torch.zeros(B, T, device=device, dtype=h.dtype)
        mu_prev = mu.detach().clone()

        for t in range(self.max_steps):
            # Delayed start: novel mechanisms only active after step 2
            late_gate = 1.0 if t >= 3 else 0.0

            # Stage transitions (switching kernel)
            K = self.transition(mu)
            pi_evolved = torch.bmm(
                pi.view(B * T, 1, N_STAGES),
                K.view(B * T, N_STAGES, N_STAGES)
            ).view(B, T, N_STAGES)

            # v0.5.4: Peri-LN around stage bank
            mu_bank = self.pre_bank_ln(mu)
            stage_out, evidence = self.stage_bank(mu_bank, pi)
            stage_out = self.post_bank_ln(stage_out)

            # Evidence-modulated transition + Top2
            pi_new = pi_evolved * F.softmax(evidence / 0.5, dim=-1)
            pi_new = pi_new / pi_new.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            pi = top2_project(pi_new)

            # v0.5.4: Peri-LN around router
            mu_route = self.pre_route_ln(mu)
            messages = self.router(mu_route) * pi[:, :, 3:4]
            messages = self.post_route_ln(messages)

            # Scratchpad context injection (from v0.5.3)
            mem_ctx = self.scratchpad.read(mu, mem)
            messages = messages + 0.1 * mem_ctx * pi[:, :, 3:4]

            # v0.5.4: Pheromone bias on messages (delayed)
            if late_gate > 0:
                phero_gate = torch.sigmoid(pheromone).unsqueeze(-1)
                messages = messages + 0.05 * phero_gate * mem_ctx * pi[:, :, 3:4]

            # v0.5.4: Peri-LN around writer
            mu_write = self.pre_write_ln(mu)
            mu, lam = self.writer(mu_write, lam, messages, pi[:, :, 4:5])
            mu = self.post_write_ln(mu)

            # Residual from stage bank
            mu = mu + stage_out * 0.1

            # Update scratchpad (from v0.5.3)
            mem = self.scratchpad.write(mu.detach(), mem.detach(), pi[:, :, 4:5].detach())

            # v0.5.4: Update pheromone trace (delayed)
            with torch.no_grad():
                delta_mag = (mu - mu_prev).pow(2).mean(dim=-1).sqrt()
                deposit = pi[:, :, 4].detach() * torch.tanh(delta_mag)
                pheromone = self.pheromone_rho * pheromone + late_gate * deposit
            mu_prev = mu.detach().clone()

        # Final output
        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(self.dim)

        aux = {
            "compute_cost": torch.tensor(0.0, device=device),
            "avg_steps": self.max_steps,
        }
        return logits, aux

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def create_v054(dim=768, ff_dim=1536, max_steps=8, window=4, k_retrieval=8):
    """Create v0.5.4 model."""
    return SutraV054(vocab_size=50257, dim=dim, ff_dim=ff_dim,
                     max_steps=max_steps, window=window, k_retrieval=k_retrieval)


def warmstart_v054(checkpoint_path, **kwargs):
    """Warm-start v0.5.4 from v0.5.3 checkpoint."""
    model = create_v054(**kwargs)
    new_state = model.state_dict()

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    old_state = ckpt["model"] if "model" in ckpt else ckpt

    transferred = 0
    for name in list(new_state.keys()):
        if name in old_state and new_state[name].shape == old_state[name].shape:
            new_state[name] = old_state[name]
            transferred += 1

    model.load_state_dict(new_state)
    total = len(new_state)
    print(f"Warm-start v0.5.4: {transferred}/{total} params ({transferred/total*100:.0f}%)")
    fresh = total - transferred
    print(f"Fresh (Peri-LN norms): {fresh} params")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmstart", default=None)
    args = parser.parse_args()

    if args.warmstart:
        model = warmstart_v054(args.warmstart)
    else:
        model = create_v054()

    print(f"v0.5.4 params: {model.count_params():,} ({model.count_params()/1e6:.1f}M)")

    # Quick test
    x = torch.randint(0, 50257, (2, 32))
    logits, aux = model(x)
    print(f"Forward OK: {logits.shape}")

    # Causality
    xa = torch.randint(0, 50257, (1, 16))
    xb = xa.clone(); xb[0, 10] = (xb[0, 10] + 1) % 50257
    with torch.no_grad():
        la, _ = model(xa); lb, _ = model(xb)
    print(f"Causality: diff={((la[0,:10]-lb[0,:10]).abs().max()):.6f}")

    # Backward
    y = torch.randint(0, 50257, (2, 32))
    loss = F.cross_entropy(logits.reshape(-1, 50257), y.reshape(-1))
    loss.backward()
    print(f"Backward OK: loss={loss.item():.4f}")
    print(f"NaN: {any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)}")
