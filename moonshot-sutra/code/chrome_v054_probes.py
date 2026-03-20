"""Chrome v0.5.4 Probes: Error Scratchpad, Pheromone Router, Depth-Drop Bootstrap.

Three novel mechanisms proposed by Codex deep analysis. Each tested at dim=128
on CPU against v0.5.3 baseline. The design principle: at 67M scale, Sutra accepts
simple shared state and coarse bias, rejects high-entropy learned control.

Kill criteria (per Codex):
  Error Scratchpad: <2% BPT gain OR <10% late-step improvement -> kill
  Pheromone Router: top-k recall not +5pts AND BPT <1% -> kill
  Depth-Drop Bootstrap: early-late gap not reduced >=25% by step 300 -> kill

Usage: python code/chrome_v054_probes.py
"""

import sys, math, time, json, copy
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sutra_v05_ssm import (SutraV05, N_STAGES, STAGE_GRAPH, top2_project,
                            StageBank, BayesianWrite, LocalRouter, Verifier)
from scratchpad import Scratchpad

# Chrome parameters
SEQ, BATCH, VOCAB = 64, 8, 50257
DIM, FF_DIM = 128, 256
MAX_STEPS = 6
N_SCRATCH = 8
TRAIN_STEPS = 300
EVAL_SEQS = 64
WINDOW, K_RET = 2, 4

print("=" * 70)
print("CHROME v0.5.4: Error Scratchpad | Pheromone Router | Depth-Drop")
print("=" * 70)

# Load data
tokens = torch.load(REPO / "data" / "minipile_tokens.pt", weights_only=True)
N_SEQ = min(512, (tokens.numel() - 1) // (SEQ + 1))
data_x = torch.stack([tokens[i*(SEQ+1):i*(SEQ+1)+SEQ] for i in range(N_SEQ)])
data_y = torch.stack([tokens[i*(SEQ+1)+1:i*(SEQ+1)+SEQ+1] for i in range(N_SEQ)])
print(f"Data: {N_SEQ} sequences, seq_len={SEQ}\n")


# ============================================================
# SWITCHING KERNEL (from v0.5.2, shared by all v0.5.3+ models)
# ============================================================
class SwitchingKernel2(nn.Module):
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


# ============================================================
# BASELINE: v0.5.3 (switching kernel + scratchpad)
# ============================================================
class SutraV053Baseline(nn.Module):
    """v0.5.3 baseline for fair comparison."""
    def __init__(self):
        super().__init__()
        self.dim = DIM
        self.max_steps = MAX_STEPS
        self.emb = nn.Embedding(VOCAB, DIM)
        self.pos_emb = nn.Embedding(2048, DIM)
        self.init_mu = nn.Linear(DIM, DIM)
        self.init_lam = nn.Linear(DIM, DIM)
        self.transition = SwitchingKernel2(DIM, hidden=DIM*2, gate_hidden=32)
        self.stage_bank = StageBank(DIM, FF_DIM)
        self.router = LocalRouter(DIM, window=WINDOW, k=K_RET)
        self.writer = BayesianWrite(DIM)
        self.scratchpad = Scratchpad(DIM, n_slots=N_SCRATCH)
        self.ln = nn.LayerNorm(DIM)

    def forward(self, x, return_per_step=False):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=x.device)
        pi[:, :, 2] = 1.0
        mem = self.scratchpad.init_memory(B, x.device)

        step_losses = [] if return_per_step else None

        for t in range(self.max_steps):
            K = self.transition(mu)
            pi_ev = torch.bmm(pi.view(B*T,1,N_STAGES), K.view(B*T,N_STAGES,N_STAGES)).view(B,T,N_STAGES)
            so, ev = self.stage_bank(mu, pi)
            pi_new = pi_ev * F.softmax(ev / 0.5, dim=-1)
            pi_new = pi_new / pi_new.sum(-1, keepdim=True).clamp(min=1e-8)
            pi = top2_project(pi_new)
            messages = self.router(mu) * pi[:,:,3:4]
            mem_ctx = self.scratchpad.read(mu, mem)
            messages = messages + 0.1 * mem_ctx * pi[:,:,3:4]
            mu, lam = self.writer(mu, lam, messages, pi[:,:,4:5])
            mu = mu + so * 0.1
            mem = self.scratchpad.write(mu.detach(), mem.detach(), pi[:,:,4:5].detach())

            if return_per_step:
                with torch.no_grad():
                    logits_t = F.linear(self.ln(mu), self.emb.weight) / math.sqrt(DIM)
                    step_losses.append(logits_t)

        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(DIM)

        if return_per_step:
            return logits, step_losses
        return logits, {}


# ============================================================
# PROBE 1: ERROR SCRATCHPAD
# Write prediction error (delta) to scratchpad, not raw state.
# ============================================================
class ErrorScratchpad(nn.Module):
    """Scratchpad that stores prediction error, not raw state."""
    def __init__(self, dim, n_slots=8, ema_decay=0.9):
        super().__init__()
        self.n_slots = n_slots
        self.ema_decay = ema_decay
        self.mem_init = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        self.read_proj = nn.Linear(dim, dim)
        self.write_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.write_val = nn.Linear(dim, dim)
        self.error_ln = nn.LayerNorm(dim)

    def init_memory(self, batch_size, device):
        return self.mem_init.expand(batch_size, -1, -1).to(device)

    def read(self, h, mem):
        q = self.read_proj(h)
        attn = F.softmax(torch.bmm(q, mem.transpose(1, 2)) / math.sqrt(h.size(-1)), dim=-1)
        return torch.bmm(attn, mem)

    def write_error(self, mu_curr, mu_prev, mem, pi_write):
        """Write prediction ERROR, not raw state."""
        # delta = LN(mu_t - sg(mu_{t-1}))
        delta = self.error_ln(mu_curr - mu_prev.detach())
        write_summary = (delta * pi_write).sum(dim=1) / pi_write.sum(dim=1).clamp(min=1e-6)
        write_exp = write_summary.unsqueeze(1).expand_as(mem)
        gate = self.write_gate(torch.cat([mem, write_exp], dim=-1))
        new_val = self.write_val(write_exp)
        return self.ema_decay * mem + (1 - self.ema_decay) * gate * new_val


class SutraErrorScratchpad(nn.Module):
    """v0.5.3 + Error Scratchpad (writes delta, not raw state)."""
    def __init__(self):
        super().__init__()
        self.dim = DIM
        self.max_steps = MAX_STEPS
        self.emb = nn.Embedding(VOCAB, DIM)
        self.pos_emb = nn.Embedding(2048, DIM)
        self.init_mu = nn.Linear(DIM, DIM)
        self.init_lam = nn.Linear(DIM, DIM)
        self.transition = SwitchingKernel2(DIM, hidden=DIM*2, gate_hidden=32)
        self.stage_bank = StageBank(DIM, FF_DIM)
        self.router = LocalRouter(DIM, window=WINDOW, k=K_RET)
        self.writer = BayesianWrite(DIM)
        self.error_scratch = ErrorScratchpad(DIM, n_slots=N_SCRATCH)
        self.ln = nn.LayerNorm(DIM)

    def forward(self, x, return_per_step=False):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=x.device)
        pi[:, :, 2] = 1.0
        mem = self.error_scratch.init_memory(B, x.device)
        mu_prev = mu.detach().clone()

        step_losses = [] if return_per_step else None

        for t in range(self.max_steps):
            K = self.transition(mu)
            pi_ev = torch.bmm(pi.view(B*T,1,N_STAGES), K.view(B*T,N_STAGES,N_STAGES)).view(B,T,N_STAGES)
            so, ev = self.stage_bank(mu, pi)
            pi_new = pi_ev * F.softmax(ev / 0.5, dim=-1)
            pi_new = pi_new / pi_new.sum(-1, keepdim=True).clamp(min=1e-8)
            pi = top2_project(pi_new)
            messages = self.router(mu) * pi[:,:,3:4]
            # Read from error memory (same interface as regular scratchpad)
            mem_ctx = self.error_scratch.read(mu, mem)
            messages = messages + 0.1 * mem_ctx * pi[:,:,3:4]
            mu, lam = self.writer(mu, lam, messages, pi[:,:,4:5])
            mu = mu + so * 0.1
            # KEY DIFFERENCE: write error signal, not raw state
            mem = self.error_scratch.write_error(
                mu.detach(), mu_prev, mem.detach(), pi[:,:,4:5].detach()
            )
            mu_prev = mu.detach().clone()

            if return_per_step:
                with torch.no_grad():
                    logits_t = F.linear(self.ln(mu), self.emb.weight) / math.sqrt(DIM)
                    step_losses.append(logits_t)

        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(DIM)
        if return_per_step:
            return logits, step_losses
        return logits, {}


# ============================================================
# PROBE 2: PHEROMONE ROUTER
# Decaying scalar trace over past positions, inspired by ant colonies.
# ============================================================
class PheromoneRouter(nn.Module):
    """Router with pheromone-style decaying traces."""
    def __init__(self, dim, window=4, k=8, rho=0.9, alpha=0.3):
        super().__init__()
        self.window = window
        self.k = k
        self.rho = rho  # decay rate
        self.alpha = alpha  # pheromone weight in retrieval
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.msg_net = nn.Sequential(nn.Linear(dim*2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.out_proj = nn.Linear(dim*2, dim)

    def forward(self, mu, pheromone=None):
        """Causal routing with pheromone traces.

        pheromone: (B, T) scalar trace of past usefulness per position.
        """
        B, N, D = mu.shape

        # Local window (same as base)
        padded = F.pad(mu, (0, 0, self.window, 0))
        neighbors = torch.stack([
            padded[:, self.window-w:self.window-w+N, :]
            for w in range(1, self.window+1)
        ], dim=2)
        self_exp = mu.unsqueeze(2).expand_as(neighbors)
        local_msgs = self.msg_net(torch.cat([self_exp, neighbors], dim=-1)).mean(dim=2)

        # Global: sparse top-k with pheromone bias
        q = self.q_proj(mu)
        k = self.k_proj(mu)
        v = self.v_proj(mu)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)

        # Add pheromone bias: positions with higher trace get boosted
        if pheromone is not None:
            scores = scores + self.alpha * pheromone.unsqueeze(1)  # (B, 1, T) broadcast

        causal_mask = torch.triu(torch.ones(N, N, device=mu.device) * float("-inf"), diagonal=1)
        scores = scores + causal_mask

        if N > self.k:
            topk_vals, topk_idx = scores.topk(self.k, dim=-1)
            sparse = torch.full_like(scores, float("-inf"))
            sparse.scatter_(2, topk_idx, topk_vals)
            attn = F.softmax(sparse, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)
        global_msgs = torch.bmm(attn, v)

        return self.out_proj(torch.cat([local_msgs, global_msgs], dim=-1))


class SutraPheromoneRouter(nn.Module):
    """v0.5.3 + Pheromone Router (decaying trace over useful positions)."""
    def __init__(self):
        super().__init__()
        self.dim = DIM
        self.max_steps = MAX_STEPS
        self.rho = 0.9  # pheromone decay
        self.emb = nn.Embedding(VOCAB, DIM)
        self.pos_emb = nn.Embedding(2048, DIM)
        self.init_mu = nn.Linear(DIM, DIM)
        self.init_lam = nn.Linear(DIM, DIM)
        self.transition = SwitchingKernel2(DIM, hidden=DIM*2, gate_hidden=32)
        self.stage_bank = StageBank(DIM, FF_DIM)
        self.router = PheromoneRouter(DIM, window=WINDOW, k=K_RET)
        self.writer = BayesianWrite(DIM)
        self.scratchpad = Scratchpad(DIM, n_slots=N_SCRATCH)
        self.ln = nn.LayerNorm(DIM)

    def forward(self, x, return_per_step=False):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=x.device)
        pi[:, :, 2] = 1.0
        mem = self.scratchpad.init_memory(B, x.device)

        # Initialize pheromone trace (zero = no history)
        pheromone = torch.zeros(B, T, device=x.device)

        step_losses = [] if return_per_step else None

        for t in range(self.max_steps):
            K = self.transition(mu)
            pi_ev = torch.bmm(pi.view(B*T,1,N_STAGES), K.view(B*T,N_STAGES,N_STAGES)).view(B,T,N_STAGES)
            so, ev = self.stage_bank(mu, pi)
            pi_new = pi_ev * F.softmax(ev / 0.5, dim=-1)
            pi_new = pi_new / pi_new.sum(-1, keepdim=True).clamp(min=1e-8)
            pi = top2_project(pi_new)

            # Route with pheromone traces
            messages = self.router(mu, pheromone=pheromone) * pi[:,:,3:4]
            mem_ctx = self.scratchpad.read(mu, mem)
            messages = messages + 0.1 * mem_ctx * pi[:,:,3:4]
            mu_pre = mu.detach().clone()
            mu, lam = self.writer(mu, lam, messages, pi[:,:,4:5])
            mu = mu + so * 0.1

            # Update pheromone: deposit = write stage probability * recurrent delta magnitude
            # High write-stage occupancy + large state change = useful position
            with torch.no_grad():
                delta_mag = (mu - mu_pre).norm(dim=-1)  # (B, T)
                deposit = pi[:,:,4].detach() * delta_mag  # write-stage gated
                pheromone = self.rho * pheromone + deposit

            mem = self.scratchpad.write(mu.detach(), mem.detach(), pi[:,:,4:5].detach())

            if return_per_step:
                with torch.no_grad():
                    logits_t = F.linear(self.ln(mu), self.emb.weight) / math.sqrt(DIM)
                    step_losses.append(logits_t)

        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(DIM)
        if return_per_step:
            return logits, step_losses
        return logits, {}


# ============================================================
# PROBE 3: DEPTH-DROP BOOTSTRAP
# Random recurrence truncation + KL to full-depth teacher.
# ============================================================
class SutraDepthDrop(nn.Module):
    """v0.5.3 + Depth-Drop Bootstrap.

    During training: randomly truncate to k steps, KL to full-depth teacher.
    This makes every recurrence step a viable stopping point.
    """
    def __init__(self):
        super().__init__()
        self.dim = DIM
        self.max_steps = MAX_STEPS
        self.emb = nn.Embedding(VOCAB, DIM)
        self.pos_emb = nn.Embedding(2048, DIM)
        self.init_mu = nn.Linear(DIM, DIM)
        self.init_lam = nn.Linear(DIM, DIM)
        self.transition = SwitchingKernel2(DIM, hidden=DIM*2, gate_hidden=32)
        self.stage_bank = StageBank(DIM, FF_DIM)
        self.router = LocalRouter(DIM, window=WINDOW, k=K_RET)
        self.writer = BayesianWrite(DIM)
        self.scratchpad = Scratchpad(DIM, n_slots=N_SCRATCH)
        self.ln = nn.LayerNorm(DIM)

    def _run_steps(self, x, n_steps):
        """Run exactly n_steps of recurrence."""
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=x.device)
        pi[:, :, 2] = 1.0
        mem = self.scratchpad.init_memory(B, x.device)

        for t in range(n_steps):
            K = self.transition(mu)
            pi_ev = torch.bmm(pi.view(B*T,1,N_STAGES), K.view(B*T,N_STAGES,N_STAGES)).view(B,T,N_STAGES)
            so, ev = self.stage_bank(mu, pi)
            pi_new = pi_ev * F.softmax(ev / 0.5, dim=-1)
            pi_new = pi_new / pi_new.sum(-1, keepdim=True).clamp(min=1e-8)
            pi = top2_project(pi_new)
            messages = self.router(mu) * pi[:,:,3:4]
            mem_ctx = self.scratchpad.read(mu, mem)
            messages = messages + 0.1 * mem_ctx * pi[:,:,3:4]
            mu, lam = self.writer(mu, lam, messages, pi[:,:,4:5])
            mu = mu + so * 0.1
            mem = self.scratchpad.write(mu.detach(), mem.detach(), pi[:,:,4:5].detach())

        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(DIM)
        return logits

    def forward(self, x, training=True, return_per_step=False):
        if return_per_step:
            # For evaluation: run all steps and collect per-step logits
            step_logits = []
            for k in range(1, self.max_steps + 1):
                with torch.no_grad():
                    step_logits.append(self._run_steps(x, k))
            return self._run_steps(x, self.max_steps), step_logits

        if training:
            # Full-depth teacher (stop gradient)
            with torch.no_grad():
                teacher_logits = self._run_steps(x, self.max_steps)

            # Random truncated student
            k = torch.randint(2, self.max_steps + 1, (1,)).item()
            student_logits = self._run_steps(x, k)

            return student_logits, {"teacher_logits": teacher_logits, "depth": k}
        else:
            logits = self._run_steps(x, self.max_steps)
            return logits, {}


# ============================================================
# TRAINING + EVALUATION LOOP
# ============================================================
def train_and_eval(name, model, train_steps=TRAIN_STEPS, depth_drop=False):
    """Train a model and return BPT + per-step analysis."""
    print(f"\n{'='*50}")
    print(f"  PROBE: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*50}")

    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    t0 = time.time()
    log = []

    for step in range(train_steps):
        idx = torch.randint(0, N_SEQ, (BATCH,))
        x, y = data_x[idx], data_y[idx]

        if depth_drop:
            logits, aux = model(x, training=True)
            Tc = min(logits.size(1), y.size(1))
            ce = F.cross_entropy(logits[:,:Tc].reshape(-1, VOCAB), y[:,:Tc].reshape(-1))

            # KL to full-depth teacher
            teacher_logits = aux["teacher_logits"]
            T_kd = 2.0
            kl = F.kl_div(
                F.log_softmax(logits[:,:Tc] / T_kd, dim=-1),
                F.softmax(teacher_logits[:,:Tc] / T_kd, dim=-1),
                reduction="batchmean"
            ) * T_kd * T_kd
            loss = ce + 0.1 * kl
        else:
            logits, _ = model(x)
            Tc = min(logits.size(1), y.size(1))
            ce = F.cross_entropy(logits[:,:Tc].reshape(-1, VOCAB), y[:,:Tc].reshape(-1))
            loss = ce

        if torch.isnan(loss):
            print(f"  NaN at step {step+1}, skipping")
            opt.zero_grad()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        bpt = ce.item() / math.log(2)
        log.append(bpt)

        if (step+1) % 50 == 0:
            avg = sum(log[-50:]) / len(log[-50:])
            print(f"  Step {step+1}/{train_steps}: BPT={bpt:.3f} (avg50={avg:.3f})", flush=True)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits, _ = model(data_x[:EVAL_SEQS], training=False) if depth_drop else model(data_x[:EVAL_SEQS])
        Tc = min(logits.size(1), data_y[:EVAL_SEQS].size(1))
        test_bpt = F.cross_entropy(
            logits[:,:Tc].reshape(-1, VOCAB),
            data_y[:EVAL_SEQS,:Tc].reshape(-1)
        ).item() / math.log(2)

    # Per-step analysis: how much does each step help?
    step_bpts = []
    with torch.no_grad():
        if depth_drop:
            _, per_step = model(data_x[:16], training=False, return_per_step=True)
        else:
            _, per_step = model(data_x[:16], return_per_step=True)

        if per_step:
            for sl in per_step:
                Tc = min(sl.size(1), data_y[:16].size(1))
                sbpt = F.cross_entropy(sl[:,:Tc].reshape(-1, VOCAB),
                                        data_y[:16,:Tc].reshape(-1)).item() / math.log(2)
                step_bpts.append(sbpt)

    elapsed = time.time() - t0
    print(f"\n  RESULT: test_BPT={test_bpt:.3f}, time={elapsed:.1f}s")
    if step_bpts:
        print(f"  Per-step BPT: {['%.3f'%s for s in step_bpts]}")
        if len(step_bpts) >= 4:
            early_drop = step_bpts[0] - step_bpts[1]
            late_drop = step_bpts[-2] - step_bpts[-1]
            print(f"  Early drop (1->2): {early_drop:.4f}")
            print(f"  Late drop ({len(step_bpts)-1}->{len(step_bpts)}): {late_drop:.4f}")

    return {
        "name": name,
        "test_bpt": round(test_bpt, 4),
        "step_bpts": [round(s, 4) for s in step_bpts],
        "train_log": [round(b, 4) for b in log[-50:]],
        "time": round(elapsed, 1),
        "params": sum(p.numel() for p in model.parameters()),
    }


# ============================================================
# RUN ALL PROBES
# ============================================================
results = {}

# 1. Baseline: v0.5.3
torch.manual_seed(42)
baseline = SutraV053Baseline()
results["v053_baseline"] = train_and_eval("v0.5.3 Baseline", baseline)

# 2. Error Scratchpad
torch.manual_seed(42)
error_scratch = SutraErrorScratchpad()
results["error_scratchpad"] = train_and_eval("Error Scratchpad", error_scratch)

# 3. Pheromone Router
torch.manual_seed(42)
pheromone = SutraPheromoneRouter()
results["pheromone_router"] = train_and_eval("Pheromone Router", pheromone)

# 4. Depth-Drop Bootstrap
torch.manual_seed(42)
depth_drop = SutraDepthDrop()
results["depth_drop"] = train_and_eval("Depth-Drop Bootstrap", depth_drop, depth_drop=True)


# ============================================================
# ANALYSIS + VERDICT
# ============================================================
print("\n" + "=" * 70)
print("CHROME v0.5.4 RESULTS")
print("=" * 70)

base_bpt = results["v053_baseline"]["test_bpt"]
print(f"\nBaseline (v0.5.3): {base_bpt:.4f} BPT")
print(f"{'Probe':<25s} {'BPT':>8s} {'Delta':>8s} {'%':>8s} {'Verdict':>12s}")
print("-" * 65)

for key in ["error_scratchpad", "pheromone_router", "depth_drop"]:
    r = results[key]
    delta = base_bpt - r["test_bpt"]
    pct = delta / base_bpt * 100
    # Apply kill criteria
    if key == "error_scratchpad":
        verdict = "PASS" if pct >= 2.0 else "KILL"
    elif key == "pheromone_router":
        verdict = "PASS" if pct >= 1.0 else "KILL"
    elif key == "depth_drop":
        # Check early-late gap reduction
        base_steps = results["v053_baseline"].get("step_bpts", [])
        probe_steps = r.get("step_bpts", [])
        if base_steps and probe_steps and len(base_steps) >= 2 and len(probe_steps) >= 2:
            base_gap = base_steps[1] - base_steps[-1]
            probe_gap = probe_steps[1] - probe_steps[-1]
            gap_reduction = (base_gap - probe_gap) / max(abs(base_gap), 1e-6) * 100
            verdict = "PASS" if gap_reduction >= 25 else "KILL"
        else:
            verdict = "INCONCLUSIVE"

    results[key]["delta_bpt"] = round(delta, 4)
    results[key]["delta_pct"] = round(pct, 2)
    results[key]["verdict"] = verdict
    print(f"  {r['name']:<23s} {r['test_bpt']:>8.4f} {delta:>+8.4f} {pct:>+7.1f}% {verdict:>12s}")

# Step analysis
print(f"\nPer-step BPT comparison:")
for key in ["v053_baseline", "error_scratchpad", "pheromone_router", "depth_drop"]:
    r = results[key]
    if r["step_bpts"]:
        print(f"  {r['name']:<25s}: {['%.3f'%s for s in r['step_bpts']]}")

# Save
out_path = REPO / "results" / "chrome_v054_probes.json"
json.dump(results, open(out_path, "w"), indent=2)
print(f"\nSaved to {out_path}")

# Summary for Codex
winners = [k for k in ["error_scratchpad", "pheromone_router", "depth_drop"]
           if results[k].get("verdict") == "PASS"]
if winners:
    print(f"\nWINNERS: {', '.join(results[w]['name'] for w in winners)}")
    print("Next: Combine winners into v0.5.4 and validate at dim=128")
else:
    print("\nAll probes killed. Need different mechanisms or scale.")
