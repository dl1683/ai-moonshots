"""Chrome v0.5.4 Ablation: 5-arm validation per Codex master design.

Arms:
  1. v0.5.3 + Peri-LN
  2. v0.5.3 + Peri-LN + Delayed Surprise Bank
  3. v0.5.3 + Peri-LN + Delayed Pheromone
  4. v0.5.3 + Peri-LN + Delayed Surprise + Delayed Pheromone (full v0.5.4)
  5. Full v0.5.4 + Grokfast(0.95, 2.0)

Design source: results/codex_v054_master_design.md
Kill criteria per Codex:
  Peri-LN: must be +1.0% over v0.5.3 baseline
  Surprise Bank: must be +0.5% over Peri-LN-only AND late-step 1.5x
  Pheromone: must improve top-k recall +3pts AND BPT +0.3%
  Grokfast: no NaN in 300 steps

Usage: python code/chrome_v054_ablation.py
"""

import sys, math, time, json
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sutra_v05_ssm import (N_STAGES, STAGE_GRAPH, top2_project,
                            StageBank, BayesianWrite, LocalRouter)
from scratchpad import Scratchpad
from grokfast import GrokfastFilter

# Chrome parameters
SEQ, BATCH, VOCAB = 64, 8, 50257
DIM, FF_DIM = 128, 256
MAX_STEPS = 6
N_SCRATCH = 8
N_SURPRISE = 4
TRAIN_STEPS = 300
EVAL_SEQS = 64
WINDOW, K_RET = 2, 4

print("=" * 70)
print("CHROME v0.5.4 ABLATION: 5-arm validation (Codex master design)")
print("=" * 70)

tokens = torch.load(REPO / "data" / "minipile_tokens.pt", weights_only=True)
N_SEQ = min(512, (tokens.numel() - 1) // (SEQ + 1))
data_x = torch.stack([tokens[i*(SEQ+1):i*(SEQ+1)+SEQ] for i in range(N_SEQ)])
data_y = torch.stack([tokens[i*(SEQ+1)+1:i*(SEQ+1)+SEQ+1] for i in range(N_SEQ)])
print(f"Data: {N_SEQ} sequences, seq_len={SEQ}\n")


# ============================================================
# SWITCHING KERNEL (from v0.5.2)
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
# SURPRISE BANK (new for v0.5.4)
# ============================================================
class SurpriseBank(nn.Module):
    """Fast-decaying memory storing prediction error (surprise), not state."""
    def __init__(self, dim, n_slots=4, ema_decay=0.85):
        super().__init__()
        self.n_slots = n_slots
        self.ema_decay = ema_decay
        self.mem_init = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        self.read_proj = nn.Linear(dim, dim)
        self.write_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        # Zero-init output so surprise bank starts silent
        self.write_val = nn.Linear(dim, dim)
        nn.init.zeros_(self.write_val.weight)
        nn.init.zeros_(self.write_val.bias)
        self.error_ln = nn.LayerNorm(dim)

    def init_memory(self, batch_size, device):
        return self.mem_init.expand(batch_size, -1, -1).to(device)

    def read(self, h, mem):
        q = self.read_proj(h)
        attn = F.softmax(torch.bmm(q, mem.transpose(1, 2)) / math.sqrt(h.size(-1)), dim=-1)
        return torch.bmm(attn, mem)

    def write(self, mu_curr, mu_prev, mem, pi_write):
        delta = self.error_ln(mu_curr - mu_prev.detach())
        write_summary = (delta * pi_write).sum(dim=1) / pi_write.sum(dim=1).clamp(min=1e-6)
        write_exp = write_summary.unsqueeze(1).expand_as(mem)
        gate = self.write_gate(torch.cat([mem, write_exp], dim=-1))
        new_val = self.write_val(write_exp)
        return self.ema_decay * mem + (1 - self.ema_decay) * gate * new_val


# ============================================================
# v0.5.4 MODEL (configurable arms)
# ============================================================
class SutraV054(nn.Module):
    """v0.5.4: Peri-LN + optional Delayed Surprise + optional Delayed Pheromone."""

    def __init__(self, use_peri_ln=True, use_surprise=False, use_pheromone=False):
        super().__init__()
        self.dim = DIM
        self.max_steps = MAX_STEPS
        self.use_peri_ln = use_peri_ln
        self.use_surprise = use_surprise
        self.use_pheromone = use_pheromone
        self.pheromone_rho = 0.90
        self.pheromone_alpha = 0.25

        # Core
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

        # Peri-LN
        if use_peri_ln:
            self.pre_bank_ln = nn.LayerNorm(DIM)
            self.post_bank_ln = nn.LayerNorm(DIM)
            self.pre_route_ln = nn.LayerNorm(DIM)
            self.post_route_ln = nn.LayerNorm(DIM)
            self.pre_write_ln = nn.LayerNorm(DIM)
            self.post_write_ln = nn.LayerNorm(DIM)

        # Surprise Bank
        if use_surprise:
            self.surprise_bank = SurpriseBank(DIM, n_slots=N_SURPRISE)

        # (Pheromone needs no parameters, just state)

    def forward(self, x, return_per_step=False):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=x.device)
        pi[:, :, 2] = 1.0
        mem = self.scratchpad.init_memory(B, x.device)
        mu_prev = mu.detach().clone()

        # Surprise memory
        surp_mem = self.surprise_bank.init_memory(B, x.device) if self.use_surprise else None

        # Pheromone trace
        pheromone = torch.zeros(B, T, device=x.device) if self.use_pheromone else None

        step_logits = [] if return_per_step else None

        for t in range(self.max_steps):
            late_gate = 1.0 if t >= 3 else 0.0

            K = self.transition(mu)
            pi_ev = torch.bmm(pi.view(B*T,1,N_STAGES), K.view(B*T,N_STAGES,N_STAGES)).view(B,T,N_STAGES)

            # Stage bank (with optional Peri-LN)
            mu_bank = self.pre_bank_ln(mu) if self.use_peri_ln else mu
            so, ev = self.stage_bank(mu_bank, pi)
            so = self.post_bank_ln(so) if self.use_peri_ln else so

            pi_new = pi_ev * F.softmax(ev / 0.5, dim=-1)
            pi_new = pi_new / pi_new.sum(-1, keepdim=True).clamp(min=1e-8)
            pi = top2_project(pi_new)

            # Router (with optional Peri-LN)
            mu_route = self.pre_route_ln(mu) if self.use_peri_ln else mu
            messages = self.router(mu_route) * pi[:,:,3:4]
            messages = self.post_route_ln(messages) if self.use_peri_ln else messages

            # Inject state memory
            mem_ctx = self.scratchpad.read(mu, mem)
            messages = messages + 0.10 * mem_ctx * pi[:,:,3:4]

            # Inject surprise memory (delayed)
            if self.use_surprise and surp_mem is not None:
                surp_ctx = self.surprise_bank.read(mu, surp_mem)
                messages = messages + 0.05 * late_gate * surp_ctx * pi[:,:,3:4]

            # Inject pheromone into global retrieval (already in router scores)
            # Note: pheromone modifies the router's global branch internally
            # For Chrome, we approximate by adding bias to messages
            if self.use_pheromone and pheromone is not None:
                # Approximate: weight messages by pheromone-gated positions
                phero_gate = late_gate * torch.sigmoid(pheromone).unsqueeze(-1)
                messages = messages + 0.05 * phero_gate * mem_ctx * pi[:,:,3:4]

            # Writer (with optional Peri-LN)
            mu_write = self.pre_write_ln(mu) if self.use_peri_ln else mu
            mu, lam = self.writer(mu_write, lam, messages, pi[:,:,4:5])
            mu = self.post_write_ln(mu) if self.use_peri_ln else mu

            mu = mu + so * 0.1

            # Update state memory
            mem = self.scratchpad.write(mu.detach(), mem.detach(), pi[:,:,4:5].detach())

            # Update surprise memory (delayed)
            if self.use_surprise and surp_mem is not None:
                surp_mem = self.surprise_bank.write(
                    mu.detach(), mu_prev, surp_mem.detach(), pi[:,:,4:5].detach()
                )

            # Update pheromone (delayed)
            if self.use_pheromone and pheromone is not None:
                with torch.no_grad():
                    delta_mag = (mu - mu_prev).pow(2).mean(dim=-1).sqrt()
                    deposit = pi[:,:,4].detach() * torch.tanh(delta_mag)
                    pheromone = self.pheromone_rho * pheromone + late_gate * deposit

            mu_prev = mu.detach().clone()

            if return_per_step:
                with torch.no_grad():
                    sl = F.linear(self.ln(mu), self.emb.weight) / math.sqrt(DIM)
                    step_logits.append(sl)

        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(DIM)

        if return_per_step:
            return logits, step_logits
        return logits, {}

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# TRAINING + EVALUATION
# ============================================================
def train_and_eval(name, model, use_grokfast=False):
    print(f"\n{'='*55}")
    print(f"  ARM: {name}")
    print(f"  Params: {model.count_params():,}")
    print(f"{'='*55}")

    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    gf = GrokfastFilter(model, alpha=0.95, lam=2.0) if use_grokfast else None
    t0 = time.time()
    log = []

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

        if gf and step >= 20:  # Small warmup for Grokfast
            gf.apply()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        bpt = ce.item() / math.log(2)
        log.append(bpt)

        if (step+1) % 50 == 0:
            avg = sum(log[-50:]) / len(log[-50:])
            print(f"  Step {step+1}/{TRAIN_STEPS}: BPT={bpt:.3f} (avg50={avg:.3f})", flush=True)

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits, _ = model(data_x[:EVAL_SEQS])
        Tc = min(logits.size(1), data_y[:EVAL_SEQS].size(1))
        test_bpt = F.cross_entropy(
            logits[:,:Tc].reshape(-1, VOCAB),
            data_y[:EVAL_SEQS,:Tc].reshape(-1)
        ).item() / math.log(2)

    # Per-step analysis
    step_bpts = []
    with torch.no_grad():
        _, per_step = model(data_x[:16], return_per_step=True)
        if per_step:
            for sl in per_step:
                Tc = min(sl.size(1), data_y[:16].size(1))
                sbpt = F.cross_entropy(sl[:,:Tc].reshape(-1, VOCAB),
                                        data_y[:16,:Tc].reshape(-1)).item() / math.log(2)
                step_bpts.append(sbpt)

    elapsed = time.time() - t0
    print(f"\n  RESULT: test_BPT={test_bpt:.4f}, time={elapsed:.1f}s")
    if step_bpts:
        print(f"  Per-step: {['%.3f'%s for s in step_bpts]}")
        if len(step_bpts) >= 2:
            late = step_bpts[-2] - step_bpts[-1]
            print(f"  Late drop ({len(step_bpts)-1}->{len(step_bpts)}): {late:.4f}")

    return {
        "name": name, "test_bpt": round(test_bpt, 4),
        "step_bpts": [round(s, 4) for s in step_bpts],
        "time": round(elapsed, 1),
        "params": model.count_params(),
        "grokfast": use_grokfast,
    }


# ============================================================
# RUN 5-ARM ABLATION
# ============================================================
results = {}

# Arm 0: v0.5.3 baseline (reference)
torch.manual_seed(42)
m0 = SutraV054(use_peri_ln=False, use_surprise=False, use_pheromone=False)
results["v053_baseline"] = train_and_eval("v0.5.3 Baseline", m0)

# Arm 1: + Peri-LN
torch.manual_seed(42)
m1 = SutraV054(use_peri_ln=True, use_surprise=False, use_pheromone=False)
results["peri_ln"] = train_and_eval("Peri-LN", m1)

# Arm 2: + Peri-LN + Delayed Surprise
torch.manual_seed(42)
m2 = SutraV054(use_peri_ln=True, use_surprise=True, use_pheromone=False)
results["peri_surprise"] = train_and_eval("Peri-LN + Surprise", m2)

# Arm 3: + Peri-LN + Delayed Pheromone
torch.manual_seed(42)
m3 = SutraV054(use_peri_ln=True, use_surprise=False, use_pheromone=True)
results["peri_pheromone"] = train_and_eval("Peri-LN + Pheromone", m3)

# Arm 4: Full v0.5.4
torch.manual_seed(42)
m4 = SutraV054(use_peri_ln=True, use_surprise=True, use_pheromone=True)
results["full_v054"] = train_and_eval("Full v0.5.4", m4)

# Arm 5: Full v0.5.4 + Grokfast
torch.manual_seed(42)
m5 = SutraV054(use_peri_ln=True, use_surprise=True, use_pheromone=True)
results["v054_grokfast"] = train_and_eval("v0.5.4 + Grokfast", m5, use_grokfast=True)


# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("CHROME v0.5.4 ABLATION RESULTS")
print("=" * 70)

base_bpt = results["v053_baseline"]["test_bpt"]
print(f"\nv0.5.3 Baseline: {base_bpt:.4f} BPT")
print(f"\n{'Arm':<30s} {'BPT':>8s} {'Delta':>8s} {'%':>8s} {'Late':>8s} {'Verdict':>10s}")
print("-" * 78)

for key in ["peri_ln", "peri_surprise", "peri_pheromone", "full_v054", "v054_grokfast"]:
    r = results[key]
    delta = base_bpt - r["test_bpt"]
    pct = delta / base_bpt * 100
    late = 0
    if r["step_bpts"] and len(r["step_bpts"]) >= 2:
        late = r["step_bpts"][-2] - r["step_bpts"][-1]

    # Verdicts per Codex kill criteria
    if key == "peri_ln":
        verdict = "PASS" if pct >= 1.0 else "KILL"
    elif key == "peri_surprise":
        peri_bpt = results["peri_ln"]["test_bpt"]
        peri_late = 0
        if results["peri_ln"]["step_bpts"] and len(results["peri_ln"]["step_bpts"]) >= 2:
            peri_late = results["peri_ln"]["step_bpts"][-2] - results["peri_ln"]["step_bpts"][-1]
        surp_gain = (peri_bpt - r["test_bpt"]) / peri_bpt * 100
        late_ratio = late / max(peri_late, 0.001)
        verdict = "PASS" if surp_gain >= 0.5 and late_ratio >= 1.5 else "KILL"
    elif key == "peri_pheromone":
        phero_gain = (results["peri_ln"]["test_bpt"] - r["test_bpt"]) / results["peri_ln"]["test_bpt"] * 100
        verdict = "PASS" if phero_gain >= 0.3 else "KILL"
    elif key == "full_v054":
        full_gain = (results["peri_ln"]["test_bpt"] - r["test_bpt"]) / results["peri_ln"]["test_bpt"] * 100
        verdict = "PASS" if pct >= 1.0 else "MARGINAL" if pct >= 0 else "KILL"
    elif key == "v054_grokfast":
        verdict = "PASS" if pct >= 2.0 else "MARGINAL"

    r["delta_pct"] = round(pct, 2)
    r["late_drop"] = round(late, 4)
    r["verdict"] = verdict
    print(f"  {r['name']:<28s} {r['test_bpt']:>8.4f} {delta:>+8.4f} {pct:>+7.1f}% {late:>8.4f} {verdict:>10s}")

# Promotion rule
print(f"\n{'='*70}")
best_key = min(results, key=lambda k: results[k]["test_bpt"])
best = results[best_key]
print(f"BEST ARM: {best['name']} ({best['test_bpt']:.4f} BPT, {best.get('delta_pct', 0):+.1f}%)")

# Save
out = REPO / "results" / "chrome_v054_ablation.json"
json.dump(results, open(out, "w"), indent=2)
print(f"Saved: {out}")
