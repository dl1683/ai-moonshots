"""Chrome: Grokfast gradient filter for Sutra v0.5.3.

Grokfast (arXiv 2405.20233) amplifies slow gradient modes via EMA filtering.
Reports 50x grokking acceleration. We test it on Sutra at dim=128.

The filter: after loss.backward(), before opt.step():
  grads_ema[p] = alpha * grads_ema[p] + (1-alpha) * p.grad
  p.grad += lam * grads_ema[p]

Alpha=0.98 (EMA decay), Lambda=5.0 (amplification factor).

Usage: python code/chrome_grokfast.py
"""

import sys, math, time, json
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sutra_v05_ssm import SutraV05

SEQ, BATCH, VOCAB = 64, 8, 50257
DIM, FF_DIM = 128, 256
TRAIN_STEPS = 300
EVAL_SEQS = 64

print("=" * 60)
print("CHROME: Grokfast Gradient Filter on Sutra v0.5")
print("=" * 60)

tokens = torch.load(REPO / "data" / "minipile_tokens.pt", weights_only=True)
N_SEQ = min(512, (tokens.numel() - 1) // (SEQ + 1))
data_x = torch.stack([tokens[i*(SEQ+1):i*(SEQ+1)+SEQ] for i in range(N_SEQ)])
data_y = torch.stack([tokens[i*(SEQ+1)+1:i*(SEQ+1)+SEQ+1] for i in range(N_SEQ)])
print(f"Data: {N_SEQ} sequences\n")


def grokfast_filter(model, grads_ema, alpha=0.98, lam=5.0):
    """Grokfast EMA gradient filter. Amplifies slow (generalization) modes."""
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name not in grads_ema:
            grads_ema[name] = torch.zeros_like(p.grad)
        grads_ema[name].mul_(alpha).add_(p.grad, alpha=1 - alpha)
        p.grad.add_(grads_ema[name], alpha=lam)


def train_and_eval(name, use_grokfast=False, alpha=0.98, lam=5.0):
    torch.manual_seed(42)
    model = SutraV05(vocab_size=VOCAB, dim=DIM, ff_dim=FF_DIM,
                     max_steps=3, window=2, k_retrieval=4)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    grads_ema = {}
    t0 = time.time()
    log = []

    print(f"  {name}: params={sum(p.numel() for p in model.parameters()):,}")

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

        if use_grokfast:
            grokfast_filter(model, grads_ema, alpha=alpha, lam=lam)

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        bpt = ce.item() / math.log(2)
        log.append(bpt)

        if (step+1) % 50 == 0:
            avg = sum(log[-50:]) / len(log[-50:])
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
    return {
        "name": name, "test_bpt": round(test_bpt, 4),
        "time": round(elapsed, 1),
        "last_50_avg": round(sum(log[-50:])/len(log[-50:]), 4),
        "grokfast": use_grokfast,
        "alpha": alpha if use_grokfast else None,
        "lam": lam if use_grokfast else None,
    }


results = {}

# Baseline
r = train_and_eval("Baseline (no Grokfast)")
results["baseline"] = r
print(f"  -> Test BPT: {r['test_bpt']:.4f}\n")

# Grokfast with different configs
for alpha, lam in [(0.98, 5.0), (0.95, 2.0), (0.99, 10.0)]:
    name = f"Grokfast(a={alpha},l={lam})"
    r = train_and_eval(name, use_grokfast=True, alpha=alpha, lam=lam)
    results[name] = r
    print(f"  -> Test BPT: {r['test_bpt']:.4f}\n")

# Summary
print("\n" + "=" * 60)
print("GROKFAST RESULTS")
print("=" * 60)
base = results["baseline"]["test_bpt"]
print(f"\nBaseline: {base:.4f} BPT")
print(f"{'Config':<30s} {'BPT':>8s} {'Delta':>8s} {'%':>8s}")
print("-" * 58)
for k, v in results.items():
    if k == "baseline":
        continue
    delta = base - v["test_bpt"]
    pct = delta / base * 100
    print(f"  {v['name']:<28s} {v['test_bpt']:>8.4f} {delta:>+8.4f} {pct:>+7.1f}%")

# Save
out = REPO / "results" / "chrome_grokfast.json"
json.dump(results, open(out, "w"), indent=2)
print(f"\nSaved: {out}")

best_key = min(results, key=lambda k: results[k]["test_bpt"])
if best_key != "baseline":
    best = results[best_key]
    gain = (base - best["test_bpt"]) / base * 100
    print(f"\nBEST: {best['name']} ({gain:+.1f}%)")
    print(f"Recommendation: add Grokfast(alpha={best['alpha']}, lam={best['lam']}) to production training")
else:
    print("\nNo Grokfast config beat baseline. Skip for now.")
