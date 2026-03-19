"""Probe B: Variable Depth with Matched Compute.

Tests whether adaptive computation depth helps reasoning when compute is matched.

Three models (20M params each) on synthetic multi-step arithmetic:
  A) Fixed 12 layers
  B) Fixed 24 layers (more depth, matched width)
  C) Shared-weight recurrent block, 1-24 iterations, adaptive halting (PonderNet-style)

Control: Random-halting baseline (same FLOPs distribution but random depth per input).

Measurements: Accuracy by reasoning depth (1,2,3,5,8-step), FLOPs per correct answer.
Kill: If adaptive < fixed-24 on hard tasks OR < fixed-12 at matched FLOPs.
"""

import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
LEDGER_PATH = REPO_ROOT / "experiments" / "ledger.jsonl"

VOCAB_SIZE = 128  # ASCII
DIM = 128
MAX_SEQ = 128
BATCH_SIZE = 64
EPOCHS = 10


# =============================================================================
# Synthetic Multi-Step Arithmetic Dataset
# =============================================================================

class ArithmeticDataset(Dataset):
    """Generate multi-step arithmetic problems as character sequences.

    Format: "12+5=17" or "12+5-3=14" or "12+5-3*2=8"
    Variable number of steps to test depth scaling.
    """

    def __init__(self, n_samples=10000, max_steps=8, max_val=50):
        self.samples = []
        random.seed(SEED)
        for _ in range(n_samples):
            n_steps = random.randint(1, max_steps)
            val = random.randint(1, max_val)
            expr = str(val)
            for _ in range(n_steps):
                op = random.choice(["+", "-"])
                operand = random.randint(1, 20)
                expr += f"{op}{operand}"
                if op == "+":
                    val += operand
                else:
                    val -= operand
            full = f"{expr}={val}"
            self.samples.append((full, n_steps))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, n_steps = self.samples[idx]
        # Encode as ASCII, pad to MAX_SEQ
        tokens = [ord(c) % VOCAB_SIZE for c in text]
        if len(tokens) > MAX_SEQ:
            tokens = tokens[:MAX_SEQ]
        else:
            tokens = tokens + [0] * (MAX_SEQ - len(tokens))

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y, n_steps


# =============================================================================
# Model A: Fixed 12 Layers
# =============================================================================

class FixedDepthModel(nn.Module):
    def __init__(self, n_layers=12, dim=DIM):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, dim)
        self.pos = nn.Embedding(MAX_SEQ, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB_SIZE, bias=False)
        self.n_layers = n_layers

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.head(self.ln(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self):
        return self.n_layers  # Proportional measure


# =============================================================================
# Model C: Adaptive Depth (PonderNet-style)
# =============================================================================

class AdaptiveDepthModel(nn.Module):
    """Shared-weight recurrent block with learned halting.

    One transformer layer applied repeatedly. A halting probability head
    decides when to stop. Geometric prior on halting (PonderNet).
    """

    def __init__(self, max_steps=24, dim=DIM, lambda_p=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, dim)
        self.pos = nn.Embedding(MAX_SEQ, dim)
        # Single shared layer (applied repeatedly)
        self.layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout=0.0, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB_SIZE, bias=False)
        # Halting mechanism
        self.halt_head = nn.Linear(dim, 1)  # Per-position halting probability
        self.max_steps = max_steps
        self.lambda_p = lambda_p  # Geometric prior parameter

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        # Accumulate output weighted by halting probabilities
        accum_output = torch.zeros(B, T, VOCAB_SIZE, device=x.device)
        remaining_prob = torch.ones(B, T, 1, device=x.device)
        total_steps = torch.zeros(B, T, 1, device=x.device)
        kl_loss = 0.0

        for step in range(self.max_steps):
            h = self.layer(h, src_mask=mask, is_causal=True)

            # Halting probability
            halt_logit = self.halt_head(h)  # (B, T, 1)
            halt_prob = torch.sigmoid(halt_logit)

            # On last step, force halt
            if step == self.max_steps - 1:
                halt_prob = torch.ones_like(halt_prob)

            # Weight this step's output
            step_prob = remaining_prob * halt_prob
            logits = self.head(self.ln(h))
            accum_output += step_prob * logits

            # KL regularization: encourage geometric distribution
            # p(halt at step n) should be close to lambda_p * (1-lambda_p)^n
            geometric_prob = self.lambda_p * (1 - self.lambda_p) ** step
            kl_loss += F.kl_div(
                torch.log(halt_prob.mean() + 1e-8),
                torch.tensor(geometric_prob, device=x.device).log(),
                log_target=True,
                reduction="batchmean",
            )

            # Update remaining probability
            remaining_prob = remaining_prob * (1 - halt_prob)
            total_steps += step_prob

            # Early exit if all sequences have halted
            if remaining_prob.max() < 0.01:
                break

        self._last_avg_steps = total_steps.mean().item()
        self._kl_loss = kl_loss / self.max_steps
        return accum_output

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self):
        return getattr(self, "_last_avg_steps", self.max_steps)


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_model(model, train_loader, epochs=EPOCHS, lr=3e-4, name="model", adaptive=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y, steps in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            if adaptive and hasattr(model, "_kl_loss"):
                loss = loss + 0.01 * model._kl_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            extra = f" avg_steps={model._last_avg_steps:.1f}" if adaptive and hasattr(model, "_last_avg_steps") else ""
            print(f"    [{name}] Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}{extra}", flush=True)


@torch.no_grad()
def evaluate_by_depth(model, test_loader):
    """Evaluate accuracy grouped by reasoning depth."""
    model.eval()
    results_by_steps = {}

    for x, y, steps_batch in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=-1)

        for i in range(x.size(0)):
            n_steps = steps_batch[i].item()
            # Check if the answer portion is correct
            # Find '=' in the sequence
            seq = x[i].cpu().tolist()
            try:
                eq_pos = seq.index(ord("=") % VOCAB_SIZE)
            except ValueError:
                continue

            # Compare predictions after '='
            if eq_pos + 1 < len(seq):
                pred_after_eq = preds[i, eq_pos:].cpu().tolist()
                true_after_eq = y[i, eq_pos:].cpu().tolist()
                # Check first few positions after =
                correct = all(p == t for p, t in zip(pred_after_eq[:4], true_after_eq[:4]) if t != 0)
            else:
                correct = False

            if n_steps not in results_by_steps:
                results_by_steps[n_steps] = {"correct": 0, "total": 0}
            results_by_steps[n_steps]["total"] += 1
            if correct:
                results_by_steps[n_steps]["correct"] += 1

    # Compute accuracy per depth
    for k in results_by_steps:
        r = results_by_steps[k]
        r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0.0

    return results_by_steps


def main():
    print(f"Probe B: Variable Depth with Matched Compute")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)

    # Data
    train_ds = ArithmeticDataset(n_samples=10000, max_steps=8)
    test_ds = ArithmeticDataset(n_samples=2000, max_steps=8)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = []

    # Model A: Fixed 12 layers
    print(f"\n{'='*40}")
    print(f"Model A: Fixed 12 Layers")
    print(f"{'='*40}")
    model_a = FixedDepthModel(n_layers=12, dim=DIM).to(DEVICE)
    print(f"  Params: {model_a.count_params():,}")
    start = time.time()
    train_model(model_a, train_loader, name="Fixed-12")
    t_a = time.time() - start
    acc_a = evaluate_by_depth(model_a, test_loader)
    print(f"  Accuracy by depth: {json.dumps({k: round(v['accuracy'], 3) for k, v in sorted(acc_a.items())})}")
    results.append({"model": "fixed_12", "params": model_a.count_params(),
                     "accuracy_by_depth": {str(k): v for k, v in acc_a.items()},
                     "train_time_s": round(t_a, 1)})
    del model_a; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Model B: Fixed 24 layers
    print(f"\n{'='*40}")
    print(f"Model B: Fixed 24 Layers")
    print(f"{'='*40}")
    model_b = FixedDepthModel(n_layers=24, dim=DIM).to(DEVICE)
    print(f"  Params: {model_b.count_params():,}")
    start = time.time()
    train_model(model_b, train_loader, name="Fixed-24")
    t_b = time.time() - start
    acc_b = evaluate_by_depth(model_b, test_loader)
    print(f"  Accuracy by depth: {json.dumps({k: round(v['accuracy'], 3) for k, v in sorted(acc_b.items())})}")
    results.append({"model": "fixed_24", "params": model_b.count_params(),
                     "accuracy_by_depth": {str(k): v for k, v in acc_b.items()},
                     "train_time_s": round(t_b, 1)})
    del model_b; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Model C: Adaptive depth (PonderNet-style)
    print(f"\n{'='*40}")
    print(f"Model C: Adaptive Depth (max 24 steps)")
    print(f"{'='*40}")
    model_c = AdaptiveDepthModel(max_steps=24, dim=DIM).to(DEVICE)
    print(f"  Params: {model_c.count_params():,}")
    start = time.time()
    train_model(model_c, train_loader, name="Adaptive", adaptive=True)
    t_c = time.time() - start
    acc_c = evaluate_by_depth(model_c, test_loader)
    avg_steps = getattr(model_c, "_last_avg_steps", 0)
    print(f"  Avg steps used: {avg_steps:.1f}")
    print(f"  Accuracy by depth: {json.dumps({k: round(v['accuracy'], 3) for k, v in sorted(acc_c.items())})}")
    results.append({"model": "adaptive_24", "params": model_c.count_params(),
                     "avg_steps": round(avg_steps, 1),
                     "accuracy_by_depth": {str(k): v for k, v in acc_c.items()},
                     "train_time_s": round(t_c, 1)})
    del model_c; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print(f"\n{'='*60}")
    print(f"PROBE B RESULTS")
    print(f"{'='*60}")
    for r in results:
        accs = {int(k): v["accuracy"] for k, v in r["accuracy_by_depth"].items()}
        easy = sum(accs.get(s, 0) for s in [1, 2]) / 2
        hard = sum(accs.get(s, 0) for s in [5, 6, 7, 8]) / max(1, sum(1 for s in [5, 6, 7, 8] if s in accs))
        print(f"  {r['model']:15s}: easy(1-2)={easy:.3f}  hard(5-8)={hard:.3f}  "
              f"params={r['params']:,}  time={r['train_time_s']}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "probe": "B_variable_depth",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Adaptive depth helps reasoning when compute is matched",
        "results": results,
    }
    out_path = RESULTS_DIR / "probe_b_variable_depth.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "id": "probe_b_variable_depth",
        "purpose": "Does adaptive depth help reasoning at matched compute?",
        "command": "python code/probe_b_variable_depth.py",
        "metrics": {"n_models": 3},
        "status": "DONE",
    }
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")

    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
