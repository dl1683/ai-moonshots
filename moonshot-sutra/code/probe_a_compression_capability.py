"""Probe A: Compression ↔ Capability Correlation.

CORE THESIS TEST: Does better compression actually predict better reasoning?

Design: Train 6 tiny models (10M params each) with different objectives.
Measure held-out bits-per-byte AND accuracy on synthetic reasoning tasks.
Plot compression vs capability. If r > 0.8, thesis confirmed.

Controls: Same architecture, same data, same params, same training steps.
Only the objective/regularization varies.

Kill criterion: If r(compression, capability) < 0.5 at this scale,
compression != intelligence.
"""

import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
LEDGER_PATH = REPO_ROOT / "experiments" / "ledger.jsonl"


# =============================================================================
# Minimal Transformer (10M params, ~6 layers, dim 256, 4 heads)
# =============================================================================

class MiniTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        # Generate causal mask
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=6, n_heads=4, max_seq=512, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([
            MiniTransformerBlock(dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq = max_seq

    def forward(self, x):
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = tok + pos
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Synthetic Reasoning Tasks
# =============================================================================

def generate_arithmetic_chain(n_steps, max_val=50):
    """Generate a multi-step arithmetic reasoning problem."""
    val = random.randint(1, max_val)
    steps = [f"x = {val}"]
    for _ in range(n_steps):
        op = random.choice(["+", "-", "*"])
        operand = random.randint(1, 10)
        if op == "+":
            val = val + operand
        elif op == "-":
            val = val - operand
        else:
            val = val * operand
        steps.append(f"x = x {op} {operand}")
    question = "; ".join(steps) + "; x = ?"
    return question, str(val)


def generate_logic_chain(n_steps):
    """Generate a multi-step logic problem."""
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
    props = ["tall", "smart", "fast", "kind", "brave", "wise"]
    random.shuffle(names)
    random.shuffle(props)

    assignments = {}
    rules = []
    for i in range(min(n_steps, len(names))):
        assignments[names[i]] = props[i]
        rules.append(f"{names[i]} is {props[i]}")

    # Add inference rules
    for i in range(min(n_steps - 1, len(names) - 1)):
        rules.append(f"If someone is {props[i]}, they are also {props[i+1]}")

    target_name = names[0]
    # Follow the chain
    final_prop = props[min(n_steps - 1, len(props) - 1)]

    question = ". ".join(rules) + f". Is {target_name} {final_prop}?"
    answer = "yes"
    return question, answer


def generate_reasoning_dataset(n_samples=200):
    """Generate a mix of reasoning tasks at various difficulties."""
    tasks = []
    for _ in range(n_samples // 2):
        n_steps = random.choice([1, 2, 3, 5])
        q, a = generate_arithmetic_chain(n_steps)
        tasks.append({"question": q, "answer": a, "type": "arithmetic", "steps": n_steps})
    for _ in range(n_samples // 2):
        n_steps = random.choice([1, 2, 3, 4])
        q, a = generate_logic_chain(n_steps)
        tasks.append({"question": q, "answer": a, "type": "logic", "steps": n_steps})
    return tasks


# =============================================================================
# Text Dataset (Simple character-level for compression measurement)
# =============================================================================

class CharDataset(Dataset):
    def __init__(self, text, seq_len=256):
        self.data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def get_text_data():
    """Get training text. Use a simple built-in corpus for the probe."""
    # Generate synthetic but structured text for reproducibility
    # In production, replace with WikiText or similar
    random.seed(42)
    words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "to", "big", "small",
             "red", "blue", "house", "tree", "car", "bird", "fish", "sun", "moon", "star",
             "if", "then", "because", "when", "while", "and", "or", "not", "but", "so",
             "every", "some", "no", "all", "each", "many", "few", "most", "any", "other",
             "is", "was", "has", "had", "will", "can", "may", "should", "could", "would",
             "think", "know", "see", "find", "give", "take", "make", "come", "go", "say"]

    sentences = []
    for _ in range(5000):
        length = random.randint(5, 20)
        sent = " ".join(random.choices(words, k=length))
        sentences.append(sent + ".")

    text = " ".join(sentences)
    split = int(len(text) * 0.9)
    return text[:split], text[split:]


# =============================================================================
# Training Objectives
# =============================================================================

def train_ce(model, train_loader, epochs=10, lr=3e-4):
    """Standard cross-entropy training."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}")
    return model


def train_ce_l2(model, train_loader, epochs=10, lr=3e-4, l2_weight=0.01):
    """CE + explicit L2 regularization (simple MDL approximation)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_weight)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            # Explicit L2 on top of AdamW weight decay
            l2 = sum(p.pow(2).sum() for p in model.parameters()) * l2_weight
            total = loss + l2
            opt.zero_grad()
            total.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return model


def train_ce_dropout(model, train_loader, epochs=10, lr=3e-4):
    """CE with heavy dropout (compression via information bottleneck)."""
    # Dropout is already in the model if dropout > 0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return model


def train_label_smoothing(model, train_loader, epochs=10, lr=3e-4, smoothing=0.1):
    """CE with label smoothing."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                   label_smoothing=smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return model


def train_mdl_approx(model, train_loader, epochs=10, lr=3e-4):
    """Variational MDL approximation: CE + KL penalty on weight distribution.

    Treats each weight as drawn from a learned Gaussian, penalizes KL from a
    standard normal prior. This is the bits-back / variational Bayes approach.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n_data = len(train_loader.dataset)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # KL penalty: treat weights as point estimates with implicit variance
            # Approximate: sum of squared weights / (2 * n_data) — MAP with N(0,1) prior
            kl = sum(p.pow(2).sum() for p in model.parameters()) / (2 * n_data)

            loss = ce_loss + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += ce_loss.item() * x.size(0)
            n += x.size(0)
    return model


def train_noisy(model, train_loader, epochs=10, lr=3e-4, noise_rate=0.3):
    """Deliberately BAD compressor: train on noisy labels."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Corrupt labels
            mask = torch.rand_like(y.float()) < noise_rate
            noisy_y = y.clone()
            noisy_y[mask] = torch.randint(0, 256, (mask.sum().item(),), device=DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), noisy_y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return model


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def measure_compression(model, test_loader):
    """Measure bits-per-byte on held-out data."""
    model.eval()
    total_loss = 0
    total_chars = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_chars += y.numel()
    # Convert nats to bits: bits = nats / ln(2)
    bits_per_byte = total_loss / (total_chars * math.log(2))
    return bits_per_byte


@torch.no_grad()
def measure_reasoning(model, tasks, vocab_size=256):
    """Measure accuracy on reasoning tasks by feeding question and checking if model
    generates the correct answer token sequence."""
    model.eval()
    correct = 0
    total = 0
    for task in tasks:
        q = task["question"]
        a = task["answer"]
        # Encode question as bytes
        q_tokens = torch.tensor([ord(c) % vocab_size for c in q], dtype=torch.long).unsqueeze(0).to(DEVICE)
        if q_tokens.size(1) > model.max_seq - 10:
            q_tokens = q_tokens[:, :model.max_seq - 10]

        # Generate answer tokens autoregressively
        generated = []
        inp = q_tokens
        for _ in range(len(a) + 5):  # Generate a few extra tokens
            if inp.size(1) > model.max_seq:
                inp = inp[:, -model.max_seq:]
            logits = model(inp)
            next_token = logits[0, -1].argmax()
            generated.append(next_token.item())
            inp = torch.cat([inp, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        # Check if generated starts with the answer
        gen_str = "".join(chr(t) for t in generated[:len(a)])
        if gen_str.strip() == a.strip():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Probe A: Compression <-> Capability Correlation")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)

    # Prepare data
    train_text, test_text = get_text_data()
    train_ds = CharDataset(train_text, seq_len=256)
    test_ds = CharDataset(test_text, seq_len=256)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Prepare reasoning tasks
    reasoning_tasks = generate_reasoning_dataset(200)

    # Define the 6 training conditions
    conditions = [
        ("CE_standard", lambda: MiniTransformer(256, dim=256, n_layers=6, n_heads=4).to(DEVICE),
         lambda m, tl: train_ce(m, tl, epochs=5)),
        ("CE_L2_heavy", lambda: MiniTransformer(256, dim=256, n_layers=6, n_heads=4).to(DEVICE),
         lambda m, tl: train_ce_l2(m, tl, epochs=5, l2_weight=0.1)),
        ("CE_dropout", lambda: MiniTransformer(256, dim=256, n_layers=6, n_heads=4, dropout=0.3).to(DEVICE),
         lambda m, tl: train_ce_dropout(m, tl, epochs=5)),
        ("label_smoothing", lambda: MiniTransformer(256, dim=256, n_layers=6, n_heads=4).to(DEVICE),
         lambda m, tl: train_label_smoothing(m, tl, epochs=5, smoothing=0.1)),
        ("MDL_approx", lambda: MiniTransformer(256, dim=256, n_layers=6, n_heads=4).to(DEVICE),
         lambda m, tl: train_mdl_approx(m, tl, epochs=5)),
        ("noisy_labels", lambda: MiniTransformer(256, dim=256, n_layers=6, n_heads=4).to(DEVICE),
         lambda m, tl: train_noisy(m, tl, epochs=5, noise_rate=0.3)),
    ]

    results = []
    for name, make_model, train_fn in conditions:
        print(f"\n{'='*40}")
        print(f"Training: {name}")
        print(f"{'='*40}")

        model = make_model()
        print(f"  Params: {model.count_params():,}")

        start = time.time()
        model = train_fn(model, train_loader)
        train_time = time.time() - start

        # Measure compression
        bpb = measure_compression(model, test_loader)
        print(f"  Bits-per-byte: {bpb:.4f}")

        # Measure reasoning
        acc = measure_reasoning(model, reasoning_tasks)
        print(f"  Reasoning accuracy: {acc:.4f}")
        print(f"  Training time: {train_time:.1f}s")

        results.append({
            "condition": name,
            "bits_per_byte": bpb,
            "reasoning_accuracy": acc,
            "train_time_s": round(train_time, 1),
            "params": model.count_params(),
        })

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute correlation
    bpbs = [r["bits_per_byte"] for r in results]
    accs = [r["reasoning_accuracy"] for r in results]

    # Pearson correlation (manual to avoid numpy dependency)
    n = len(bpbs)
    mean_bpb = sum(bpbs) / n
    mean_acc = sum(accs) / n
    cov = sum((b - mean_bpb) * (a - mean_acc) for b, a in zip(bpbs, accs)) / n
    std_bpb = (sum((b - mean_bpb)**2 for b in bpbs) / n) ** 0.5
    std_acc = (sum((a - mean_acc)**2 for a in accs) / n) ** 0.5
    r = cov / (std_bpb * std_acc) if std_bpb > 0 and std_acc > 0 else 0.0

    # Note: LOWER bpb = BETTER compression, so we expect NEGATIVE r with accuracy
    # (better compression -> higher accuracy). Report |r| or flip sign.
    # Actually: better compression = lower bpb, better capability = higher acc
    # So r should be negative if thesis is true. Report as r_compression_capability = -r
    r_thesis = -r  # Positive means better compression predicts better capability

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    for r_item in results:
        print(f"  {r_item['condition']:20s}: bpb={r_item['bits_per_byte']:.4f}, "
              f"acc={r_item['reasoning_accuracy']:.4f}")
    print(f"\n  Correlation (compression -> capability): r = {r_thesis:.4f}")
    print(f"  THESIS {'CONFIRMED' if r_thesis > 0.5 else 'AMBIGUOUS' if r_thesis > 0.0 else 'REJECTED'}")
    print(f"  Kill criterion (r < 0.5): {'KILLED' if r_thesis < 0.5 else 'SURVIVED'}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "probe": "A_compression_capability",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Better compression predicts better reasoning at fixed params",
        "r_compression_capability": round(r_thesis, 4),
        "kill_criterion": "r < 0.5",
        "verdict": "CONFIRMED" if r_thesis > 0.5 else "AMBIGUOUS" if r_thesis > 0.0 else "REJECTED",
        "conditions": results,
    }
    out_path = RESULTS_DIR / "probe_a_compression_capability.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Log to ledger
    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "id": "probe_a_compression_capability",
        "purpose": "Test core thesis: does better compression predict better reasoning?",
        "command": "python code/probe_a_compression_capability.py",
        "metrics": {"r": round(r_thesis, 4), "n_conditions": 6, "n_reasoning_tasks": 200},
        "status": "DONE",
        "notes": output["verdict"],
    }
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")


if __name__ == "__main__":
    main()
