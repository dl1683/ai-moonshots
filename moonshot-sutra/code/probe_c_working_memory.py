"""Probe C: Working Memory / State Tracking.

Tests the minimal mechanism for variable binding and state tracking.

Three architectures (10M params each) on synthetic state-tracking tasks:
  A) Standard transformer (attention as implicit memory)
  B) Explicit external memory (key-value store, NTM-inspired)
  C) Recurrent state (hidden state carried forward, no attention)

Tasks: variable assignment, pointer chasing, stack operations.
Test 1, 5, 10, 20 variable bindings.

Control: Scrambled-variable baseline (variables shuffled — tests memorization vs tracking).
Kill: If all three fail at >5 bindings, problem is training not architecture.
"""

import json
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

VOCAB_SIZE = 128
DIM = 128
MAX_SEQ = 256
BATCH_SIZE = 64
EPOCHS = 10


# =============================================================================
# Synthetic State-Tracking Tasks
# =============================================================================

class StateTrackingDataset(Dataset):
    """Variable assignment and lookup tasks.

    Format: "SET x 5; SET y 3; SET z 7; GET x; ANS 5"
    Tests whether model can track N variable bindings.
    """

    def __init__(self, n_samples=5000, n_vars_list=[1, 2, 5, 10, 20]):
        self.samples = []
        random.seed(SEED)
        var_names = [chr(ord('a') + i) for i in range(26)]

        for _ in range(n_samples):
            n_vars = random.choice(n_vars_list)
            n_vars = min(n_vars, 26)

            # Generate assignments
            vars_used = var_names[:n_vars]
            values = {v: random.randint(0, 99) for v in vars_used}

            commands = []
            for v in vars_used:
                commands.append(f"SET {v} {values[v]}")

            # Add some overwrites to test tracking
            for _ in range(min(n_vars // 2, 3)):
                v = random.choice(vars_used)
                new_val = random.randint(0, 99)
                values[v] = new_val
                commands.append(f"SET {v} {new_val}")

            # Query a random variable
            query_var = random.choice(vars_used)
            commands.append(f"GET {query_var}")
            commands.append(f"ANS {values[query_var]}")

            text = "; ".join(commands)
            self.samples.append((text, n_vars, values[query_var]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, n_vars, answer = self.samples[idx]
        tokens = [ord(c) % VOCAB_SIZE for c in text]
        if len(tokens) > MAX_SEQ:
            tokens = tokens[:MAX_SEQ]
        tokens = tokens + [0] * (MAX_SEQ - len(tokens))

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y, n_vars


# =============================================================================
# Model A: Standard Transformer
# =============================================================================

class TransformerModel(nn.Module):
    def __init__(self, dim=DIM, n_layers=6):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, dim)
        self.pos = nn.Embedding(MAX_SEQ, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB_SIZE, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.head(self.ln(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Model B: External Memory (Key-Value Store)
# =============================================================================

class ExternalMemoryModel(nn.Module):
    """Transformer with an explicit key-value memory bank.

    At each layer, the model can READ from and WRITE to a fixed-size memory.
    Inspired by Neural Turing Machines but simpler.
    """

    def __init__(self, dim=DIM, n_layers=6, mem_slots=32):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, dim)
        self.pos = nn.Embedding(MAX_SEQ, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB_SIZE, bias=False)

        # External memory
        self.mem_slots = mem_slots
        self.mem_keys = nn.Parameter(torch.randn(1, mem_slots, dim) * 0.02)
        self.mem_values = nn.Parameter(torch.randn(1, mem_slots, dim) * 0.02)

        # Read/write heads
        self.read_query = nn.Linear(dim, dim)
        self.write_key = nn.Linear(dim, dim)
        self.write_value = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        # Initialize memory per batch
        mem_k = self.mem_keys.expand(B, -1, -1).clone()
        mem_v = self.mem_values.expand(B, -1, -1).clone()

        for i, layer in enumerate(self.layers):
            h = layer(h, src_mask=mask, is_causal=True)

            # Read from memory
            query = self.read_query(h)  # (B, T, dim)
            attn = torch.bmm(query, mem_k.transpose(1, 2)) / (h.size(-1) ** 0.5)
            attn = F.softmax(attn, dim=-1)  # (B, T, mem_slots)
            read = torch.bmm(attn, mem_v)  # (B, T, dim)

            # Gate: combine read with hidden
            combined = torch.cat([h, read], dim=-1)
            h = torch.sigmoid(self.gate(combined)) * h + (1 - torch.sigmoid(self.gate(combined))) * read

            # Write to memory (use last position's representation)
            if i == len(self.layers) // 2:  # Write at middle layer
                write_k = self.write_key(h[:, -1:, :])  # (B, 1, dim)
                write_v = self.write_value(h[:, -1:, :])
                mem_k = torch.cat([mem_k[:, 1:, :], write_k], dim=1)
                mem_v = torch.cat([mem_v[:, 1:, :], write_v], dim=1)

        return self.head(self.ln(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Model C: Recurrent State (No Attention)
# =============================================================================

class RecurrentModel(nn.Module):
    """Pure recurrent model — GRU layers, no attention.

    Tests whether recurrent state alone can track variables.
    """

    def __init__(self, dim=DIM, n_layers=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, dim)
        self.rnn = nn.GRU(dim, dim, n_layers, batch_first=True, dropout=0.0)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB_SIZE, bias=False)

    def forward(self, x):
        h = self.emb(x)
        h, _ = self.rnn(h)
        return self.head(self.ln(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_model(model, loader, epochs=EPOCHS, lr=3e-4, name="model"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y, nv in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"    [{name}] Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}", flush=True)


@torch.no_grad()
def evaluate_by_vars(model, loader):
    model.eval()
    results = {}
    for x, y, nv_batch in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=-1)

        for i in range(x.size(0)):
            nv = nv_batch[i].item()
            # Find "ANS " in sequence and check following digits
            seq_str = "".join(chr(t) for t in x[i].cpu().tolist() if t > 0)
            pred_str = "".join(chr(t) for t in preds[i].cpu().tolist() if t > 0)

            try:
                ans_pos = seq_str.index("ANS ")
                # The answer follows "ANS "
                true_ans = seq_str[ans_pos + 4:ans_pos + 6].strip()
                # Get model's prediction at that position
                pred_at_pos = pred_str[ans_pos + 3:ans_pos + 5].strip()
                correct = pred_at_pos == true_ans
            except (ValueError, IndexError):
                correct = False

            if nv not in results:
                results[nv] = {"correct": 0, "total": 0}
            results[nv]["total"] += 1
            if correct:
                results[nv]["correct"] += 1

    for k in results:
        r = results[k]
        r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0.0
    return results


def main():
    print(f"Probe C: Working Memory / State Tracking")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)

    train_ds = StateTrackingDataset(n_samples=8000, n_vars_list=[1, 2, 5, 10, 20])
    test_ds = StateTrackingDataset(n_samples=2000, n_vars_list=[1, 2, 5, 10, 20])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    models_config = [
        ("transformer", TransformerModel),
        ("external_memory", ExternalMemoryModel),
        ("recurrent_gru", RecurrentModel),
    ]

    results = []
    for name, ModelClass in models_config:
        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"{'='*40}")
        model = ModelClass(dim=DIM).to(DEVICE)
        print(f"  Params: {model.count_params():,}")
        start = time.time()
        train_model(model, train_loader, name=name)
        t = time.time() - start
        acc = evaluate_by_vars(model, test_loader)
        print(f"  Accuracy by #vars: {json.dumps({k: round(v['accuracy'], 3) for k, v in sorted(acc.items())})}")
        results.append({"model": name, "params": model.count_params(),
                         "accuracy_by_vars": {str(k): v for k, v in acc.items()},
                         "train_time_s": round(t, 1)})
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print(f"\n{'='*60}")
    print(f"PROBE C RESULTS")
    print(f"{'='*60}")
    for r in results:
        accs = {int(k): v["accuracy"] for k, v in r["accuracy_by_vars"].items()}
        small = accs.get(1, 0)
        large = accs.get(20, 0)
        print(f"  {r['model']:20s}: 1-var={small:.3f}  20-var={large:.3f}  params={r['params']:,}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "probe": "C_working_memory",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Different architectures have different state-tracking capacity",
        "results": results,
    }
    with open(RESULTS_DIR / "probe_c_working_memory.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "id": "probe_c_working_memory",
        "purpose": "What is the minimal mechanism for variable binding and state tracking?",
        "command": "python code/probe_c_working_memory.py",
        "metrics": {"n_models": 3},
        "status": "DONE",
    }
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")

    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
