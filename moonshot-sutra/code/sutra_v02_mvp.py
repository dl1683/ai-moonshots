"""Sutra v0.2-MVP: Minimal Viable Architecture.

The SIMPLEST possible architecture that tests the core hypothesis:
Can local compression + chunk-level message passing match a transformer?

Architecture:
1. Byte-level input → fixed-window chunking (e.g., 8 bytes per chunk)
2. Local model: small transformer within each chunk (processes 8 bytes)
3. Chunk-level message passing: each chunk exchanges info with neighbors
4. Optional: tiny global scratchpad (8 memory tokens visible to all chunks)
5. Standard autoregressive next-byte prediction

This is MEGABYTE-like but with message passing instead of a global transformer
at the chunk level. The question: can O(n) message passing replace O(n²) attention
between chunks?

Usage:
    python code/sutra_v02_mvp.py  # Train and evaluate vs transformer baseline
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


# =============================================================================
# Data (use the rich training data)
# =============================================================================

class ByteDataset(Dataset):
    def __init__(self, text_path, seq_len=256):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.data = torch.tensor([b for b in text.encode("utf-8")], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# =============================================================================
# Transformer Baseline (standard, for comparison)
# =============================================================================

class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size=256, dim=256, n_layers=6, n_heads=4, max_seq=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, n_heads, dim * 4, dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq = max_seq

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
# Sutra v0.2-MVP
# =============================================================================

class LocalChunkProcessor(nn.Module):
    """Process one chunk of bytes locally. Small transformer or MLP."""

    def __init__(self, vocab_size, chunk_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(chunk_size, dim)
        # Simple 2-layer MLP (faster than attention for tiny chunks)
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, chunk_size) -> (B, chunk_size, dim)"""
        B, C = x.shape
        h = self.emb(x) + self.pos(torch.arange(C, device=x.device))
        h = h + self.layers(self.ln(h))
        return h


class ChunkMessagePassing(nn.Module):
    """Message passing between chunk summaries.

    Each chunk exchanges information with its causal neighbors (left only).
    This is the O(n) replacement for the global transformer.
    """

    def __init__(self, dim, n_rounds=3, window=4):
        super().__init__()
        self.n_rounds = n_rounds
        self.window = window  # How many previous chunks each chunk can see

        # Message computation
        self.msg_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Update computation
        self.update_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, chunk_summaries):
        """chunk_summaries: (B, N_chunks, dim) -> updated_summaries: (B, N_chunks, dim)"""
        B, N, D = chunk_summaries.shape
        h = chunk_summaries

        for round_idx in range(self.n_rounds):
            new_h = torch.zeros_like(h)

            for i in range(N):
                # Gather causal neighbors (left window only)
                start = max(0, i - self.window)
                neighbors = h[:, start:i + 1, :]  # (B, window_size, D)

                # Compute messages from all neighbors
                self_expanded = h[:, i:i + 1, :].expand_as(neighbors)
                msg_input = torch.cat([self_expanded, neighbors], dim=-1)  # (B, w, D*2)
                messages = self.msg_net(msg_input)  # (B, w, D)

                # Aggregate messages (mean)
                agg_msg = messages.mean(dim=1)  # (B, D)

                # Update
                update_input = torch.cat([h[:, i, :], agg_msg], dim=-1)  # (B, D*2)
                new_h[:, i, :] = h[:, i, :] + self.update_net(update_input)

            h = self.ln(new_h)

        return h


class GlobalScratchpad(nn.Module):
    """Tiny global memory: a few learnable tokens visible to all chunks.

    This is Codex's suggested compromise for long-range dependencies.
    """

    def __init__(self, n_slots=8, dim=256):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        self.read_proj = nn.Linear(dim, dim)
        self.write_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)

    def read(self, query):
        """query: (B, dim) -> (B, dim) read from scratchpad"""
        B = query.shape[0]
        slots = self.slots.expand(B, -1, -1)  # (B, n_slots, dim)
        q = self.read_proj(query).unsqueeze(1)  # (B, 1, dim)
        attn = torch.bmm(q, slots.transpose(1, 2)).softmax(dim=-1)  # (B, 1, n_slots)
        read_val = torch.bmm(attn, slots).squeeze(1)  # (B, dim)
        return read_val

    def write(self, value):
        """value: (B, dim) -> update scratchpad"""
        B = value.shape[0]
        write_val = self.write_proj(value).unsqueeze(1)  # (B, 1, dim)
        # Soft update: blend new value into closest slot
        slots = self.slots.expand(B, -1, -1)
        similarity = torch.bmm(write_val, slots.transpose(1, 2)).softmax(dim=-1)  # (B, 1, n_slots)
        # Update slots (detached to prevent gradient explosion)
        update = torch.bmm(similarity.transpose(1, 2), write_val)  # (B, n_slots, dim)
        self.slots.data = self.slots.data + 0.01 * update.mean(0, keepdim=True).detach()


class SutraMVP(nn.Module):
    """Sutra v0.2 Minimum Viable Architecture.

    Byte-level input → chunk → local process → message passing → predict.
    """

    def __init__(self, vocab_size=256, chunk_size=8, dim=128, n_msg_rounds=3,
                 msg_window=4, use_scratchpad=False, n_scratchpad_slots=8, max_seq=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.dim = dim
        self.max_seq = max_seq
        self.use_scratchpad = use_scratchpad

        # Local processor (within each chunk)
        self.local_proc = LocalChunkProcessor(vocab_size, chunk_size, dim)

        # Chunk summarizer: pool chunk features into one vector
        self.chunk_pool = nn.Linear(dim, dim)

        # Message passing between chunks
        self.msg_pass = ChunkMessagePassing(dim, n_rounds=n_msg_rounds, window=msg_window)

        # Broadcast chunk summary back to positions
        self.broadcast = nn.Linear(dim, dim)

        # Final prediction head
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Optional scratchpad
        if use_scratchpad:
            self.scratchpad = GlobalScratchpad(n_scratchpad_slots, dim)

    def forward(self, x):
        """x: (B, T) -> logits: (B, T, vocab_size)"""
        B, T = x.shape

        # Pad to multiple of chunk_size
        n_chunks = math.ceil(T / self.chunk_size)
        padded_T = n_chunks * self.chunk_size
        x_pad = F.pad(x, (0, padded_T - T), value=0)

        # Reshape into chunks
        chunks = x_pad.view(B, n_chunks, self.chunk_size)  # (B, N, C)

        # Local processing per chunk
        local_features = []
        for i in range(n_chunks):
            feat = self.local_proc(chunks[:, i, :])  # (B, C, dim)
            local_features.append(feat)
        local_features = torch.stack(local_features, dim=1)  # (B, N, C, dim)

        # Summarize each chunk
        chunk_summaries = self.chunk_pool(local_features.mean(dim=2))  # (B, N, dim)

        # Optional: read from scratchpad
        if self.use_scratchpad:
            for i in range(n_chunks):
                read_val = self.scratchpad.read(chunk_summaries[:, i, :])
                chunk_summaries[:, i, :] = chunk_summaries[:, i, :] + 0.1 * read_val

        # Message passing between chunks
        updated_summaries = self.msg_pass(chunk_summaries)  # (B, N, dim)

        # Optional: write to scratchpad
        if self.use_scratchpad:
            for i in range(n_chunks):
                self.scratchpad.write(updated_summaries[:, i, :])

        # Broadcast chunk info back to positions
        broadcast = self.broadcast(updated_summaries)  # (B, N, dim)
        broadcast_expanded = broadcast.unsqueeze(2).expand(-1, -1, self.chunk_size, -1)  # (B, N, C, dim)

        # Combine local + global info
        combined = local_features + broadcast_expanded  # (B, N, C, dim)
        combined = combined.view(B, padded_T, -1)[:, :T, :]  # (B, T, dim)

        # Predict
        logits = self.head(self.ln(combined))  # (B, T, vocab_size)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_model(model, loader, epochs=10, lr=3e-4, name="model"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        print(f"    [{name}] Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}", flush=True)


@torch.no_grad()
def eval_bpb(model, loader):
    model.eval()
    total_loss = 0
    total_bytes = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_bytes += y.numel()
    return total_loss / (total_bytes * math.log(2))


def main():
    print(f"Sutra v0.2-MVP: Local Compression + Message Passing")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)

    # Data — use rich training data
    data_dir = REPO_ROOT / "data"
    if not (data_dir / "train.txt").exists():
        print("Generating training data first...")
        import subprocess
        subprocess.run(["python", "-u", "code/generate_training_data.py"], cwd=str(REPO_ROOT), check=True)

    train_ds = ByteDataset(data_dir / "train.txt", seq_len=256)
    test_ds = ByteDataset(data_dir / "test.txt", seq_len=256)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds):,} samples")
    print(f"Test: {len(test_ds):,} samples")

    configs = [
        ("Transformer (baseline)", lambda: TransformerBaseline(dim=128, n_layers=6, n_heads=4).to(DEVICE)),
        ("Sutra-MVP (no scratchpad)", lambda: SutraMVP(dim=128, chunk_size=8, n_msg_rounds=3, use_scratchpad=False).to(DEVICE)),
        ("Sutra-MVP (with scratchpad)", lambda: SutraMVP(dim=128, chunk_size=8, n_msg_rounds=3, use_scratchpad=True, n_scratchpad_slots=8).to(DEVICE)),
    ]

    results = []
    for name, make_model in configs:
        print(f"\n{'='*40}")
        print(f"{name}")
        print(f"{'='*40}")
        model = make_model()
        print(f"  Params: {model.count_params():,}")
        start = time.time()
        train_model(model, train_loader, epochs=10, name=name[:15])
        elapsed = time.time() - start
        bpb = eval_bpb(model, test_loader)
        print(f"  BPB: {bpb:.4f}")
        print(f"  Time: {elapsed:.1f}s")
        results.append({"model": name, "params": model.count_params(), "bpb": round(bpb, 4),
                         "time_s": round(elapsed, 1)})
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print(f"\n{'='*60}")
    print(f"SUTRA v0.2-MVP RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['model']:35s}: BPB={r['bpb']:.4f}  params={r['params']:,}  time={r['time_s']}s")

    baseline_bpb = results[0]["bpb"]
    for r in results[1:]:
        ratio = r["bpb"] / baseline_bpb
        status = "BETTER" if ratio < 1.0 else "CLOSE" if ratio < 1.1 else "WORSE"
        print(f"  {r['model']:35s}: {ratio:.3f}x baseline ({status})")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "sutra_v02_mvp",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Local compression + message passing can match transformer",
        "results": results,
    }
    with open(RESULTS_DIR / "sutra_v02_mvp.json", "w") as f:
        json.dump(output, f, indent=2)

    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "id": "sutra_v02_mvp",
        "purpose": "Test v0.2-MVP: local chunks + message passing vs transformer baseline",
        "command": "python code/sutra_v02_mvp.py",
        "metrics": {m["model"]: m["bpb"] for m in results},
        "status": "DONE",
    }
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")

    print(f"\nResults saved to results/sutra_v02_mvp.json")


if __name__ == "__main__":
    main()
