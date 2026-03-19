"""Sutra 100M-param: Production-Scale Training.

Trains on MiniPile (5.6GB, ~1.4B tokens) with the v0.3 architecture.
This is where Sutra proves itself against real production models.

Architecture (same v0.3, scaled):
- dim=512
- 4-byte patches (best from sweep)
- 6 message passing rounds
- k=16 sparse retrieval
- PonderNet adaptive halting
- 512 sequence length

Compares against matched-param transformer.

Usage:
    python code/sutra_100m.py  # Requires GPU (~12GB VRAM)
"""

import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = Path(__file__).parent.parent

# 100M-scale hyperparameters
DIM = 512
PATCH_SIZE = 4
MAX_ROUNDS = 6
K_RETRIEVAL = 16
SEQ_LEN = 512
BATCH_SIZE = 32
EPOCHS = 1  # Single epoch over 1.4B tokens is enough
LR = 1e-4
WARMUP_STEPS = 2000
MAX_BYTES = None  # Use all data


class StreamingByteDataset(IterableDataset):
    """Memory-efficient streaming dataset for large files.

    Reads chunks from file without loading everything into memory.
    """

    def __init__(self, path, seq_len=SEQ_LEN, chunk_size=10_000_000):
        self.path = path
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.file_size = os.path.getsize(path)

    def __iter__(self):
        with open(self.path, "rb") as f:
            buffer = b""
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                buffer += chunk
                while len(buffer) >= self.seq_len + 1:
                    x = torch.tensor(list(buffer[: self.seq_len]), dtype=torch.long)
                    y = torch.tensor(list(buffer[1 : self.seq_len + 1]), dtype=torch.long)
                    yield x, y
                    # Slide by seq_len (non-overlapping for efficiency)
                    buffer = buffer[self.seq_len :]

    def __len__(self):
        return self.file_size // self.seq_len


# Import architecture
import sys
sys.path.insert(0, str(REPO / "code"))
from sutra_v03_mvp import SutraV03, TransformerBaseline


def get_lr(step, warmup, max_lr, total_steps):
    """Cosine learning rate with warmup."""
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, loader, opt, total_steps, use_kl=False, max_batches=None):
    """Train for one epoch with cosine LR schedule."""
    model.train()
    total_loss = n = step = 0
    start = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        x, y = x.to(DEVICE), y.to(DEVICE)

        # LR schedule
        lr = get_lr(step, WARMUP_STEPS, LR, total_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        logits, aux = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss = ce + (0.01 * aux if use_kl and isinstance(aux, torch.Tensor) else 0.0)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += ce.item() * x.size(0)
        n += x.size(0)
        step += 1

        if (batch_idx + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = (batch_idx + 1) / elapsed
            print(
                f"    Batch {batch_idx+1}: loss={total_loss/n:.4f} "
                f"lr={lr:.2e} {rate:.1f} batch/s ({elapsed:.0f}s)",
                flush=True,
            )

    return total_loss / n if n > 0 else 0


@torch.no_grad()
def eval_bpb(model, loader, max_batches=500):
    model.eval()
    total = n = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x)
        total += F.cross_entropy(
            logits.reshape(-1, 256), y.reshape(-1), reduction="sum"
        ).item()
        n += y.numel()
    return total / (n * math.log(2)) if n > 0 else float("inf")


def main():
    print(f"Sutra 100M-param: PRODUCTION-SCALE TRAINING")
    print(f"Device: {DEVICE}")
    print(f"Data: MiniPile (~5.6GB, ~1.4B tokens)")
    print(f"=" * 60)

    # Data paths
    train_path = REPO / "data" / "minipile_train.txt"
    test_path = REPO / "data" / "corpus_test.txt"  # Use existing test set

    if not train_path.exists():
        print("ERROR: MiniPile not found. Run download first.")
        return

    train_ds = StreamingByteDataset(train_path, seq_len=SEQ_LEN)
    print(f"Train: ~{len(train_ds):,} samples ({train_ds.file_size / 1024 / 1024:.0f}MB)")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False
    )

    # Test loader (from existing corpus)
    from sutra_10m import ByteDataset
    test_ds = ByteDataset(test_path, seq_len=SEQ_LEN, max_bytes=1_000_000)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    # Estimate total steps
    total_steps = len(train_ds) // BATCH_SIZE
    max_batches_per_epoch = min(total_steps, 50000)  # Cap at 50K batches per epoch
    print(f"Total steps (capped): {max_batches_per_epoch:,}")

    configs = [
        (
            "Transformer-100M",
            lambda: TransformerBaseline(
                dim=DIM, n_layers=12, n_heads=8, max_seq=SEQ_LEN
            ).to(DEVICE),
            False,
        ),
        (
            "Sutra-100M",
            lambda: SutraV03(
                dim=DIM,
                patch_size=PATCH_SIZE,
                max_rounds=MAX_ROUNDS,
                k_retrieval=K_RETRIEVAL,
                max_seq=SEQ_LEN,
            ).to(DEVICE),
            True,
        ),
    ]

    results = []
    for name, make_model, use_kl in configs:
        print(f"\n{'='*40}")
        print(f"{name}")
        print(f"{'='*40}")

        model = make_model()
        params = model.count_params()
        print(f"  Params: {params:,}")
        print(f"  VRAM estimate: ~{params * 4 / 1024 / 1024:.0f}MB (fp32)")

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        start = time.time()
        train_loss = train_epoch(
            model, train_loader, opt, max_batches_per_epoch,
            use_kl=use_kl, max_batches=max_batches_per_epoch,
        )
        elapsed = time.time() - start

        bpb = eval_bpb(model, test_loader)
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Test BPB: {bpb:.4f}")
        print(f"  Time: {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

        results.append({
            "model": name,
            "params": params,
            "train_loss": round(train_loss, 4),
            "bpb": round(bpb, 4),
            "time_s": round(elapsed, 1),
        })

        # Save model weights
        torch.save(model.state_dict(), REPO / "results" / f"{name.lower()}_weights.pt")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print(f"\n{'='*60}")
    print(f"100M-PARAM RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['model']:25s}: BPB={r['bpb']:.4f}  params={r['params']:,}")

    # Save
    out = REPO / "results" / "sutra_100m.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": "sutra_100m",
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "id": "sutra_100m",
        "purpose": "100M-param production-scale test on MiniPile",
        "metrics": {r["model"]: r["bpb"] for r in results},
        "status": "DONE",
    }
    with open(REPO / "experiments" / "ledger.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
