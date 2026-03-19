"""Sutra Production Training: 300M-500M params on MiniPile.

THIS is the model that competes with Qwen3-0.6B, SmolLM-360M, Pythia-410M.
No more toy comparisons — real models, real data, real benchmarks.

Architecture: Sutra v0.4 (GRU patches + message passing + sparse retrieval + adaptive depth)
Data: MiniPile 5.6GB (~1.4B tokens)
Target: competitive BPB with best models in the 300-500M param class

Usage:
    python code/sutra_production.py
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
from torch.utils.data import DataLoader, IterableDataset

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = Path(__file__).parent.parent

# Production hyperparameters
DIM = 3072          # 300M+ params
PATCH_SIZE = 4      # Validated as best
MAX_ROUNDS = 4      # Balanced depth
K_RETRIEVAL = 16    # Strong retrieval for long-range
SEQ_LEN = 512       # Standard context
BATCH_SIZE = 8      # Fits in VRAM with gradient accumulation
GRAD_ACCUM = 4      # Effective batch = 32
LR = 1e-4           # Conservative for larger model
WARMUP_STEPS = 1000
MAX_STEPS = 50000   # ~50K steps × 32 effective batch × 512 seq = ~800M tokens
EVAL_EVERY = 2000
SAVE_EVERY = 10000


class StreamingByteDataset(IterableDataset):
    def __init__(self, path, seq_len=SEQ_LEN):
        self.path = path
        self.seq_len = seq_len
        self.file_size = os.path.getsize(path)

    def __iter__(self):
        with open(self.path, "rb") as f:
            buffer = b""
            while True:
                chunk = f.read(1_000_000)
                if not chunk:
                    f.seek(0)  # Loop over data
                    continue
                buffer += chunk
                while len(buffer) >= self.seq_len + 1:
                    x = torch.tensor(list(buffer[:self.seq_len]), dtype=torch.long)
                    y = torch.tensor(list(buffer[1:self.seq_len + 1]), dtype=torch.long)
                    yield x, y
                    buffer = buffer[self.seq_len:]


import sys
sys.path.insert(0, str(REPO / "code"))
from sutra_v04 import SutraV04


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    print(f"SUTRA PRODUCTION TRAINING")
    print(f"Target: compete with Qwen3-0.6B, SmolLM-360M, Pythia-410M")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)

    # Data
    train_path = REPO / "data" / "minipile_train.txt"
    if not train_path.exists():
        print("ERROR: MiniPile not found. Run build_training_corpus.py first.")
        return

    train_ds = StreamingByteDataset(train_path, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

    # Test data
    test_path = REPO / "data" / "corpus_test.txt"
    with open(test_path, "r", encoding="utf-8") as f:
        test_bytes = list(f.read().encode("utf-8")[:500_000])
    test_data = torch.tensor(test_bytes, dtype=torch.long)

    # Model
    model = SutraV04(
        dim=DIM, patch_size=PATCH_SIZE, max_rounds=MAX_ROUNDS,
        k_retrieval=K_RETRIEVAL, max_seq=SEQ_LEN
    ).to(DEVICE)

    params = model.count_params()
    print(f"Model: Sutra v0.4")
    print(f"Params: {params:,}")
    print(f"dim={DIM}, patch={PATCH_SIZE}, rounds={MAX_ROUNDS}, k={K_RETRIEVAL}")
    print(f"Data: {train_ds.file_size / 1e9:.1f}GB MiniPile")
    print(f"Training: {MAX_STEPS:,} steps, effective batch={BATCH_SIZE * GRAD_ACCUM}")
    print(f"Tokens: ~{MAX_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / 1e9:.1f}B")
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"VRAM: {vram:.1f}GB initial")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))

    # Training loop
    model.train()
    step = 0
    total_loss = 0
    start = time.time()
    best_bpb = float("inf")

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Update LR
        lr = get_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        logits, kl = model(x)
        Tc = min(logits.size(1), y.size(1))
        ce = F.cross_entropy(logits[:, :Tc].reshape(-1, 256), y[:, :Tc].reshape(-1))
        loss = (ce + 0.01 * kl) / GRAD_ACCUM

        loss.backward()
        total_loss += ce.item()

        if (batch_idx + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % 100 == 0:
                avg_loss = total_loss / (100 * GRAD_ACCUM)
                elapsed = time.time() - start
                tokens_per_sec = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / elapsed
                vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"Step {step:>6d}/{MAX_STEPS}: loss={avg_loss:.4f} lr={lr:.2e} "
                      f"{tokens_per_sec:.0f} tok/s {vram:.1f}GB", flush=True)
                total_loss = 0

            if step % EVAL_EVERY == 0:
                model.eval()
                eval_loss = 0
                eval_n = 0
                with torch.no_grad():
                    for i in range(0, min(len(test_data) - SEQ_LEN - 1, 100000), SEQ_LEN):
                        tx = test_data[i:i + SEQ_LEN].unsqueeze(0).to(DEVICE)
                        ty = test_data[i + 1:i + SEQ_LEN + 1].unsqueeze(0).to(DEVICE)
                        lo, _ = model(tx)
                        Tc = min(lo.size(1), ty.size(1))
                        eval_loss += F.cross_entropy(
                            lo[:, :Tc].reshape(-1, 256), ty[:, :Tc].reshape(-1), reduction="sum"
                        ).item()
                        eval_n += ty[:, :Tc].numel()
                bpb = eval_loss / (eval_n * math.log(2))
                print(f"  EVAL Step {step}: BPB={bpb:.4f} {'*BEST*' if bpb < best_bpb else ''}", flush=True)
                if bpb < best_bpb:
                    best_bpb = bpb
                    torch.save(model.state_dict(), REPO / "results" / "sutra_production_best.pt")
                model.train()

            if step % SAVE_EVERY == 0:
                torch.save(model.state_dict(), REPO / "results" / f"sutra_production_step{step}.pt")

            if step >= MAX_STEPS:
                break

    # Final eval
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Steps: {step:,}")
    print(f"Best BPB: {best_bpb:.4f}")
    print(f"Time: {(time.time()-start)/3600:.1f}h")
    print(f"Params: {params:,}")

    # Save final results
    result = {
        "model": "sutra_v04_production",
        "params": params,
        "dim": DIM,
        "best_bpb": round(best_bpb, 4),
        "steps": step,
        "tokens_seen": step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN,
        "timestamp": datetime.now().isoformat(),
    }
    with open(REPO / "results" / "sutra_production.json", "w") as f:
        json.dump(result, f, indent=2)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "id": "sutra_production",
        "purpose": f"Production Sutra v0.4 ({params:,} params) on MiniPile",
        "metrics": {"best_bpb": round(best_bpb, 4), "steps": step},
        "status": "DONE",
    }
    with open(REPO / "experiments" / "ledger.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
