"""Sutra Combo 5 Pragmatist: Token-level + bf16 + GRUCell write + RoPE.

THE fastest path to a competitive model. Based on Codex analysis:
- Token-level BPE (4x shorter sequences, more semantic per position)
- bf16 mixed precision (2x throughput, half VRAM)
- v0.4 architecture with GRUCell write upgrade
- Trained on MiniPile + TinyStories

Target: compete with Pythia-410M, SmolLM-360M on standard benchmarks.
"""

import json, math, os, random, time
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

# Combo 5 hyperparameters
DIM = 1024
PATCH_SIZE = 4       # 4 BPE tokens per patch ~ 4-5 words
MAX_ROUNDS = 4
K_RETRIEVAL = 8
SEQ_LEN = 256        # In tokens (shorter than byte-level)
BATCH_SIZE = 16      # Larger batch possible with bf16
GRAD_ACCUM = 4       # Effective batch = 64
LR = 3e-4
WARMUP_STEPS = 500   # Faster warmup
MAX_STEPS = 50000
EVAL_EVERY = 2000
SAVE_EVERY = 2000    # More frequent saves!

import sys
sys.path.insert(0, str(REPO / "code"))
from sutra_v04 import SutraV04


class TokenStreamDataset(IterableDataset):
    """Streams BPE-tokenized text."""
    def __init__(self, token_file, seq_len=SEQ_LEN):
        self.tokens = torch.load(token_file, weights_only=True)
        self.seq_len = seq_len

    def __iter__(self):
        idx = 0
        while True:
            if idx + self.seq_len + 1 > len(self.tokens):
                idx = 0
            x = self.tokens[idx:idx + self.seq_len]
            y = self.tokens[idx + 1:idx + self.seq_len + 1]
            yield x, y
            idx += self.seq_len


def tokenize_and_save(text_path, out_path, max_chars=None):
    """Tokenize text file with GPT-2 BPE and save as tensor."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read(max_chars) if max_chars else f.read()
    tokens = tokenizer.encode(text)
    torch.save(torch.tensor(tokens, dtype=torch.long), out_path)
    return len(tokens), tokenizer.vocab_size


def main():
    print(f"SUTRA COMBO 5 PRAGMATIST: Token-Level + bf16")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)

    # Tokenize data if needed
    token_file = REPO / "data" / "minipile_tokens.pt"
    if not token_file.exists():
        print("Tokenizing MiniPile with GPT-2 BPE...")
        n_tokens, vocab_size = tokenize_and_save(
            REPO / "data" / "minipile_train.txt", token_file, max_chars=50_000_000
        )
        print(f"Tokenized: {n_tokens:,} tokens, vocab={vocab_size}")
    else:
        vocab_size = 50257  # GPT-2
        print("Using cached tokens.")

    # Model (token-level v0.4)
    model = SutraV04(
        vocab_size=vocab_size, dim=DIM, patch_size=PATCH_SIZE,
        max_rounds=MAX_ROUNDS, k_retrieval=K_RETRIEVAL, max_seq=SEQ_LEN
    ).to(DEVICE)

    params = model.count_params()
    print(f"Model: Sutra v0.4 (token-level)")
    print(f"Params: {params:,}")
    print(f"Vocab: {vocab_size} (GPT-2 BPE)")
    print()

    # Checkpoint resume
    ckpt_dir = REPO / "results" / "checkpoints_combo5"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0

    latest = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if latest:
        ckpt = torch.load(latest[-1], weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        print(f"RESUMED from step {start_step}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    if latest and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    # Data
    train_ds = TokenStreamDataset(token_file, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

    # Training with bf16
    model.train()
    step = start_step
    total_loss = 0
    start = time.time()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        lr = LR * min(1.0, (step + 1) / WARMUP_STEPS) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)))
        )
        for pg in opt.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, kl = model(x)
            Tc = min(logits.size(1), y.size(1))
            ce = F.cross_entropy(logits[:, :Tc].reshape(-1, vocab_size), y[:, :Tc].reshape(-1))
            loss = ce / GRAD_ACCUM

        loss.backward()
        total_loss += ce.item()

        if (batch_idx + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % 100 == 0:
                avg = total_loss / (100 * GRAD_ACCUM)
                elapsed = time.time() - start
                tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / elapsed
                print(f"Step {step:>6d}/{MAX_STEPS}: loss={avg:.4f} lr={lr:.2e} {tps:.0f}tok/s", flush=True)
                total_loss = 0

            if step % SAVE_EVERY == 0:
                ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}
                torch.save(ckpt, ckpt_dir / f"step_{step}.pt")
                torch.save(model.state_dict(), REPO / "results" / f"combo5_step{step}.pt")
                old = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
                for o in old[:-3]:
                    o.unlink()

            if step >= MAX_STEPS:
                break

    print(f"\nDone. {step} steps, {(time.time()-start)/3600:.1f}h")


if __name__ == "__main__":
    main()
