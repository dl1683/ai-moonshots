"""Sutra Token-Level: Same architecture, BPE tokenizer input.

Parallel branch per Codex recommendation: if byte-level BPB is mediocre,
token-level with BPE gives much shorter sequences and better sample efficiency.

Architecture: identical v0.4 (GRU patches + msg passing + sparse retrieval)
Input: BPE tokens (vocab ~32K) instead of raw bytes (vocab 256)
Patches: groups of 4 BPE tokens instead of 4 bytes

Key advantage: 4 BPE tokens ≈ 16-20 bytes ≈ 4-5 words.
The model processes 4-5 words per patch instead of 4 characters.
Much more semantic information per patch.

Usage:
    python code/sutra_token_level.py
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

# Token-level hyperparameters
VOCAB_SIZE = 32000  # BPE vocab size (will be set by tokenizer)
DIM = 2048          # Smaller dim since tokens carry more info than bytes
PATCH_SIZE = 4      # 4 BPE tokens per patch ≈ 4-5 words
MAX_ROUNDS = 4
K_RETRIEVAL = 8
SEQ_LEN = 256       # In tokens (≈1024 bytes ≈ 256 words)
BATCH_SIZE = 8
GRAD_ACCUM = 4      # Effective batch = 32
LR = 3e-4
MAX_STEPS = 50000


class TokenLevelSutra(nn.Module):
    """Sutra v0.4 adapted for token-level input.

    Same architecture as byte-level but with:
    - Larger vocab (32K vs 256)
    - Shorter sequences (256 tokens vs 512 bytes)
    - Each patch processes 4 tokens ≈ 4-5 words (more semantic info per patch)
    """

    def __init__(self, vocab_size=VOCAB_SIZE, patch_size=PATCH_SIZE, dim=DIM,
                 max_rounds=MAX_ROUNDS, k_retrieval=K_RETRIEVAL, max_seq=SEQ_LEN):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.max_seq = max_seq

        # GRU patch processor (same as v0.4)
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(patch_size, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.patch_ln = nn.LayerNorm(dim)

        # Summarize patches
        self.summarize = nn.Linear(dim, dim)

        # Message passing (vectorized, from v0.4)
        self.msg_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.update_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.msg_ln = nn.LayerNorm(dim)
        self.max_rounds = max_rounds
        self.window = 4

        # Sparse retrieval
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.k_retrieval = k_retrieval

        # Output
        self.broadcast = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        P = self.patch_size
        n_patches = math.ceil(T / P)
        x_pad = F.pad(x, (0, n_patches * P - T))
        patches = x_pad.view(B, n_patches, P)

        # GRU patch processing (vectorized)
        flat = patches.reshape(B * n_patches, P)
        h = self.emb(flat) + self.pos_emb(torch.arange(P, device=x.device))
        h, _ = self.gru(h)
        local = self.patch_ln(h).reshape(B, n_patches, P, -1)

        # Summarize
        summaries = self.summarize(local.mean(dim=2))

        # Message passing (vectorized)
        h = summaries
        for _ in range(self.max_rounds):
            padded = F.pad(h, (0, 0, self.window, 0))
            neighbors = torch.stack([
                padded[:, self.window - w:self.window - w + n_patches, :]
                for w in range(self.window + 1)
            ], dim=2)
            self_exp = h.unsqueeze(2).expand_as(neighbors)
            msg_input = torch.cat([self_exp, neighbors], dim=-1)
            msgs = self.msg_net(msg_input).mean(dim=2)
            upd_input = torch.cat([h, msgs], dim=-1)
            h = self.msg_ln(h + self.update_net(upd_input))

        # Sparse retrieval
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        N = h.size(1)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dim)
        causal = torch.triu(torch.ones(N, N, device=x.device) * float("-inf"), diagonal=1)
        scores = scores + causal
        if N > self.k_retrieval:
            topk_vals, topk_idx = scores.topk(self.k_retrieval, dim=-1)
            sparse = torch.full_like(scores, float("-inf"))
            sparse.scatter_(2, topk_idx, topk_vals)
            attn = F.softmax(sparse, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)
        retrieved = self.out_proj(torch.bmm(attn, v))

        combined = h + retrieved
        broad = self.broadcast(combined).unsqueeze(2).expand(-1, -1, P, -1)
        final = local + broad
        final = final.view(B, n_patches * P, -1)[:, :T, :]

        return self.head(self.ln(final)), torch.tensor(0.0, device=x.device)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def main():
    print(f"SUTRA TOKEN-LEVEL BRANCH")
    print(f"Same architecture, BPE tokenizer input")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)

    # Use HuggingFace tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
        print(f"Tokenizer: GPT-2 BPE, vocab={vocab_size}")
    except ImportError:
        print("ERROR: transformers not installed")
        return

    # Tokenize MiniPile
    data_path = REPO / "data" / "minipile_train.txt"
    if not data_path.exists():
        print("ERROR: MiniPile not found")
        return

    print("Tokenizing MiniPile (first 50MB)...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read(50_000_000)  # 50MB

    tokens = tokenizer.encode(text)
    print(f"Tokens: {len(tokens):,} (from {len(text):,} chars)")
    print(f"Compression: {len(text)/len(tokens):.1f} chars/token")

    token_data = torch.tensor(tokens, dtype=torch.long)

    # Model
    model = TokenLevelSutra(
        vocab_size=vocab_size, dim=DIM, patch_size=PATCH_SIZE,
        max_rounds=MAX_ROUNDS, k_retrieval=K_RETRIEVAL, max_seq=SEQ_LEN
    ).to(DEVICE)
    print(f"Params: {model.count_params():,}")
    print()

    # Quick training test
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()

    for step in range(200):
        idx = torch.randint(0, len(token_data) - SEQ_LEN - 1, (BATCH_SIZE,))
        x = torch.stack([token_data[i:i + SEQ_LEN] for i in idx]).to(DEVICE)
        y = torch.stack([token_data[i + 1:i + SEQ_LEN + 1] for i in idx]).to(DEVICE)

        logits, _ = model(x)
        Tc = min(logits.size(1), y.size(1))
        loss = F.cross_entropy(logits[:, :Tc].reshape(-1, vocab_size), y[:, :Tc].reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: loss={loss.item():.4f}", flush=True)

    print(f"\nToken-level Sutra training works.")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Ready for full production training.")


if __name__ == "__main__":
    main()
