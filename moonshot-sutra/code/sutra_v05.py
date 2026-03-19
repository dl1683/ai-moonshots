"""Sutra v0.5: Routing-focused upgrade from v0.4.

Three targeted fixes from 3 rounds of Codex pipeline debate:
1. RoPE on retrieval Q/K (Stage 2: better addressing)
2. Global causal attention refresh every 2 local rounds (Stage 4: better routing)
3. GRUCell gated write instead of additive merge (Stage 5: better memory)

Everything else unchanged from v0.4. Clean, attributable improvements.
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

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


# =============================================================================
# NEW: Rotary Position Embedding for retrieval Q/K
# =============================================================================

class RotaryEmbedding(nn.Module):
    """RoPE: relative position encoding via rotation in complex plane."""

    def __init__(self, dim, max_seq=1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq = max_seq

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(x, cos, sin):
    """Apply rotary embedding to x."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos[..., :d] - x2 * sin[..., :d],
                      x2 * cos[..., :d] + x1 * sin[..., :d]], dim=-1)


# =============================================================================
# NEW: Global causal attention refresh layer
# =============================================================================

class GlobalAttentionRefresh(nn.Module):
    """One layer of standard causal multi-head attention over all patches.

    Used every 2 local message-passing rounds to ensure global information flow.
    This is the key Stage 4 fix: "wrong answers are born when evidence isn't routed."
    """

    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, h):
        B, N, D = h.shape
        mask = nn.Transformer.generate_square_subsequent_mask(N, device=h.device)
        residual = h
        h = self.ln(h)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        return residual + h


# =============================================================================
# Sutra v0.5 Architecture
# =============================================================================

class SutraV05(nn.Module):
    """Sutra v0.5: v0.4 + RoPE retrieval + attention refresh + gated write.

    Three routing-focused fixes from Codex pipeline debate (3 rounds).
    """

    def __init__(self, vocab_size=256, patch_size=4, dim=256, max_rounds=6,
                 k_retrieval=16, max_seq=512):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.max_seq = max_seq
        self.max_rounds = max_rounds
        self.k_retrieval = k_retrieval

        # Stage 1: Same as v0.4 (byte-level, fixed patches)
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(patch_size, dim)

        # Stage 3: GRU patch processor (same as v0.4)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.patch_ln = nn.LayerNorm(dim)

        # Summarize
        self.summarize = nn.Linear(dim, dim)

        # Stage 2 FIX: RoPE for retrieval addressing
        self.rope = RotaryEmbedding(dim)

        # Stage 4: Message passing (vectorized, same as v0.4)
        self.msg_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.msg_ln = nn.LayerNorm(dim)
        self.window = 4

        # Stage 4 FIX: Global attention refresh every 2 rounds
        self.attention_refresh = GlobalAttentionRefresh(dim, n_heads=min(8, dim // 64))

        # Stage 4: Sparse retrieval (same Q/K/V but with RoPE)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Stage 5 FIX: GRUCell gated write instead of additive merge
        self.gated_write = nn.GRUCell(dim * 2, dim)  # Input: msgs + retrieved, State: h

        # Stage 7: Readout (same as v0.4)
        self.broadcast = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        P = self.patch_size
        n_patches = math.ceil(T / P)
        x_pad = F.pad(x, (0, n_patches * P - T))
        patches = x_pad.view(B, n_patches, P)

        # Stage 3: GRU patch processing (vectorized)
        flat = patches.reshape(B * n_patches, P)
        h = self.emb(flat) + self.pos_emb(torch.arange(P, device=x.device))
        h, _ = self.gru(h)
        local = self.patch_ln(h).reshape(B, n_patches, P, -1)

        # Summarize patches
        summaries = self.summarize(local.mean(dim=2))  # (B, N, D)

        # Precompute RoPE for retrieval
        cos, sin = self.rope(n_patches, x.device)

        # Inner loop: local message passing + sparse retrieval + attention refresh
        h = summaries
        for round_idx in range(self.max_rounds):
            # Stage 4a: Local message passing (vectorized)
            padded = F.pad(h, (0, 0, self.window, 0))
            neighbors = torch.stack([
                padded[:, self.window - w:self.window - w + n_patches, :]
                for w in range(self.window + 1)
            ], dim=2)
            self_exp = h.unsqueeze(2).expand_as(neighbors)
            msg_input = torch.cat([self_exp, neighbors], dim=-1)
            msgs = self.msg_net(msg_input).mean(dim=2)  # (B, N, D)

            # Stage 4b: Sparse retrieval with RoPE
            q = apply_rotary(self.q_proj(h), cos, sin)
            k = apply_rotary(self.k_proj(h), cos, sin)
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
            retrieved = self.out_proj(torch.bmm(attn, v))  # (B, N, D)

            # Stage 5 FIX: Gated write (GRUCell) instead of additive merge
            write_input = torch.cat([msgs, retrieved], dim=-1)  # (B, N, 2D)
            h_flat = h.reshape(B * N, -1)
            write_flat = write_input.reshape(B * N, -1)
            h_updated = self.gated_write(write_flat, h_flat)
            h = self.msg_ln(h_updated.reshape(B, N, -1))

            # Stage 4 FIX: Global attention refresh every 2 rounds
            if (round_idx + 1) % 2 == 0:
                h = self.attention_refresh(h)

        # Stage 7: Broadcast and predict
        combined = h
        broad = self.broadcast(combined).unsqueeze(2).expand(-1, -1, P, -1)
        final = local + broad
        final = final.view(B, n_patches * P, -1)[:, :T, :]

        return self.head(self.ln(final)), torch.tensor(0.0, device=x.device)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = SutraV05(dim=256, patch_size=4, max_rounds=4, k_retrieval=8, max_seq=128)
    print(f"Sutra v0.5 params: {model.count_params():,}")
    x = torch.randint(0, 256, (2, 128))
    logits, _ = model(x)
    print(f"Forward OK: {logits.shape}")
    loss = F.cross_entropy(logits.reshape(-1, 256), x[:, :logits.size(1)].reshape(-1))
    loss.backward()
    print(f"Backward OK. v0.5 WORKING.")
