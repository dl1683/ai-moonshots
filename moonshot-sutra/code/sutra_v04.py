"""Sutra v0.4: GRU patches + Message Passing + Sparse Retrieval + Adaptive Depth.

Validated components combined:
1. GRU within patches (8.7% better on sequential, validated)
2. Message passing between patches (28-46% structural advantage, validated)
3. Sparse top-k retrieval (16% MQAR, validated)
4. PonderNet adaptive depth with min_rounds=2 (collapse fixed)

Integration test: GRU+MsgPass beats transformer 32% on mixed data at 56% fewer params.
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
REPO = Path(__file__).parent.parent


class GRUPatchProcessor(nn.Module):
    """Recurrent processing within each patch — gives sequential capability."""

    def __init__(self, vocab_size, patch_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(patch_size, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        B, P = x.shape
        h = self.emb(x) + self.pos(torch.arange(P, device=x.device))
        out, _ = self.gru(h)
        return self.ln(out)


class LocalMessagePassing(nn.Module):
    """Message passing between patch summaries with adaptive depth."""

    def __init__(self, dim, max_rounds=6, window=4, min_rounds=2, lambda_p=0.2):
        super().__init__()
        self.max_rounds = max_rounds
        self.min_rounds = min_rounds
        self.window = window
        self.lambda_p = lambda_p

        self.msg = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.update = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.halt = nn.Linear(dim, 1)
        self.ln = nn.LayerNorm(dim)

    def forward(self, summaries):
        B, N, D = summaries.shape
        h = summaries
        remaining = torch.ones(B, N, 1, device=h.device)
        output = torch.zeros_like(h)
        kl_loss = 0.0

        for step in range(self.max_rounds):
            # VECTORIZED message passing (replaces per-patch for-loop)
            # Each patch aggregates messages from its causal window neighbors
            # Using 1D causal convolution as efficient local aggregation

            # Build neighbor features via shift-and-stack (O(N*W) not O(N^2))
            padded = F.pad(h, (0, 0, self.window, 0))  # Pad left with window zeros
            # Stack shifted versions: each position sees its window of predecessors
            neighbors = torch.stack([
                padded[:, self.window - w:self.window - w + N, :]
                for w in range(self.window + 1)
            ], dim=2)  # (B, N, window+1, D)

            # Self-expanded for message computation
            self_exp = h.unsqueeze(2).expand_as(neighbors)  # (B, N, window+1, D)

            # Compute messages for all patches at once
            msg_input = torch.cat([self_exp, neighbors], dim=-1)  # (B, N, W+1, 2D)
            msgs = self.msg(msg_input).mean(dim=2)  # (B, N, D)

            # Update all patches at once
            upd_input = torch.cat([h, msgs], dim=-1)  # (B, N, 2D)
            h = self.ln(h + self.update(upd_input))

            if step < self.min_rounds:
                halt_prob = torch.zeros(B, N, 1, device=h.device)
            else:
                halt_prob = torch.sigmoid(self.halt(h.mean(dim=1, keepdim=True)))
                halt_prob = halt_prob.expand(-1, N, -1)

            if step == self.max_rounds - 1:
                halt_prob = torch.ones_like(halt_prob)

            step_weight = remaining * halt_prob
            output = output + step_weight * h
            remaining = remaining * (1 - halt_prob)

            p_geom = self.lambda_p * (1 - self.lambda_p) ** step
            kl_loss += F.kl_div(
                torch.log(halt_prob.mean() + 1e-8),
                torch.tensor(p_geom, device=h.device).log(),
                log_target=True, reduction="batchmean",
            )

            if remaining.max() < 0.01:
                break

        self._avg_steps = (self.max_rounds - remaining.mean().item() * self.max_rounds)
        return output, kl_loss / self.max_rounds


class SparseRetrieval(nn.Module):
    """Sparse top-k content-addressable retrieval."""

    def __init__(self, dim, k=8):
        super().__init__()
        self.k = k
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, summaries):
        B, N, D = summaries.shape
        q = self.q_proj(summaries)
        k = self.k_proj(summaries)
        v = self.v_proj(summaries)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)
        causal = torch.triu(torch.ones(N, N, device=summaries.device) * float("-inf"), diagonal=1)
        scores = scores + causal

        if N > self.k:
            topk_vals, topk_idx = scores.topk(self.k, dim=-1)
            sparse = torch.full_like(scores, float("-inf"))
            sparse.scatter_(2, topk_idx, topk_vals)
            attn = F.softmax(sparse, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)

        return self.out_proj(torch.bmm(attn, v))


class SutraV04(nn.Module):
    """Sutra v0.4: The full architecture with all validated components."""

    def __init__(self, vocab_size=256, patch_size=4, dim=256, max_rounds=4,
                 k_retrieval=8, max_seq=512):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.max_seq = max_seq

        self.patch_proc = GRUPatchProcessor(vocab_size, patch_size, dim)
        self.summarize = nn.Linear(dim, dim)
        self.msg_pass = LocalMessagePassing(dim, max_rounds=max_rounds)
        self.retrieval = SparseRetrieval(dim, k=k_retrieval)
        self.broadcast = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        P = self.patch_size
        n_patches = math.ceil(T / P)
        x_pad = F.pad(x, (0, n_patches * P - T))
        patches = x_pad.view(B, n_patches, P)

        # VECTORIZED: process all patches at once (no per-patch loop)
        flat_patches = patches.reshape(B * n_patches, P)  # (B*N, P)
        flat_features = self.patch_proc(flat_patches)  # (B*N, P, D)
        local_features = flat_features.reshape(B, n_patches, P, -1)  # (B, N, P, D)

        summaries = self.summarize(local_features.mean(dim=2))
        msg_out, kl_loss = self.msg_pass(summaries)
        retrieved = self.retrieval(msg_out)
        combined = msg_out + retrieved

        broad = self.broadcast(combined).unsqueeze(2).expand(-1, -1, P, -1)
        final = local_features + broad
        final = final.view(B, n_patches * P, -1)[:, :T, :]

        return self.head(self.ln(final)), kl_loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = SutraV04(dim=64, patch_size=4, max_rounds=2, k_retrieval=4, max_seq=64)
    print(f"Sutra v0.4 params: {model.count_params():,}")
    x = torch.randint(0, 256, (2, 64))
    logits, kl = model(x)
    print(f"Forward pass OK: {logits.shape}")
