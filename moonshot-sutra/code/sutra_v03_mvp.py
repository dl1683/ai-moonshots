"""Sutra v0.3-MVP: The Final MVP Architecture.

THREE mechanisms, clean and falsifiable:
1. Local message passing between patches (O(n))
2. Sparse top-k attention to k distant patches (content-addressable retrieval)
3. PonderNet adaptive depth on processing rounds (1-8)

Byte-level input -> 8-byte patches -> process -> predict next byte.
Compare against transformer baseline at matched parameters.

Usage:
    python code/sutra_v03_mvp.py
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


# =============================================================================
# Data
# =============================================================================

class ByteDataset(Dataset):
    def __init__(self, path, seq_len=256):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self.data = torch.tensor(list(text.encode("utf-8")[:2_000_000]), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len], self.data[idx + 1:idx + self.seq_len + 1]


# =============================================================================
# Transformer Baseline
# =============================================================================

class TransformerBaseline(nn.Module):
    def __init__(self, vocab=256, dim=256, n_layers=6, n_heads=4, max_seq=256, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, n_heads, dim * 4, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.drop(self.emb(x) + self.pos(torch.arange(T, device=x.device)))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.head(self.ln(h)), 0.0

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Sutra v0.3-MVP
# =============================================================================

class PatchProcessor(nn.Module):
    """Shared-weight local processor for one patch of bytes."""

    def __init__(self, vocab, patch_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(patch_size, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        """x: (B, P) -> (B, P, dim)"""
        B, P = x.shape
        h = self.emb(x) + self.pos(torch.arange(P, device=x.device))
        return h + self.net(h)


class SparseRetrieval(nn.Module):
    """Sparse top-k attention: each patch attends to k most relevant distant patches.

    O(n*k) complexity where k is fixed (e.g., 4).
    Content-addressable: uses query-key matching to find relevant patches.
    """

    def __init__(self, dim, k=4):
        super().__init__()
        self.k = k
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, summaries):
        """summaries: (B, N, dim) -> (B, N, dim) with sparse retrieval."""
        B, N, D = summaries.shape
        q = self.q_proj(summaries)  # (B, N, D)
        k = self.k_proj(summaries)  # (B, N, D)
        v = self.v_proj(summaries)  # (B, N, D)

        # Compute all pairwise scores (this is O(N^2) but N = seq/8 is small)
        # For production, would use approximate top-k (LSH, etc.)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)  # (B, N, N)

        # Causal masking: can only attend to past patches
        causal = torch.triu(torch.ones(N, N, device=summaries.device) * float("-inf"), diagonal=1)
        scores = scores + causal

        # Top-k selection per query
        if N > self.k:
            topk_vals, topk_idx = scores.topk(self.k, dim=-1)  # (B, N, k)
            # Create sparse attention mask
            sparse_scores = torch.full_like(scores, float("-inf"))
            sparse_scores.scatter_(2, topk_idx, topk_vals)
            attn = F.softmax(sparse_scores, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)

        retrieved = torch.bmm(attn, v)  # (B, N, D)
        return self.out_proj(retrieved)


class MessagePassingWithHalting(nn.Module):
    """Local message passing with PonderNet-style adaptive halting.

    Each round: exchange info with neighbor patches.
    Halting: geometric distribution decides when to stop.
    """

    def __init__(self, dim, max_rounds=8, window=3, lambda_p=0.2):
        super().__init__()
        self.max_rounds = max_rounds
        self.window = window
        self.lambda_p = lambda_p

        # Message function (shared across rounds)
        self.msg = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        # Update function
        self.update = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        # Halting head
        self.halt = nn.Linear(dim, 1)
        self.ln = nn.LayerNorm(dim)

    def forward(self, summaries):
        """summaries: (B, N, dim) -> (B, N, dim), kl_loss"""
        B, N, D = summaries.shape
        h = summaries
        remaining = torch.ones(B, N, 1, device=h.device)
        output = torch.zeros_like(h)
        kl_loss = 0.0
        total_steps = 0.0

        for step in range(self.max_rounds):
            # Message passing: each patch aggregates from causal neighbors
            new_h = torch.zeros_like(h)
            for i in range(N):
                start = max(0, i - self.window)
                neighbors = h[:, start:i + 1, :]
                self_rep = h[:, i:i + 1, :].expand_as(neighbors)
                msgs = self.msg(torch.cat([self_rep, neighbors], dim=-1))
                agg = msgs.mean(dim=1)
                upd_in = torch.cat([h[:, i, :], agg], dim=-1)
                new_h[:, i, :] = h[:, i, :] + self.update(upd_in)

            h = self.ln(new_h)

            # Halting probability
            halt_prob = torch.sigmoid(self.halt(h.mean(dim=1, keepdim=True)))  # (B, 1, 1)
            halt_prob = halt_prob.expand(-1, N, -1)

            if step == self.max_rounds - 1:
                halt_prob = torch.ones_like(halt_prob)

            step_weight = remaining * halt_prob
            output = output + step_weight * h
            remaining = remaining * (1 - halt_prob)
            total_steps += step_weight.mean().item()

            # KL loss (PonderNet geometric prior)
            p_geom = self.lambda_p * (1 - self.lambda_p) ** step
            kl_loss += F.kl_div(
                torch.log(halt_prob.mean() + 1e-8),
                torch.tensor(p_geom, device=h.device).log(),
                log_target=True,
                reduction="batchmean",
            )

            if remaining.max() < 0.01:
                break

        self._avg_steps = total_steps
        return output, kl_loss / self.max_rounds


class SutraV03(nn.Module):
    """Sutra v0.3-MVP: patch processing + message passing + sparse retrieval + adaptive depth."""

    def __init__(self, vocab=256, patch_size=8, dim=128, max_rounds=8, k_retrieval=4, max_seq=256):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.max_seq = max_seq

        # Patch-level processing
        self.patch_proc = PatchProcessor(vocab, patch_size, dim)

        # Summarize patch -> single vector
        self.summarize = nn.Linear(dim, dim)

        # Message passing with halting
        self.msg_pass = MessagePassingWithHalting(dim, max_rounds=max_rounds)

        # Sparse retrieval
        self.retrieval = SparseRetrieval(dim, k=k_retrieval)

        # Broadcast summary back + predict
        self.broadcast = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        P = self.patch_size

        # Chunk into patches
        n_patches = math.ceil(T / P)
        x_pad = F.pad(x, (0, n_patches * P - T))
        patches = x_pad.view(B, n_patches, P)  # (B, N, P)

        # Local processing per patch
        local_features = []
        for i in range(n_patches):
            feat = self.patch_proc(patches[:, i, :])  # (B, P, dim)
            local_features.append(feat)
        local_features = torch.stack(local_features, dim=1)  # (B, N, P, dim)

        # Summarize patches
        summaries = self.summarize(local_features.mean(dim=2))  # (B, N, dim)

        # Message passing with adaptive depth
        msg_output, kl_loss = self.msg_pass(summaries)  # (B, N, dim)

        # Sparse retrieval
        retrieved = self.retrieval(msg_output)  # (B, N, dim)

        # Combine message passing + retrieval
        combined = msg_output + retrieved

        # Broadcast back to positions
        broad = self.broadcast(combined)  # (B, N, dim)
        broad_exp = broad.unsqueeze(2).expand(-1, -1, P, -1)  # (B, N, P, dim)

        # Combine local + global
        final = local_features + broad_exp
        final = final.view(B, n_patches * P, -1)[:, :T, :]  # (B, T, dim)

        logits = self.head(self.ln(final))
        return logits, kl_loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Training
# =============================================================================

def train_model(model, loader, epochs=10, lr=3e-4, name="model", use_kl=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, aux = model(x)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = ce + (0.01 * aux if use_kl else 0.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += ce.item() * x.size(0)
            n += x.size(0)
        avg_steps = getattr(model, "msg_pass", None)
        step_info = ""
        if avg_steps and hasattr(avg_steps, "_avg_steps"):
            step_info = f" steps={avg_steps._avg_steps:.1f}"
        print(f"  [{name}] Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}{step_info}", flush=True)


@torch.no_grad()
def eval_bpb(model, loader):
    model.eval()
    total = 0
    n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    return total / (n * math.log(2))


def main():
    print(f"Sutra v0.3-MVP: THE CRITICAL ARCHITECTURE TEST")
    print(f"Device: {DEVICE}")
    print(f"Message passing + sparse retrieval + adaptive depth vs transformer")
    print(f"=" * 60)

    # Data
    data_dir = REPO / "data"
    corpus = data_dir / "corpus_train.txt"
    if not corpus.exists():
        corpus = data_dir / "train.txt"
    test_corpus = data_dir / "corpus_test.txt"
    if not test_corpus.exists():
        test_corpus = data_dir / "test.txt"

    train_ds = ByteDataset(corpus, seq_len=256)
    test_ds = ByteDataset(test_corpus, seq_len=256)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds):,} samples")
    print(f"Test: {len(test_ds):,} samples")

    results = []
    configs = [
        ("Transformer-6L", lambda: TransformerBaseline(dim=128, n_layers=6).to(DEVICE), False),
        ("Sutra-v0.3-MVP", lambda: SutraV03(dim=128, patch_size=8, max_rounds=4, k_retrieval=4).to(DEVICE), True),
    ]

    for name, make_model, use_kl in configs:
        print(f"\n{'='*40}")
        print(f"{name}")
        print(f"{'='*40}")
        model = make_model()
        print(f"  Params: {model.count_params():,}")
        start = time.time()
        train_model(model, train_loader, epochs=5, name=name[:15], use_kl=use_kl)
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
    print(f"RESULTS")
    print(f"{'='*60}")
    baseline_bpb = results[0]["bpb"]
    for r in results:
        ratio = r["bpb"] / baseline_bpb if baseline_bpb > 0 else 0
        tag = "BASELINE" if r == results[0] else f"{ratio:.3f}x"
        print(f"  {r['model']:25s}: BPB={r['bpb']:.4f}  params={r['params']:,}  {tag}")

    # Save
    out = REPO / "results" / "sutra_v03_mvp.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"experiment": "sutra_v03_mvp", "results": results,
                    "timestamp": datetime.now().isoformat()}, f, indent=2)

    entry = {"timestamp": datetime.now().isoformat(), "id": "sutra_v03_mvp",
             "purpose": "v0.3-MVP vs transformer: message passing + sparse retrieval + adaptive depth",
             "metrics": {r["model"]: r["bpb"] for r in results}, "status": "DONE"}
    with open(REPO / "experiments" / "ledger.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
