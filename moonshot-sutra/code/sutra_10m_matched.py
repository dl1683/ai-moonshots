"""Sutra 10M Matched-Params: Fair comparison at equal parameter count.

Previous 10M test had 5.5x param mismatch (Sutra 1.2M vs TF 6.5M).
This test matches params: Sutra dim=608 (6.2M) vs Transformer dim=256 (6.5M).

If Sutra wins at matched params on real text, the architecture is genuinely better.
If it loses, the prior wins were from regularization (smaller model), not architecture.

Usage:
    python code/sutra_10m_matched.py
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

SEQ_LEN = 512
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-4


class ByteDataset(Dataset):
    def __init__(self, path, seq_len=SEQ_LEN, max_bytes=None):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        raw = text.encode("utf-8")
        if max_bytes:
            raw = raw[:max_bytes]
        self.data = torch.tensor(list(raw), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len], self.data[idx + 1:idx + self.seq_len + 1]


import sys
sys.path.insert(0, str(REPO / "code"))
from sutra_v03_mvp import SutraV03, TransformerBaseline


def train_and_eval(model, train_loader, test_loader, name, epochs=EPOCHS, use_kl=False):
    params = model.count_params()
    print(f"  Params: {params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    start = time.time()
    for epoch in range(epochs):
        total_loss = n = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, aux = model(x)
            ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            loss = ce + (0.01 * aux if use_kl and isinstance(aux, torch.Tensor) else 0.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += ce.item() * x.size(0)
            n += x.size(0)
            if (batch_idx + 1) % 500 == 0:
                print(f"    Batch {batch_idx+1}: loss={total_loss/n:.4f}", flush=True)

        model.eval()
        test_loss = test_n = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x)
                test_loss += F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum").item()
                test_n += y.numel()
        bpb = test_loss / (test_n * math.log(2))
        model.train()
        print(f"  [{name}] Epoch {epoch+1}/{epochs}: test_bpb={bpb:.4f} ({time.time()-start:.0f}s)", flush=True)

    return {"model": name, "params": params, "bpb": round(bpb, 4), "time_s": round(time.time() - start, 1)}


def main():
    print(f"Sutra 10M MATCHED-PARAMS: Fair comparison")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)

    train_ds = ByteDataset(REPO / "data" / "corpus_train.txt", seq_len=SEQ_LEN, max_bytes=2_000_000)
    test_ds = ByteDataset(REPO / "data" / "corpus_test.txt", seq_len=SEQ_LEN, max_bytes=500_000)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train: {len(train_ds):,} samples")
    print(f"Test: {len(test_ds):,} samples")

    results = []

    # Transformer baseline: dim=256, 8 layers, 8 heads = ~6.5M params
    print(f"\n{'='*40}\nTransformer (baseline)\n{'='*40}")
    tf = TransformerBaseline(dim=256, n_layers=8, n_heads=8, max_seq=SEQ_LEN, dropout=0.1).to(DEVICE)
    results.append(train_and_eval(tf, train_loader, test_loader, "Transformer-6.5M"))
    del tf; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Sutra MATCHED: dim=608, patch=4, 4 rounds, k=8 = ~6.2M params
    print(f"\n{'='*40}\nSutra MATCHED (dim=608)\n{'='*40}")
    sutra = SutraV03(dim=608, patch_size=4, max_rounds=4, k_retrieval=8, max_seq=SEQ_LEN).to(DEVICE)
    results.append(train_and_eval(sutra, train_loader, test_loader, "Sutra-6.2M", use_kl=True))
    del sutra; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Sutra SMALL (original dim=256) for comparison = ~1.2M params
    print(f"\n{'='*40}\nSutra SMALL (dim=256) for efficiency comparison\n{'='*40}")
    sutra_s = SutraV03(dim=256, patch_size=4, max_rounds=4, k_retrieval=8, max_seq=SEQ_LEN).to(DEVICE)
    results.append(train_and_eval(sutra_s, train_loader, test_loader, "Sutra-1.2M", use_kl=True))
    del sutra_s; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print(f"\n{'='*60}\nMATCHED-PARAM RESULTS\n{'='*60}")
    for r in results:
        ratio = r["bpb"] / results[0]["bpb"] if results[0]["bpb"] > 0 else 0
        tag = "BASELINE" if r == results[0] else f"{ratio:.3f}x"
        print(f"  {r['model']:20s}: BPB={r['bpb']:.4f}  params={r['params']:,}  {tag}")

    out = REPO / "results" / "sutra_10m_matched.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"experiment": "sutra_10m_matched_params", "results": results,
                    "timestamp": datetime.now().isoformat()}, f, indent=2)

    entry = {"timestamp": datetime.now().isoformat(), "id": "sutra_10m_matched",
             "purpose": "Fair matched-param comparison: Sutra 6.2M vs Transformer 6.5M",
             "metrics": {r["model"]: r["bpb"] for r in results}, "status": "DONE"}
    with open(REPO / "experiments" / "ledger.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
