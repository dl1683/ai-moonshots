#!/usr/bin/env python
"""
cgp_headline_small_beats_large.py

THE HEADLINE EXPERIMENT: Small Model + Compiled Geometry > Large Model + Standard Training

This is the Day-90 demonstration of "Intelligence = Geometry, not Scale."

Setup:
  - Small model: bge-small-en-v1.5 (33M params, 384 dim)
  - Large model: bge-base-en-v1.5 (110M params, 768 dim)
  - Small model gets optimal class-sep regularization (lambda_sep from Week 2)
  - Large model gets standard contrastive training (no geometry programming)
  - Same compute budget: matched training steps
  - Evaluate on multiple hierarchical datasets

Success criterion:
  Small+compiled knn_l1 > Large+standard knn_l1 on majority of datasets

This would prove: you don't need a bigger model, you need better geometry.

Usage:
    python -u src/cgp_headline_small_beats_large.py [--lambda_sep 1.0]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset

# ── Models ──────────────────────────────────────────────────────────

SMALL_MODEL = {
    "name": "bge-small",
    "hf_path": "BAAI/bge-small-en-v1.5",
    "params": "33M",
    "hidden_dim": 384,
}

LARGE_MODEL = {
    "name": "bge-base",
    "hf_path": "BAAI/bge-base-en-v1.5",
    "params": "110M",
    "hidden_dim": 768,
}

# ── Configuration ────────────────────────────────────────────────────

EVAL_DATASETS = ["clinc", "dbpedia_classes", "agnews", "trec", "yahoo"]
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 500
EVAL_SAMPLES = 1000
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 16
MAX_SEQ_LEN = 128
LR = 2e-4


# ── LoRA ─────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.original = original
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, original.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, r))
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(model, r, alpha):
    trainable = 0
    total = sum(p.numel() for p in model.parameters())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("query" in name or "value" in name):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name]
            lora = LoRALinear(module, r, alpha)
            setattr(parent, child_name, lora)
            trainable += r * module.in_features + module.out_features * r
    print(f"  LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    return model


# ── Losses ───────────────────────────────────────────────────────────

def contrastive_loss(embeddings, labels, temperature=0.05):
    emb = F.normalize(embeddings.float(), dim=-1)
    sim = emb @ emb.T / temperature
    n = sim.size(0)
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = labels_eq & ~torch.eye(n, dtype=torch.bool, device=sim.device)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=sim.device, requires_grad=True)
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    return -(log_prob * mask).sum() / mask.sum()


def class_sep_loss(embeddings, labels, eps=1e-6):
    emb = F.normalize(embeddings.float(), dim=-1)
    unique = labels.unique()
    if len(unique) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    centroids, intras = [], []
    for lbl in unique:
        m = labels == lbl
        ce = emb[m]
        if len(ce) < 2:
            continue
        c = ce.mean(0)
        centroids.append(c)
        intras.append(torch.norm(ce - c.unsqueeze(0), dim=-1).mean())
    if len(centroids) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    ct = torch.stack(centroids)
    inter = torch.cdist(ct.unsqueeze(0), ct.unsqueeze(0)).squeeze(0)
    triu = torch.triu(torch.ones_like(inter), diagonal=1).bool()
    return torch.stack(intras).mean() / (inter[triu].mean() + eps)


# ── Training ─────────────────────────────────────────────────────────

def train_model(model, tokenizer, texts, labels, device,
                lambda_sep=0.0, steps=500):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    label_set = sorted(set(labels))
    lmap = {l: i for i, l in enumerate(label_set)}
    lids = [lmap[l] for l in labels]
    n = len(texts)
    model.train()

    for step in range(steps):
        idx = np.random.choice(n, BATCH_SIZE, replace=False)
        batch_texts = [texts[i] for i in idx]
        batch_labels = torch.tensor([lids[i] for i in idx], device=device)
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                       max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
        out = model(**enc)
        emb = out.last_hidden_state[:, 0, :]

        loss = contrastive_loss(emb, batch_labels)
        if lambda_sep > 0:
            loss = loss + lambda_sep * class_sep_loss(emb, batch_labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{steps}: loss={loss.item():.4f}")


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, device, ds_name, max_samples=1000):
    model.eval()
    ds = load_hierarchical_dataset(ds_name, split="test", max_samples=max_samples)
    texts = [s.text for s in ds.samples]
    l0 = [s.level0_label for s in ds.samples]
    l1 = [s.level1_label for s in ds.samples]

    all_embs = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i+64]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
        out = model(**enc)
        all_embs.append(out.last_hidden_state[:, 0, :].cpu().float())

    emb = F.normalize(torch.cat(all_embs), dim=-1)
    n = emb.shape[0]

    # kNN
    sim = emb @ emb.T
    sim.fill_diagonal_(-1e9)
    _, topk = sim.topk(5, dim=1)

    l0_t = torch.tensor(l0)
    l1_t = torch.tensor(l1)
    knn_l0 = (l0_t[topk].mode(1).values == l0_t).float().mean().item()
    knn_l1 = (l1_t[topk].mode(1).values == l1_t).float().mean().item()

    # Fisher Q
    global_mean = emb.mean(0)
    s_b = torch.zeros(emb.shape[1], emb.shape[1])
    s_w = torch.zeros(emb.shape[1], emb.shape[1])
    for lbl in l1_t.unique():
        m = l1_t == lbl
        ce = emb[m]
        if len(ce) < 2:
            continue
        c = ce.mean(0)
        db = (c - global_mean).unsqueeze(1)
        s_b += len(ce) * (db @ db.T)
        cent = ce - c.unsqueeze(0)
        s_w += cent.T @ cent
    s_b /= n
    s_w /= n
    fisher_q = s_b.trace().item() / max(s_w.trace().item(), 1e-8)

    # Centroid regularity (kappa) and composite G
    centroids_list = []
    for lbl in l1_t.unique():
        m = l1_t == lbl
        ce = emb[m]
        if len(ce) >= 2:
            centroids_list.append(ce.mean(0))

    C_cls = len(centroids_list)
    d_dim = emb.shape[1]
    kappa = 0.0
    composite_G = 0.0

    if C_cls >= 2:
        ct = torch.stack(centroids_list)
        pw_sq = torch.cdist(ct.unsqueeze(0), ct.unsqueeze(0)).squeeze(0) ** 2
        triu_m = torch.triu(torch.ones_like(pw_sq), diagonal=1).bool()
        pw_vals = pw_sq[triu_m]
        d_min_sq = pw_vals.min().item()
        d_avg_sq = pw_vals.mean().item()
        kappa = d_min_sq / max(d_avg_sq, 1e-8)
        composite_G = kappa * C_cls * d_dim * fisher_q / max(C_cls - 1, 1)

    return {
        "knn_l0": knn_l0,
        "knn_l1": knn_l1,
        "fisher_q": fisher_q,
        "kappa": kappa,
        "composite_G": composite_G,
        "n_samples": n,
        "n_l0_classes": len(set(l0)),
        "n_l1_classes": len(set(l1)),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_sep", type=float, default=1.0,
                       help="Lambda for class-sep regularizer on small model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Lambda_sep for small model: {args.lambda_sep}")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {EVAL_DATASETS}")
    print()

    results = {"small_compiled": [], "large_standard": []}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED: {seed}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load training data
        train_data = load_hierarchical_dataset("clinc", split="train", max_samples=5000)
        texts = [s.text for s in train_data.samples]
        l1_labels = [s.level1_label for s in train_data.samples]

        # --- Small model + compiled geometry ---
        print(f"\n  SMALL MODEL: {SMALL_MODEL['name']} ({SMALL_MODEL['params']}) + compiled geometry")
        tok_s = AutoTokenizer.from_pretrained(SMALL_MODEL["hf_path"])
        model_s = AutoModel.from_pretrained(
            SMALL_MODEL["hf_path"], torch_dtype=torch.float16
        ).to(device)
        model_s = apply_lora(model_s, LORA_R, LORA_ALPHA)

        t0 = time.time()
        train_model(model_s, tok_s, texts, l1_labels, device,
                   lambda_sep=args.lambda_sep, steps=STEPS)
        train_time_s = time.time() - t0
        print(f"  Training time: {train_time_s:.1f}s")

        small_evals = {}
        for ds in EVAL_DATASETS:
            try:
                m = evaluate(model_s, tok_s, device, ds, EVAL_SAMPLES)
                small_evals[ds] = m
                print(f"    {ds}: knn_l0={m['knn_l0']:.3f} knn_l1={m['knn_l1']:.3f} Q={m['fisher_q']:.3f}")
            except Exception as e:
                print(f"    {ds}: ERROR - {e}")
                small_evals[ds] = {"error": str(e)}

        results["small_compiled"].append({
            "seed": seed, "model": SMALL_MODEL["name"],
            "lambda_sep": args.lambda_sep, "train_time": train_time_s,
            "eval": small_evals,
        })

        del model_s
        gc.collect()
        torch.cuda.empty_cache()

        # --- Large model + standard training ---
        print(f"\n  LARGE MODEL: {LARGE_MODEL['name']} ({LARGE_MODEL['params']}) + standard training")
        tok_l = AutoTokenizer.from_pretrained(LARGE_MODEL["hf_path"])
        model_l = AutoModel.from_pretrained(
            LARGE_MODEL["hf_path"], torch_dtype=torch.float16
        ).to(device)
        model_l = apply_lora(model_l, LORA_R, LORA_ALPHA)

        t0 = time.time()
        train_model(model_l, tok_l, texts, l1_labels, device,
                   lambda_sep=0.0, steps=STEPS)  # NO geometry programming
        train_time_l = time.time() - t0
        print(f"  Training time: {train_time_l:.1f}s")

        large_evals = {}
        for ds in EVAL_DATASETS:
            try:
                m = evaluate(model_l, tok_l, device, ds, EVAL_SAMPLES)
                large_evals[ds] = m
                print(f"    {ds}: knn_l0={m['knn_l0']:.3f} knn_l1={m['knn_l1']:.3f} Q={m['fisher_q']:.3f}")
            except Exception as e:
                print(f"    {ds}: ERROR - {e}")
                large_evals[ds] = {"error": str(e)}

        results["large_standard"].append({
            "seed": seed, "model": LARGE_MODEL["name"],
            "lambda_sep": 0.0, "train_time": train_time_l,
            "eval": large_evals,
        })

        del model_l
        gc.collect()
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  HEADLINE RESULTS: Small+Compiled vs Large+Standard")
    print(f"{'='*60}")

    wins = 0
    total = 0
    for ds in EVAL_DATASETS:
        small_vals = [r["eval"].get(ds, {}).get("knn_l1", None)
                     for r in results["small_compiled"]]
        large_vals = [r["eval"].get(ds, {}).get("knn_l1", None)
                     for r in results["large_standard"]]

        small_vals = [v for v in small_vals if v is not None]
        large_vals = [v for v in large_vals if v is not None]

        if small_vals and large_vals:
            s_mean = np.mean(small_vals)
            l_mean = np.mean(large_vals)
            diff = s_mean - l_mean
            won = diff > 0
            if won:
                wins += 1
            total += 1
            symbol = "WIN" if won else "LOSE"
            print(f"  {ds:20s}: small={s_mean:.3f} large={l_mean:.3f} "
                  f"diff={diff:+.3f} [{symbol}]")

    print(f"\n  SCORE: Small+Compiled wins {wins}/{total} datasets")
    if wins > total / 2:
        print("  >>> HEADLINE CONFIRMED: Intelligence = Geometry <<<")
    else:
        print("  >>> Headline NOT confirmed. Need stronger geometry programming. <<<")

    # Save
    output = {
        "experiment": "CGP Headline: Small+Compiled vs Large+Standard",
        "small_model": SMALL_MODEL,
        "large_model": LARGE_MODEL,
        "lambda_sep": args.lambda_sep,
        "seeds": SEEDS,
        "datasets": EVAL_DATASETS,
        "results": results,
        "wins": wins,
        "total": total,
        "headline_confirmed": wins > total / 2,
    }

    out_path = RESULTS_DIR / "cgp_headline_small_beats_large.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
