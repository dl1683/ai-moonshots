#!/usr/bin/env python
"""
cgp_week3_cross_arch.py

CGP Week 3: Cross-Architecture Replication

Tests whether the class separation -> quality relationship is UNIVERSAL,
not specific to Pythia-160M.

Design:
  - Models: bge-small-en-v1.5, e5-small-v2, all-MiniLM-L6-v2
  - Same lambda_sep sweep: [0.0, 0.1, 0.3, 1.0]
  - Same datasets: clinc, dbpedia_classes + 2 held-out (agnews, trec)
  - 3 seeds per condition
  - Compute both S (scalar) and Q (multivariate Fisher)
  - Test: same S/Q -> same quality across architectures?

Success criteria:
  1. Monotonic dose-response on at least 2/3 models
  2. Cross-model R2(Q, quality) > 0.4 (pooled across models)
  3. Mediation holds: adding model indicator doesn't improve R2 beyond Q

Usage:
    python -u src/cgp_week3_cross_arch.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset

# ── Models ──────────────────────────────────────────────────────────

MODELS = {
    "bge-small": {
        "hf_path": "BAAI/bge-small-en-v1.5",
        "type": "encoder",
        "hidden_dim": 384,
        "num_layers": 12,
    },
    "e5-small": {
        "hf_path": "intfloat/e5-small-v2",
        "type": "encoder",
        "hidden_dim": 384,
        "num_layers": 12,
    },
    "minilm": {
        "hf_path": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "encoder",
        "hidden_dim": 384,
        "num_layers": 6,
    },
}

# ── Configuration ────────────────────────────────────────────────────

LAMBDA_SEP_VALUES = [0.0, 0.1, 0.3, 1.0]
SEEDS = [42, 123, 456]
EVAL_DATASETS = ["clinc", "dbpedia_classes", "agnews", "trec"]
STEPS = 500
EVAL_SAMPLES = 800
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 16
MAX_SEQ_LEN = 128
LR = 2e-4


# ── LoRA for Encoder Models ─────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.original = original
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, original.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, r))

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


def apply_lora_encoder(model, r, alpha):
    """Apply LoRA to encoder model query/value projections."""
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


# ── Loss Functions ───────────────────────────────────────────────────

def contrastive_loss(embeddings, labels, temperature=0.05):
    """NT-Xent contrastive loss."""
    emb = F.normalize(embeddings.float(), dim=-1)
    sim = emb @ emb.T / temperature
    n = sim.size(0)

    # Create positive mask
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = labels_eq & ~torch.eye(n, dtype=torch.bool, device=sim.device)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=sim.device, requires_grad=True)

    # Log-softmax over negatives
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

    # Mean of positive log-probs
    loss = -(log_prob * mask).sum() / mask.sum()
    return loss


def class_separation_loss(embeddings, labels, eps=1e-6):
    """Fisher-like class separation regularizer."""
    emb = F.normalize(embeddings.float(), dim=-1)
    unique_labels = labels.unique()

    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    centroids = []
    intra_dists = []

    for lbl in unique_labels:
        mask = labels == lbl
        class_emb = emb[mask]
        if len(class_emb) < 2:
            continue
        centroid = class_emb.mean(dim=0)
        centroids.append(centroid)
        dists = torch.norm(class_emb - centroid.unsqueeze(0), dim=-1)
        intra_dists.append(dists.mean())

    if len(centroids) < 2 or len(intra_dists) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    centroids_t = torch.stack(centroids)
    mean_intra = torch.stack(intra_dists).mean()
    inter_dists = torch.cdist(centroids_t.unsqueeze(0), centroids_t.unsqueeze(0)).squeeze(0)
    triu_mask = torch.triu(torch.ones_like(inter_dists), diagonal=1).bool()
    mean_inter = inter_dists[triu_mask].mean()

    return mean_intra / (mean_inter + eps)


# ── Evaluation Metrics ───────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, tokenizer, texts, device, max_len=128, batch_size=64):
    """Extract [CLS] embeddings from encoder model."""
    model.eval()
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=max_len, return_tensors="pt").to(device)
        out = model(**enc)
        # Use [CLS] token embedding
        embs = out.last_hidden_state[:, 0, :]
        all_embs.append(embs.cpu().float())

    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def compute_metrics(embeddings, l0_labels, l1_labels, k=5):
    """Compute kNN accuracy and geometric metrics."""
    emb = F.normalize(embeddings, dim=-1)
    n = emb.shape[0]

    # kNN
    sim = emb @ emb.T
    sim.fill_diagonal_(-1e9)
    _, topk_idx = sim.topk(k, dim=1)

    # L0 kNN accuracy
    l0_t = torch.tensor(l0_labels)
    l0_neighbors = l0_t[topk_idx]
    knn_l0_preds = l0_neighbors.mode(dim=1).values
    knn_l0 = (knn_l0_preds == l0_t).float().mean().item()

    # L1 kNN accuracy
    l1_t = torch.tensor(l1_labels)
    l1_neighbors = l1_t[topk_idx]
    knn_l1_preds = l1_neighbors.mode(dim=1).values
    knn_l1 = (knn_l1_preds == l1_t).float().mean().item()

    # Class separation (L1 level)
    unique_l1 = l1_t.unique()
    centroids = []
    intra_dists = []
    for lbl in unique_l1:
        mask = l1_t == lbl
        cls_emb = emb[mask]
        if len(cls_emb) < 2:
            continue
        centroid = cls_emb.mean(dim=0)
        centroids.append(centroid)
        dists = torch.norm(cls_emb - centroid.unsqueeze(0), dim=-1)
        intra_dists.append(dists.mean().item())

    if len(centroids) >= 2:
        cent_t = torch.stack(centroids)
        inter = torch.cdist(cent_t.unsqueeze(0), cent_t.unsqueeze(0)).squeeze(0)
        triu = torch.triu(torch.ones_like(inter), diagonal=1).bool()
        mean_inter = inter[triu].mean().item()
        mean_intra = np.mean(intra_dists)
        class_sep = mean_inter / (mean_intra + 1e-8)
    else:
        class_sep = 0.0
        mean_inter = 0.0
        mean_intra = 0.0

    # Multivariate Fisher Q = tr(Sigma_between) / tr(Sigma_within)
    global_mean = emb.mean(dim=0)
    sigma_between = torch.zeros(emb.shape[1], emb.shape[1])
    sigma_within = torch.zeros(emb.shape[1], emb.shape[1])

    for lbl in unique_l1:
        mask = l1_t == lbl
        cls_emb = emb[mask]
        if len(cls_emb) < 2:
            continue
        centroid = cls_emb.mean(dim=0)
        diff_b = (centroid - global_mean).unsqueeze(1)
        sigma_between += len(cls_emb) * (diff_b @ diff_b.T)

        centered = cls_emb - centroid.unsqueeze(0)
        sigma_within += centered.T @ centered

    sigma_between /= n
    sigma_within /= n

    trace_between = sigma_between.trace().item()
    trace_within = max(sigma_within.trace().item(), 1e-8)
    fisher_q = trace_between / trace_within

    # Centroid regularity (kappa) and composite G
    C_classes = len(centroids) if len(centroids) >= 2 else 0
    d_dim = emb.shape[1]
    kappa = 0.0
    composite_G = 0.0

    if len(centroids) >= 2:
        cent_t_full = torch.stack(centroids)
        pairwise_sq = torch.cdist(cent_t_full.unsqueeze(0),
                                   cent_t_full.unsqueeze(0)).squeeze(0) ** 2
        triu_kappa = torch.triu(torch.ones_like(pairwise_sq), diagonal=1).bool()
        pairwise_vals = pairwise_sq[triu_kappa]
        d_min_sq = pairwise_vals.min().item()
        d_avg_sq = pairwise_vals.mean().item()
        kappa = d_min_sq / max(d_avg_sq, 1e-8)
        # G = kappa * C * d * Q / (C-1)
        composite_G = kappa * C_classes * d_dim * fisher_q / max(C_classes - 1, 1)

    # Anisotropy
    _, S, _ = torch.linalg.svd(emb, full_matrices=False)
    S = S / S.sum()
    anisotropy = S[0].item()
    effective_rank = torch.exp(-torch.sum(S * torch.log(S + 1e-10))).item()

    return {
        "knn_l0": knn_l0,
        "knn_l1": knn_l1,
        "class_sep_l1": class_sep,
        "fisher_q": fisher_q,
        "trace_between": trace_between,
        "trace_within": trace_within,
        "kappa": kappa,
        "composite_G": composite_G,
        "n_classes": C_classes,
        "dim": d_dim,
        "anisotropy": anisotropy,
        "effective_rank": effective_rank,
        "mean_inter": mean_inter,
        "mean_intra": mean_intra,
    }


# ── Training ─────────────────────────────────────────────────────────

def train_encoder_with_sep(model, tokenizer, texts, labels, device,
                           lambda_sep=0.0, num_steps=500):
    """Train encoder model with contrastive + class-sep regularizer."""
    # Only train LoRA params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    label_set = sorted(set(labels))
    label_map = {l: i for i, l in enumerate(label_set)}
    label_ids = [label_map[l] for l in labels]

    n = len(texts)
    model.train()

    avg_primary = avg_sep = 0.0
    count = 0

    for step in range(num_steps):
        # Sample batch
        idx = np.random.choice(n, BATCH_SIZE, replace=False)
        batch_texts = [texts[i] for i in idx]
        batch_labels = torch.tensor([label_ids[i] for i in idx], device=device)

        enc = tokenizer(batch_texts, padding=True, truncation=True,
                       max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)

        out = model(**enc)
        emb = out.last_hidden_state[:, 0, :]  # [CLS]

        # Contrastive loss
        loss_primary = contrastive_loss(emb, batch_labels)

        # Class separation regularizer
        if lambda_sep > 0:
            loss_sep = class_separation_loss(emb, batch_labels)
        else:
            loss_sep = torch.tensor(0.0, device=device)

        loss = loss_primary + lambda_sep * loss_sep

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_primary += loss_primary.item()
        avg_sep += loss_sep.item()
        count += 1

        if (step + 1) % 200 == 0:
            print(f"    Step {step+1}/{num_steps}: primary={avg_primary/count:.4f} "
                  f"sep={avg_sep/count:.4f}")

    return avg_primary / max(count, 1), avg_sep / max(count, 1)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Lambda_sep: {LAMBDA_SEP_VALUES}")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {EVAL_DATASETS}")

    all_results = []

    for model_name, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name} ({model_cfg['hf_path']})")
        print(f"{'='*60}")

        # Load training data
        train_data = load_hierarchical_dataset("clinc", split="train", max_samples=5000)
        texts = [s.text for s in train_data.samples]
        l1_labels = [s.level1_label for s in train_data.samples]

        for lam_sep, seed in product(LAMBDA_SEP_VALUES, SEEDS):
            print(f"\n  --- {model_name} lam_sep={lam_sep} seed={seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_path"])
            base_model = AutoModel.from_pretrained(
                model_cfg["hf_path"], torch_dtype=torch.float16
            ).to(device)

            if lam_sep == 0.0 and seed == SEEDS[0]:
                # Baseline (no training) - only need once
                model = base_model
                train_time = 0.0
                avg_primary = avg_sep = 0.0
                is_baseline = True
            else:
                model = apply_lora_encoder(base_model, LORA_R, LORA_ALPHA)
                t0 = time.time()
                avg_primary, avg_sep = train_encoder_with_sep(
                    model, tokenizer, texts, l1_labels, device,
                    lambda_sep=lam_sep, num_steps=STEPS,
                )
                train_time = time.time() - t0
                print(f"  Trained in {train_time:.1f}s")
                is_baseline = False

            # Evaluate
            eval_results = {}
            for ds_name in EVAL_DATASETS:
                try:
                    ds = load_hierarchical_dataset(ds_name, split="test",
                                                  max_samples=EVAL_SAMPLES)
                    eval_texts = [s.text for s in ds.samples]
                    eval_l0 = [s.level0_label for s in ds.samples]
                    eval_l1 = [s.level1_label for s in ds.samples]

                    embs = extract_embeddings(model, tokenizer, eval_texts, device)
                    metrics = compute_metrics(embs, eval_l0, eval_l1)
                    eval_results[ds_name] = metrics

                    print(f"    {ds_name}: knn_l0={metrics['knn_l0']:.3f} "
                          f"knn_l1={metrics['knn_l1']:.3f} "
                          f"sep={metrics['class_sep_l1']:.3f} "
                          f"Q={metrics['fisher_q']:.3f}")
                except Exception as e:
                    print(f"    {ds_name}: ERROR - {e}")
                    eval_results[ds_name] = {"error": str(e)}

            result = {
                "model": model_name,
                "lambda_sep": lam_sep,
                "seed": seed,
                "is_baseline": is_baseline,
                "avg_primary_loss": float(avg_primary),
                "avg_sep_loss": float(avg_sep),
                "train_time_sec": float(train_time),
                "eval": eval_results,
            }
            all_results.append(result)

            del model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save
    output = {
        "experiment": "CGP Week 3: Cross-Architecture Replication",
        "models": list(MODELS.keys()),
        "lambda_sep_values": LAMBDA_SEP_VALUES,
        "seeds": SEEDS,
        "eval_datasets": EVAL_DATASETS,
        "results": all_results,
    }

    out_path = RESULTS_DIR / "cgp_week3_cross_arch.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
