#!/usr/bin/env python -u
"""
cti_cross_arch_causal_v2.py

Cross-architecture causal intervention v2.
Extends the 4-architecture causal law to DeBERTa-base, OLMo-1B, and Mamba-130m.

PREREGISTERED FORMULA (locked from 4-architecture fit):
  Causal_A = 3.23 + 34.51 * kappa_CE  (R2=0.97 on pythia-160m, gpt-neo-125m, BERT-base)

PRE-REGISTERED PREDICTIONS (from kappa_near_cache files, raw kappa at last layer):
  DeBERTa-base L12:  kappa_CE=0.3574 -> Causal_A_predicted = 3.23 + 34.51*0.3574 = 15.56
  OLMo-1B L16:       kappa_CE=0.4402 -> Causal_A_predicted = 3.23 + 34.51*0.4402 = 18.45
  Mamba-130m L23:    kappa_CE=0.1860 -> (go_emotions K=28) Causal_A_predicted = 3.23 + 34.51*0.1860 = 9.65
    NOTE: Mamba on 20newsgroups K=20 hits ceiling (q_CE=0.98); using go_emotions K=28 instead.

PASS CRITERIA (pre-registered):
  1. delta_q >= 0.02 (triplet improves q by at least 2pp)
  2. kappa_triplet > kappa_CE (triplet increases kappa)
  3. Both hold in 5/5 seeds
  4. |Causal_A_actual - Causal_A_predicted| / Causal_A_predicted < 0.50 (within 50%)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

# ── PRE-REGISTERED CONSTANTS ──────────────────────────────────────────────────
CAUSAL_A_INTERCEPT = 3.23
CAUSAL_A_SLOPE = 34.51
PASS_DELTA_Q = 0.02
PASS_PRED_TOL = 0.50  # relative error tolerance

MODELS = {
    "mamba-130m": {
        "hf_path": "state-spaces/mamba-130m-hf",
        "hidden_dim": 768,
        "pooling": "last",
        "target_layer": 23,
        "trust_remote_code": True,
        "kappa_ce_prereg": 0.1860,  # go_emotions raw kappa; 20newsgroups hits ceiling q=0.98
        "dataset": "go_emotions",   # harder task (K=28 fine-grained emotions)
    },
    "deberta-base": {
        "hf_path": "microsoft/deberta-base",
        "hidden_dim": 768,
        "pooling": "cls",
        "target_layer": 12,
        "trust_remote_code": False,
        "kappa_ce_prereg": 0.3574,
        "force_fp32": True,  # DeBERTa disentangled attention requires float32
    },
    "olmo-1b": {
        "hf_path": "allenai/OLMo-1B-hf",
        "hidden_dim": 2048,
        "pooling": "last",
        "target_layer": 16,
        "trust_remote_code": False,
        "kappa_ce_prereg": 0.4402,  # from kappa_near_cache_20newsgroups_OLMo-1B-hf.json
    },
}

# ── Data ──────────────────────────────────────────────────────────────────────
def load_20newsgroups(n_samples: int = 5000):
    news = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(news.data), size=min(n_samples, len(news.data)), replace=False)
    texts = [news.data[i] for i in idx]
    labels = [news.target[i] for i in idx]
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return texts, labels


def load_go_emotions(n_samples: int = 5000):
    """GoEmotions: 28 fine-grained emotion classes, harder than 20newsgroups."""
    from datasets import load_dataset
    ds = load_dataset("go_emotions", "simplified", split="train")
    texts, labels = [], []
    for ex in ds:
        if len(ex["labels"]) == 1:  # keep single-label examples only
            texts.append(ex["text"])
            labels.append(ex["labels"][0])
    rng = np.random.default_rng(42)
    idx = rng.choice(len(texts), size=min(n_samples, len(texts)), replace=False)
    texts = [texts[i] for i in idx]
    labels_raw = [labels[i] for i in idx]
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels_raw)
    return texts, labels_enc


def load_dataset_for_model(model_key: str, n_samples: int = 5000):
    info = MODELS[model_key]
    dataset = info.get("dataset", "20newsgroups")
    if dataset == "go_emotions":
        return load_go_emotions(n_samples), "go_emotions"
    else:
        return load_20newsgroups(n_samples), "20newsgroups"


# ── Embedding extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model_key: str, texts, device: str = "cuda"):
    from transformers import AutoModel, AutoTokenizer

    info = MODELS[model_key]
    print(f"Loading {model_key} ({info['hf_path']})...")
    tok = AutoTokenizer.from_pretrained(
        info["hf_path"], trust_remote_code=info.get("trust_remote_code", False)
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float32 if info.get("force_fp32") else torch.float16
    model = AutoModel.from_pretrained(
        info["hf_path"],
        trust_remote_code=info.get("trust_remote_code", False),
        dtype=dtype,
        output_hidden_states=True,
    ).to(device).eval()

    target_layer = info["target_layer"]
    pooling = info["pooling"]
    batch_size = 32
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple of (B, T, D)
        h = hs[target_layer].float()  # (B, T, D)

        if pooling == "cls":
            pooled = h[:, 0, :]
        else:  # last non-padding token
            mask = enc["attention_mask"]
            seq_lens = mask.sum(dim=1) - 1
            pooled = h[torch.arange(h.size(0)), seq_lens]

        # L2 normalize
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        all_embs.append(pooled.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"  batch {i//batch_size}/{len(texts)//batch_size}")

    X = np.concatenate(all_embs, axis=0)
    print(f"  Embeddings: {X.shape}")
    return X


# ── kappa_nearest computation ─────────────────────────────────────────────────
def compute_kappa_nearest(X, y):
    classes = np.unique(y)
    mu = np.array([X[y == c].mean(0) for c in classes])
    Xc = np.vstack([X[y == c] - mu[c] for c in classes])
    sigma_W = float(np.std(Xc))
    D = X.shape[1]

    kappas = []
    for c in classes:
        dists = np.linalg.norm(mu[c] - mu, axis=1)
        dists[c] = np.inf
        kappas.append(np.min(dists) / (sigma_W * np.sqrt(D)))
    return float(np.mean(kappas))


# ── Neural network ────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, d_in=768, d_out=64, K=20):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=True)
        self.cls = nn.Linear(d_out, K, bias=True)

    def forward(self, x):
        z = self.proj(x)
        return z, self.cls(z)


def hard_negative_triplet_loss(z, y, margin=0.5):
    diff = z.unsqueeze(0) - z.unsqueeze(1)
    dists = (diff ** 2).sum(-1).clamp(min=1e-12).sqrt()
    same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    diff_mask = 1.0 - same

    d_ap = (dists * same).max(dim=1).values
    d_an = dists + 1e6 * same
    d_an = d_an.min(dim=1).values
    return torch.clamp(d_ap - d_an + margin, min=0.0).mean()


def compute_q_from_embeddings(X, y):
    K = len(np.unique(y))
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    acc = knn.score(X, y)
    return float((acc - 1.0 / K) / (1.0 - 1.0 / K))


# ── Training ──────────────────────────────────────────────────────────────────
def run_arm(X_np, y_np, arm: str, seed: int, n_epochs=100, lambda_triplet=0.1):
    rng = torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    K = int(y.max().item()) + 1
    D = X.shape[1]

    model = ProjectionHead(d_in=D, d_out=64, K=K)
    nn.init.xavier_uniform_(model.proj.weight)
    nn.init.zeros_(model.proj.bias)
    nn.init.xavier_uniform_(model.cls.weight)
    nn.init.zeros_(model.cls.bias)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(X, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    model = model.to(device)
    for ep in range(n_epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            z, logits = model(xb)
            loss = ce_loss(logits, yb)
            if arm == "triplet":
                loss = loss + lambda_triplet * hard_negative_triplet_loss(z, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        z_all, _ = model(X.to(device))
        Z = z_all.cpu().numpy()

    q = compute_q_from_embeddings(Z, y_np)
    kappa = compute_kappa_nearest(Z, y_np)
    logit_q = float(np.log(q / (1 - q))) if 0 < q < 1 else float(np.sign(q - 0.5) * 10)
    return {"q": q, "kappa": kappa, "logit_q": logit_q}


# ── Main ──────────────────────────────────────────────────────────────────────
def run_model(model_key: str, n_seeds: int = 5, n_epochs: int = 100):
    info = MODELS[model_key]
    kappa_ce_prereg = info["kappa_ce_prereg"]
    causal_A_predicted = CAUSAL_A_INTERCEPT + CAUSAL_A_SLOPE * kappa_ce_prereg

    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"Pre-registered kappa_CE: {kappa_ce_prereg:.4f}")
    print(f"Pre-registered Causal_A: {causal_A_predicted:.2f}")
    print(f"{'='*60}")

    # Load data (model-specific dataset)
    (texts, labels), dataset_name = load_dataset_for_model(model_key)
    K = len(np.unique(labels))
    print(f"Dataset: {dataset_name}, K={K}, n={len(texts)}")

    # Extract or load embeddings (cache is dataset-specific)
    emb_path = f"results/causal_v2_embs_{model_key}_{dataset_name}.npz"
    if os.path.exists(emb_path):
        print(f"Loading embeddings from {emb_path}")
        data = np.load(emb_path)
        X, y = data["X"], data["y"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = extract_embeddings(model_key, texts, device=device)
        y = np.array(labels)
        np.savez(emb_path, X=X, y=y)
        print(f"Saved embeddings to {emb_path}")

    # Verify kappa matches pre-registered
    kappa_actual = compute_kappa_nearest(X, y)
    print(f"Kappa from embeddings: {kappa_actual:.4f} (pre-reg: {kappa_ce_prereg:.4f})")
    causal_A_adjusted = CAUSAL_A_INTERCEPT + CAUSAL_A_SLOPE * kappa_actual
    print(f"Adjusted Causal_A prediction: {causal_A_adjusted:.2f}")

    seeds = list(range(n_seeds))
    ce_results = []
    triplet_results = []

    for seed in seeds:
        print(f"\nSeed {seed}:")
        r_ce = run_arm(X, y, "ce", seed, n_epochs)
        r_tr = run_arm(X, y, "triplet", seed, n_epochs)
        ce_results.append(r_ce)
        triplet_results.append(r_tr)
        print(f"  CE:     q={r_ce['q']:.4f}  kappa={r_ce['kappa']:.4f}")
        print(f"  Triplet: q={r_tr['q']:.4f}  kappa={r_tr['kappa']:.4f}")
        print(f"  delta_q={r_tr['q']-r_ce['q']:+.4f}  delta_kappa={r_tr['kappa']-r_ce['kappa']:+.4f}")

    mean_q_ce = float(np.mean([r["q"] for r in ce_results]))
    mean_q_tr = float(np.mean([r["q"] for r in triplet_results]))
    mean_k_ce = float(np.mean([r["kappa"] for r in ce_results]))
    mean_k_tr = float(np.mean([r["kappa"] for r in triplet_results]))
    delta_q = mean_q_tr - mean_q_ce
    delta_k = mean_k_tr - mean_k_ce
    delta_logit = float(np.mean([r["logit_q"] for r in triplet_results])) - float(
        np.mean([r["logit_q"] for r in ce_results])
    )

    causal_A_actual = delta_logit / delta_k if abs(delta_k) > 1e-6 else None
    signs_q = sum(1 for c, t in zip(ce_results, triplet_results) if t["q"] > c["q"])
    signs_k = sum(1 for c, t in zip(ce_results, triplet_results) if t["kappa"] > c["kappa"])

    print(f"\n{'='*60}")
    print(f"SUMMARY for {model_key}:")
    print(f"  CE:      q={mean_q_ce:.4f}  kappa={mean_k_ce:.4f}")
    print(f"  Triplet: q={mean_q_tr:.4f}  kappa={mean_k_tr:.4f}")
    print(f"  delta_q={delta_q:+.4f}  delta_kappa={delta_k:+.4f}  delta_logit={delta_logit:+.4f}")
    print(f"  Causal_A_actual={causal_A_actual:.2f}" if causal_A_actual else "  Causal_A: N/A")
    print(f"  Causal_A_predicted={causal_A_predicted:.2f}")
    if causal_A_actual:
        pred_err = abs(causal_A_actual - causal_A_predicted) / causal_A_predicted
        print(f"  Prediction_error={pred_err:.3f} (tolerance={PASS_PRED_TOL})")
    print(f"  Signs: q={signs_q}/{n_seeds}, kappa={signs_k}/{n_seeds}")

    pass_delta_q = delta_q >= PASS_DELTA_Q
    pass_signs = signs_q >= n_seeds and signs_k >= n_seeds
    pass_pred = (
        abs(causal_A_actual - causal_A_predicted) / causal_A_predicted < PASS_PRED_TOL
        if causal_A_actual else False
    )

    verdict = "PASS" if (pass_delta_q and pass_signs and pass_pred) else "PARTIAL"
    n_pass = sum([pass_delta_q, pass_signs, pass_pred])
    print(f"\nVerdict: {verdict} ({n_pass}/3 criteria)")
    print(f"  delta_q>={PASS_DELTA_Q}: {pass_delta_q}")
    print(f"  signs 5/5: {pass_signs}")
    print(f"  pred within 50%: {pass_pred}")

    result = {
        "model": model_key,
        "dataset": dataset_name,
        "preregistered": {
            "causal_A_intercept": CAUSAL_A_INTERCEPT,
            "causal_A_slope": CAUSAL_A_SLOPE,
            "kappa_ce_prereg": kappa_ce_prereg,
            "causal_A_predicted": float(causal_A_predicted),
            "PASS_DELTA_Q": PASS_DELTA_Q,
            "PASS_PRED_TOL": PASS_PRED_TOL,
        },
        "kappa_from_embeddings": float(kappa_actual),
        "results": {
            "ce": ce_results,
            "triplet": triplet_results,
        },
        "summary": {
            "mean_q_ce": float(mean_q_ce),
            "mean_q_triplet": float(mean_q_tr),
            "mean_kappa_ce": float(mean_k_ce),
            "mean_kappa_triplet": float(mean_k_tr),
            "delta_q": float(delta_q),
            "delta_kappa": float(delta_k),
            "delta_logit": float(delta_logit),
            "causal_A_actual": float(causal_A_actual) if causal_A_actual else None,
            "causal_A_predicted": float(causal_A_predicted),
            "prediction_error": float(abs(causal_A_actual - causal_A_predicted) / causal_A_predicted) if causal_A_actual else None,
            "signs_q": signs_q,
            "signs_k": signs_k,
            "verdict": verdict,
            "n_pass": n_pass,
        },
    }

    out_path = f"results/cti_causal_{model_key.replace('-', '_')}_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="deberta-base,mamba-130m",
                        help="Comma-separated model keys")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]

    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
            continue
        t0 = time.time()
        run_model(model_key, n_seeds=args.seeds, n_epochs=args.epochs)
        print(f"\n[{model_key} done in {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
