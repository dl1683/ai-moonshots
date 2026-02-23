#!/usr/bin/env python -u
"""
K-INDEPENDENCE TEST FOR EQUICORRELATION rho
============================================

If class centroids form a regular simplex in Sigma_W-whitened space (Neural Collapse),
then rho = avg cosine similarity = 0.5 REGARDLESS OF K.

Test: pythia-160m on 3 datasets with different K
  agnews K=4, dbpedia K=14, banking77 K=77

Prediction (regular simplex = Neural Collapse ETF):
  rho ~ 0.45-0.50 for all K
  d_eff_comp = 1/(1-rho) ~ 1.7-2.0 for all K

If rho is approximately constant across K: simplex geometry confirmed.
If rho decreases with K: not a simplex, some other geometry.

Pre-registered: commit 8c55183 (extend of equicorrelation test framework)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCH = "pythia-160m"
MODEL_ID = "EleutherAI/pythia-160m"
N_SAMPLES = 2000
N_PCA = 256

# Datasets with different K
DATASETS = [
    ("agnews", 4),
    ("dbpedia_14", 14),
    ("banking77", 77),
]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_ID}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE)
    model.eval()
    return model, tok


def load_dataset_texts(dataset_name, n_samples):
    from datasets import load_dataset
    import random
    random.seed(42)
    if dataset_name == "agnews":
        ds = load_dataset("ag_news", split="train")
        text_col, label_col = "text", "label"
    elif dataset_name == "dbpedia_14":
        ds = load_dataset("dbpedia_14", split="train")
        text_col, label_col = "content", "label"
    elif dataset_name == "banking77":
        ds = load_dataset("banking77", split="train")
        text_col, label_col = "text", "label"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:n_samples]
    texts = [ds[text_col][i] for i in indices]
    labels = np.array([ds[label_col][i] for i in indices])
    return texts, labels


@torch.no_grad()
def extract_at_layer(model, tok, texts, layer_idx, batch_size=32):
    embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer_idx + 1]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
        embeds.append(pooled.cpu().numpy())
    return np.concatenate(embeds, axis=0)


def find_best_layer_kappa(model, tok, texts, labels, n_layers):
    """Sample 6 layers, pick best kappa."""
    layer_idx = list(np.linspace(0, n_layers - 1, 6, dtype=int))
    best_k, best_l = -1, 0
    for l in layer_idx:
        X = extract_at_layer(model, tok, texts, l)
        classes = np.unique(labels)
        cents = {c: X[labels==c].mean(0) for c in classes if (labels==c).sum() >= 2}
        if len(cents) < len(classes):
            continue
        Xc = np.concatenate([X[labels==c] - cents[c]
                              for c in classes if (labels==c).sum() >= 2])
        trSW = np.sum(Xc**2) / len(Xc)
        from sklearn.metrics import pairwise_distances
        ca = np.array([cents[c] for c in sorted(cents)])
        D = pairwise_distances(ca)
        np.fill_diagonal(D, np.inf)
        dmin = D.min()
        kappa = dmin / (np.sqrt(trSW) + 1e-12)
        if kappa > best_k:
            best_k, best_l = kappa, l
    return best_l, float(best_k)


def compute_rho(X, labels, n_pca=256):
    """Compute competition equicorrelation rho and d_eff_comp=1/(1-rho)."""
    classes = np.unique(labels)
    K = len(classes)
    n, d = X.shape
    centroids = {c: X[labels==c].mean(0).astype(np.float64)
                 for c in classes if (labels==c).sum() >= 2}
    if len(centroids) < K:
        return None, None

    centroid_arr = np.array([centroids[c] for c in sorted(centroids)])  # K x d
    Xc = np.concatenate([(X[labels==c] - centroids[c]).astype(np.float64)
                          for c in classes if (labels==c).sum() >= 2])
    N_tot = len(Xc)

    n_comp = min(n_pca, d, N_tot - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(Xc)
    V = svd.components_.T  # d x n_comp
    lam = (svd.singular_values_**2) / N_tot
    sqrt_lam = np.sqrt(lam + 1e-12)

    classes_sorted = sorted(centroids.keys())
    rho_per_class = []
    for ci, c in enumerate(classes_sorted):
        other = [i for i in range(K) if i != ci]
        deltas = centroid_arr[other] - centroids[c]  # (K-1) x d
        proj = deltas @ V  # (K-1) x n_comp
        wh = proj * sqrt_lam[None, :]
        norms = np.linalg.norm(wh, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        wh_n = wh / norms
        cos_mat = wh_n @ wh_n.T
        off = ~np.eye(K-1, dtype=bool)
        rho_per_class.append(float(cos_mat[off].mean()))

    rho = float(np.mean(rho_per_class))
    rho_std = float(np.std(rho_per_class))
    d_eff = 1.0 / (1.0 - rho) if rho < 1.0 else float("inf")
    return rho, d_eff, rho_std


def main():
    print("K-INDEPENDENCE TEST FOR rho", flush=True)
    print(f"Arch: {ARCH}, N={N_SAMPLES}, n_pca={N_PCA}", flush=True)
    print(f"Prediction: rho ~ 0.45-0.50 for ALL K (regular simplex)", flush=True)
    print(f"d_eff_comp = 1/(1-rho) ~ 1.7-2.0 for ALL K", flush=True)

    model, tok = load_model()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"Model: {n_layers} layers, d={d}", flush=True)

    results = {}
    for (ds_name, K_expected) in DATASETS:
        print(f"\n--- {ds_name} (K={K_expected}) ---", flush=True)
        t0 = time.time()

        texts, labels = load_dataset_texts(ds_name, N_SAMPLES)
        K_actual = len(np.unique(labels))
        print(f"  Loaded {len(texts)} samples, K={K_actual}", flush=True)

        best_layer, best_kappa = find_best_layer_kappa(model, tok, texts, labels, n_layers)
        print(f"  Best layer: {best_layer}, kappa={best_kappa:.4f}", flush=True)

        X = extract_at_layer(model, tok, texts, best_layer)
        X = X.astype(np.float64)

        rho, d_eff, rho_std = compute_rho(X, labels, N_PCA)
        elapsed = time.time() - t0

        print(f"  rho = {rho:.4f} +/- {rho_std:.4f}", flush=True)
        print(f"  d_eff_comp = {d_eff:.4f}", flush=True)
        print(f"  Elapsed: {elapsed:.1f}s", flush=True)

        results[ds_name] = {
            "K": K_actual,
            "rho": float(rho),
            "rho_std_per_class": float(rho_std),
            "d_eff_comp": float(d_eff),
            "best_layer": int(best_layer),
            "kappa_nearest": float(best_kappa),
            "elapsed_s": float(elapsed),
        }

    # Summary
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "="*60, flush=True)
    print("K-INDEPENDENCE SUMMARY", flush=True)
    print("="*60, flush=True)
    ks = [results[d]["K"] for d in results]
    rhos = [results[d]["rho"] for d in results]
    deffs = [results[d]["d_eff_comp"] for d in results]
    for ds in results:
        r = results[ds]
        print(f"  K={r['K']:3d}: rho={r['rho']:.4f}, d_eff_comp={r['d_eff_comp']:.4f}", flush=True)

    rho_range = max(rhos) - min(rhos)
    cv_rho = np.std(rhos) / (np.mean(rhos) + 1e-12)
    cv_deff = np.std(deffs) / (np.mean(deffs) + 1e-12)
    print(f"\n  rho range: {min(rhos):.4f} to {max(rhos):.4f} (range={rho_range:.4f})", flush=True)
    print(f"  CV(rho) = {cv_rho:.4f}, CV(d_eff_comp) = {cv_deff:.4f}", flush=True)
    print(f"  SIMPLEX PREDICTION: rho=0.5 (K-independent)", flush=True)

    # K-independence pass: CV(rho) < 0.15
    k_indep = cv_rho < 0.15
    print(f"  K-independence (CV(rho)<0.15): {'PASS' if k_indep else 'FAIL'}", flush=True)

    out = {
        "experiment": "cti_equicorr_K_sweep",
        "arch": ARCH,
        "n_samples": N_SAMPLES,
        "prediction": "rho ~ 0.5 (regular simplex, K-independent)",
        "results": results,
        "summary": {
            "mean_rho": float(np.mean(rhos)),
            "std_rho": float(np.std(rhos)),
            "cv_rho": float(cv_rho),
            "mean_d_eff_comp": float(np.mean(deffs)),
            "cv_d_eff_comp": float(cv_deff),
            "k_independence_pass": bool(k_indep),
        },
    }

    out_path = RESULTS_DIR / "cti_equicorr_K_sweep.json"
    with open(out_path, "w", encoding="ascii") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
