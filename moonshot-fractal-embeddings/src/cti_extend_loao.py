#!/usr/bin/env python -u
"""
EXTEND LOAO: Add GPT-2 and BERT to Cross-Architecture Universality Test
========================================================================
GPT-2 = causal decoder (12L, 768d, OpenAI original)
BERT  = bidirectional encoder (12L, 768d, masked LM)

These add two NEW architectural paradigms to the existing 5:
  GPT-NeoX (Pythia), GPT-Neo, Qwen, OLMo, LLaMA (TinyLlama)
  + GPT-2 (original GPT architecture)
  + BERT (encoder-only, VERY different training objective)

If alpha ~ 1.5 for BERT (encoder) as well as all decoders, it's a stronger
universality claim spanning encoder/decoder paradigms.
"""

import json
import sys
import os
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

DEVICE_STR = "cuda"
import torch
DEVICE = torch.device(DEVICE_STR if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# ================================================================
# CONFIG
# ================================================================
NEW_MODELS = {
    "gpt2": {
        "name": "gpt2",
        "layers": [3, 6, 9, 12],
        "is_encoder": False,
    },
    "bert-base-uncased": {
        "name": "bert-base-uncased",
        "layers": [4, 8, 10, 12],
        "is_encoder": True,
    },
}

DATASETS = ["agnews", "dbpedia", "20newsgroups", "go_emotions"]
DATASET_K = {"agnews": 4, "dbpedia": 14, "20newsgroups": 20, "go_emotions": 28}
DATASET_TEXTS_N = {"agnews": 2000, "dbpedia": 2000, "20newsgroups": 2000, "go_emotions": 5000}

BATCH_SIZE = 64
CACHE_DIR = "results"


# ================================================================
# DATA LOADING (same as universal test)
# ================================================================
def load_dataset_texts(dataset_name, n_sample):
    """Load text/label pairs."""
    if dataset_name == "agnews":
        from datasets import load_dataset
        ds = load_dataset("ag_news", split="test")
        texts = [x["text"] for x in ds]
        labels = [x["label"] for x in ds]
    elif dataset_name == "dbpedia":
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/dbpedia_14", split="test")
        texts = [x["content"] for x in ds]
        labels = [x["label"] for x in ds]
    elif dataset_name == "20newsgroups":
        from sklearn.datasets import fetch_20newsgroups
        bunch = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
        texts = bunch.data
        labels = list(bunch.target)
    elif dataset_name == "go_emotions":
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
        texts, labels = [], []
        for x in ds:
            if len(x["labels"]) == 1:
                texts.append(x["text"])
                labels.append(x["labels"][0])

    rng = np.random.default_rng(42)
    n = min(n_sample, len(texts))
    idx = rng.choice(len(texts), n, replace=False)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    return texts, np.array(labels)


# ================================================================
# EMBEDDING EXTRACTION
# ================================================================
def get_embeddings_at_layers(model_name, texts, labels, layers, is_encoder=False):
    """Extract mean-pooled embeddings at specified layers."""
    print(f"  Loading model: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()

    all_layer_embs = defaultdict(list)
    all_labels = []

    print(f"  Extracting embeddings ({len(texts)} texts, {len(layers)} layers)...", flush=True)
    t0 = time.time()

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=128
        ).to(DEVICE)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16) if DEVICE.type == "cuda" else torch.no_grad():
                outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of (n_layers+1,) tensors
        mask = inputs["attention_mask"].unsqueeze(-1).float()

        for layer_idx in layers:
            if layer_idx < len(hidden_states):
                h = hidden_states[layer_idx]  # (B, seq, d)
                emb = (h * mask).sum(1) / mask.sum(1)  # (B, d)
                all_layer_embs[layer_idx].append(emb.cpu().float().numpy())

        all_labels.extend(batch_labels.tolist() if hasattr(batch_labels, 'tolist') else list(batch_labels))

    del model
    torch.cuda.empty_cache()

    print(f"  Done in {int(time.time()-t0)}s", flush=True)

    result = {}
    for l in layers:
        if all_layer_embs[l]:
            result[l] = np.vstack(all_layer_embs[l])
        else:
            result[l] = None

    return result, np.array(all_labels)


# ================================================================
# kappa_nearest and q computation (same as universal test)
# ================================================================
def compute_kappa_nearest(embeddings, labels, K, subsample=500):
    X, y = embeddings, labels
    classes = np.unique(y)
    if len(classes) < 2:
        return None, None
    K = len(classes)

    if len(X) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), subsample, replace=False)
        X, y = X[idx], y[idx]

    mu = {}
    for k in classes:
        idx_k = (y == k)
        if idx_k.sum() < 2:
            continue
        mu[k] = X[idx_k].mean(0)

    if len(mu) < 2:
        return None, None

    within_var = 0.0
    n_total = 0
    for k, mean_k in mu.items():
        idx_k = (y == k)
        Xk = X[idx_k]
        within_var += np.sum((Xk - mean_k)**2)
        n_total += len(Xk)

    sigma_W = float(np.sqrt(within_var / (n_total * X.shape[1])))
    if sigma_W < 1e-10:
        return None, None

    all_kappa = []
    for k, mean_k in mu.items():
        min_dist = np.inf
        for j, mean_j in mu.items():
            if j == k:
                continue
            dist_kj = float(np.linalg.norm(mean_k - mean_j))
            if dist_kj < min_dist:
                min_dist = dist_kj
        kappa_k = min_dist / (sigma_W * np.sqrt(X.shape[1]))
        all_kappa.append(kappa_k)

    return float(np.mean(all_kappa)), float(np.min(all_kappa))


def compute_knn_q(embeddings, labels, K, subsample=1000):
    if len(embeddings) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), subsample, replace=False)
        X, y = embeddings[idx], labels[idx]
    else:
        X, y = embeddings, labels

    # Filter rare classes
    counts = Counter(y.tolist())
    valid_set = {lbl for lbl, cnt in counts.items() if cnt >= 2}
    if len(valid_set) < 2:
        return None
    mask = np.array([l in valid_set for l in y])
    X, y = X[mask], y[mask]
    K_eff = len(valid_set)

    # Filter NaN
    valid = np.isfinite(X).all(axis=1)
    X, y = X[valid], y[valid]
    if len(X) < 10:
        return None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = (acc - 1.0/K_eff) / (1.0 - 1.0/K_eff)
    return float(q)


# ================================================================
# LOAO ANALYSIS
# ================================================================
def run_loao(all_points, label=""):
    from numpy.linalg import lstsq

    models_uniq = sorted(set(p['model'] for p in all_points))
    arch_map = {m: m for m in models_uniq}  # for display

    kappa_arr = np.array([p['kappa_nearest'] for p in all_points])
    logKm1_arr = np.array([p['logKm1'] for p in all_points])
    logit_q_arr = np.array([p['logit_q'] for p in all_points])

    # Global fit
    X = np.column_stack([kappa_arr, logKm1_arr, np.ones(len(all_points))])
    coef, _, _, _ = lstsq(X, logit_q_arr, rcond=None)
    pred = X @ coef
    r2 = 1 - np.var(pred - logit_q_arr) / np.var(logit_q_arr)
    print(f"  Global fit: alpha={coef[0]:.4f}  beta={coef[1]:.4f}  C={coef[2]:.4f}  R2={r2:.4f}")

    # LOAO (by architecture family)
    arch_labels = set()
    for p in all_points:
        m = p['model']
        if 'pythia' in m:
            arch_labels.add('GPT-NeoX')
        elif 'gpt-neo' in m.lower() or m == 'gpt-neo-125m':
            arch_labels.add('GPT-Neo')
        elif 'Qwen' in m:
            arch_labels.add('Qwen')
        elif 'OLMo' in m:
            arch_labels.add('OLMo')
        elif 'TinyLlama' in m or 'llama' in m.lower():
            arch_labels.add('LLaMA')
        elif m == 'gpt2':
            arch_labels.add('GPT-2')
        elif 'bert' in m.lower():
            arch_labels.add('BERT')

    def get_arch(m):
        if 'pythia' in m: return 'GPT-NeoX'
        if 'gpt-neo' in m.lower() or m == 'gpt-neo-125m': return 'GPT-Neo'
        if 'Qwen' in m: return 'Qwen'
        if 'OLMo' in m: return 'OLMo'
        if 'TinyLlama' in m or 'llama' in m.lower(): return 'LLaMA'
        if m == 'gpt2': return 'GPT-2'
        if 'bert' in m.lower(): return 'BERT'
        return m

    # LOAO by architecture
    loao_alphas = []
    loao_betas = []
    print(f"\n  LOAO by architecture family:")
    for arch in sorted(arch_labels):
        train_pts = [p for p in all_points if get_arch(p['model']) != arch]
        test_pts  = [p for p in all_points if get_arch(p['model']) == arch]
        if not train_pts or not test_pts:
            continue

        kappa_tr = np.array([p['kappa_nearest'] for p in train_pts])
        logKm1_tr = np.array([p['logKm1'] for p in train_pts])
        logit_tr = np.array([p['logit_q'] for p in train_pts])
        X_tr = np.column_stack([kappa_tr, logKm1_tr, np.ones(len(train_pts))])
        coef_tr, _, _, _ = lstsq(X_tr, logit_tr, rcond=None)
        loao_alphas.append(coef_tr[0])
        loao_betas.append(coef_tr[1])

        kappa_te = np.array([p['kappa_nearest'] for p in test_pts])
        logKm1_te = np.array([p['logKm1'] for p in test_pts])
        logit_te = np.array([p['logit_q'] for p in test_pts])
        X_te = np.column_stack([kappa_te, logKm1_te, np.ones(len(test_pts))])
        logit_pred = X_te @ coef_tr
        mae = float(np.abs(logit_pred - logit_te).mean())
        print(f"    Hold-out {arch:12s}: alpha={coef_tr[0]:.4f}  beta={coef_tr[1]:.4f}  MAE={mae:.4f}")

    if loao_alphas:
        arr_a = np.array(loao_alphas)
        arr_b = np.array(loao_betas)
        cv_a = np.std(arr_a) / np.abs(np.mean(arr_a))
        cv_b = np.std(arr_b) / np.abs(np.mean(arr_b))
        print(f"  alpha: mean={np.mean(arr_a):.4f}  CV={cv_a:.3f}  {'PASS' if cv_a < 0.25 else 'FAIL'}")
        print(f"  beta:  mean={np.mean(arr_b):.4f}  CV={cv_b:.3f}  {'PASS' if cv_b < 0.25 else 'FAIL'}")
        return {"alpha_mean": float(np.mean(arr_a)), "alpha_cv": float(cv_a),
                "beta_mean": float(np.mean(arr_b)), "beta_cv": float(cv_b),
                "n_architectures": len(loao_alphas)}
    return {}


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("EXTENDING LOAO: Adding GPT-2 and BERT-base")
    print("=" * 70)

    # Load existing points
    existing_path = "results/cti_kappa_nearest_universal.json"
    with open(existing_path) as f:
        existing = json.load(f)
    all_points = list(existing["all_points"])
    print(f"Loaded {len(all_points)} existing points from {len(set(p['model'] for p in all_points))} models")

    new_points = []

    for model_short, model_cfg in NEW_MODELS.items():
        model_name = model_cfg["name"]
        layers = model_cfg["layers"]
        is_encoder = model_cfg["is_encoder"]

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for dataset_name in DATASETS:
            K = DATASET_K[dataset_name]
            cache_path = f"results/kappa_near_cache_{dataset_name}_{model_short}.json"

            # Load from cache if available
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    cached = json.load(f)
                if cached:
                    print(f"  {dataset_name}: Loaded from cache ({len(cached)} pts)")
                    for c in cached:
                        c['model'] = model_short
                    new_points.extend(cached)
                    continue

            # Otherwise compute
            n_sample = DATASET_TEXTS_N[dataset_name]
            print(f"  {dataset_name}: Computing (n={n_sample}, K={K})...", flush=True)

            try:
                texts, y = load_dataset_texts(dataset_name, n_sample)
            except Exception as e:
                print(f"  {dataset_name}: SKIP - {e}", flush=True)
                continue

            try:
                layer_embs, y_emb = get_embeddings_at_layers(model_name, texts, y, layers, is_encoder)
            except Exception as e:
                print(f"  {dataset_name}: embedding error - {e}", flush=True)
                continue

            dataset_points = []
            for layer in layers:
                embs = layer_embs.get(layer)
                if embs is None:
                    continue

                # Filter NaN
                valid = np.isfinite(embs).all(axis=1)
                if (~valid).sum() > 0:
                    print(f"    Layer {layer}: filtering {(~valid).sum()} NaN rows", flush=True)
                embs = embs[valid]
                y_l = y_emb[valid]

                if len(embs) < 20:
                    continue

                q = compute_knn_q(embs, y_l, K)
                if q is None:
                    continue

                kn, km = compute_kappa_nearest(embs, y_l, K)
                if kn is None:
                    continue

                logit_q = float(np.log(q / (1 - q) + 1e-10)) if 0 < q < 1 else (4.0 if q >= 1 else -4.0)
                logKm1 = float(np.log(K - 1))

                pt = {
                    "model": model_short,
                    "dataset": dataset_name,
                    "layer": int(layer),
                    "K": K,
                    "q": float(q),
                    "kappa_nearest": float(kn),
                    "kappa_min": float(km),
                    "logit_q": float(logit_q),
                    "logKm1": float(logKm1),
                }
                dataset_points.append(pt)
                print(f"    Layer {layer}: q={q:.4f}  kappa={kn:.4f}  logit={logit_q:.4f}", flush=True)

            if dataset_points:
                with open(cache_path, "w") as f:
                    json.dump(dataset_points, f, indent=2)
                print(f"  Cached {len(dataset_points)} pts to {cache_path}", flush=True)
                new_points.extend(dataset_points)

    print(f"\nNew points from GPT-2 + BERT: {len(new_points)}")

    # Combine all
    combined = all_points + new_points
    print(f"Total combined: {len(combined)} points from {len(set(p['model'] for p in combined))} models")

    # Run LOAO on combined
    print("\n" + "=" * 70)
    print("LOAO ANALYSIS (ALL MODELS INCLUDING GPT-2 AND BERT)")
    print("=" * 70)

    result = run_loao(combined, label="combined")

    # Save combined results
    out = {
        "experiment": "kappa_nearest_universal_extended",
        "new_models": list(NEW_MODELS.keys()),
        "n_total_points": len(combined),
        "n_models": len(set(p['model'] for p in combined)),
        "all_points": combined,
        "loao_result": result,
    }
    out_path = "results/cti_kappa_nearest_extended.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")

    # Also run per-task-intercept LOAO
    print("\n" + "=" * 70)
    print("PER-TASK-INTERCEPT LOAO (alpha without K collinearity)")
    print("=" * 70)

    from numpy.linalg import lstsq
    ds_list = sorted(set(p['dataset'] for p in combined))
    models_uniq = sorted(set(p['model'] for p in combined))

    def get_arch_combined(m):
        if 'pythia' in m: return 'GPT-NeoX'
        if 'gpt-neo' in m.lower() or m == 'gpt-neo-125m': return 'GPT-Neo'
        if 'Qwen' in m: return 'Qwen'
        if 'OLMo' in m: return 'OLMo'
        if 'TinyLlama' in m or 'llama' in m.lower(): return 'LLaMA'
        if m == 'gpt2': return 'GPT-2'
        if 'bert' in m.lower(): return 'BERT'
        return m

    arch_set = sorted(set(get_arch_combined(p['model']) for p in combined))

    def fit_alpha_per_task(pts_train):
        n = len(pts_train)
        ds_train = sorted(set(p['dataset'] for p in pts_train))
        X = np.zeros((n, 1 + len(ds_train)))
        y_arr = np.zeros(n)
        for i, p in enumerate(pts_train):
            X[i, 0] = p['kappa_nearest']
            X[i, 1 + ds_train.index(p['dataset'])] = 1.0
            y_arr[i] = p['logit_q']
        coef, _, _, _ = lstsq(X, y_arr, rcond=None)
        return coef[0]  # alpha

    loao_alphas_pt = []
    for arch in arch_set:
        train_pts = [p for p in combined if get_arch_combined(p['model']) != arch]
        if not train_pts:
            continue
        a = fit_alpha_per_task(train_pts)
        loao_alphas_pt.append(a)
        print(f"  Hold-out {arch:12s}: alpha (per-task intercept) = {a:.4f}")

    arr_a = np.array(loao_alphas_pt)
    cv_a = np.std(arr_a) / np.abs(np.mean(arr_a))
    print(f"\n  alpha: mean={np.mean(arr_a):.4f}  std={np.std(arr_a):.4f}  CV={cv_a:.3f}  {'PASS' if cv_a < 0.25 else 'FAIL'}")
    print(f"  N architectures: {len(arr_a)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
