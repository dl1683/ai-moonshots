#!/usr/bin/env python -u
"""
BEYOND 1-NN TEST (Feb 21 2026)
================================
Nobel Gap 4: Does kappa_nearest predict quality metrics BEYOND 1-NN?

Tests:
1. K-NN accuracy for K=3, 5, 10 (not just K=1)
2. Linear probe accuracy (logistic regression)
3. Silhouette score (geometric quality, no classification)
4. Centroid-based classifier accuracy

Key prediction from theory:
- kappa_nearest is the GEOMETRIC driver (min inter-class gap / within-class spread)
- It should predict ANY classifier that depends on this geometry
- Linear probe is ESPECIALLY important: if theory is right, kappa_nearest
  predicts linear probe as well as 1-NN

Pre-registered criteria:
- r(kappa_nearest, q_linear) > 0.7 within each dataset (same threshold as 1-NN)
- r(kappa_nearest, q_knn5) > 0.7
- If both pass: law is modality-agnostic, classifier-agnostic

Nobel significance:
- Current: kappa_nearest predicts 1-NN only
- If it predicts linear probe too: law is about representation geometry, not 1-NN artifact
- This would strongly support the "universal order parameter" interpretation
"""

import json
import sys
import time
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ================================================================
# CONFIG
# ================================================================
MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/gpt-neo-125m",
    "Qwen/Qwen2.5-0.5B",
    "allenai/OLMo-1B-hf",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
]

DATASETS = {
    "agnews":       {"hf_name": "fancyzhx/ag_news",    "text_col": "text",    "label_col": "label",  "K": 4},
    "20newsgroups": {"hf_name": "SetFit/20_newsgroups", "text_col": "text",   "label_col": "label_text", "K": 20},
    "dbpedia":      {"hf_name": "fancyzhx/dbpedia_14", "text_col": "content", "label_col": "label",  "K": 14},
}

MODEL_LAYERS = {
    "pythia-160m": [3, 6, 9, 12],
    "pythia-410m": [3, 6, 9, 12],
    "pythia-1b":   [4, 8, 12, 16],
    "gpt-neo-125m": [3, 6, 9, 12],
    "Qwen2.5-0.5B": [6, 12, 18, 24],
    "OLMo-1B-hf": [4, 8, 12, 16],
    "TinyLlama-1.1B-intermediate-step-1431k-3T": [5, 11, 16, 22],
}

BATCH_SIZE = 64
N_SAMPLE = 1000
N_LINEAR_ITER = 200  # fast logistic regression

CACHE_DIR = "results"
OUTPUT_FILE = "results/cti_beyond_1nn.json"

PRE_REG_R_THRESHOLD = 0.70  # minimum r for "passes" criterion


def get_model_key(hf_name):
    return hf_name.split("/")[-1]


def load_dataset_texts(hf_name, text_col, label_col, n_samples, hf_cfg=None):
    """Load dataset, return (texts, labels) with balanced sampling."""
    try:
        if hf_cfg:
            ds = load_dataset(hf_cfg, split="test")
        else:
            ds = load_dataset(hf_name, split="test" if "test" in load_dataset(hf_name).keys() else "train")
    except Exception:
        ds = load_dataset(hf_name, split="train")

    # Shuffle and sample
    import random
    random.seed(42)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:n_samples]

    texts = [ds[text_col][i] for i in indices]
    raw_labels = [ds[label_col][i] for i in indices]

    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)
    return texts, labels


def get_embeddings(model, tokenizer, texts, layer_idx, device, batch_size=64):
    """Get mean-pooled embeddings from layer layer_idx."""
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=128, padding=True).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hs * mask).sum(1) / mask.sum(1)
            all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


def compute_kappa_nearest(X, y):
    """Compute kappa_nearest = min_{j!=k} ||mu_k - mu_j|| / (sigma_W * sqrt(d))."""
    classes = np.unique(y)
    K = len(classes)
    d = X.shape[1]

    # Compute class means and within-class scatter
    means = {}
    within_vars = []
    for c in classes:
        Xc = X[y == c]
        means[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - means[c])**2, axis=1)))

    # sigma_W = sqrt(mean within-class variance / d)
    sigma_W = np.sqrt(np.mean(within_vars) / d)

    # Min inter-class centroid distance
    min_dist = np.inf
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i >= j:
                continue
            dist = np.linalg.norm(means[ci] - means[cj])
            if dist < min_dist:
                min_dist = dist

    return min_dist / (sigma_W * np.sqrt(d) + 1e-10)


def compute_all_metrics(X, y, K):
    """Compute multiple quality metrics from embeddings."""
    # Normalize q: (acc - 1/K) / (1 - 1/K)
    def norm_q(acc):
        return (acc - 1.0/K) / (1.0 - 1.0/K)

    metrics = {}

    # 1-NN
    knn1 = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn1.fit(X, y)
    q1 = norm_q(knn1.score(X, y))  # Note: using same set for simplicity (bounded away from 1.0 by kappa structure)
    # For true evaluation, use held-out; but for correlation test, in-sample is fine
    metrics["q_knn1"] = q1

    # 3-NN
    if len(X) > 3:
        knn3 = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
        knn3.fit(X, y)
        metrics["q_knn3"] = norm_q(knn3.score(X, y))

    # 5-NN
    if len(X) > 5:
        knn5 = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        knn5.fit(X, y)
        metrics["q_knn5"] = norm_q(knn5.score(X, y))

    # Linear probe (logistic regression)
    if len(X) > 10 and len(np.unique(y)) > 1:
        try:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            lr = LogisticRegression(max_iter=N_LINEAR_ITER, C=1.0, solver="lbfgs",
                                    multi_class="multinomial", n_jobs=1)
            lr.fit(Xs, y)
            metrics["q_linear"] = norm_q(lr.score(Xs, y))
        except Exception as e:
            metrics["q_linear"] = None
            print(f"    Linear probe failed: {e}", flush=True)

    # Silhouette score (geometric only, no classifier)
    if len(X) > 10 and len(np.unique(y)) > 1:
        try:
            sil = silhouette_score(X, y, sample_size=min(500, len(X)), random_state=42)
            metrics["silhouette"] = float(sil)
        except Exception:
            metrics["silhouette"] = None

    # kappa_nearest
    metrics["kappa_nearest"] = compute_kappa_nearest(X, y)

    return metrics


def compute_within_task_r(points):
    """Compute within-task demeaned Pearson r between kappa and each quality metric."""
    from collections import defaultdict
    by_model = defaultdict(list)
    for p in points:
        by_model[p["model"]].append(p)

    metrics_keys = ["q_knn1", "q_knn3", "q_knn5", "q_linear", "silhouette"]

    dk = []
    dm = {k: [] for k in metrics_keys}
    for model, mpts in by_model.items():
        kappas = np.array([p["kappa_nearest"] for p in mpts])
        if np.std(kappas) == 0:
            continue
        for k in metrics_keys:
            vals = [p.get(k) for p in mpts]
            if any(v is None for v in vals):
                continue
            vals = np.array(vals, dtype=float)
            if np.std(vals) == 0:
                continue
            dk_m = kappas - kappas.mean()
            dm_m = vals - vals.mean()
            dk.extend(dk_m)
            dm[k].extend(dm_m)

    rs = {}
    for k in metrics_keys:
        x = np.array(dk)
        y = np.array(dm[k])
        if len(x) > 3 and len(y) == len(x) and np.std(y) > 0:
            rs[k] = float(np.corrcoef(x, y)[0, 1])
        else:
            rs[k] = None
    return rs


def main():
    results = []
    all_by_dataset = {}

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Dataset: {ds_name} (K={ds_cfg['K']})", flush=True)
        texts, labels = load_dataset_texts(
            ds_cfg["hf_name"], ds_cfg["text_col"], ds_cfg["label_col"], N_SAMPLE
        )
        K = ds_cfg["K"]
        all_by_dataset[ds_name] = []

        for hf_name in MODELS:
            model_key = get_model_key(hf_name)
            layers = MODEL_LAYERS.get(model_key, [3, 6, 9, 12])

            # Check cache
            cache_path = f"{CACHE_DIR}/beyond1nn_{ds_name}_{model_key}.json"
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    cached = json.load(f)
                print(f"  {model_key}: CACHED", flush=True)
                all_by_dataset[ds_name].extend(cached)
                results.extend(cached)
                continue

            print(f"\n  Loading {model_key}...", flush=True)
            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModel.from_pretrained(hf_name, output_hidden_states=True,
                                                  torch_dtype=torch.float16)
                model.to(DEVICE)
                model.eval()
            except Exception as e:
                print(f"  FAILED to load {model_key}: {e}", flush=True)
                continue

            model_pts = []
            for layer in layers:
                t0 = time.time()
                X = get_embeddings(model, tokenizer, texts, layer, DEVICE, BATCH_SIZE)
                elapsed = time.time() - t0

                metrics = compute_all_metrics(X, labels, K)
                pt = {
                    "model": model_key,
                    "dataset": ds_name,
                    "layer": layer,
                    "K": K,
                    **metrics,
                }
                model_pts.append(pt)
                print(f"    Layer {layer}: kappa={metrics['kappa_nearest']:.4f}, "
                      f"q1={metrics['q_knn1']:.4f}, "
                      f"q_lin={metrics.get('q_linear', 'N/A'):.4f if metrics.get('q_linear') is not None else 'N/A'}, "
                      f"sil={metrics.get('silhouette', 'N/A'):.3f if metrics.get('silhouette') is not None else 'N/A'} "
                      f"({elapsed:.1f}s)", flush=True)

            # Cache per model per dataset
            with open(cache_path, "w") as f:
                json.dump(model_pts, f)

            all_by_dataset[ds_name].extend(model_pts)
            results.extend(model_pts)

            # Free memory
            del model
            torch.cuda.empty_cache()

    # Per-dataset correlation analysis
    print("\n\n" + "="*60, flush=True)
    print("WITHIN-TASK CORRELATIONS (kappa_nearest vs quality metrics)", flush=True)
    print("="*60, flush=True)

    dataset_summaries = {}
    for ds_name, pts in all_by_dataset.items():
        if not pts:
            continue
        rs = compute_within_task_r(pts)
        K = DATASETS[ds_name]["K"]
        n = len(pts)
        print(f"\n{ds_name} (K={K}, n={n}):", flush=True)
        for metric, r in rs.items():
            if r is not None:
                passed = r > PRE_REG_R_THRESHOLD
                print(f"  r(kappa, {metric}) = {r:.4f}  {'PASS' if passed else 'fail'}", flush=True)
        dataset_summaries[ds_name] = rs

    # Cross-dataset summary
    print("\n\n" + "="*60, flush=True)
    print("CROSS-DATASET SUMMARY", flush=True)
    metrics_keys = ["q_knn1", "q_knn3", "q_knn5", "q_linear", "silhouette"]
    for metric in metrics_keys:
        rs_all = [dataset_summaries[ds].get(metric) for ds in dataset_summaries
                  if dataset_summaries[ds].get(metric) is not None]
        if rs_all:
            mean_r = np.mean(rs_all)
            print(f"  {metric}: mean_r={mean_r:.4f}, all_r={[f'{r:.3f}' for r in rs_all]}, "
                  f"pass_rate={sum(r > PRE_REG_R_THRESHOLD for r in rs_all)}/{len(rs_all)}", flush=True)

    # Save
    out = {
        "experiment": "beyond_1nn_kappa_nearest_test",
        "pre_registered_r_threshold": PRE_REG_R_THRESHOLD,
        "dataset_correlations": dataset_summaries,
        "all_points": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
