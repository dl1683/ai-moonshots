"""
Strong Model Kappa Sweep
Tests CTI law (logit(q) = A * kappa * sqrt(d_eff) + C) on stronger embedding models.

Models tested (in capability order):
  Small/Medium embedding models:  bge-small, bge-base, bge-large, bge-m3
  SOTA embedding models:          Qwen3-Embedding-0.6B, e5-large, nomic-embed
  Larger LLM-based:               Qwen3-1.7B, Qwen3-4B (as zero-shot embedding)

Hypothesis: A is UNIVERSAL across capability levels (~1.054).
Prediction: Stronger embedding models have higher kappa AND higher q.
"""

import torch
import numpy as np
import json
import os
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from scipy.special import logit as logit_fn
from scipy import stats

RESULTS_FILE = "results/cti_strong_model_sweep.json"
CACHE_DIR = "results"
A_PREREGISTERED = 1.0535

# Pre-registered A (from LOAO across 7 architectures)
A_RENORM = 1.0535

# Models to test (all via SentenceTransformer)
EMBEDDING_MODELS = [
    # Classic BGE scaling (33M -> 109M -> 335M -> 568M)
    ("BAAI/bge-small-en-v1.5",   "bge-small",   384),
    ("BAAI/bge-base-en-v1.5",    "bge-base",    768),
    ("BAAI/bge-large-en-v1.5",   "bge-large",   1024),
    ("BAAI/bge-m3",              "bge-m3",      1024),  # 568M, multilingual

    # E5 scaling (33M -> 109M -> 335M)
    ("intfloat/e5-small-v2",     "e5-small",    384),
    ("intfloat/e5-base-v2",      "e5-base",     768),
    ("intfloat/e5-large-v2",     "e5-large",    1024),

    # SOTA embedding models
    ("Qwen/Qwen3-Embedding-0.6B",   "Qwen3-Embed-0.6B",   1024),  # SOTA <1B
    ("nomic-ai/nomic-embed-text-v1.5", "nomic-embed-1.5", 768),   # Matryoshka
    ("google/embedding-gemma-308m", "embedding-gemma-308m", 768),  # Google SOTA <500M
]

DATASETS = {
    "agnews":  {"hf_name": "fancyzhx/ag_news",    "text_col": "text",    "label_col": "label", "K": 4,  "n": 1000},
    "dbpedia": {"hf_name": "fancyzhx/dbpedia_14", "text_col": "content", "label_col": "label", "K": 14, "n": 1000},
}


def load_texts_labels(ds_name, cfg):
    """Load dataset texts and labels."""
    print(f"  Loading {ds_name}...")
    if ds_name == "agnews":
        ds = load_dataset(cfg["hf_name"], split="test")
    else:
        ds = load_dataset(cfg["hf_name"], split="test[:2000]")

    texts = ds[cfg["text_col"]][:cfg["n"]]
    labels = ds[cfg["label_col"]][:cfg["n"]]

    # For label_text -> int mapping
    if isinstance(labels[0], str):
        unique = sorted(set(labels))
        l2i = {l: i for i, l in enumerate(unique)}
        labels = [l2i[l] for l in labels]

    return texts, np.array(labels)


def compute_kappa_and_q(X, y, K):
    """Compute kappa_nearest, d_eff_formula, q."""
    d = X.shape[1]
    centroids = np.array([X[y == k].mean(0) for k in range(K)])

    # Within-class covariance (pooled)
    X_c = np.zeros_like(X)
    for k in range(K):
        mask = y == k
        X_c[mask] = X[mask] - centroids[k]
    Sigma_W = (X_c.T @ X_c) / len(X)
    tr_W = float(np.trace(Sigma_W))
    sigma_W_sq = tr_W / d
    sigma_W = float(np.sqrt(sigma_W_sq))

    # Nearest centroid pair
    min_kappa = np.inf
    for i in range(K):
        for j in range(i+1, K):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            kappa_ij = dist / (sigma_W * np.sqrt(d))
            if kappa_ij < min_kappa:
                min_kappa = kappa_ij

    # d_eff_formula: direction-specific
    # Find centroid direction of nearest pair
    min_kappa_pair = (None, None)
    min_dist = np.inf
    for i in range(K):
        for j in range(i+1, K):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            kappa_ij = dist / (sigma_W * np.sqrt(d))
            if kappa_ij < min_kappa * 1.001:
                min_kappa_pair = (i, j)
                min_dist = dist

    if min_kappa_pair[0] is not None:
        i, j = min_kappa_pair
        dir_hat = (centroids[i] - centroids[j])
        dir_hat = dir_hat / (np.linalg.norm(dir_hat) + 1e-10)
        # Within-class variance in centroid direction
        sigma_cdir_sq = float(dir_hat @ Sigma_W @ dir_hat)
        d_eff_formula = tr_W / sigma_cdir_sq if sigma_cdir_sq > 0 else d
    else:
        d_eff_formula = d

    # kNN accuracy
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    n_train = len(X) * 3 // 4
    knn.fit(X[:n_train], y[:n_train])
    acc = knn.score(X[n_train:], y[n_train:])
    q = (acc - 1.0/K) / (1.0 - 1.0/K)

    return float(min_kappa), float(d_eff_formula), float(q), float(tr_W), float(sigma_W)


def run_model_on_dataset(model_name, model_short, ds_name, ds_cfg, batch_size=128):
    """Extract embeddings and compute CTI metrics."""
    cache_path = os.path.join(CACHE_DIR, f"kappa_strong_{ds_name}_{model_short}.json")

    if os.path.exists(cache_path):
        print(f"  [CACHE HIT] {model_short} on {ds_name}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"  Loading {model_short} (model_name={model_name})...")
    try:
        model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"  ERROR loading {model_short}: {e}")
        return None

    texts, labels = load_texts_labels(ds_name, ds_cfg)
    K = ds_cfg["K"]

    print(f"  Encoding {len(texts)} texts with {model_short}...")
    try:
        X = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    except Exception as e:
        print(f"  ERROR encoding: {e}")
        return None

    print(f"  X.shape={X.shape}, computing CTI metrics...")
    kappa, d_eff, q, tr_W, sigma_W = compute_kappa_and_q(X, labels, K)
    logit_q = float(logit_fn(np.clip(q, 1e-6, 1-1e-6)))
    pred_logit = A_RENORM * kappa * np.sqrt(d_eff)  # zero-param prediction (C=0)

    result = {
        "model": model_short,
        "model_name": model_name,
        "dataset": ds_name,
        "K": K,
        "d_embed": int(X.shape[1]),
        "n": len(texts),
        "kappa_nearest": float(kappa),
        "d_eff_formula": float(d_eff),
        "q": float(q),
        "logit_q": float(logit_q),
        "kappa_eff": float(kappa * np.sqrt(d_eff)),
        "pred_logit_zero_C": float(pred_logit),
        "tr_W": float(tr_W),
        "sigma_W": float(sigma_W),
    }

    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Done: kappa={kappa:.4f}, d_eff={d_eff:.2f}, q={q:.3f}, logit_q={logit_q:.3f}")
    del model
    torch.cuda.empty_cache()
    return result


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 70)
    print("Strong Model Kappa Sweep - CTI Universal Law Test")
    print("=" * 70)
    print(f"Models: {len(EMBEDDING_MODELS)}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Pre-registered A = {A_RENORM}")
    print()

    all_results = []

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name} (K={ds_cfg['K']})")
        print(f"{'='*50}")

        for model_name, model_short, embed_dim in EMBEDDING_MODELS:
            print(f"\n  [{model_short}]")
            result = run_model_on_dataset(model_name, model_short, ds_name, ds_cfg)
            if result:
                all_results.append(result)

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for ds_name in DATASETS:
        ds_results = [r for r in all_results if r["dataset"] == ds_name]
        if not ds_results:
            continue

        print(f"\n{ds_name}:")
        print(f"{'Model':25s} {'kappa':>8s} {'d_eff':>8s} {'q':>7s} {'logit_q':>8s} {'pred_logit':>10s} {'resid':>7s}")
        print("-" * 80)

        # Fit C (intercept) for each model - or use zero C
        logit_q_vals = [r["logit_q"] for r in ds_results]
        kappa_eff_vals = [r["kappa_eff"] for r in ds_results]

        if len(logit_q_vals) >= 3:
            C_fitted = np.mean(np.array(logit_q_vals) - A_RENORM * np.array(kappa_eff_vals))
        else:
            C_fitted = 0.0

        for r in sorted(ds_results, key=lambda x: x["kappa_nearest"]):
            pred = A_RENORM * r["kappa_eff"] + C_fitted
            resid = r["logit_q"] - pred
            print(f"{r['model']:25s} {r['kappa_nearest']:8.4f} {r['d_eff_formula']:8.2f} "
                  f"{r['q']:7.3f} {r['logit_q']:8.3f} {pred:10.3f} {resid:7.3f}")

        if len(logit_q_vals) >= 3:
            # R2 with fitted C
            pred_vals = [A_RENORM * r["kappa_eff"] + C_fitted for r in ds_results]
            ss_res = sum((o-p)**2 for o,p in zip(logit_q_vals, pred_vals))
            ss_tot = sum((o - np.mean(logit_q_vals))**2 for o in logit_q_vals)
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
            pearson_r, _ = stats.pearsonr(kappa_eff_vals, logit_q_vals)
            print(f"\n  R2(kappa_eff, logit_q) = {r2:.4f}")
            print(f"  Pearson r = {pearson_r:.4f}")
            print(f"  Fitted C = {C_fitted:.3f}")

    # Cross-model A estimate
    print("\n" + "=" * 70)
    print("CROSS-MODEL A ESTIMATION")
    print("=" * 70)
    if len(all_results) >= 5:
        kappa_eff_all = [r["kappa_eff"] for r in all_results]
        logit_q_all = [r["logit_q"] for r in all_results]
        slope, intercept, r_val, p_val, se = stats.linregress(kappa_eff_all, logit_q_all)
        print(f"  OLS slope (A_empirical) = {slope:.4f} (pre-registered: {A_RENORM})")
        print(f"  OLS intercept (C)       = {intercept:.4f}")
        print(f"  Pearson r               = {r_val:.4f}")
        print(f"  R2                      = {r_val**2:.4f}")
        print(f"  Slope error             = {abs(slope - A_RENORM) / A_RENORM * 100:.1f}%")

    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "records": all_results,
            "A_preregistered": A_RENORM,
            "n_models": len(EMBEDDING_MODELS),
            "n_datasets": len(DATASETS),
        }, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
