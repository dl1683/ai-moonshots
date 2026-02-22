#!/usr/bin/env python -u
"""
KAPPA_NEAREST UNIVERSAL LAW TEST (Feb 21 2026)
================================================
Hypothesis: logit(q_norm) = alpha * kappa_nearest - b_eff * log(K-1) + C_0

where:
  kappa_nearest = min_{j != k} ||mu_k - mu_j|| / sigma_W   (gap to nearest competing class)
  sigma_W = sqrt(tr(S_W) / (n * d_eff))                    (pooled within-class std, per feature)
  OR simpler: sigma_W = sqrt(mean over all within-class distances)

This is DIFFERENT from:
  - kappa_spec = tr(S_B)/tr(S_W): uses ALL class pairs, not just nearest
  - dist_ratio: uses SAMPLE NN distances (not class means)

kappa_nearest is the DIRECT causal variable from Gumbel Race theory.
If logit(q) = alpha * kappa_nearest - b_eff * log(K-1) + C_0 holds
CROSS-TASK (across different K), this is the Nobel-worthy universal law.

Pre-registered: alpha and b_eff have CV < 0.25 across datasets (LOAO test)
Nobel-track: if PASS, we have a first-principles universal law for kNN quality.
"""

import json
import sys
import time
import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset

# Use cached model results if available
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ================================================================
# CONFIG
# ================================================================
# Multi-family models: Pythia (GPT-NeoX), GPT-Neo, Qwen2.5
# Tests CROSS-ARCHITECTURE universality of kappa_nearest law
MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/gpt-neo-125m",       # GPT-Neo family (different arch from Pythia)
    "Qwen/Qwen2.5-0.5B",             # Qwen2.5 family (totally different arch/training)
    "allenai/OLMo-1B-hf",            # OLMo family (Allen AI, yet another arch)
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",  # LLaMA family (5th arch)
]

# Per-model layer indices (proportional depth: 25%, 50%, 75%, 100%)
MODEL_LAYERS = {
    "pythia-160m": [3, 6, 9, 12],   # 12 total layers
    "pythia-410m": [3, 6, 9, 12],   # 12 total layers
    "pythia-1b":   [4, 8, 12, 16],  # 16 total layers
    "gpt-neo-125m": [3, 6, 9, 12],  # 12 total layers
    "Qwen2.5-0.5B": [6, 12, 18, 24],  # 28 total layers
    "OLMo-1B-hf": [4, 8, 12, 16],  # 16 total layers
    "TinyLlama-1.1B-intermediate-step-1431k-3T": [5, 11, 16, 22],  # 22 total layers
}

DATASETS = {
    "agnews":       {"hf_name": "fancyzhx/ag_news",    "text_col": "text",    "label_col": "label",       "K": 4,  "n_sample": 1000},
    "dbpedia":      {"hf_name": "fancyzhx/dbpedia_14", "text_col": "content", "label_col": "label",       "K": 14, "n_sample": 1000},
    "20newsgroups": {"hf_name": "SetFit/20_newsgroups", "text_col": "text",   "label_col": "label_text",  "K": 20, "n_sample": 1000},
    "go_emotions":  {"hf_name": "google-research-datasets/go_emotions", "hf_cfg": "simplified",
                     "text_col": "text", "label_col": "labels", "K": 28, "n_sample": 5000, "multilabel": True},
}
LAYERS_TO_EVAL = [3, 6, 9, 12]  # default layers (overridden per-model by MODEL_LAYERS)
BATCH_SIZE = 64
CACHE_DIR = "results"

PRE_REG_CV_ALPHA = 0.25   # alpha CV threshold (across datasets)
PRE_REG_CV_BEFF  = 0.25   # b_eff CV threshold

# ================================================================
# EMBEDDING EXTRACTION
# ================================================================
@torch.no_grad()
def get_embeddings_at_layers(model_name, texts, labels, layers, batch_size=64):
    """Extract embeddings at specified layers."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, torch_dtype=torch.float16
    ).to(DEVICE).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_layer_embs = {l: [] for l in layers}
    all_labels = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=128
        ).to(DEVICE)

        outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of (n_layers+1,) tensors

        # Mean pool each layer
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        for layer_idx in layers:
            if layer_idx < len(hidden_states):
                h = hidden_states[layer_idx].float()  # (B, seq, d) cast to fp32
                emb = (h * mask).sum(1) / mask.sum(1)  # (B, d)
                emb_np = np.nan_to_num(emb.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                all_layer_embs[layer_idx].append(emb_np)

        all_labels.extend(batch_labels)

    del model
    torch.cuda.empty_cache()

    for l in layers:
        if all_layer_embs[l]:
            all_layer_embs[l] = np.vstack(all_layer_embs[l])
        else:
            all_layer_embs[l] = None

    return all_layer_embs, np.array(all_labels)


# ================================================================
# COMPUTE kappa_nearest DIRECTLY
# ================================================================
def compute_kappa_nearest(embeddings, labels, K, subsample=500):
    """
    Compute kappa_nearest = min_{j!=k} ||mu_k - mu_j|| / sigma_W

    sigma_W = sqrt(mean within-class variance per dimension)
    mu_k = class mean embedding
    """
    X, y = embeddings, labels
    classes = np.unique(y)
    if len(classes) != K:
        K = len(classes)
    if K < 2:
        return None, None

    # Subsample for speed
    if len(X) > subsample:
        idx = np.random.choice(len(X), subsample, replace=False)
        X = X[idx]
        y = y[idx]

    # Class means
    mu = {}
    for k in classes:
        idx_k = (y == k)
        if idx_k.sum() < 2:
            continue
        mu[k] = X[idx_k].mean(0)

    if len(mu) < 2:
        return None, None

    # Pooled within-class variance (diagonal, i.e., per-dimension mean variance)
    within_var = 0.0
    n_total = 0
    for k, mean_k in mu.items():
        idx_k = (y == k)
        Xk = X[idx_k]
        within_var += np.sum((Xk - mean_k)**2)
        n_total += len(Xk)

    sigma_W = float(np.sqrt(within_var / (n_total * X.shape[1])))  # per feature std
    if sigma_W < 1e-10:
        return None, None

    # kappa_nearest: for each class, find minimum distance to any other class
    all_kappa = []
    for k, mean_k in mu.items():
        min_dist = np.inf
        for j, mean_j in mu.items():
            if j == k:
                continue
            dist_kj = float(np.linalg.norm(mean_k - mean_j))
            if dist_kj < min_dist:
                min_dist = dist_kj

        kappa_k = min_dist / (sigma_W * np.sqrt(X.shape[1]))  # normalize by sqrt(d)
        all_kappa.append(kappa_k)

    kappa_nearest = float(np.mean(all_kappa))  # mean over classes
    kappa_min = float(np.min(all_kappa))       # minimum over classes (tightest bottleneck)

    return kappa_nearest, kappa_min


# ================================================================
# COMPUTE kNN quality
# ================================================================
def compute_knn_q(embeddings, labels, K, subsample=1000):
    """1-NN quality q_norm = (acc - 1/K) / (1 - 1/K)."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedShuffleSplit

    # Random subsample (not just first N, to avoid ordering bias)
    if len(embeddings) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), subsample, replace=False)
        X, y = embeddings[idx], labels[idx]
    else:
        X, y = embeddings, labels

    # Filter out rare classes (need >= 2 for StratifiedShuffleSplit)
    from collections import Counter
    counts = Counter(y.tolist())
    valid_set = {lbl for lbl, cnt in counts.items() if cnt >= 2}
    if len(valid_set) < 2:
        return None
    mask = np.array([l in valid_set for l in y])
    X, y = X[mask], y[mask]
    K_eff = len(valid_set)  # use actual number of valid classes

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = (acc - 1.0/K_eff) / (1.0 - 1.0/K_eff)
    return float(q)


# ================================================================
# LOAD AND CACHE DATA
# ================================================================
def load_dataset_texts_labels(dataset_name, config):
    """Load dataset, return (texts, labels) subsample."""
    hf_name = config["hf_name"]
    text_col = config["text_col"]
    label_col = config["label_col"]
    n_sample = config["n_sample"]

    hf_cfg = config.get("hf_cfg")
    multilabel = config.get("multilabel", False)
    try:
        if hf_cfg:
            ds = load_dataset(hf_name, hf_cfg, split="test")
        else:
            ds = load_dataset(hf_name, split="test")

        texts = [str(x[text_col]) for x in ds]
        labels_raw = [x[label_col] for x in ds]
    except Exception as e:
        print(f"  Error loading {hf_name}: {e}", flush=True)
        return None, None

    # Handle multilabel: take first label
    if multilabel:
        labels_raw = [l[0] if isinstance(l, list) and l else 0 for l in labels_raw]

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)

    # Subsample
    np.random.seed(42)
    idx = np.random.choice(len(texts), min(n_sample, len(texts)), replace=False)
    texts = [texts[i] for i in idx]
    labels = labels[idx]

    return texts, labels


# ================================================================
# MAIN
# ================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("KAPPA_NEAREST UNIVERSAL LAW TEST")
    print(f"Models: {[m.split('/')[-1] for m in MODELS]}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Pre-registered: alpha CV < {PRE_REG_CV_ALPHA}, b_eff CV < {PRE_REG_CV_BEFF}")
    print("=" * 70)

    # Collect all data points
    all_points = []

    for dataset_name, config in DATASETS.items():
        K = config["K"]
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name} (K={K})")

        texts, labels = load_dataset_texts_labels(dataset_name, config)
        if texts is None:
            continue

        print(f"  Loaded {len(texts)} samples", flush=True)

        for model_name in MODELS:
            model_short = model_name.split("/")[-1]
            print(f"\n  Model: {model_short}...", flush=True)

            cache_path = os.path.join(CACHE_DIR, f"kappa_near_cache_{dataset_name}_{model_short}.json")

            if os.path.exists(cache_path):
                try:
                    with open(cache_path) as f:
                        content = f.read().strip()
                    if not content:
                        print(f"    Cache empty, regenerating: {cache_path}")
                        os.remove(cache_path)
                    else:
                        model_data = json.loads(content)
                        print(f"    Loaded from cache: {cache_path} ({len(model_data)} pts)")
                        for pt in model_data:
                            all_points.append(pt)
                        continue
                except Exception as e:
                    print(f"    Cache corrupt ({e}), regenerating")
                    os.remove(cache_path)

            # Per-model layer selection (proportional depth)
            layers_for_model = MODEL_LAYERS.get(model_short, LAYERS_TO_EVAL)
            try:
                layer_embs, y = get_embeddings_at_layers(
                    model_name, texts, labels, layers_for_model, BATCH_SIZE
                )
            except Exception as e:
                print(f"    Error: {e}", flush=True)
                continue

            model_points = []
            for layer in layers_for_model:
                embs = layer_embs.get(layer)
                if embs is None:
                    continue

                # Filter NaN/Inf embeddings (some HF datasets produce these)
                valid_mask = np.isfinite(embs).all(axis=1)
                n_invalid = (~valid_mask).sum()
                if n_invalid > 0:
                    print(f"    Layer {layer}: filtering {n_invalid} NaN/Inf rows", flush=True)
                embs = embs[valid_mask]
                y_layer = y[valid_mask]
                if len(embs) < 20:
                    print(f"    Layer {layer}: too few valid embeddings ({len(embs)}), skipping", flush=True)
                    continue

                q = compute_knn_q(embs, y_layer, K)
                kappa_near, kappa_min = compute_kappa_nearest(embs, y_layer, K)

                if q is None or kappa_near is None:
                    continue

                pt = {
                    "model": model_short, "dataset": dataset_name,
                    "layer": layer, "K": K,
                    "q": q, "kappa_nearest": kappa_near, "kappa_min": kappa_min,
                    "logit_q": float(np.log(max(q,0.001)/max(1-q,0.001))),
                    "logKm1": float(np.log(K-1)) if K > 1 else 0.0
                }
                model_points.append(pt)
                all_points.append(pt)

                print(f"    Layer {layer}: q={q:.4f}  kappa_near={kappa_near:.4f}  "
                      f"logit(q)={pt['logit_q']:.4f}", flush=True)

            # Save cache
            if model_points:
                with open(cache_path, "w") as f:
                    json.dump(model_points, f, indent=2,
                              default=lambda x: float(x) if hasattr(x, "__float__") else str(x))

    print(f"\n\nTotal valid points: {len(all_points)}")

    if len(all_points) < 10:
        print("Not enough points for analysis")
        return

    # ============================================================
    # ANALYSIS: Fit logit(q) = alpha * kappa_nearest + b_eff * logKm1 + C_0
    # ============================================================
    print("\n" + "=" * 70)
    print("GLOBAL FIT: logit(q) = alpha * kappa_nearest + beta * log(K-1) + C")
    print("=" * 70)

    kappa_arr = np.array([p["kappa_nearest"] for p in all_points])
    logKm1_arr = np.array([p["logKm1"] for p in all_points])
    logit_arr = np.array([p["logit_q"] for p in all_points])

    # 2D OLS: logit(q) = alpha * kappa + beta * logKm1 + C
    X_design = np.column_stack([kappa_arr, logKm1_arr, np.ones(len(kappa_arr))])
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_design, logit_arr, rcond=None)
        alpha, beta, C0 = coeffs
        logit_pred = X_design @ coeffs
        ss_res = np.sum((logit_arr - logit_pred)**2)
        ss_tot = np.sum((logit_arr - logit_arr.mean())**2)
        r2_global = 1 - ss_res / ss_tot
        print(f"  alpha (kappa_nearest): {alpha:.4f}")
        print(f"  beta (log(K-1)):       {beta:.4f}  [Gumbel theory: -1.0 to -b_eff]")
        print(f"  C_0 (intercept):       {C0:.4f}")
        print(f"  R2 (global): {r2_global:.3f}")
        print(f"  MAE: {np.mean(np.abs(logit_arr - logit_pred)):.4f}")
    except Exception as e:
        print(f"  OLS failed: {e}")
        alpha, beta, C0 = None, None, None

    # ============================================================
    # LOAO: Leave-One-Architecture-Out cross-validation
    # ============================================================
    print("\n" + "=" * 70)
    print("LOAO: Leave-One-Architecture-Out")
    print("=" * 70)

    models_in_data = list(set(p["model"] for p in all_points))
    loao_results = {}
    alpha_values, beta_values = [], []

    for held_out in models_in_data:
        train = [p for p in all_points if p["model"] != held_out]
        test  = [p for p in all_points if p["model"] == held_out]
        if len(train) < 5 or len(test) < 2:
            continue

        kappa_tr = np.array([p["kappa_nearest"] for p in train])
        logKm1_tr = np.array([p["logKm1"] for p in train])
        logit_tr = np.array([p["logit_q"] for p in train])

        X_tr = np.column_stack([kappa_tr, logKm1_tr, np.ones(len(kappa_tr))])
        try:
            coeffs_tr, _, _, _ = np.linalg.lstsq(X_tr, logit_tr, rcond=None)
        except:
            continue

        kappa_te = np.array([p["kappa_nearest"] for p in test])
        logKm1_te = np.array([p["logKm1"] for p in test])
        logit_te = np.array([p["logit_q"] for p in test])

        X_te = np.column_stack([kappa_te, logKm1_te, np.ones(len(kappa_te))])
        logit_pred_te = X_te @ coeffs_tr

        mae = float(np.mean(np.abs(logit_te - logit_pred_te)))
        loao_results[held_out] = {
            "mae": mae, "n_test": len(test),
            "alpha": float(coeffs_tr[0]), "beta": float(coeffs_tr[1])
        }
        alpha_values.append(float(coeffs_tr[0]))
        beta_values.append(float(coeffs_tr[1]))
        print(f"  Hold-out {held_out}: MAE={mae:.4f}  alpha={coeffs_tr[0]:.4f}  beta={coeffs_tr[1]:.4f}")

    if alpha_values:
        cv_alpha = float(np.std(alpha_values) / (np.abs(np.mean(alpha_values)) + 1e-10))
        cv_beta = float(np.std(beta_values) / (np.abs(np.mean(beta_values)) + 1e-10))
        print(f"\n  alpha: mean={np.mean(alpha_values):.4f}  CV={cv_alpha:.3f}  "
              f"{'PASS' if cv_alpha < PRE_REG_CV_ALPHA else 'FAIL'}")
        print(f"  beta:  mean={np.mean(beta_values):.4f}  CV={cv_beta:.3f}  "
              f"{'PASS' if cv_beta < PRE_REG_CV_BEFF else 'FAIL'}")

    # ============================================================
    # LODO: Leave-One-Dataset-Out cross-validation
    # ============================================================
    print("\n" + "=" * 70)
    print("LODO: Leave-One-Dataset-Out (CRITICAL: tests cross-task universality)")
    print("=" * 70)

    datasets_in_data = list(set(p["dataset"] for p in all_points))
    alpha_lodo, beta_lodo = [], []

    for held_out_ds in datasets_in_data:
        train = [p for p in all_points if p["dataset"] != held_out_ds]
        test  = [p for p in all_points if p["dataset"] == held_out_ds]
        if len(train) < 5 or len(test) < 2:
            continue

        kappa_tr = np.array([p["kappa_nearest"] for p in train])
        logKm1_tr = np.array([p["logKm1"] for p in train])
        logit_tr = np.array([p["logit_q"] for p in train])

        X_tr = np.column_stack([kappa_tr, logKm1_tr, np.ones(len(kappa_tr))])
        try:
            coeffs_tr, _, _, _ = np.linalg.lstsq(X_tr, logit_tr, rcond=None)
        except:
            continue

        kappa_te = np.array([p["kappa_nearest"] for p in test])
        logKm1_te = np.array([p["logKm1"] for p in test])
        logit_te = np.array([p["logit_q"] for p in test])

        X_te = np.column_stack([kappa_te, logKm1_te, np.ones(len(kappa_te))])
        logit_pred_te = X_te @ coeffs_tr

        mae = float(np.mean(np.abs(logit_te - logit_pred_te)))
        K_held = test[0]["K"]
        print(f"  Hold-out {held_out_ds} (K={K_held}): MAE={mae:.4f}  "
              f"n={len(test)}  alpha={coeffs_tr[0]:.4f}  beta={coeffs_tr[1]:.4f}")
        alpha_lodo.append(float(coeffs_tr[0]))
        beta_lodo.append(float(coeffs_tr[1]))

    if alpha_lodo:
        cv_a_lodo = float(np.std(alpha_lodo) / (np.abs(np.mean(alpha_lodo)) + 1e-10))
        cv_b_lodo = float(np.std(beta_lodo) / (np.abs(np.mean(beta_lodo)) + 1e-10))
        print(f"\n  LODO alpha CV: {cv_a_lodo:.3f}  {'PASS' if cv_a_lodo < PRE_REG_CV_ALPHA else 'FAIL'}")
        print(f"  LODO beta  CV: {cv_b_lodo:.3f}  {'PASS' if cv_b_lodo < PRE_REG_CV_BEFF else 'FAIL'}")

    # Save
    output = {
        "experiment": "kappa_nearest_universal_law",
        "pre_registered": {
            "cv_alpha": PRE_REG_CV_ALPHA, "cv_beff": PRE_REG_CV_BEFF
        },
        "all_points": all_points,
        "global_fit": {
            "alpha": float(alpha) if alpha else None,
            "beta": float(beta) if beta else None,
            "C0": float(C0) if C0 else None,
            "r2": float(r2_global) if alpha else None
        },
        "loao": loao_results,
        "lodo_alpha_cv": float(cv_a_lodo) if alpha_lodo else None,
        "lodo_beta_cv": float(cv_b_lodo) if alpha_lodo else None,
        "runtime_s": int(time.time() - t0)
    }

    out_path = "results/cti_kappa_nearest_universal.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")
    print(f"Runtime: {int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
