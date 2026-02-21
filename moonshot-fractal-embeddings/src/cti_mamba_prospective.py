#!/usr/bin/env python -u
"""
MAMBA-130M PROSPECTIVE VALIDATION (Feb 21 2026)
================================================
TRUE PROSPECTIVE TEST: Mamba-130M (state-spaces, SSM architecture) was NOT
used to fit any parameters. This is an SSM (State Space Model), NOT a Transformer.
No attention mechanism. Recurrent updates. Completely different computation graph.

If the same kappa_nearest law holds for SSMs as for Transformers, it would be a
profound universality result: the law is MODALITY-ARCHITECTURE-AGNOSTIC.

All parameters are FROZEN from the 9-model fit on Transformer-only data.

Pre-registered criterion (same as Phi-2):
  - Pearson r(q_pred, q_actual) > 0.80 on topic classification tasks
  - MAE < 0.10 in q space
  - Same pre-registered as Phi-2 prospective

Frozen parameters (identical to Phi-2 test):
  GLOBAL MODEL: logit(q) = 3.07 * kappa_nearest - 0.72 * log(K-1) + 0.79
  PER-TASK MODEL: logit(q) = 1.54 * kappa_nearest + C_task
    where C_task = {agnews: 0.483, dbpedia: 1.424, 20newsgroups: -0.348, go_emotions: -1.056}

Model: state-spaces/mamba-130m
  - 130M parameters
  - 24 pre-trained layers (layers 0-23 in checkpoint)
  - d_model=768 (same as BERT-base)
  - Uses NeoX tokenizer
  - Recurrent SSM architecture (NO attention)
  - Never used in any prior CTI experiment

Test layers (hidden state indices from 0-indexed output):
  6, 12, 18, 24 => correspond to Mamba residual blocks 5, 11, 17, 23

This test directly answers: Is the kappa_nearest law a Transformer phenomenon
or a universal law of learned representations?
"""

import json
import sys
import os
import time
import numpy as np
from collections import defaultdict

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

from transformers import MambaModel, AutoTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# ================================================================
# FROZEN PARAMETERS (DO NOT CHANGE - pre-registered, same as Phi-2)
# ================================================================
FROZEN_GLOBAL_ALPHA = 3.07
FROZEN_GLOBAL_BETA = -0.72
FROZEN_GLOBAL_C = 0.79

FROZEN_TASK_ALPHA = 1.54
FROZEN_C_TASK = {
    'agnews': 0.483,
    'dbpedia': 1.424,
    '20newsgroups': -0.348,
    'go_emotions': -1.056,
}

R_THRESHOLD = 0.80
MAE_THRESHOLD = 0.10

# ================================================================
# MODEL: Mamba-130M (SSM - NO attention)
# ================================================================
MODEL_NAME = "state-spaces/mamba-130m"
# Test at hidden state indices 6, 12, 18, 24
# These correspond to pre-trained SSM residual blocks 5, 11, 17, 23
# (Mamba-130M has 24 pre-trained layers; hidden_state[i] = layer i-1 output for i>=1)
LAYER_INDICES = [6, 12, 18, 24]  # hidden state indices
LAYER_LABELS = [5, 11, 17, 23]   # human-readable block numbers (0-indexed)

# ================================================================
# DATASETS
# ================================================================
DATASETS = {
    "agnews": {"K": 4, "n_sample": 2000},
    "dbpedia": {"K": 14, "n_sample": 2000},
    "20newsgroups": {"K": 20, "n_sample": 2000},
    "go_emotions": {"K": 28, "n_sample": 5000},
}
BATCH_SIZE = 64  # Mamba-130M is small


# ================================================================
# DATA LOADING (same as Phi-2 prospective)
# ================================================================
def load_dataset_texts(dataset_name, n_sample):
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
    return [texts[i] for i in idx], np.array([labels[i] for i in idx])


# ================================================================
# EMBEDDING EXTRACTION
# ================================================================
def get_mamba_embeddings(texts, labels):
    print(f"  Loading {MODEL_NAME} (SSM - no attention)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token

    # Load Mamba model - use float16 for efficiency
    model = MambaModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()

    print(f"  Mamba-130M loaded. Extracting embeddings ({len(texts)} texts)...", flush=True)
    t0 = time.time()

    all_layer_embs = defaultdict(list)
    all_labels = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=128
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of tensors
        mask = inputs["attention_mask"].unsqueeze(-1).float()

        for layer_idx in LAYER_INDICES:
            if layer_idx < len(hidden_states):
                h = hidden_states[layer_idx].float()  # Convert to fp32
                emb = (h * mask).sum(1) / mask.sum(1)  # mean pooling
                all_layer_embs[layer_idx].append(emb.cpu().numpy())

        all_labels.extend(batch_labels.tolist() if hasattr(batch_labels, 'tolist') else list(batch_labels))

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    Batch {i//BATCH_SIZE}/{len(texts)//BATCH_SIZE}...", flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"  Done in {int(time.time()-t0)}s", flush=True)

    result = {}
    for l in LAYER_INDICES:
        if all_layer_embs[l]:
            result[l] = np.vstack(all_layer_embs[l])
    return result, np.array(all_labels)


# ================================================================
# COMPUTE q AND kappa_nearest
# ================================================================
def compute_kappa_nearest(embeddings, labels, K, subsample=500):
    X, y = embeddings, labels
    classes = np.unique(y)
    K_eff = len(classes)
    if K_eff < 2:
        return None, None

    if len(X) > subsample:
        idx = np.random.default_rng(42).choice(len(X), subsample, replace=False)
        X, y = X[idx], y[idx]

    mu = {}
    for k in classes:
        mask_k = (y == k)
        if mask_k.sum() < 2:
            continue
        mu[k] = X[mask_k].mean(0)

    if len(mu) < 2:
        return None, None

    within_var = 0.0
    n_total = 0
    for k, mean_k in mu.items():
        Xk = X[y == k]
        within_var += np.sum((Xk - mean_k)**2)
        n_total += len(Xk)

    sigma_W = float(np.sqrt(within_var / (n_total * X.shape[1])))
    if sigma_W < 1e-10:
        return None, None

    all_kappa = []
    for k, mean_k in mu.items():
        min_dist = min(float(np.linalg.norm(mean_k - mean_j))
                       for j, mean_j in mu.items() if j != k)
        all_kappa.append(min_dist / (sigma_W * np.sqrt(X.shape[1])))

    return float(np.mean(all_kappa)), float(np.min(all_kappa))


def compute_knn_q(embeddings, labels, K, subsample=1000):
    if len(embeddings) > subsample:
        idx = np.random.default_rng(42).choice(len(embeddings), subsample, replace=False)
        X, y = embeddings[idx], labels[idx]
    else:
        X, y = embeddings, labels

    counts = Counter(y.tolist())
    valid_set = {lbl for lbl, cnt in counts.items() if cnt >= 2}
    if len(valid_set) < 2:
        return None
    mask = np.array([l in valid_set for l in y])
    X, y = X[mask], y[mask]
    K_eff = len(valid_set)

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
    return float((acc - 1.0/K_eff) / (1.0 - 1.0/K_eff))


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("MAMBA-130M PROSPECTIVE VALIDATION (SSM - NO ATTENTION)")
    print(f"Model: {MODEL_NAME} (130M params, SSM, UNSEEN during parameter fitting)")
    print(f"Frozen global: logit(q) = {FROZEN_GLOBAL_ALPHA} * kappa - {abs(FROZEN_GLOBAL_BETA)} * log(K-1) + {FROZEN_GLOBAL_C}")
    print(f"Success criterion: r > {R_THRESHOLD}, MAE < {MAE_THRESHOLD}")
    print(f"KEY QUESTION: Does the Transformer law hold for SSMs?")
    print("=" * 70, flush=True)

    all_results = []

    for ds_name, ds_cfg in DATASETS.items():
        K = ds_cfg["K"]
        n_sample = ds_cfg["n_sample"]

        per_ds_cache = f"results/kappa_near_cache_{ds_name}_mamba-130m.json"
        if os.path.exists(per_ds_cache):
            with open(per_ds_cache) as f:
                cached = json.load(f)
            if cached:
                print(f"\n{ds_name}: Loaded from cache ({len(cached)} pts)")
                for p in cached:
                    p['model'] = 'mamba-130m'
                all_results.extend(cached)
                continue

        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name} (K={K})")
        print(f"{'='*50}", flush=True)

        try:
            texts, y = load_dataset_texts(ds_name, n_sample)
        except Exception as e:
            print(f"  SKIP: {e}", flush=True)
            continue

        try:
            layer_embs, y_emb = get_mamba_embeddings(texts, y)
        except Exception as e:
            print(f"  Embedding error: {e}", flush=True)
            continue

        ds_points = []
        for layer_idx, layer_label in zip(LAYER_INDICES, LAYER_LABELS):
            embs = layer_embs.get(layer_idx)
            if embs is None:
                continue

            valid = np.isfinite(embs).all(axis=1)
            embs = embs[valid]
            y_l = y_emb[valid]

            q = compute_knn_q(embs, y_l, K)
            if q is None:
                continue

            kn, km = compute_kappa_nearest(embs, y_l, K)
            if kn is None:
                continue

            logit_q = float(np.log(max(q, 1e-6) / (1 - min(q, 1-1e-6))))
            logKm1 = float(np.log(K - 1))

            # Frozen predictions
            logit_pred_global = FROZEN_GLOBAL_ALPHA * kn + FROZEN_GLOBAL_BETA * logKm1 + FROZEN_GLOBAL_C
            q_pred_global = float(1.0 / (1.0 + np.exp(-logit_pred_global)))

            C_task = FROZEN_C_TASK.get(ds_name, 0.0)
            logit_pred_task = FROZEN_TASK_ALPHA * kn + C_task
            q_pred_task = float(1.0 / (1.0 + np.exp(-logit_pred_task)))

            pt = {
                "model": "mamba-130m",
                "dataset": ds_name,
                "layer": int(layer_label),  # block index (0-indexed)
                "hidden_state_idx": int(layer_idx),
                "K": K,
                "q": float(q),
                "kappa_nearest": float(kn),
                "kappa_min": float(km),
                "logit_q": float(logit_q),
                "logKm1": float(logKm1),
                "q_pred_global": q_pred_global,
                "q_pred_task": q_pred_task,
                "error_global": float(abs(q_pred_global - q)),
                "error_task": float(abs(q_pred_task - q)),
            }
            ds_points.append(pt)
            print(f"  Block {layer_label:2d}: q={q:.4f}  kappa={kn:.4f}  "
                  f"pred_global={q_pred_global:.4f}  err={abs(q_pred_global-q):.4f}", flush=True)

        if ds_points:
            with open(per_ds_cache, "w") as f:
                json.dump(ds_points, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
            print(f"  Cached {len(ds_points)} pts", flush=True)
            all_results.extend(ds_points)

    # ================================================================
    # EVALUATION
    # ================================================================
    print("\n" + "=" * 70)
    print("MAMBA-130M PROSPECTIVE VALIDATION RESULTS")
    print("=" * 70)

    if not all_results:
        print("No results!")
        return

    topic_pts = [p for p in all_results if p['dataset'] != 'go_emotions']
    all_pts = all_results

    def evaluate_predictions(pts, model_name):
        if not pts:
            return None
        q_actual = np.array([p['q'] for p in pts])
        q_pred = np.array([p[f'q_pred_{model_name}'] for p in pts])
        errors = np.array([p[f'error_{model_name}'] for p in pts])

        r = float(np.corrcoef(q_actual, q_pred)[0, 1])
        mae = float(np.mean(errors))
        passed_r = r > R_THRESHOLD
        passed_mae = mae < MAE_THRESHOLD

        print(f"  Model: {model_name}")
        print(f"    Pearson r: {r:.4f}  threshold={R_THRESHOLD}  {'PASS' if passed_r else 'FAIL'}")
        print(f"    MAE: {mae:.4f}  threshold={MAE_THRESHOLD}  {'PASS' if passed_mae else 'FAIL'}")
        print(f"    Overall: {'PASS (both criteria met)' if passed_r and passed_mae else 'FAIL'}")
        return {"r": r, "mae": mae, "passed_r": bool(passed_r), "passed_mae": bool(passed_mae)}

    print("\nTOPIC TASKS (agnews, dbpedia, 20newsgroups):")
    result_global = evaluate_predictions(topic_pts, "global")
    print()
    result_task = evaluate_predictions(topic_pts, "task")

    print("\nALL TASKS (including go_emotions - expected FAIL by theory):")
    evaluate_predictions(all_pts, "global")

    print("\nPer-dataset MAE (global model):")
    for ds in set(p['dataset'] for p in all_results):
        pts_ds = [p for p in all_results if p['dataset'] == ds]
        mae_ds = np.mean([p['error_global'] for p in pts_ds])
        print(f"  {ds:15s}: MAE={mae_ds:.4f}  N={len(pts_ds)}")

    # Save
    output = {
        "experiment": "mamba_prospective_validation",
        "architecture": "SSM (State Space Model, NO attention)",
        "pre_registered": {
            "r_threshold": R_THRESHOLD,
            "mae_threshold": MAE_THRESHOLD,
            "frozen_global_alpha": FROZEN_GLOBAL_ALPHA,
            "frozen_global_beta": FROZEN_GLOBAL_BETA,
            "frozen_global_C": FROZEN_GLOBAL_C,
            "frozen_task_alpha": FROZEN_TASK_ALPHA,
        },
        "results": all_results,
        "evaluation": {
            "topic_global": result_global,
            "topic_task": result_task,
        }
    }
    out_path = "results/cti_mamba_prospective.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
