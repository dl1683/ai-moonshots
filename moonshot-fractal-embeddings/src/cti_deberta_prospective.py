#!/usr/bin/env python -u
"""
DEBERTA-BASE PROSPECTIVE VALIDATION (Feb 21 2026)
===================================================
TRUE PROSPECTIVE TEST: DeBERTa-base (Microsoft, 140M params) was NOT used to
fit any parameters. DeBERTa uses DISENTANGLED ATTENTION with relative position
embeddings - fundamentally different from BERT's absolute position attention.

Same d=768 as BERT-base and GPT-2-based models in the training set.
Key test: Does C_task transfer between BERT-like MLM models when d is the same?

If yes: intercept is purely task-specific, not architecture-specific.
If no: attention mechanism or pretraining details affect C_task too.

DeBERTa advantages over BERT:
  - Disentangled attention: positions and content represented separately
  - Enhanced mask decoder for MLM
  - State-of-art on many NLP benchmarks when BERT was released

Model: microsoft/deberta-base
  - 14M parameters (fastest model we've tested)
  - 12 layers, d=768 (different from BERT's d=768)
  - DISCRIMINATIVE pretraining (novel, never used in CTI fits)

Pre-registered criterion (same as Phi-2, Mamba):
  - Pearson r > 0.80 on topic tasks
  - MAE < 0.10

Frozen parameters (identical):
  GLOBAL: logit(q) = 3.07 * kappa_nearest - 0.72 * log(K-1) + 0.79
  TASK:   logit(q) = 1.54 * kappa_nearest + C_task
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

from transformers import AutoModel, AutoTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# ================================================================
# FROZEN PARAMETERS (same as Phi-2 and Mamba tests)
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
# MODEL: ELECTRA-small-discriminator
# ================================================================
MODEL_NAME = "microsoft/deberta-base"
LAYERS = [3, 6, 9, 12]  # 4 layers to match other models (12 total)

# ================================================================
# DATASETS
# ================================================================
DATASETS = {
    "agnews": {"K": 4, "n_sample": 2000},
    "dbpedia": {"K": 14, "n_sample": 2000},
    "20newsgroups": {"K": 20, "n_sample": 2000},
    "go_emotions": {"K": 28, "n_sample": 5000},
}
BATCH_SIZE = 128  # ELECTRA-small is tiny


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


def get_embeddings(texts, labels):
    print(f"  Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
    ).to(DEVICE)
    model.eval()

    print(f"  Model loaded. Extracting embeddings ({len(texts)} texts)...", flush=True)
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

        hidden_states = outputs.hidden_states
        mask = inputs["attention_mask"].unsqueeze(-1).float()

        for layer_idx in LAYERS:
            if layer_idx < len(hidden_states):
                h = hidden_states[layer_idx].float()
                emb = (h * mask).sum(1) / mask.sum(1)
                all_layer_embs[layer_idx].append(emb.cpu().numpy())

        all_labels.extend(batch_labels.tolist() if hasattr(batch_labels, 'tolist') else list(batch_labels))

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    Batch {i//BATCH_SIZE}/{len(texts)//BATCH_SIZE}...", flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"  Done in {int(time.time()-t0)}s", flush=True)

    result = {}
    for l in LAYERS:
        if all_layer_embs[l]:
            result[l] = np.vstack(all_layer_embs[l])
    return result, np.array(all_labels)


def compute_kappa_nearest(embeddings, labels, subsample=500):
    X, y = embeddings, labels
    classes = np.unique(y)
    if len(classes) < 2:
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


def main():
    print("=" * 70)
    print("DEBERTA-BASE PROSPECTIVE VALIDATION (MLM pretraining, disentangled attention)")
    print(f"Model: {MODEL_NAME} (140M params, UNSEEN during fitting)")
    print(f"Pretraining: DISCRIMINATIVE (generator/discriminator, NOT MLM or CLM)")
    print(f"Success criterion: r > {R_THRESHOLD}, MAE < {MAE_THRESHOLD}")
    print("=" * 70, flush=True)

    all_results = []

    for ds_name, ds_cfg in DATASETS.items():
        K = ds_cfg["K"]
        n_sample = ds_cfg["n_sample"]

        per_ds_cache = f"results/kappa_near_cache_{ds_name}_deberta-base.json"
        if os.path.exists(per_ds_cache):
            with open(per_ds_cache) as f:
                cached = json.load(f)
            if cached:
                print(f"\n{ds_name}: Loaded from cache ({len(cached)} pts)")
                for p in cached:
                    p['model'] = 'deberta-base'
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
            layer_embs, y_emb = get_embeddings(texts, y)
        except Exception as e:
            print(f"  Embedding error: {e}", flush=True)
            continue

        ds_points = []
        for layer in LAYERS:
            embs = layer_embs.get(layer)
            if embs is None:
                continue

            valid = np.isfinite(embs).all(axis=1)
            embs = embs[valid]
            y_l = y_emb[valid]

            q = compute_knn_q(embs, y_l, K)
            if q is None:
                continue

            kn, km = compute_kappa_nearest(embs, y_l)
            if kn is None:
                continue

            logit_q = float(np.log(max(q, 1e-6) / (1 - min(q, 1-1e-6))))
            logKm1 = float(np.log(K - 1))

            logit_pred_global = FROZEN_GLOBAL_ALPHA * kn + FROZEN_GLOBAL_BETA * logKm1 + FROZEN_GLOBAL_C
            q_pred_global = float(1.0 / (1.0 + np.exp(-logit_pred_global)))

            C_task = FROZEN_C_TASK.get(ds_name, 0.0)
            logit_pred_task = FROZEN_TASK_ALPHA * kn + C_task
            q_pred_task = float(1.0 / (1.0 + np.exp(-logit_pred_task)))

            pt = {
                "model": "deberta-base",
                "dataset": ds_name,
                "layer": int(layer),
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
            print(f"  Layer {layer:2d}: q={q:.4f}  kappa={kn:.4f}  "
                  f"pred_global={q_pred_global:.4f}  err={abs(q_pred_global-q):.4f}", flush=True)

        if ds_points:
            with open(per_ds_cache, "w") as f:
                json.dump(ds_points, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
            print(f"  Cached {len(ds_points)} pts", flush=True)
            all_results.extend(ds_points)

    # EVALUATION
    print("\n" + "=" * 70)
    print("DEBERTA-BASE PROSPECTIVE VALIDATION RESULTS")
    print("=" * 70)

    if not all_results:
        print("No results!")
        return

    topic_pts = [p for p in all_results if p['dataset'] != 'go_emotions']

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
        print(f"  {model_name}: r={r:.4f} {'PASS' if passed_r else 'FAIL'}  MAE={mae:.4f} {'PASS' if passed_mae else 'FAIL'}  Overall={'PASS' if passed_r and passed_mae else 'FAIL'}")
        return {"r": r, "mae": mae, "passed_r": bool(passed_r), "passed_mae": bool(passed_mae)}

    print("\nTOPIC TASKS:")
    result_global = evaluate_predictions(topic_pts, "global")
    result_task = evaluate_predictions(topic_pts, "task")

    output = {
        "experiment": "deberta_base_prospective_validation",
        "architecture": "ELECTRA-small-discriminator (MLM pretraining, NOT MLM/CLM)",
        "pre_registered": {
            "r_threshold": R_THRESHOLD,
            "mae_threshold": MAE_THRESHOLD,
            "frozen_global_alpha": FROZEN_GLOBAL_ALPHA,
            "frozen_global_beta": FROZEN_GLOBAL_BETA,
            "frozen_global_C": FROZEN_GLOBAL_C,
            "frozen_task_alpha": FROZEN_TASK_ALPHA,
        },
        "results": all_results,
        "evaluation": {"topic_global": result_global, "topic_task": result_task},
    }
    out_path = "results/cti_deberta_prospective.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
