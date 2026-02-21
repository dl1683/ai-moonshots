#!/usr/bin/env python -u
"""
BGE-SMALL CAUSAL COMPARISON (Feb 21 2026)
==========================================
NATURAL EXPERIMENT: BGE-small-en-v1.5 vs BERT-base-uncased

BGE-small is a BERT-base backbone fine-tuned with contrastive learning (InfoNCE
on text similarity pairs). BERT-base has NO fine-tuning for class separation.

Theory prediction:
  Contrastive training DIRECTLY optimizes kappa_nearest:
    - Pushes same-meaning texts together (reduces sigma_W)
    - Pushes different-meaning texts apart (increases mu_k - mu_j distances)
  Therefore: kappa_BGE > kappa_BERT -> q_BGE > q_BERT

This is a NATURAL CAUSAL EXPERIMENT:
  - Control: BERT-base (no class-separation training)
  - Treatment: BGE-small (contrastive training for separation)
  - Same backbone architecture (BERT)
  - Same dimension (d=768 for bge-small-en-v1.5... actually d=384)

NOTE: BGE-small-en-v1.5 has d=384 (smaller than BERT's 768)
This means kappa values are not directly comparable (different d).
Solution: Compare RELATIVE improvement (kappa_{BGE}/kappa_{BERT} and q_{BGE}/q_{BERT})
and check consistency with alpha predictions.

Pre-registered hypothesis:
  1. q_BGE > q_BERT for all topic classification tasks (directional)
  2. (q_BGE - q_BERT) consistent with alpha * (kappa_BGE - kappa_BERT)
     after accounting for dimension difference in kappa formula

Use bge-base-en-v1.5 (d=768, same as BERT) for cleaner comparison.

Models:
  - BERT-base: bert-base-uncased (d=768, MLM pretraining, no class training)
  - BGE-base:  BAAI/bge-base-en-v1.5 (d=768, contrastive fine-tuned)
"""

import json
import os
import sys
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
# CONFIG
# ================================================================
MODELS = {
    "bert-base-uncased": {
        "hf_name": "bert-base-uncased",
        "d": 768,
        "layers": [4, 8, 10, 12],   # match existing BERT cache (layers from kappa_nearest_extended)
        "type": "MLM (no class training)",
    },
    "bge-base-v1.5": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "d": 768,
        "layers": [4, 8, 10, 12],   # same as BERT for fair comparison
        "type": "Contrastive fine-tuned (InfoNCE)",
    },
}

DATASETS = {
    "agnews": {"K": 4, "n_sample": 2000},
    "dbpedia": {"K": 14, "n_sample": 2000},
    "20newsgroups": {"K": 20, "n_sample": 2000},
    "go_emotions": {"K": 28, "n_sample": 5000},
}

BATCH_SIZE = 128
FROZEN_TASK_ALPHA = 1.54
FROZEN_C_TASK = {
    'agnews': 0.483,
    'dbpedia': 1.424,
    '20newsgroups': -0.348,
    'go_emotions': -1.056,
}


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


def get_embeddings(model_name, hf_name, layers, texts, labels):
    print(f"  Loading {hf_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        hf_name,
        output_hidden_states=True,
    ).to(DEVICE)
    model.eval()

    print(f"  Extracting embeddings ({len(texts)} texts)...", flush=True)
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

        for layer_idx in layers:
            if layer_idx < len(hidden_states):
                h = hidden_states[layer_idx].float()
                emb = (h * mask).sum(1) / mask.sum(1)
                all_layer_embs[layer_idx].append(emb.cpu().numpy())

        all_labels.extend(
            batch_labels.tolist() if hasattr(batch_labels, 'tolist') else list(batch_labels)
        )

        if (i // BATCH_SIZE) % 5 == 0:
            print(f"    Batch {i//BATCH_SIZE}/{(len(texts)+BATCH_SIZE-1)//BATCH_SIZE}...", flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"  Done in {int(time.time()-t0)}s", flush=True)

    result = {}
    for l in layers:
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
    print("BGE vs BERT CAUSAL COMPARISON")
    print("Hypothesis: contrastive training -> higher kappa_nearest -> higher q")
    print("=" * 70, flush=True)

    all_results = {}

    for model_key, model_cfg in MODELS.items():
        hf_name = model_cfg["hf_name"]
        layers = model_cfg["layers"]
        model_results = []

        # Check per-dataset cache
        cache_prefix = f"results/kappa_near_cache"

        for ds_name, ds_cfg in DATASETS.items():
            K = ds_cfg["K"]

            # Map model key to cache filename
            cache_key = model_key.replace("/", "_").replace(".", "-")
            per_ds_cache = f"{cache_prefix}_{ds_name}_{cache_key}.json"

            if os.path.exists(per_ds_cache):
                with open(per_ds_cache) as f:
                    cached = json.load(f)
                if cached:
                    print(f"\n{model_key} / {ds_name}: Loaded from cache ({len(cached)} pts)")
                    for p in cached:
                        p['model_key'] = model_key
                    model_results.extend(cached)
                    continue

            print(f"\n{'='*50}")
            print(f"Model: {model_key} | Dataset: {ds_name} (K={K})")
            print(f"{'='*50}", flush=True)

            try:
                texts, y = load_dataset_texts(ds_name, ds_cfg["n_sample"])
            except Exception as e:
                print(f"  SKIP: {e}", flush=True)
                continue

            try:
                layer_embs, y_emb = get_embeddings(
                    model_key, hf_name, layers, texts, y
                )
            except Exception as e:
                print(f"  Embedding error: {e}", flush=True)
                continue

            ds_points = []
            for layer in layers:
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
                C_task = FROZEN_C_TASK.get(ds_name, 0.0)
                logit_pred = FROZEN_TASK_ALPHA * kn + C_task
                q_pred = float(1.0 / (1.0 + np.exp(-logit_pred)))

                pt = {
                    "model": model_key,
                    "model_type": model_cfg["type"],
                    "dataset": ds_name,
                    "layer": int(layer),
                    "K": K,
                    "q": float(q),
                    "kappa_nearest": float(kn),
                    "kappa_min": float(km),
                    "logit_q": float(logit_q),
                    "q_pred_task": float(q_pred),
                    "error_task": float(abs(q_pred - q)),
                }
                ds_points.append(pt)
                print(f"  Layer {layer:2d}: q={q:.4f}  kappa={kn:.4f}  "
                      f"pred={q_pred:.4f}  err={abs(q_pred-q):.4f}", flush=True)

            if ds_points:
                with open(per_ds_cache, "w") as f:
                    json.dump(ds_points, f, indent=2,
                              default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
                print(f"  Cached {len(ds_points)} pts -> {per_ds_cache}", flush=True)
                model_results.extend(ds_points)

        all_results[model_key] = model_results

    # ================================================================
    # CAUSAL COMPARISON: BGE vs BERT
    # ================================================================
    print("\n" + "=" * 70)
    print("CAUSAL COMPARISON: BGE-BASE vs BERT-BASE")
    print("=" * 70)

    bert_pts = all_results.get("bert-base-uncased", [])
    bge_pts = all_results.get("bge-base-v1.5", [])

    if not bert_pts or not bge_pts:
        print("Missing data for comparison!")
        return

    print("\nPer-dataset-per-layer comparison:")
    print(f"{'Dataset':15s} {'Layer':5s} {'q_BERT':8s} {'q_BGE':8s} {'Dq':8s} {'k_BERT':8s} {'k_BGE':8s} {'Dk':8s}")
    print("-" * 70)

    # Get common layers from data (not hardcoded)
    bert_layers = sorted(set(p['layer'] for p in bert_pts))
    bge_layers = sorted(set(p['layer'] for p in bge_pts))
    common_layers = sorted(set(bert_layers) & set(bge_layers))
    print(f"BERT layers: {bert_layers}, BGE layers: {bge_layers}, common: {common_layers}")

    comparison_pts = []
    for ds in ['agnews', 'dbpedia', '20newsgroups', 'go_emotions']:
        for layer in common_layers:
            b_pt = next((p for p in bert_pts if p['dataset'] == ds and p['layer'] == layer), None)
            g_pt = next((p for p in bge_pts if p['dataset'] == ds and p['layer'] == layer), None)
            if b_pt is None or g_pt is None:
                continue

            dq = g_pt['q'] - b_pt['q']
            dk = g_pt['kappa_nearest'] - b_pt['kappa_nearest']

            print(f"  {ds:13s} {layer:5d} {b_pt['q']:8.4f} {g_pt['q']:8.4f} {dq:+8.4f} "
                  f"{b_pt['kappa_nearest']:8.4f} {g_pt['kappa_nearest']:8.4f} {dk:+8.4f}")

            comparison_pts.append({
                "dataset": ds,
                "layer": layer,
                "q_bert": b_pt['q'],
                "q_bge": g_pt['q'],
                "delta_q": dq,
                "kappa_bert": b_pt['kappa_nearest'],
                "kappa_bge": g_pt['kappa_nearest'],
                "delta_kappa": dk,
            })

    # Summary: directional hypothesis test
    dqs = [p['delta_q'] for p in comparison_pts]
    dks = [p['delta_kappa'] for p in comparison_pts]

    n_dq_positive = sum(1 for dq in dqs if dq > 0)
    n_dk_positive = sum(1 for dk in dks if dk > 0)
    n = len(comparison_pts)

    print(f"\nDirectional test: q_BGE > q_BERT in {n_dq_positive}/{n} = {n_dq_positive/n*100:.0f}%")
    print(f"Directional test: kappa_BGE > kappa_BERT in {n_dk_positive}/{n} = {n_dk_positive/n*100:.0f}%")

    # Binomial p-value (H0: equal probability)
    from scipy import stats
    try:
        binom_q = stats.binomtest(n_dq_positive, n, 0.5, alternative='greater').pvalue
        binom_k = stats.binomtest(n_dk_positive, n, 0.5, alternative='greater').pvalue
    except AttributeError:
        # older scipy
        binom_q = stats.binom_test(n_dq_positive, n, 0.5, alternative='greater') if n_dq_positive > 0 else 1.0
        binom_k = stats.binom_test(n_dk_positive, n, 0.5, alternative='greater') if n_dk_positive > 0 else 1.0
    print(f"Binomial p-value (q): {binom_q:.4f}")
    print(f"Binomial p-value (kappa): {binom_k:.4f}")

    # Quantitative: check if delta_q consistent with alpha * delta_kappa
    # Expected: delta_logit_q ~= alpha * delta_kappa (first-order approximation)
    # delta_q ~= alpha * delta_kappa * q*(1-q) [logistic derivative]
    print(f"\nQuantitative consistency check:")
    print(f"Theory: delta_logit_q = alpha * delta_kappa  (alpha=1.54)")
    for ds in ['agnews', 'dbpedia', '20newsgroups']:
        pts_ds = [p for p in comparison_pts if p['dataset'] == ds]
        for p in sorted(pts_ds, key=lambda x: x['layer']):
            # logit of BERT and BGE
            b_logit = float(np.log(max(p['q_bert'], 1e-4) / max(1-p['q_bert'], 1e-4)))
            g_logit = float(np.log(max(p['q_bge'], 1e-4) / max(1-p['q_bge'], 1e-4)))
            d_logit = g_logit - b_logit
            predicted_d_logit = FROZEN_TASK_ALPHA * p['delta_kappa']
            print(f"  {ds}/{p['layer']}: delta_logit={d_logit:+.4f}  predicted={predicted_d_logit:+.4f}  ratio={d_logit/predicted_d_logit:.2f}"
                  if abs(predicted_d_logit) > 0.01 else
                  f"  {ds}/{p['layer']}: delta_kappa~0 (skip)")

    # Save results
    output = {
        "experiment": "bge_bert_causal_comparison",
        "hypothesis": "contrastive training (BGE) -> higher kappa_nearest -> higher q",
        "frozen_alpha": FROZEN_TASK_ALPHA,
        "model_types": {k: v["type"] for k, v in MODELS.items()},
        "comparison": comparison_pts,
        "summary": {
            "n_total": n,
            "n_dq_positive": int(n_dq_positive),
            "n_dk_positive": int(n_dk_positive),
            "pct_dq_positive": float(n_dq_positive/n) if n > 0 else 0.0,
            "pct_dk_positive": float(n_dk_positive/n) if n > 0 else 0.0,
            "binom_p_q": float(binom_q),
            "binom_p_kappa": float(binom_k),
            "passed_directional": bool(n_dq_positive/n > 0.75),
        },
        "bert_points": bert_pts,
        "bge_points": bge_pts,
    }

    out_path = "results/cti_bge_bert_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT:")
    if n_dq_positive/n > 0.75 and binom_q < 0.05:
        print("  PASS: BGE has significantly higher q than BERT (directional causal)")
        print("  Supports: contrastive training -> kappa_nearest increase -> q increase")
    elif n_dq_positive/n > 0.6:
        print("  WEAK PASS: BGE tends to have higher q (not significant)")
    else:
        print("  FAIL: BGE does not consistently outperform BERT in q")
        print("  Note: BGE might not improve q on topic classification tasks")
        print("  (BGE optimizes sentence similarity, not topic classification)")


if __name__ == "__main__":
    main()
