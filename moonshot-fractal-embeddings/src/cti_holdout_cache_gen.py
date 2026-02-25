#!/usr/bin/env python -u
"""
Generate kappa_nearest cache for HOLDOUT models (H8+ expanded factorized test).

Pre-registered at commit 719da25 (PREREGISTRATION_expanded_holdout_h8plus.md).
Target: 12 models x 8 datasets = 96 predictions.

Models (all UNSEEN during CTI law training):
  1. roberta-base          (encoder, 125M, 12L)
  2. distilbert-base-uncased (encoder, 66M, 6L)
  3. albert-base-v2        (encoder/ALBERT, 12M, 12L)
  4. facebook/opt-125m     (decoder/OPT, 125M, 12L)
  5. EleutherAI/pythia-2.8b (decoder/Pythia, 2.8B, 32L)
  6. stabilityai/stablelm-3b-4e1t (decoder/StableLM, 3B, 32L)
  7. gemma-3-1b            (decoder/Gemma, 1B, 26L)
  8. bigscience/bloom-560m (decoder/BLOOM, 560M, 24L)
  9. meta-llama/Llama-3.2-1B (decoder/Llama, 1B, 16L) — NEW family
 10. microsoft/phi-1_5      (decoder/Phi, 1.3B, 24L) — NEW family
 11. Qwen/Qwen2.5-1.5B     (decoder/Qwen, 1.5B, 28L) — size generalization
 12. tiiuae/falcon-rw-1b    (decoder/Falcon, 1B, 24L) — NEW family

Datasets: 4 core + 4 extended = 8 total
"""

import json, os, sys, time, gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from datasets import load_dataset as hf_load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
BATCH_SIZE = 64
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# 8 datasets: 4 core + 4 extended (matching training pipeline configs)
DATASETS = {
    # Core 4
    "agnews":       {"hf_name": "fancyzhx/ag_news",    "text_col": "text",    "label_col": "label",       "K": 4,  "n_sample": 1000},
    "dbpedia":      {"hf_name": "fancyzhx/dbpedia_14", "text_col": "content", "label_col": "label",       "K": 14, "n_sample": 1000},
    "20newsgroups": {"hf_name": "SetFit/20_newsgroups", "text_col": "text",   "label_col": "label_text",  "K": 20, "n_sample": 1000},
    "go_emotions":  {"hf_name": "google-research-datasets/go_emotions", "hf_cfg": "simplified",
                     "text_col": "text", "label_col": "labels", "K": 28, "n_sample": 1000, "multilabel": True},
    # Extended 4
    "banking77":       {"hf_name": "mteb/banking77",         "text_col": "text", "label_col": "label", "K": 77, "n_sample": 1500},
    "emotion":         {"hf_name": "dair-ai/emotion",        "text_col": "text", "label_col": "label", "K": 6,  "n_sample": 1000},
    "yahoo":           {"hf_name": "yahoo_answers_topics",   "text_col": "question_content", "label_col": "topic", "K": 10, "n_sample": 1000},
    "amazon_massive":  {"hf_name": "mteb/amazon_massive_intent", "text_col": "text", "label_col": "label", "K": 60, "n_sample": 1000,
                        "filter_lang": "en"},
}

# (hf_id, short_name, layers_at_25_50_75_100_pct)
HOLDOUT_MODELS = [
    ("roberta-base",                      "roberta-base",           [3, 6, 9, 12]),
    ("distilbert-base-uncased",           "distilbert-base-uncased", [2, 3, 5, 6]),
    ("albert-base-v2",                    "albert-base-v2",         [3, 6, 9, 12]),
    ("facebook/opt-125m",                 "opt-125m",               [3, 6, 9, 12]),
    ("EleutherAI/pythia-2.8b",            "pythia-2.8b",            [8, 16, 24, 32]),
    ("stabilityai/stablelm-3b-4e1t",      "stablelm-3b-4e1t",      [8, 16, 24, 32]),
    ("google/gemma-3-1b-pt",              "gemma-3-1b",             [5, 12, 19, 25]),
    ("bigscience/bloom-560m",             "bloom-560m",             [6, 12, 18, 24]),
    # Round 2: 4 more models to push n > 60
    ("meta-llama/Llama-3.2-1B",           "llama-3.2-1b",          [4, 8, 12, 16]),
    ("microsoft/phi-1_5",                 "phi-1.5",               [6, 12, 18, 24]),
    ("Qwen/Qwen2.5-1.5B",                "qwen2.5-1.5b",          [7, 14, 21, 28]),
    ("tiiuae/falcon-rw-1b",              "falcon-rw-1b",           [6, 12, 18, 24]),
]


def load_dataset_texts_labels(dataset_name, config):
    """Load dataset, return (texts, labels) subsample."""
    hf_name = config["hf_name"]
    text_col = config["text_col"]
    label_col = config["label_col"]
    n_sample = config["n_sample"]
    hf_cfg = config.get("hf_cfg")
    multilabel = config.get("multilabel", False)

    filter_lang = config.get("filter_lang")

    try:
        if hf_cfg:
            ds = hf_load_dataset(hf_name, hf_cfg, split="test")
        else:
            # Try test split first, fall back to train
            try:
                ds = hf_load_dataset(hf_name, split="test")
            except Exception:
                ds = hf_load_dataset(hf_name, split="train")

        # Filter by language if needed (amazon_massive)
        if filter_lang:
            ds = ds.filter(lambda x: x.get("lang", x.get("language", "")) == filter_lang)

        texts = [str(x[text_col]) for x in ds]
        labels_raw = [x[label_col] for x in ds]
    except Exception as e:
        print(f"  Error loading {hf_name}: {e}", flush=True)
        return None, None

    if multilabel:
        labels_raw = [l[0] if isinstance(l, list) and l else 0 for l in labels_raw]

    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)

    np.random.seed(42)
    idx = np.random.choice(len(texts), min(n_sample, len(texts)), replace=False)
    texts = [texts[i] for i in idx]
    labels = labels[idx]

    return texts, labels


def compute_kappa_nearest(embeddings, labels, K, subsample=500):
    """Compute kappa_nearest = min_{j!=k} ||mu_k - mu_j|| / (sigma_W * sqrt(d))."""
    X, y = embeddings, labels
    classes = np.unique(y)
    if len(classes) != K:
        K = len(classes)
    if K < 2:
        return None, None

    if len(X) > subsample:
        idx = np.random.choice(len(X), subsample, replace=False)
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
        within_var += np.sum((Xk - mean_k) ** 2)
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
    """1-NN quality q_norm = (acc - 1/K) / (1 - 1/K)."""
    if len(embeddings) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), subsample, replace=False)
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

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = (acc - 1.0 / K_eff) / (1.0 - 1.0 / K_eff)
    return float(q)


@torch.no_grad()
def process_model(hf_id, short_name, layers, dataset_cache):
    """Load model ONCE, extract embeddings for all needed datasets."""
    print(f"\n{'='*60}")
    print(f"MODEL: {short_name} ({hf_id})")
    print(f"{'='*60}")

    # Check which datasets need computation
    needed = {}
    for ds_name, (texts, labels, K) in dataset_cache.items():
        cache_path = os.path.join(CACHE_DIR, f"kappa_near_cache_{ds_name}_{short_name}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    content = f.read().strip()
                if content and json.loads(content):
                    print(f"  {ds_name}: SKIP (cached)", flush=True)
                    continue
            except Exception:
                pass
        needed[ds_name] = (texts, labels, K)

    if not needed:
        print(f"  All datasets cached, skipping model entirely")
        return 0, 0

    # Load model
    t_load = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        cfg = AutoConfig.from_pretrained(hf_id)
        n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
        print(f"  Config: {n_layers} layers, hidden={getattr(cfg, 'hidden_size', '?')}", flush=True)

        model = AutoModel.from_pretrained(
            hf_id, output_hidden_states=True, torch_dtype=torch.float16,
        ).to(DEVICE).eval()
    except Exception as e:
        print(f"  ERROR loading model: {e}", flush=True)
        return 0, len(needed)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Clamp layers to actual model depth
    max_layer = n_layers if n_layers else max(layers)
    safe_layers = [l for l in layers if l <= max_layer]
    if safe_layers != layers:
        print(f"  WARNING: clamped layers {layers} -> {safe_layers} (model has {n_layers} layers)")
        layers = safe_layers

    print(f"  Model loaded in {time.time()-t_load:.1f}s, layers={layers}", flush=True)

    generated = 0
    failed = 0

    for ds_name, (texts, labels, K) in needed.items():
        print(f"\n  Dataset: {ds_name} (K={K}, n={len(texts)})", flush=True)
        t_ds = time.time()

        all_layer_embs = {l: [] for l in layers}
        all_labels = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_labels = labels[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=128,
            ).to(DEVICE)

            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            for layer_idx in layers:
                if layer_idx < len(hidden_states):
                    h = hidden_states[layer_idx].float()
                    emb = (h * mask).sum(1) / mask.sum(1)
                    emb_np = np.nan_to_num(emb.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                    all_layer_embs[layer_idx].append(emb_np)

            all_labels.extend(batch_labels)

        y = np.array(all_labels)
        for l in layers:
            if all_layer_embs[l]:
                all_layer_embs[l] = np.vstack(all_layer_embs[l])
            else:
                all_layer_embs[l] = None

        # Compute kappa and q for each layer
        model_points = []
        for layer in layers:
            embs = all_layer_embs.get(layer)
            if embs is None:
                continue

            valid_mask = np.isfinite(embs).all(axis=1)
            n_invalid = (~valid_mask).sum()
            if n_invalid > 0:
                print(f"    Layer {layer}: filtering {n_invalid} NaN/Inf rows", flush=True)
            embs = embs[valid_mask]
            y_layer = y[valid_mask]
            if len(embs) < 20:
                print(f"    Layer {layer}: too few valid ({len(embs)}), skip", flush=True)
                continue

            q = compute_knn_q(embs, y_layer, K)
            kappa_near, kappa_min = compute_kappa_nearest(embs, y_layer, K)

            if q is None or kappa_near is None:
                continue

            pt = {
                "model": short_name, "dataset": ds_name,
                "layer": layer, "K": K,
                "q": q, "kappa_nearest": kappa_near, "kappa_min": kappa_min,
                "logit_q": float(np.log(max(q, 0.001) / max(1 - q, 0.001))),
                "logKm1": float(np.log(K - 1)) if K > 1 else 0.0,
            }
            model_points.append(pt)
            print(f"    L{layer}: q={q:.4f}  kappa={kappa_near:.4f}  logit(q)={pt['logit_q']:.4f}",
                  flush=True)

        cache_path = os.path.join(CACHE_DIR, f"kappa_near_cache_{ds_name}_{short_name}.json")
        if model_points:
            with open(cache_path, "w") as f:
                json.dump(model_points, f, indent=2,
                          default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
            print(f"    -> Saved {len(model_points)} pts ({time.time()-t_ds:.1f}s)", flush=True)
            generated += 1
        else:
            print(f"    WARNING: no valid points for {ds_name}", flush=True)
            failed += 1

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return generated, failed


def main():
    t0 = time.time()
    print("=" * 70)
    print("HOLDOUT KAPPA CACHE GENERATION (factorized prospective test)")
    print(f"Models: {[m[1] for m in HOLDOUT_MODELS]}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Total combinations: {len(HOLDOUT_MODELS) * len(DATASETS)}")
    print("=" * 70)

    # Preload all datasets
    print("\nPreloading datasets...")
    dataset_cache = {}
    for ds_name, config in DATASETS.items():
        texts, labels = load_dataset_texts_labels(ds_name, config)
        if texts is not None:
            dataset_cache[ds_name] = (texts, labels, config["K"])
            print(f"  {ds_name}: {len(texts)} samples, K={config['K']}")
        else:
            print(f"  {ds_name}: FAILED to load")

    total_gen = 0
    total_fail = 0

    for hf_id, short_name, layers in HOLDOUT_MODELS:
        gen, fail = process_model(hf_id, short_name, layers, dataset_cache)
        total_gen += gen
        total_fail += fail

    elapsed = int(time.time() - t0)
    print(f"\n{'='*70}")
    print(f"COMPLETE. Generated={total_gen}, Failed={total_fail}")
    print(f"Runtime: {elapsed}s ({elapsed // 60}m {elapsed % 60}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
