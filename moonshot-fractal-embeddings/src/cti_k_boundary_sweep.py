#!/usr/bin/env python -u
"""
K-BOUNDARY SWEEP (Session 43 follow-up)
========================================
Pre-registered BEFORE running.

Codex recommendation: narrow the K-threshold window from (10, 59) by
testing K~26 and K~41.

Current picture:
  K=6  (emotion):        rho=0.613 FAIL
  K=10 (yahoo):          rho=0.835 FAIL
  K=59 (amazon_massive): rho=0.922 PASS
  K=77 (banking77):      rho=0.890 PASS

PRE-REGISTERED: Test two new intermediate-K datasets with frozen A(K) formula.
  Frozen A(K): a=1.1636, b=1.3013 (same throughout all prospective tests)
  Dataset 1: papluca/language-identification (K=20 languages) => A(K=20)=1.1636/log(20)+1.3013
  Dataset 2: khalidalt/HuffPost (K=41 news categories) => A(K=41)=1.1636/log(41)+1.3013

PRE-REGISTERED HYPOTHESES:
  H_bound: At least one of the two new datasets has rho >= 0.85
  H2: MAE <= 0.08 for both datasets
  H_threshold: rho >= 0.85 for K >= 30 (bisects the 10-59 window)

Same 6-architecture protocol as previous prospective tests.
"""

import json
import math
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = RESULTS_DIR
OUT_JSON = RESULTS_DIR / "cti_k_boundary_sweep.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

A_GLOBAL_a = 1.1636
A_GLOBAL_b = 1.3013


def A_pred(K):
    return A_GLOBAL_a / math.log(K) + A_GLOBAL_b


RHO_THRESH = 0.85
MAE_THRESH = 0.08

# Pre-register K values and predicted A
K_LANGID = 20
K_HUFFPOST = 41
A_LANGID = A_pred(K_LANGID)
A_HUFFPOST = A_pred(K_HUFFPOST)
print(f"Pre-registered: A(K={K_LANGID})={A_LANGID:.4f}, A(K={K_HUFFPOST})={A_HUFFPOST:.4f}", flush=True)

# Dataset configs
# papluca/language-identification: K=20 language classes (string labels "fr","en",...)
# khalidalt/HuffPost: K=41 news categories (integer labels 0-40)
# Note: K=20 was in A(K) fitting set (20newsgroups K=20), but the dataset itself is new.
# The K-threshold test is prospective: outcomes unknown, datasets never run through protocol.
NEW_DATASETS = {
    "langid": {
        "hf_name": "papluca/language-identification",
        "hf_cfg": None,
        "text_col": "text",
        "label_col": "labels",  # string codes like "fr","en",...
        "K_expected": K_LANGID,  # K=20 expected
        "n_sample_per_class": 30,
        "split": "test",
    },
    "news_category": {
        "hf_name": "heegyu/news-category-dataset",
        "hf_cfg": None,
        "text_col": "headline",
        "label_col": "category",  # string labels e.g. "POLITICS","WELLNESS",...
        "K_expected": K_HUFFPOST,  # K~41 expected
        "n_sample_per_class": 25,
        "split": "train",       # uses train split (no test split for news-category)
    },
}

MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/gpt-neo-125m",
    "Qwen/Qwen3-0.6B",
    "allenai/OLMo-1B-hf",
    "tiiuae/Falcon-H1-0.5B-Base",
    "RWKV/rwkv-4-169m-pile",
]

MODEL_LAYERS = {
    "pythia-160m": [3, 6, 9, 12],
    "gpt-neo-125m": [3, 6, 9, 12],
    "Qwen3-0.6B": [7, 14, 21, 28],
    "OLMo-1B-hf": [4, 8, 12, 16],
    "Falcon-H1-0.5B-Base": [9, 18, 27, 36],
    "rwkv-4-169m-pile": [3, 6, 9, 12],
}

TRUST_REMOTE_CODE_MODELS = {"Falcon-H1-0.5B-Base"}
BATCH_SIZE = 32


def load_dataset_texts(ds_name, config):
    hf_name = config["hf_name"]
    hf_cfg = config.get("hf_cfg")
    text_col = config["text_col"]
    label_col = config["label_col"]
    n_per_class = config.get("n_sample_per_class", 25)
    split = config.get("split", "test")
    try:
        if hf_cfg:
            ds = load_dataset(hf_name, hf_cfg, split=split)
        else:
            ds = load_dataset(hf_name, split=split)
        texts_raw = [str(x[text_col]) for x in ds]
        labels_raw = [x[label_col] for x in ds]
    except Exception as e:
        print(f"  Error loading {hf_name}: {e}", flush=True)
        return None, None, 0
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels_raw)
    # Stratified sampling: at least n_per_class per class
    np.random.seed(42)
    classes = np.unique(labels_enc)
    selected_idx = []
    for c in classes:
        c_idx = np.where(labels_enc == c)[0]
        take = min(n_per_class, len(c_idx))
        if take < 2:
            continue  # skip degenerate classes
        chosen = np.random.choice(c_idx, take, replace=False)
        selected_idx.extend(chosen.tolist())
    selected_idx = np.array(selected_idx)
    texts = [texts_raw[i] for i in selected_idx]
    labels = labels_enc[selected_idx]
    K_actual = len(np.unique(labels))
    return texts, labels, K_actual


@torch.no_grad()
def get_embeddings_at_layers(model_name, texts, layers, batch_size=32):
    model_short = model_name.split("/")[-1]
    trust_remote = model_short in TRUST_REMOTE_CODE_MODELS
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, dtype=torch.float16,
        trust_remote_code=trust_remote,
    ).to(DEVICE).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    all_layer_embs = {l: [] for l in layers}
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(DEVICE)
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        for layer_idx in layers:
            if layer_idx < len(hidden_states):
                h = hidden_states[layer_idx].float()
                emb = (h * mask).sum(1) / mask.sum(1)
                emb_np = np.nan_to_num(emb.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                all_layer_embs[layer_idx].append(emb_np)
    del model
    torch.cuda.empty_cache()
    result = {}
    for l in layers:
        if all_layer_embs[l]:
            result[l] = np.vstack(all_layer_embs[l])
    return result


def compute_kappa_nearest(X, labels, K):
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    mu = {k: X[labels == k].mean(0) for k in classes if (labels == k).sum() >= 2}
    if len(mu) < 2:
        return None
    within_var = sum(np.sum((X[labels == k] - mu[k])**2)
                     for k in mu if (labels == k).sum() >= 2)
    n_total = sum((labels == k).sum() for k in mu if (labels == k).sum() >= 2)
    sigma_W = float(np.sqrt(within_var / (n_total * X.shape[1])))
    if sigma_W < 1e-10:
        return None
    denom = sigma_W * math.sqrt(X.shape[1])
    kappas = []
    for k in mu:
        min_dist = min(np.linalg.norm(mu[k] - mu[j]) for j in mu if j != k)
        kappas.append(min_dist / denom)
    return float(np.mean(kappas))


def compute_q(X, labels, K, n_splits=5):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedShuffleSplit
    if len(X) < 20:
        return None
    # Drop classes with < 2 samples
    counts = np.bincount(labels)
    valid_classes = np.where(counts >= 2)[0]
    if len(valid_classes) < 2:
        return None
    mask = np.isin(labels, valid_classes)
    X, labels = X[mask], labels[mask]
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    accs = []
    for train_idx, test_idx in cv.split(X, labels):
        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
        knn.fit(X[train_idx], labels[train_idx])
        accs.append(knn.score(X[test_idx], labels[test_idx]))
    acc = float(np.mean(accs))
    K_eff = float(len(np.unique(labels)))
    q = (acc - 1.0/K_eff) / (1.0 - 1.0/K_eff)
    return float(np.clip(q, 1e-5, 1 - 1e-5))


def json_default(obj):
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    print("=" * 70)
    print("K-BOUNDARY SWEEP (Session 43 follow-up)")
    print(f"Frozen A(K): a={A_GLOBAL_a}, b={A_GLOBAL_b}")
    print(f"Datasets: langid K={K_LANGID}, huffpost K={K_HUFFPOST}")
    print(f"A(K={K_LANGID})={A_LANGID:.4f}, A(K={K_HUFFPOST})={A_HUFFPOST:.4f}")
    print(f"H1: rho>={RHO_THRESH}, H2: MAE<={MAE_THRESH}")
    print("=" * 70)

    all_results = []

    for ds_name, ds_config in NEW_DATASETS.items():
        K_expected = ds_config["K_expected"]
        A = A_pred(K_expected)
        print(f"\n{'='*60}", flush=True)
        print(f"Dataset: {ds_name}  K_expected={K_expected}  A_pred={A:.4f}", flush=True)

        texts, labels, K_actual = load_dataset_texts(ds_name, ds_config)
        if texts is None:
            print(f"  SKIPPED: failed to load", flush=True)
            continue
        print(f"  Loaded {len(texts)} samples, {K_actual} classes", flush=True)

        # Use actual K for A prediction
        A_actual = A_pred(K_actual)
        print(f"  Actual K={K_actual}, A_pred={A_actual:.4f}", flush=True)

        cache_by_model = defaultdict(list)

        for model_name in MODELS:
            model_short = model_name.split("/")[-1]
            cache_path = CACHE_DIR / f"kappa_near_cache_{ds_name}_{model_short}.json"

            if cache_path.exists():
                with open(cache_path) as f:
                    pts = json.load(f)
                for pt in pts:
                    cache_by_model[model_short].append(pt)
                print(f"  {model_short}: loaded from cache ({len(pts)} pts)", flush=True)
                continue

            print(f"  {model_short}: extracting embeddings...", flush=True)
            layers = MODEL_LAYERS.get(model_short, [3, 6, 9, 12])
            try:
                layer_embs = get_embeddings_at_layers(model_name, texts, layers)
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                continue

            model_pts = []
            for layer_idx, embs in layer_embs.items():
                kappa = compute_kappa_nearest(embs, labels, K_actual)
                q = compute_q(embs, labels, K_actual)
                if kappa is None or q is None:
                    continue
                q_norm = float(np.clip(q, 1e-5, 1 - 1e-5))
                logit_q = float(math.log(q_norm / (1 - q_norm)))
                if not math.isfinite(logit_q) or not math.isfinite(kappa) or kappa <= 0:
                    continue
                pt = {
                    "model": model_short,
                    "dataset": ds_name,
                    "layer": int(layer_idx),
                    "K": K_actual,
                    "q": q_norm,
                    "kappa_nearest": kappa,
                    "logit_q": logit_q,
                }
                model_pts.append(pt)
                cache_by_model[model_short].append(pt)

            if model_pts:
                with open(cache_path, "w") as f:
                    json.dump(model_pts, f, indent=2, default=json_default)
                print(f"    -> {len(model_pts)} points saved", flush=True)

        models_with_data = [m for m in cache_by_model if len(cache_by_model[m]) >= 2]
        print(f"\n  Models with data: {models_with_data}", flush=True)
        if len(models_with_data) < 3:
            continue

        anchor_rhos = []
        anchor_maes = []

        for anchor_model in models_with_data:
            anchor_pts = cache_by_model[anchor_model]
            C_dataset = float(np.mean([p["logit_q"] - A_actual * p["kappa_nearest"]
                                       for p in anchor_pts]))
            q_pred_all, q_actual_all = [], []
            for other_model in models_with_data:
                if other_model == anchor_model:
                    continue
                for p in cache_by_model[other_model]:
                    logit_pred = A_actual * p["kappa_nearest"] + C_dataset
                    q_pred = float(np.clip(1.0 / (1.0 + math.exp(-logit_pred)), 1e-6, 1-1e-6))
                    q_pred_all.append(q_pred)
                    q_actual_all.append(p["q"])

            if len(q_actual_all) < 4:
                continue
            rho, _ = spearmanr(q_pred_all, q_actual_all)
            mae = float(np.mean(np.abs(np.array(q_pred_all) - np.array(q_actual_all))))
            anchor_rhos.append(float(rho))
            anchor_maes.append(float(mae))
            print(f"    anchor={anchor_model}: rho={rho:.4f}, MAE={mae:.4f}", flush=True)

        if not anchor_rhos:
            continue

        mean_rho = float(np.mean(anchor_rhos))
        mean_mae = float(np.mean(anchor_maes))
        pass_H1 = bool(mean_rho >= RHO_THRESH)
        pass_H2 = bool(mean_mae <= MAE_THRESH)
        print(f"\n  RESULT: rho={mean_rho:.4f} H1={'PASS' if pass_H1 else 'FAIL'}, "
              f"MAE={mean_mae:.4f} H2={'PASS' if pass_H2 else 'FAIL'}", flush=True)

        all_results.append({
            "dataset": ds_name,
            "K_expected": K_expected,
            "K_actual": K_actual,
            "A_pred": float(A_actual),
            "n_models": len(models_with_data),
            "mean_rho": mean_rho,
            "std_rho": float(np.std(anchor_rhos)),
            "mean_mae": mean_mae,
            "pass_H1": pass_H1,
            "pass_H2": pass_H2,
        })

    print("\n" + "=" * 70)
    print("SUMMARY: K-BOUNDARY SWEEP")
    print("=" * 70)
    for r in all_results:
        p1 = "PASS" if r["pass_H1"] else "FAIL"
        p2 = "PASS" if r["pass_H2"] else "FAIL"
        print(f"  {r['dataset']:>15} K={r['K_actual']:>3}: rho={r['mean_rho']:.4f} {p1}, "
              f"MAE={r['mean_mae']:.4f} {p2}")

    print("\nK-THRESHOLD FULL PICTURE (all prospective datasets):")
    full_picture = [
        ("emotion",       6,  0.613, False),
        ("yahoo",        10,  0.835, False),
    ]
    for r in all_results:
        full_picture.append((r["dataset"], r["K_actual"], r["mean_rho"], r["pass_H1"]))
    full_picture += [
        ("amazon_massive", 59, 0.922, True),
        ("banking77",      77, 0.890, True),
    ]
    full_picture.sort(key=lambda x: x[1])
    for name, K, rho, passed in full_picture:
        p = "PASS" if passed else "FAIL"
        print(f"  K={K:<3} ({name:<16}): rho={rho:.3f} {p}")

    n_H1 = sum(1 for r in all_results if r["pass_H1"])
    n_H2 = sum(1 for r in all_results if r["pass_H2"])
    print(f"\nNew datasets: H1 {n_H1}/{len(all_results)}, H2 {n_H2}/{len(all_results)}")

    output = {
        "experiment": "k_boundary_sweep",
        "session": 43,
        "preregistered": {
            "A_K_formula": "A(K) = 1.1636/log(K) + 1.3013",
            "A_langid_K20": float(A_LANGID),
            "A_huffpost_K41": float(A_HUFFPOST),
            "hypothesis": "rho >= 0.85 requires K >= 30 (bisects 10-59 window)",
            "H1_rho_threshold": RHO_THRESH,
            "H2_mae_threshold": MAE_THRESH,
        },
        "results": all_results,
        "full_k_sweep": [
            {"dataset": name, "K": K, "rho": rho, "pass_H1": passed}
            for name, K, rho, passed in full_picture
        ],
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
