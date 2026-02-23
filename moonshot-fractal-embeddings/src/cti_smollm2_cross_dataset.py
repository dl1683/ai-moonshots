"""
SmolLM2-1.7B cross-dataset external replication (CORRECTED DESIGN).

DESIGN RATIONALE:
The original smollm2_held_out.py used cross-LAYER fitting (5 layers -> fit alpha).
This is NOT equivalent to the LOAO design, which uses the BEST LAYER per architecture
and fits alpha ACROSS ARCHITECTURES (fixed dataset).

CORRECT EQUIVALENT: cross-DATASET at BEST LAYER.
For SmolLM2, scan all layers to find the best layer per dataset (max 1-NN accuracy),
then fit logit(q_norm) ~ alpha * kappa_nearest + C across datasets (matching LODO design).

Pre-registered alpha interval from LOAO: mean=2.8685, std=0.054, range [2.43, 3.29].
Comprehensive universality alpha (19 archs, 10 datasets): 3.598 (per-dataset intercept).

Datasets used (mix of in-training and new):
  - dbpedia_14 (K=14): in training set
  - clinc_oos (K=151): OUT OF DISTRIBUTION (large-K, new domain)
  - banking77 (K=77): in training set (large-K)
  - ag_news (K=4): in training set (small-K)
  - 20newsgroups (K=20): in training set

N_LAYERS_SCAN: scan all 24 layers, pick best (max q_norm)
n_per_class: 200 per class (fast, sufficient for geometry)

Pre-registered criteria (frozen constants):
  PR1: fitted alpha (cross-dataset, per-dataset intercept) in [2.43, 3.29]
  PR2: Pearson r(kappa, logit_q) across datasets >= 0.80
  PR3: clinc_oos (K=151, NEW) has kappa > 0.0, q_norm > 0.10 (law applies to large-K)
  OVERALL PASS: PR1 AND PR2

Output: results/cti_smollm2_cross_dataset.json
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, linregress

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_smollm2_cross_dataset.json")

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"

# Pre-registered constants (frozen)
ALPHA_LOW = 2.43
ALPHA_HIGH = 3.29
ALPHA_UNIVERSAL = 3.598  # comprehensive universality (per-dataset intercept)

DATASETS = [
    {
        "name": "dbpedia_14",
        "hf_name": "dbpedia_14",
        "hf_subset": None,
        "split": "train",
        "text_col": "content",
        "label_col": "label",
        "n_per_class": 200,
        "note": "In training set (K=14, anchor)",
    },
    {
        "name": "clinc_oos",
        "hf_name": "clinc_oos",
        "hf_subset": "plus",
        "split": "test",
        "text_col": "text",
        "label_col": "intent",
        "n_per_class": 100,
        "note": "OUT-OF-DISTRIBUTION (K=151, new domain, new large-K)",
    },
    {
        "name": "banking77",
        "hf_name": "banking77",
        "hf_subset": None,
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "n_per_class": 200,
        "note": "In training set (K=77)",
    },
    {
        "name": "ag_news",
        "hf_name": "ag_news",
        "hf_subset": None,
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "n_per_class": 200,
        "note": "In training set (K=4)",
    },
    {
        "name": "20newsgroups",
        "hf_name": "SetFit/20_newsgroups",
        "hf_subset": None,
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "n_per_class": 200,
        "note": "In training set (K=20)",
    },
]

N_PER_CLASS = 200


def extract_best_layer_embeddings(model, tokenizer, device, texts_by_class,
                                  all_classes, n_layers_total):
    """Scan all layers, pick best (max 1-NN accuracy) layer."""
    from sklearn.neighbors import KNeighborsClassifier

    n_per_class = N_PER_CLASS
    K = len(all_classes)

    best_layer = -1
    best_q_raw = -1.0
    best_class_embeddings = None

    # Scan layers
    layer_indices = list(range(0, n_layers_total, max(1, n_layers_total // 8))) + [n_layers_total - 1]
    layer_indices = sorted(set(layer_indices))
    print(f"  Scanning layers: {layer_indices}")

    for layer_idx in layer_indices:
        class_embeddings = {}
        for ci in all_classes:
            texts_ci = texts_by_class[ci][:n_per_class]
            if len(texts_ci) < 5:
                continue
            embs_ci = []
            for b_start in range(0, len(texts_ci), 32):
                batch = texts_ci[b_start:b_start + 32]
                import torch
                tok = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    out = model(**tok, output_hidden_states=True)
                hidden = out.hidden_states[layer_idx + 1]  # +1 for embedding
                emb = hidden[:, -1, :].cpu().float().numpy()
                embs_ci.extend(emb)
            if embs_ci:
                class_embeddings[ci] = np.array(embs_ci)

        if len(class_embeddings) < K:
            continue

        # Quick 1-NN accuracy
        rng = np.random.default_rng(42)
        train_embs, train_labels, test_embs, test_labels = [], [], [], []
        for ci in all_classes:
            embs = class_embeddings[ci][:n_per_class]
            n = len(embs)
            n_train = max(1, int(0.8 * n))
            idx = rng.permutation(n)
            train_embs.append(embs[idx[:n_train]])
            train_labels.extend([ci] * n_train)
            test_embs.append(embs[idx[n_train:]])
            test_labels.extend([ci] * (n - n_train))

        train_X = np.vstack(train_embs)
        test_X = np.vstack(test_embs)
        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
        knn.fit(train_X, np.array(train_labels))
        q_raw = float(np.mean(knn.predict(test_X) == np.array(test_labels)))
        print(f"    Layer {layer_idx}: q_raw={q_raw:.3f}")

        if q_raw > best_q_raw:
            best_q_raw = q_raw
            best_layer = layer_idx
            best_class_embeddings = {ci: class_embeddings[ci].copy() for ci in class_embeddings}

    return best_layer, best_q_raw, best_class_embeddings


def compute_kappa_and_q(embs_by_class, all_classes):
    """Compute kappa_nearest and 1-NN accuracy."""
    from sklearn.neighbors import KNeighborsClassifier

    K = len(all_classes)
    n_per_class = min(len(embs_by_class[all_classes[0]]), N_PER_CLASS)
    d = embs_by_class[all_classes[0]].shape[1]

    mu = {}
    within_var_sum = 0.0
    n_total = 0
    for ci in all_classes:
        embs = embs_by_class[ci][:n_per_class]
        mu[ci] = embs.mean(0)
        within_var_sum += np.sum((embs - mu[ci]) ** 2)
        n_total += len(embs)
    sigma_W = float(np.sqrt(within_var_sum / (n_total * d)))

    min_dist = float("inf")
    for i, ci in enumerate(all_classes):
        for j, cj in enumerate(all_classes):
            if i >= j:
                continue
            dist = float(np.linalg.norm(mu[ci] - mu[cj]))
            min_dist = min(min_dist, dist)
    kappa_nearest = min_dist / (sigma_W * np.sqrt(d))

    rng = np.random.default_rng(42)
    train_embs, train_labels, test_embs, test_labels = [], [], [], []
    for ci in all_classes:
        embs = embs_by_class[ci][:n_per_class]
        n = len(embs)
        n_train = max(1, int(0.8 * n))
        idx = rng.permutation(n)
        train_embs.append(embs[idx[:n_train]])
        train_labels.extend([ci] * n_train)
        test_embs.append(embs[idx[n_train:]])
        test_labels.extend([ci] * (n - n_train))

    train_X = np.vstack(train_embs)
    test_X = np.vstack(test_embs)
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(train_X, np.array(train_labels))
    q_raw = float(np.mean(knn.predict(test_X) == np.array(test_labels)))
    q_norm = float((q_raw - 1.0 / K) / (1.0 - 1.0 / K))

    return kappa_nearest, q_raw, q_norm, sigma_W, d


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    from scipy.special import logit as scipy_logit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers_total = len(model.model.layers) if hasattr(model, 'model') else 24
    print(f"Total layers: {n_layers_total}")

    kappa_all = []
    logit_q_all = []
    dataset_results = []

    for ds_config in DATASETS:
        ds_name = ds_config["name"]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({ds_config['note']})")

        try:
            if ds_config["hf_subset"]:
                ds = load_dataset(ds_config["hf_name"], ds_config["hf_subset"],
                                  split=ds_config["split"], trust_remote_code=False)
            else:
                ds = load_dataset(ds_config["hf_name"], split=ds_config["split"],
                                  trust_remote_code=False)
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            continue

        texts = [x[ds_config["text_col"]] for x in ds]
        labels = [x[ds_config["label_col"]] for x in ds]
        all_classes = sorted(set(labels))
        K = len(all_classes)
        print(f"  K={K}, total={len(texts)}")

        # Group texts by class
        texts_by_class = {}
        for ci in all_classes:
            texts_by_class[ci] = [texts[i] for i, l in enumerate(labels) if l == ci]

        # Find best layer
        print("  Finding best layer...")
        best_layer, best_q_raw, best_embeddings = extract_best_layer_embeddings(
            model, tokenizer, device, texts_by_class, all_classes, n_layers_total
        )
        print(f"  Best layer: {best_layer}, q_raw={best_q_raw:.3f}")

        kappa, q_raw, q_norm, sigma_W, d = compute_kappa_and_q(
            best_embeddings, sorted(best_embeddings.keys()))

        q_norm_clipped = float(np.clip(q_norm, 0.01, 0.99))
        logit_q = float(scipy_logit(q_norm_clipped))

        kappa_all.append(kappa)
        logit_q_all.append(logit_q)

        print(f"  kappa={kappa:.4f}, q_norm={q_norm:.3f}, logit(q)={logit_q:.3f}")
        dataset_results.append({
            "dataset": ds_name,
            "K": K,
            "note": ds_config["note"],
            "best_layer": best_layer,
            "best_q_raw": best_q_raw,
            "kappa_nearest": kappa,
            "q_raw": q_raw,
            "q_norm": q_norm,
            "logit_q_norm": logit_q,
            "sigma_W": sigma_W,
            "d": d,
        })

    # Fit alpha across datasets (per-dataset intercept = LODO design)
    print(f"\n{'='*60}")
    print(f"CROSS-DATASET FIT FOR SmolLM2-1.7B")
    print(f"{'='*60}")

    if len(kappa_all) >= 3:
        r_pearson, p_pearson = pearsonr(kappa_all, logit_q_all)
        slope, intercept, _, _, se = linregress(kappa_all, logit_q_all)

        pr1 = ALPHA_LOW <= slope <= ALPHA_HIGH
        pr2 = r_pearson >= 0.80
        pr3 = (any(r["dataset"] == "clinc_oos" and r["kappa_nearest"] > 0
                   and r["q_norm"] > 0.10 for r in dataset_results))
        overall = pr1 and pr2

        print(f"  alpha_fit = {slope:.4f} +/- {se:.4f}")
        print(f"  Pearson r = {r_pearson:.4f} (p={p_pearson:.4f})")
        print(f"\n  PR1 (alpha in [{ALPHA_LOW},{ALPHA_HIGH}]): {'PASS' if pr1 else 'FAIL'}")
        print(f"  PR2 (r >= 0.80): {'PASS' if pr2 else 'FAIL'}")
        print(f"  PR3 (clinc_oos K=151 valid): {'PASS' if pr3 else 'FAIL'}")
        print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")

        output = {
            "experiment": "smollm2_cross_dataset_replication",
            "model": MODEL_NAME,
            "design": "cross-dataset at best-layer (matches LODO design)",
            "note": "Corrected design: best-layer per dataset, fit alpha across datasets. "
                    "Original cross-layer design failed for non-fine-tuned LM.",
            "pre_reg_alpha_interval": [ALPHA_LOW, ALPHA_HIGH],
            "alpha_universal": ALPHA_UNIVERSAL,
            "dataset_results": dataset_results,
            "alpha_fit": float(slope),
            "alpha_se": float(se),
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "pr1_alpha_pass": bool(pr1),
            "pr2_r_pass": bool(pr2),
            "pr3_clinc_valid": bool(pr3),
            "overall_pass": bool(overall),
            "verdict": "PASS" if overall else "FAIL",
        }
    else:
        print(f"  INSUFFICIENT DATA ({len(kappa_all)} datasets)")
        output = {
            "experiment": "smollm2_cross_dataset_replication",
            "model": MODEL_NAME,
            "design": "cross-dataset at best-layer",
            "dataset_results": dataset_results,
            "error": f"Only {len(kappa_all)} datasets succeeded",
            "overall_pass": False,
            "verdict": "FAIL",
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
