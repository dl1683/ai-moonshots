"""
SmolLM2-1.7B LOAO-equivalent external replication.

DESIGN: Exactly matches the original kappa_nearest_universal LOAO protocol:
  - Mean-pool embeddings (NOT last-token)
  - 4 proportional layers: 6, 12, 18, 23 (out of 24 total)
  - logit(q_raw) [NOT q_norm] -- same convention as kappa_near_cache files
  - 3 main datasets: dbpedia (K=14), agnews (K=4), 20newsgroups (K=20)
  - Generates kappa_near_cache_*.json files for SmolLM2

ANALYSIS:
  - Load all existing kappa cache files for 19 architectures
  - Add SmolLM2 data
  - Do LOAO: hold out SmolLM2, fit on remaining 19
  - Check if SmolLM2 LOAO alpha falls in [2.43, 3.29]

Pre-registered constants:
  alpha_low=2.43, alpha_high=3.29 (from LOAO with 12-arch pre-reg)

Output: results/kappa_near_cache_{dataset}_SmolLM2-1.7B.json (3 files)
        results/cti_smollm2_loao_replication.json
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, linregress
from scipy.special import logit as scipy_logit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_smollm2_loao_replication.json")

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_SHORT = "SmolLM2-1.7B"

ALPHA_LOW = 2.43
ALPHA_HIGH = 3.29

# Proportional layers for 24-layer SmolLM2: 25%, 50%, 75%, 100% depth
# 0.25*24=6, 0.50*24=12, 0.75*24=18, 1.0*24-1=23
LAYER_INDICES = [6, 12, 18, 23]

# 3 main datasets (same as LOAO training)
DATASETS = [
    {
        "name": "dbpedia",
        "hf_name": "dbpedia_14",
        "hf_subset": None,
        "split": "train",
        "text_col": "content",
        "label_col": "label",
        "K": 14,
        "n_per_class": 500,
    },
    {
        "name": "agnews",
        "hf_name": "ag_news",
        "hf_subset": None,
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "K": 4,
        "n_per_class": 500,
    },
    {
        "name": "20newsgroups",
        "hf_name": "SetFit/20_newsgroups",
        "hf_subset": None,
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "K": 20,
        "n_per_class": 500,
    },
]


def compute_kappa_nearest_mean_pool(embs, labels, K, subsample=500):
    """Compute kappa_nearest using mean-pool convention (same as LOAO script).
    kappa_nearest = mean over classes of (min_dist_to_other_class / (sigma_W * sqrt(d)))
    """
    unique_classes = sorted(set(labels))
    K_actual = len(unique_classes)
    d = embs.shape[1]

    # Subsample per class
    class_embs = {}
    for ci in unique_classes:
        idx = np.where(labels == ci)[0][:subsample]
        class_embs[ci] = embs[idx]

    # Class centroids and within-class variance
    centroids = {}
    within_var_sum = 0.0
    n_total = 0
    for ci in unique_classes:
        e = class_embs[ci]
        centroids[ci] = e.mean(0)
        within_var_sum += np.sum((e - centroids[ci]) ** 2)
        n_total += len(e)
    sigma_W = float(np.sqrt(within_var_sum / (n_total * d)))

    # Per-class kappa (distance to nearest other class centroid)
    kappa_list = []
    for ci in unique_classes:
        min_dist = float("inf")
        for cj in unique_classes:
            if ci == cj:
                continue
            dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            min_dist = min(min_dist, dist)
        kappa_list.append(min_dist / (sigma_W * np.sqrt(d)))

    kappa_nearest = float(np.mean(kappa_list))  # mean over classes
    kappa_min = float(np.min(kappa_list))        # minimum over classes

    return kappa_nearest, kappa_min, sigma_W


def compute_1nn_accuracy(embs, labels, subsample=500):
    """Compute 1-NN accuracy (80/20 split per class)."""
    from sklearn.neighbors import KNeighborsClassifier
    unique_classes = sorted(set(labels))

    rng = np.random.default_rng(42)
    train_embs, train_labels, test_embs, test_labels = [], [], [], []
    for ci in unique_classes:
        idx = np.where(labels == ci)[0][:subsample]
        n = len(idx)
        n_train = max(1, int(0.8 * n))
        perm = rng.permutation(n)
        train_embs.append(embs[idx[perm[:n_train]]])
        train_labels.extend([ci] * n_train)
        test_embs.append(embs[idx[perm[n_train:]]])
        test_labels.extend([ci] * (n - n_train))

    train_X = np.vstack(train_embs)
    test_X = np.vstack(test_embs)
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(train_X, np.array(train_labels))
    q_raw = float(np.mean(knn.predict(test_X) == np.array(test_labels)))
    return q_raw


def extract_mean_pool_embeddings_at_layer(model, tokenizer, device, texts, layer_idx):
    """Extract mean-pooled embeddings at a given layer (hidden_states[layer_idx+1])."""
    import torch
    all_embs = []
    for b_start in range(0, len(texts), 32):
        batch = texts[b_start:b_start + 32]
        tok = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**tok, output_hidden_states=True)
        # Mean pool over non-padding tokens
        hidden = out.hidden_states[layer_idx + 1]  # (B, seq, d)
        mask = tok["attention_mask"].unsqueeze(-1).float()  # (B, seq, 1)
        mean_emb = (hidden * mask).sum(1) / mask.sum(1)  # (B, d)
        all_embs.append(mean_emb.cpu().float().numpy())
    return np.vstack(all_embs)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYER_INDICES} (proportional depth 25/50/75/100%)")
    print(f"Convention: mean-pool, logit(q_raw) -- matches LOAO cache format\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_smollm2_points = []

    for ds_config in DATASETS:
        ds_name = ds_config["name"]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (K={ds_config['K']})")

        # Load dataset
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

        texts_raw = [x[ds_config["text_col"]] for x in ds]
        labels_raw = [x[ds_config["label_col"]] for x in ds]
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(labels_raw)
        all_classes = list(range(len(le.classes_)))
        K = len(all_classes)
        n_per_class = ds_config["n_per_class"]
        print(f"  K={K}, total={len(texts_raw)}, n_per_class={n_per_class}")

        # Subsample n_per_class per class
        texts_sub = []
        labels_sub = []
        for ci in all_classes:
            idx_ci = [i for i, l in enumerate(labels) if l == ci][:n_per_class]
            texts_sub.extend([texts_raw[i] for i in idx_ci])
            labels_sub.extend([ci] * len(idx_ci))
        labels_arr = np.array(labels_sub)

        # Build cache
        cache_points = []
        logKm1 = float(np.log(K - 1))

        for layer_idx in LAYER_INDICES:
            print(f"  Layer {layer_idx}...", flush=True)
            embs = extract_mean_pool_embeddings_at_layer(
                model, tokenizer, device, texts_sub, layer_idx)

            kappa_near, kappa_min, sigma_W = compute_kappa_nearest_mean_pool(
                embs, labels_arr, K, subsample=n_per_class)
            q_raw = compute_1nn_accuracy(embs, labels_arr, subsample=n_per_class)
            logit_q = float(scipy_logit(np.clip(q_raw, 0.001, 0.999)))

            pt = {
                "model": MODEL_SHORT,
                "dataset": ds_name,
                "layer": layer_idx,
                "K": K,
                "q": q_raw,
                "kappa_nearest": kappa_near,
                "kappa_min": kappa_min,
                "logit_q": logit_q,
                "logKm1": logKm1,
            }
            cache_points.append(pt)
            all_smollm2_points.append(pt)
            print(f"    kappa={kappa_near:.4f}, q={q_raw:.3f}, logit={logit_q:.3f}")

        # Save cache file (same format as other kappa_near_cache files)
        cache_path = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds_name}_{MODEL_SHORT}.json")
        with open(cache_path, "w") as f:
            json.dump(cache_points, f, indent=2)
        print(f"  Saved cache: {cache_path}")

    # =========================================================
    # LOAO ANALYSIS: Add SmolLM2 to existing architecture pool
    # =========================================================
    print(f"\n{'='*60}")
    print("LOAO ANALYSIS: SmolLM2 as new architecture")
    print(f"{'='*60}")

    # Load all existing kappa cache files for the 3 datasets
    all_points = []
    for ds_config in DATASETS:
        ds_name = ds_config["name"]
        import glob
        pattern = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds_name}_*.json")
        for cache_file in glob.glob(pattern):
            model_short = os.path.basename(cache_file).replace(
                f"kappa_near_cache_{ds_name}_", "").replace(".json", "")
            with open(cache_file) as f:
                pts = json.load(f)
            all_points.extend(pts)

    print(f"  Total points from all architectures: {len(all_points)}")
    models_in_data = sorted(set(p["model"] for p in all_points))
    print(f"  Architectures: {len(models_in_data)}")

    # LOAO: leave SmolLM2 out, fit on others, predict SmolLM2
    train_pts = [p for p in all_points if p["model"] != MODEL_SHORT]
    test_pts = [p for p in all_points if p["model"] == MODEL_SHORT]
    print(f"\n  Train: {len(train_pts)} points ({len(set(p['model'] for p in train_pts))} archs)")
    print(f"  Test (SmolLM2): {len(test_pts)} points")

    if len(train_pts) < 5 or len(test_pts) < 2:
        print("  INSUFFICIENT DATA")
        return

    # Fit on train
    kappa_tr = np.array([p["kappa_nearest"] for p in train_pts])
    logKm1_tr = np.array([p["logKm1"] for p in train_pts])
    logit_tr = np.array([p["logit_q"] for p in train_pts])
    X_tr = np.column_stack([kappa_tr, logKm1_tr, np.ones(len(train_pts))])
    coeffs, _, _, _ = np.linalg.lstsq(X_tr, logit_tr, rcond=None)
    alpha_loao, beta_loao, C0_loao = coeffs
    print(f"\n  Train fit: alpha={alpha_loao:.4f}, beta={beta_loao:.4f}, C={C0_loao:.4f}")

    # Evaluate on SmolLM2
    kappa_te = np.array([p["kappa_nearest"] for p in test_pts])
    logKm1_te = np.array([p["logKm1"] for p in test_pts])
    logit_te = np.array([p["logit_q"] for p in test_pts])
    X_te = np.column_stack([kappa_te, logKm1_te, np.ones(len(test_pts))])
    logit_pred = X_te @ coeffs

    # Fit SmolLM2 alpha (holding beta and C fixed)
    # Actually: fit alpha for SmolLM2 independently
    kappa_smol = kappa_te
    logit_smol = logit_te
    logKm1_smol = logKm1_te
    X_smol = np.column_stack([kappa_smol, logKm1_smol, np.ones(len(test_pts))])
    coeffs_smol, _, _, _ = np.linalg.lstsq(X_smol, logit_smol, rcond=None)
    alpha_smol, beta_smol, C_smol = coeffs_smol
    print(f"\n  SmolLM2 fit: alpha={alpha_smol:.4f}, beta={beta_smol:.4f}, C={C_smol:.4f}")

    # Prediction error
    logit_pred_frozen = X_te @ coeffs  # using train-fit coefficients
    mae_frozen = float(np.mean(np.abs(logit_smol - logit_pred_frozen)))
    r_pred, p_pred = pearsonr(logit_smol, logit_pred_frozen)

    pr1 = ALPHA_LOW <= alpha_smol <= ALPHA_HIGH
    pr2 = r_pred >= 0.80

    print(f"\n  PR1 (SmolLM2 alpha in [{ALPHA_LOW},{ALPHA_HIGH}]): {'PASS' if pr1 else 'FAIL'}")
    print(f"       alpha_smol={alpha_smol:.4f}")
    print(f"  PR2 (Pearson r(pred, obs) >= 0.80): {'PASS' if pr2 else 'FAIL'}")
    print(f"       r={r_pred:.4f}, MAE_frozen={mae_frozen:.4f}")
    print(f"\n  OVERALL: {'PASS' if (pr1 and pr2) else 'FAIL'}")

    output = {
        "experiment": "smollm2_loao_replication",
        "model": MODEL_NAME,
        "design": "LOAO-equivalent: mean-pool, logit(q_raw), 4 proportional layers (6,12,18,23)",
        "convention": "logit(q_raw) -- matches original kappa_near_cache format",
        "pre_reg_alpha_interval": [ALPHA_LOW, ALPHA_HIGH],
        "datasets_used": [d["name"] for d in DATASETS],
        "layers_used": LAYER_INDICES,
        "n_train_points": len(train_pts),
        "n_train_architectures": len(set(p["model"] for p in train_pts)),
        "n_smollm2_points": len(test_pts),
        "smollm2_points": all_smollm2_points,
        "train_fit": {
            "alpha": float(alpha_loao),
            "beta": float(beta_loao),
            "C0": float(C0_loao),
        },
        "smollm2_fit": {
            "alpha": float(alpha_smol),
            "beta": float(beta_smol),
            "C": float(C_smol),
        },
        "evaluation": {
            "mae_frozen": mae_frozen,
            "pearson_r_pred": float(r_pred),
            "pearson_p_pred": float(p_pred),
        },
        "pr1_alpha_pass": bool(pr1),
        "pr2_r_pass": bool(pr2),
        "overall_pass": bool(pr1 and pr2),
        "verdict": "PASS" if (pr1 and pr2) else "FAIL",
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
