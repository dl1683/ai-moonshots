"""
Resume SmolLM2 LOAO replication - 20newsgroups only.
dbpedia and agnews caches already created. Just run 20newsgroups, then do LOAO analysis.
"""
import json
import os
import sys
import glob
import numpy as np
from scipy.stats import pearsonr
from scipy.special import logit as scipy_logit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_smollm2_loao_replication.json")

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_SHORT = "SmolLM2-1.7B"
ALPHA_LOW = 2.43
ALPHA_HIGH = 3.29
LAYER_INDICES = [6, 12, 18, 23]

DATASETS_ALL = ["dbpedia", "agnews", "20newsgroups"]


def compute_kappa_nearest_mean_pool(embs, labels, K, subsample=500):
    unique_classes = sorted(set(labels))
    d = embs.shape[1]
    class_embs = {}
    for ci in unique_classes:
        idx = np.where(labels == ci)[0][:subsample]
        class_embs[ci] = embs[idx]
    centroids = {}
    within_var_sum = 0.0
    n_total = 0
    for ci in unique_classes:
        e = class_embs[ci]
        centroids[ci] = e.mean(0)
        within_var_sum += np.sum((e - centroids[ci]) ** 2)
        n_total += len(e)
    sigma_W = float(np.sqrt(within_var_sum / (n_total * d)))
    kappa_list = []
    for ci in unique_classes:
        min_dist = float("inf")
        for cj in unique_classes:
            if ci == cj:
                continue
            dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            min_dist = min(min_dist, dist)
        kappa_list.append(min_dist / (sigma_W * np.sqrt(d)))
    return float(np.mean(kappa_list)), float(np.min(kappa_list)), sigma_W


def compute_1nn_accuracy(embs, labels, subsample=500):
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
    # Final NaN check
    if np.isnan(train_X).any() or np.isnan(test_X).any():
        raise ValueError("NaN in embeddings passed to KNN")
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(train_X, np.array(train_labels))
    q_raw = float(np.mean(knn.predict(test_X) == np.array(test_labels)))
    return q_raw


def extract_mean_pool_embeddings_at_layer(model, tokenizer, device, texts, layer_idx):
    import torch
    all_embs = []
    for b_start in range(0, len(texts), 32):
        batch = texts[b_start:b_start + 32]
        tok = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**tok, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx + 1].float()
        mask = tok["attention_mask"].unsqueeze(-1).float()
        mask_sum = mask.sum(1).clamp(min=1e-9)
        mean_emb = (hidden * mask).sum(1) / mask_sum
        emb_np = mean_emb.cpu().numpy()
        valid_mask = ~np.isnan(emb_np).any(axis=1)
        if not valid_mask.all():
            print(f"    Warning: {(~valid_mask).sum()} NaN rows filtered from batch")
        all_embs.append(emb_np[valid_mask])
    return np.vstack(all_embs)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    from sklearn.preprocessing import LabelEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Running 20newsgroups only (dbpedia+agnews already cached)\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run 20newsgroups
    ds_name = "20newsgroups"
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name} (K=20)")

    ds = load_dataset("SetFit/20_newsgroups", split="train", trust_remote_code=False)
    texts_raw = [x["text"] for x in ds]
    labels_raw = [x["label"] for x in ds]
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    all_classes = list(range(len(le.classes_)))
    K = len(all_classes)
    n_per_class = 500
    print(f"  K={K}, total={len(texts_raw)}, n_per_class={n_per_class}")

    texts_sub = []
    labels_sub = []
    for ci in all_classes:
        idx_ci = [i for i, l in enumerate(labels) if l == ci][:n_per_class]
        texts_sub.extend([texts_raw[i] for i in idx_ci])
        labels_sub.extend([ci] * len(idx_ci))
    labels_arr = np.array(labels_sub)

    cache_points = []
    logKm1 = float(np.log(K - 1))

    for layer_idx in LAYER_INDICES:
        print(f"  Layer {layer_idx}...", flush=True)
        embs = extract_mean_pool_embeddings_at_layer(
            model, tokenizer, device, texts_sub, layer_idx)
        print(f"    Got {len(embs)} valid embeddings")

        kappa_near, kappa_min, sigma_W = compute_kappa_nearest_mean_pool(
            embs, labels_arr[:len(embs)], K, subsample=n_per_class)
        q_raw = compute_1nn_accuracy(embs, labels_arr[:len(embs)], subsample=n_per_class)
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
        print(f"    kappa={kappa_near:.4f}, q={q_raw:.3f}, logit={logit_q:.3f}")

    cache_path = os.path.join(RESULTS_DIR, f"kappa_near_cache_20newsgroups_SmolLM2-1.7B.json")
    with open(cache_path, "w") as f:
        json.dump(cache_points, f, indent=2)
    print(f"  Saved cache: {cache_path}")

    # =========================================================
    # LOAO ANALYSIS
    # =========================================================
    print(f"\n{'='*60}")
    print("LOAO ANALYSIS: SmolLM2 as new architecture")
    print(f"{'='*60}")

    all_points = []
    for ds_name_loop in DATASETS_ALL:
        pattern = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds_name_loop}_*.json")
        for cache_file in glob.glob(pattern):
            model_short = os.path.basename(cache_file).replace(
                f"kappa_near_cache_{ds_name_loop}_", "").replace(".json", "")
            with open(cache_file) as f:
                pts = json.load(f)
            # Some older cache files (e.g. bge) store K but not logKm1 -- compute it
            for pt in pts:
                if "logKm1" not in pt and "K" in pt:
                    pt["logKm1"] = float(np.log(pt["K"] - 1))
            all_points.extend(pts)

    print(f"  Total points: {len(all_points)}")
    models_in_data = sorted(set(p["model"] for p in all_points))
    print(f"  Architectures: {len(models_in_data)}")

    train_pts = [p for p in all_points if p["model"] != MODEL_SHORT]
    test_pts = [p for p in all_points if p["model"] == MODEL_SHORT]
    print(f"\n  Train: {len(train_pts)} pts ({len(set(p['model'] for p in train_pts))} archs)")
    print(f"  Test (SmolLM2): {len(test_pts)} pts")

    # Fit on train
    kappa_tr = np.array([p["kappa_nearest"] for p in train_pts])
    logKm1_tr = np.array([p["logKm1"] for p in train_pts])
    logit_tr = np.array([p["logit_q"] for p in train_pts])
    X_tr = np.column_stack([kappa_tr, logKm1_tr, np.ones(len(train_pts))])
    coeffs, _, _, _ = np.linalg.lstsq(X_tr, logit_tr, rcond=None)
    alpha_loao, beta_loao, C0_loao = coeffs
    print(f"\n  Train fit: alpha={alpha_loao:.4f}, beta={beta_loao:.4f}, C={C0_loao:.4f}")

    # Fit SmolLM2 independently
    kappa_te = np.array([p["kappa_nearest"] for p in test_pts])
    logKm1_te = np.array([p["logKm1"] for p in test_pts])
    logit_te = np.array([p["logit_q"] for p in test_pts])
    X_te = np.column_stack([kappa_te, logKm1_te, np.ones(len(test_pts))])
    coeffs_smol, _, _, _ = np.linalg.lstsq(X_te, logit_te, rcond=None)
    alpha_smol, beta_smol, C_smol = coeffs_smol
    print(f"\n  SmolLM2 fit: alpha={alpha_smol:.4f}, beta={beta_smol:.4f}, C={C_smol:.4f}")

    # Prediction r using train-fit coefficients
    logit_pred_frozen = X_te @ coeffs
    r_pred, p_pred = pearsonr(logit_te, logit_pred_frozen)
    mae_frozen = float(np.mean(np.abs(logit_te - logit_pred_frozen)))

    pr1 = ALPHA_LOW <= alpha_smol <= ALPHA_HIGH
    pr2 = r_pred >= 0.80

    print(f"\n  PR1 (SmolLM2 alpha in [{ALPHA_LOW},{ALPHA_HIGH}]): {'PASS' if pr1 else 'FAIL'}")
    print(f"       alpha_smol={alpha_smol:.4f}")
    print(f"  PR2 (r(pred,obs) >= 0.80): {'PASS' if pr2 else 'FAIL'}")
    print(f"       r={r_pred:.4f}, MAE={mae_frozen:.4f}")
    print(f"\n  OVERALL: {'PASS' if (pr1 and pr2) else 'FAIL'}")

    # Print all SmolLM2 points for verification
    print(f"\n  SmolLM2 data points:")
    for p in test_pts:
        print(f"    {p['dataset']} L{p['layer']}: kappa={p['kappa_nearest']:.4f}, "
              f"q={p['q']:.3f}, logit={p['logit_q']:.3f}")

    output = {
        "experiment": "smollm2_loao_replication",
        "model": MODEL_NAME,
        "design": "LOAO-equivalent: mean-pool, logit(q_raw), 4 proportional layers (6,12,18,23)",
        "convention": "logit(q_raw) -- matches original kappa_near_cache format",
        "pre_reg_alpha_interval": [ALPHA_LOW, ALPHA_HIGH],
        "datasets_used": DATASETS_ALL,
        "layers_used": LAYER_INDICES,
        "n_train_points": len(train_pts),
        "n_train_architectures": len(set(p["model"] for p in train_pts)),
        "n_smollm2_points": len(test_pts),
        "smollm2_points": test_pts,
        "train_fit": {"alpha": float(alpha_loao), "beta": float(beta_loao), "C0": float(C0_loao)},
        "smollm2_fit": {"alpha": float(alpha_smol), "beta": float(beta_smol), "C": float(C_smol)},
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
