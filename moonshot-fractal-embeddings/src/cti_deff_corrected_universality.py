#!/usr/bin/env python -u
"""
d_eff-CORRECTED UNIVERSALITY TEST (Feb 21 2026)
================================================
Motivation from held-out universality failure (0/3):
- Fixed A from NLP doesn't transfer to DBpedia (R2=-2.214)
- BUT rho=0.87 (shape is universal, SCALE is not)
- Theory: A = C_corr * sqrt(d_eff * log(n_per))
- d_eff = tr(Sigma_W)^2 / tr(Sigma_W^2) (effective within-class dimension)

Hypothesis: C_corr IS universal, A is only APPARENTLY non-universal because
d_eff varies across model sizes and task types.

Test design:
- Fit C_corr on training (Pythia-160m, 410m, CLINC+AGNews)
- For each held-out point: compute d_eff, predict A = C_corr * sqrt(d_eff * log(n_per))
- Check if R2 and MAE improve vs fixed-A baseline

Pre-registered criterion: R2 > 0.80 on all 3 splits (same as before but now d_eff-corrected)
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.special import expit, logit
from scipy.stats import spearmanr
import torch
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same splits as cti_held_out_universality.py
TRAINING_MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
]
TRAINING_DATASETS = ["clinc_oos", "agnews"]

HELD_OUT_A_MODELS = ["EleutherAI/pythia-1b"]
HELD_OUT_A_DATASETS = ["clinc_oos", "agnews"]

HELD_OUT_B_MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
]
HELD_OUT_B_DATASETS = ["dbpedia_14"]

HELD_OUT_C_MODELS = ["EleutherAI/pythia-1b"]
HELD_OUT_C_DATASETS = ["dbpedia_14"]

MAE_THRESHOLD = 0.10
R2_THRESHOLD = 0.80
N_LAYERS_SAMPLE = 8
N_SAMPLES = 1000


def load_model_and_tokenizer(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16
    ).to(DEVICE)
    model.eval()
    return model, tok


def load_dataset_texts_labels(dataset_name, n_samples=1000):
    from datasets import load_dataset
    if dataset_name == "clinc_oos":
        ds = load_dataset("clinc_oos", "plus", split="train")
        texts = list(ds["text"])[:n_samples]
        labels = np.array(ds["intent"])[:n_samples]
    elif dataset_name == "agnews":
        ds = load_dataset("ag_news", split="train")
        texts = list(ds["text"])[:n_samples]
        labels = np.array(ds["label"])[:n_samples]
    elif dataset_name == "dbpedia_14":
        ds = load_dataset("dbpedia_14", split="train")
        import random
        random.seed(42)
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[:n_samples]
        texts = [ds["content"][i] for i in indices]
        labels = np.array([ds["label"][i] for i in indices])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return texts, labels


def get_embeddings_at_layer(model, tokenizer, texts, layer_idx, batch_size=32):
    all_hidden = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx + 1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        all_hidden.append(pooled.cpu().float().numpy())
    return np.concatenate(all_hidden, axis=0)


def compute_metrics(embeddings, labels):
    """Compute dist_ratio, q, AND d_eff from embeddings."""
    X = np.nan_to_num(embeddings, nan=0.0)
    y = labels

    K = len(np.unique(y))
    N = len(X)
    d = X.shape[1]

    if N < K * 5 or K < 2:
        return None

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import pairwise_distances
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = max(min((acc - 1.0/K) / (1.0 - 1.0/K), 0.999), 0.001)

    # dist_ratio
    n_sub = min(N, 500)
    idx_sub = np.random.choice(N, n_sub, replace=False)
    X_sub = X[idx_sub]; y_sub = y[idx_sub]
    D = pairwise_distances(X_sub, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    intra_mins, inter_mins = [], []
    for i in range(n_sub):
        same = (y_sub == y_sub[i]); same[i] = False
        diff = ~same; diff[i] = False
        if same.any(): intra_mins.append(D[i][same].min())
        if diff.any(): inter_mins.append(D[i][diff].min())
    if not intra_mins or not inter_mins:
        return None
    dist_ratio = float(np.mean(inter_mins)) / (float(np.mean(intra_mins)) + 1e-10)

    # d_eff from within-class covariance
    # Sigma_W = within-class scatter / N
    # d_eff = tr(Sigma_W)^2 / tr(Sigma_W^2)
    within_class_covs = []
    class_counts = []
    for k in np.unique(y):
        Xk = X[y == k]
        nk = len(Xk)
        if nk < 2:
            continue
        mu_k = Xk.mean(0)
        Xk_c = Xk - mu_k
        # Sigma_k = (1/nk) * Xk_c^T Xk_c
        # tr(Sigma_k) = sum of variances
        # tr(Sigma_k^2) = ||Xk_c/sqrt(nk)||_F^4 / d (not exactly right)
        # Use: tr(Sigma_k) = mean(||x - mu_k||^2) (trace = sum of eigenvalues)
        # tr(Sigma_k^2) = ||Sigma_k||_F^2 = mean_i,j [(x_i - mu_k) dot (x_j - mu_k)]^2 / nk^2
        # For efficiency: use random projection or direct formula
        tr_k = float(np.mean(np.sum(Xk_c**2, axis=1)))  # tr(Sigma_k) = E[||x-mu||^2]
        # tr(Sigma_k^2): use ||Xk_c||_F^2 for trace but need trace of square
        # tr(Sigma^2) = (1/n^2) * ||Xk_c^T Xk_c||_F^2
        # For large d, approximate: tr(Sigma^2) ~ tr(Sigma)^2 / d_eff (definition)
        # Direct: tr(Sigma^2) = sum_{i,j} cov(x_i,x_j)^2
        # Most efficient: Gram matrix approach for n << d
        # Use the identity: tr(Sigma^2) = (1/n^2) * ||G||_F^2 where G is Gram matrix
        G = Xk_c @ Xk_c.T / nk  # (nk, nk) Gram matrix
        tr_sq_k = float(np.sum(G**2)) / nk  # tr(Sigma^2) = ||G||_F^2 / n
        within_class_covs.append((tr_k, tr_sq_k, nk))
        class_counts.append(nk)

    if not within_class_covs:
        return None

    # Pooled Sigma_W
    total_n = sum(c[2] for c in within_class_covs)
    tr_W = sum(c[0] * c[2] for c in within_class_covs) / total_n
    tr_W_sq = sum(c[1] * c[2] for c in within_class_covs) / total_n

    d_eff = float(tr_W**2 / (tr_W_sq + 1e-10))

    # n_per = mean samples per class
    n_per = float(total_n / K)

    return {
        "knn_acc": acc,
        "q": q,
        "dist_ratio": dist_ratio,
        "d_eff": d_eff,
        "n_per": n_per,
        "K": K,
        "N": N,
        "d": d,
        "A_feature": float(np.sqrt(max(d_eff * np.log(max(n_per, 2)), 1e-10))),
    }


def collect_data_for_split(model_ids, dataset_names, max_layers=N_LAYERS_SAMPLE):
    import gc
    data_points = []
    for model_id in model_ids:
        try:
            model, tokenizer = load_model_and_tokenizer(model_id)
            n_layers = model.config.num_hidden_layers
            layer_indices = list(np.linspace(0, n_layers-1, max_layers, dtype=int))
            for dataset_name in dataset_names:
                print(f"  {model_id.split('/')[-1]} + {dataset_name}...", end="", flush=True)
                texts, labels = load_dataset_texts_labels(dataset_name, N_SAMPLES)
                count = 0
                for layer_idx in layer_indices:
                    emb = get_embeddings_at_layer(model, tokenizer, texts, layer_idx)
                    result = compute_metrics(emb, labels)
                    if result is not None:
                        result["model"] = model_id
                        result["dataset"] = dataset_name
                        result["layer"] = int(layer_idx)
                        data_points.append(result)
                        count += 1
                print(f" {count} pts", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR: {e}]", flush=True)
    return data_points


def fit_deff_corrected(data_points, verbose=True):
    """
    Fit logit(q) = C_corr * sqrt(d_eff * log(n_per)) * (DR-1) + C
    Returns C_corr, C, R2, MAE.
    """
    drs = np.array([p["dist_ratio"] for p in data_points])
    qs = np.array([p["q"] for p in data_points])
    a_feats = np.array([p["A_feature"] for p in data_points])  # sqrt(d_eff * log(n_per))
    logit_qs = logit(qs)

    # Design matrix: [a_feat * (DR-1), 1]
    X_design = np.column_stack([a_feats * (drs - 1), np.ones(len(drs))])
    theta = np.linalg.lstsq(X_design, logit_qs, rcond=None)[0]
    C_corr, C = float(theta[0]), float(theta[1])

    pred = C_corr * a_feats * (drs - 1) + C
    r2 = 1 - np.sum((logit_qs - pred)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)
    q_pred = expit(pred)
    mae_q = float(np.mean(np.abs(qs - q_pred)))

    if verbose:
        print(f"    d_eff-corrected fit: C_corr={C_corr:.4f}, C={C:.3f}")
        print(f"    R2={r2:.4f}, MAE_q={mae_q:.4f}", flush=True)

    return C_corr, C, float(r2), mae_q


def predict_held_out_deff(data_points, C_corr, C, split_name, verbose=True):
    """Predict held-out using d_eff-corrected formula with frozen C_corr."""
    drs = np.array([p["dist_ratio"] for p in data_points])
    qs = np.array([p["q"] for p in data_points])
    a_feats = np.array([p["A_feature"] for p in data_points])
    logit_qs = logit(qs)

    # Prediction with frozen C_corr
    pred = C_corr * a_feats * (drs - 1) + C
    r2 = 1 - np.sum((logit_qs - pred)**2) / max(np.sum((logit_qs - logit_qs.mean())**2), 1e-10)
    q_pred = expit(pred)
    mae_q = float(np.mean(np.abs(qs - q_pred)))
    rho = float(spearmanr(qs, q_pred).correlation)

    passed = (mae_q < MAE_THRESHOLD) and (r2 > R2_THRESHOLD)
    n = len(data_points)

    if verbose:
        print(f"    Held-out: n={n}, MAE_q={mae_q:.4f}, R2={r2:.4f}, rho={rho:.4f}")
        print(f"    MAE threshold: {MAE_THRESHOLD}, R2 threshold: {R2_THRESHOLD}")
        print(f"    PASS: {passed}", flush=True)

    return {
        "n_points": n,
        "mae_q": mae_q,
        "r2": float(r2),
        "rho": rho,
        "passed": bool(passed),
        "mean_d_eff": float(np.mean([p["d_eff"] for p in data_points])),
        "mean_A_implied": float(C_corr * np.mean(a_feats)),
    }


def main():
    print("=" * 70)
    print("d_eff-CORRECTED UNIVERSALITY TEST")
    print("logit(q) = C_corr * sqrt(d_eff * log(n)) * (DR-1) + C")
    print(f"Pre-registered: MAE < {MAE_THRESHOLD}, R2 > {R2_THRESHOLD}")
    print("=" * 70, flush=True)

    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Collecting training data")
    print(f"  Models: {[m.split('/')[-1] for m in TRAINING_MODELS]}")
    print(f"  Datasets: {TRAINING_DATASETS}")
    print("=" * 70, flush=True)

    train_pts = collect_data_for_split(TRAINING_MODELS, TRAINING_DATASETS)
    print(f"\n  Training data: {len(train_pts)} valid points", flush=True)

    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Fitting d_eff-corrected law on training data")
    print("=" * 70, flush=True)

    C_corr, C_const, train_r2, train_mae = fit_deff_corrected(train_pts)
    print(f"  C_corr={C_corr:.4f}, C_const={C_const:.3f}")
    print(f"  Training R2={train_r2:.4f}, MAE={train_mae:.4f}", flush=True)

    # Also fit the plain law for comparison
    drs_tr = np.array([p["dist_ratio"] for p in train_pts])
    qs_tr = np.array([p["q"] for p in train_pts])
    logit_qs_tr = logit(qs_tr)
    X_plain = np.column_stack([drs_tr - 1, np.ones(len(drs_tr))])
    theta_plain = np.linalg.lstsq(X_plain, logit_qs_tr, rcond=None)[0]
    A_plain, C_plain = float(theta_plain[0]), float(theta_plain[1])
    pred_plain = A_plain * (drs_tr - 1) + C_plain
    r2_plain = 1 - np.sum((logit_qs_tr - pred_plain)**2) / max(np.sum((logit_qs_tr - logit_qs_tr.mean())**2), 1e-10)
    print(f"\n  Plain law (for comparison): A={A_plain:.4f}, C={C_plain:.3f}, R2={r2_plain:.4f}", flush=True)

    # ================================================================
    results = {
        "pre_registered_criteria": {"mae_threshold": MAE_THRESHOLD, "r2_threshold": R2_THRESHOLD},
        "training": {
            "models": TRAINING_MODELS,
            "datasets": TRAINING_DATASETS,
            "n_points": len(train_pts),
            "C_corr": C_corr,
            "C_const": C_const,
            "r2": train_r2,
            "mae": train_mae,
            "plain_A": A_plain,
            "plain_C": C_plain,
            "plain_r2": float(r2_plain),
            "mean_d_eff": float(np.mean([p["d_eff"] for p in train_pts])),
        },
        "held_out_splits": {},
        "scorecard": {},
    }

    total_pass = 0
    total_splits = 0

    for split_name, split_models, split_datasets in [
        ("A_new_model_size", HELD_OUT_A_MODELS, HELD_OUT_A_DATASETS),
        ("B_new_dataset", HELD_OUT_B_MODELS, HELD_OUT_B_DATASETS),
        ("C_both_new", HELD_OUT_C_MODELS, HELD_OUT_C_DATASETS),
    ]:
        print(f"\n{'='*70}")
        print(f"STEP 3{split_name[0]}: Held-out split {split_name}")
        print(f"  Models: {[m.split('/')[-1] for m in split_models]}")
        print(f"  Datasets: {split_datasets}")
        print("=" * 70, flush=True)

        pts = collect_data_for_split(split_models, split_datasets)
        if not pts:
            print("  No valid points!", flush=True)
            results["held_out_splits"][split_name] = {"n_points": 0, "passed": False}
            total_splits += 1
            continue

        print(f"  Held-out data: {len(pts)} valid points", flush=True)
        split_result = predict_held_out_deff(pts, C_corr, C_const, split_name)
        split_result["models"] = [m.split('/')[-1] for m in split_models]
        split_result["datasets"] = split_datasets
        results["held_out_splits"][split_name] = split_result

        verdict = "PASS" if split_result["passed"] else "FAIL"
        print(f"  -> {verdict}", flush=True)

        if split_result["passed"]:
            total_pass += 1
        total_splits += 1

    # ================================================================
    print(f"\n{'='*70}")
    print("FINAL SCORECARD")
    print("=" * 70, flush=True)

    for sn, sr in results["held_out_splits"].items():
        verdict = "PASS" if sr.get("passed", False) else "FAIL"
        print(f"  {sn}: [{verdict}] MAE={sr.get('mae_q', 'N/A'):.4f if isinstance(sr.get('mae_q'), float) else 'N/A'}, R2={sr.get('r2', 'N/A'):.4f if isinstance(sr.get('r2'), float) else 'N/A'}, mean_d_eff={sr.get('mean_d_eff', 'N/A'):.1f if isinstance(sr.get('mean_d_eff'), float) else 'N/A'}")

    print(f"\n  TOTAL: {total_pass}/{total_splits}")
    results["scorecard"] = {"passed": total_pass, "total": total_splits, "overall_pass": total_pass == total_splits}

    out_path = RESULTS_DIR / "cti_deff_corrected_universality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
