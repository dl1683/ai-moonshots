#!/usr/bin/env python -u
"""
CTI Prospective K-based Alpha Prediction Test
==============================================
Pre-registered prospective test of A_single(K) = C - a * log(K)
on NEW NLP datasets with K NOT in calibration set {4, 14, 20, 28}.

Formula locked from NLP calibration (cti_K_alpha_analysis.json):
  C = 26.09915367667568
  a = 7.7798097585482395
  A_pred(K) = C - a * log(K)

Test datasets (all K outside calibration range):
  - glue/sst2       K=2  A_pred = 20.71
  - tweet_eval/sent K=3  A_pred = 17.56
  - dair-ai/emotion K=6  A_pred = 12.17

Architectures: Pythia-160m, Pythia-410m, Qwen3-0.6B, Falcon-H1-0.5B
Layers: 4 proportional-depth points (25/50/75/100%)

Usage: python -u src/cti_prospective_K_test.py
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.special import logit as sp_logit
from scipy.stats import pearsonr

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================================================
# PRE-REGISTERED FORMULA (LOCKED - do NOT modify)
# ===================================================
C_LOCKED = 26.09915367667568
A_LOCKED = 7.7798097585482395

def A_pred(K):
    return C_LOCKED - A_LOCKED * np.log(float(K))

PREREG_PREDICTIONS = {
    "sst2":               {"K": 2, "A_pred": A_pred(2)},
    "tweet_sentiment":    {"K": 3, "A_pred": A_pred(3)},
    "emotion":            {"K": 6, "A_pred": A_pred(6)},
}

# ===================================================
# TEST DATASETS
# ===================================================
DATASETS = {
    "sst2": {
        "hf_name": "glue", "hf_cfg": "sst2",
        "split": "validation", "text_col": "sentence", "label_col": "label",
        "K": 2, "n_sample": 1000,
    },
    "tweet_sentiment": {
        "hf_name": "tweet_eval", "hf_cfg": "sentiment",
        "split": "test", "text_col": "text", "label_col": "label",
        "K": 3, "n_sample": 1000,
    },
    "emotion": {
        "hf_name": "dair-ai/emotion", "hf_cfg": None,
        "split": "test", "text_col": "text", "label_col": "label",
        "K": 6, "n_sample": 1000,
    },
}

# ===================================================
# TEST ARCHITECTURES
# ===================================================
MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    # falcon-h1 excluded: naive SSM path too slow on Windows
]

TRUST_REMOTE_CODE_MODELS = {"falcon-h1-0.5b-base"}


# ===================================================
# EMBEDDING EXTRACTION
# ===================================================
def get_embeddings_at_layers(model_name, texts, batch_size=64):
    """Extract embeddings at 4 proportional-depth layers."""
    model_short = model_name.split("/")[-1]
    trust_remote = model_short in TRUST_REMOTE_CODE_MODELS

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote)
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, torch_dtype=torch.float16,
        trust_remote_code=trust_remote,
    ).to(DEVICE).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get number of layers to compute proportional-depth indices
    try:
        n_layers = model.config.num_hidden_layers
    except AttributeError:
        n_layers = len(model.base_model.layers) if hasattr(model, 'base_model') else 12

    layer_indices = sorted(set([
        max(1, round(n_layers * 0.25)),
        max(1, round(n_layers * 0.50)),
        max(1, round(n_layers * 0.75)),
        n_layers,  # final layer
    ]))
    print(f"  {model_short}: {n_layers} layers, testing at {layer_indices}")

    all_layer_embs = {l: [] for l in layer_indices}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # (n_layers+1, B, seq, d)
        mask = inputs["attention_mask"].unsqueeze(-1).float()

        for li in layer_indices:
            if li < len(hidden_states):
                h = hidden_states[li].float()
                emb = (h * mask).sum(1) / mask.sum(1)
                all_layer_embs[li].append(
                    np.nan_to_num(emb.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                )

    del model
    torch.cuda.empty_cache()

    for l in layer_indices:
        if all_layer_embs[l]:
            all_layer_embs[l] = np.vstack(all_layer_embs[l])
        else:
            all_layer_embs[l] = None

    return all_layer_embs, layer_indices


def compute_kappa_nearest(X, y, K):
    """kappa = mean_class(min_{j!=k} ||mu_k - mu_j|| / (sigma_W * sqrt(d)))"""
    classes = np.unique(y)
    if len(classes) < 2:
        return None
    K_eff = len(classes)

    mu = {c: X[y == c].mean(0) for c in classes if (y == c).sum() >= 2}
    if len(mu) < 2:
        return None

    within_var = sum(np.sum((X[y == c] - mu[c])**2) for c in mu)
    n_total = sum((y == c).sum() for c in mu)
    sigma_W = float(np.sqrt(within_var / (n_total * X.shape[1])))
    if sigma_W < 1e-10:
        return None

    kappa_list = []
    for c in mu:
        dists = [np.linalg.norm(mu[c] - mu[j]) for j in mu if j != c]
        kappa_list.append(min(dists) / (sigma_W * np.sqrt(X.shape[1])))

    return float(np.mean(kappa_list))


def compute_knn_q(X, y, K):
    """q_norm = (1NN_acc - 1/K) / (1 - 1/K) using 80/20 split."""
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(42)
    n = len(X)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    classes_tr = set(np.unique(y_tr))
    classes_te = set(np.unique(y_te))
    if len(classes_tr) < 2 or not classes_te.issubset(classes_tr):
        return None

    K_eff = len(classes_tr)
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
    knn.fit(X_tr, y_tr)
    acc = float(knn.score(X_te, y_te))
    return (acc - 1.0/K_eff) / (1.0 - 1.0/K_eff)


# ===================================================
# FIT SINGLE-INTERCEPT SLOPE
# ===================================================
def fit_A_single(points):
    """Fit logit(q) = A * kappa + C; return A, r, p."""
    kappas = np.array([p["kappa"] for p in points])
    qs = np.array([p["q"] for p in points])

    # Clip q to avoid logit(0) or logit(1)
    qs = np.clip(qs, 0.01, 0.99)
    logit_q = np.array([sp_logit(float(q)) for q in qs])

    # Exclude invalid
    valid = np.isfinite(kappas) & np.isfinite(logit_q) & (kappas > 0)
    if valid.sum() < 3:
        return None, None, None

    k, l = kappas[valid], logit_q[valid]
    coeffs = np.polyfit(k, l, deg=1)
    A = float(coeffs[0])
    r, p = pearsonr(k, l)
    return A, float(r), float(p)


# ===================================================
# MAIN
# ===================================================
def main():
    print("=" * 65)
    print("CTI Prospective K-based Alpha Prediction Test")
    print("=" * 65)
    print(f"Formula (LOCKED): A_single(K) = {C_LOCKED:.4f} - {A_LOCKED:.4f} * log(K)")
    print()

    # Write pre-registration file FIRST (before any data loading)
    prereg = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "formula": "A_single = C - a * log(K)",
        "C": C_LOCKED,
        "a": A_LOCKED,
        "predictions": {ds: {"K": v["K"], "A_pred": float(v["A_pred"])}
                        for ds, v in PREREG_PREDICTIONS.items()},
        "threshold": "A_obs within 20% of A_pred = PASS",
        "calibration_datasets": [4, 14, 20, 28],
        "note": "All test datasets have K outside calibration range",
    }
    prereg_path = RESULTS_DIR / "cti_prospective_K_prereg.json"
    with open(prereg_path, "w") as f:
        json.dump(prereg, f, indent=2)
    print(f"Pre-registration saved: {prereg_path.name}")
    print(f"Timestamp: {prereg['timestamp']}")
    print()
    print("PRE-REGISTERED PREDICTIONS:")
    for ds, v in prereg["predictions"].items():
        print(f"  {ds:<20} K={v['K']:<3} A_pred = {v['A_pred']:.3f}")
    print()
    print("=" * 65)
    print("NOW LOADING DATA AND RUNNING EXPERIMENTS")
    print("(All predictions locked above - cannot be changed)")
    print("=" * 65)
    print()

    # Collect all data points per dataset
    all_points = {ds: [] for ds in DATASETS}

    for model_name in MODELS:
        model_short = model_name.split("/")[-1]
        print(f"\nModel: {model_short}")
        print("-" * 40)

        for ds_name, ds_cfg in DATASETS.items():
            print(f"  Dataset: {ds_name} (K={ds_cfg['K']})")

            # Load dataset
            try:
                if ds_cfg["hf_cfg"]:
                    raw = load_dataset(ds_cfg["hf_name"], ds_cfg["hf_cfg"],
                                       split=ds_cfg["split"])
                else:
                    raw = load_dataset(ds_cfg["hf_name"], split=ds_cfg["split"])

                texts = [str(x[ds_cfg["text_col"]]) for x in raw]
                labels = np.array([x[ds_cfg["label_col"]] for x in raw])

                # Subsample
                n = min(len(texts), ds_cfg["n_sample"])
                rng = np.random.default_rng(0)
                idx = rng.choice(len(texts), n, replace=False)
                texts = [texts[i] for i in idx]
                labels = labels[idx]

                print(f"    Loaded {len(texts)} samples, K={len(np.unique(labels))}")
            except Exception as e:
                print(f"    ERROR loading: {e}")
                continue

            # Extract embeddings
            try:
                layer_embs, layer_indices = get_embeddings_at_layers(
                    model_name, texts)
            except Exception as e:
                print(f"    ERROR embedding: {e}")
                continue

            # Compute kappa + q per layer
            for li in layer_indices:
                embs = layer_embs.get(li)
                if embs is None:
                    continue

                kappa = compute_kappa_nearest(embs, labels, ds_cfg["K"])
                q = compute_knn_q(embs, labels, ds_cfg["K"])

                if kappa is None or q is None:
                    continue

                all_points[ds_name].append({
                    "model": model_short,
                    "layer": li,
                    "kappa": float(kappa),
                    "q": float(q),
                    "K": ds_cfg["K"],
                })
                print(f"    Layer {li:2d}: kappa={kappa:.4f}, q={q:.4f}")

    # FIT AND COMPARE
    print()
    print("=" * 65)
    print("RESULTS: A_obs vs A_pred")
    print("=" * 65)
    print(f"  {'Dataset':<20} {'K':<4} {'A_pred':<10} {'A_obs':<10} {'r':<8} {'n':<5} {'Err%':<8} {'Pass'}")
    print("  " + "-" * 75)

    results = {}
    for ds_name, points in all_points.items():
        K_ds = DATASETS[ds_name]["K"]
        A_pred_val = PREREG_PREDICTIONS[ds_name]["A_pred"]

        if len(points) < 3:
            print(f"  {ds_name:<20} {K_ds:<4} {A_pred_val:<10.3f} INSUFFICIENT DATA (n={len(points)})")
            results[ds_name] = {"K": K_ds, "A_pred": A_pred_val, "status": "insufficient_data"}
            continue

        A_obs, r, p = fit_A_single(points)
        if A_obs is None:
            print(f"  {ds_name:<20} {K_ds:<4} {A_pred_val:<10.3f} FIT FAILED")
            results[ds_name] = {"K": K_ds, "A_pred": A_pred_val, "status": "fit_failed"}
            continue

        err_pct = abs(A_obs - A_pred_val) / abs(A_pred_val) * 100
        passed = err_pct < 20.0

        print(f"  {ds_name:<20} {K_ds:<4} {A_pred_val:<10.3f} {A_obs:<10.3f} "
              f"{r:<8.3f} {len(points):<5} {err_pct:<8.1f}% {'PASS' if passed else 'FAIL'}")

        results[ds_name] = {
            "K": K_ds,
            "A_pred": float(A_pred_val),
            "A_obs": float(A_obs),
            "r": float(r),
            "p": float(p),
            "n_points": len(points),
            "rel_error_pct": float(err_pct),
            "pass": bool(passed),
            "points": points,
        }

    # Summary
    passed = [ds for ds, v in results.items() if v.get("pass") is True]
    total_fit = [ds for ds, v in results.items() if "A_obs" in v]
    print()
    print(f"Overall: {len(passed)}/{len(total_fit)} datasets PASS (<20% error)")

    # Combined r(log K, A_obs) over all new + calibration data
    calib_K_A = [(4, 16.0), (14, 3.76), (20, 2.55), (28, 1.54), (10, 7.506), (10, 8.568)]
    new_K_A = [(v["K"], v["A_obs"]) for ds, v in results.items() if "A_obs" in v]
    all_K_A = calib_K_A + new_K_A
    log_K = np.log([x[0] for x in all_K_A])
    A_all = np.array([x[1] for x in all_K_A])
    r_all, p_all = pearsonr(log_K, A_all)
    print(f"Combined r(log K, A) = {r_all:.4f} (p={p_all:.5f}, n={len(all_K_A)} = calib + new)")

    # Save results
    out = {
        "experiment": "prospective_K_alpha_prediction",
        "prereg_timestamp": prereg["timestamp"],
        "formula": "A_single = C - a * log(K)",
        "C": C_LOCKED,
        "a": A_LOCKED,
        "datasets": results,
        "summary": {
            "pass_count": len(passed),
            "total_fit": len(total_fit),
            "combined_r_logK_A": float(r_all),
            "combined_p": float(p_all),
            "combined_n": len(all_K_A),
        },
    }
    out_path = RESULTS_DIR / "cti_prospective_K_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path.name}")


if __name__ == "__main__":
    main()
