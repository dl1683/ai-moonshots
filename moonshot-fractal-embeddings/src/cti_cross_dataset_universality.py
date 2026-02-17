#!/usr/bin/env python
"""
CROSS-DATASET UNIVERSALITY: q = sigmoid(kappa / sqrt(K)).

Tests the universal law across 6 datasets with varying K (number of classes):
  AGNews (K=18), 20Newsgroups (K=20), GoEmotions (K=28),
  TREC (K=42), DBPedia_Classes (K=68), CLINC (K=150)

Protocol:
  1. Run kappa sweeps on 4 new datasets (AGNews, 20Newsgroups, GoEmotions,
     DBPedia_Classes) with 2 models (Qwen2-0.5B, Qwen3-0.6B)
  2. Pool with existing CLINC (4 models) and TREC (2 models) data
  3. Fit sigmoid: q = sigmoid(kappa / sqrt(K)) on training split
  4. Leave-one-dataset-out cross-validation
  5. Report per-dataset MAE

Pre-registered criterion: Leave-one-dataset-out MAE < 0.05.

All models from MODEL_DIRECTORY.md.
"""

import json
import sys
import time
import gc
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model, ResidualScaler
from hierarchical_datasets import load_hierarchical_dataset


def extract_all_layer_reps(model, tokenizer, texts, alpha, device="cuda", batch_size=32):
    """Extract all layer representations with residual scaling."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**enc, output_hidden_states=True, return_dict=True)
            mask = enc.get("attention_mask",
                           torch.ones(enc["input_ids"].shape, device=device))
            for idx, hs in enumerate(outputs.hidden_states):
                hs_f = hs.float()
                m = mask.unsqueeze(-1).float()
                pooled = (hs_f * m).sum(1) / m.sum(1).clamp(min=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if idx not in all_hidden:
                    all_hidden[idx] = []
                all_hidden[idx].append(pooled.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in all_hidden.items()}


def compute_kappa(X, labels):
    """Compute spectral margin-to-noise ratio kappa = trace(S_B)/trace(S_W)."""
    try:
        if np.isnan(X).any():
            return 0.0
        unique_labels = np.unique(labels)
        grand_mean = X.mean(0)
        trace_sb = 0.0
        trace_sw = 0.0
        for lbl in unique_labels:
            mask = labels == lbl
            n_k = mask.sum()
            if n_k < 2:
                continue
            X_k = X[mask]
            mu_k = X_k.mean(0)
            trace_sb += n_k * np.sum((mu_k - grand_mean) ** 2)
            trace_sw += np.sum((X_k - mu_k) ** 2)
        if trace_sw < 1e-12:
            return 100.0 if trace_sb > 0 else 0.0
        return float(min(trace_sb / trace_sw, 100.0))
    except Exception:
        return 0.0


def compute_knn(X, labels, n_train_frac=0.7):
    """kNN accuracy."""
    n = len(labels)
    n_train = int(n_train_frac * n)
    if n_train < 5 or n - n_train < 5:
        return 0.0
    try:
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X[:n_train], labels[:n_train])
        return float(knn.score(X[n_train:], labels[n_train:]))
    except Exception:
        return 0.0


def compute_mean_obs(reps, labels):
    """Compute mean kNN and kappa across all layers."""
    knn_vals, kappa_vals = [], []
    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue
        knn_vals.append(compute_knn(X, labels))
        kv = compute_kappa(X, labels)
        if np.isfinite(kv):
            kappa_vals.append(kv)
    return {
        "knn": float(np.mean(knn_vals)) if knn_vals else 0,
        "kappa": float(np.mean(kappa_vals)) if kappa_vals else 0,
    }


def sigmoid(x, a, b, c, d):
    """Generalized sigmoid: d + (a - d) / (1 + exp(-b * (x - c)))"""
    return d + (a - d) / (1 + np.exp(-b * (x - c)))


def run_sweep(model_id, dataset_name, alphas, device):
    """Run alpha sweep for one model+dataset, return list of point dicts."""
    ds = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])
    K = len(np.unique(labels))

    model, tokenizer, n_layers, n_params = load_model(model_id, device)

    points = []
    for alpha in alphas:
        print(f"    alpha={alpha:.2f}", end="", flush=True)
        t0 = time.time()
        reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)
        obs = compute_mean_obs(reps, labels)
        elapsed = time.time() - t0
        print(f"  kNN={obs['knn']:.3f}  kappa={obs['kappa']:.4f}  ({elapsed:.1f}s)")
        points.append({
            "model": model_id,
            "dataset": dataset_name,
            "K": K,
            "alpha": alpha,
            "kappa": obs["kappa"],
            "knn": obs["knn"],
        })
        sys.stdout.flush()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return points


def main():
    print("=" * 70)
    print("CROSS-DATASET UNIVERSALITY: q = sigmoid(kappa / sqrt(K))")
    print("6 datasets, K from 18 to 150")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    # Models from MODEL_DIRECTORY.md
    sweep_models = ["Qwen/Qwen2-0.5B", "Qwen/Qwen3-0.6B"]

    # Datasets to run NEW sweeps on
    new_datasets = ["agnews", "20newsgroups", "goemotions", "dbpedia_classes"]

    # ============================================================
    # STEP 1: Load existing data
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 1: LOAD EXISTING DATA")
    print(f"{'='*70}")

    all_points = []

    # Load CLINC data from spectral collapse (4 models)
    sc_path = RESULTS_DIR / "cti_spectral_collapse.json"
    if sc_path.exists():
        with open(sc_path) as f:
            sc = json.load(f)
        for p in sc["all_points"]:
            all_points.append({
                "model": p["model"],
                "dataset": "clinc",
                "K": 150,
                "alpha": p["alpha"],
                "kappa": p["kappa"],
                "knn": p["knn"],
            })
        print(f"  Loaded {len(sc['all_points'])} CLINC points (4 models)")
    else:
        print("  WARNING: No CLINC spectral collapse data found")

    # Load TREC data from prospective prediction (2 models: Qwen2, Pythia-160m)
    pp_path = RESULTS_DIR / "cti_prospective_kappa.json"
    if pp_path.exists():
        with open(pp_path) as f:
            pp = json.load(f)
        trec_count = 0
        for p in pp["holdout"]["points"]:
            if p["dataset"] == "trec":
                all_points.append({
                    "model": p["model"],
                    "dataset": "trec",
                    "K": 42,  # Actual K from dataset loading
                    "alpha": p["alpha"],
                    "kappa": p["kappa"],
                    "knn": p["knn"],
                })
                trec_count += 1
        print(f"  Loaded {trec_count} TREC points (2 models)")
    else:
        print("  WARNING: No TREC data found")

    print(f"  Total existing points: {len(all_points)}")

    # ============================================================
    # STEP 2: Run NEW sweeps
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 2: NEW DATASET SWEEPS")
    print(f"{'='*70}")

    for ds_name in new_datasets:
        for model_id in sweep_models:
            short_model = model_id.split("/")[-1]
            print(f"\n  --- {short_model} on {ds_name} ---")
            pts = run_sweep(model_id, ds_name, alphas, device)
            all_points.extend(pts)
            print(f"  Added {len(pts)} points (K={pts[0]['K']})")

    # ============================================================
    # STEP 3: Compute normalized quality q
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 3: UNIVERSAL LAW TEST")
    print(f"{'='*70}")

    n_total = len(all_points)
    datasets_in_data = sorted(set(p["dataset"] for p in all_points))
    print(f"\n  Total points: {n_total}")
    print(f"  Datasets: {datasets_in_data}")
    for ds in datasets_in_data:
        ds_pts = [p for p in all_points if p["dataset"] == ds]
        K = ds_pts[0]["K"]
        models = sorted(set(p["model"] for p in ds_pts))
        print(f"    {ds:>20}: K={K}, {len(ds_pts)} points, {len(models)} models")

    # Compute normalized quality: q = (kNN - 1/K) / (1 - 1/K)
    for p in all_points:
        K = p["K"]
        q = (p["knn"] - 1.0/K) / (1.0 - 1.0/K) if K > 1 else p["knn"]
        p["q"] = q
        p["kappa_norm"] = p["kappa"] / np.sqrt(K)

    # ============================================================
    # OPTION A: Global correlation (kappa/sqrt(K) vs q)
    # ============================================================
    kappa_norm_arr = np.array([p["kappa_norm"] for p in all_points])
    q_arr = np.array([p["q"] for p in all_points])
    kappa_raw_arr = np.array([p["kappa"] for p in all_points])
    knn_arr = np.array([p["knn"] for p in all_points])

    # Compare normalizations
    norms = {
        "raw kappa vs kNN": (kappa_raw_arr, knn_arr),
        "raw kappa vs q": (kappa_raw_arr, q_arr),
        "kappa/sqrt(K) vs q": (kappa_norm_arr, q_arr),
        "kappa/K vs q": (np.array([p["kappa"]/p["K"] for p in all_points]), q_arr),
        "kappa/log(K) vs q": (np.array([p["kappa"]/np.log(p["K"]) for p in all_points]), q_arr),
    }

    print(f"\n  {'Normalization':>25} {'Spearman':>10} {'Pearson':>10}")
    print(f"  {'-'*50}")
    best_name, best_rho, best_r = None, 0, 0
    norm_results = {}
    for name, (x, y) in norms.items():
        rho, _ = spearmanr(x, y)
        r, _ = pearsonr(x, y)
        print(f"  {name:>25} {rho:>10.4f} {r:>10.4f}")
        norm_results[name] = {"rho": float(rho), "r": float(r)}
        if abs(r) > abs(best_r):
            best_name, best_rho, best_r = name, rho, r

    print(f"\n  BEST: {best_name} (r={best_r:.4f})")

    # ============================================================
    # OPTION B: Leave-one-dataset-out cross-validation
    # ============================================================
    print(f"\n{'='*70}")
    print("LEAVE-ONE-DATASET-OUT CROSS-VALIDATION")
    print(f"{'='*70}")

    lodo_results = {}

    for held_out_ds in datasets_in_data:
        # Training: all other datasets
        train_pts = [p for p in all_points if p["dataset"] != held_out_ds]
        test_pts = [p for p in all_points if p["dataset"] == held_out_ds]

        x_train = np.array([p["kappa_norm"] for p in train_pts])
        y_train = np.array([p["q"] for p in train_pts])
        x_test = np.array([p["kappa_norm"] for p in test_pts])
        y_test = np.array([p["q"] for p in test_pts])

        try:
            popt, _ = curve_fit(sigmoid, x_train, y_train,
                                p0=[0.6, 10, 0.03, -0.05], maxfev=10000)
            pred = sigmoid(x_test, *popt)
            mae = float(np.mean(np.abs(y_test - pred)))
            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            K = test_pts[0]["K"]
            n_models = len(set(p["model"] for p in test_pts))
            status = "PASS" if mae < 0.05 else "FAIL"
            print(f"  {held_out_ds:>20} (K={K:>3}): MAE={mae:.4f}  R2={r2:.4f}  "
                  f"[{n_models} models]  [{status}]")
            lodo_results[held_out_ds] = {
                "K": K,
                "n_points": len(test_pts),
                "n_models": n_models,
                "mae": mae,
                "r2": float(r2),
                "passes": mae < 0.05,
            }
        except Exception as e:
            print(f"  {held_out_ds:>20}: FIT FAILED ({e})")
            lodo_results[held_out_ds] = {"error": str(e)}

    # Overall LODO MAE
    lodo_maes = [v["mae"] for v in lodo_results.values() if "mae" in v]
    if lodo_maes:
        mean_lodo = np.mean(lodo_maes)
        n_pass = sum(1 for m in lodo_maes if m < 0.05)
        print(f"\n  OVERALL LODO MAE: {mean_lodo:.4f}")
        print(f"  Datasets passing (MAE < 0.05): {n_pass}/{len(lodo_maes)}")
        print(f"  Pre-registered: mean LODO MAE < 0.05")
        if mean_lodo < 0.05:
            print(f"  UNIVERSAL LAW CONFIRMED: q = sigmoid(kappa / sqrt(K))")
        elif mean_lodo < 0.10:
            print(f"  PARTIAL UNIVERSALITY (MAE < 0.10)")
        else:
            print(f"  NOT UNIVERSAL (MAE >= 0.10)")

    # ============================================================
    # OPTION C: Fit global sigmoid and report
    # ============================================================
    print(f"\n{'='*70}")
    print("GLOBAL SIGMOID FIT: q = sigmoid(kappa / sqrt(K))")
    print(f"{'='*70}")

    try:
        popt_global, pcov_global = curve_fit(
            sigmoid, kappa_norm_arr, q_arr,
            p0=[0.6, 10, 0.03, -0.05], maxfev=10000
        )
        pred_global = sigmoid(kappa_norm_arr, *popt_global)
        mae_global = float(np.mean(np.abs(q_arr - pred_global)))
        ss_res = np.sum((q_arr - pred_global) ** 2)
        ss_tot = np.sum((q_arr - q_arr.mean()) ** 2)
        r2_global = 1 - ss_res / ss_tot

        print(f"\n  Sigmoid params: a={popt_global[0]:.4f}, b={popt_global[1]:.2f}, "
              f"c={popt_global[2]:.4f}, d={popt_global[3]:.4f}")
        print(f"  Global R2 = {r2_global:.4f}")
        print(f"  Global MAE = {mae_global:.4f}")

        # Per-dataset residuals
        print(f"\n  PER-DATASET RESIDUALS:")
        for ds in datasets_in_data:
            ds_mask = np.array([p["dataset"] == ds for p in all_points])
            ds_resid = np.abs(q_arr[ds_mask] - pred_global[ds_mask])
            K = [p["K"] for p in all_points if p["dataset"] == ds][0]
            print(f"    {ds:>20} (K={K:>3}): MAE={np.mean(ds_resid):.4f}, "
                  f"max={np.max(ds_resid):.4f}")

        global_fit = {
            "params": popt_global.tolist(),
            "r2": float(r2_global),
            "mae": float(mae_global),
        }
    except Exception as e:
        print(f"  Global fit failed: {e}")
        global_fit = {"error": str(e)}

    # ============================================================
    # SAVE
    # ============================================================
    out = {
        "experiment": "cross_dataset_universality",
        "universal_law": "q = sigmoid(kappa / sqrt(K))",
        "preregistered_criterion": "LODO MAE < 0.05",
        "datasets": {ds: {"K": [p["K"] for p in all_points if p["dataset"] == ds][0],
                          "n_points": sum(1 for p in all_points if p["dataset"] == ds)}
                     for ds in datasets_in_data},
        "n_total_points": n_total,
        "normalizations": norm_results,
        "best_normalization": best_name,
        "lodo_cv": lodo_results,
        "lodo_mean_mae": float(mean_lodo) if lodo_maes else None,
        "global_fit": global_fit,
        "all_points": all_points,
    }

    out_path = RESULTS_DIR / "cti_cross_dataset_universality.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
