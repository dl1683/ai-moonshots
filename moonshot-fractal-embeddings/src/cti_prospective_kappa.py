#!/usr/bin/env python
"""
PRE-REGISTERED PROSPECTIVE PREDICTION: kNN from kappa alone.

Step 1: Fit master curve kNN = f(kappa) on 3 training models (CLINC dataset)
Step 2: Predict kNN for held-out conditions:
  - Held-out model: Pythia-160M (from MODEL_DIRECTORY.md, not in training)
  - Held-out dataset: TREC (different from CLINC training)
  - Held-out model+dataset: Pythia-160M on TREC

Pre-registered criterion:
  Mean absolute prediction error < 0.05 across all held-out conditions.

If predictions land on the master curve with no refit, this is strong
evidence for a universal law, not just a good empirical fit.

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
    """Compute spectral margin-to-noise ratio."""
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
    """Mean kNN and kappa across layers."""
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
        "knn_acc": float(np.mean(knn_vals)) if knn_vals else 0,
        "kappa": float(np.mean(kappa_vals)) if kappa_vals else 0,
    }


def sigmoid(x, a, b, c, d):
    """Generalized sigmoid: d + (a - d) / (1 + exp(-b * (x - c)))"""
    return d + (a - d) / (1 + np.exp(-b * (x - c)))


def run_sweep(model_id, dataset_name, alphas, device):
    """Run alpha sweep and return list of (kappa, knn) points."""
    ds = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])

    model, tokenizer, n_layers, n_params = load_model(model_id, device)

    points = []
    for alpha in alphas:
        print(f"  alpha={alpha:.2f}", end="", flush=True)
        t0 = time.time()
        reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)
        obs = compute_mean_obs(reps, labels)
        elapsed = time.time() - t0
        print(f"  kNN={obs['knn_acc']:.3f}  kappa={obs['kappa']:.4f}  ({elapsed:.1f}s)")
        points.append({
            "model": model_id,
            "dataset": dataset_name,
            "alpha": alpha,
            "kappa": obs["kappa"],
            "knn": obs["knn_acc"],
        })
        sys.stdout.flush()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return points


def main():
    print("=" * 70)
    print("PRE-REGISTERED PROSPECTIVE PREDICTION")
    print("Can we predict kNN from kappa alone on unseen models/datasets?")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    # ============================================================
    # STEP 1: Fit master curve on training models + CLINC
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 1: FIT MASTER CURVE (training set)")
    print(f"{'='*70}")

    # Load existing data from spectral collapse experiment
    collapse_path = RESULTS_DIR / "cti_spectral_collapse.json"
    if collapse_path.exists():
        with open(collapse_path) as f:
            collapse_data = json.load(f)
        train_points = [p for p in collapse_data["all_points"]
                        if p["model"] != "EleutherAI/pythia-410m"]  # Hold out Pythia
        print(f"\n  Loaded {len(train_points)} training points (Qwen2, SmolLM2, Qwen3 on CLINC)")
    else:
        print("  No collapse data found, running fresh...")
        train_models = ["Qwen/Qwen2-0.5B", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen3-0.6B"]
        train_points = []
        for model_id in train_models:
            print(f"\n  {model_id}:")
            pts = run_sweep(model_id, "clinc", alphas, device)
            train_points.extend(pts)

    # Fit sigmoid: kNN = sigmoid(kappa)
    train_kappas = np.array([p["kappa"] for p in train_points])
    train_knns = np.array([p["knn"] for p in train_points])

    try:
        popt, pcov = curve_fit(sigmoid, train_kappas, train_knns,
                               p0=[0.6, 10, 0.4, 0.1], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        pred_train = sigmoid(train_kappas, *popt)
        ss_res = np.sum((train_knns - pred_train) ** 2)
        ss_tot = np.sum((train_knns - train_knns.mean()) ** 2)
        r2_train = 1 - ss_res / ss_tot
        mae_train = np.mean(np.abs(train_knns - pred_train))

        print(f"\n  Sigmoid fit: kNN = {popt[3]:.3f} + ({popt[0]:.3f} - {popt[3]:.3f}) / "
              f"(1 + exp(-{popt[1]:.1f} * (kappa - {popt[2]:.3f})))")
        print(f"  R^2 = {r2_train:.4f}")
        print(f"  MAE (train) = {mae_train:.4f}")
    except Exception as e:
        print(f"  Sigmoid fit failed: {e}")
        # Fallback: linear fit
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(train_kappas, train_knns, 2)
        popt = coeffs
        pred_train = np.polyval(coeffs, train_kappas)
        mae_train = np.mean(np.abs(train_knns - pred_train))
        r2_train = 1 - np.sum((train_knns - pred_train)**2) / np.sum((train_knns - train_knns.mean())**2)
        print(f"  Polynomial fit R^2 = {r2_train:.4f}, MAE = {mae_train:.4f}")

    # ============================================================
    # STEP 2: Held-out predictions
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 2: HELD-OUT PREDICTIONS (no refit)")
    print(f"{'='*70}")

    holdout_conditions = [
        # Held-out model (same dataset)
        ("EleutherAI/pythia-410m", "clinc"),
        # Held-out model (from MODEL_DIRECTORY.md, completely new)
        ("EleutherAI/pythia-160m", "clinc"),
        # Held-out dataset (training model)
        ("Qwen/Qwen2-0.5B", "trec"),
        # Held-out model + dataset
        ("EleutherAI/pythia-160m", "trec"),
    ]

    all_holdout_points = []
    all_errors = []

    for model_id, dataset_name in holdout_conditions:
        print(f"\n  --- {model_id} on {dataset_name} ---")
        pts = run_sweep(model_id, dataset_name, alphas, device)

        for p in pts:
            kappa = p["kappa"]
            knn_actual = p["knn"]

            # Predict from master curve (NO REFIT)
            try:
                knn_pred = float(sigmoid(kappa, *popt))
            except Exception:
                knn_pred = float(np.polyval(popt, kappa))

            error = abs(knn_actual - knn_pred)
            all_errors.append(error)
            p["knn_predicted"] = knn_pred
            p["error"] = error
            all_holdout_points.append(p)

    # ============================================================
    # STEP 3: EVALUATE PREDICTIONS
    # ============================================================
    print(f"\n{'='*70}")
    print("PROSPECTIVE PREDICTION RESULTS")
    print(f"{'='*70}")

    print(f"\n  {'Model':>30} {'Dataset':>8} {'alpha':>6} {'kappa':>8} "
          f"{'kNN_act':>8} {'kNN_pred':>9} {'error':>7}")
    print(f"  {'-'*85}")

    for p in all_holdout_points:
        short = p["model"].split("/")[-1]
        print(f"  {short:>30} {p['dataset']:>8} {p['alpha']:>6.2f} "
              f"{p['kappa']:>8.4f} {p['knn']:>8.3f} {p['knn_predicted']:>9.3f} "
              f"{p['error']:>7.3f}")

    errors = np.array(all_errors)
    mae = np.mean(errors)
    max_error = np.max(errors)
    median_error = np.median(errors)

    print(f"\n  SUMMARY:")
    print(f"    N predictions: {len(errors)}")
    print(f"    MAE: {mae:.4f}")
    print(f"    Median error: {median_error:.4f}")
    print(f"    Max error: {max_error:.4f}")
    print(f"    Pre-registered: MAE < 0.05")

    if mae < 0.05:
        print(f"\n    PREDICTION PASSES (MAE={mae:.4f} < 0.05)")
        print(f"    kappa is a UNIVERSAL predictor of kNN quality")
    else:
        print(f"\n    PREDICTION FAILS (MAE={mae:.4f} >= 0.05)")
        print(f"    kappa may not generalize perfectly across conditions")

    # Per-condition breakdown
    print(f"\n  PER-CONDITION MAE:")
    for model_id, dataset_name in holdout_conditions:
        cond_errors = [p["error"] for p in all_holdout_points
                       if p["model"] == model_id and p["dataset"] == dataset_name]
        cond_mae = np.mean(cond_errors)
        short = model_id.split("/")[-1]
        status = "PASS" if cond_mae < 0.05 else "FAIL"
        print(f"    {short:>20} / {dataset_name:>8}: MAE = {cond_mae:.4f}  [{status}]")

    # Correlation on held-out data
    from scipy.stats import spearmanr, pearsonr
    ho_kappas = np.array([p["kappa"] for p in all_holdout_points])
    ho_knns = np.array([p["knn"] for p in all_holdout_points])
    rho, p_rho = spearmanr(ho_kappas, ho_knns)
    r, p_r = pearsonr(ho_kappas, ho_knns)
    print(f"\n  HELD-OUT CORRELATION (kappa vs kNN):")
    print(f"    Spearman rho = {rho:.4f} (p = {p_rho:.6f})")
    print(f"    Pearson r = {r:.4f} (p = {p_r:.6f})")

    # Save
    out = {
        "experiment": "prospective_kappa_prediction",
        "preregistered_criterion": "MAE < 0.05",
        "training": {
            "models": ["Qwen/Qwen2-0.5B", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen3-0.6B"],
            "dataset": "clinc",
            "n_points": len(train_points),
            "sigmoid_params": popt.tolist() if hasattr(popt, 'tolist') else list(popt),
            "r2": float(r2_train),
            "mae": float(mae_train),
        },
        "holdout": {
            "conditions": [{"model": m, "dataset": d} for m, d in holdout_conditions],
            "points": all_holdout_points,
            "mae": float(mae),
            "max_error": float(max_error),
            "median_error": float(median_error),
            "passes": bool(mae < 0.05),
        },
        "holdout_correlation": {
            "rho": float(rho),
            "p_rho": float(p_rho),
            "r": float(r),
            "p_r": float(p_r),
        },
    }
    out_path = RESULTS_DIR / "cti_prospective_kappa.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
