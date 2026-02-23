"""
CTI Per-Dataset Beta Estimation
Tests whether beta varies by dataset semantic density.

Hypothesis: emotion/sentiment datasets (semantically dense) -> beta closer to -1
           topic/task datasets (semantically sparse) -> beta closer to -0.5

Pre-registered: Feb 23, 2026
"""

import json
import os
import numpy as np
from scipy.special import logit as scipy_logit
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CACHE_PATTERN = "kappa_near_cache_"
TIMESTAMP = datetime.now().isoformat()

print("=" * 70)
print("CTI PER-DATASET BETA ESTIMATION")
print(f"Timestamp: {TIMESTAMP}")
print("=" * 70)

# Pre-registered semantic density hypothesis
DENSE_DATASETS = {"go_emotions", "emotion"}        # all classes semantically similar
SPARSE_DATASETS = {"banking77", "amazon_massive", "langid", "dbpedia"}  # distinct classes
MEDIUM_DATASETS = {"20newsgroups", "agnews", "news_category", "yahoo"}


def load_all_points():
    points = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.startswith(CACHE_PATTERN) or not fname.endswith(".json"):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            kappa = entry.get("kappa_nearest")
            K = entry.get("K")
            logit_q = entry.get("logit_q")
            model = entry.get("model", "")
            dataset = entry.get("dataset", "")
            if logit_q is None:
                q = entry.get("q")
                if q is None or K is None or q <= 1.0 / K or q >= 1.0:
                    continue
                logit_q = float(scipy_logit(q))
            else:
                logit_q = float(logit_q)
            if kappa is None or K is None or kappa <= 0 or K < 3:
                continue
            if not np.isfinite(logit_q):
                continue
            log_km1 = float(np.log(K - 1))
            points.append({
                "kappa": float(kappa),
                "logit_q": logit_q,
                "log_km1": log_km1,
                "K": K,
                "model": model,
                "dataset": dataset,
            })
    return points


def fit_unconstrained(kappas, log_km1s, logit_qs):
    X = np.column_stack([kappas, log_km1s, np.ones(len(kappas))])
    y = np.array(logit_qs)
    result = np.linalg.lstsq(X, y, rcond=None)
    c = result[0]
    pred = X @ c
    residuals = y - pred
    r2 = float(1 - np.var(residuals) / np.var(y))
    return float(c[0]), float(c[1]), float(c[2]), r2


points = load_all_points()
print(f"Loaded {len(points)} valid points")
datasets = sorted(set(p["dataset"] for p in points))
print(f"Datasets: {datasets}")

# ============================================================
# Per-dataset beta estimation
# ============================================================
print("\nPer-dataset beta estimates:")
print(f"{'Dataset':25s} {'K':>4s} {'n':>4s} {'alpha':>7s} {'beta':>7s} {'R2':>6s} {'density':12s}")
print("-" * 75)

dataset_betas = {}
for d in datasets:
    pts = [p for p in points if p["dataset"] == d]
    if len(pts) < 10:
        continue
    kappas = [p["kappa"] for p in pts]
    log_km1s = [p["log_km1"] for p in pts]
    logit_qs = [p["logit_q"] for p in pts]
    K_vals = sorted(set(p["K"] for p in pts))
    K_str = ",".join(str(k) for k in K_vals)

    if len(set(log_km1s)) < 2:
        # Single K — can't estimate beta from K variation
        # Fit kappa-only
        X = np.column_stack([kappas, np.ones(len(kappas))])
        y = np.array(logit_qs)
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        density = ("DENSE" if d in DENSE_DATASETS else
                   "SPARSE" if d in SPARSE_DATASETS else "MEDIUM")
        print(f"{d:25s} {K_str:>4s} {len(pts):>4d} {float(c[0]):>7.3f} {'N/A':>7s} {'N/A':>6s} {density:12s}")
        dataset_betas[d] = None
        continue

    alpha, beta, C, r2 = fit_unconstrained(kappas, log_km1s, logit_qs)
    density = ("DENSE" if d in DENSE_DATASETS else
               "SPARSE" if d in SPARSE_DATASETS else "MEDIUM")
    print(f"{d:25s} {K_str:>4s} {len(pts):>4d} {alpha:>7.3f} {beta:>7.3f} {r2:>6.3f} {density:12s}")
    dataset_betas[d] = beta

# ============================================================
# Density hypothesis test
# ============================================================
print("\n" + "=" * 60)
print("SEMANTIC DENSITY HYPOTHESIS TEST")
print("=" * 60)
print("Pre-registered: dense -> beta closer to -1, sparse -> beta closer to -0.5")

dense_betas = [dataset_betas[d] for d in datasets
               if d in DENSE_DATASETS and dataset_betas.get(d) is not None]
sparse_betas = [dataset_betas[d] for d in datasets
                if d in SPARSE_DATASETS and dataset_betas.get(d) is not None]
medium_betas = [dataset_betas[d] for d in datasets
                if d in MEDIUM_DATASETS and dataset_betas.get(d) is not None]

print(f"\nDense datasets: {[d for d in datasets if d in DENSE_DATASETS]}")
if dense_betas:
    print(f"  Beta values: {[f'{b:.3f}' for b in dense_betas]}")
    print(f"  Mean beta: {np.mean(dense_betas):.3f}")
else:
    print("  No multi-K dense datasets with beta estimates")

print(f"\nSparse datasets: {[d for d in datasets if d in SPARSE_DATASETS]}")
if sparse_betas:
    print(f"  Beta values: {[f'{b:.3f}' for b in sparse_betas]}")
    print(f"  Mean beta: {np.mean(sparse_betas):.3f}")

print(f"\nMedium datasets: {[d for d in datasets if d in MEDIUM_DATASETS]}")
if medium_betas:
    print(f"  Beta values: {[f'{b:.3f}' for b in medium_betas]}")
    print(f"  Mean beta: {np.mean(medium_betas):.3f}")

# ============================================================
# Per-K beta for models that appear at multiple K values
# ============================================================
print("\n" + "=" * 60)
print("PER-MODEL BETA (models at multiple K values)")
print("=" * 60)
models = sorted(set(p["model"] for p in points))
multi_K_models = []
for m in models:
    pts_m = [p for p in points if p["model"] == m]
    Ks = sorted(set(p["K"] for p in pts_m))
    if len(Ks) >= 3:
        multi_K_models.append(m)

print(f"Models with >= 3 K values: {len(multi_K_models)}")
print(f"\n{'Model':35s} {'n_K':>3s} {'n':>4s} {'alpha':>7s} {'beta':>7s} {'R2':>6s}")
print("-" * 65)

model_betas = {}
for m in multi_K_models:
    pts_m = [p for p in points if p["model"] == m]
    kappas = [p["kappa"] for p in pts_m]
    log_km1s = [p["log_km1"] for p in pts_m]
    logit_qs = [p["logit_q"] for p in pts_m]
    Ks = sorted(set(p["K"] for p in pts_m))

    if len(set(log_km1s)) < 2:
        continue
    alpha, beta, C, r2 = fit_unconstrained(kappas, log_km1s, logit_qs)
    print(f"{m:35s} {len(Ks):>3d} {len(pts_m):>4d} {alpha:>7.3f} {beta:>7.3f} {r2:>6.3f}")
    model_betas[m] = beta

if model_betas:
    beta_vals = list(model_betas.values())
    print(f"\nAll-model beta mean: {np.mean(beta_vals):.3f} +/- {np.std(beta_vals):.3f}")
    print(f"All-model beta CV: {np.std(beta_vals)/abs(np.mean(beta_vals))*100:.1f}%")

# ============================================================
# Save results
# ============================================================
output = {
    "experiment": "per_dataset_beta_estimation",
    "timestamp": TIMESTAMP,
    "n_total": len(points),
    "dataset_betas": {k: v for k, v in dataset_betas.items() if v is not None},
    "model_betas": model_betas,
    "density_groups": {
        "dense": {d: dataset_betas.get(d) for d in DENSE_DATASETS if d in dataset_betas},
        "sparse": {d: dataset_betas.get(d) for d in SPARSE_DATASETS if d in dataset_betas},
        "medium": {d: dataset_betas.get(d) for d in MEDIUM_DATASETS if d in dataset_betas},
    },
    "hypothesis": "dense datasets beta closer to -1, sparse closer to -0.5",
}
out_path = os.path.join(RESULTS_DIR, "cti_per_dataset_beta.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {out_path}")
