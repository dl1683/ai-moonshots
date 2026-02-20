#!/usr/bin/env python
"""
NON-GAUSSIAN CORRECTION: ELIMINATE DATASET-SPECIFIC OFFSETS

The Gaussian theory predicts q = sigmoid(kappa/sqrt(K)) but real neural
network representations have dataset-specific offsets (universal collapse
R^2=0.587). This script:

1. Extracts representations from reference model at alpha=1.0
2. Computes non-Gaussian statistics per dataset:
   - eta: within-class isotropy
   - inter-class distance CV (coefficient of variation)
   - nearest-class ratio (how clustered are classes?)
   - kurtosis excess
   - effective dimensionality
3. Tests which statistics explain the residual offsets
4. Builds a corrected formula with parameter-free prediction

Goal: universal collapse WITHOUT per-dataset fitting.
"""

import json
import sys
import gc
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model, ResidualScaler
from hierarchical_datasets import load_hierarchical_dataset


def sigmoid(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def extract_reps(model, tokenizer, texts, alpha=1.0, device="cuda", batch_size=32):
    """Extract final-layer representations with given alpha."""
    all_reps = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, return_dict=True)
            mask = enc.get("attention_mask",
                           torch.ones(enc["input_ids"].shape, device=device))
            # Use penultimate layer (more informative than final)
            hs = out.hidden_states[-2].float()
            m = mask.unsqueeze(-1).float()
            pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            all_reps.append(pooled.cpu().numpy())

    return np.concatenate(all_reps, axis=0)


def compute_nongaussian_stats(X, labels):
    """Compute a battery of non-Gaussian statistics.

    Returns dict with:
    - kappa: tr(S_B)/tr(S_W)
    - eta: within-class isotropy
    - interclass_cv: coefficient of variation of inter-class distances
    - nearest_class_ratio: mean(min inter-class dist) / mean(all inter-class dist)
    - kurtosis_excess: average excess kurtosis of within-class residuals
    - effective_dim: effective dimensionality of representations
    - class_overlap: fraction of nearest neighbors from different classes
    """
    classes = np.unique(labels)
    K = len(classes)
    d = X.shape[1]
    grand_mean = X.mean(axis=0)

    # Class centroids
    centroids = np.zeros((K, d))
    class_sizes = []
    for i, c in enumerate(classes):
        mask = labels == c
        centroids[i] = X[mask].mean(axis=0)
        class_sizes.append(mask.sum())
    class_sizes = np.array(class_sizes)

    # --- kappa and eta ---
    S_W_trace = 0.0
    S_W_sq_trace = 0.0
    S_B_trace = 0.0

    within_residuals = []
    for i, c in enumerate(classes):
        X_c = X[labels == c]
        n_c = len(X_c)
        diff = X_c - centroids[i]
        S_W_trace += np.sum(diff ** 2)

        # For S_W^2 trace, need eigenvalues of S_W
        # Approximate: sum of squared column variances
        col_vars = np.var(diff, axis=0)
        S_W_sq_trace += n_c * np.sum(col_vars ** 2)

        within_residuals.append(diff)

        mean_diff = centroids[i] - grand_mean
        S_B_trace += n_c * np.sum(mean_diff ** 2)

    kappa = S_B_trace / max(S_W_trace, 1e-10)
    eta = S_W_trace ** 2 / max(d * S_W_sq_trace, 1e-10) if S_W_sq_trace > 0 else 0

    # --- Inter-class distance statistics ---
    # Pairwise distances between class centroids
    if K > 1:
        centroid_dists = pdist(centroids, metric='euclidean')
        interclass_cv = float(np.std(centroid_dists) / max(np.mean(centroid_dists), 1e-10))

        # Nearest-class ratio
        dist_matrix = squareform(centroid_dists)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = dist_matrix.min(axis=1)
        nearest_class_ratio = float(np.mean(min_dists) / max(np.mean(centroid_dists), 1e-10))
    else:
        interclass_cv = 0.0
        nearest_class_ratio = 1.0

    # --- Kurtosis excess ---
    # Average excess kurtosis of within-class residuals along each dimension
    all_residuals = np.concatenate(within_residuals, axis=0)
    n_samples = len(all_residuals)
    if n_samples > 4:
        # Per-dimension kurtosis
        col_means = all_residuals.mean(axis=0)
        col_stds = all_residuals.std(axis=0)
        col_stds[col_stds < 1e-10] = 1.0
        standardized = (all_residuals - col_means) / col_stds
        kurtosis_excess = float(np.mean(np.mean(standardized ** 4, axis=0) - 3.0))
    else:
        kurtosis_excess = 0.0

    # --- Effective dimensionality ---
    # Using eigenvalue entropy
    cov = np.cov(X.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    probs = eigvals / eigvals.sum()
    eff_dim = float(np.exp(-np.sum(probs * np.log(probs + 1e-20))))

    # --- Class overlap (1-NN from different class) ---
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, labels)
    _, indices = knn.kneighbors(X)
    # For each point, check if 2nd nearest neighbor (1st is self) is from same class
    different_class = 0
    for i in range(len(X)):
        if labels[indices[i, 1]] != labels[i]:
            different_class += 1
    class_overlap = different_class / len(X)

    return {
        "kappa": float(kappa),
        "eta": float(eta),
        "interclass_cv": interclass_cv,
        "nearest_class_ratio": nearest_class_ratio,
        "kurtosis_excess": kurtosis_excess,
        "effective_dim": eff_dim,
        "class_overlap": class_overlap,
        "K": int(K),
        "n": int(len(labels)),
        "d": int(d),
    }


def load_all_kappa_q_data():
    """Load all (kappa, q, dataset) points from cached results."""
    all_points = []

    # CLINC from geometry mediator
    with open(RESULTS_DIR / "cti_geometry_mediator.json") as f:
        clinc_raw = json.load(f)
    for p in clinc_raw["all_points"]:
        K = 150
        q = (p["knn"] - 1.0 / K) / (1.0 - 1.0 / K)
        all_points.append({
            "dataset": "clinc", "K": K,
            "kappa": p["kappa"], "knn": p["knn"], "q": q,
        })

    # AGNews and DBPedia
    for ds in ["agnews", "dbpedia_classes"]:
        with open(RESULTS_DIR / f"cti_multidata_{ds}_cache.json") as f:
            data = json.load(f)
        for p in data:
            K = p["n_classes"]
            q = (p["knn"] - 1.0 / K) / (1.0 - 1.0 / K)
            all_points.append({
                "dataset": p["dataset"], "K": K,
                "kappa": p["kappa"], "knn": p["knn"], "q": q,
            })

    # Yahoo and arXiv
    with open(RESULTS_DIR / "cti_blind_prediction.json") as f:
        blind = json.load(f)
    for p in blind["blind_points"]:
        K = p["K"]
        q = (p["knn"] - 1.0 / K) / (1.0 - 1.0 / K)
        all_points.append({
            "dataset": p["dataset"], "K": K,
            "kappa": p["kappa"], "knn": p["knn"], "q": q,
        })

    return all_points


def main():
    print("=" * 70)
    print("NON-GAUSSIAN CORRECTION FOR UNIVERSAL COLLAPSE")
    print("=" * 70)

    # Step 1: Load all (kappa, q) data
    all_points = load_all_kappa_q_data()
    print(f"Total points: {len(all_points)}")

    datasets_seen = sorted(set(p["dataset"] for p in all_points))
    for ds in datasets_seen:
        n_pts = sum(1 for p in all_points if p["dataset"] == ds)
        K = [p["K"] for p in all_points if p["dataset"] == ds][0]
        print(f"  {ds:>20}: {n_pts} points, K={K}")

    # Step 2: Extract non-Gaussian statistics from reference model
    print(f"\n{'='*70}")
    print("EXTRACTING NON-GAUSSIAN STATISTICS (Qwen2-0.5B, alpha=1.0)")
    print(f"{'='*70}")

    model_id = "Qwen/Qwen2-0.5B"
    model, tokenizer, _, _ = load_model(model_id, device="cuda")
    model.eval()

    ds_stats = {}
    for ds_name in datasets_seen:
        print(f"\n  Processing {ds_name}...")
        ds = load_hierarchical_dataset(ds_name, split="test", max_samples=2000)
        texts = [s.text for s in ds.samples]
        labels = np.array([s.level1_label for s in ds.samples])

        # Encode labels as integers
        unique_labels = np.unique(labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labels_int = np.array([label_map[l] for l in labels])

        # Extract representations
        X = extract_reps(model, tokenizer, texts, alpha=1.0)
        stats = compute_nongaussian_stats(X, labels_int)
        ds_stats[ds_name] = stats

        print(f"    kappa={stats['kappa']:.4f}, eta={stats['eta']:.4f}")
        print(f"    interclass_cv={stats['interclass_cv']:.4f}, "
              f"nearest_class_ratio={stats['nearest_class_ratio']:.4f}")
        print(f"    kurtosis_excess={stats['kurtosis_excess']:.4f}, "
              f"eff_dim={stats['effective_dim']:.1f}")
        print(f"    class_overlap={stats['class_overlap']:.4f}")

    # Clean up model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Compute residual offsets from naive sigmoid
    print(f"\n{'='*70}")
    print("RESIDUAL OFFSETS FROM NAIVE kappa/sqrt(K) SIGMOID")
    print(f"{'='*70}")

    kappas = np.array([p["kappa"] for p in all_points])
    qs = np.array([p["q"] for p in all_points])
    Ks = np.array([p["K"] for p in all_points])
    ds_labels = np.array([p["dataset"] for p in all_points])

    x_norm = kappas / np.sqrt(Ks)

    # Fit global sigmoid
    popt_global, _ = curve_fit(sigmoid, x_norm, qs,
                                p0=[0.6, 10, np.median(x_norm), 0.0],
                                maxfev=10000)
    pred_global = sigmoid(x_norm, *popt_global)
    residuals = qs - pred_global

    # Per-dataset offsets
    offsets = {}
    for ds_name in datasets_seen:
        mask = ds_labels == ds_name
        offset = float(residuals[mask].mean())
        offsets[ds_name] = offset
        print(f"  {ds_name:>20}: offset={offset:+.4f}")

    # Step 4: Correlate offsets with non-Gaussian statistics
    print(f"\n{'='*70}")
    print("CORRELATION: OFFSETS vs NON-GAUSSIAN STATISTICS")
    print(f"{'='*70}")

    stat_names = ["eta", "interclass_cv", "nearest_class_ratio",
                  "kurtosis_excess", "effective_dim", "class_overlap"]

    offset_arr = np.array([offsets[ds] for ds in datasets_seen])

    correlations = {}
    for stat_name in stat_names:
        stat_arr = np.array([ds_stats[ds][stat_name] for ds in datasets_seen])
        if len(set(stat_arr)) >= 3:
            rho, p = spearmanr(stat_arr, offset_arr)
            r, pr = pearsonr(stat_arr, offset_arr)
        else:
            rho = r = 0.0
            p = pr = 1.0

        correlations[stat_name] = {"rho": float(rho), "r": float(r),
                                    "p_spearman": float(p), "p_pearson": float(pr)}
        print(f"  {stat_name:>25}: rho={rho:+.4f} (p={p:.4f}), r={r:+.4f} (p={pr:.4f})")

    # Step 5: Build corrected formula using best predictor
    print(f"\n{'='*70}")
    print("CORRECTED UNIVERSAL FORMULA")
    print(f"{'='*70}")

    # Try multiple corrections
    corrections = {}

    # Correction 1: kappa / sqrt(K) with nearest_class_ratio adjustment
    ncr_arr = np.array([ds_stats[p["dataset"]]["nearest_class_ratio"] for p in all_points])
    x_corr1 = kappas / np.sqrt(Ks) * ncr_arr  # scale by nearest-class ratio
    try:
        popt1, _ = curve_fit(sigmoid, x_corr1, qs,
                              p0=[0.6, 10, np.median(x_corr1), 0.0], maxfev=10000)
        pred1 = sigmoid(x_corr1, *popt1)
        r2_1 = 1 - np.sum((qs - pred1) ** 2) / np.sum((qs - qs.mean()) ** 2)
        mae_1 = float(np.mean(np.abs(qs - pred1)))
        corrections["kappa*ncr/sqrt(K)"] = {"r2": float(r2_1), "mae": mae_1}
        print(f"  kappa*ncr/sqrt(K): R^2={r2_1:.4f}, MAE={mae_1:.4f}")
    except Exception as e:
        print(f"  kappa*ncr/sqrt(K): FAILED ({e})")

    # Correction 2: kappa / (sqrt(K) * interclass_cv)
    icv_arr = np.array([ds_stats[p["dataset"]]["interclass_cv"] for p in all_points])
    x_corr2 = kappas / (np.sqrt(Ks) * np.clip(icv_arr, 0.01, 10))
    try:
        popt2, _ = curve_fit(sigmoid, x_corr2, qs,
                              p0=[0.6, 10, np.median(x_corr2), 0.0], maxfev=10000)
        pred2 = sigmoid(x_corr2, *popt2)
        r2_2 = 1 - np.sum((qs - pred2) ** 2) / np.sum((qs - qs.mean()) ** 2)
        mae_2 = float(np.mean(np.abs(qs - pred2)))
        corrections["kappa/(sqrt(K)*icv)"] = {"r2": float(r2_2), "mae": mae_2}
        print(f"  kappa/(sqrt(K)*icv): R^2={r2_2:.4f}, MAE={mae_2:.4f}")
    except Exception as e:
        print(f"  kappa/(sqrt(K)*icv): FAILED ({e})")

    # Correction 3: kappa / sqrt(K) with class_overlap adjustment
    co_arr = np.array([ds_stats[p["dataset"]]["class_overlap"] for p in all_points])
    x_corr3 = kappas / np.sqrt(Ks) * (1 - co_arr)  # more overlap = harder
    try:
        popt3, _ = curve_fit(sigmoid, x_corr3, qs,
                              p0=[0.6, 10, np.median(x_corr3), 0.0], maxfev=10000)
        pred3 = sigmoid(x_corr3, *popt3)
        r2_3 = 1 - np.sum((qs - pred3) ** 2) / np.sum((qs - qs.mean()) ** 2)
        mae_3 = float(np.mean(np.abs(qs - pred3)))
        corrections["kappa*(1-overlap)/sqrt(K)"] = {"r2": float(r2_3), "mae": mae_3}
        print(f"  kappa*(1-overlap)/sqrt(K): R^2={r2_3:.4f}, MAE={mae_3:.4f}")
    except Exception as e:
        print(f"  kappa*(1-overlap)/sqrt(K): FAILED ({e})")

    # Correction 4: kappa * eta / sqrt(K) (isotropy correction)
    eta_arr = np.array([ds_stats[p["dataset"]]["eta"] for p in all_points])
    x_corr4 = kappas * eta_arr / np.sqrt(Ks)
    try:
        popt4, _ = curve_fit(sigmoid, x_corr4, qs,
                              p0=[0.6, 10, np.median(x_corr4), 0.0], maxfev=10000)
        pred4 = sigmoid(x_corr4, *popt4)
        r2_4 = 1 - np.sum((qs - pred4) ** 2) / np.sum((qs - qs.mean()) ** 2)
        mae_4 = float(np.mean(np.abs(qs - pred4)))
        corrections["kappa*eta/sqrt(K)"] = {"r2": float(r2_4), "mae": mae_4}
        print(f"  kappa*eta/sqrt(K): R^2={r2_4:.4f}, MAE={mae_4:.4f}")
    except Exception as e:
        print(f"  kappa*eta/sqrt(K): FAILED ({e})")

    # Correction 5: kappa / sqrt(K) with kurtosis correction
    kurt_arr = np.array([ds_stats[p["dataset"]]["kurtosis_excess"] for p in all_points])
    x_corr5 = kappas / np.sqrt(Ks) / (1 + np.abs(kurt_arr))
    try:
        popt5, _ = curve_fit(sigmoid, x_corr5, qs,
                              p0=[0.6, 10, np.median(x_corr5), 0.0], maxfev=10000)
        pred5 = sigmoid(x_corr5, *popt5)
        r2_5 = 1 - np.sum((qs - pred5) ** 2) / np.sum((qs - qs.mean()) ** 2)
        mae_5 = float(np.mean(np.abs(qs - pred5)))
        corrections["kappa/(sqrt(K)*(1+|kurt|))"] = {"r2": float(r2_5), "mae": mae_5}
        print(f"  kappa/(sqrt(K)*(1+|kurt|)): R^2={r2_5:.4f}, MAE={mae_5:.4f}")
    except Exception as e:
        print(f"  kappa/(sqrt(K)*(1+|kurt|)): FAILED ({e})")

    # Baseline comparison
    r2_baseline = 1 - np.sum((qs - pred_global) ** 2) / np.sum((qs - qs.mean()) ** 2)
    mae_baseline = float(np.mean(np.abs(qs - pred_global)))
    print(f"\n  BASELINE kappa/sqrt(K): R^2={r2_baseline:.4f}, MAE={mae_baseline:.4f}")

    # Best correction
    if corrections:
        best_name = max(corrections, key=lambda k: corrections[k]["r2"])
        best = corrections[best_name]
        print(f"  BEST: {best_name} (R^2={best['r2']:.4f}, MAE={best['mae']:.4f})")
        improvement = best["r2"] - r2_baseline
        print(f"  Improvement: dR^2={improvement:+.4f}")

    # Step 6: LODO test with best correction
    print(f"\n{'='*70}")
    print("LODO TEST WITH BEST CORRECTION")
    print(f"{'='*70}")

    # Test all corrections with LODO
    for corr_name in corrections:
        # Recompute the corrected x for this correction
        if corr_name == "kappa*ncr/sqrt(K)":
            x_corr = kappas / np.sqrt(Ks) * ncr_arr
        elif corr_name == "kappa/(sqrt(K)*icv)":
            x_corr = kappas / (np.sqrt(Ks) * np.clip(icv_arr, 0.01, 10))
        elif corr_name == "kappa*(1-overlap)/sqrt(K)":
            x_corr = kappas / np.sqrt(Ks) * (1 - co_arr)
        elif corr_name == "kappa*eta/sqrt(K)":
            x_corr = kappas * eta_arr / np.sqrt(Ks)
        elif corr_name == "kappa/(sqrt(K)*(1+|kurt|))":
            x_corr = kappas / np.sqrt(Ks) / (1 + np.abs(kurt_arr))
        else:
            continue

        lodo_maes = []
        for held_out in datasets_seen:
            train_mask = ds_labels != held_out
            test_mask = ds_labels == held_out

            try:
                popt_lodo, _ = curve_fit(sigmoid, x_corr[train_mask], qs[train_mask],
                                          p0=[0.6, 10, np.median(x_corr[train_mask]), 0.0],
                                          maxfev=10000)
                pred_test = sigmoid(x_corr[test_mask], *popt_lodo)
                mae_lodo = float(np.mean(np.abs(qs[test_mask] - pred_test)))
            except Exception:
                mae_lodo = 1.0

            lodo_maes.append(mae_lodo)

        mean_lodo = np.mean(lodo_maes)
        print(f"  {corr_name:>35}: LODO MAE={mean_lodo:.4f}")

    # Baseline LODO
    lodo_base = []
    for held_out in datasets_seen:
        train_mask = ds_labels != held_out
        test_mask = ds_labels == held_out
        try:
            popt_b, _ = curve_fit(sigmoid, x_norm[train_mask], qs[train_mask],
                                  p0=[0.6, 10, np.median(x_norm[train_mask]), 0.0],
                                  maxfev=10000)
            pred_b = sigmoid(x_norm[test_mask], *popt_b)
            lodo_base.append(float(np.mean(np.abs(qs[test_mask] - pred_b))))
        except Exception:
            lodo_base.append(1.0)

    print(f"  {'BASELINE kappa/sqrt(K)':>35}: LODO MAE={np.mean(lodo_base):.4f}")

    # ============================================================
    # SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    best_r2 = max(corrections.values(), key=lambda x: x["r2"])["r2"] if corrections else 0
    best_mae = min(corrections.values(), key=lambda x: x["mae"])["mae"] if corrections else 1

    checks = [
        ("Some correction beats baseline R^2",
         best_r2 > r2_baseline, f"best={best_r2:.4f} vs base={r2_baseline:.4f}"),
        ("Best correction R^2 > 0.90 (universal collapse)",
         best_r2 > 0.90, f"R^2={best_r2:.4f}"),
        ("Best correction MAE < 0.05",
         best_mae < 0.05, f"MAE={best_mae:.4f}"),
        ("At least one offset-stat has |rho| > 0.8",
         any(abs(v["rho"]) > 0.8 for v in correlations.values()),
         f"max |rho|={max(abs(v['rho']) for v in correlations.values()):.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "nongaussian_correction",
        "n_points": len(all_points),
        "n_datasets": len(datasets_seen),
        "dataset_stats": {ds: ds_stats[ds] for ds in datasets_seen},
        "offsets": offsets,
        "correlations": correlations,
        "corrections": corrections,
        "baseline": {"r2": float(r2_baseline), "mae": mae_baseline},
        "scorecard": {
            "passes": passes, "total": len(checks),
            "details": [{"criterion": c, "passed": bool(p), "value": v}
                        for c, p, v in checks],
        },
    }

    out_path = RESULTS_DIR / "cti_nongaussian_correction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
