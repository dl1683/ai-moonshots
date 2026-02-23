#!/usr/bin/env python -u
"""
Confusion Matrix Gumbel Test (Session 41)
==========================================
Pre-registered mechanistic test: does the 1-NN confusion matrix follow
the Gumbel race prediction C[ci,j] ~ exp(-kappa_ij/tau_err)?

Gumbel race prediction:
  log(C[ci,j]) = const_ci - kappa_ij / tau_err
  where C[ci,j] = fraction of class ci examples whose 1-NN is class j

If this holds, the Gumbel mechanism is the source of CTI law quantitatively.
This uses observed confusion WITHOUT interventions (no matched-delta issues).

PRE-REGISTERED CRITERIA (tau*=0.20 FIXED a priori):
  H1 (directional): pooled Pearson r(log(C), -kappa) > 0.50
  H2 (tau range): median per-class tau_err in [0.10, 0.50]
  H3 (tau* consistency): |log(tau_err / 0.20)| < log(2)  [within factor 2]
  H4 (confusion rank): pooled Spearman rho(-kappa, C) > 0.50

Dataset: pythia-160m step512, K=14 DBpedia14, N=7000
Cache: results/checkpoint_embs_pythia-160m_step512.npz
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_NPZ = RESULTS_DIR / "checkpoint_embs_pythia-160m_step512.npz"
OUT_JSON = RESULTS_DIR / "cti_confusion_gumbel_test.json"

TAU_STAR = 0.20
N_SPLITS = 5
EPSILON = 1e-6  # floor for log(C) to avoid -inf

# Pre-registered thresholds
R_DIRECTIONAL_THRESH = 0.50
RHO_RANK_THRESH = 0.50
TAU_MIN = 0.10
TAU_MAX = 0.50


def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    for c in classes:
        centroids[c] = X[y == c].mean(0)
    R = np.vstack([X[y == c] - centroids[c] for c in classes])
    sigma_W = float(np.sqrt(np.mean(R ** 2)))
    return centroids, sigma_W


def compute_kappa_matrix(centroids, sigma_W, d):
    """Return kappa[ci][cj] = dist(mu_ci, mu_cj) / (sigma_W * sqrt(d))."""
    classes = sorted(centroids.keys())
    K = len(classes)
    kappa = {}
    for ci in classes:
        kappa[ci] = {}
        for cj in classes:
            if ci == cj:
                kappa[ci][cj] = 0.0
            else:
                dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
                kappa[ci][cj] = dist / (sigma_W * np.sqrt(d) + 1e-12)
    return kappa


def compute_confusion_matrix(X, y, n_splits=N_SPLITS):
    """Compute 5-fold 1-NN confusion matrix: C[ci,cj] = P(classified as cj | true=ci)."""
    classes = sorted(np.unique(y).tolist())
    K = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    C = np.zeros((K, K))
    counts = np.zeros(K)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X_tr, y_tr)
        preds = knn.predict(X_te)

        for true_c, pred_c in zip(y_te, preds):
            i = cls_to_idx[true_c]
            j = cls_to_idx[pred_c]
            C[i, j] += 1
            counts[i] += 1

    # Normalize rows
    for i in range(K):
        if counts[i] > 0:
            C[i] /= counts[i]

    return C, classes


def fit_tau_per_class(kappas, confusions, tau_grid=None):
    """Fit tau s.t. log(C[ci,j]) = const - kappa_j/tau minimizes residuals."""
    if tau_grid is None:
        tau_grid = np.logspace(-2, 1, 100)  # 0.01 to 10

    # Filter: only use j with C[ci,j] > 0
    mask = confusions > 0
    if mask.sum() < 2:
        return None, None, None

    kappas_valid = np.array(kappas)[mask]
    log_C = np.log(np.array(confusions)[mask] + EPSILON)

    best_tau = None
    best_r2 = -np.inf
    for tau in tau_grid:
        xs = -kappas_valid / tau
        # Fit y = a*x + b
        if np.std(xs) < 1e-10:
            continue
        slope, intercept, r, p, _ = linregress(xs, log_C)
        r2 = r ** 2
        if r2 > best_r2:
            best_r2 = r2
            best_tau = tau

    # Also compute Pearson r at tau*=0.20
    xs_prereg = -kappas_valid / TAU_STAR
    if np.std(xs_prereg) > 1e-10 and np.std(log_C) > 1e-10:
        r_prereg, _ = pearsonr(xs_prereg, log_C)
    else:
        r_prereg = 0.0

    return best_tau, best_r2, r_prereg


def json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    print("=" * 70)
    print("CONFUSION MATRIX GUMBEL TEST")
    print(f"tau*={TAU_STAR} FIXED, H1: pooled r > {R_DIRECTIONAL_THRESH}")
    print("=" * 70)

    if not CACHE_NPZ.exists():
        raise FileNotFoundError(f"Cache not found: {CACHE_NPZ}")

    data = np.load(str(CACHE_NPZ))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    K = len(classes)
    print(f"Data: N={len(X)}, d={d}, K={K}")

    # Compute geometry
    centroids, sigma_W = compute_class_stats(X, y)
    kappa = compute_kappa_matrix(centroids, sigma_W, d)
    print(f"sigma_W={sigma_W:.4f}")

    # Compute confusion matrix
    print("\nComputing 5-fold confusion matrix...")
    C, classes_list = compute_confusion_matrix(X, y)

    print("\nDiagonal (per-class accuracy):")
    for i, ci in enumerate(classes_list):
        print(f"  ci={ci}: C[{ci},{ci}]={C[i,i]:.4f} (kappa_j1={min(kappa[ci][cj] for cj in classes if cj != ci):.4f})")

    # For each ci: collect (kappa_ij, C[ci,j]) for j != ci
    per_class_results = []
    all_kappas = []
    all_log_C = []
    all_C = []

    for i, ci in enumerate(classes_list):
        kappas_ci = []
        confusions_ci = []
        for j, cj in enumerate(classes_list):
            if ci == cj:
                continue
            kappas_ci.append(kappa[ci][cj])
            confusions_ci.append(C[i, j])

        kappas_ci = np.array(kappas_ci)
        confusions_ci = np.array(confusions_ci)

        # Per-class tau fit
        best_tau, best_r2, r_prereg = fit_tau_per_class(kappas_ci, confusions_ci)

        # Pearson r(-kappa, C) directional test
        if np.std(kappas_ci) > 1e-10 and np.std(confusions_ci) > 1e-10:
            r_dir, p_dir = pearsonr(-kappas_ci, confusions_ci)
        else:
            r_dir, p_dir = 0.0, 1.0

        # Spearman rho(-kappa, C)
        if np.std(kappas_ci) > 1e-10:
            rho_dir, p_rho = spearmanr(-kappas_ci, confusions_ci)
        else:
            rho_dir, p_rho = 0.0, 1.0

        per_class_results.append({
            "ci": int(ci),
            "kappas": kappas_ci.tolist(),
            "confusions": confusions_ci.tolist(),
            "tau_fit": float(best_tau) if best_tau else None,
            "tau_r2": float(best_r2) if best_r2 is not None else None,
            "r_prereg": float(r_prereg) if r_prereg is not None else None,
            "r_dir_pearson": float(r_dir),
            "p_dir": float(p_dir),
            "rho_spearman": float(rho_dir),
            "p_rho": float(p_rho),
        })

        tau_str = f"{best_tau:.3f}" if best_tau is not None else "N/A"
        rp_str = f"{r_prereg:.3f}" if r_prereg is not None else "N/A"
        print(f"  ci={ci}: tau_fit={tau_str}, r_dir={r_dir:.3f}, rho={rho_dir:.3f}, r_prereg={rp_str}")

        # Accumulate pooled data (only non-zero confusions)
        for kj, Cij in zip(kappas_ci, confusions_ci):
            if Cij > 0:
                all_kappas.append(kj)
                all_log_C.append(np.log(Cij + EPSILON))
                all_C.append(Cij)

    # Pooled tests
    all_kappas = np.array(all_kappas)
    all_log_C = np.array(all_log_C)
    all_C = np.array(all_C)

    print(f"\nPooled non-zero C pairs: {len(all_kappas)}")

    # H1: pooled r(log(C), -kappa) > 0.50
    if len(all_kappas) >= 4 and np.std(all_kappas) > 1e-10 and np.std(all_log_C) > 1e-10:
        r_pooled, p_pooled = pearsonr(-all_kappas, all_log_C)
    else:
        r_pooled, p_pooled = 0.0, 1.0

    # H4: pooled Spearman rho(-kappa, C) > 0.50
    if len(all_kappas) >= 4:
        rho_pooled, p_rho_pooled = spearmanr(-all_kappas, all_C)
    else:
        rho_pooled, p_rho_pooled = 0.0, 1.0

    # Fit pooled tau_err
    best_tau_pooled, best_r2_pooled, r_pooled_prereg = fit_tau_per_class(all_kappas, all_C)

    # H2, H3: tau range
    tau_fits = [r["tau_fit"] for r in per_class_results if r["tau_fit"] is not None]
    tau_median = float(np.median(tau_fits)) if tau_fits else None
    tau_in_range = bool(TAU_MIN <= tau_median <= TAU_MAX) if tau_median else False
    tau_consistent = bool(abs(np.log(tau_median / TAU_STAR)) < np.log(2)) if tau_median else False

    pass_H1 = bool(r_pooled > R_DIRECTIONAL_THRESH)
    pass_H4 = bool(rho_pooled > RHO_RANK_THRESH)
    pass_H2 = tau_in_range
    pass_H3 = tau_consistent
    pass_prereg = bool(pass_H1 and pass_H4 and pass_H2 and pass_H3)

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)
    print(f"H1 Pooled r(log(C), -kappa) = {r_pooled:.4f} (thresh {R_DIRECTIONAL_THRESH}): {'PASS' if pass_H1 else 'FAIL'}")
    print(f"H4 Pooled rho(-kappa, C)    = {rho_pooled:.4f} (thresh {RHO_RANK_THRESH}): {'PASS' if pass_H4 else 'FAIL'}")
    tm_str = f"{tau_median:.3f}" if tau_median is not None else "N/A"
    ratio_str = f"{TAU_STAR/tau_median:.2f}" if tau_median else "N/A"
    btp_str = f"{best_tau_pooled:.3f}" if best_tau_pooled is not None else "N/A"
    br2_str = f"{best_r2_pooled:.3f}" if best_r2_pooled is not None else "N/A"
    print(f"H2 Median tau_err           = {tm_str} (range [{TAU_MIN},{TAU_MAX}]): {'PASS' if pass_H2 else 'FAIL'}")
    print(f"H3 tau*/tau_err ratio       = {ratio_str} (factor-2 OK): {'PASS' if pass_H3 else 'FAIL'}")
    print(f"Pooled tau_fit              = {btp_str} (R2={br2_str})")
    print(f"OVERALL pre-reg pass        = {'PASS' if pass_prereg else 'FAIL'}")

    output = {
        "experiment": "confusion_gumbel_test",
        "session": 41,
        "tau_star_prereg": TAU_STAR,
        "preregistered": {
            "H1_threshold": R_DIRECTIONAL_THRESH,
            "H4_threshold": RHO_RANK_THRESH,
            "H2_tau_range": [TAU_MIN, TAU_MAX],
            "H3_tau_factor": 2.0,
        },
        "pooled": {
            "n_pairs": len(all_kappas),
            "r_pearson_log_C": float(r_pooled),
            "p_pearson": float(p_pooled),
            "rho_spearman_C": float(rho_pooled),
            "p_spearman": float(p_rho_pooled),
            "tau_fit_pooled": float(best_tau_pooled) if best_tau_pooled else None,
            "tau_r2_pooled": float(best_r2_pooled) if best_r2_pooled is not None else None,
        },
        "per_class_tau": {
            "median": float(tau_median) if tau_median else None,
            "mean": float(np.mean(tau_fits)) if tau_fits else None,
            "std": float(np.std(tau_fits)) if tau_fits else None,
            "values": [float(t) for t in tau_fits],
        },
        "pass_H1": pass_H1,
        "pass_H2": pass_H2,
        "pass_H3": pass_H3,
        "pass_H4": pass_H4,
        "pass_prereg": pass_prereg,
        "per_class": per_class_results,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
