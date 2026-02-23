#!/usr/bin/env python -u
"""
Jacobian Delta-Window Sweep: Artifact Disambiguation (Session 40)
=================================================================
Codex priority: run Jacobian test across 4 DELTA_OUT windows to test whether
the pre-registered FAIL (r=0.756) is a saturation artifact or a real theory failure.

PREDICTION (artifact hypothesis): as window shrinks, intercept c weakens and r improves.
PREDICTION (theory-failure hypothesis): mismatch persists even in near-zero window.

Pre-registered (from session 40 original run):
  tau* = 0.20 FIXED
  r_threshold = 0.85
  Window: 0..5 --> r=0.756, FAIL, tau_fit=0.46, c?-2.9

Secondary: fit log_w = c - gap/tau (intercept model) for each window.
If c?0 as window shrinks, the intercept is a saturation artifact.
If c stays large, it is a genuine floor effect independent of window.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, linregress
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_NPZ = RESULTS_DIR / "checkpoint_embs_pythia-160m_step512.npz"
OUT_JSON = RESULTS_DIR / "cti_jacobian_delta_sweep.json"

TAU_STAR = 0.20
R_THRESHOLD = 0.85
N_SPLITS_CV = 5

DELTA_WINDOWS = {
    "local_0.05":  np.linspace(0.0, 0.05, 11),
    "local_0.1":   np.linspace(0.0, 0.10, 11),
    "local_0.3":   np.linspace(0.0, 0.30, 11),
    "local_1.0":   np.linspace(0.0, 1.00, 11),
    "prereg_5.0":  np.linspace(0.0, 5.00, 11),
}


def compute_class_stats(X, y):
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(0)
        centroids[c] = mu
        resids.append(Xc - mu)
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R**2)))
    return centroids, sigma_W


def compute_all_kappas_sorted(centroids, sigma_W, d, ci):
    mu_i = centroids[ci]
    kappas = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        k = dist / (sigma_W * np.sqrt(d) + 1e-10)
        kappas.append((k, cj))
    kappas.sort(key=lambda x: x[0])
    return kappas


def compute_per_class_q(X, y, ci):
    K_local = len(np.unique(y))
    skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
    recalls = []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if (y_tr == ci).sum() < 1:
            continue
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        knn.fit(X_tr, y_tr)
        mask = y_te == ci
        if mask.sum() == 0:
            continue
        preds = knn.predict(X_te[mask])
        recalls.append(float((preds == ci).mean()))
    if not recalls:
        return None
    q_raw = float(np.mean(recalls))
    return float((q_raw - 1.0 / K_local) / (1.0 - 1.0 / K_local))


def safe_logit(q):
    q = float(np.clip(q, 1e-5, 1 - 1e-5))
    return float(np.log(q / (1.0 - q)))


def apply_competitor_shift(X, y, centroids, ci, cj, delta):
    mu_i, mu_j = centroids[ci], centroids[cj]
    diff = mu_j - mu_i
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        return X.copy()
    direction = diff / dist
    X_new = X.copy()
    X_new[y == cj] += delta * direction
    return X_new


def fit_slope_zero_intercept(deltas, logit_qs):
    """Fit slope assuming zero intercept."""
    if len(deltas) < 4 or np.std(deltas) < 1e-8 or np.std(logit_qs) < 1e-8:
        return 0.0, 0.0, 1.0
    r, p = pearsonr(deltas, logit_qs)
    slope, _, _, _, _ = linregress(deltas, logit_qs)
    return float(slope), float(r), float(p)


def fit_slope_with_intercept(xs, ys):
    """Fit y = a*x + b, return slope, intercept."""
    if len(xs) < 4 or np.std(xs) < 1e-8:
        return 0.0, float(np.mean(ys))
    slope, intercept, _, _, _ = linregress(xs, ys)
    return float(slope), float(intercept)


def run_jacobian_window(X, y, d, centroids, sigma_W, per_class_geom, delta_arr, window_name):
    """Run Jacobian test for a given DELTA_OUT window."""
    print(f"\n  Window: {window_name} (max_delta={delta_arr[-1]:.3f})")
    jacobian_results = []
    for ci in sorted(per_class_geom.keys()):
        geom = per_class_geom[ci]
        j1, j2 = geom["j1"], geom["j2"]
        kappa_j1, kappa_j2, gap = geom["kappa_j1"], geom["kappa_j2"], geom["gap"]
        blq = geom["logit_q"]

        # alpha_j1
        dk_j1, dlq_j1 = [], []
        for delta in delta_arr[1:]:
            X_mod = apply_competitor_shift(X, y, centroids, ci, j1, delta)
            c_mod, s_mod = compute_class_stats(X_mod, y)
            kp = compute_all_kappas_sorted(c_mod, s_mod, d, ci)
            if not kp:
                continue
            k1_mod = kp[0][0]
            q_mod = compute_per_class_q(X_mod, y, ci)
            if q_mod is None:
                continue
            dk_j1.append(k1_mod - kappa_j1)
            dlq_j1.append(safe_logit(q_mod) - blq)

        alpha_j1, r_j1, _ = fit_slope_zero_intercept(dk_j1, dlq_j1)

        # alpha_j2
        dk_j2, dlq_j2 = [], []
        for delta in delta_arr[1:]:
            X_mod = apply_competitor_shift(X, y, centroids, ci, j2, delta)
            c_mod, s_mod = compute_class_stats(X_mod, y)
            kp = compute_all_kappas_sorted(c_mod, s_mod, d, ci)
            if not kp:
                continue
            j2_ks = [k for k, c in kp if c == j2]
            if not j2_ks:
                continue
            k2_mod = j2_ks[0]
            q_mod = compute_per_class_q(X_mod, y, ci)
            if q_mod is None:
                continue
            dk_j2.append(k2_mod - kappa_j2)
            dlq_j2.append(safe_logit(q_mod) - blq)

        alpha_j2, r_j2, _ = fit_slope_zero_intercept(dk_j2, dlq_j2)

        if alpha_j1 > 0.01 and alpha_j2 is not None:
            w_j2 = alpha_j2 / alpha_j1
            log_w = float(np.log(max(w_j2, 1e-6)))
            jacobian_results.append({
                "ci": int(ci),
                "gap": float(gap),
                "alpha_j1": float(alpha_j1),
                "alpha_j2": float(alpha_j2),
                "r_j1": float(r_j1),
                "r_j2": float(r_j2),
                "w_j2": float(w_j2),
                "log_w_j2": log_w,
                "theory_log_w": float(-gap / TAU_STAR),
            })

    if len(jacobian_results) < 4:
        print(f"    Too few valid: {len(jacobian_results)}")
        return {
            "window": window_name,
            "n_valid": len(jacobian_results),
            "error": "too_few",
        }

    log_ws = np.array([r["log_w_j2"] for r in jacobian_results])
    gaps = np.array([r["gap"] for r in jacobian_results])
    theory_log_ws = -gaps / TAU_STAR

    r_pearson, p_pearson = pearsonr(log_ws, theory_log_ws)
    # Zero-intercept fit: log_w = -gap/tau
    slope_zero, intercept_zero = fit_slope_with_intercept(-gaps, log_ws)
    tau_fit_zero = 1.0 / slope_zero if abs(slope_zero) > 1e-8 else None
    # Intercept model: log_w = c - gap/tau_int
    slope_int, intercept_int = fit_slope_with_intercept(-gaps, log_ws)
    tau_fit_int = 1.0 / slope_int if abs(slope_int) > 1e-8 else None

    pass_prereg = bool(r_pearson > R_THRESHOLD)
    print(f"    n_valid={len(jacobian_results)}, r={r_pearson:.4f}, pass={pass_prereg}, tau_fit={tau_fit_zero:.3f}, intercept={intercept_int:.3f}")

    return {
        "window": window_name,
        "max_delta": float(delta_arr[-1]),
        "n_valid": len(jacobian_results),
        "r_pearson": float(r_pearson),
        "p_pearson": float(p_pearson),
        "tau_fit": float(tau_fit_zero) if tau_fit_zero else None,
        "intercept_c": float(intercept_int),
        "pass_prereg": pass_prereg,
        "per_class": jacobian_results,
    }


def json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    print("=" * 70)
    print("JACOBIAN DELTA-WINDOW SWEEP: ARTIFACT DISAMBIGUATION")
    print(f"tau*={TAU_STAR} FIXED, r_threshold={R_THRESHOLD}")
    print("Hypothesis: artifact -> c->0 as window shrinks; failure -> c stays large")
    print("=" * 70)

    # Load cached embeddings
    if not CACHE_NPZ.exists():
        raise FileNotFoundError(f"Cache not found at {CACHE_NPZ}. Run cti_jacobian_early_checkpoint.py first.")

    print(f"Loading cached embeddings from {CACHE_NPZ}")
    data = np.load(str(CACHE_NPZ))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    K = len(classes)
    print(f"Data: N={len(X)}, d={d}, K={K}")

    centroids, sigma_W = compute_class_stats(X, y)

    # Compute baseline geometry
    print("\nBaseline geometry:")
    per_class_geom = {}
    for ci in classes:
        kappas_list = compute_all_kappas_sorted(centroids, sigma_W, d, ci)
        if len(kappas_list) < 2:
            continue
        kappa_j1, j1 = kappas_list[0]
        kappa_j2, j2 = kappas_list[1]
        gap = kappa_j2 - kappa_j1
        q = compute_per_class_q(X, y, ci)
        if q is None:
            continue
        per_class_geom[ci] = {
            "j1": j1, "j2": j2,
            "kappa_j1": kappa_j1, "kappa_j2": kappa_j2,
            "gap": gap, "logit_q": safe_logit(q),
        }
        print(f"  ci={ci}: kappa_j1={kappa_j1:.4f}, kappa_j2={kappa_j2:.4f}, gap={gap:.4f}, q={q:.4f}")

    # Run sweep
    print("\n" + "=" * 70)
    print("SWEEPING WINDOWS")
    print("=" * 70)

    sweep_results = []
    for window_name, delta_arr in DELTA_WINDOWS.items():
        res = run_jacobian_window(X, y, d, centroids, sigma_W, per_class_geom, delta_arr, window_name)
        sweep_results.append(res)

    # Print summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Window':<15} {'max_delta':>9} {'n_valid':>7} {'r_pearson':>10} {'tau_fit':>8} {'intercept_c':>12} {'pass':>6}")
    for res in sweep_results:
        if "error" in res:
            print(f"{res['window']:<15} {'N/A':>9} {res['n_valid']:>7} {'N/A':>10} {'N/A':>8} {'N/A':>12} {'N/A':>6}")
        else:
            p = "YES" if res["pass_prereg"] else "NO"
            print(f"{res['window']:<15} {res['max_delta']:>9.3f} {res['n_valid']:>7} {res['r_pearson']:>10.4f} {res['tau_fit']:>8.3f} {res['intercept_c']:>12.3f} {p:>6}")

    output = {
        "experiment": "jacobian_delta_sweep",
        "session": 40,
        "tau_star_prereg": TAU_STAR,
        "r_threshold_prereg": R_THRESHOLD,
        "artifact_hypothesis": "if intercept_c -> 0 as window shrinks, artifact confirmed",
        "theory_failure_hypothesis": "if intercept_c stays large across all windows, genuine floor",
        "sweep": sweep_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
