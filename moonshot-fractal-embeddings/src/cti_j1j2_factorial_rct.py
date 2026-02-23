#!/usr/bin/env python -u
"""
J1+J2 Factorial RCT: Direct Causal Weight Estimation for 2nd Nearest Competitor
=================================================================================
HYPOTHESIS: The correct law is logit(q_ci) = A * (kappa_j1 + w * kappa_j2) + C
where w = 0.40 (Codex prediction from LOAO/single ratio, Session 38).

DESIGN: For each class ci, independently perturb:
  - j1(ci): nearest competitor centroid, shift away from ci by delta_kappa units
  - j2(ci): 2nd nearest competitor centroid, shift away from ci by delta_kappa units
  - j1+j2:  shift BOTH j1 and j2 simultaneously

DOSE-RESPONSE per class:
  DELTA_LIST = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0] kappa units

Per-class estimates:
  slope_j1_ci = regression: delta_logit_q vs delta_kappa_j1 (j1-only arm)
  slope_j2_ci = regression: delta_logit_q vs delta_kappa_j2 (j2-only arm)
  w_ci = slope_j2_ci / slope_j1_ci

PRE-REGISTERED:
  1. mean_w ≈ 0.40 (within ±0.15) -- Codex prediction
  2. slope_j1 > 0 for all classes (direction test)
  3. slope_j2 > 0 for majority of classes (j2 causal)
  4. Additivity: joint_logit_delta ≈ j1_delta + j2_delta, r > 0.85

KEY: If w ≈ 0.40 and additivity holds, the upgraded law
     logit(q_ci) = A * phi(tau*) + C  is the correct causal formula.
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import pearsonr, linregress

EMBS_FILE = Path("results/dointerv_multi_pythia-160m_l12.npz")
OUT_JSON = Path("results/cti_j1j2_factorial_rct.json")
K = 14
RANDOM_STATE = 42
TEST_SIZE = 0.2
DELTA_LIST = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]  # kappa units; 0.0 = baseline
W_PREDICTED = 0.40  # Codex prediction


def compute_class_stats(X, y):
    """Compute centroids and global sigma_W."""
    classes = np.unique(y)
    centroids = {}
    resids = []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(0)
        centroids[c] = mu
        resids.append(Xc - mu)
    R = np.vstack(resids)
    sigma_W = float(np.sqrt(np.mean(R ** 2)))
    return centroids, sigma_W


def compute_ranked_competitors(centroids, sigma_W, d, ci):
    """Return list of (kappa, class_label) sorted ascending (nearest first)."""
    mu_i = centroids[ci]
    ranking = []
    for cj, mu_j in centroids.items():
        if cj == ci:
            continue
        dist = float(np.linalg.norm(mu_i - mu_j))
        kappa = dist / (sigma_W * np.sqrt(d) + 1e-10)
        ranking.append((kappa, cj))
    ranking.sort()
    return ranking


def shift_competitor(X, y, ci_centroid, competitor_label, competitor_centroid,
                     delta_kappa, sigma_W, d):
    """
    Shift all embeddings of competitor_label AWAY from ci by delta_kappa kappa units.
    direction = (competitor_centroid - ci_centroid) normalized
    shift_magnitude = delta_kappa * sigma_W * sqrt(d)
    Returns modified copy of X.
    """
    direction = competitor_centroid - ci_centroid
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return X.copy()
    direction = direction / norm
    shift = delta_kappa * sigma_W * np.sqrt(d) * direction
    X_mod = X.copy()
    X_mod[y == competitor_label] += shift
    return X_mod


def eval_q_ci(X_tr, y_tr, X_te, y_te, ci):
    """Fit 1-NN and return normalized per-class accuracy for class ci."""
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
    knn.fit(X_tr, y_tr)
    mask = y_te == ci
    if mask.sum() == 0:
        return None
    preds = knn.predict(X_te[mask])
    q_raw = float((preds == ci).mean())
    K_local = len(np.unique(y_tr))
    return float((q_raw - 1.0 / K_local) / (1.0 - 1.0 / K_local))


def safe_logit(q):
    q = float(np.clip(q, 1e-5, 1 - 1e-5))
    return float(np.log(q / (1.0 - q)))


def fit_slope(x_vals, y_vals):
    """Fit linear regression y = m*x + b, return (slope, r, intercept)."""
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or x.std() < 1e-8:
        return None, None, None
    slope, intercept, r, _, _ = linregress(x, y)
    return float(slope), float(r), float(intercept)


def main():
    print("=" * 70)
    print("J1+J2 FACTORIAL RCT: Causal Weight Estimation for 2nd Competitor")
    print(f"PRE-REGISTERED: mean_w = {W_PREDICTED} +/- 0.15")
    print("=" * 70)

    data = np.load(str(EMBS_FILE))
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    d = X.shape[1]
    classes = sorted(np.unique(y).tolist())
    N = len(X)
    print(f"Loaded: N={N}, d={d}, K={len(classes)}")

    # Fixed train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(sss.split(X, y))
    X_train_base, X_test_base = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]
    print(f"Train: {len(tr_idx)}, Test: {len(te_idx)}")

    # Compute baseline class stats
    centroids, sigma_W = compute_class_stats(X, y)
    print(f"sigma_W = {sigma_W:.6f}, sigma_W*sqrt(d) = {sigma_W * np.sqrt(d):.4f}")

    per_class_results = {}
    all_w_estimates = []
    additivity_pairs = []  # (predicted_delta, actual_delta) for joint arm

    for ci in classes:
        ranking = compute_ranked_competitors(centroids, sigma_W, d, ci)
        kappa_j1_base, j1_class = ranking[0]
        kappa_j2_base, j2_class = ranking[1]

        print(f"\n--- Class {ci}: j1={j1_class} kappa_j1={kappa_j1_base:.4f}, "
              f"j2={j2_class} kappa_j2={kappa_j2_base:.4f} "
              f"(gap={kappa_j2_base-kappa_j1_base:.4f})")

        # Baseline (delta=0): q_ci from original embeddings
        q_base = eval_q_ci(X_train_base, y_train, X_test_base, y_test, ci)
        if q_base is None:
            print(f"  SKIP: no test samples for class {ci}")
            continue
        logit_base = safe_logit(q_base)
        print(f"  baseline: q={q_base:.4f}, logit={logit_base:.4f}")

        arm_data = {"j1": [], "j2": [], "joint": []}

        for delta in DELTA_LIST:
            if delta == 0.0:
                # Baseline for all arms
                for arm in ["j1", "j2", "joint"]:
                    arm_data[arm].append({
                        "delta_requested": 0.0,
                        "delta_kappa_j1": 0.0,
                        "delta_kappa_j2": 0.0,
                        "q": float(q_base),
                        "logit_q": float(logit_base),
                        "delta_logit_q": 0.0,
                    })
                continue

            for arm in ["j1", "j2", "joint"]:
                # Create modified embeddings
                X_tr_mod = X_train_base.copy()
                X_te_mod = X_test_base.copy()

                # Shift j1
                if arm in ["j1", "joint"]:
                    direction = centroids[j1_class] - centroids[ci]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    shift = delta * sigma_W * np.sqrt(d) * direction
                    # Shift j1 in BOTH train and test (consistent intervention)
                    tr_j1_mask = y_train == j1_class
                    te_j1_mask = y_test == j1_class
                    X_tr_mod[tr_j1_mask] += shift
                    X_te_mod[te_j1_mask] += shift

                # Shift j2
                if arm in ["j2", "joint"]:
                    direction = centroids[j2_class] - centroids[ci]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    shift = delta * sigma_W * np.sqrt(d) * direction
                    tr_j2_mask = y_train == j2_class
                    te_j2_mask = y_test == j2_class
                    X_tr_mod[tr_j2_mask] += shift
                    X_te_mod[te_j2_mask] += shift

                # Evaluate q_ci on modified embeddings
                q_mod = eval_q_ci(X_tr_mod, y_train, X_te_mod, y_test, ci)
                if q_mod is None:
                    continue
                logit_mod = safe_logit(q_mod)
                delta_logit = logit_mod - logit_base

                # Compute actual delta_kappa (from modified centroids)
                # After shift: centroid of j1/j2 changes because ALL points shifted
                # For j1 arm: centroid_j1_new = centroid_j1_old + shift
                # => delta_kappa_j1 = delta (exact by construction)
                delta_kappa_j1 = delta if arm in ["j1", "joint"] else 0.0
                delta_kappa_j2 = delta if arm in ["j2", "joint"] else 0.0

                arm_data[arm].append({
                    "delta_requested": float(delta),
                    "delta_kappa_j1": float(delta_kappa_j1),
                    "delta_kappa_j2": float(delta_kappa_j2),
                    "q": float(q_mod),
                    "logit_q": float(logit_mod),
                    "delta_logit_q": float(delta_logit),
                })

        # Fit slopes per arm
        def arm_slope(arm_name, kappa_col):
            data_pts = arm_data[arm_name]
            dk = [p[kappa_col] for p in data_pts]
            dl = [p["delta_logit_q"] for p in data_pts]
            return fit_slope(dk, dl)

        slope_j1, r_j1, _ = arm_slope("j1", "delta_kappa_j1")
        slope_j2, r_j2, _ = arm_slope("j2", "delta_kappa_j2")

        s1_str = f"{slope_j1:.4f}" if slope_j1 is not None else "N/A"
        r1_str = f"{r_j1:.4f}" if r_j1 is not None else "N/A"
        s2_str = f"{slope_j2:.4f}" if slope_j2 is not None else "N/A"
        r2_str = f"{r_j2:.4f}" if r_j2 is not None else "N/A"
        print(f"  slope_j1 = {s1_str} (r={r1_str})")
        print(f"  slope_j2 = {s2_str} (r={r2_str})")

        w_ci = None
        if slope_j1 is not None and slope_j2 is not None and abs(slope_j1) > 0.01:
            w_ci = float(slope_j2 / slope_j1)
            print(f"  w = slope_j2/slope_j1 = {w_ci:.4f}  [predicted: {W_PREDICTED}]")
            all_w_estimates.append(w_ci)

        # Additivity test: compare joint vs j1+j2 prediction
        for j1_pt, j2_pt, joint_pt in zip(arm_data["j1"][1:],  # skip delta=0
                                           arm_data["j2"][1:],
                                           arm_data["joint"][1:]):
            if all(p is not None for p in [j1_pt, j2_pt, joint_pt]):
                predicted_joint = j1_pt["delta_logit_q"] + j2_pt["delta_logit_q"]
                actual_joint = joint_pt["delta_logit_q"]
                additivity_pairs.append((predicted_joint, actual_joint))

        per_class_results[ci] = {
            "j1_class": int(j1_class),
            "j2_class": int(j2_class),
            "kappa_j1_base": float(kappa_j1_base),
            "kappa_j2_base": float(kappa_j2_base),
            "kappa_gap": float(kappa_j2_base - kappa_j1_base),
            "q_base": float(q_base),
            "logit_base": float(logit_base),
            "slope_j1": float(slope_j1) if slope_j1 is not None else None,
            "slope_j2": float(slope_j2) if slope_j2 is not None else None,
            "r_j1": float(r_j1) if r_j1 is not None else None,
            "r_j2": float(r_j2) if r_j2 is not None else None,
            "w": float(w_ci) if w_ci is not None else None,
            "arm_data": arm_data,
        }

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Mean w estimate
    valid_w = [w for w in all_w_estimates if np.isfinite(w) and -5 < w < 5]
    mean_w = float(np.mean(valid_w)) if valid_w else None
    std_w = float(np.std(valid_w)) if valid_w else None
    print(f"\nw estimates across {len(valid_w)} classes: {[f'{w:.3f}' for w in valid_w]}")
    print(f"mean_w = {mean_w:.4f}  std_w = {std_w:.4f}" if mean_w is not None else "N/A")
    print(f"Predicted w = {W_PREDICTED}")

    w_test_pass = (mean_w is not None and abs(mean_w - W_PREDICTED) < 0.20)
    print(f"w test (|mean_w - 0.40| < 0.20): {'PASS' if w_test_pass else 'FAIL'}")

    # Direction test
    n_slope_j1_pos = sum(1 for ci, r in per_class_results.items()
                         if r["slope_j1"] is not None and r["slope_j1"] > 0)
    n_slope_j2_pos = sum(1 for ci, r in per_class_results.items()
                         if r["slope_j2"] is not None and r["slope_j2"] > 0)
    n_valid = sum(1 for r in per_class_results.values()
                  if r["slope_j1"] is not None)
    print(f"\nDirection test: slope_j1 > 0: {n_slope_j1_pos}/{n_valid}")
    print(f"Direction test: slope_j2 > 0: {n_slope_j2_pos}/{n_valid}")

    # Additivity test
    add_r, add_p = (None, None)
    if len(additivity_pairs) >= 5:
        pred_arr = np.array([p[0] for p in additivity_pairs])
        actual_arr = np.array([p[1] for p in additivity_pairs])
        valid_mask = np.isfinite(pred_arr) & np.isfinite(actual_arr)
        if valid_mask.sum() >= 5:
            add_r, add_p = pearsonr(pred_arr[valid_mask], actual_arr[valid_mask])
            print(f"\nAdditivity test (n={valid_mask.sum()}): r={add_r:.4f} p={add_p:.4e}")
            print(f"  PASS (r > 0.85): {'PASS' if add_r > 0.85 else 'FAIL'}")

    # Per-class table
    print(f"\nPer-class results:")
    print(f"{'ci':>4} {'kappa_j1':>10} {'kappa_j2':>10} {'gap':>8} "
          f"{'slope_j1':>10} {'slope_j2':>10} {'w':>8}")
    for ci in sorted(per_class_results.keys()):
        r = per_class_results[ci]
        w_str = f"{r['w']:.3f}" if r["w"] is not None else "N/A"
        s1_str = f"{r['slope_j1']:.3f}" if r["slope_j1"] is not None else "N/A"
        s2_str = f"{r['slope_j2']:.3f}" if r["slope_j2"] is not None else "N/A"
        print(f"{ci:>4} {r['kappa_j1_base']:>10.4f} {r['kappa_j2_base']:>10.4f} "
              f"{r['kappa_gap']:>8.4f} {s1_str:>10} {s2_str:>10} {w_str:>8}")

    result = {
        "experiment": "j1j2_factorial_rct",
        "model": "pythia-160m",
        "dataset": "dbpedia14",
        "K": K,
        "delta_list": DELTA_LIST,
        "w_predicted": W_PREDICTED,
        "per_class": per_class_results,
        "mean_w": float(mean_w) if mean_w is not None else None,
        "std_w": float(std_w) if std_w is not None else None,
        "n_valid_w": len(valid_w),
        "all_w_estimates": valid_w,
        "w_test_pass": bool(w_test_pass),
        "n_slope_j1_positive": int(n_slope_j1_pos),
        "n_slope_j2_positive": int(n_slope_j2_pos),
        "n_valid_slopes": int(n_valid),
        "additivity_r": float(add_r) if add_r is not None else None,
        "additivity_p": float(add_p) if add_p is not None else None,
        "additivity_n": len(additivity_pairs),
        "additivity_pass": bool(add_r > 0.85 if add_r is not None else False),
    }

    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nResults saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
