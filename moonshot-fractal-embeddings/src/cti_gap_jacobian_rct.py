#!/usr/bin/env python -u
"""
Gap-Conditioned Jacobian RCT: Phi Model Mechanistic Test
=========================================================
HYPOTHESIS: The local Jacobian of logit(q) with respect to kappa_j
follows the phi model prediction:
    d(logit_q) / d(kappa_j) = A * softmax(-kappa_j / tau*)_j

Ratio (w = j2/j1 slope):
    w_local = exp(-(kappa_j2 - kappa_j1) / tau*)

PRE-REGISTERED:
  1. r(log(w_empirical), -gap) > 0.60 across (arch, class) pairs
     [phi functional form: log(w) = -gap/tau*]
  2. tau*_empirical (from regression) is within [0.05, 0.50]
     [physiclally reasonable tau range]
  3. 5 architectures x 14 classes = 70 (arch, class) pairs

DESIGN:
- 5 architectures (same as phi_upgrade_pooled)
- For each (arch, class) pair: compute kappa_j1, kappa_j2, gap, q_base
- USE ONLY stable-rank pairs: delta < gap / 2 (j1/j2 don't swap during perturbation)
- Small dose-response: delta in [0.02, 0.04, 0.06, 0.08, 0.10] kappa units
- slope_j1 = regression of delta_logit_q vs delta_kappa_j1 (j1-only arm)
- slope_j2 = regression of delta_logit_q vs delta_kappa_j2 (j2-only arm)
- w_empirical = slope_j2 / slope_j1

GAP-CONDITIONED REGRESSION:
  log(w_empirical) = -gap / tau_star + epsilon
  tau_star = -gap / log(w_empirical) per pair
  Test: r(log(w), -gap) > 0.60 (phi is functionally correct)
  Test: median(tau_star) in [0.05, 0.50] (tau calibration)
  Test: tau_median ≈ 0.2 (best tau from phi_upgrade_pooled)
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import pearsonr, linregress, spearmanr

EMBS = {
    "pythia-160m": "results/dointerv_multi_pythia-160m_l12.npz",
    "pythia-410m": "results/dointerv_multi_pythia-410m_l3.npz",
    "electra-small": "results/dointerv_multi_electra-small_l3.npz",
    "rwkv-4-169m": "results/dointerv_multi_rwkv-4-169m_l12.npz",
    "bert-base": "results/dointerv_multi_bert-base-uncased_l10.npz",
}
OUT_JSON = Path("results/cti_gap_jacobian_rct.json")
K = 14
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Dose-response: small deltas only for stable-rank regime
DELTA_LIST_SMALL = [0.02, 0.04, 0.06, 0.08, 0.10]
TAU_STAR_POOLED = 0.2  # from phi_upgrade_pooled


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
    sigma_W = float(np.sqrt(np.mean(R ** 2)))
    return centroids, sigma_W


def compute_ranked_competitors(centroids, sigma_W, d, ci):
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


def shift_competitor_train_test(X_tr, X_te, y_tr, y_te,
                                 ci_centroid, competitor_label, competitor_centroid,
                                 delta_kappa, sigma_W, d):
    direction = competitor_centroid - ci_centroid
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return X_tr.copy(), X_te.copy()
    direction = direction / norm
    shift = delta_kappa * sigma_W * np.sqrt(d) * direction
    X_tr_mod = X_tr.copy()
    X_te_mod = X_te.copy()
    X_tr_mod[y_tr == competitor_label] += shift
    X_te_mod[y_te == competitor_label] += shift
    return X_tr_mod, X_te_mod


def eval_q_ci(X_tr, y_tr, X_te, y_te, ci):
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
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or x.std() < 1e-8:
        return None, None
    slope, _, r, _, _ = linregress(x, y)
    return float(slope), float(r)


def main():
    print("=" * 70)
    print("GAP-CONDITIONED JACOBIAN RCT: Phi Model Mechanistic Test")
    print(f"PRE-REG: r(log(w), -gap) > 0.60, tau_star in [0.05, 0.50]")
    print("=" * 70)

    all_records = []  # (arch, class, gap, w_empirical, w_phi, tau_star_est, ...)

    for model_name, emb_path in EMBS.items():
        path = Path(emb_path)
        if not path.exists():
            print(f"  MISSING: {emb_path}")
            continue
        data = np.load(str(path))
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int64)
        d = X.shape[1]
        classes = sorted(np.unique(y).tolist())
        N = len(X)
        print(f"\n=== {model_name} (N={N}, d={d}) ===")

        # Fixed train/test split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(sss.split(X, y))
        X_tr_base, X_te_base = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        centroids, sigma_W = compute_class_stats(X, y)

        for ci in classes:
            ranking = compute_ranked_competitors(centroids, sigma_W, d, ci)
            kappa_j1, j1_class = ranking[0]
            kappa_j2, j2_class = ranking[1]
            gap = float(kappa_j2 - kappa_j1)

            # Baseline q
            q_base = eval_q_ci(X_tr_base, y_tr, X_te_base, y_te, ci)
            if q_base is None:
                continue
            logit_base = safe_logit(q_base)

            # Determine delta list: only use deltas in stable-rank regime (delta < gap/2)
            stable_deltas = [d for d in DELTA_LIST_SMALL if d < gap / 2.0]
            if len(stable_deltas) < 3:
                # Not enough stable-rank deltas — skip or use rank-switch data
                print(f"  ci={ci}: gap={gap:.4f} too small for stable-rank "
                      f"(need gap > {2*DELTA_LIST_SMALL[2]:.3f}). "
                      f"Regime: rank-switch. Skipping Jacobian estimate.")
                all_records.append({
                    "arch": model_name, "ci": ci,
                    "kappa_j1": float(kappa_j1), "kappa_j2": float(kappa_j2),
                    "gap": float(gap), "q_base": float(q_base),
                    "regime": "rank_switch",
                    "w_empirical": None, "tau_star_est": None,
                    "w_phi_tau02": float(np.exp(-gap / TAU_STAR_POOLED)),
                })
                continue

            # Dose-response for j1-only and j2-only (stable-rank deltas only)
            j1_deltas, j1_dlogits = [0.0], [0.0]
            j2_deltas, j2_dlogits = [0.0], [0.0]

            for delta in stable_deltas:
                # j1-only arm
                X_tr_j1, X_te_j1 = shift_competitor_train_test(
                    X_tr_base, X_te_base, y_tr, y_te,
                    centroids[ci], j1_class, centroids[j1_class],
                    delta, sigma_W, d)
                q_j1 = eval_q_ci(X_tr_j1, y_tr, X_te_j1, y_te, ci)
                if q_j1 is not None:
                    j1_deltas.append(delta)
                    j1_dlogits.append(safe_logit(q_j1) - logit_base)

                # j2-only arm
                X_tr_j2, X_te_j2 = shift_competitor_train_test(
                    X_tr_base, X_te_base, y_tr, y_te,
                    centroids[ci], j2_class, centroids[j2_class],
                    delta, sigma_W, d)
                q_j2 = eval_q_ci(X_tr_j2, y_tr, X_te_j2, y_te, ci)
                if q_j2 is not None:
                    j2_deltas.append(delta)
                    j2_dlogits.append(safe_logit(q_j2) - logit_base)

            slope_j1, r_j1 = fit_slope(j1_deltas, j1_dlogits)
            slope_j2, r_j2 = fit_slope(j2_deltas, j2_dlogits)

            w_emp = None
            tau_est = None
            if (slope_j1 is not None and slope_j2 is not None
                    and abs(slope_j1) > 0.001 and slope_j1 > 0 and slope_j2 > 0):
                w_emp = float(slope_j2 / slope_j1)
                # Estimate tau_star from: log(w) = -gap / tau
                if w_emp > 0 and w_emp < 1.0:  # phi model predicts w < 1
                    tau_est = float(-gap / np.log(w_emp))

            w_phi = float(np.exp(-gap / TAU_STAR_POOLED))

            s1 = f"{slope_j1:.4f}" if slope_j1 is not None else "N/A"
            s2 = f"{slope_j2:.4f}" if slope_j2 is not None else "N/A"
            w_str = f"{w_emp:.3f}" if w_emp is not None else "N/A"
            tau_str = f"{tau_est:.3f}" if tau_est is not None else "N/A"
            print(f"  ci={ci}: gap={gap:.4f} q={q_base:.3f} "
                  f"slope_j1={s1} slope_j2={s2} "
                  f"w={w_str} tau*={tau_str} w_phi={w_phi:.3f}")

            all_records.append({
                "arch": model_name, "ci": ci,
                "kappa_j1": float(kappa_j1), "kappa_j2": float(kappa_j2),
                "gap": float(gap), "q_base": float(q_base),
                "regime": "stable_rank",
                "slope_j1": float(slope_j1) if slope_j1 is not None else None,
                "slope_j2": float(slope_j2) if slope_j2 is not None else None,
                "r_j1": float(r_j1) if r_j1 is not None else None,
                "r_j2": float(r_j2) if r_j2 is not None else None,
                "w_empirical": float(w_emp) if w_emp is not None else None,
                "tau_star_est": float(tau_est) if tau_est is not None else None,
                "w_phi_tau02": float(w_phi),
                "n_stable_deltas": len(stable_deltas),
            })

    # ================================================================
    # ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: Gap-Conditioned Phi Jacobian Test")
    print("=" * 70)

    # Filter to valid stable-rank records with valid w_empirical
    valid = [r for r in all_records
             if r["regime"] == "stable_rank"
             and r.get("w_empirical") is not None
             and np.isfinite(r["w_empirical"])]

    n_stable = sum(1 for r in all_records if r["regime"] == "stable_rank")
    n_rank_switch = sum(1 for r in all_records if r["regime"] == "rank_switch")
    print(f"\nTotal (arch, class) pairs: {len(all_records)}")
    print(f"  Stable-rank regime: {n_stable}")
    print(f"  Rank-switch regime: {n_rank_switch}")
    print(f"  Valid w estimates: {len(valid)}")

    # Among valid: split by w < 1 (phi consistent) vs w >= 1 (phi inconsistent)
    w_lt1 = [r for r in valid if r["w_empirical"] < 1.0]
    w_gte1 = [r for r in valid if r["w_empirical"] >= 1.0]
    print(f"  w < 1 (phi-consistent): {len(w_lt1)}")
    print(f"  w >= 1 (phi-inconsistent): {len(w_gte1)}")

    # Correlation test: log(w_empirical) vs -gap (among all valid)
    # Skip w_gte1 that have log(w) > 0 (violate phi model)
    corr_r, corr_p = None, None
    if len(valid) >= 5:
        gaps = np.array([r["gap"] for r in valid])
        log_w = np.array([np.log(r["w_empirical"]) for r in valid])
        mask = np.isfinite(log_w)
        if mask.sum() >= 5:
            corr_r, corr_p = pearsonr(-gaps[mask], log_w[mask])
            sp_r, sp_p = spearmanr(-gaps[mask], log_w[mask])
            print(f"\nCorrelation: log(w) vs -gap (n={mask.sum()})")
            print(f"  Pearson r = {corr_r:.4f}, p = {corr_p:.4e}")
            print(f"  Spearman r = {sp_r:.4f}, p = {sp_p:.4e}")
            corr_pass = bool(corr_r > 0.60) if corr_r is not None else False
            print(f"  PASS (r > 0.60): {'PASS' if corr_pass else 'FAIL'}")

    # Tau* estimation (only from w < 1 pairs)
    tau_estimates = [r["tau_star_est"] for r in w_lt1 if r["tau_star_est"] is not None]
    tau_estimates = [t for t in tau_estimates if np.isfinite(t) and 0 < t < 10]
    if tau_estimates:
        tau_median = float(np.median(tau_estimates))
        tau_mean = float(np.mean(tau_estimates))
        tau_std = float(np.std(tau_estimates))
        print(f"\nTau* estimates (n={len(tau_estimates)} w<1 pairs):")
        print(f"  median = {tau_median:.4f}, mean = {tau_mean:.4f}, std = {tau_std:.4f}")
        print(f"  Pooled phi_upgrade_pooled tau* = {TAU_STAR_POOLED}")
        tau_pass = bool(0.05 < tau_median < 0.50)
        tau_close = bool(abs(tau_median - TAU_STAR_POOLED) < 0.15)
        print(f"  PASS (tau in [0.05, 0.50]): {'PASS' if tau_pass else 'FAIL'}")
        print(f"  MATCH (|tau_median - 0.2| < 0.15): {'PASS' if tau_close else 'FAIL'}")
    else:
        tau_median, tau_mean, tau_std = None, None, None
        tau_pass, tau_close = False, False

    # w vs w_phi comparison (among w < 1)
    if len(w_lt1) >= 3:
        w_emp_arr = np.array([r["w_empirical"] for r in w_lt1])
        w_phi_arr = np.array([r["w_phi_tau02"] for r in w_lt1])
        if len(w_emp_arr) >= 3 and w_emp_arr.std() > 0:
            r_ww, p_ww = pearsonr(w_phi_arr, w_emp_arr)
            print(f"\nw_empirical vs w_phi(tau=0.2) correlation (n={len(w_lt1)}):")
            print(f"  r = {r_ww:.4f}, p = {p_ww:.4e}")

    # Per-arch summary
    print(f"\nPer-architecture summary:")
    for arch_name in EMBS.keys():
        arch_recs = [r for r in all_records if r["arch"] == arch_name]
        stable = [r for r in arch_recs if r["regime"] == "stable_rank"]
        valid_w = [r for r in stable if r.get("w_empirical") is not None]
        w_lt1_arch = [r for r in valid_w if r["w_empirical"] < 1.0]
        print(f"  {arch_name}: {len(arch_recs)} pairs, "
              f"{len(stable)} stable-rank, "
              f"{len(valid_w)} valid w, "
              f"{len(w_lt1_arch)} phi-consistent (w<1)")

    # Print gap distribution for stable-rank pairs
    stable_pairs = [r for r in all_records if r["regime"] == "stable_rank"]
    if stable_pairs:
        gaps = [r["gap"] for r in stable_pairs]
        print(f"\nStable-rank gap distribution:")
        print(f"  min={min(gaps):.4f}, max={max(gaps):.4f}, "
              f"mean={np.mean(gaps):.4f}, median={np.median(gaps):.4f}")

    result = {
        "experiment": "gap_jacobian_rct",
        "tau_star_pooled": TAU_STAR_POOLED,
        "total_pairs": len(all_records),
        "n_stable_rank": n_stable,
        "n_rank_switch": n_rank_switch,
        "n_valid_w": len(valid),
        "n_w_lt1": len(w_lt1),
        "n_w_gte1": len(w_gte1),
        "corr_log_w_vs_neg_gap_r": float(corr_r) if corr_r is not None else None,
        "corr_log_w_vs_neg_gap_p": float(corr_p) if corr_p is not None else None,
        "corr_pass": bool(corr_r > 0.60 if corr_r is not None else False),
        "tau_star_estimates": tau_estimates,
        "tau_star_median": float(tau_median) if tau_median is not None else None,
        "tau_star_mean": float(tau_mean) if tau_mean is not None else None,
        "tau_star_std": float(tau_std) if tau_std is not None else None,
        "tau_pass": bool(tau_pass),
        "tau_close_to_pooled": bool(tau_close),
        "records": all_records,
    }

    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nResults saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
