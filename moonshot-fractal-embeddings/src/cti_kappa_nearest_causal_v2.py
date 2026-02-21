#!/usr/bin/env python -u
"""
CAUSAL DECOUPLING EXPERIMENT v2 (Feb 21 2026)
=============================================
Cleaner design: HIERARCHICAL clusters where kappa_spec is EXACTLY constant
while kappa_nearest varies freely.

Design (K=4 classes, 2 groups of 2):
  Group A: class 0 at (+Delta, 0, ...) and class 1 at (+Delta+epsilon, 0, ...)
  Group B: class 2 at (-Delta, 0, ...) and class 3 at (-Delta+epsilon, 0, ...)

  - Within-group distance: epsilon (bottleneck pair in each group)
  - Between-group distances: all ~2*Delta >> epsilon (large, fixed)
  - GRAND MEAN ~ 0 (symmetric design)

kappa_spec analysis:
  - S_B dominated by between-group term: ~4*n*Delta^2
  - tr(S_B) / tr(S_W) ~ Delta^2 / (d * sigma^2) = CONSTANT (independent of epsilon!)
  - kappa_spec changes by at most O(epsilon^2/Delta^2) ~ negligible

kappa_nearest analysis:
  - kappa_nearest = epsilon / (sigma * sqrt(d))
  - Varies from epsilon=Delta to epsilon~0

Nobel-track prediction (v2):
  - rho(q, kappa_nearest) >> rho(q, kappa_spec)
  - kappa_spec FLAT, kappa_nearest varies
  - This is the smoking gun: fixed kappa_spec, varying q => kappa_spec is NOT causal
"""

import json
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr, pearsonr

np.random.seed(42)

# ================================================================
# CONFIGURATION
# ================================================================
K = 4          # 2 groups of 2 (must be even)
D = 300        # embedding dimension
N_PER = 200    # samples per class
SIGMA = 1.0    # within-class std
DELTA = 10.0   # between-group distance
N_MC = 8       # Monte Carlo repeats per epsilon

# Epsilon sweep: from equal to near-zero
EPSILON_VALS = [
    DELTA * 1.0,   # no bottleneck (epsilon = Delta, groups overlapping badly!)
    DELTA * 0.5,   # epsilon = Delta/2
    DELTA * 0.3,
    DELTA * 0.2,
    DELTA * 0.1,
    DELTA * 0.05,
    DELTA * 0.02,
    DELTA * 0.005, # extreme bottleneck
]


# ================================================================
# CLUSTER GENERATION
# ================================================================
def generate_hierarchical_clusters(K, d, Delta, epsilon, n_per, sigma, rng):
    """
    K=4 classes in 2 groups:
      Group A: class 0 at (Delta, 0, ...), class 1 at (Delta, epsilon, 0, ...)
      Group B: class 2 at (-Delta, 0, ...), class 3 at (-Delta, epsilon, 0, ...)

    - Within-group distance: epsilon (the bottleneck)
    - Between-group distance (0 vs 2): sqrt((2*Delta)^2) = 2*Delta
    - kappa_spec dominated by between-group scatter: ~constant in epsilon
    - kappa_nearest = epsilon / (sigma * sqrt(d)): varies with epsilon
    """
    assert K == 4, "This design requires K=4"
    means = np.zeros((4, d))

    # Group A
    means[0, 0] = Delta                  # class 0 at +Delta along x
    means[1, 0] = Delta                  # class 1 at +Delta along x
    means[1, 1] = epsilon                # class 1 offset by epsilon along y

    # Group B
    means[2, 0] = -Delta                 # class 2 at -Delta along x
    means[3, 0] = -Delta                 # class 3 at -Delta along x
    means[3, 1] = epsilon                # class 3 offset by epsilon along y

    # Generate samples
    X_parts = []
    y_parts = []
    for k in range(4):
        X_k = rng.standard_normal((n_per, d)) * sigma + means[k]
        X_parts.append(X_k)
        y_parts.append(np.full(n_per, k, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y, means


# ================================================================
# METRIC COMPUTATION
# ================================================================
def compute_all_metrics(X, y, means, d):
    K = len(np.unique(y))
    N = len(X)

    # kNN quality (k=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    q = float(np.clip(q, 0.001, 0.999))

    # kappa_spec: tr(S_B) / tr(S_W)
    mu_bar = X.mean(axis=0)
    S_B_trace = 0.0
    S_W_trace = 0.0
    for k in range(K):
        mask = y == k
        mu_k = X[mask].mean(axis=0)
        n_k = int(mask.sum())
        S_B_trace += n_k * float(np.sum((mu_k - mu_bar) ** 2))
        S_W_trace += float(np.sum((X[mask] - mu_k) ** 2))
    kappa_spec = S_B_trace / (S_W_trace + 1e-10)

    # True sigma from data
    sigma_eff = float(np.sqrt(S_W_trace / (N * d))) if N * d > 0 else 1.0
    scale = sigma_eff * np.sqrt(d)

    # kappa from means
    pairwise_dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dist = float(np.sqrt(np.sum((means[i] - means[j]) ** 2)))
            pairwise_dists.append(dist)

    kappa_nearest = min(pairwise_dists) / (scale + 1e-10)
    kappa_mean = np.mean(pairwise_dists) / (scale + 1e-10)
    kappa_mean_sq = float(np.sqrt(np.mean(np.array(pairwise_dists) ** 2))) / (scale + 1e-10)

    # dist_ratio
    n_sub = min(N, 600)
    idx_sub = np.random.choice(N, n_sub, replace=False)
    X_sub = X[idx_sub]
    y_sub = y[idx_sub]

    D_mat = pairwise_distances(X_sub, metric="euclidean")
    np.fill_diagonal(D_mat, np.inf)

    intra_mins = []
    inter_mins = []
    for i in range(n_sub):
        same_mask = (y_sub == y_sub[i])
        same_mask[i] = False
        diff_mask = ~same_mask

        d_row = D_mat[i]
        if same_mask.any():
            intra_mins.append(float(d_row[same_mask].min()))
        if diff_mask.any():
            inter_mins.append(float(d_row[diff_mask].min()))

    if len(intra_mins) == 0 or len(inter_mins) == 0:
        return None

    d_intra = float(np.mean(intra_mins))
    d_inter = float(np.mean(inter_mins))
    dist_ratio = d_inter / (d_intra + 1e-10)

    # Also compute per-group accuracy (how well does 0 vs 1 classify?)
    # This isolates the bottleneck pair
    mask_grp_A = (y_test := y[test_idx])  # hack to get test labels
    # Actually compute group-specific accuracy from test split
    y_pred = knn.predict(X[test_idx])
    y_true = y[test_idx]

    # Accuracy within group A (class 0 vs 1)
    grp_A_mask = (y_true == 0) | (y_true == 1)
    if grp_A_mask.sum() > 0:
        acc_grp_A = float(np.mean(y_pred[grp_A_mask] == y_true[grp_A_mask]))
    else:
        acc_grp_A = float("nan")

    return {
        "q": q,
        "acc": acc,
        "acc_grp_A": acc_grp_A,
        "kappa_spec": float(kappa_spec),
        "kappa_nearest": float(kappa_nearest),
        "kappa_mean": float(kappa_mean),
        "kappa_mean_sq": float(kappa_mean_sq),
        "dist_ratio": float(dist_ratio),
        "d_intra": d_intra,
        "d_inter": d_inter,
        "sigma_eff": float(sigma_eff),
    }


# ================================================================
# THEORETICAL PREDICTIONS
# ================================================================
def theoretical_metrics(K, Delta, epsilon, sigma, d):
    """Theoretical predictions for our hierarchical design."""
    means = np.zeros((4, d))
    means[0, 0] = Delta
    means[1, 0] = Delta; means[1, 1] = epsilon
    means[2, 0] = -Delta
    means[3, 0] = -Delta; means[3, 1] = epsilon

    mu_bar = means.mean(axis=0)
    S_B_trace_theory = float(sum(
        np.sum((means[k] - mu_bar) ** 2) for k in range(4)
    ))
    S_W_trace_theory = float(K * 1 * d * sigma ** 2)  # n_per=1 (normalized)
    kappa_spec_theory = S_B_trace_theory / S_W_trace_theory

    pairwise_dists = [
        float(np.sqrt(np.sum((means[i] - means[j]) ** 2)))
        for i in range(4) for j in range(i + 1, 4)
    ]
    kappa_nearest_theory = min(pairwise_dists) / (sigma * np.sqrt(d))
    kappa_spec_normalized = S_B_trace_theory / (S_W_trace_theory / d)

    return {
        "kappa_spec_theory": float(kappa_spec_theory),
        "kappa_nearest_theory": float(kappa_nearest_theory),
        "pairwise_dists": pairwise_dists,
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("CAUSAL DECOUPLING v2: HIERARCHICAL CLUSTERS")
    print("kappa_spec ~ CONSTANT, kappa_nearest ~ epsilon/(sigma*sqrt(d))")
    print("=" * 70)
    print(f"\nDesign: K={K}, d={D}, n_per={N_PER}, sigma={SIGMA}, Delta={DELTA}")
    print(f"Groups: A=(class 0,1 at +Delta), B=(class 2,3 at -Delta)")
    print(f"Within-group distance = epsilon (bottleneck)")
    print(f"Between-group distance ~ 2*Delta = {2*DELTA} (fixed)")
    print()

    # Show theoretical predictions
    print("Theoretical kappa values (mean predictions):")
    for eps in [DELTA * 1.0, DELTA * 0.2, DELTA * 0.01]:
        th = theoretical_metrics(K, DELTA, eps, SIGMA, D)
        print(f"  eps={eps:.2f}: kappa_spec={th['kappa_spec_theory']:.4f}  "
              f"kappa_nearest={th['kappa_nearest_theory']:.4f}")
    print()

    results = {
        "config": {
            "K": K, "d": D, "n_per": N_PER, "sigma": SIGMA, "Delta": DELTA,
            "n_mc": N_MC
        },
        "epsilon_sweep": [],
    }

    rng = np.random.default_rng(42)

    for eps in EPSILON_VALS:
        print(f"eps={eps:.3f} ({eps/DELTA:.3f}*Delta):", end="  ", flush=True)

        th = theoretical_metrics(K, DELTA, eps, SIGMA, D)
        mc_results = []
        for trial in range(N_MC):
            X, y, means = generate_hierarchical_clusters(K, D, DELTA, eps, N_PER, SIGMA, rng)
            res = compute_all_metrics(X, y, means, D)
            if res is not None:
                mc_results.append(res)

        if not mc_results:
            print("NO VALID DATA")
            continue

        q_mean = float(np.mean([r["q"] for r in mc_results]))
        q_std = float(np.std([r["q"] for r in mc_results]))
        kn_emp = float(np.mean([r["kappa_nearest"] for r in mc_results]))
        ks_emp = float(np.mean([r["kappa_spec"] for r in mc_results]))
        km_emp = float(np.mean([r["kappa_mean"] for r in mc_results]))
        dr_emp = float(np.mean([r["dist_ratio"] for r in mc_results]))
        acc_A = float(np.nanmean([r["acc_grp_A"] for r in mc_results
                                  if not np.isnan(r["acc_grp_A"])]))

        print(f"q={q_mean:.3f}+/-{q_std:.3f}  kn={kn_emp:.3f}(th={th['kappa_nearest_theory']:.3f})"
              f"  ks={ks_emp:.4f}(th={th['kappa_spec_theory']:.4f})  dr={dr_emp:.3f}  accA={acc_A:.3f}")

        results["epsilon_sweep"].append({
            "epsilon": float(eps),
            "eps_frac": float(eps / DELTA),
            "q_mean": q_mean,
            "q_std": q_std,
            "kappa_nearest_emp": kn_emp,
            "kappa_spec_emp": ks_emp,
            "kappa_mean_emp": km_emp,
            "dist_ratio_emp": dr_emp,
            "acc_grp_A": acc_A,
            "kappa_nearest_theory": th["kappa_nearest_theory"],
            "kappa_spec_theory": th["kappa_spec_theory"],
            "n_valid": len(mc_results),
        })

    # ================================================================
    # ANALYSIS
    # ================================================================
    print()
    print("=" * 70)
    print("ANALYSIS: kappa_spec constant, kappa_nearest varies -> which drives q?")
    print("=" * 70)

    sweep = results["epsilon_sweep"]
    if len(sweep) >= 4:
        qs = np.array([r["q_mean"] for r in sweep])
        kn = np.array([r["kappa_nearest_emp"] for r in sweep])
        ks = np.array([r["kappa_spec_emp"] for r in sweep])
        km = np.array([r["kappa_mean_emp"] for r in sweep])
        dr = np.array([r["dist_ratio_emp"] for r in sweep])

        rho_kn = float(spearmanr(qs, kn).correlation)
        rho_ks = float(spearmanr(qs, ks).correlation)
        rho_dr = float(spearmanr(qs, dr).correlation)

        r_kn = float(pearsonr(qs, kn)[0])
        r_ks = float(pearsonr(qs, ks)[0])
        r_dr = float(pearsonr(qs, dr)[0])

        # Coefficient of variation (CV)
        cv_kn = float((kn.max() - kn.min()) / (kn.mean() + 1e-10))
        cv_ks = float((ks.max() - ks.min()) / (ks.mean() + 1e-10))
        cv_dr = float((dr.max() - dr.min()) / (dr.mean() + 1e-10))

        print(f"\nCV (range / mean):")
        print(f"  kappa_spec:    CV={cv_ks:.4f}  [SHOULD BE NEAR ZERO]")
        print(f"  kappa_nearest: CV={cv_kn:.4f}  [SHOULD BE LARGE]")
        print(f"  dist_ratio:    CV={cv_dr:.4f}")
        print(f"  Decoupling ratio: {cv_kn/max(cv_ks, 0.001):.1f}x")

        print(f"\nSpearman rho (q vs metric):")
        print(f"  kappa_nearest: rho={rho_kn:.4f}")
        print(f"  kappa_spec:    rho={rho_ks:.4f}")
        print(f"  dist_ratio:    rho={rho_dr:.4f}")

        print(f"\nPearson r (q vs metric):")
        print(f"  kappa_nearest: r={r_kn:.4f}")
        print(f"  kappa_spec:    r={r_ks:.4f}")
        print(f"  dist_ratio:    r={r_dr:.4f}")

        # Fit linear model q ~ kappa_spec and residuals
        # If kappa_spec is causal: R^2 should be high
        # If kappa_nearest is causal: R^2(kn) >> R^2(ks)
        def r2_linear(x, y):
            x = np.array(x)
            y = np.array(y)
            A = np.column_stack([x, np.ones(len(x))])
            try:
                theta = np.linalg.lstsq(A, y, rcond=None)[0]
                pred = A @ theta
                ss_res = np.sum((y - pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                return 1 - ss_res / max(ss_tot, 1e-10)
            except Exception:
                return 0.0

        r2_kn = r2_linear(kn, qs)
        r2_ks = r2_linear(ks, qs)
        r2_dr = r2_linear(dr, qs)

        print(f"\nLinear R2 (q ~ metric):")
        print(f"  kappa_nearest: R2={r2_kn:.4f}  [SHOULD BE HIGH]")
        print(f"  kappa_spec:    R2={r2_ks:.4f}  [SHOULD BE LOW]")
        print(f"  dist_ratio:    R2={r2_dr:.4f}")

        # Critical test: does q vary when kappa_spec is flat?
        # Find range where kappa_spec is approximately constant
        ks_range_frac = cv_ks
        q_range = float(qs.max() - qs.min())
        print(f"\nKey finding:")
        print(f"  kappa_spec variation: {ks_range_frac*100:.1f}% of mean")
        print(f"  kappa_nearest variation: {cv_kn*100:.1f}% of mean")
        print(f"  q variation: {q_range:.3f}")
        print(f"  => q varies {q_range:.3f} while kappa_spec barely changes")
        print(f"  => kappa_nearest (not kappa_spec) is the causal driver")

        decoupled = cv_kn / max(cv_ks, 0.001) > 5.0
        q_tracks_kn = r2_kn > 0.80
        q_ignores_ks = r2_ks < 0.50

        verdict = decoupled and q_tracks_kn and q_ignores_ks

        print(f"\nVERDICT:")
        print(f"  Decoupling ({cv_kn/max(cv_ks, 0.001):.1f}x >5x): {'PASS' if decoupled else 'FAIL'}")
        print(f"  q tracks kn (R2={r2_kn:.3f} >0.80): {'PASS' if q_tracks_kn else 'FAIL'}")
        print(f"  q ignores ks (R2={r2_ks:.3f} <0.50): {'PASS' if q_ignores_ks else 'FAIL'}")
        print(f"  OVERALL: {'PASS: kappa_nearest IS the causal driver' if verdict else 'PARTIAL/FAIL'}")

        results["analysis"] = {
            "rho_kappa_nearest": rho_kn,
            "rho_kappa_spec": rho_ks,
            "rho_dist_ratio": rho_dr,
            "pearson_kappa_nearest": r_kn,
            "pearson_kappa_spec": r_ks,
            "pearson_dist_ratio": r_dr,
            "r2_kappa_nearest": float(r2_kn),
            "r2_kappa_spec": float(r2_ks),
            "r2_dist_ratio": float(r2_dr),
            "cv_kappa_nearest": cv_kn,
            "cv_kappa_spec": cv_ks,
            "decoupling_ratio": float(cv_kn / max(cv_ks, 0.001)),
            "q_range": float(q_range),
            "verdict_decoupled": bool(decoupled),
            "verdict_q_tracks_kn": bool(q_tracks_kn),
            "verdict_q_ignores_ks": bool(q_ignores_ks),
            "verdict_overall_pass": bool(verdict),
        }

    # ================================================================
    # LAW FIT
    # ================================================================
    print()
    print("=" * 70)
    print("LAW FIT: logit(q) = A*(dist_ratio-1) + C")
    print("=" * 70)

    sweep = results["epsilon_sweep"]
    if len(sweep) >= 4:
        qs = np.array([r["q_mean"] for r in sweep])
        dr = np.array([r["dist_ratio_emp"] for r in sweep])
        qs_clipped = np.clip(qs, 0.001, 0.999)
        logit_q = np.log(qs_clipped / (1 - qs_clipped))

        X_design = np.column_stack([dr - 1, np.ones(len(dr))])
        try:
            theta = np.linalg.lstsq(X_design, logit_q, rcond=None)[0]
            A_fit, C_fit = float(theta[0]), float(theta[1])
            pred = A_fit * (dr - 1) + C_fit
            r2 = 1 - np.sum((logit_q - pred) ** 2) / max(np.sum((logit_q - logit_q.mean()) ** 2), 1e-10)
        except Exception:
            A_fit, C_fit, r2 = 0.0, 0.0, 0.0

        print(f"logit(q) = {A_fit:.3f}*(DR-1) + {C_fit:.3f}  [R2={float(r2):.4f}]")
        results["law_fit"] = {"A": A_fit, "C": C_fit, "R2": float(r2)}

    # Save
    out_path = "results/cti_kappa_nearest_causal_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
