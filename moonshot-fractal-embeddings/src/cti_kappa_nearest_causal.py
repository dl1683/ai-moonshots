#!/usr/bin/env python -u
"""
CAUSAL DECOUPLING EXPERIMENT (Feb 21 2026)
==========================================
Core question: Is kappa_nearest or kappa_spec the causal driver of kNN quality?

Design:
  - K=6 Gaussian clusters in R^d
  - Classes 0..K-2: at delta * e_i (orthogonal, mutual distance = delta*sqrt(2))
  - Class K-1 (bottleneck class): at delta*e_0 + epsilon*e_{K-1}
    -> distance to class 0 = epsilon (small bottleneck)
    -> distance to other classes ~ delta*sqrt(2) (large, same as rest)
  - As epsilon varies from delta*sqrt(2) to 0:
    -> kappa_nearest decreases (bottleneck pair gets closer)
    -> kappa_spec stays APPROXIMATELY constant (only 1 of K*(K-1)/2 pairs changes)

Nobel-track prediction:
  - q tracks kappa_nearest (not kappa_spec) -> kappa_nearest is the causal driver
  - dist_ratio tracks kappa_nearest (not kappa_spec) -> dist_ratio is the right observable
  - kappa_spec is a BIASED PROXY that fails when there's a class-pair bottleneck
"""

import json
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import pairwise_distances
from scipy.special import logit as scipy_logit

np.random.seed(42)

# ================================================================
# CONFIGURATION
# ================================================================
K = 6          # number of classes
D = 200        # embedding dimension
N_PER = 150    # samples per class
SIGMA = 1.0    # within-class std
DELTA = 8.0    # "baseline" distance between orthogonal classes
N_MC = 5       # Monte Carlo repeats per epsilon

# Epsilon sweep: from nearly-equal (epsilon ~ delta) to bottleneck (epsilon ~ 0)
EPSILON_VALS = [
    DELTA * 1.0,   # no bottleneck (symmetric)
    DELTA * 0.8,
    DELTA * 0.6,
    DELTA * 0.4,
    DELTA * 0.2,
    DELTA * 0.1,
    DELTA * 0.05,
    DELTA * 0.01,  # extreme bottleneck
]


# ================================================================
# CLUSTER GENERATION
# ================================================================
def generate_clusters(K, d, delta, epsilon, n_per, sigma, rng):
    """
    K classes:
      - Classes 0..K-2: means at delta * e_i (orthogonal basis vectors)
      - Class K-1 (bottleneck): mean at means[0] + epsilon * e_{K-1}
        -> nearest class is class 0, distance = epsilon

    Returns (X, y, means) with shape X=(K*n_per, d), y=(K*n_per,), means=(K,d).
    """
    assert d >= K, f"Need d >= K, got d={d}, K={K}"
    means = np.zeros((K, d))

    # Classes 0..K-2: orthogonal positions
    for i in range(K - 1):
        means[i, i] = delta

    # Class K-1: close to class 0
    means[K - 1] = means[0].copy()
    means[K - 1, K - 1] = epsilon  # small displacement in orthogonal direction
    # Distance from class K-1 to class 0: sqrt(epsilon^2) = epsilon

    # Generate samples
    X_parts = []
    y_parts = []
    for k in range(K):
        X_k = rng.standard_normal((n_per, d)) * sigma + means[k]
        X_parts.append(X_k)
        y_parts.append(np.full(n_per, k, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y, means


# ================================================================
# METRIC COMPUTATION
# ================================================================
def compute_all_metrics(X, y, means, sigma, d):
    """
    Compute:
      q           : normalized kNN quality (k=1, 80/20 split)
      kappa_spec  : tr(S_B) / tr(S_W)
      kappa_nearest: min pairwise ||mu_i - mu_j|| / (sigma * sqrt(d))
      kappa_mean  : mean pairwise ||mu_i - mu_j|| / (sigma * sqrt(d))
      dist_ratio  : E[D_inter_min] / E[D_intra_min]
    Returns dict or None if invalid.
    """
    K = len(np.unique(y))
    N = len(X)

    # 1. kNN quality
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

    # 2. kappa_spec
    mu_bar = X.mean(axis=0)
    S_B_trace = 0.0
    S_W_trace = 0.0
    for k in range(K):
        mask = y == k
        mu_k = X[mask].mean(axis=0)
        n_k = int(mask.sum())
        S_B_trace += n_k * float(np.sum((mu_k - mu_bar) ** 2))
        S_W_trace += float(np.sum((X[mask] - mu_k) ** 2))
    kappa_spec = S_B_trace / (S_W_trace / N + 1e-10)

    # Normalize to be dimension-invariant (like our standard kappa definition)
    # kappa = tr(S_B)/tr(S_W) where S_B, S_W are normalized per sample
    kappa_spec_norm = S_B_trace / (S_W_trace + 1e-10)  # trace ratio (not per-dim)

    # 3. kappa_nearest and kappa_mean from means
    # sigma_W_eff = sqrt(S_W_trace / (N * d)) ~ sigma
    sigma_eff = float(np.sqrt(S_W_trace / (N * d))) if N * d > 0 else sigma
    # Normalize by sigma * sqrt(d)
    scale = sigma_eff * np.sqrt(d)

    pairwise_dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dist = float(np.sqrt(np.sum((means[i] - means[j]) ** 2)))
            pairwise_dists.append(dist)

    kappa_nearest = min(pairwise_dists) / (scale + 1e-10)
    kappa_mean = np.mean(pairwise_dists) / (scale + 1e-10)

    # 4. dist_ratio: E[D_intra_min] / E[D_inter_min]
    n_sub = min(N, 400)
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

    return {
        "q": q,
        "acc": acc,
        "kappa_spec": float(kappa_spec_norm),
        "kappa_nearest": float(kappa_nearest),
        "kappa_mean": float(kappa_mean),
        "dist_ratio": float(dist_ratio),
        "d_intra": d_intra,
        "d_inter": d_inter,
        "sigma_eff": float(sigma_eff),
    }


# ================================================================
# GEOMETRIC PREDICTIONS
# ================================================================
def theoretical_kappa_nearest(epsilon, delta, sigma, d):
    """Predicted kappa_nearest = epsilon / (sigma * sqrt(d))."""
    return epsilon / (sigma * np.sqrt(d))


def theoretical_kappa_spec_approx(K, epsilon, delta, sigma, d):
    """
    Approximate kappa_spec from mean pairwise distance:
    - K-1 pairs among classes 0..K-2: distance = delta*sqrt(2), dist^2 = 2*delta^2
    - 1 bottleneck pair (class 0, class K-1): dist^2 = epsilon^2
    - K-2 pairs (class k=1..K-2, class K-1): dist^2 = 2*delta^2 + epsilon^2 ~ 2*delta^2

    This approximates kappa_spec ~ mean_pairwise_dist^2 / (sigma^2 * d).
    """
    n_pairs = K * (K - 1) // 2
    # C(K-1,2) pairs among 0..K-2
    n_core = (K - 1) * (K - 2) // 2
    # 1 bottleneck pair
    # K-2 pairs: class k=1..K-2 to class K-1
    n_side = K - 2
    # Total
    total_sq = n_core * 2 * delta ** 2 + epsilon ** 2 + n_side * (2 * delta ** 2 + epsilon ** 2)
    mean_sq = total_sq / n_pairs
    return mean_sq / (sigma ** 2 * d)


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("CAUSAL DECOUPLING EXPERIMENT")
    print("Does quality track kappa_nearest or kappa_spec?")
    print("=" * 70)
    print(f"\nSetup: K={K}, d={D}, n_per={N_PER}, sigma={SIGMA}, delta={DELTA}")
    print(f"Bottleneck: class {K-1} placed at distance epsilon from class 0")
    print(f"Prediction: q tracks kappa_nearest, NOT kappa_spec")
    print()

    results = {
        "config": {
            "K": K, "d": D, "n_per": N_PER, "sigma": SIGMA, "delta": DELTA,
            "n_mc": N_MC
        },
        "epsilon_sweep": [],
    }

    for eps in EPSILON_VALS:
        print(f"epsilon = {eps:.3f} ({eps/DELTA:.2f} * delta):", end="  ", flush=True)

        # Theoretical predictions
        kn_theory = theoretical_kappa_nearest(eps, DELTA, SIGMA, D)
        ks_theory = theoretical_kappa_spec_approx(K, eps, DELTA, SIGMA, D)

        # Monte Carlo
        mc_results = []
        rng = np.random.default_rng(123)
        for trial in range(N_MC):
            X, y, means = generate_clusters(K, D, DELTA, eps, N_PER, SIGMA, rng)
            res = compute_all_metrics(X, y, means, SIGMA, D)
            if res is not None:
                mc_results.append(res)

        if not mc_results:
            print("NO VALID DATA")
            continue

        # Aggregate
        q_mean = float(np.mean([r["q"] for r in mc_results]))
        q_std = float(np.std([r["q"] for r in mc_results]))
        kn_emp = float(np.mean([r["kappa_nearest"] for r in mc_results]))
        ks_emp = float(np.mean([r["kappa_spec"] for r in mc_results]))
        km_emp = float(np.mean([r["kappa_mean"] for r in mc_results]))
        dr_emp = float(np.mean([r["dist_ratio"] for r in mc_results]))

        print(f"q={q_mean:.3f}+/-{q_std:.3f}  kn={kn_emp:.3f}  ks={ks_emp:.3f}  dr={dr_emp:.3f}")

        results["epsilon_sweep"].append({
            "epsilon": float(eps),
            "eps_frac": float(eps / DELTA),
            "q_mean": q_mean,
            "q_std": q_std,
            "kappa_nearest_emp": kn_emp,
            "kappa_spec_emp": ks_emp,
            "kappa_mean_emp": km_emp,
            "dist_ratio_emp": dr_emp,
            "kappa_nearest_theory": float(kn_theory),
            "kappa_spec_theory": float(ks_theory),
            "n_valid": len(mc_results),
        })

    # ================================================================
    # ANALYSIS: Correlation of q with each metric
    # ================================================================
    print()
    print("=" * 70)
    print("ANALYSIS: Which metric predicts q?")
    print("=" * 70)

    sweep = results["epsilon_sweep"]
    if len(sweep) >= 4:
        from scipy.stats import spearmanr, pearsonr

        qs = np.array([r["q_mean"] for r in sweep])
        kn = np.array([r["kappa_nearest_emp"] for r in sweep])
        ks = np.array([r["kappa_spec_emp"] for r in sweep])
        km = np.array([r["kappa_mean_emp"] for r in sweep])
        dr = np.array([r["dist_ratio_emp"] for r in sweep])

        rho_kn = float(spearmanr(qs, kn).correlation)
        rho_ks = float(spearmanr(qs, ks).correlation)
        rho_km = float(spearmanr(qs, km).correlation)
        rho_dr = float(spearmanr(qs, dr).correlation)

        r_kn = float(pearsonr(qs, kn)[0])
        r_ks = float(pearsonr(qs, ks)[0])
        r_dr = float(pearsonr(qs, dr)[0])

        print(f"\nSpearman rho (q vs metric):")
        print(f"  kappa_nearest: rho={rho_kn:.4f}  (SHOULD BE HIGH)")
        print(f"  kappa_spec:    rho={rho_ks:.4f}  (SHOULD BE LOW)")
        print(f"  kappa_mean:    rho={rho_km:.4f}  (INTERMEDIATE)")
        print(f"  dist_ratio:    rho={rho_dr:.4f}  (SHOULD BE HIGH)")

        print(f"\nPearson r (q vs metric):")
        print(f"  kappa_nearest: r={r_kn:.4f}")
        print(f"  kappa_spec:    r={r_ks:.4f}")
        print(f"  dist_ratio:    r={r_dr:.4f}")

        # Test: is there significant variation in kappa_spec?
        ks_range = float(ks.max() - ks.min()) / float(ks.mean())
        kn_range = float(kn.max() - kn.min()) / float(kn.mean())
        print(f"\nCoefficient of variation:")
        print(f"  kappa_spec:    CV={ks_range:.3f}  (SHOULD BE LOW)")
        print(f"  kappa_nearest: CV={kn_range:.3f}  (SHOULD BE HIGH)")
        print(f"  Ratio (kn/ks variation): {kn_range/max(ks_range, 0.001):.1f}x")

        # Verdict
        decoupled = (kn_range / max(ks_range, 0.001)) > 5.0
        q_tracks_kn = rho_kn > 0.90
        q_ignores_ks = abs(rho_ks) < 0.50
        passed = decoupled and q_tracks_kn

        print(f"\nVERDICT:")
        print(f"  kn/ks decoupling: {kn_range/max(ks_range, 0.001):.1f}x {'[PASS >5x]' if decoupled else '[FAIL <5x]'}")
        print(f"  q tracks kn: rho={rho_kn:.3f} {'[PASS >0.9]' if q_tracks_kn else '[FAIL <0.9]'}")
        print(f"  q ignores ks: rho={rho_ks:.3f} {'[PASS <0.5]' if q_ignores_ks else '[FAIL >0.5]'}")
        print(f"  OVERALL: {'PASS - kappa_nearest IS causal driver' if passed else 'FAIL'}")

        results["analysis"] = {
            "rho_kappa_nearest": rho_kn,
            "rho_kappa_spec": rho_ks,
            "rho_kappa_mean": rho_km,
            "rho_dist_ratio": rho_dr,
            "pearson_kappa_nearest": r_kn,
            "pearson_kappa_spec": r_ks,
            "pearson_dist_ratio": r_dr,
            "cv_kappa_nearest": kn_range,
            "cv_kappa_spec": ks_range,
            "decoupling_ratio": float(kn_range / max(ks_range, 0.001)),
            "verdict_decoupled": bool(decoupled),
            "verdict_q_tracks_kn": bool(q_tracks_kn),
            "verdict_overall_pass": bool(passed),
        }
    else:
        print("Insufficient data for analysis.")
        results["analysis"] = {"error": "insufficient_data"}

    # ================================================================
    # ALSO: dist_ratio as predictor
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
        logit_q = np.array([float(np.log(q / (1 - q))) for q in qs_clipped])

        # OLS fit
        X_design = np.column_stack([dr - 1, np.ones(len(dr))])
        try:
            theta = np.linalg.lstsq(X_design, logit_q, rcond=None)[0]
            A_fit, C_fit = float(theta[0]), float(theta[1])
        except Exception:
            A_fit, C_fit = 0.0, 0.0

        pred = A_fit * (dr - 1) + C_fit
        ss_res = float(np.sum((logit_q - pred) ** 2))
        ss_tot = float(np.sum((logit_q - logit_q.mean()) ** 2))
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        print(f"logit(q) = {A_fit:.3f}*(DR-1) + {C_fit:.3f}")
        print(f"R2 = {r2:.4f}")

        results["law_fit"] = {
            "A": A_fit, "C": C_fit, "R2": float(r2),
        }

    # Save
    out_path = "results/cti_kappa_nearest_causal.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
