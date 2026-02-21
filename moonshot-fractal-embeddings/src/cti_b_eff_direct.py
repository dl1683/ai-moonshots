#!/usr/bin/env python -u
"""
DIRECT b_eff MEASUREMENT (Feb 21 2026)
=======================================
Goal: Directly measure b_eff in logit(q) = A*kappa - b_eff*log(K-1) + C
by varying K at FIXED kappa values.

Setup:
  - Synthetic isotropic Gaussians
  - For each kappa in [0.3, 0.5, 0.7, 1.0]:
    - Generate clusters at K = 5, 10, 20, 50, 100, 200 with kappa fixed
    - Measure logit(q) for each K
    - Slope of logit(q) vs log(K-1) = -b_eff

If b_eff = 1 (Gumbel Race exact): logit(q) decreases by 1 per log(K) unit
If b_eff < 1 (finite-sample): logit(q) decreases less per log(K) unit

Nobel-track significance:
  - b_eff tells us how close real distributions are to the ETF/Gumbel ideal
  - b_eff(n, d, K) formula from first principles is the key remaining gap
  - Measuring b_eff directly tests the Gumbel Race prediction at each kappa
"""

import json
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import spearmanr

np.random.seed(42)

# ================================================================
# CONFIG
# ================================================================
D = 200
N_PER = 100    # samples per class (fixed)
SIGMA = 1.0
N_MC = 10      # Monte Carlo repeats per (K, kappa) cell

# Fixed kappa values to test
KAPPA_VALS = [0.25, 0.35, 0.50, 0.70, 1.00]
# K values to sweep
K_VALS = [4, 6, 10, 20, 50, 100, 200]


# ================================================================
# CLUSTER GENERATION AT FIXED KAPPA
# ================================================================
def generate_fixed_kappa_clusters(K, d, kappa_nearest, n_per, sigma, rng):
    """
    Generate K isotropic Gaussians in R^d with kappa_nearest ≈ target.
    Use simplex / regular arrangement: all means equidistant from origin.
    For K <= d, place means on vertices of regular simplex scaled to give desired kappa.

    kappa_nearest = delta / (sigma * sqrt(d)) where delta = pairwise distance between nearest means.
    For regular simplex: all pairwise distances are equal to delta = sqrt(2) * ||mu||.
    So ||mu|| = delta / sqrt(2) = kappa_nearest * sigma * sqrt(d) / sqrt(2).
    """
    # Target nearest-class distance in actual units
    delta_target = kappa_nearest * sigma * np.sqrt(d)

    # Generate K means on a regular simplex in R^d
    # For K <= d+1: use standard simplex construction
    # For K > d: use approximate random placement
    if K <= d + 1:
        # Standard regular simplex in K-1 dimensions, embedded in R^d
        # Use Gram-Schmidt to get K orthogonal-ish directions
        means = np.zeros((K, d))
        # Random orthogonal vectors from QR decomposition
        rand_matrix = rng.standard_normal((d, K))
        Q, _ = np.linalg.qr(rand_matrix)
        directions = Q[:, :K].T  # K x d, orthonormal

        # Place means at equal distance from origin
        # For regular simplex: ||mu_i - mu_j|| = delta for all i != j
        # With orthonormal vectors: ||directions_i - directions_j||^2 = 2 for i != j
        # So scale factor r such that ||r*(d_i - d_j)|| = delta -> r = delta/sqrt(2)
        r = delta_target / np.sqrt(2)
        means = r * directions  # (K, d)
    else:
        # K > d+1: place means randomly in R^d with approximately equal spacing
        # Use random unit vectors scaled to target radius
        rand_matrix = rng.standard_normal((K, d))
        norms = np.linalg.norm(rand_matrix, axis=1, keepdims=True)
        unit_vecs = rand_matrix / (norms + 1e-10)
        # Scale to achieve approximately delta_target pairwise distances
        # For random unit vectors in R^d: E[||u_i - u_j||^2] = 2 (nearly orthogonal)
        r = delta_target / np.sqrt(2)
        means = r * unit_vecs

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


def compute_q(X, y, K):
    """Compute normalized kNN quality (k=1, 80/20 split)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    try:
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X[train_idx], y[train_idx])
    acc = float(knn.score(X[test_idx], y[test_idx]))
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    return float(np.clip(q, 0.001, 0.999))


def compute_kappa_nearest(means, sigma, d):
    """Compute kappa_nearest from class means."""
    K = len(means)
    min_dist = float("inf")
    for i in range(K):
        for j in range(i + 1, K):
            dist = float(np.sqrt(np.sum((means[i] - means[j]) ** 2)))
            if dist < min_dist:
                min_dist = dist
    return min_dist / (sigma * np.sqrt(d))


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("DIRECT b_eff MEASUREMENT")
    print("logit(q) = A*kappa - b_eff*log(K-1) + C")
    print("=" * 70)
    print(f"\nSetup: d={D}, n_per={N_PER}, sigma={SIGMA}, n_mc={N_MC}")
    print(f"kappa values: {KAPPA_VALS}")
    print(f"K values: {K_VALS}")
    print()

    results = {
        "config": {
            "d": D, "n_per": N_PER, "sigma": SIGMA, "n_mc": N_MC,
            "kappa_vals": KAPPA_VALS, "K_vals": K_VALS,
        },
        "data": {},
        "b_eff_per_kappa": {},
        "summary": {},
    }

    rng = np.random.default_rng(42)

    for kappa in KAPPA_VALS:
        print(f"\nkappa = {kappa:.2f}:")
        results["data"][str(kappa)] = []

        logit_q_list = []
        log_K_list = []

        for K in K_VALS:
            if K > D:
                print(f"  K={K}: skip (K > d={D})")
                continue

            mc_qs = []
            mc_kn = []
            for trial in range(N_MC):
                X, y, means = generate_fixed_kappa_clusters(K, D, kappa, N_PER, SIGMA, rng)
                q = compute_q(X, y, K)
                kn = compute_kappa_nearest(means, SIGMA, D)
                if q is not None:
                    mc_qs.append(q)
                    mc_kn.append(kn)

            if not mc_qs:
                continue

            q_mean = float(np.mean(mc_qs))
            q_std = float(np.std(mc_qs))
            kn_mean = float(np.mean(mc_kn))
            logit_q = float(np.log(q_mean / (1 - q_mean)))

            print(f"  K={K:3d}: q={q_mean:.3f}+/-{q_std:.3f}  "
                  f"kn={kn_mean:.3f}  logit(q)={logit_q:.3f}  log(K-1)={np.log(K-1):.3f}")

            results["data"][str(kappa)].append({
                "K": K,
                "q_mean": q_mean,
                "q_std": q_std,
                "kappa_nearest_mean": kn_mean,
                "logit_q": logit_q,
                "log_K_minus_1": float(np.log(K - 1)),
                "n_valid": len(mc_qs),
            })

            logit_q_list.append(logit_q)
            log_K_list.append(np.log(K - 1))

        # Fit b_eff at this kappa
        if len(logit_q_list) >= 3:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, se = linregress(log_K_list, logit_q_list)
            b_eff = -slope  # slope = -b_eff

            print(f"  -> b_eff = {b_eff:.4f} (slope={slope:.4f}, r={r_value:.4f}, p={p_value:.4f})")
            print(f"     Theoretical: b_eff = 1.0")

            results["b_eff_per_kappa"][str(kappa)] = {
                "b_eff": float(b_eff),
                "slope": float(slope),
                "intercept": float(intercept),
                "r": float(r_value),
                "p": float(p_value),
                "se": float(se),
                "n_K_vals": len(logit_q_list),
            }

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("=" * 70)
    print("SUMMARY: b_eff vs kappa")
    print("=" * 70)

    b_effs = []
    kappas = []
    for kappa_str, res in results["b_eff_per_kappa"].items():
        kappa = float(kappa_str)
        b_eff = res["b_eff"]
        b_effs.append(b_eff)
        kappas.append(kappa)
        print(f"  kappa={kappa:.2f}: b_eff={b_eff:.4f} (r={res['r']:.3f})")

    if len(b_effs) >= 2:
        b_eff_mean = float(np.mean(b_effs))
        b_eff_std = float(np.std(b_effs))
        print(f"\nMean b_eff = {b_eff_mean:.4f} +/- {b_eff_std:.4f}")
        print(f"Theoretical b_eff = 1.0 (Gumbel Race, exact)")
        print(f"Deviation from theory = {b_eff_mean - 1.0:.4f}")

        # Is b_eff constant or kappa-dependent?
        if len(b_effs) >= 3:
            from scipy.stats import pearsonr, spearmanr
            r_kappa = pearsonr(kappas, b_effs)[0]
            rho_kappa = spearmanr(kappas, b_effs).correlation
            print(f"b_eff vs kappa: Pearson r={r_kappa:.3f}, Spearman rho={rho_kappa:.3f}")
            print(f"{'b_eff varies with kappa' if abs(rho_kappa) > 0.7 else 'b_eff CONSTANT across kappa (universal)'}")

        results["summary"] = {
            "b_eff_mean": b_eff_mean,
            "b_eff_std": b_eff_std,
            "b_eff_theoretical": 1.0,
            "deviation": float(b_eff_mean - 1.0),
            "n_kappa_vals": len(b_effs),
        }

    # Save
    out_path = "results/cti_b_eff_direct.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
