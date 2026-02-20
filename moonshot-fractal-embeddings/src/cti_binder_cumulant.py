#!/usr/bin/env python -u
"""
BINDER CUMULANT: Test for true phase transition in kNN classification.

Codex protocol: if this is a genuine phase transition, Binder cumulant U4
curves for different K should cross at a SINGLE kappa_c.

DEFINITIONS:
  m = q (normalized kNN quality, 0 at chance, 1 at perfect)
  U4 = 1 - <m^4> / (3 * <m^2>^2)   [Binder cumulant]

  For true 2nd-order transition:
    U4 -> 0 for t < 0 (disordered: m -> 0 in thermodynamic limit)
    U4 -> 2/3 for t > 0 (ordered: m -> const)
    All U4 curves CROSS at t = 0 (at kappa_c)

  For crossover (no true transition):
    No universal crossing point
    U4 curves converge but don't cross sharply

THERMODYNAMIC LIMIT: K -> infinity with d/K and n/K fixed.

CONTROL PARAMETER: kappa_nearest (nearest class separation), normalized as
  t = kappa_nearest / kappa_c - 1
  We sweep kappa by varying sigma_B (class mean spread) directly.

KEY PREDICTION: If true phase transition exists, the kappa_c should satisfy:
  q = 0.5 at kappa_nearest = kappa_c
  q ~ |t|^beta for small t (critical exponent beta)
  chi ~ K^{gamma/nu} at kappa_c (diverging susceptibility)
  Crossing of U4 curves at universal kappa_c/sqrt(K)
"""

import json
import sys
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(0)

# ============================================================
# PARAMETERS
# ============================================================
K_vals = [10, 20, 40, 80]  # System sizes (classes)
D_PER_K = 20              # d = D_PER_K * K (fixed ratio)
N_PER = 50                # samples per class (fixed)
N_SEEDS = 40              # disorder realizations per (K, kappa) point
SIGMA_W = 1.0             # within-class noise

# Sweep sigma_B to vary kappa (sigma_B^2 = sigma_W^2 * kappa)
# Need to sweep around the transition point kappa_c ~ 0.1
# Use 25 log-spaced sigma_B values
SIGMA_B_VALS = np.logspace(-0.7, 0.3, 25)  # ~0.2 to ~2.0


def knn_quality_single(K, n_per, d, sigma_B, seed):
    """Single realization: draw means, generate data, compute kNN quality."""
    rng = np.random.RandomState(seed)
    # Class means span full d dimensions (rank = d, so kappa_nearest ~= kappa_spec)
    means = rng.randn(K, d) * sigma_B
    X = means[np.repeat(np.arange(K), n_per)] + rng.randn(K * n_per, d) * SIGMA_W
    y = np.repeat(np.arange(K), n_per)

    grand_mean = X.mean(0)
    tr_sb, tr_sw = 0.0, 0.0
    for k in range(K):
        Xk = X[y == k]
        mu_k = means[k]
        tr_sb += n_per * np.sum((mu_k - grand_mean)**2)
        tr_sw += np.sum((Xk - mu_k)**2)
    kappa_spec = tr_sb / max(tr_sw, 1e-10)

    # Nearest-class delta
    dists = []
    for i in range(K):
        nearest = min(np.sum((means[i] - means[j])**2) for j in range(K) if j != i)
        dists.append(nearest)
    kappa_nearest = float(np.mean(dists)) / d

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, y)
    _, idxs = knn.kneighbors(X)
    correct = sum(y[idxs[i, 1]] == y[i] for i in range(K * n_per))
    knn_acc = correct / (K * n_per)
    q = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)

    return float(q), float(kappa_spec), float(kappa_nearest)


print("=" * 70)
print("BINDER CUMULANT: Testing for true phase transition in kNN")
print("=" * 70)
print(f"\nK values: {K_vals}")
print(f"d/K = {D_PER_K}, n_per = {N_PER}, n_seeds = {N_SEEDS}")
print(f"Sweeping sigma_B: {SIGMA_B_VALS[0]:.3f} to {SIGMA_B_VALS[-1]:.3f}\n")

all_results = {}

for K in K_vals:
    d = D_PER_K * K
    print(f"\n[K={K}, d={d}] Running {N_SEEDS} seeds x {len(SIGMA_B_VALS)} sigma_B values...")
    K_results = []

    for sigma_B in SIGMA_B_VALS:
        # Gather q values over multiple seeds
        q_vals = []
        kappa_n_vals = []
        kappa_s_vals = []

        for seed in range(N_SEEDS):
            q, ks, kn = knn_quality_single(K, N_PER, d, sigma_B, seed)
            q_vals.append(q)
            kappa_n_vals.append(kn)
            kappa_s_vals.append(ks)

        q_arr = np.array(q_vals)
        kappa_n_arr = np.array(kappa_n_vals)
        kappa_s_arr = np.array(kappa_s_vals)

        # Moments for Binder cumulant
        m1 = float(np.mean(q_arr))
        m2 = float(np.mean(q_arr**2))
        m4 = float(np.mean(q_arr**4))
        var_m = float(np.var(q_arr))

        # Binder cumulant U4 = 1 - <m^4>/(3*<m^2>^2)
        U4 = 1.0 - m4 / (3.0 * m2**2) if m2 > 1e-10 else 0.0

        # Susceptibility chi = K * Var(m)
        chi = K * var_m

        kn_mean = float(np.mean(kappa_n_arr))
        ks_mean = float(np.mean(kappa_s_arr))
        kappa_true = sigma_B**2 / SIGMA_W**2  # population kappa

        K_results.append({
            "K": K, "d": d, "sigma_B": float(sigma_B),
            "kappa_true": float(kappa_true),
            "kappa_spec": float(ks_mean),
            "kappa_nearest": float(kn_mean),
            "m_mean": m1,
            "m2": m2,
            "m4": m4,
            "var_m": float(var_m),
            "U4": float(U4),
            "chi": float(chi),
            # Key rescaled variable for collapse
            "t_spec": float(ks_mean / np.sqrt(K)),
            "t_nearest": float(kn_mean / np.sqrt(K)),
            "t_true": float(kappa_true / np.sqrt(K)),
        })

        sys.stdout.flush()

    all_results[str(K)] = K_results

    # Print summary for this K
    q_arr_all = [r["m_mean"] for r in K_results]
    sig_B_arr = [r["sigma_B"] for r in K_results]
    U4_arr = [r["U4"] for r in K_results]
    chi_arr = [r["chi"] for r in K_results]
    print(f"  K={K}: m range [{min(q_arr_all):.3f}, {max(q_arr_all):.3f}], "
          f"max chi={max(chi_arr):.3f} at sigma_B={sig_B_arr[np.argmax(chi_arr)]:.3f}, "
          f"U4 range [{min(U4_arr):.3f}, {max(U4_arr):.3f}]")

# ============================================================
# ANALYSIS: Look for Binder cumulant crossing
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS: Binder cumulant crossing")
print("=" * 70)

# Find where U4 crosses 1/3 (midpoint) for each K
print("\nU4 midpoint crossing (U4 ~ 1/3) for each K:")
for K in K_vals:
    K_results = all_results[str(K)]
    # Find kappa_nearest where U4 crosses 0.33
    for i in range(len(K_results) - 1):
        u4_1 = K_results[i]["U4"]
        u4_2 = K_results[i + 1]["U4"]
        if u4_1 < 0.33 < u4_2 or u4_2 < 0.33 < u4_1:
            # Linear interpolation
            t1 = K_results[i]["t_nearest"]
            t2 = K_results[i + 1]["t_nearest"]
            kn_cross = t1 + (0.33 - u4_1) / (u4_2 - u4_1) * (t2 - t1)
            print(f"  K={K:>3}: kappa_nearest/sqrt(K) crossing at t={kn_cross:.4f} (U4 interp)")
            break

# Find where m=0.5 for each K
print("\nTransition midpoint (q=0.5) for each K:")
kappa_c_list = []
for K in K_vals:
    K_results = all_results[str(K)]
    for i in range(len(K_results) - 1):
        m1 = K_results[i]["m_mean"]
        m2 = K_results[i + 1]["m_mean"]
        if m1 < 0.5 < m2 or m2 < 0.5 < m1:
            kn1 = K_results[i]["kappa_nearest"]
            kn2 = K_results[i + 1]["kappa_nearest"]
            kappa_c = kn1 + (0.5 - m1) / (m2 - m1) * (kn2 - kn1)
            kappa_c_list.append((K, kappa_c))
            print(f"  K={K:>3}: kappa_nearest_c = {kappa_c:.4f}, kappa_c/sqrt(K) = {kappa_c/np.sqrt(K):.4f}")
            break

if len(kappa_c_list) >= 2:
    kappa_c_scaled = [(K, kc / np.sqrt(K)) for K, kc in kappa_c_list]
    vals = [v for _, v in kappa_c_scaled]
    print(f"\n  kappa_c/sqrt(K) values: {[f'{v:.4f}' for v in vals]}")
    print(f"  Mean: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    is_universal = np.std(vals) / np.mean(vals) < 0.1 if np.mean(vals) > 0 else False
    print(f"  CV = {np.std(vals)/np.mean(vals):.4f} (< 10% = universal)")
    print(f"  VERDICT: {'UNIVERSAL kappa_c -- supports phase transition!' if is_universal else 'NOT universal -- suggests crossover'}")

# ============================================================
# Susceptibility scaling
# ============================================================
print("\n" + "=" * 70)
print("SUSCEPTIBILITY SCALING: chi_max ~ K^(gamma/nu)?")
print("=" * 70)

chi_max_vals = []
for K in K_vals:
    K_results = all_results[str(K)]
    chi_arr = [r["chi"] for r in K_results]
    chi_max = max(chi_arr)
    chi_max_vals.append((K, chi_max))
    print(f"  K={K:>3}: chi_max = {chi_max:.4f}")

if len(chi_max_vals) >= 3:
    log_K = np.log([K for K, _ in chi_max_vals])
    log_chi = np.log([c for _, c in chi_max_vals])
    slope, intercept, r, p, se = stats.linregress(log_K, log_chi)
    print(f"\n  Power law fit: chi_max ~ K^{slope:.3f} (R={r:.4f}, p={p:.4f})")
    print(f"  gamma/nu = {slope:.3f}")
    if slope > 0.1:
        print("  -> chi_max GROWS with K: supports critical divergence")
    else:
        print("  -> chi_max ~CONSTANT: no critical divergence (crossover only)")

# ============================================================
# Print kappa_nearest vs m table for visual inspection
# ============================================================
print("\n" + "=" * 70)
print("TRANSITION PROFILE (kappa_nearest/sqrt(K) vs m and U4)")
print("=" * 70)

print(f"\n{'t_near':>8}", end="")
for K in K_vals:
    print(f"  K={K}(m)  K={K}(U4)", end="")
print()

# Use shared t-grid
t_grid = np.linspace(0.01, 0.3, 15)

for t_target in t_grid:
    print(f"{t_target:>8.3f}", end="")
    for K in K_vals:
        K_results = all_results[str(K)]
        # Find closest t_nearest value
        ts = [r["t_nearest"] for r in K_results]
        ms = [r["m_mean"] for r in K_results]
        u4s = [r["U4"] for r in K_results]

        # Interpolate
        if t_target <= ts[0]:
            m_val, u4_val = ms[0], u4s[0]
        elif t_target >= ts[-1]:
            m_val, u4_val = ms[-1], u4s[-1]
        else:
            idx = np.searchsorted(ts, t_target)
            t1, t2 = ts[idx - 1], ts[idx]
            frac = (t_target - t1) / (t2 - t1) if t2 > t1 else 0.5
            m_val = ms[idx - 1] + frac * (ms[idx] - ms[idx - 1])
            u4_val = u4s[idx - 1] + frac * (u4s[idx] - u4s[idx - 1])

        print(f"  {m_val:.3f}  {u4_val:.3f}", end="")
    print()

# Save
out = {
    "parameters": {
        "K_vals": K_vals,
        "D_PER_K": D_PER_K,
        "N_PER": N_PER,
        "N_SEEDS": N_SEEDS,
    },
    "K_results": all_results,
    "kappa_c_at_q05": kappa_c_list,
}
out_path = RESULTS_DIR / "cti_binder_cumulant.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nResults saved to {out_path.name}")
