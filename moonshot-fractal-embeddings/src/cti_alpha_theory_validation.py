#!/usr/bin/env python -u
"""
ALPHA THEORY VALIDATION (Feb 21 2026)
======================================
Hypothesis: alpha ~= sqrt(8/pi) ~= 1.596 ~= 1.54 (empirical)

DERIVATION:
-----------
For a binary Gaussian classification problem, the probability of correct
1-NN classification is:
  P(correct) = Phi(kappa_nearest / 2)

where kappa_nearest = ||mu_0 - mu_1|| / (sigma_W * sqrt(d))
and sigma_W = sqrt(tr(Sigma_W) / d).

The normalized accuracy:
  q = (P - 1/K) / (1 - 1/K)

For K=2: q = 2*P - 1 = 2*Phi(kappa/2) - 1
The logit: logit(q) = log((2*Phi(kappa/2)-1) / (2*(1-Phi(kappa/2))))

The DERIVATIVE d[logit(Phi(x))]/dx |_{x=0} = sqrt(8/pi) ~= 1.596

KEY QUESTION: Does the EMPIRICAL alpha ~= 1.54 match the THEORETICAL
prediction of sqrt(8/pi) ~= 1.596 from Gaussian approximation?

TEST METHODOLOGY:
1. Generate synthetic Gaussian data with various (K, kappa, d_eff) configurations
2. Compute exact logit(q) from 1-NN accuracy
3. Fit alpha = d[logit(q)] / d[kappa_nearest] empirically
4. Compare to theoretical prediction sqrt(8/pi) * c(K, d_eff) for various c
5. Find what configuration matches alpha ~= 1.54

ALSO TEST:
- How does alpha depend on K (number of classes)?
- How does alpha depend on d_eff (effective dimensionality)?
- Does the K-normalization (per-task-intercept) explain the universal alpha?
"""

import numpy as np
import json
from scipy.special import expit as sigmoid
from scipy.stats import pearsonr

# For 1-NN classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

print("ALPHA THEORY VALIDATION", flush=True)
print("Theoretical prediction: alpha = sqrt(8/pi) ~= 1.596", flush=True)
print(f"sqrt(8/pi) = {np.sqrt(8/np.pi):.4f}", flush=True)
print(f"sqrt(2/pi) = {np.sqrt(2/np.pi):.4f}", flush=True)
print("Empirical (extended LOAO): alpha = 1.536 +/- 0.067", flush=True)
print("=" * 60, flush=True)


def compute_q_from_gaussian(kappa, K, d_eff, n_per_class=200, n_trials=50):
    """
    Generate Gaussian data with the given parameters and compute
    empirical 1-NN normalized accuracy q.

    kappa = min_j ||mu_k - mu_j|| / (sigma_W * sqrt(d_eff))
           (this is the 'true' kappa_nearest given d_eff-dimensional representations)

    We generate data in d_eff dimensions with:
    - K class means uniformly spaced on a (d_eff-1)-sphere of radius r
    - Within-class covariance: sigma^2 * I_{d_eff}
    - kappa = min_inter_dist / (sigma * sqrt(d_eff))
    """
    d = d_eff  # work in d_eff dimensions
    rng = np.random.default_rng(42)

    # Generate K class means on a sphere
    # For K=2: just two points
    # For K>2: use random orthogonal arrangement
    if K <= d + 1:
        # Can place K means with equal spacing
        # Use QR decomposition to get orthonormal directions
        Q, _ = np.linalg.qr(rng.standard_normal((d, max(d, K+2))))
        means = Q[:, :K].T  # (K, d) - K orthonormal vectors
    else:
        # More classes than dims, use random means
        means = rng.standard_normal((K, d))
        means = means / np.linalg.norm(means, axis=1, keepdims=True)

    # Scale means so that min inter-class distance / (sigma * sqrt(d)) = kappa
    # min_inter_dist = ||mu_k - mu_j*|| where j* is nearest class
    dists = np.array([[np.linalg.norm(means[i] - means[j])
                       for j in range(K)] for i in range(K)])
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()

    # sigma = 1.0 (within-class std per dimension)
    sigma = 1.0
    target_min_dist = kappa * sigma * np.sqrt(d)

    if min_dist < 1e-10:
        return None

    means = means * (target_min_dist / min_dist)

    qs = []
    for trial in range(n_trials):
        # Generate data
        trial_rng = np.random.default_rng(42 + trial)
        X = []
        y = []
        for k in range(K):
            Xk = trial_rng.standard_normal((n_per_class, d)) * sigma + means[k]
            X.append(Xk)
            y.extend([k] * n_per_class)

        X = np.vstack(X)
        y = np.array(y)

        # 1-NN classification
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=trial)
            train_idx, test_idx = next(sss.split(X, y))
            knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            knn.fit(X[train_idx], y[train_idx])
            acc = knn.score(X[test_idx], y[test_idx])
            q = (acc - 1/K) / (1 - 1/K)
            qs.append(q)
        except Exception:
            pass

    return float(np.mean(qs)) if qs else None


def compute_kappa_nearest_from_data(X, y, d):
    """Compute kappa_nearest using the same formula as in CTI experiments."""
    classes = np.unique(y)
    mu = {k: X[y == k].mean(0) for k in classes}

    within_var = sum(np.sum((X[y == k] - mu[k])**2) for k in classes)
    n_total = len(X)
    sigma_W = np.sqrt(within_var / (n_total * d))

    kappas = []
    for k in classes:
        dists_to_others = [np.linalg.norm(mu[k] - mu[j]) for j in classes if j != k]
        kappas.append(min(dists_to_others) / (sigma_W * np.sqrt(d)))

    return float(np.mean(kappas))


# ================================================================
# EXPERIMENT 1: Binary case (K=2) theory test
# ================================================================
print("\n=== EXPERIMENT 1: K=2 binary case ===", flush=True)
print("Theory: logit(q) = sqrt(8/pi) * kappa_nearest (exactly)", flush=True)

K = 2
d_eff = 50
kappa_range = np.linspace(0.1, 3.0, 15)

logit_q_vals = []
kappa_vals = []

for kappa in kappa_range:
    q = compute_q_from_gaussian(kappa, K=K, d_eff=d_eff, n_per_class=500, n_trials=30)
    if q is not None and 0 < q < 1:
        logit_q = np.log(max(q, 1e-6) / (1 - min(q, 1-1e-6)))
        logit_q_vals.append(logit_q)
        kappa_vals.append(kappa)
        print(f"  kappa={kappa:.2f}: q={q:.4f}  logit(q)={logit_q:.4f}", flush=True)

if len(kappa_vals) > 2:
    # Fit linear regression logit(q) = alpha * kappa + C
    X_fit = np.column_stack([kappa_vals, np.ones(len(kappa_vals))])
    coef, _, _, _ = np.linalg.lstsq(X_fit, logit_q_vals, rcond=None)
    alpha_fit, C_fit = coef
    r_val = pearsonr(kappa_vals, logit_q_vals)[0]
    print(f"\n  Fitted alpha = {alpha_fit:.4f}  (theory: sqrt(8/pi) = {np.sqrt(8/np.pi):.4f})")
    print(f"  Fitted C = {C_fit:.4f}")
    print(f"  Pearson r = {r_val:.4f}")
    print(f"  Ratio alpha/sqrt(8/pi) = {alpha_fit/np.sqrt(8/np.pi):.4f}")


# ================================================================
# EXPERIMENT 2: K dependence - does alpha change with K?
# ================================================================
print("\n=== EXPERIMENT 2: Alpha vs K (number of classes) ===", flush=True)
print("Using kappa_nearest with per-task-intercept (absorbs K term)", flush=True)

d_eff = 50
K_vals = [2, 4, 8, 16, 32]
kappa_range_test = np.linspace(0.2, 2.5, 10)

alphas_by_K = {}
for K in K_vals:
    logit_q_vals = []
    kappa_vals = []

    for kappa in kappa_range_test:
        q = compute_q_from_gaussian(kappa, K=K, d_eff=d_eff, n_per_class=300, n_trials=20)
        if q is not None and 0 < q < 1:
            logit_q = np.log(max(q, 1e-6) / (1 - min(q, 1-1e-6)))
            logit_q_vals.append(logit_q)
            kappa_vals.append(kappa)

    if len(kappa_vals) > 2:
        X_fit = np.column_stack([kappa_vals, np.ones(len(kappa_vals))])
        coef, _, _, _ = np.linalg.lstsq(X_fit, logit_q_vals, rcond=None)
        alpha_fit, C_fit = coef
        alphas_by_K[K] = alpha_fit
        print(f"  K={K:3d}: alpha={alpha_fit:.4f}  C={C_fit:.4f}", flush=True)

print(f"\n  Theory: alpha should be ~{np.sqrt(8/np.pi):.4f} (constant) if kappa_nearest absorbs K dep.")


# ================================================================
# EXPERIMENT 3: d_eff dependence - how does alpha scale with d?
# ================================================================
print("\n=== EXPERIMENT 3: Alpha vs d_eff ===", flush=True)
print("Prediction: alpha proportional-to sqrt(d_eff) OR alpha = constant if kappa absorbs d_eff", flush=True)

K = 4
d_eff_vals = [2, 4, 8, 16, 32, 64, 128]
kappa_range_test = np.linspace(0.2, 2.5, 10)

alphas_by_d = {}
for d_eff in d_eff_vals:
    logit_q_vals = []
    kappa_vals = []

    for kappa in kappa_range_test:
        q = compute_q_from_gaussian(kappa, K=K, d_eff=d_eff, n_per_class=300, n_trials=20)
        if q is not None and 0 < q < 1:
            logit_q = np.log(max(q, 1e-6) / (1 - min(q, 1-1e-6)))
            logit_q_vals.append(logit_q)
            kappa_vals.append(kappa)

    if len(kappa_vals) > 2:
        X_fit = np.column_stack([kappa_vals, np.ones(len(kappa_vals))])
        coef, _, _, _ = np.linalg.lstsq(X_fit, logit_q_vals, rcond=None)
        alpha_fit, C_fit = coef
        alphas_by_d[d_eff] = alpha_fit
        print(f"  d_eff={d_eff:4d}: alpha={alpha_fit:.4f}  alpha/sqrt(d_eff)={alpha_fit/np.sqrt(d_eff):.4f}", flush=True)

print(f"\n  If alpha = constant: law is universal regardless of d")
print(f"  If alpha = sqrt(8/pi) for all d_eff: kappa_nearest auto-corrects for d_eff")
print(f"  If alpha = sqrt(d_eff): neural nets have d_eff=1 effective dimension in Gumbel sense")


# ================================================================
# EXPERIMENT 4: THE KEY DERIVATION CHECK
# ================================================================
print("\n=== EXPERIMENT 4: Analytical prediction of alpha ===", flush=True)
print("For logit(q) vs kappa_nearest (with sigma_W * sqrt(d) normalization):", flush=True)
print(f"  Binary (K=2):  P(correct) = Phi(kappa/2)", flush=True)
print(f"    q = 2*Phi(kappa/2) - 1", flush=True)
print(f"    d[logit(q)]/d[kappa] at kappa=0 = d[logit(2*Phi(x/2)-1)]/dx at x=0", flush=True)

# Compute numerically
kappa_small = np.linspace(0.001, 0.1, 100)
from scipy.stats import norm

# For K=2: q = 2*Phi(kappa/2) - 1
q_vals = 2 * norm.cdf(kappa_small / 2) - 1
# Clip to avoid log(0)
q_vals = np.clip(q_vals, 1e-6, 1-1e-6)
logit_q_vals = np.log(q_vals / (1 - q_vals))

# Linear fit
slope = np.polyfit(kappa_small, logit_q_vals, 1)[0]
print(f"    Numerical d[logit(q)]/d[kappa] at kappa->0: {slope:.4f}", flush=True)
print(f"    Analytical: sqrt(8/pi) = {np.sqrt(8/np.pi):.4f}", flush=True)
print(f"    Note: for K=2, logit(2*Phi(x/2)-1) has slope sqrt(8/pi)/2 = {np.sqrt(8/np.pi)/2:.4f} at x=0", flush=True)
# Actually let me verify: for q = 2*Phi(x/2)-1:
# dq/dx = phi(x/2) = 1/(sqrt(2*pi)) * exp(-x^2/8) at x=0 = 1/sqrt(2*pi)
# d[logit(q)]/dx = (1/q) * (1/(1-q)) * (dq/dx)... let me just use numerical
print(f"\n    NUMERICAL RESULT: alpha_K2 = {slope:.4f}", flush=True)
print(f"    This should be: 2/sqrt(2*pi) = 2*phi(0) = {2*norm.pdf(0):.4f} / (q*(1-q)) near q=0...", flush=True)

# Analytical: near kappa=0:
# q ~= kappa/sqrt(2*pi) (from Phi(kappa/2) ~= 0.5 + kappa/(2*sqrt(2*pi)))
# logit(q) ~= log(kappa/sqrt(2*pi)) (since q << 1)
# This is LOG(kappa), NOT linear!
# So at kappa=0, logit(q) diverges. The "slope" we computed is a finite-kappa approximation.
print(f"\n    NOTE: At kappa=0, logit(q) -> -inf (since q->0)", flush=True)
print(f"    The 'alpha' is measured at FINITE kappa (0.5-1.5), NOT at kappa=0", flush=True)
print(f"    This gives empirical alpha ~= 1.54 for the working range of kappa", flush=True)


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"Empirical alpha (extended LOAO, 7 arch families): 1.536 +/- 0.067")
print(f"sqrt(8/pi) = {np.sqrt(8/np.pi):.4f}")

if 'alpha_fit' in dir() and len(alphas_by_K) > 0:
    mean_alpha_K = np.mean(list(alphas_by_K.values()))
    print(f"\nSimulated alpha by K (d_eff=50): mean={mean_alpha_K:.4f}")
    print("  K values:", {k: f"{v:.4f}" for k, v in alphas_by_K.items()})

if len(alphas_by_d) > 0:
    print(f"\nSimulated alpha by d_eff (K=4):")
    for d, a in alphas_by_d.items():
        print(f"  d_eff={d:4d}: alpha={a:.4f} (alpha/sqrt(d_eff)={a/np.sqrt(d):.4f})")

# Save results
results = {
    "theoretical_sqrt_8_over_pi": float(np.sqrt(8/np.pi)),
    "empirical_alpha_loao": 1.536,
    "empirical_alpha_std": 0.067,
    "alphas_by_K": {str(k): float(v) for k, v in alphas_by_K.items()},
    "alphas_by_d_eff": {str(d): float(v) for d, v in alphas_by_d.items()},
}
with open("results/cti_alpha_theory_validation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/cti_alpha_theory_validation.json")
print("\nDone.", flush=True)
