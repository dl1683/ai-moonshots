#!/usr/bin/env python -u
"""
DIST_RATIO THEORY: Derive the relationship between dist_ratio and kappa.

dist_ratio = E[min_{j!=i} d(x, NN_j)] / E[min_{k=i} d(x, NN_k)]
           = the ratio of nearest inter-class to nearest intra-class distance

HYPOTHESIS: dist_ratio = 1 + C * kappa_nearest for some constant C(K, d, n)
or more precisely: dist_ratio = sqrt(1 + kappa_nearest * g(K, d, n))

KEY QUESTION: Is dist_ratio = f(kappa_nearest) with a simple functional form?
If YES: dist_ratio is a MORE DIRECT order parameter than kappa_spec.
If YES + derivable from first principles: THEOREM about training dynamics.

EXPERIMENT:
1. Generate K-class Gaussians with varying kappa_nearest
2. Compute dist_ratio, kappa_spec, kappa_nearest
3. Fit dist_ratio = f(kappa_nearest) and compare to q = sigmoid(...)
4. Compare predictive power: kappa_spec vs dist_ratio for q
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit, minimize
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)

print("=" * 70)
print("DIST_RATIO THEORY: dist_ratio = f(kappa_nearest)?")
print("=" * 70)


def h_rK(r, K):
    """h(r, K) = 2 * E[chi^2(r)_min_(K-1)] / r."""
    r = int(r)
    K = int(K)
    m = K - 1
    if m < 1:
        return 2.0
    x_max = float(stats.chi2.ppf(1 - 1e-10, df=r))
    xs = np.linspace(0, x_max, 2000)
    dx = xs[1] - xs[0]
    survival = (1.0 - stats.chi2.cdf(xs, df=r)) ** m
    return 2.0 * float(np.sum(survival) * dx) / r


def generate_gaussian_data(K, d, r, sigma_b, sigma_w, n_per=50, seed=None):
    """Generate K-class Gaussian data."""
    if seed is not None:
        np.random.seed(seed)
    # Class means: r effective dimensions
    means = np.zeros((K, d))
    means[:, :r] = np.random.randn(K, r) * sigma_b
    # Samples
    X = []
    y = []
    for k in range(K):
        samples = means[k] + np.random.randn(n_per, d) * sigma_w
        X.append(samples)
        y.extend([k] * n_per)
    return np.vstack(X), np.array(y), means


def compute_metrics(X, y, K, sample_size=200):
    """Compute kappa_spec, kappa_nearest, dist_ratio, knn_q."""
    n, d = X.shape
    labels = y
    unique_labels = sorted(set(labels))

    # Scatter matrices
    grand_mean = X.mean(0)
    S_B = np.zeros((d, d))
    S_W = np.zeros((d, d))
    centroids = []
    for k in unique_labels:
        mask = labels == k
        mu_k = X[mask].mean(0)
        centroids.append(mu_k)
        n_k = mask.sum()
        diff = (mu_k - grand_mean).reshape(-1, 1)
        S_B += n_k * (diff @ diff.T)
        centered = X[mask] - mu_k
        S_W += centered.T @ centered

    tr_SB = float(np.trace(S_B))
    tr_SW = float(np.trace(S_W))
    kappa_spec = float(tr_SB / max(tr_SW, 1e-10))

    centroids = np.array(centroids)

    # kappa_nearest: mean minimum inter-centroid distance
    min_inter_dists = []
    for i, mu_i in enumerate(centroids):
        dists = np.sum((centroids - mu_i) ** 2, axis=1)
        dists[i] = np.inf
        min_inter_dists.append(np.min(dists))
    kappa_nearest_raw = float(np.mean(min_inter_dists)) / d  # normalized by d

    # Effective rank of S_B
    eigvals = np.linalg.eigvalsh(S_B)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) > 0:
        tr_SB2 = float(np.sum(eigvals**2))
        eff_rank = float(tr_SB**2 / tr_SB2) if tr_SB2 > 1e-15 else 1.0
    else:
        eff_rank = 1.0

    h = h_rK(max(int(eff_rank), 1), K)
    kappa_nearest_theorem = kappa_spec * h

    # dist_ratio: nearest inter/intra class distance ratio
    idx = np.random.choice(n, min(sample_size, n), replace=False)
    same_dists = []
    diff_dists = []
    for i in idx:
        dists = np.sum((X - X[i])**2, axis=1)
        dists[i] = np.inf
        same_mask = labels == labels[i]
        diff_mask = labels != labels[i]
        same_mask[i] = False
        if same_mask.any():
            same_dists.append(np.min(dists[same_mask]))
        if diff_mask.any():
            diff_dists.append(np.min(dists[diff_mask]))

    if same_dists and diff_dists:
        dist_ratio = float(np.mean(diff_dists) / max(np.mean(same_dists), 1e-10))
    else:
        dist_ratio = 1.0

    # kNN accuracy
    from sklearn.neighbors import KNeighborsClassifier
    n_test = min(n // 5, 200)
    idx_all = np.random.permutation(n)
    test_idx = idx_all[:n_test]
    train_idx = idx_all[n_test:]
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(X[train_idx], labels[train_idx])
    acc = float(knn.score(X[test_idx], labels[test_idx]))
    q = max(min((acc - 1.0/K) / (1.0 - 1.0/K), 0.999), 0.001)

    return {
        "kappa_spec": kappa_spec,
        "kappa_nearest_raw": kappa_nearest_raw,
        "kappa_nearest_theorem": kappa_nearest_theorem,
        "eff_rank": eff_rank,
        "h": h,
        "dist_ratio": dist_ratio,
        "q": q,
        "knn_acc": acc,
        "K": K,
        "d": d,
        "n": n,
    }


# ============================================================
# EXPERIMENT 1: Vary kappa with fixed K, d — study dist_ratio vs kappa
# ============================================================

print("\nEXPERIMENT 1: dist_ratio vs kappa (fixed K=50, d=200, r=10)")
print("-" * 60)
K_fixed = 50
d_fixed = 200
r_fixed = 10
sigma_w_fixed = 1.0
sigma_b_values = np.logspace(-1.5, 0.5, 12)  # wide range
n_per_fixed = 80

exp1_results = []
for i, sigma_b in enumerate(sigma_b_values):
    X_i, y_i, _ = generate_gaussian_data(K_fixed, d_fixed, r_fixed, sigma_b, sigma_w_fixed,
                                          n_per=n_per_fixed, seed=i)
    result = compute_metrics(X_i, y_i, K_fixed)
    exp1_results.append(result)
    print(f"  sigma_b={sigma_b:.3f}: kappa={result['kappa_spec']:.4f}, "
          f"kappa_near_th={result['kappa_nearest_theorem']:.4f}, "
          f"dist_ratio={result['dist_ratio']:.4f}, q={result['q']:.4f}")

# Fit dist_ratio = f(kappa_spec)
kappas_e1 = np.array([r["kappa_spec"] for r in exp1_results])
drs_e1 = np.array([r["dist_ratio"] for r in exp1_results])
qs_e1 = np.array([r["q"] for r in exp1_results])
kn_e1 = np.array([r["kappa_nearest_theorem"] for r in exp1_results])

# Try: dist_ratio = 1 + a * kappa_spec
def linear_model(x, a): return 1 + a * x
try:
    popt, _ = curve_fit(linear_model, kappas_e1, drs_e1)
    r2_lin_kappa = 1 - np.sum((drs_e1 - linear_model(kappas_e1, *popt))**2) / np.sum((drs_e1 - drs_e1.mean())**2)
    print(f"\n  dist_ratio ~ 1 + {popt[0]:.3f}*kappa_spec: R2={r2_lin_kappa:.4f}")
except:
    r2_lin_kappa = 0

# Try: dist_ratio = 1 + a * kappa_nearest
try:
    popt2, _ = curve_fit(linear_model, kn_e1, drs_e1)
    r2_lin_kn = 1 - np.sum((drs_e1 - linear_model(kn_e1, *popt2))**2) / np.sum((drs_e1 - drs_e1.mean())**2)
    print(f"  dist_ratio ~ 1 + {popt2[0]:.3f}*kappa_nearest: R2={r2_lin_kn:.4f}")
    a_dr_kn = float(popt2[0])
except:
    r2_lin_kn = 0
    a_dr_kn = 1.0

# Try: dist_ratio = sqrt(1 + a * kappa_spec)
def sqrt_model(x, a): return np.sqrt(1 + a * x)
try:
    popt3, _ = curve_fit(sqrt_model, kappas_e1, drs_e1, p0=[1.0])
    r2_sqrt_kappa = 1 - np.sum((drs_e1 - sqrt_model(kappas_e1, *popt3))**2) / np.sum((drs_e1 - drs_e1.mean())**2)
    print(f"  dist_ratio ~ sqrt(1 + {popt3[0]:.3f}*kappa_spec): R2={r2_sqrt_kappa:.4f}")
except:
    r2_sqrt_kappa = 0

# Try: dist_ratio = kappa_spec^a
def power_model(x, a, b): return a * x**b
try:
    popt4, _ = curve_fit(power_model, kappas_e1[kappas_e1>0.05], drs_e1[kappas_e1>0.05], p0=[2.0, 0.5])
    r2_pow = 1 - np.sum((drs_e1[kappas_e1>0.05] - power_model(kappas_e1[kappas_e1>0.05], *popt4))**2) / np.sum((drs_e1[kappas_e1>0.05] - drs_e1[kappas_e1>0.05].mean())**2)
    print(f"  dist_ratio ~ {popt4[0]:.3f}*kappa^{popt4[1]:.3f}: R2={r2_pow:.4f}")
except Exception as e:
    r2_pow = 0
    print(f"  Power model failed: {e}")

sys.stdout.flush()


# ============================================================
# EXPERIMENT 2: Vary K with fixed kappa — study dist_ratio scaling with K
# ============================================================

print("\nEXPERIMENT 2: dist_ratio vs K (fixed kappa~0.3, d=200, r=10)")
print("-" * 60)
K_values = [5, 10, 20, 50, 100, 200]
d_fixed2 = 200
r_fixed2 = 10
sigma_w_fixed2 = 1.0
sigma_b_target = 0.5  # moderate kappa
n_per_fixed2 = 80

exp2_results = []
for K in K_values:
    X_k, y_k, _ = generate_gaussian_data(K, d_fixed2, r_fixed2, sigma_b_target,
                                          sigma_w_fixed2, n_per=n_per_fixed2, seed=K)
    result = compute_metrics(X_k, y_k, K)
    exp2_results.append(result)
    print(f"  K={K:4d}: kappa={result['kappa_spec']:.4f}, kappa_near={result['kappa_nearest_theorem']:.4f}, "
          f"dist_ratio={result['dist_ratio']:.4f}, q={result['q']:.4f}")

# Does dist_ratio decrease with log(K)?
Ks_e2 = np.array([r["K"] for r in exp2_results], float)
drs_e2 = np.array([r["dist_ratio"] for r in exp2_results])
qs_e2 = np.array([r["q"] for r in exp2_results])
kappas_e2 = np.array([r["kappa_spec"] for r in exp2_results])
kn_e2 = np.array([r["kappa_nearest_theorem"] for r in exp2_results])

# For fixed kappa, dist_ratio as function of K:
from scipy.stats import spearmanr
rho_dr_logK, p1 = spearmanr(drs_e2, -np.log(Ks_e2))
rho_dr_sqrtK, p2 = spearmanr(drs_e2, -np.sqrt(Ks_e2))
rho_dr_K, p3 = spearmanr(drs_e2, -Ks_e2)
rho_q_logK, _ = spearmanr(qs_e2, -np.log(Ks_e2))
rho_q_sqrtK, _ = spearmanr(qs_e2, -np.sqrt(Ks_e2))
print(f"\n  rho(dist_ratio, -log(K)) = {rho_dr_logK:.4f}")
print(f"  rho(dist_ratio, -sqrt(K)) = {rho_dr_sqrtK:.4f}")
print(f"  rho(dist_ratio, -K) = {rho_dr_K:.4f}")
print(f"  rho(q, -log(K)) = {rho_q_logK:.4f}")
print(f"  rho(q, -sqrt(K)) = {rho_q_sqrtK:.4f}")
print(f"  -> dist_ratio more correlated with log(K) or sqrt(K): {'log' if abs(rho_dr_logK) > abs(rho_dr_sqrtK) else 'sqrt'}")

sys.stdout.flush()


# ============================================================
# EXPERIMENT 3: Joint (kappa, K) prediction of dist_ratio
# ============================================================

print("\nEXPERIMENT 3: Joint prediction dist_ratio = f(kappa_nearest, K)")
print("-" * 60)

all_results = []
K_range = [10, 20, 50, 100, 150]
sigma_b_range = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
d_e3 = 200
r_e3 = 10
n_per_e3 = 60

for K in K_range:
    for sigma_b in sigma_b_range:
        X_e, y_e, _ = generate_gaussian_data(K, d_e3, r_e3, sigma_b, 1.0,
                                              n_per=n_per_e3, seed=K*100+int(sigma_b*100))
        result = compute_metrics(X_e, y_e, K)
        all_results.append(result)

print(f"  Total: {len(all_results)} data points")

kappas_e3 = np.array([r["kappa_spec"] for r in all_results])
kn_e3 = np.array([r["kappa_nearest_theorem"] for r in all_results])
drs_e3 = np.array([r["dist_ratio"] for r in all_results])
qs_e3 = np.array([r["q"] for r in all_results])
Ks_e3 = np.array([float(r["K"]) for r in all_results])

# Model A: dist_ratio = 1 + a * kappa_nearest (no K term)
def model_a(X, a): return 1 + a * X[0]
try:
    popt_a, _ = curve_fit(model_a, [kn_e3], drs_e3, p0=[1.0])
    pred_a = model_a([kn_e3], *popt_a)
    r2_a = 1 - np.sum((drs_e3 - pred_a)**2) / np.sum((drs_e3 - drs_e3.mean())**2)
    print(f"  Model A (dist_ratio = 1 + a*kappa_near): R2={r2_a:.4f}, a={popt_a[0]:.3f}")
except Exception as e:
    print(f"  Model A failed: {e}")
    r2_a = 0

# Model B: dist_ratio = 1 + a * kappa_nearest + b * log(K)
def model_b(X, a, b): return 1 + a * X[0] + b * np.log(X[1])
try:
    popt_b, _ = curve_fit(model_b, [kn_e3, Ks_e3], drs_e3, p0=[1.0, -0.1])
    pred_b = model_b([kn_e3, Ks_e3], *popt_b)
    r2_b = 1 - np.sum((drs_e3 - pred_b)**2) / np.sum((drs_e3 - drs_e3.mean())**2)
    print(f"  Model B (dist_ratio = 1 + a*kn + b*log(K)): R2={r2_b:.4f}, a={popt_b[0]:.3f}, b={popt_b[1]:.3f}")
except Exception as e:
    print(f"  Model B failed: {e}")
    r2_b = 0

# Model C: dist_ratio = 1 + a * kappa_nearest / K^gamma
def model_c(X, a, gamma): return 1 + a * X[0] / X[1]**gamma
try:
    popt_c, _ = curve_fit(model_c, [kn_e3, Ks_e3], drs_e3, p0=[1.0, 0.5])
    pred_c = model_c([kn_e3, Ks_e3], *popt_c)
    r2_c = 1 - np.sum((drs_e3 - pred_c)**2) / np.sum((drs_e3 - drs_e3.mean())**2)
    print(f"  Model C (dist_ratio = 1 + a*kn/K^gamma): R2={r2_c:.4f}, a={popt_c[0]:.3f}, gamma={popt_c[1]:.3f}")
except Exception as e:
    print(f"  Model C failed: {e}")
    r2_c = 0

# Now compare: q ~ dist_ratio vs q ~ kappa_nearest
valid = (qs_e3 > 0.05) & (qs_e3 < 0.95)
if valid.sum() > 5:
    from scipy.stats import spearmanr as sp
    rho_q_dr, _ = sp(qs_e3[valid], drs_e3[valid])
    rho_q_kn, _ = sp(qs_e3[valid], kn_e3[valid])
    rho_q_ks, _ = sp(qs_e3[valid], kappas_e3[valid])
    print(f"\n  Within (K,sigma_b) variation:")
    print(f"  rho(q, dist_ratio) = {rho_q_dr:.4f}")
    print(f"  rho(q, kappa_nearest) = {rho_q_kn:.4f}")
    print(f"  rho(q, kappa_spec) = {rho_q_ks:.4f}")
    print(f"  -> Best predictor of q: {'dist_ratio' if rho_q_dr >= max(rho_q_kn, rho_q_ks) else ('kappa_nearest' if rho_q_kn >= rho_q_ks else 'kappa_spec')}")

sys.stdout.flush()


# ============================================================
# THEORY PREDICTION: For isotropic Gaussian, dist_ratio = ?
# ============================================================

print("\nTHEORETICAL DERIVATION:")
print("-" * 60)
print("""
For K-class isotropic Gaussian (within-class sigma_W^2*I_d, r effective signal dims):
  kappa_spec = (K-1)/K * r * sigma_B^2 / (d * sigma_W^2)  [scatter ratio]
  kappa_nearest = kappa_spec * h(r, K)                     [corrected for EVT]

WITHIN-CLASS nearest neighbor (n_per samples per class):
  E[min_{k=i} ||x - x_k||^2] = d * sigma_W^2 * f_intra(d, n_per)
  where f_intra ~ 1 - 2*log(n_per)/d (approximately)
  For large d: f_intra ~ 1 (concentrates)

CROSS-CLASS nearest neighbor (K-1 other classes):
  E[min_{j!=i} ||x - x_j||^2] = d * sigma_W^2 * f_inter(d, n_per, K) + sigma_B^2 * r * g(r, K)
  where g(r, K) = E[min_K chi^2(r)] = r - sqrt(2r*log(K)) + O(log(log(K)))

DIST_RATIO = E[cross] / E[intra]
  ~ (d * sigma_W^2 + sigma_B^2 * r * g(r, K)) / (d * sigma_W^2)
  = 1 + (sigma_B^2 * r / (d * sigma_W^2)) * g(r, K)
  = 1 + kappa_spec * g(r, K)

where g(r, K) = E[min of K-1 chi^2(r)] / r = h(r,K)/2 ... approximately.

PREDICTION: dist_ratio = 1 + C * kappa_nearest
where C is a constant close to 1 (depending on n_per, d).

This explains why dist_ratio > kappa as predictor:
- kappa_spec is a GLOBAL ratio (tr(S_B)/tr(S_W))
- dist_ratio is a SAMPLE ratio (nearest-neighbor inter/intra)
- dist_ratio directly measures what kNN uses for classification
- For training dynamics: kappa grows monotonically (more between-class),
  but dist_ratio may not if within-class clusters also SHRINK simultaneously
""")
sys.stdout.flush()


# ============================================================
# SAVE RESULTS
# ============================================================

out = {
    "experiment": "dist_ratio_theory",
    "experiment1": {
        "description": "dist_ratio vs kappa (fixed K=50, d=200)",
        "r2_linear_kappa": float(r2_lin_kappa),
        "r2_linear_kappa_nearest": float(r2_lin_kn),
        "r2_sqrt_kappa": float(r2_sqrt_kappa),
        "r2_power_kappa": float(r2_pow),
        "data": exp1_results,
    },
    "experiment2": {
        "description": "dist_ratio vs K (fixed kappa~0.3)",
        "rho_dr_logK": float(rho_dr_logK),
        "rho_dr_sqrtK": float(rho_dr_sqrtK),
        "data": exp2_results,
    },
    "experiment3": {
        "description": "joint (kappa, K) prediction of dist_ratio",
        "r2_model_a": float(r2_a),
        "r2_model_b": float(r2_b),
        "r2_model_c": float(r2_c),
        "rho_q_dist_ratio": float(rho_q_dr) if valid.sum() > 5 else None,
        "rho_q_kappa_nearest": float(rho_q_kn) if valid.sum() > 5 else None,
        "rho_q_kappa_spec": float(rho_q_ks) if valid.sum() > 5 else None,
        "data": all_results,
    },
    "theory_prediction": "dist_ratio = 1 + C * kappa_nearest (C near 1)",
}

out_path = RESULTS_DIR / "cti_dist_ratio_theory.json"
with open(out_path, "w") as f:
    import json as js
    js.dump(out, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

print(f"\nSaved: {out_path.name}")
