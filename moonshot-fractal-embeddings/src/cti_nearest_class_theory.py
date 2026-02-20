#!/usr/bin/env python -u
"""
NEAREST-CLASS THEORY: Fix large-K failure using nearest competitor modeling.

Root cause of large-K failure:
  Current theory uses AVERAGE inter-class distance (delta_sq = 2*kappa*d).
  But kNN quality is determined by the NEAREST competing class, not average.
  With K=200 random centroids, the nearest competitor is MUCH closer than average.

Three models compared:
  1. Original: average delta (current, fails at K>20)
  2. Plug-in: use empirical nearest-class delta (requires data, upper bound on theory)
  3. Analytic EVT: E[delta_min] from CDF F_min(x) = 1 - (1-F_delta(x))^(K-1)

For random isotropic centroids mu_k ~ N(0, sigma_B^2 I):
  ||mu_i - mu_j||^2 ~ Gamma(d/2, 4*sigma_B^2) [since 2*sigma_B^2 * chi^2(d)/d]
  Actually: sum of d iid (N(0,sigma_B^2) - N(0,sigma_B^2))^2 = sum of d iid N(0, 2*sigma_B^2)^2
         = 2*sigma_B^2 * chi^2(d)
  So delta_sq ~ 2*kappa*chi^2(d)/d * d = 2*kappa * chi^2(d) [wrong]

  Actually: let X = mu_i ~ N(0, sigma_B^2 I_d), Y = mu_j ~ N(0, sigma_B^2 I_d)
  X - Y ~ N(0, 2*sigma_B^2 I_d)
  ||X-Y||^2 = 2*sigma_B^2 * chi^2(d)  [where sigma_B^2 = kappa * sigma_W^2 = kappa]
  E[||X-Y||^2] = 2*kappa*d, Var = 4*kappa^2 * 2d = 8*kappa^2*d

  For large d: chi^2(d)/d -> 1, so ||X-Y||^2 -> 2*kappa*d (concentrated!)
  The coefficient of variation = sqrt(8*kappa^2*d) / (2*kappa*d) = sqrt(2/d) -> 0

KEY INSIGHT: In high-d, inter-class distances concentrate around 2*kappa*d.
  But with K-1 classes, the MINIMUM over K-1 concentrated values still departs:
  E[delta_min] = 2*kappa*d - sigma_delta * E[|min of K-1 N(0,1)|]
               = 2*kappa*d - sqrt(8*kappa^2*d) * z_{1/(K-1+1)}
               where z_p = Phi^{-1}(p) (the p-quantile of N(0,1))

  For K=200: z_{1/200} ~ -3.5, so delta_min = 2*kappa*d + 3.5*sqrt(8*kappa^2*d)
  Wait, minimum is the SMALLEST, so delta_min ~ 2*kappa*d + sqrt(8*kappa^2*d) * z_{1/K}
  where z_{1/K} < 0 for K > 1, so delta_min < 2*kappa*d.

  E[min of K-1 iid N(mu, sigma^2)] = mu + sigma * E[min of K-1 iid N(0,1)]
  E[min of m iid N(0,1)] = -E[max of m iid N(0,1)] ~ -sqrt(2*log(m)) for large m

So the nearest-class delta_sq: E[delta_min] = 2*kappa*d - sqrt(8*kappa^2*d) * sqrt(2*log(K-1))
This can be MUCH smaller than 2*kappa*d when K is large!

Success criterion: cut K=100,200 error by >50% while preserving small-K fit.
"""

import json
import sys
import numpy as np
from scipy import stats
from scipy.special import ndtri
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def theory_q_from_delta(delta_sq, K, n_per_class, d_eff):
    """Zero-parameter theory given nearest-class delta_sq.

    Uses conditional (shared query point) variance:
    - D_same | x ~ N(2d, 6d)
    - D_other | x from nearest class ~ N(2d + delta_sq, 6d + 4*delta_sq)
    where delta_sq = nearest competitor squared distance.
    """
    if d_eff < 2 or delta_sq <= 0:
        return 0.0

    n = max(int(n_per_class), 2)
    m = max(int(n_per_class), 2)  # only nearest class (1 * n_per_class)

    # Same-class distances (conditional on shared query)
    mu_s = 2.0 * d_eff
    sigma_s = np.sqrt(6.0 * d_eff)

    # Nearest-class distances (conditional)
    mu_o = 2.0 * d_eff + delta_sq
    sigma_o = np.sqrt(6.0 * d_eff + 4.0 * delta_sq)

    # Order statistics (Normal approx, expected minimum)
    p_n = 1.0 / (n + 1)
    z_n = ndtri(max(p_n, 1e-10))
    phi_z_n = max(stats.norm.pdf(z_n), 1e-20)
    mu_s_min = mu_s + sigma_s * z_n
    tau_s = sigma_s / (n * phi_z_n)

    # Min of n samples from nearest class
    p_m = 1.0 / (m + 1)
    z_m = ndtri(max(p_m, 1e-10))
    phi_z_m = max(stats.norm.pdf(z_m), 1e-20)
    mu_o_min = mu_o + sigma_o * z_m
    tau_o = sigma_o / (m * phi_z_m)

    mu_M = mu_o_min - mu_s_min
    sigma_M = np.sqrt(tau_o**2 + tau_s**2)

    if sigma_M < 1e-20:
        return 1.0 if mu_M > 0 else 0.0

    z = mu_M / sigma_M
    q = float(stats.norm.cdf(z))
    return float(np.clip(q, 0.0, 1.0))


def theory_q_average(kappa, K, n_per_class, d_eff):
    """Original theory: uses average inter-class distance (FAILS at large K)."""
    delta_sq = 2.0 * kappa * d_eff  # average
    if d_eff < 2 or delta_sq <= 0:
        return 0.0

    n = max(int(n_per_class), 2)
    m = max(int((K - 1) * n_per_class), 2)  # all other classes

    mu_s = 2.0 * d_eff
    sigma_s = np.sqrt(6.0 * d_eff)
    mu_o = 2.0 * d_eff + delta_sq
    sigma_o = np.sqrt(6.0 * d_eff + 4.0 * delta_sq)

    p_n = 1.0 / (n + 1)
    z_n = ndtri(max(p_n, 1e-10))
    phi_z_n = max(stats.norm.pdf(z_n), 1e-20)
    mu_s_min = mu_s + sigma_s * z_n
    tau_s = sigma_s / (n * phi_z_n)

    p_m = 1.0 / (m + 1)
    z_m = ndtri(max(p_m, 1e-10))
    phi_z_m = max(stats.norm.pdf(z_m), 1e-20)
    mu_o_min = mu_o + sigma_o * z_m
    tau_o = sigma_o / (m * phi_z_m)

    mu_M = mu_o_min - mu_s_min
    sigma_M = np.sqrt(tau_o**2 + tau_s**2)
    if sigma_M < 1e-20:
        return 1.0 if mu_M > 0 else 0.0
    z = mu_M / sigma_M
    return float(np.clip(stats.norm.cdf(z), 0.0, 1.0))


def analytic_nearest_delta(kappa, K, d_eff):
    """Analytic expected nearest-class delta_sq (no free parameters).

    For random centroids mu_k ~ N(0, kappa * I_d):
      delta_sq = ||mu_i - mu_j||^2 ~ 2*kappa * chi^2(d)
      Mean delta_sq = 2*kappa*d
      Std delta_sq = sqrt(4*kappa^2 * 2d) = 2*kappa*sqrt(2d)

    Nearest of (K-1) deltas (all iid, approximated as Normal in high-d):
      E[delta_min] ~ mu_delta + sigma_delta * z_{1/K}
      where z_{1/K} = Phi^{-1}(1/K) -- expected minimum quantile
    """
    if K <= 1:
        return 2.0 * kappa * d_eff

    mu_delta = 2.0 * kappa * d_eff
    sigma_delta = 2.0 * kappa * np.sqrt(2.0 * d_eff)

    # Expected minimum of (K-1) iid ~ N(mu_delta, sigma_delta^2)
    # E[min] = mu + sigma * E[min of K-1 iid N(0,1)]
    # Use expected-order-statistic approximation: E[X_(1:m)] ~ mu + sigma * Phi^{-1}(1/(m+1))
    m = K - 1
    p_min = 1.0 / (m + 1)
    z_min = ndtri(max(p_min, 1e-15))

    delta_min = mu_delta + sigma_delta * z_min
    return max(float(delta_min), 0.01)  # can't be negative


def analytic_nearest_delta_gamma(kappa, K, d_eff):
    """More accurate: use Gamma CDF for delta_sq distribution.

    ||mu_i - mu_j||^2 = 2*kappa * chi^2(d)
    chi^2(d) ~ Gamma(d/2, 2)
    so delta_sq ~ Gamma(d/2, 4*kappa) [shape=d/2, scale=4*kappa]

    For min of (K-1) iid Gamma(a, b):
    F_min(x) = 1 - (1-F_gamma(x; a, b))^(K-1)
    E[delta_min] = integral_0^inf P(delta_min > x) dx
                 = integral_0^inf (1 - F_gamma(x; a, b))^(K-1) dx
    """
    if K <= 1:
        return 2.0 * kappa * d_eff

    a = d_eff / 2.0  # shape
    b = 4.0 * kappa   # scale

    # Numerical integration
    # E[delta_min] = sum_x P(delta_min > x)
    # = sum_x (1 - F_gamma(x; a, b))^(K-1)
    # Integrate from 0 to ~5*mean
    mean_delta = a * b  # = 2*kappa*d
    x_max = 5.0 * mean_delta
    n_pts = 500
    xs = np.linspace(0, x_max, n_pts + 1)[1:]  # exclude 0
    dx = xs[1] - xs[0]

    survival_gamma = 1.0 - stats.gamma.cdf(xs, a=a, scale=b)
    survival_min = survival_gamma ** (K - 1)
    e_delta_min = float(np.sum(survival_min) * dx)

    return max(e_delta_min, 0.01)


# ============================================================
# EXPERIMENT 1: Controlled K sweep with homogeneous centroids
# ============================================================
print("=" * 60)
print("EXP 1: Homogeneous centroids (equidistant, iid assumption HOLDS)")
print("=" * 60)

def run_knn_experiment(K, n_per_class, d, sigma_B, seed=42, homogeneous=False):
    """Generate synthetic Gaussians, compute kappa, run kNN, return (kappa, q_obs)."""
    rng = np.random.RandomState(seed)
    n_total = K * n_per_class
    d_eff = d

    if homogeneous:
        # Put class means on vertices of a regular simplex scaled to have
        # approximately the same average pairwise distance as random
        # Random centroids: E[delta^2] = 2*sigma_B^2*d
        # Simplex: average pairwise distance^2 = 2*sigma_B^2*d/(K/(K-1)) ~ 2*sigma_B^2*d
        # Use random orthogonal construction for K <= d+1
        target_delta_sq = 2 * sigma_B**2 * d
        # Generate K points with equal pairwise distance = target_delta_sq
        # Simple: place on scaled orthonormal basis
        k_use = min(K, d)
        means = np.zeros((K, d))
        r = np.sqrt(target_delta_sq * K / (2 * (K - 1)))  # radius for equidistant
        for k in range(K):
            if k < k_use:
                means[k, k] = r
            # else duplicate dimensions (still equidistant in 2d subspace)
        # Better: random equidistant points via Gram-Schmidt
        # Use random unit vectors, then scale to equal pairwise distance
        if K <= d:
            # QR of (d x K) matrix gives orthonormal columns Q of shape (d, K)
            basis = rng.randn(d, K)
            Q, _ = np.linalg.qr(basis)
            means = Q[:, :K].T * r  # shape (K, d): each row is orthonormal * r
            means -= means.mean(0)  # center
    else:
        means = rng.randn(K, d) * sigma_B

    # Generate data
    X = []
    y = []
    for k in range(K):
        samples = rng.randn(n_per_class, d) + means[k]
        X.append(samples)
        y.extend([k] * n_per_class)
    X = np.vstack(X)
    y = np.array(y)

    # Compute kappa = trace(S_B) / trace(S_W)
    grand_mean = X.mean(0)
    S_B = np.zeros(d)
    S_W = np.zeros(d)
    for k in range(K):
        mask = y == k
        Xk = X[mask]
        diff = means[k] - grand_mean
        S_B += n_per_class * diff**2
        S_W += ((Xk - means[k])**2).sum(0)
    kappa = float(S_B.sum() / S_W.sum())

    # kNN accuracy (LOO-kNN with k=1)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    preds = knn.predict(X)  # LOO would be better but slow; use fit-predict for speed
    # For a proper LOO: use leave-one-out
    from sklearn.model_selection import cross_val_score
    score = cross_val_score(KNeighborsClassifier(n_neighbors=1), X, y, cv=5).mean()
    knn_acc = float(score)
    q_obs = (knn_acc - 1.0/K) / (1.0 - 1.0/K)

    # Compute empirical nearest-class delta
    pairwise_dist_sq = []
    for i in range(K):
        for j in range(i+1, K):
            pairwise_dist_sq.append(float(np.sum((means[i] - means[j])**2)))
    avg_delta_sq = float(np.mean(pairwise_dist_sq))

    # Nearest class per class
    nearest_deltas = []
    for i in range(K):
        dists = [np.sum((means[i] - means[j])**2) for j in range(K) if j != i]
        nearest_deltas.append(min(dists))
    nearest_delta_sq = float(np.mean(nearest_deltas))  # avg over classes

    return {
        "kappa": kappa,
        "q_obs": q_obs,
        "avg_delta_sq": avg_delta_sq,
        "nearest_delta_sq": nearest_delta_sq,
        "d_eff": d_eff,
        "n_per_class": n_per_class,
        "K": K,
    }


print("\n[Homogeneous centroids]")
print(f"{'K':>5} {'kappa':>8} {'q_obs':>7} {'q_avg':>7} {'q_near_emp':>11} {'q_near_anl':>11} {'q_near_gam':>11}")
K_list = [5, 10, 20, 50, 100, 200]
sigma_B = 0.3
n_total = 2000
d = 300

results_homo = []
results_hetero = []

for K in K_list:
    n_per = max(int(n_total / K), 5)
    row_h = run_knn_experiment(K, n_per, d, sigma_B, homogeneous=True)
    kappa = row_h["kappa"]
    d_eff = row_h["d_eff"]

    q_avg = theory_q_average(kappa, K, n_per, d_eff)
    q_near_emp = theory_q_from_delta(row_h["nearest_delta_sq"], K, n_per, d_eff)
    delta_anl = analytic_nearest_delta(kappa, K, d_eff)
    delta_gam = analytic_nearest_delta_gamma(kappa, K, d_eff)
    q_near_anl = theory_q_from_delta(delta_anl, K, n_per, d_eff)
    q_near_gam = theory_q_from_delta(delta_gam, K, n_per, d_eff)

    q_obs = row_h["q_obs"]
    print(f"{K:>5} {kappa:>8.4f} {q_obs:>7.3f} {q_avg:>7.3f} {q_near_emp:>11.3f} {q_near_anl:>11.3f} {q_near_gam:>11.3f}")
    results_homo.append({
        "K": K, "kappa": kappa, "q_obs": q_obs,
        "q_avg": q_avg, "q_near_emp": q_near_emp,
        "q_near_anl": q_near_anl, "q_near_gam": q_near_gam,
        "nearest_delta_sq": row_h["nearest_delta_sq"],
        "delta_anl": delta_anl, "delta_gam": delta_gam,
        "avg_delta_sq": row_h["avg_delta_sq"],
    })
    sys.stdout.flush()

print("\n[Heterogeneous centroids]")
print(f"{'K':>5} {'kappa':>8} {'q_obs':>7} {'q_avg':>7} {'q_near_emp':>11} {'q_near_anl':>11} {'q_near_gam':>11}")

for K in K_list:
    n_per = max(int(n_total / K), 5)
    row_r = run_knn_experiment(K, n_per, d, sigma_B, homogeneous=False)
    kappa = row_r["kappa"]
    d_eff = row_r["d_eff"]

    q_avg = theory_q_average(kappa, K, n_per, d_eff)
    q_near_emp = theory_q_from_delta(row_r["nearest_delta_sq"], K, n_per, d_eff)
    delta_anl = analytic_nearest_delta(kappa, K, d_eff)
    delta_gam = analytic_nearest_delta_gamma(kappa, K, d_eff)
    q_near_anl = theory_q_from_delta(delta_anl, K, n_per, d_eff)
    q_near_gam = theory_q_from_delta(delta_gam, K, n_per, d_eff)

    q_obs = row_r["q_obs"]
    print(f"{K:>5} {kappa:>8.4f} {q_obs:>7.3f} {q_avg:>7.3f} {q_near_emp:>11.3f} {q_near_anl:>11.3f} {q_near_gam:>11.3f}")
    results_hetero.append({
        "K": K, "kappa": kappa, "q_obs": q_obs,
        "q_avg": q_avg, "q_near_emp": q_near_emp,
        "q_near_anl": q_near_anl, "q_near_gam": q_near_gam,
        "nearest_delta_sq": row_r["nearest_delta_sq"],
        "delta_anl": delta_anl, "delta_gam": delta_gam,
        "avg_delta_sq": row_r["avg_delta_sq"],
    })
    sys.stdout.flush()


# ============================================================
# SCORECARD
# ============================================================
print("\n" + "=" * 60)
print("SCORECARD")
print("=" * 60)

def mae(results, key):
    errors = [abs(r["q_obs"] - r[key]) for r in results]
    return float(np.mean(errors))

def score_reduction(results, key_new, key_base, large_K_only=True):
    """How much does new model reduce error vs baseline at large K?"""
    rows = [r for r in results if r["K"] >= 50] if large_K_only else results
    if not rows:
        return 0.0
    err_base = np.mean([abs(r["q_obs"] - r[key_base]) for r in rows])
    err_new = np.mean([abs(r["q_obs"] - r[key_new]) for r in rows])
    if err_base < 1e-10:
        return 0.0
    return float(1.0 - err_new / err_base)

checks = []
for label, results in [("Homogeneous", results_homo), ("Heterogeneous", results_hetero)]:
    for model, key in [("NearEmp", "q_near_emp"), ("NearAnl", "q_near_anl"), ("NearGam", "q_near_gam")]:
        mae_all = mae(results, key)
        mae_base = mae(results, "q_avg")
        red = score_reduction(results, key, "q_avg")
        passed = red > 0.5
        checks.append({
            "label": label,
            "model": model,
            "mae": mae_all,
            "mae_baseline": mae_base,
            "large_K_reduction": red,
            "passed": passed,
        })
        print(f"  {label:14s} {model:10s}: MAE={mae_all:.3f} baseline={mae_base:.3f} large-K-reduction={red:.1%} {'PASS' if passed else 'fail'}")

passes = sum(1 for c in checks if c["passed"])
total = len(checks)
print(f"\n{passes}/{total} checks passed")

results_out = {
    "homogeneous": results_homo,
    "heterogeneous": results_hetero,
    "scorecard": {
        "checks": checks,
        "passes": passes,
        "total": total,
    }
}

out_path = RESULTS_DIR / "cti_nearest_class_theory.json"
out_path.write_text(json.dumps(results_out, indent=2))
print(f"\nResults saved to {out_path.name}")
