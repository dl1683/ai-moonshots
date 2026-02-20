#!/usr/bin/env python -u
"""
THEORETICAL DERIVATION: Universal function h(r,K) connects kappa_nearest to kappa_spec.

THEOREM: For K random Gaussian class means in R^r with total pairwise spread:
  kappa_nearest = kappa_spec * h(r, K)

where h(r, K) = 2 * E[chi^2(r)_(1:K-1)] / r

  E[chi^2(r)_(1:K-1)] = expected MINIMUM of (K-1) iid chi^2(r) random variables

Derivation:
  Class means mu_k iid N(0, sigma_a^2 I_r) with sigma_a^2 = delta^2/(2r)

  Pairwise distances: ||mu_i - mu_j||^2 = 2*sigma_a^2 * chi^2(r)
  (sum of r iid (N(0,sigma_a) - N(0,sigma_a))^2/sigma_a^2 = chi^2(r))

  kappa_spec = tr(S_B)/tr(S_W) = delta^2/(2d) [CONSTANT, d-independent of r!]

  kappa_nearest = E[min_j ||mu_i - mu_j||^2] / d [average over classes]
               = 2*sigma_a^2 * E[chi^2(r)_(1:K-1)] / d
               = (delta^2/r) * E[chi^2(r)_(1:K-1)] / d
               = kappa_spec * 2 * E[chi^2(r)_(1:K-1)] / r

So: h(r, K) = 2 * E[chi^2(r)_(1:K-1)] / r

LIMITS:
  r >> log(K): chi^2(r) concentrates around r, so E[chi^2_min] ~ r - sqrt(2r)*sqrt(2*log(K))
    h(r, K) -> 2 * (1 - sqrt(4*log(K)/r)) -> 2 as r -> inf

  r small: Weibull extreme value: E[chi^2(r)_min] ~ (K-1)^{-2/r} * Gamma(1+2/r)
    h(r, K) ~ 2*(K-1)^{-2/r} * Gamma(1+2/r) / r -> 0 as r -> 1

This is a ZERO-PARAMETER universal prediction.

COMPLETE UNIVERSAL LAW:
  q = sigmoid(kappa_spec * h(rank(S_B), K) / sqrt(K))

where rank(S_B) = rank of between-class scatter matrix (observable from representations).
"""

import json
import sys
import numpy as np
from scipy import stats
from scipy.special import gamma
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def h_analytic_large_r(r, K):
    """h(r,K) for r >> log(K): Normal approximation."""
    # E[chi^2(r)_min] ~ r + sqrt(2r) * Phi^{-1}(1/(K-1+1))
    from scipy.special import ndtri
    m = K - 1
    if m < 1:
        return 2.0
    p_min = 1.0 / (m + 1)
    z_min = ndtri(max(p_min, 1e-15))  # negative
    E_min = r + np.sqrt(2.0 * r) * z_min
    E_min = max(E_min, 0.01)
    return 2.0 * E_min / r


def h_analytic_weibull(r, K):
    """h(r,K) using Weibull extreme value theory for small r.

    For chi^2(r) with r/2 being the shape parameter:
    F(x) ~ x^{r/2} / (2^{r/2} * Gamma(r/2+1)) for small x (density ~ x^{r/2-1})
    Min of m: E[min] ~ (m * C_r)^{-2/r} where C_r = 1/(2^{r/2}*Gamma(r/2+1))

    Exact: E[min of m iid chi^2(r)] = integral_0^inf P(min>x) dx
         = integral_0^inf (1 - F_chi2_r(x))^m dx
    """
    m = K - 1
    if m < 1:
        return 2.0
    # Numerical integration
    # E[min] = integral_0^inf (1-F(x))^m dx
    # For chi^2(r), support [0, inf)
    # Use Gamma(r/2, 2) distribution
    x_max = float(stats.chi2.ppf(1 - 1e-10, df=r))  # upper limit
    xs = np.linspace(0, x_max, 1000)
    dx = xs[1] - xs[0]
    survival = (1.0 - stats.chi2.cdf(xs, df=r))**m
    E_min = float(np.sum(survival) * dx)
    return 2.0 * E_min / r


def h_numerical(r, K, n_monte_carlo=10000, seed=None):
    """h(r,K) from Monte Carlo: E[chi^2(r)_min]."""
    rng = np.random.RandomState(seed)
    m = K - 1
    samples = rng.chisquare(df=r, size=(n_monte_carlo, m))
    mins = samples.min(axis=1)
    E_min = float(np.mean(mins))
    return 2.0 * E_min / r


print("=" * 70)
print("THEORETICAL DERIVATION: h(r, K) = 2*E[chi^2(r)_min] / r")
print("=" * 70)

# ============================================================
# PART 1: Validate h formula against data from rank sweep
# ============================================================
print("\n[PART 1: Validate h(r,K) against observed kappa_nearest/kappa_spec]")

K = 20
d = 500
delta_pair = 10.0
kappa_spec_true = delta_pair**2 / (2 * d)

print(f"K={K}, d={d}, delta_pair={delta_pair}, kappa_spec={kappa_spec_true:.4f}")
print(f"\n{'r':>4} {'h_obs':>8} {'h_weibull':>11} {'h_large_r':>11} {'h_mc':>8} {'kappa_n_obs':>12} {'kappa_n_pred':>13} {'err%':>6}")

# Load from rank sweep experiment
rank_sweep_data = json.loads((RESULTS_DIR / "cti_rank_missing_variable.json").read_text())

results_part1 = []
for row in rank_sweep_data["rank_sweep"]:
    r = row["rank_r"]
    kappa_n_obs = row["kappa_nearest"]
    kappa_s_obs = row["kappa_spec"]

    # Observed h
    h_obs = kappa_n_obs / kappa_spec_true if kappa_spec_true > 0 else 0.0

    # Theoretical h
    h_w = h_analytic_weibull(r, K)
    h_lr = h_analytic_large_r(r, K)
    h_mc = h_numerical(r, K, seed=42)

    # Predicted kappa_nearest
    kappa_n_pred_w = kappa_spec_true * h_w
    kappa_n_pred_mc = kappa_spec_true * h_mc

    err_pct = abs(kappa_n_obs - kappa_n_pred_mc) / max(kappa_n_obs, 1e-6) * 100

    print(f"{r:>4} {h_obs:>8.4f} {h_w:>11.4f} {h_lr:>11.4f} {h_mc:>8.4f} "
          f"{kappa_n_obs:>12.4f} {kappa_n_pred_mc:>13.4f} {err_pct:>6.1f}%")
    results_part1.append({
        "r": r, "h_obs": h_obs, "h_weibull": h_w, "h_large_r": h_lr, "h_mc": h_mc,
        "kappa_n_obs": kappa_n_obs, "kappa_n_pred": kappa_n_pred_mc,
        "err_pct": err_pct,
    })
    sys.stdout.flush()

mae_abs = np.mean([abs(r["kappa_n_obs"] - r["kappa_n_pred"]) for r in results_part1])
print(f"\nMAE = {mae_abs:.4f}")

# Spearman correlation
h_mc_arr = np.array([r["h_mc"] for r in results_part1])
h_obs_arr = np.array([r["h_obs"] for r in results_part1])
rho, p = stats.spearmanr(h_mc_arr, h_obs_arr)
print(f"Spearman rho(h_mc, h_obs) = {rho:.4f} (p={p:.4f})")

# ============================================================
# PART 2: Complete universal law test
# ============================================================
print("\n" + "=" * 70)
print("[PART 2: Complete Universal Law q = sigmoid(kappa_spec * h(r,K) / sqrt(K))]")
print("=" * 70)

print(f"\n{'r':>4} {'q_obs':>7} {'q_spec':>7} {'q_complete':>11} {'err_spec':>9} {'err_comp':>9}")
for row in rank_sweep_data["rank_sweep"]:
    r = row["rank_r"]
    q_obs = row["q_obs"]
    kappa_s = row["kappa_spec"]
    h_mc = h_numerical(r, K, seed=42)
    kappa_n_pred = kappa_s * h_mc

    q_spec = float(stats.logistic.cdf(kappa_s / np.sqrt(K)))
    q_complete = float(stats.logistic.cdf(kappa_n_pred / np.sqrt(K)))
    print(f"{r:>4} {q_obs:>7.3f} {q_spec:>7.3f} {q_complete:>11.3f} {abs(q_obs-q_spec):>9.3f} {abs(q_obs-q_complete):>9.3f}")

# ============================================================
# PART 3: Vary K - test K-dependence of h
# ============================================================
print("\n" + "=" * 70)
print("[PART 3: K-dependence of h (fixed r=20)]")
print("=" * 70)

r_fixed = 20
K_vals = [5, 10, 20, 50, 100, 200]
print(f"r={r_fixed}, K varies")
print(f"\n{'K':>5} {'h_obs':>8} {'h_mc':>8} {'h_weibull':>11} {'ratio':>7}")

for K_val in K_vals:
    row = next((r for r in rank_sweep_data["K_sweep"] if r["K"] == K_val), None)
    if row is None:
        continue
    kappa_s = row["kappa_spec"]
    kappa_n = row["kappa_nearest"]
    h_obs = kappa_n / kappa_s if kappa_s > 0 else 0

    h_mc = h_numerical(r_fixed, K_val, seed=42)
    h_w = h_analytic_weibull(r_fixed, K_val)
    ratio = h_obs / h_mc if h_mc > 0 else 0

    print(f"{K_val:>5} {h_obs:>8.4f} {h_mc:>8.4f} {h_w:>11.4f} {ratio:>7.3f}")
    sys.stdout.flush()

# ============================================================
# PART 4: The theorem statement
# ============================================================
print("\n" + "=" * 70)
print("THEOREM (ZERO-PARAMETER)")
print("=" * 70)
print("""
For K-class Gaussian mixture with class means mu_k iid N(0, sigma_B^2 I_r)
embedded in R^d (r <= d), within-class noise N(0, sigma_W^2 I_d):

  kappa_nearest = kappa_spec * h(r, K) + o(1/K)

where:
  kappa_spec = tr(S_B) / tr(S_W) [the measurable LDA ratio]
  kappa_nearest = mean(min_j ||mu_i - mu_j||^2) / d [nearest-class SNR]

  h(r, K) = 2 * E[chi2(r)_(1:K-1)] / r

  E[chi2(r)_(1:K-1)] = expected minimum of (K-1) iid chi-squared(r) random variables

COROLLARY (Universal Law):
  q = sigmoid(kappa_spec * h(r, K) / sqrt(K)) [up to scale and offset]

where r = rank(S_B) (the rank of the between-class scatter, measurable from data).

SPECIAL CASES:
  r >> log(K):  h(r,K) -> 2  (high-d, equidistant means, kappa_nearest = 2*kappa_spec)
  r = K-1:      h(K-1,K) = ETF limit, intermediate value
  r -> 1:       h(1,K) ~ 4/(K-1)^2  -> 0 as K grows

IMPLICATION for Neural Networks:
  Real representations have high rank (r >> K for typical tasks).
  Therefore kappa_nearest ~ 2*kappa_spec and the universal law holds.
  kappa_spec IS a valid proxy for kappa_nearest when rank >> K.
""")

# Compute h values for typical neural network regime
print("h(r, K) values for neural network regime (r >> K):")
for K_val in [5, 10, 50, 150, 1000]:
    for r_val in [K_val, 10*K_val, 100*K_val]:
        h = h_numerical(r_val, K_val, n_monte_carlo=5000, seed=42)
        print(f"  K={K_val:>5}, r={r_val:>7}: h={h:.4f}")

# Save
out = {
    "theorem": "kappa_nearest = kappa_spec * h(r,K) where h(r,K) = 2*E[chi^2(r)_min]/r",
    "part1_validation": results_part1,
    "part2_complete_law": "sigmoid(kappa_spec * h(rank(S_B), K) / sqrt(K))",
    "special_cases": {
        "r_gg_logK": "h -> 2 (equidistant, neural net regime)",
        "r_is_1": "h ~ 4/(K-1)^2 (1D case, huge penalty)",
    },
}
out_path = RESULTS_DIR / "cti_theoretical_derivation.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nResults saved to {out_path.name}")
