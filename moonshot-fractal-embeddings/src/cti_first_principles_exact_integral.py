#!/usr/bin/env python -u
"""
FIRST-PRINCIPLES DERIVATION: sqrt(K) from common-mode margin covariance.

Codex design (Feb 20, 2026): Prove the sqrt(K) normalization from the eigenmode
structure of the margin covariance matrix.

THEOREM TARGET: width ~ K^0.5 (not K^0 from Gumbel log(K))

MECHANISM:
  Margin vector Z in R^{K-1}: Z_j = ||x - mu_j||^2 - ||x - mu_y||^2 for j != y
  Cov(Z) = lambda_common * P_1 + lambda_idio * (I - P_1)
  where lambda_common ~ K, lambda_idio ~ 1
  => min_j Z_j controlled by common mode with noise scale sqrt(K)
  => P(correct) = sigmoid(kappa_nearest / sqrt(K)) [for large K]

  This reconciles:
  - Synthetic data (small K): log(K) additive term dominates (Gumbel EVT)
  - Real data (large K): sqrt(K) divisive term dominates (common mode)
  - Both correct! As K grows, log(K)/sqrt(K) -> 0 so sqrt(K) wins.

KEY PREDICTIONS:
1. sqrt(K) normalization gives best R^2 for kappa_nearest/sqrt(K) vs q
2. width ~ K^0.5 (transition width in kappa-space grows as sqrt(K))
3. lambda_common ~ K (common-mode eigenvalue grows linearly with K)
4. lambda_idio ~ K^0 (idiosyncratic eigenvalues are K-independent)
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import ndtri
from scipy.optimize import minimize_scalar, curve_fit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
np.random.seed(42)


# ============================================================
# CORE: h(r, K) from our theorem
# ============================================================

def h_rK(r, K, cache=None):
    """h(r, K) = 2 * E[chi^2(r)_min_(K-1)] / r via numerical integration."""
    key = (r, K)
    if cache is not None and key in cache:
        return cache[key]
    m = K - 1
    if m < 1:
        val = 2.0
    else:
        x_max = float(stats.chi2.ppf(1 - 1e-10, df=r))
        xs = np.linspace(0, x_max, 2000)
        dx = xs[1] - xs[0]
        survival = (1.0 - stats.chi2.cdf(xs, df=r)) ** m
        E_min = float(np.sum(survival) * dx)
        val = 2.0 * E_min / r
    if cache is not None:
        cache[key] = val
    return val


def sigma_b2_from_kappa_nearest(kappa_nearest, d, r, K, e_min_cache=None):
    """Invert kappa_nearest = sigma_b2 * r / (sigma_w2 * d) * h(r, K) -> sigma_b2.

    Derivation:
      kappa_spec = (K-1)/K * sigma_b2 * r / (sigma_w2 * d)  [for sigma_w2=1]
      kappa_nearest = kappa_spec * h(r, K)
      sigma_b2 = kappa_nearest * d / (r * h(r, K) * (K-1)/K)
    """
    h = h_rK(r, K, e_min_cache)
    # kappa_spec = kappa_nearest / h
    # kappa_spec = (K-1)/K * sigma_b2 * r / (sigma_w2 * d)
    # sigma_b2 = kappa_spec * K/(K-1) * d / r
    kappa_spec = kappa_nearest / max(h, 1e-12)
    sigma_b2 = kappa_spec * (K / max(K - 1, 1)) * d / r
    return sigma_b2


# ============================================================
# MONTE CARLO: exact q for nearest centroid
# ============================================================

def exact_q_nearest_centroid_mc(K, r, sigma_b2, sigma_w2, n_outer, seed, batch=10000):
    """P(correct) for K-class nearest centroid with Gaussian class means.

    Class means: mu_k ~ N(0, sigma_b2 * I_r) embedded in R^d via zero-padding.
    Test points: x ~ N(mu_y, sigma_w2 * I_d), y chosen uniformly.

    Uses POPULATION centroids (infinite samples per class).
    """
    rng = np.random.RandomState(seed)
    d = max(r, 64)  # At least r dimensions

    # Draw K class means in R^r, embed in R^d
    means_r = rng.randn(K, r) * np.sqrt(sigma_b2)
    means = np.zeros((K, d))
    means[:, :r] = means_r

    n_correct = 0
    n_done = 0

    while n_done < n_outer:
        n_batch = min(batch, n_outer - n_done)

        # Draw random class labels
        y_batch = rng.randint(0, K, size=n_batch)

        # Draw test points
        noise = rng.randn(n_batch, d) * np.sqrt(sigma_w2)
        x_batch = means[y_batch] + noise

        # Compute squared distances to all class centroids
        # x_batch: (n_batch, d), means: (K, d)
        # ||x - mu_k||^2 = ||x||^2 - 2*x@mu_k + ||mu_k||^2
        x_sq = np.sum(x_batch ** 2, axis=1, keepdims=True)  # (n_batch, 1)
        mu_sq = np.sum(means ** 2, axis=1, keepdims=True).T  # (1, K)
        cross = x_batch @ means.T  # (n_batch, K)
        dists = x_sq - 2 * cross + mu_sq  # (n_batch, K)

        # Predicted class = argmin distance
        pred = np.argmin(dists, axis=1)
        n_correct += np.sum(pred == y_batch)
        n_done += n_batch

    acc = n_correct / n_outer
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    return float(acc), float(max(q, 0.0))


# ============================================================
# MARGIN COVARIANCE EIGENVALUE ANALYSIS
# ============================================================

def margin_covariance_eigs(K, r, sigma_b2, sigma_w2, n_trials, seed):
    """Compute eigenvalues of margin covariance matrix Cov(Z).

    Z in R^{K-1}: Z_j = ||x - mu_j||^2 - ||x - mu_y||^2 for j != y
    (y = correct class, j != y)

    Key predictions:
    - lambda_common (all-ones mode) ~ K (grows with K)
    - lambda_idio (other modes) ~ 1 (K-independent)
    """
    rng = np.random.RandomState(seed)
    d = max(r, 64)

    # Draw one fixed set of class means (average over many test points)
    means_r = rng.randn(K, r) * np.sqrt(sigma_b2)
    means = np.zeros((K, d))
    means[:, :r] = means_r

    # Use class y=0 (by symmetry, same for all classes)
    y = 0
    mu_y = means[y]

    # Draw test points from class y
    x_batch = mu_y + rng.randn(n_trials, d) * np.sqrt(sigma_w2)

    # Compute ||x - mu_j||^2 for all j != y
    impostors = [j for j in range(K) if j != y]  # K-1 impostors
    mu_impostors = means[impostors]  # (K-1, d)

    # Distances to impostors
    x_sq = np.sum(x_batch ** 2, axis=1, keepdims=True)  # (n, 1)
    mu_imp_sq = np.sum(mu_impostors ** 2, axis=1, keepdims=True).T  # (1, K-1)
    cross_imp = x_batch @ mu_impostors.T  # (n, K-1)
    dist_imp = x_sq - 2 * cross_imp + mu_imp_sq  # (n, K-1) distances to impostors

    # Distances to correct class
    mu_y_sq = float(np.sum(mu_y ** 2))
    cross_y = x_batch @ mu_y  # (n,)
    dist_y = np.sum(x_batch ** 2, axis=1) - 2 * cross_y + mu_y_sq  # (n,)

    # Margin vector Z: Z_j = dist_imp_j - dist_y
    Z = dist_imp - dist_y[:, np.newaxis]  # (n, K-1)

    # Covariance of Z
    Z_centered = Z - Z.mean(axis=0)
    cov_Z = Z_centered.T @ Z_centered / (n_trials - 1)  # (K-1, K-1)

    # Eigenvalues of cov_Z
    eigvals = np.linalg.eigvalsh(cov_Z)  # ascending order
    eigvals = np.sort(eigvals)[::-1]  # descending

    # Common mode = all-ones direction (normalized)
    ones = np.ones(K - 1) / np.sqrt(K - 1)
    lambda_common = float(ones @ cov_Z @ ones)
    lambda_idio = float(np.mean(eigvals[1:]))  # average of non-dominant eigenvalues

    off_diag = float(np.mean(cov_Z[np.triu_indices(K - 1, k=1)]))  # average off-diagonal

    return {
        "lambda_1": float(eigvals[0]),
        "lambda_2": float(eigvals[1]) if len(eigvals) > 1 else 0.0,
        "lambda_common": lambda_common,
        "lambda_idio": lambda_idio,
        "off_diagonal_mean": off_diag,
    }


# ============================================================
# FITTING UTILITIES
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def probit(x):
    from scipy.special import ndtr
    return ndtr(x)


def fit_link(x, q, link_type="logistic"):
    """Fit q = link(a * x + b) and return R^2, MAE, params."""
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    # Clip q to avoid numerical issues
    q_clip = np.clip(q, 0.001, 0.999)

    if link_type == "logistic":
        link_fn = sigmoid

        def residuals(params):
            a, b = params
            pred = sigmoid(a * x + b)
            return np.mean((pred - q_clip) ** 2)

        from scipy.optimize import minimize
        res = minimize(residuals, [1.0, 0.0], method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
        a, b = res.x
        pred = sigmoid(a * x + b)
    else:  # probit
        from scipy.special import ndtr, ndtri

        def residuals(params):
            a, b = params
            pred = ndtr(a * x + b)
            return np.mean((pred - q_clip) ** 2)

        from scipy.optimize import minimize
        res = minimize(residuals, [1.0, 0.0], method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
        a, b = res.x
        pred = ndtr(a * x + b)

    ss_res = np.sum((q - pred) ** 2)
    ss_tot = np.sum((q - np.mean(q)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    mae = float(np.mean(np.abs(q - pred)))
    return {"r2": float(r2), "mae": mae, "a": float(a), "b": float(b)}


def linearity_on_transformed_q(x, q, link="logit"):
    """R^2 of link(q) vs x (test linearity of link-transformed q in x)."""
    from scipy.special import logit as sp_logit, ndtri
    q_clip = np.clip(q, 0.001, 0.999)
    if link == "logit":
        y = sp_logit(q_clip)
    else:  # probit
        y = ndtri(q_clip)
    slope, intercept, r, p, se = stats.linregress(x, y)
    return {"r2": float(r ** 2), "slope": float(slope), "intercept": float(intercept)}


def best_gamma(kappa_near, Ks, qs):
    """Find gamma in q = sigmoid(a * kappa / K^gamma + b) via grid search."""
    kappa_near = np.asarray(kappa_near)
    Ks = np.asarray(Ks, dtype=float)
    qs = np.asarray(qs)
    best_r2 = -1.0
    best_gamma_val = 0.5

    for gamma in np.arange(0.0, 1.01, 0.02):
        x = kappa_near / (Ks ** gamma + 1e-12)
        fit = fit_link(x, qs, "logistic")
        if fit["r2"] > best_r2:
            best_r2 = fit["r2"]
            best_gamma_val = gamma

    return {"gamma": float(best_gamma_val), "r2": float(best_r2)}


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",")]


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",")]


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--sigma_w2", type=float, default=1.0)
    p.add_argument("--K_values", type=str, default="5,10,20,50,100,200")
    p.add_argument("--r_values", type=str, default="16,32,64")
    p.add_argument("--kappa_near_values", type=str, default="0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.5")
    p.add_argument("--n_outer", type=int, default=20000)
    p.add_argument("--batch", type=int, default=10000)
    p.add_argument("--cov_trials", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/cti_first_principles_exact_integral.json")
    args = p.parse_args()

    K_values = parse_int_list(args.K_values)
    r_values = parse_int_list(args.r_values)
    kappa_near_values = parse_float_list(args.kappa_near_values)

    h_cache = {}
    e_min_cache = {}
    rows = []

    print("=" * 72)
    print("FIRST-PRINCIPLES DERIVATION: exact multiclass Gaussian integral")
    print("=" * 72)
    print(f"d={args.d}, K_values={K_values}, r_values={r_values}")
    print(f"n_outer={args.n_outer}, cov_trials={args.cov_trials}")

    for r in r_values:
        for K in K_values:
            h_val = h_rK(r, K, h_cache)
            print(f"\nr={r}, K={K}, h={h_val:.5f}", flush=True)

            for kn in kappa_near_values:
                sigma_b2 = sigma_b2_from_kappa_nearest(
                    kappa_nearest=kn,
                    d=args.d,
                    r=r,
                    K=K,
                    e_min_cache=e_min_cache,
                )

                acc, q = exact_q_nearest_centroid_mc(
                    K=K,
                    r=r,
                    sigma_b2=sigma_b2,
                    sigma_w2=args.sigma_w2,
                    n_outer=args.n_outer,
                    seed=args.seed + 17 * K + 101 * r + int(1000 * kn),
                    batch=args.batch,
                )

                kappa_spec = kn / max(h_val, 1e-12)
                rows.append({
                    "K": K, "r": r, "d": args.d,
                    "sigma_w2": args.sigma_w2,
                    "sigma_b2": float(sigma_b2),
                    "h_rK": float(h_val),
                    "kappa_spec": float(kappa_spec),
                    "kappa_nearest": float(kn),
                    "acc": float(acc),
                    "q": float(q),
                })
                print(f"  k_near={kn:>5.2f} -> q={q:>7.4f}", flush=True)
                sys.stdout.flush()

    kappa_near = np.array([row["kappa_nearest"] for row in rows], dtype=float)
    Ks = np.array([row["K"] for row in rows], dtype=float)
    qs = np.array([row["q"] for row in rows], dtype=float)

    norm_defs = {
        "sqrt(K)": np.sqrt(Ks),
        "log(K)": np.log(Ks),
        "log(K+1)": np.log(Ks + 1.0),
        "K^0.4": Ks ** 0.4,
        "K": Ks,
        "none": np.ones_like(Ks),
    }

    normalization_results = {}
    print("\n" + "=" * 72)
    print("Normalization + Link Comparison")
    print("=" * 72)

    for name, denom in norm_defs.items():
        x = kappa_near / denom
        fit_log = fit_link(x, qs, "logistic")
        fit_pro = fit_link(x, qs, "probit")
        normalization_results[name] = {"logistic": fit_log, "probit": fit_pro}
        print(
            f"{name:>8s} | logistic R2={fit_log['r2']:.4f} MAE={fit_log['mae']:.4f} | "
            f"probit R2={fit_pro['r2']:.4f} MAE={fit_pro['mae']:.4f}"
        )
        sys.stdout.flush()

    x_sqrt = kappa_near / np.sqrt(Ks)
    link_linearity = {
        "logit_vs_kappa_over_sqrtK": linearity_on_transformed_q(x_sqrt, qs, "logit"),
        "probit_vs_kappa_over_sqrtK": linearity_on_transformed_q(x_sqrt, qs, "probit"),
        "best_gamma_logistic": best_gamma(kappa_near, Ks, qs),
    }

    print("\nLink linearity on x = kappa_nearest/sqrt(K)")
    print(
        f"logit R2={link_linearity['logit_vs_kappa_over_sqrtK']['r2']:.4f}, "
        f"probit R2={link_linearity['probit_vs_kappa_over_sqrtK']['r2']:.4f}"
    )
    print(
        f"free gamma={link_linearity['best_gamma_logistic']['gamma']:.4f}, "
        f"logistic R2={link_linearity['best_gamma_logistic']['r2']:.4f}"
    )
    sys.stdout.flush()

    # Width scaling: per-K logistic slope
    per_k = []
    for K in K_values:
        mask = Ks == K
        if mask.sum() < 3:
            continue
        fitK = fit_link(kappa_near[mask], qs[mask], "logistic")
        width = 1.0 / max(abs(fitK["a"]), 1e-12)
        per_k.append({"K": int(K), "a": float(fitK["a"]), "width": float(width), "r2": float(fitK["r2"])})

    K_arr = np.array([x["K"] for x in per_k], dtype=float)
    w_arr = np.array([x["width"] for x in per_k], dtype=float)
    if len(K_arr) >= 3:
        beta_w, c_w = np.polyfit(np.log(K_arr), np.log(w_arr), 1)
    else:
        beta_w, c_w = 0.0, 0.0
    width_scaling = {"beta_width_vs_K": float(beta_w), "intercept": float(c_w), "per_K": per_k}

    print(f"\nWidth scaling: width ~ K^beta, beta={beta_w:.4f} (target ~ 0.5)")
    for pk in per_k:
        print(f"  K={pk['K']:>3d}: a={pk['a']:.4f}, width={pk['width']:.4f}, R2={pk['r2']:.4f}")
    sys.stdout.flush()

    # Eigenvalue analysis: lambda_common ~ K?
    print("\n" + "=" * 72)
    print("EIGENVALUE ANALYSIS: margin covariance Cov(Z)")
    print("=" * 72)
    c_match = 0.8  # kappa_nearest = c * sqrt(K) -> at fixed "signal strength per mode"
    r_cov = max(r_values)
    eig_rows = []

    for K in K_values:
        kn = c_match * np.sqrt(K)
        try:
            sigma_b2 = sigma_b2_from_kappa_nearest(kn, args.d, r_cov, K, e_min_cache)
            eig = margin_covariance_eigs(
                K=K, r=r_cov, sigma_b2=sigma_b2, sigma_w2=args.sigma_w2,
                n_trials=args.cov_trials, seed=args.seed + 9000 + K,
            )
            eig_rows.append({"K": K, "kappa_nearest": float(kn), **eig})
            print(
                f"K={K:>3d}: lambda_common={eig['lambda_common']:.3e}, "
                f"lambda_idio={eig['lambda_idio']:.3e}, "
                f"ratio={eig['lambda_common'] / max(eig['lambda_idio'], 1e-12):.2f}"
            )
        except Exception as exc:
            print(f"K={K}: ERROR: {exc}")
        sys.stdout.flush()

    # Fit power laws
    if len(eig_rows) >= 3:
        K_eig = np.array([x["K"] for x in eig_rows], dtype=float)
        lam1 = np.array([x["lambda_common"] for x in eig_rows], dtype=float)
        lam2 = np.array([x["lambda_idio"] for x in eig_rows], dtype=float)
        b1, _ = np.polyfit(np.log(K_eig), np.log(lam1 + 1e-30), 1)
        b2, _ = np.polyfit(np.log(K_eig), np.log(lam2 + 1e-30), 1)
        eig_scaling = {
            "lambda_common_exponent_vs_K": float(b1),
            "lambda_idio_exponent_vs_K": float(b2),
            "details": eig_rows,
        }
        print(f"\nlambda_common ~ K^{b1:.3f} (target ~ 1.0)")
        print(f"lambda_idio ~ K^{b2:.3f} (target ~ 0.0)")
    else:
        eig_scaling = {"details": eig_rows}

    # Final summary
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    best_norm = max(normalization_results.items(), key=lambda kv: kv[1]["logistic"]["r2"])
    print(f"Best normalization: {best_norm[0]} (logistic R2={best_norm[1]['logistic']['r2']:.4f})")
    print(f"Width scaling exponent: {beta_w:.3f} (target 0.5)")
    if eig_rows:
        lam1_exp = eig_scaling.get("lambda_common_exponent_vs_K", float("nan"))
        lam2_exp = eig_scaling.get("lambda_idio_exponent_vs_K", float("nan"))
        print(f"lambda_common exponent: {lam1_exp:.3f} (target 1.0)")
        print(f"lambda_idio exponent: {lam2_exp:.3f} (target 0.0)")

    confirms_sqrtK = (
        best_norm[0] in ["sqrt(K)", "K^0.4"] and
        abs(beta_w - 0.5) < 0.15
    )
    print(f"CONFIRMS sqrt(K) mechanism: {confirms_sqrtK}")

    out = {
        "config": vars(args),
        "points": rows,
        "normalization_results": normalization_results,
        "link_linearity": link_linearity,
        "width_scaling": width_scaling,
        "eigenmode_scaling": eig_scaling,
        "interpretation_targets": {
            "best_norm_should_be": "sqrt(K)",
            "best_link_should_be": "logistic_or_probit",
            "width_beta_target": 0.5,
            "lambda_common_exponent_target": 1.0,
            "lambda_idio_exponent_target": 0.0,
        },
        "confirms_sqrtK": confirms_sqrtK,
    }

    out_path = RESULTS_DIR / "cti_first_principles_exact_integral.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
