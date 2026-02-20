#!/usr/bin/env python
"""
UNIVERSALITY TEST: Does the Gumbel Race Law hold for non-Gaussian distributions?

The theorem was proved for isotropic Gaussian clusters.
But the key mechanism (Gumbel EVT -> logistic margin) should hold for ANY
distribution in the Gumbel maximum domain of attraction, which includes:
- Sub-Gaussian (e.g., bounded, sub-exponential tails)
- Log-normal (light tails)
- Uniform (bounded)

If the law holds for non-Gaussian distributions, the theorem is UNIVERSAL.
This would be a major upgrade (from narrow Gaussian result to general theory).

Test distributions:
1. Gaussian (reference)
2. Uniform (bounded support)
3. t-distribution with df=10 (heavier tails)
4. Laplace (exponential tails)
5. Mixed: Gaussian + outliers
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.special import expit, logit as sp_logit
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def safe_logit(q, eps=0.001):
    q_clip = np.clip(q, eps, 1.0 - eps)
    return sp_logit(q_clip)


def generate_cluster(dist_type, n, d, mu, scale=1.0):
    """Generate n samples in d dimensions around mu with given distribution."""
    if dist_type == "gaussian":
        return mu + np.random.randn(n, d) * scale
    elif dist_type == "uniform":
        # Uniform on [-sqrt(3), sqrt(3)] * scale (same variance as standard normal)
        w = np.sqrt(3) * scale
        return mu + np.random.uniform(-w, w, (n, d))
    elif dist_type == "t10":
        # t-distribution with df=10 (heavier tails)
        # Scale to match variance: var of t(10) is 10/(10-2) = 1.25
        raw = np.random.standard_t(10, (n, d))
        return mu + raw * scale / np.sqrt(1.25)
    elif dist_type == "laplace":
        # Laplace (exponential tails)
        # Scale: var of Laplace(0,b) is 2b^2, so b = scale/sqrt(2)
        b = scale / np.sqrt(2)
        return mu + np.random.laplace(0, b, (n, d))
    elif dist_type == "mixed":
        # 90% Gaussian + 10% outliers (5x scale)
        n_normal = int(0.9 * n)
        n_outlier = n - n_normal
        normal = mu + np.random.randn(n_normal, d) * scale
        outlier = mu + np.random.randn(n_outlier, d) * scale * 5.0
        combined = np.concatenate([normal, outlier], axis=0)
        np.random.shuffle(combined)
        return combined
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")


def simulate_knn(d, K, m, kappa_target, dist_type="gaussian", n_test=2000):
    """Simulate 1-NN for K clusters with given distribution."""
    sigma2 = 1.0
    delta2 = kappa_target * K * d / max(K - 1, 1)

    # Simplex means
    if K - 1 <= d:
        V = np.eye(K, min(K-1, d))
        V = V - V.mean(0)
        norms = np.sqrt((V ** 2).sum(1, keepdims=True))
        norms[norms < 1e-10] = 1.0
        V = V / norms * np.sqrt((K - 1) / K)
        means = np.zeros((K, d))
        means[:, :min(K-1, d)] = V * np.sqrt(delta2)
    else:
        means = np.random.randn(K, d)
        means = means - means.mean(0)
        norms = np.sqrt((means ** 2).sum(1, keepdims=True))
        norms[norms < 1e-10] = 1.0
        means = means / norms * np.sqrt(delta2) * np.sqrt((K - 1) / K)

    # Verify kappa
    grand_mean = means.mean(0)
    tr_sb = sum(np.sum((means[k] - grand_mean)**2) for k in range(K)) / K
    tr_sw = d * sigma2
    actual_kappa = tr_sb / tr_sw

    # Train set
    train_labels = np.repeat(np.arange(K), m)
    train_X = np.zeros((K * m, d))
    for k in range(K):
        train_X[k*m:(k+1)*m] = generate_cluster(dist_type, m, d, means[k])

    # Test set
    test_labels = np.random.randint(0, K, n_test)
    test_X = np.zeros((n_test, d))
    for i in range(n_test):
        test_X[i] = generate_cluster(dist_type, 1, d, means[test_labels[i]])[0]

    # 1-NN
    correct = 0
    bs = 500
    for s in range(0, n_test, bs):
        e = min(s + bs, n_test)
        diff = test_X[s:e, None, :] - train_X[None, :, :]
        dists = (diff ** 2).sum(2)
        nn = dists.argmin(1)
        correct += (train_labels[nn] == test_labels[s:e]).sum()

    acc = correct / n_test
    q = (acc - 1.0/K) / (1.0 - 1.0/K)
    return acc, q, actual_kappa


def main():
    print("=" * 70)
    print("UNIVERSALITY TEST: Gumbel Race Law for Non-Gaussian Distributions")
    print("=" * 70)

    d = 300
    m = 40
    K_values = [10, 50, 100]
    kappa_range = np.linspace(0.02, 0.5, 10)
    dist_types = ["gaussian", "uniform", "t10", "laplace", "mixed"]

    all_results = {}

    for dist in dist_types:
        print(f"\n  --- Distribution: {dist} ---", flush=True)
        data = []

        for K in K_values:
            print(f"    K={K}...", end="", flush=True)
            for kappa_t in kappa_range:
                if kappa_t * K > d * 0.8:
                    continue
                acc, q, actual_kappa = simulate_knn(d, K, m, kappa_t, dist, n_test=1500)
                data.append({"K": K, "kappa": actual_kappa, "q": q})
                sys.stdout.write(".")
                sys.stdout.flush()
            print(" done")

        # Fit logit-linear model
        kappas = np.array([p["kappa"] for p in data])
        qs = np.array([p["q"] for p in data])
        Ks = np.array([p["K"] for p in data])
        logKs = np.log(Ks)

        mask = (qs > 0.005) & (qs < 0.995)
        if mask.sum() < 5:
            print(f"    Too few points in logit range ({mask.sum()})")
            continue

        kf = kappas[mask]
        qf = safe_logit(qs[mask])
        lKf = logKs[mask]

        # Model: logit(q) = A*kappa - B*log(K) + C
        X = np.column_stack([kf, lKf, np.ones(len(kf))])
        beta, _, _, _ = np.linalg.lstsq(X, qf, rcond=None)
        pred = X @ beta
        r2 = 1 - ((qf - pred)**2).sum() / ((qf - qf.mean())**2).sum()

        # Also fit sigmoid(kappa/log(K))
        def sigmoid_model(x, a, b):
            return expit(a * x + b)
        x_norm = kappas / np.log(Ks)
        try:
            popt, _ = curve_fit(sigmoid_model, x_norm, qs, p0=[10.0, -1.0], maxfev=5000)
            q_pred = sigmoid_model(x_norm, *popt)
            r2_sig = 1 - ((qs - q_pred)**2).sum() / ((qs - qs.mean())**2).sum()
        except Exception:
            r2_sig = 0
            popt = [0, 0]

        all_results[dist] = {
            "n_points": len(data),
            "logit_r2": float(r2),
            "A": float(beta[0]),
            "B": float(-beta[1]),
            "C": float(beta[2]),
            "sigmoid_r2": float(r2_sig),
            "sigmoid_a": float(popt[0]),
            "sigmoid_b": float(popt[1]),
        }

        print(f"    Logit-linear: A={beta[0]:.3f}, B={-beta[1]:.3f}, C={beta[2]:.3f}, "
              f"R^2={r2:.4f}")
        print(f"    Sigmoid(kappa/logK): a={popt[0]:.3f}, b={popt[1]:.3f}, "
              f"R^2={r2_sig:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("UNIVERSALITY SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Distribution':>12} | {'Logit R^2':>10} | {'B coeff':>8} | {'Sigmoid R^2':>12}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")

    for dist, res in all_results.items():
        print(f"  {dist:>12} | {res['logit_r2']:>10.4f} | {res['B']:>8.4f} | {res['sigmoid_r2']:>12.4f}")

    # Key test: is B close to 1.0 for all distributions?
    B_values = [res["B"] for res in all_results.values()]
    B_mean = np.mean(B_values)
    B_std = np.std(B_values)

    print(f"\n  B coefficient: mean={B_mean:.4f}, std={B_std:.4f}")
    print(f"  Theory predicts B = 1.0")

    # Scorecard
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    checks = [
        ("All distributions have logit R^2 > 0.90",
         all(r["logit_r2"] > 0.90 for r in all_results.values()),
         f"min={min(r['logit_r2'] for r in all_results.values()):.4f}"),
        ("B coefficient within [0.5, 1.5] for all",
         all(0.5 <= r["B"] <= 1.5 for r in all_results.values()),
         f"range=[{min(r['B'] for r in all_results.values()):.3f}, "
         f"{max(r['B'] for r in all_results.values()):.3f}]"),
        ("Non-Gaussian B within 20% of Gaussian B",
         all(abs(r["B"] - all_results.get("gaussian", {}).get("B", 1.0)) /
             max(all_results.get("gaussian", {}).get("B", 1.0), 0.01) < 0.20
             for dist, r in all_results.items() if dist != "gaussian"),
         f"see table above"),
        ("Sigmoid R^2 > 0.85 for all distributions",
         all(r["sigmoid_r2"] > 0.85 for r in all_results.values()),
         f"min={min(r['sigmoid_r2'] for r in all_results.values()):.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    save_data = {
        "experiment": "universality_test",
        "d": d, "m": m,
        "K_values": K_values,
        "distributions": list(all_results.keys()),
        "results": all_results,
        "B_statistics": {"mean": float(B_mean), "std": float(B_std)},
        "scorecard": {"passes": passes, "total": len(checks)},
    }

    out_path = RESULTS_DIR / "cti_universality.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
