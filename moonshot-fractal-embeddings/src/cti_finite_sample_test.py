#!/usr/bin/env python
"""
FINITE SAMPLE TEST: How does the Gumbel Race Law degrade with small m?

The theorem requires EVT (m -> infinity). Real experiments have m = 10-50 per class.
Key question: At what m does the law break down?

Test m = 5, 10, 20, 50, 100, 200 with fixed K=50, d=300.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.special import expit, logit as sp_logit
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def safe_logit(q, eps=0.001):
    return sp_logit(np.clip(q, eps, 1.0 - eps))


def simulate_knn(d, K, m, kappa_target, n_test=2000):
    delta2 = kappa_target * K * d / max(K - 1, 1)
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

    grand_mean = means.mean(0)
    tr_sb = sum(np.sum((means[k] - grand_mean)**2) for k in range(K)) / K
    actual_kappa = tr_sb / (d * 1.0)

    train_labels = np.repeat(np.arange(K), m)
    train_X = np.zeros((K * m, d))
    for k in range(K):
        train_X[k*m:(k+1)*m] = means[k] + np.random.randn(m, d)

    test_labels = np.random.randint(0, K, n_test)
    test_X = np.zeros((n_test, d))
    for i in range(n_test):
        test_X[i] = means[test_labels[i]] + np.random.randn(d)

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
    print("FINITE SAMPLE TEST: Gumbel Race Law vs sample size m")
    print("=" * 70)

    d = 300
    K = 50
    m_values = [5, 10, 20, 50, 100, 200]
    kappa_range = np.linspace(0.02, 0.4, 10)

    results = {}

    for m in m_values:
        print(f"\n  m={m}...", end="", flush=True)
        data = []
        for kappa_t in kappa_range:
            if kappa_t * K > d * 0.8:
                continue
            acc, q, actual_kappa = simulate_knn(d, K, m, kappa_t, n_test=1500)
            data.append({"kappa": actual_kappa, "q": q})
            sys.stdout.write(".")
            sys.stdout.flush()
        print(" done")

        kappas = np.array([p["kappa"] for p in data])
        qs = np.array([p["q"] for p in data])
        logK = np.log(K)

        mask = (qs > 0.005) & (qs < 0.995)
        if mask.sum() < 3:
            results[m] = {"r2": 0, "B": 0, "n_valid": int(mask.sum())}
            continue

        kf = kappas[mask]
        qf = safe_logit(qs[mask])

        # Fit: logit(q) = A*kappa + C (single K, so B*log(K) is absorbed into C)
        X = np.column_stack([kf, np.ones(len(kf))])
        beta, _, _, _ = np.linalg.lstsq(X, qf, rcond=None)
        pred = X @ beta
        r2 = 1 - ((qf - pred)**2).sum() / ((qf - qf.mean())**2).sum()

        # Sigmoid fit
        from scipy.optimize import curve_fit
        def sig(x, a, b):
            return expit(a * x + b)
        try:
            popt, _ = curve_fit(sig, kappas, qs, p0=[30.0, -2.0], maxfev=5000)
            q_pred = sig(kappas, *popt)
            r2_sig = 1 - ((qs - q_pred)**2).sum() / ((qs - qs.mean())**2).sum()
        except Exception:
            r2_sig = 0
            popt = [0, 0]

        results[m] = {
            "n_valid": int(mask.sum()),
            "logit_r2": float(r2),
            "A": float(beta[0]),
            "C": float(beta[1]),
            "sigmoid_r2": float(r2_sig),
            "sigmoid_a": float(popt[0]),
            "sigmoid_b": float(popt[1]),
        }

        print(f"    Logit-linear: A={beta[0]:.3f}, C={beta[1]:.3f}, R^2={r2:.4f}")
        print(f"    Sigmoid: a={popt[0]:.3f}, b={popt[1]:.3f}, R^2={r2_sig:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("FINITE SAMPLE SUMMARY (K=50, d=300)")
    print(f"{'='*70}")

    print(f"\n  {'m':>5} | {'Logit R^2':>10} | {'A (slope)':>10} | {'Sigmoid R^2':>12}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    for m in m_values:
        res = results.get(m, {})
        lr2 = res.get("logit_r2", 0)
        A = res.get("A", 0)
        sr2 = res.get("sigmoid_r2", 0)
        print(f"  {m:>5} | {lr2:>10.4f} | {A:>10.3f} | {sr2:>12.4f}")

    # Does slope A change with m? Theory: A = alpha/beta where beta doesn't depend on m
    # but alpha may depend on m through the Gumbel location
    A_values = [results[m]["A"] for m in m_values if "A" in results.get(m, {})]
    m_arr = [m for m in m_values if "A" in results.get(m, {})]

    if len(A_values) >= 3:
        r, p = pearsonr(np.log(m_arr), A_values)
        print(f"\n  Correlation A vs log(m): r={r:.4f}, p={p:.4f}")
        print(f"  Theory: A should increase with log(m) (more samples -> sharper transition)")

    # Scorecard
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    checks = [
        ("Law holds at m=5 (logit R^2 > 0.8)",
         results.get(5, {}).get("logit_r2", 0) > 0.8,
         f"R^2={results.get(5, {}).get('logit_r2', 0):.4f}"),
        ("Law holds at m=10 (logit R^2 > 0.9)",
         results.get(10, {}).get("logit_r2", 0) > 0.9,
         f"R^2={results.get(10, {}).get('logit_r2', 0):.4f}"),
        ("Sigmoid R^2 > 0.95 for m >= 20",
         all(results.get(m, {}).get("sigmoid_r2", 0) > 0.95 for m in [20, 50, 100, 200]),
         f"min={min(results.get(m, {}).get('sigmoid_r2', 0) for m in [20, 50, 100, 200]):.4f}"),
        ("Slope A increases monotonically with m",
         all(A_values[i] <= A_values[i+1] for i in range(len(A_values)-1)) if len(A_values) >= 2 else False,
         f"A_values={[f'{a:.1f}' for a in A_values]}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    save_data = {
        "experiment": "finite_sample_test",
        "d": d, "K": K,
        "m_values": m_values,
        "results": {str(k): v for k, v in results.items()},
        "scorecard": {"passes": passes, "total": len(checks)},
    }

    out_path = RESULTS_DIR / "cti_finite_sample.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
