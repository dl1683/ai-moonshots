"""Scaling Trend Robustness: Leave-one-out and prediction interval analysis.

Addresses Codex review concern that rho=0.83 on 6 points is fragile.
Computes:
1. Leave-one-out Spearman correlation (drop each dataset, recompute)
2. Prediction interval for meta-analysis (not just CI)
3. Pearson regression diagnostics (Cook's distance, leverage)
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from itertools import combinations

RESULTS_DIR = Path(__file__).parent.parent / "results"

DATASETS = ["yahoo", "goemotions", "newsgroups", "trec", "arxiv", "clinc", "dbpedia_classes", "wos"]


def compute_steer(pa):
    return (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
           (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))


def load_dataset_stats():
    """Load per-dataset mean steerability gap and H(L1|L0)."""
    results = []
    for ds in DATASETS:
        f = RESULTS_DIR / f"benchmark_bge-small_{ds}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        seeds = d.get('seeds', [])
        v5_steers, mrl_steers = [], []
        for seed in seeds:
            sk = str(seed)
            if sk in d.get('v5', {}) and isinstance(d['v5'][sk], dict) and 'prefix_accuracy' in d['v5'][sk]:
                if sk in d.get('mrl', {}) and isinstance(d['mrl'][sk], dict) and 'prefix_accuracy' in d['mrl'][sk]:
                    v5_steers.append(compute_steer(d['v5'][sk]['prefix_accuracy']))
                    mrl_steers.append(compute_steer(d['mrl'][sk]['prefix_accuracy']))
        if len(v5_steers) < 2:
            continue
        v5_arr = np.array(v5_steers)
        mrl_arr = np.array(mrl_steers)
        gap = np.mean(v5_arr - mrl_arr)
        gap_se = np.std(v5_arr - mrl_arr, ddof=1) / np.sqrt(len(v5_arr))
        results.append({
            "name": ds,
            "n": len(v5_arr),
            "v5_steer": float(np.mean(v5_arr)),
            "gap": float(gap),
            "gap_se": float(gap_se),
        })
    return results


def get_hlo():
    """Get H(L1|L0) from hierarchy profiles."""
    hlo_map = {}
    for ds in DATASETS:
        f = RESULTS_DIR / f"benchmark_bge-small_{ds}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        # Try to get from any seed's hierarchy profile
        for method in ['v5', 'mrl']:
            for sk in d.get(method, {}):
                entry = d[method][sk]
                if isinstance(entry, dict) and 'hierarchy_profile' in entry:
                    hp = entry['hierarchy_profile']
                    hlo_map[ds] = hp.get('H_L1_given_L0', hp.get('h_l1_given_l0', None))
                    break
            if ds in hlo_map:
                break
    # Fallback to known values
    known = {
        "yahoo": 1.2288, "goemotions": 1.8815, "newsgroups": 1.8815,
        "trec": 2.2082, "arxiv": 2.6238, "dbpedia_classes": 3.17, "clinc": 3.9069,
        "wos": 5.05
    }
    for ds in DATASETS:
        if ds not in hlo_map:
            hlo_map[ds] = known.get(ds)
    return hlo_map


def main():
    print("=" * 80)
    print("  SCALING TREND ROBUSTNESS ANALYSIS")
    print("=" * 80)

    ds_stats = load_dataset_stats()
    hlo_map = get_hlo()

    names = [d['name'] for d in ds_stats]
    gaps = np.array([d['v5_steer'] for d in ds_stats])
    hlos = np.array([hlo_map[d['name']] for d in ds_stats])
    n = len(names)

    print(f"\n  Datasets: {names}")
    print(f"  H(L1|L0): {[f'{h:.3f}' for h in hlos]}")
    print(f"  V5 steer: {[f'{g:+.4f}' for g in gaps]}")

    # Full correlation
    rho_full, p_full = stats.spearmanr(hlos, gaps)
    r_full, pr_full = stats.pearsonr(hlos, gaps)
    print(f"\n  FULL: Spearman rho={rho_full:.3f} (p={p_full:.4f}), "
          f"Pearson r={r_full:.3f} (p={pr_full:.4f})")

    # === 1. Leave-one-out Spearman ===
    print(f"\n{'=' * 80}")
    print("  LEAVE-ONE-OUT ANALYSIS")
    print(f"{'=' * 80}")

    loo_results = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        hlos_loo = hlos[idx]
        gaps_loo = gaps[idx]
        rho_loo, p_loo = stats.spearmanr(hlos_loo, gaps_loo)
        r_loo, pr_loo = stats.pearsonr(hlos_loo, gaps_loo)
        print(f"  Drop {names[i]:12s}: rho={rho_loo:.3f} (p={p_loo:.4f}), "
              f"r={r_loo:.3f} (p={pr_loo:.4f})")
        loo_results.append({
            "dropped": names[i],
            "spearman_rho": float(rho_loo),
            "spearman_p": float(p_loo),
            "pearson_r": float(r_loo),
            "pearson_p": float(pr_loo),
        })

    rhos = [r['spearman_rho'] for r in loo_results]
    print(f"\n  LOO Spearman range: [{min(rhos):.3f}, {max(rhos):.3f}]")
    print(f"  LOO Spearman mean: {np.mean(rhos):.3f}")

    # === 2. Regression diagnostics (Cook's distance) ===
    print(f"\n{'=' * 80}")
    print("  REGRESSION DIAGNOSTICS")
    print(f"{'=' * 80}")

    # OLS fit
    slope, intercept, r_val, p_val, se = stats.linregress(hlos, gaps)
    predicted = intercept + slope * hlos
    residuals = gaps - predicted
    hat_diag = 1/n + (hlos - np.mean(hlos))**2 / np.sum((hlos - np.mean(hlos))**2)
    mse = np.sum(residuals**2) / (n - 2)

    print(f"\n  OLS: steer = {slope:.4f} * H(L1|L0) + {intercept:.4f}")
    print(f"  R^2 = {r_val**2:.3f}, p = {p_val:.4f}")
    print(f"  MSE = {mse:.6f}")

    # Cook's distance
    cooks_d = residuals**2 * hat_diag / (2 * mse * (1 - hat_diag)**2)
    print(f"\n  Cook's distance (threshold: {4/n:.2f}):")
    for i in range(n):
        flag = " **INFLUENTIAL**" if cooks_d[i] > 4/n else ""
        print(f"    {names[i]:12s}: D={cooks_d[i]:.4f}, leverage={hat_diag[i]:.3f}, "
              f"residual={residuals[i]:+.4f}{flag}")

    # === 3. Bootstrap CI for rho ===
    print(f"\n{'=' * 80}")
    print("  BOOTSTRAP CI FOR SPEARMAN RHO")
    print(f"{'=' * 80}")

    np.random.seed(42)
    boot_rhos = []
    for _ in range(10000):
        idx = np.random.choice(n, n, replace=True)
        if len(set(idx)) < 3:
            continue
        rho_b, _ = stats.spearmanr(hlos[idx], gaps[idx])
        if not np.isnan(rho_b):
            boot_rhos.append(rho_b)

    boot_rhos = np.array(boot_rhos)
    ci_lo = np.percentile(boot_rhos, 2.5)
    ci_hi = np.percentile(boot_rhos, 97.5)
    print(f"\n  Bootstrap rho: {np.mean(boot_rhos):.3f}")
    print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Fraction > 0: {np.mean(boot_rhos > 0):.4f}")

    # === 4. Meta-analysis prediction interval ===
    print(f"\n{'=' * 80}")
    print("  META-ANALYSIS PREDICTION INTERVAL")
    print(f"{'=' * 80}")

    ma = json.load(open(RESULTS_DIR / "meta_analysis.json"))
    theta = ma['cohens_d']['pooled']
    se_theta = ma['cohens_d']['se']
    tau2 = ma['cohens_d']['tau2']

    # Prediction interval = pooled +/- t_{k-2, 0.025} * sqrt(se^2 + tau^2)
    k = ma['n_studies']
    t_crit = stats.t.ppf(0.975, k - 2)
    pi_se = np.sqrt(se_theta**2 + tau2)
    pi_lo = theta - t_crit * pi_se
    pi_hi = theta + t_crit * pi_se

    print(f"\n  Pooled d = {theta:.3f}")
    print(f"  95% CI: [{theta - 1.96*se_theta:.3f}, {theta + 1.96*se_theta:.3f}]")
    print(f"  95% PI: [{pi_lo:.3f}, {pi_hi:.3f}]")
    print(f"  PI includes zero: {'YES' if pi_lo < 0 else 'NO'}")
    print(f"  tau^2 = {tau2:.4f}")

    # === Save results ===
    output = {
        "full_correlation": {
            "spearman_rho": float(rho_full),
            "spearman_p": float(p_full),
            "pearson_r": float(r_full),
            "pearson_p": float(pr_full),
        },
        "leave_one_out": loo_results,
        "loo_rho_range": [float(min(rhos)), float(max(rhos))],
        "loo_rho_mean": float(np.mean(rhos)),
        "regression": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_val**2),
            "p": float(p_val),
        },
        "cooks_distance": {n: float(d) for n, d in zip(names, cooks_d)},
        "bootstrap_rho": {
            "mean": float(np.mean(boot_rhos)),
            "ci_95": [float(ci_lo), float(ci_hi)],
            "frac_positive": float(np.mean(boot_rhos > 0)),
        },
        "prediction_interval": {
            "pooled_d": float(theta),
            "ci_95": [float(theta - 1.96*se_theta), float(theta + 1.96*se_theta)],
            "pi_95": [float(pi_lo), float(pi_hi)],
            "includes_zero": bool(pi_lo < 0),
        },
    }

    out_path = RESULTS_DIR / "scaling_robustness.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
