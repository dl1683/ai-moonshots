"""
Metric Robustness Battery for Fractal Embeddings.
Computes alternative steerability formulations to demonstrate construct validity.

Metrics:
1. S_orig: Original steerability (Eq 1) = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)
2. S_auc: Area-under-curve steerability = sum over all j of normalized scores
3. S_mono: Monotonicity index = fraction of adjacent prefix pairs with correct ordering
4. S_gap: SpecGap = (L0@j1 - L1@j1) - (L0@j4 - L1@j4) = how much more L0-specialized is j1 vs j4

All four should agree on V5 >> MRL if the construct is valid.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

DATASETS = ["yahoo", "goemotions", "newsgroups", "trec", "arxiv", "clinc", "dbpedia_classes", "wos"]
DATASET_NAMES = {
    "yahoo": "Yahoo", "goemotions": "GoEmotions", "newsgroups": "Newsgroups",
    "trec": "TREC", "arxiv": "arXiv", "clinc": "CLINC",
    "dbpedia_classes": "DBPedia", "wos": "WOS"
}
DATASET_H = {
    "yahoo": 1.23, "goemotions": 1.88, "newsgroups": 1.88,
    "trec": 2.21, "arxiv": 2.62, "clinc": 3.90,
    "dbpedia_classes": 3.17, "wos": 5.05
}


def extract_prefix_curves(data, method):
    """Extract per-seed prefix accuracy curves: returns list of dicts with j1..j4 l0/l1."""
    seeds_data = data.get(method, {})
    curves = []
    for seed_key, seed_val in seeds_data.items():
        if not isinstance(seed_val, dict) or "prefix_accuracy" not in seed_val:
            continue
        pa = seed_val["prefix_accuracy"]
        if "j1_l0" not in pa:
            continue
        curve = {}
        for j in range(1, 5):
            curve[f"j{j}_l0"] = pa.get(f"j{j}_l0", 0)
            curve[f"j{j}_l1"] = pa.get(f"j{j}_l1", 0)
        curves.append(curve)
    return curves


def s_orig(curve):
    """Original steerability: (L0@j1 - L0@j4) + (L1@j4 - L1@j1)."""
    return (curve["j1_l0"] - curve["j4_l0"]) + (curve["j4_l1"] - curve["j1_l1"])


def s_auc(curve):
    """AUC steerability: average over all prefix pairs of normalized score.
    Uses trapezoidal integration of the L0-L1 specialization difference across j."""
    # For each j, compute L0_specialization = L0@j - mean(L0), L1_specialization = L1@j - mean(L1)
    # Or simpler: sum of (L0@j1 - L0@j) + (L1@j - L1@j1) for j=2,3,4 normalized
    total = 0
    for j in range(2, 5):
        total += (curve["j1_l0"] - curve[f"j{j}_l0"]) + (curve[f"j{j}_l1"] - curve["j1_l1"])
    return total / 3.0  # Average over 3 pairs


def s_mono(curve):
    """Monotonicity index: fraction of adjacent pairs where L0 decreases and L1 increases.
    Perfect steerability = 1.0, random = 0.5, inverted = 0.0."""
    correct = 0
    total = 0
    for j in range(1, 4):
        # L0 should decrease (or stay) as j increases
        if curve[f"j{j}_l0"] >= curve[f"j{j+1}_l0"]:
            correct += 1
        total += 1
        # L1 should increase (or stay) as j increases
        if curve[f"j{j}_l1"] <= curve[f"j{j+1}_l1"]:
            correct += 1
        total += 1
    return correct / total


def s_gap(curve):
    """SpecGap: change in L0-L1 specialization from j=1 to j=4.
    = (L0@j1 - L1@j1) - (L0@j4 - L1@j4)
    High when j=1 is L0-specialized and j=4 is L1-specialized."""
    gap_j1 = curve["j1_l0"] - curve["j1_l1"]
    gap_j4 = curve["j4_l0"] - curve["j4_l1"]
    return gap_j1 - gap_j4


METRICS = {
    "S_orig": s_orig,
    "S_auc": s_auc,
    "S_mono": s_mono,
    "S_gap": s_gap,
}


def main():
    print("=" * 100)
    print("  METRIC ROBUSTNESS BATTERY")
    print("=" * 100)

    all_results = {}

    for ds in DATASETS:
        path = RESULTS_DIR / f"benchmark_bge-small_{ds}.json"
        if not path.exists():
            continue
        data = json.load(open(path))

        v5_curves = extract_prefix_curves(data, "v5")
        mrl_curves = extract_prefix_curves(data, "mrl")
        n = min(len(v5_curves), len(mrl_curves))

        if n < 2:
            continue

        ds_results = {"n": n, "metrics": {}}

        for metric_name, metric_fn in METRICS.items():
            v5_vals = np.array([metric_fn(c) for c in v5_curves[:n]])
            mrl_vals = np.array([metric_fn(c) for c in mrl_curves[:n]])
            diffs = v5_vals - mrl_vals

            t_stat, p_val = stats.ttest_rel(v5_vals, mrl_vals)
            d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else float('inf')

            ds_results["metrics"][metric_name] = {
                "v5_mean": float(np.mean(v5_vals)),
                "v5_sd": float(np.std(v5_vals, ddof=1)),
                "mrl_mean": float(np.mean(mrl_vals)),
                "mrl_sd": float(np.std(mrl_vals, ddof=1)),
                "gap": float(np.mean(diffs)),
                "t": float(t_stat),
                "p": float(p_val),
                "d": float(d),
                "v5_gt_mrl": bool(np.mean(v5_vals) > np.mean(mrl_vals)),
            }

        all_results[ds] = ds_results

    # Print results table
    print(f"\n{'Dataset':<12}", end="")
    for mn in METRICS:
        print(f" | {mn:>28}", end="")
    print()
    print("-" * 130)

    for ds in DATASETS:
        if ds not in all_results:
            continue
        print(f"{DATASET_NAMES[ds]:<12}", end="")
        for mn in METRICS:
            m = all_results[ds]["metrics"][mn]
            sig = "*" if m["p"] < 0.05 else " "
            sign = "+" if m["v5_gt_mrl"] else "-"
            print(f" | {sign} gap={m['gap']:+.4f} d={m['d']:>5.1f}{sig}", end="")
        print()

    # Summary: count how many datasets show V5 > MRL for each metric
    print(f"\n\n{'Metric':<12} {'V5>MRL count':<15} {'Sign test p':<15}")
    print("-" * 50)
    for mn in METRICS:
        count = sum(1 for ds in all_results if all_results[ds]["metrics"][mn]["v5_gt_mrl"])
        total = len(all_results)
        # One-sided binomial test
        p_sign = stats.binomtest(count, total, 0.5, alternative='greater').pvalue if count > 0 else 1.0
        print(f"{mn:<12} {count}/{total:<14} p={p_sign:.4f}")

    # Cross-metric correlation (do all metrics agree on ranking?)
    print("\n\n--- Cross-Metric Rank Correlation ---")
    ds_list = [ds for ds in DATASETS if ds in all_results]
    metric_names = list(METRICS.keys())
    for i, m1 in enumerate(metric_names):
        for m2 in metric_names[i+1:]:
            vals1 = [all_results[ds]["metrics"][m1]["gap"] for ds in ds_list]
            vals2 = [all_results[ds]["metrics"][m2]["gap"] for ds in ds_list]
            rho, p = stats.spearmanr(vals1, vals2)
            print(f"  {m1} vs {m2}: Spearman rho={rho:.3f} (p={p:.4f})")

    # LaTeX table
    print("\n\n--- LaTeX Table (for appendix) ---")
    print(r"Dataset & $\hlo$ & $\steer_\text{orig}$ & $\steer_\text{AUC}$ & $\steer_\text{mono}$ & $\steer_\text{gap}$ \\")
    print(r"\midrule")
    for ds in DATASETS:
        if ds not in all_results:
            continue
        name = DATASET_NAMES[ds]
        h = DATASET_H[ds]
        metrics = all_results[ds]["metrics"]
        orig = metrics["S_orig"]
        auc = metrics["S_auc"]
        mono = metrics["S_mono"]
        gap = metrics["S_gap"]
        print(f"{name} & {h:.2f} & ${orig['gap']:+.3f}$ & ${auc['gap']:+.3f}$ & "
              f"${mono['gap']:+.2f}$ & ${gap['gap']:+.3f}$ \\\\")

    # Save
    output = {
        "description": "Metric robustness battery: 4 alternative steerability formulations",
        "metrics": list(METRICS.keys()),
        "datasets": {}
    }
    for ds in DATASETS:
        if ds in all_results:
            output["datasets"][ds] = all_results[ds]

    out_path = RESULTS_DIR / "metric_robustness_battery.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
