"""
Compute statistics for external baseline comparisons.
======================================================

Produces:
- Per-method steerability summary (mean +/- std across seeds)
- Head-to-head comparison: V5 vs each baseline (sign test, effect size)
- Combined comparison table for paper
- LaTeX table snippet

Usage:
    python src/compute_external_baseline_stats.py
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from external_baselines.common import ALL_DATASETS, ALL_SEEDS, RESULTS_DIR

RESULTS_ROOT = Path(__file__).parent.parent / "results"
METHODS = ["heal", "csr", "smec"]


def load_v5_mrl_steerability():
    """Load V5 and MRL steerability from benchmark files."""
    v5_data = {}  # dataset -> {seed -> steer}
    mrl_data = {}

    for ds in ALL_DATASETS:
        v5_data[ds] = {}
        mrl_data[ds] = {}

        bench_path = RESULTS_ROOT / f"benchmark_bge-small_{ds}.json"
        if not bench_path.exists():
            continue

        bench = json.load(open(bench_path))

        for seed in ALL_SEEDS:
            seed_key = str(seed)

            # V5
            if seed_key in bench.get("v5", {}):
                pa = bench["v5"][seed_key].get("prefix_accuracy", {})
                s = (pa.get("j1_l0", 0) - pa.get("j4_l0", 0)) + \
                    (pa.get("j4_l1", 0) - pa.get("j1_l1", 0))
                v5_data[ds][seed] = s

            # MRL
            if seed_key in bench.get("mrl", {}):
                pa = bench["mrl"][seed_key].get("prefix_accuracy", {})
                s = (pa.get("j1_l0", 0) - pa.get("j4_l0", 0)) + \
                    (pa.get("j4_l1", 0) - pa.get("j1_l1", 0))
                mrl_data[ds][seed] = s

    return v5_data, mrl_data


def load_baseline_steerability(method):
    """Load steerability for an external baseline method."""
    data = {}  # dataset -> {seed -> steer}

    for ds in ALL_DATASETS:
        data[ds] = {}
        for seed in ALL_SEEDS:
            path = RESULTS_DIR / method / f"{ds}_seed{seed}.json"
            if path.exists():
                r = json.load(open(path))
                data[ds][seed] = r["steerability"]

    return data


def compute_head_to_head(v5_data, baseline_data, method_name):
    """
    Compute head-to-head comparison stats between V5 and a baseline.

    Returns:
    - Per-dataset: paired t-test or sign test
    - Overall: win rate, sign test, meta-analytic summary
    """
    results = {
        "method": method_name,
        "per_dataset": {},
        "overall": {},
    }

    v5_wins = 0
    total_datasets = 0
    all_v5_means = []
    all_base_means = []

    for ds in ALL_DATASETS:
        v5_seeds = v5_data.get(ds, {})
        base_seeds = baseline_data.get(ds, {})

        # Get paired seeds
        common_seeds = sorted(set(v5_seeds.keys()) & set(base_seeds.keys()))
        if len(common_seeds) < 2:
            continue

        v5_vals = np.array([v5_seeds[s] for s in common_seeds])
        base_vals = np.array([base_seeds[s] for s in common_seeds])
        diffs = v5_vals - base_vals

        v5_mean = np.mean(v5_vals)
        base_mean = np.mean(base_vals)
        diff_mean = np.mean(diffs)

        all_v5_means.append(v5_mean)
        all_base_means.append(base_mean)

        if v5_mean > base_mean:
            v5_wins += 1
        total_datasets += 1

        # Paired t-test
        if len(common_seeds) >= 3:
            t_stat, p_val = stats.ttest_rel(v5_vals, base_vals)
        else:
            t_stat, p_val = float("nan"), float("nan")

        # Cohen's d (paired)
        if np.std(diffs) > 0:
            d = np.mean(diffs) / np.std(diffs)
        else:
            d = float("inf") if diff_mean > 0 else float("-inf") if diff_mean < 0 else 0

        results["per_dataset"][ds] = {
            "v5_mean": float(v5_mean),
            "base_mean": float(base_mean),
            "diff_mean": float(diff_mean),
            "v5_std": float(np.std(v5_vals)),
            "base_std": float(np.std(base_vals)),
            "cohens_d": float(d),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "n_seeds": len(common_seeds),
        }

    # Overall stats
    if total_datasets > 0:
        # Sign test
        sign_p = stats.binomtest(
            v5_wins, total_datasets, 0.5, alternative="greater"
        ).pvalue

        results["overall"] = {
            "v5_wins": v5_wins,
            "total_datasets": total_datasets,
            "win_rate": v5_wins / total_datasets,
            "sign_test_p": float(sign_p),
            "v5_mean_steer": float(np.mean(all_v5_means)),
            "base_mean_steer": float(np.mean(all_base_means)),
            "mean_advantage": float(np.mean(all_v5_means) - np.mean(all_base_means)),
        }

    return results


def generate_latex_table(v5_data, mrl_data, baseline_results):
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Steerability comparison: V5 vs.\ external baselines. "
                 r"Values are mean steerability $S$ across 5 seeds. "
                 r"Bold = highest per dataset.}")
    lines.append(r"\label{tab:external_baselines}")
    lines.append(r"\small")

    n_methods = 2 + len(METHODS)  # V5 + MRL + externals
    cols = "l" + "c" * n_methods
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")

    header = r"Dataset & V5 (Ours) & MRL"
    for m in METHODS:
        header += f" & {m.upper()}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for ds in ALL_DATASETS:
        vals = {}

        # V5
        if ds in v5_data:
            seeds = list(v5_data[ds].values())
            if seeds:
                vals["v5"] = np.mean(seeds)

        # MRL
        if ds in mrl_data:
            seeds = list(mrl_data[ds].values())
            if seeds:
                vals["mrl"] = np.mean(seeds)

        # Baselines
        for m in METHODS:
            for br in baseline_results:
                if br["method"] == m and ds in br.get("_raw", {}):
                    seeds = list(br["_raw"][ds].values())
                    if seeds:
                        vals[m] = np.mean(seeds)

        if not vals:
            continue

        # Find max
        max_val = max(vals.values()) if vals else 0

        row = ds.replace("_", r"\_")
        for key in ["v5", "mrl"] + METHODS:
            if key in vals:
                v = vals[key]
                fmt = f"{v:+.3f}"
                if abs(v - max_val) < 1e-6:
                    fmt = r"\textbf{" + fmt + "}"
                row += f" & {fmt}"
            else:
                row += " & --"

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  External Baseline Comparison Statistics")
    print("=" * 70)

    # Load data
    v5_data, mrl_data = load_v5_mrl_steerability()

    baseline_data = {}
    for m in METHODS:
        baseline_data[m] = load_baseline_steerability(m)

    # Head-to-head comparisons
    all_results = []
    for m in METHODS:
        h2h = compute_head_to_head(v5_data, baseline_data[m], m)
        h2h["_raw"] = baseline_data[m]
        all_results.append(h2h)

        print(f"\n--- V5 vs {m.upper()} ---")
        ovr = h2h.get("overall", {})
        if ovr:
            print(f"  Win rate: {ovr['v5_wins']}/{ovr['total_datasets']} "
                  f"({ovr['win_rate']:.0%})")
            print(f"  Sign test p = {ovr['sign_test_p']:.6f}")
            print(f"  V5 mean S = {ovr['v5_mean_steer']:+.4f}")
            print(f"  {m.upper()} mean S = {ovr['base_mean_steer']:+.4f}")
            print(f"  Advantage = {ovr['mean_advantage']:+.4f}")
        else:
            print("  No data available yet")

    # Also compare against MRL
    print(f"\n--- V5 vs MRL (internal) ---")
    mrl_h2h = compute_head_to_head(v5_data, mrl_data, "mrl")
    ovr = mrl_h2h.get("overall", {})
    if ovr:
        print(f"  Win rate: {ovr['v5_wins']}/{ovr['total_datasets']} "
              f"({ovr['win_rate']:.0%})")
        print(f"  Sign test p = {ovr['sign_test_p']:.6f}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("  STEERABILITY COMPARISON TABLE")
    print(f"{'=' * 70}")

    header = f"{'Dataset':<18} {'V5':>8} {'MRL':>8}"
    for m in METHODS:
        header += f" {m.upper():>8}"
    print(header)
    print("-" * len(header))

    for ds in ALL_DATASETS:
        line = f"{ds:<18}"

        # V5
        if ds in v5_data and v5_data[ds]:
            v5_mean = np.mean(list(v5_data[ds].values()))
            line += f" {v5_mean:>+8.4f}"
        else:
            line += f" {'--':>8}"

        # MRL
        if ds in mrl_data and mrl_data[ds]:
            mrl_mean = np.mean(list(mrl_data[ds].values()))
            line += f" {mrl_mean:>+8.4f}"
        else:
            line += f" {'--':>8}"

        # Baselines
        for m in METHODS:
            if ds in baseline_data[m] and baseline_data[m][ds]:
                b_mean = np.mean(list(baseline_data[m][ds].values()))
                line += f" {b_mean:>+8.4f}"
            else:
                line += f" {'--':>8}"

        print(line)

    # Save results
    output = {
        "v5_vs_baselines": {m: {k: v for k, v in r.items() if k != "_raw"}
                            for m, r in zip(METHODS, all_results)},
        "v5_vs_mrl": {k: v for k, v in mrl_h2h.items() if k != "_raw"},
    }

    output_path = RESULTS_ROOT / "external_baseline_stats.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # LaTeX table
    latex = generate_latex_table(v5_data, mrl_data, all_results)
    latex_path = RESULTS_ROOT / "external_baselines_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
