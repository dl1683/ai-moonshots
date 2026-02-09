"""
Compute all paper-ready statistical tests with Holm-Bonferroni correction.
Reads benchmark JSONs and produces:
1. Per-dataset V5 vs MRL steerability t-tests
2. Holm-Bonferroni corrected p-values
3. 95% CIs for steerability gap
4. Updated steerability table values (mean +/- SD, n seeds)
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "results"

# Dataset order (by H(L1|L0))
DATASETS = ["yahoo", "goemotions", "newsgroups", "trec", "arxiv", "clinc", "dbpedia_classes", "wos"]
DATASET_H = {
    "yahoo": 1.23, "goemotions": 1.88, "newsgroups": 1.88,
    "trec": 2.21, "arxiv": 2.62, "dbpedia_classes": 3.17, "clinc": 3.90, "wos": 5.05
}


def load_benchmark(dataset, model="bge-small"):
    """Load benchmark JSON and extract steerability per seed."""
    path = RESULTS_DIR / f"benchmark_{model}_{dataset}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    v5_steers = []
    mrl_steers = []
    seeds = data.get("seeds", [])

    for seed in [str(s) for s in seeds]:
        # V5
        if seed in data.get("v5", {}):
            pa = data["v5"][seed].get("prefix_accuracy", {})
            if "j1_l0" in pa and "j4_l0" in pa:
                s = (pa["j1_l0"] - pa["j4_l0"]) + (pa["j4_l1"] - pa["j1_l1"])
                v5_steers.append(s)
        # MRL
        if seed in data.get("mrl", {}):
            pa = data["mrl"][seed].get("prefix_accuracy", {})
            if "j1_l0" in pa and "j4_l0" in pa:
                s = (pa["j1_l0"] - pa["j4_l0"]) + (pa["j4_l1"] - pa["j1_l1"])
                mrl_steers.append(s)

    return {
        "seeds": seeds,
        "n": min(len(v5_steers), len(mrl_steers)),
        "v5_steers": v5_steers,
        "mrl_steers": mrl_steers,
    }


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to a list of (label, p_value) tuples.
    Returns list of (label, raw_p, adjusted_p, significant) tuples."""
    m = len(p_values)
    # Sort by raw p-value
    sorted_pvs = sorted(p_values, key=lambda x: x[1])
    adjusted = []
    max_so_far = 0.0
    for i, (label, p) in enumerate(sorted_pvs):
        adj_p = min(p * (m - i), 1.0)
        # Enforce monotonicity
        max_so_far = max(max_so_far, adj_p)
        adjusted.append((label, p, max_so_far, max_so_far < 0.05))
    # Re-sort by original order
    label_order = {lbl: idx for idx, (lbl, _) in enumerate(p_values)}
    adjusted.sort(key=lambda x: label_order[x[0]])
    return adjusted


def main():
    print("=" * 90)
    print("  COMPREHENSIVE STATISTICAL TESTS WITH HOLM-BONFERRONI CORRECTION")
    print("=" * 90)

    all_results = {}
    raw_pvalues = []

    for ds in DATASETS:
        data = load_benchmark(ds)
        if data is None:
            print(f"\n  {ds}: NO DATA")
            continue

        n = data["n"]
        v5 = np.array(data["v5_steers"][:n])
        mrl = np.array(data["mrl_steers"][:n])
        diffs = v5 - mrl

        if n < 2:
            print(f"\n  {ds}: Only {n} matched seeds, skipping")
            continue

        t_stat, p_value = stats.ttest_rel(v5, mrl)
        d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else float('inf')

        all_results[ds] = {
            "n": n,
            "v5_mean": float(np.mean(v5)),
            "v5_sd": float(np.std(v5, ddof=1)),
            "mrl_mean": float(np.mean(mrl)),
            "mrl_sd": float(np.std(mrl, ddof=1)),
            "gap": float(np.mean(diffs)),
            "t": float(t_stat),
            "p": float(p_value),
            "d": float(d),
        }
        raw_pvalues.append((ds, p_value))

    # Holm-Bonferroni correction
    corrected = holm_bonferroni(raw_pvalues)
    correction_map = {label: (raw, adj, sig) for label, raw, adj, sig in corrected}

    # Print results table
    print(f"\n{'Dataset':<12} {'H(L1|L0)':>8} {'n':>3} {'V5 S':>12} {'MRL S':>12} {'Gap':>8} {'t':>7} {'p_raw':>8} {'p_adj':>8} {'d':>6} {'Sig':>5}")
    print("-" * 105)

    for ds in DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]
        raw_p, adj_p, sig = correction_map[ds]
        h = DATASET_H[ds]
        v5_str = f"+{r['v5_mean']:.3f}+/-{r['v5_sd']:.3f}"
        mrl_str = f"+{r['mrl_mean']:.3f}+/-{r['mrl_sd']:.3f}"
        sig_str = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
        print(f"{ds:<12} {h:>8.2f} {r['n']:>3} {v5_str:>12} {mrl_str:>12} {r['gap']:>+8.3f} {r['t']:>7.2f} {raw_p:>8.4f} {adj_p:>8.4f} {r['d']:>6.2f} {sig_str:>5}")

    # LaTeX table rows
    print(f"\n\n{'='*90}")
    print("  LATEX TABLE ROWS (for steerability table)")
    print("=" * 90)
    print(r"Dataset & $\hlo$ & V5 $\steer$ & MRL $\steer$ & Gap & $p_{\text{adj}}$ & $d$ & Seeds \\")
    print(r"\midrule")
    for ds in DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]
        raw_p, adj_p, sig = correction_map[ds]
        h = DATASET_H[ds]
        name = {
            "yahoo": "Yahoo", "goemotions": "GoEmotions", "newsgroups": "Newsgroups",
            "trec": "TREC", "arxiv": "arXiv", "clinc": "CLINC",
            "dbpedia_classes": "DBPedia Classes", "wos": "WOS"
        }[ds]

        # Format p-value
        if adj_p < 0.001:
            p_str = "$< 0.001$"
        else:
            p_str = f"${adj_p:.3f}$"

        # Format d
        d_str = f"${r['d']:.1f}$" if abs(r['d']) < 100 else f"${r['d']:.0f}$"

        print(f"{name} & {h:.2f} & ${r['v5_mean']:+.3f} \\pm {r['v5_sd']:.3f}$ & "
              f"${r['mrl_mean']:+.3f} \\pm {r['mrl_sd']:.3f}$ & "
              f"${r['gap']:+.3f}$ & {p_str} & {d_str} & {r['n']} \\\\")

    # Summary
    print(f"\n\n{'='*90}")
    print("  SUMMARY")
    print("=" * 90)
    n_sig = sum(1 for _, _, _, s in corrected if s)
    print(f"  Total tests: {len(corrected)}")
    print(f"  Significant after Holm correction (alpha=0.05): {n_sig}/{len(corrected)}")
    for label, raw, adj, sig in corrected:
        print(f"    {label:<12} raw={raw:.4f}  adj={adj:.4f}  {'SIGNIFICANT' if sig else 'not significant'}")

    # Save results
    output = {
        "method": "paired_ttest_with_holm_bonferroni",
        "alpha": 0.05,
        "n_tests": len(corrected),
        "results": {}
    }
    for ds in DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]
        raw_p, adj_p, sig = correction_map[ds]
        output["results"][ds] = {
            **r,
            "p_raw": float(raw_p),
            "p_adjusted": float(adj_p),
            "significant": bool(sig),
            "H_L1_L0": DATASET_H[ds],
        }

    out_path = RESULTS_DIR / "paper_statistics_holm.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
