"""
Update paper tables with latest benchmark data.
Reads all benchmark JSONs and outputs updated LaTeX table rows
for the steerability table and accuracy table.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

DATASETS = ["yahoo", "goemotions", "newsgroups", "trec", "arxiv", "dbpedia_classes", "clinc", "wos"]
DATASET_NAMES = {
    "yahoo": "Yahoo", "goemotions": "GoEmotions", "newsgroups": "Newsgroups",
    "trec": "TREC", "arxiv": "arXiv", "dbpedia_classes": "DBPedia Classes",
    "clinc": "CLINC", "wos": "WOS"
}
DATASET_H = {
    "yahoo": 1.23, "goemotions": 1.88, "newsgroups": 1.88,
    "trec": 2.21, "arxiv": 2.62, "dbpedia_classes": 3.17, "clinc": 3.90, "wos": 5.05
}


def load_benchmark(dataset, model="bge-small"):
    """Load benchmark JSON."""
    path = RESULTS_DIR / f"benchmark_{model}_{dataset}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_per_seed(data, method="v5"):
    """Extract per-seed steerability and accuracy from benchmark data."""
    results = []
    seeds = data.get("seeds", [])
    method_data = data.get(method, {})
    for seed in [str(s) for s in seeds]:
        if seed not in method_data:
            continue
        entry = method_data[seed]
        pa = entry.get("prefix_accuracy", {})
        if "j1_l0" not in pa:
            continue
        steer = (pa["j1_l0"] - pa["j4_l0"]) + (pa["j4_l1"] - pa["j1_l1"])
        acc_key = method if method == "mrl" else method
        acc_data = entry.get(acc_key, entry.get("v5", entry.get("mrl", {})))
        l0_acc = acc_data.get("l0_accuracy", pa.get("j4_l0", 0))
        l1_acc = acc_data.get("l1_accuracy", pa.get("j4_l1", 0))
        results.append({
            "seed": seed,
            "steer": steer,
            "l0": l0_acc,
            "l1": l1_acc,
            "j4_l0": pa["j4_l0"],
            "j4_l1": pa["j4_l1"],
        })
    return results


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction. Input: list of (label, p). Output: list of (label, raw, adj, sig)."""
    m = len(p_values)
    sorted_pvs = sorted(p_values, key=lambda x: x[1])
    adjusted = []
    max_so_far = 0.0
    for i, (label, p) in enumerate(sorted_pvs):
        adj_p = min(p * (m - i), 1.0)
        max_so_far = max(max_so_far, adj_p)
        adjusted.append((label, p, max_so_far, max_so_far < 0.05))
    label_order = {lbl: idx for idx, (lbl, _) in enumerate(p_values)}
    adjusted.sort(key=lambda x: label_order[x[0]])
    return adjusted


def main():
    print("=" * 90)
    print("  UPDATED PAPER TABLES")
    print("=" * 90)

    # ===== STEERABILITY TABLE =====
    print("\n--- Steerability Table ---")
    print(r"Dataset & $\hlo$ & V5 $\steer$ & MRL $\steer$ & Gap & Seeds \\")
    print(r"\midrule")

    raw_pvalues = []
    table_data = {}

    for ds in DATASETS:
        data = load_benchmark(ds)
        if data is None:
            continue
        v5_seeds = extract_per_seed(data, "v5")
        mrl_seeds = extract_per_seed(data, "mrl")
        n = min(len(v5_seeds), len(mrl_seeds))

        v5_steers = np.array([s["steer"] for s in v5_seeds[:n]])
        mrl_steers = np.array([s["steer"] for s in mrl_seeds[:n]])

        v5_mean = np.mean(v5_steers)
        v5_sd = np.std(v5_steers, ddof=1) if n > 1 else 0
        mrl_mean = np.mean(mrl_steers)
        mrl_sd = np.std(mrl_steers, ddof=1) if n > 1 else 0
        gap = v5_mean - mrl_mean

        if n >= 2:
            t, p = stats.ttest_rel(v5_steers, mrl_steers)
            d = gap / np.std(v5_steers - mrl_steers, ddof=1) if np.std(v5_steers - mrl_steers, ddof=1) > 0 else float('inf')
            raw_pvalues.append((ds, float(p)))
        else:
            t, p, d = 0, 1, 0

        table_data[ds] = {
            "n": n, "v5_mean": v5_mean, "v5_sd": v5_sd,
            "mrl_mean": mrl_mean, "mrl_sd": mrl_sd,
            "gap": gap, "t": float(t), "p": float(p), "d": float(d)
        }

        name = DATASET_NAMES[ds]
        h = DATASET_H[ds]
        print(f"{name} & {h:.2f} & ${v5_mean:+.3f} \\pm {v5_sd:.3f}$ & "
              f"${mrl_mean:+.3f} \\pm {mrl_sd:.3f}$ & ${gap:+.3f}$ & {n} \\\\")

    # Holm correction
    if raw_pvalues:
        corrected = holm_bonferroni(raw_pvalues)
        print(f"\n--- Holm-Bonferroni Corrected p-values (m={len(corrected)}) ---")
        for label, raw, adj, sig in corrected:
            print(f"  {label:<12} raw={raw:.4f}  adj={adj:.4f}  {'*' if sig else 'ns'}")

    # ===== ACCURACY TABLE =====
    print("\n\n--- Accuracy Table ---")
    print(r"Dataset & Baseline L0 & V5 L0 & MRL L0 & Baseline L1 & V5 L1 & MRL L1 \\")
    print(r"\midrule")

    for ds in DATASETS:
        data = load_benchmark(ds)
        if data is None:
            continue
        v5_seeds = extract_per_seed(data, "v5")
        mrl_seeds = extract_per_seed(data, "mrl")

        # Get baseline from first seed
        first_v5 = data["v5"][str(data["seeds"][0])]
        bl_l0 = first_v5.get("baseline", {}).get("l0_accuracy", 0)
        bl_l1 = first_v5.get("baseline", {}).get("l1_accuracy", 0)

        v5_l0 = np.mean([s["j4_l0"] for s in v5_seeds])
        v5_l1 = np.mean([s["j4_l1"] for s in v5_seeds])
        mrl_l0 = np.mean([s["j4_l0"] for s in mrl_seeds])
        mrl_l1 = np.mean([s["j4_l1"] for s in mrl_seeds])

        name = DATASET_NAMES[ds]
        print(f"{name} & {bl_l0:.3f} & {v5_l0:.3f} & {mrl_l0:.3f} & "
              f"{bl_l1:.3f} & {v5_l1:.3f} & {mrl_l1:.3f} \\\\")

    # ===== ABSTRACT STATS =====
    if "clinc" in table_data:
        cd = table_data["clinc"]
        print(f"\n\n--- Abstract Stats (CLINC) ---")
        print(f"  V5: +{cd['v5_mean']:.3f} +/- {cd['v5_sd']:.3f}")
        print(f"  MRL: +{cd['mrl_mean']:.3f} +/- {cd['mrl_sd']:.3f}")
        print(f"  p = {cd['p']:.4f}, d = {cd['d']:.1f}, n = {cd['n']}")


if __name__ == "__main__":
    main()
