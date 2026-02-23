"""
Practical utility demo: kappa as architecture selector.

The law says logit(q) = alpha*kappa + C_dataset (for fixed K, dataset).
=> Ranking by kappa = Ranking by q (architecture selector without running classification).

This demo:
1. For each dataset (fixed K), rank architectures by kappa_nearest
2. Rank the same architectures by actual q (1-NN accuracy)
3. Compute Spearman rho (kappa-rank vs q-rank) per dataset
4. Pre-registered: mean Spearman rho >= 0.80 across datasets

Key claim: kappa extracts architectural quality signal WITHOUT training a classifier.
Just compute centroid distances from raw embeddings.

Output: results/cti_kappa_ranking_demo.json
"""

import json
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_kappa_ranking_demo.json")

# Pre-registered threshold
SPEARMAN_THRESHOLD = 0.80
MIN_ARCHS_PER_DATASET = 5  # need at least 5 architectures for meaningful ranking


def load_cache():
    """Load all valid points from kappa_near_cache files."""
    pts = []
    for fname in os.listdir(RESULTS_DIR):
        if not (fname.startswith("kappa_near_cache_") and fname.endswith(".json")):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for entry in data:
            q = entry.get("q")
            kappa = entry.get("kappa_nearest")
            K = entry.get("K")
            model = entry.get("model", "")
            dataset = entry.get("dataset", "")
            if q is None or kappa is None or K is None:
                continue
            if q <= 0 or q >= 1.0:
                continue
            if kappa <= 0:
                continue
            pts.append({
                "model": model,
                "dataset": dataset,
                "K": int(K),
                "q": float(q),
                "kappa": float(kappa),
            })
    return pts


def mean_per_model_dataset(pts):
    """For each (model, dataset), average kappa and q across all available layers."""
    groups = {}
    for p in pts:
        key = (p["model"], p["dataset"])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    mean_pts = []
    for (model, dataset), plist in groups.items():
        mean_pts.append({
            "model": model,
            "dataset": dataset,
            "K": plist[0]["K"],
            "q": float(np.mean([p["q"] for p in plist])),
            "kappa": float(np.mean([p["kappa"] for p in plist])),
            "n_layers": len(plist),
        })
    return mean_pts


def main():
    print("Loading cache points...")
    all_pts = load_cache()
    print(f"Loaded {len(all_pts)} raw points")

    # Get mean kappa/q per (model, dataset) across all layers
    pts = mean_per_model_dataset(all_pts)
    print(f"After best-layer selection: {len(pts)} points")

    # Group by dataset
    by_dataset = {}
    for p in pts:
        d = p["dataset"]
        if d not in by_dataset:
            by_dataset[d] = []
        by_dataset[d].append(p)

    results = {}
    spearman_rhos = []

    print("\nPer-dataset kappa vs q ranking:")
    print(f"{'Dataset':>20} {'K':>4} {'N_arch':>7} {'Spearman_rho':>13} {'Pearson_r':>10} {'PR_PASS':>8}")
    for dataset, plist in sorted(by_dataset.items()):
        if len(plist) < MIN_ARCHS_PER_DATASET:
            print(f"{dataset:>20} {plist[0]['K']:>4} {len(plist):>7} -- (too few archs)")
            continue

        K = plist[0]["K"]
        kappas = np.array([p["kappa"] for p in plist])
        qs = np.array([p["q"] for p in plist])
        models = [p["model"] for p in plist]

        rho, p_spear = spearmanr(kappas, qs)
        r_pearson, p_pearson = pearsonr(kappas, qs)
        pr = abs(rho) >= SPEARMAN_THRESHOLD

        print(f"{dataset:>20} {K:>4} {len(plist):>7} {rho:>13.4f} {r_pearson:>10.4f} {'PASS' if pr else 'FAIL':>8}")

        results[dataset] = {
            "K": K,
            "n_architectures": len(plist),
            "spearman_rho": float(rho),
            "spearman_p": float(p_spear),
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "pr_pass": bool(pr),
            "architectures": models,
            "kappas": kappas.tolist(),
            "accuracies": qs.tolist(),
        }
        spearman_rhos.append(rho)

    n_pass = sum(1 for d in results.values() if d["pr_pass"])
    n_total = len(results)
    mean_rho = np.mean(spearman_rhos) if spearman_rhos else 0.0

    print(f"\nSummary:")
    print(f"  Datasets with >= {MIN_ARCHS_PER_DATASET} architectures: {n_total}")
    print(f"  Mean Spearman rho: {mean_rho:.4f}")
    print(f"  Datasets passing rho >= {SPEARMAN_THRESHOLD}: {n_pass}/{n_total}")
    print(f"  Pre-registered: mean rho >= {SPEARMAN_THRESHOLD} -> "
          f"{'PASS' if mean_rho >= SPEARMAN_THRESHOLD else 'FAIL'}")

    output = {
        "experiment": "kappa_ranking_demo",
        "pre_registered_threshold": SPEARMAN_THRESHOLD,
        "min_archs_per_dataset": MIN_ARCHS_PER_DATASET,
        "n_datasets": n_total,
        "mean_spearman_rho": float(mean_rho),
        "n_pass": n_pass,
        "pr_mean_pass": bool(mean_rho >= SPEARMAN_THRESHOLD),
        "per_dataset": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
