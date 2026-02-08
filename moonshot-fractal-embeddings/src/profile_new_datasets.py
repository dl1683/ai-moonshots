"""Compute H(L1|L0) hierarchy profiles for new datasets."""

import sys
import os
import json
import numpy as np
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset

RESULTS_DIR = Path(__file__).parent.parent / "results"


def compute_hierarchy_profile(dataset_name, max_samples=10000):
    """Compute H(L1|L0) and other entropy metrics for a dataset."""
    print(f"\n{'='*60}")
    print(f"  Profiling: {dataset_name}")
    print(f"{'='*60}")

    try:
        ds = load_hierarchical_dataset(dataset_name, split="train", max_samples=max_samples)
    except Exception as e:
        print(f"  ERROR loading {dataset_name}: {e}")
        return None

    if len(ds.samples) == 0:
        print(f"  No samples loaded for {dataset_name}")
        return None

    # Extract labels
    l0_labels = np.array([s.level0_label for s in ds.samples])
    l1_labels = np.array([s.level1_label for s in ds.samples])

    n = len(l0_labels)
    n_l0 = len(set(l0_labels))
    n_l1 = len(set(l1_labels))

    # H(L0) - marginal entropy of coarse labels
    l0_counts = Counter(l0_labels)
    l0_probs = np.array([l0_counts[k] / n for k in sorted(l0_counts.keys())])
    h_l0 = -np.sum(l0_probs * np.log2(l0_probs + 1e-12))

    # H(L1) - marginal entropy of fine labels
    l1_counts = Counter(l1_labels)
    l1_probs = np.array([l1_counts[k] / n for k in sorted(l1_counts.keys())])
    h_l1 = -np.sum(l1_probs * np.log2(l1_probs + 1e-12))

    # H(L1|L0) = H(L0,L1) - H(L0) = H(L1) - I(L0;L1)
    # Compute via conditional: H(L1|L0) = sum_l0 P(l0) * H(L1|L0=l0)
    h_l1_given_l0 = 0.0
    for l0_val in sorted(l0_counts.keys()):
        mask = l0_labels == l0_val
        p_l0 = mask.sum() / n
        l1_sub = l1_labels[mask]
        l1_sub_counts = Counter(l1_sub)
        l1_sub_probs = np.array([v / len(l1_sub) for v in l1_sub_counts.values()])
        h_cond = -np.sum(l1_sub_probs * np.log2(l1_sub_probs + 1e-12))
        h_l1_given_l0 += p_l0 * h_cond

    # MI(L0; L1) = H(L1) - H(L1|L0)
    mi = h_l1 - h_l1_given_l0
    branching = n_l1 / n_l0 if n_l0 > 0 else 0

    profile = {
        "dataset": dataset_name,
        "n_samples": n,
        "n_l0": n_l0,
        "n_l1": n_l1,
        "branching_factor": branching,
        "h_l0": float(h_l0),
        "h_l1": float(h_l1),
        "h_l1_given_l0": float(h_l1_given_l0),
        "mi_levels": float(mi),
        "depth_ratio": float(h_l1_given_l0 / h_l0) if h_l0 > 0 else 0,
        "avg_samples_per_l1": n / n_l1 if n_l1 > 0 else 0,
    }

    print(f"  Samples: {n}")
    print(f"  Hierarchy: {n_l0} L0 -> {n_l1} L1 (branch={branching:.1f})")
    print(f"  H(L0)={h_l0:.3f}, H(L1)={h_l1:.3f}")
    print(f"  H(L1|L0) = {h_l1_given_l0:.3f}")
    print(f"  MI(L0;L1) = {mi:.3f}")

    return profile


def main():
    new_datasets = ["wos", "goemotions", "arxiv", "dbpedia_classes"]

    # Load existing profiles
    profile_path = RESULTS_DIR / "hierarchy_profiles.json"
    if profile_path.exists():
        with open(profile_path) as f:
            profiles = json.load(f)
    else:
        profiles = {}

    for ds_name in new_datasets:
        profile = compute_hierarchy_profile(ds_name, max_samples=10000)
        if profile:
            profiles[ds_name] = profile

    # Save updated profiles
    with open(profile_path, 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f"\nSaved to {profile_path}")

    # Print summary table sorted by H(L1|L0)
    print(f"\n{'='*70}")
    print(f"  ALL DATASETS SORTED BY H(L1|L0)")
    print(f"{'='*70}")
    print(f"  {'Dataset':<18} {'L0':>4} {'L1':>5} {'H(L1|L0)':>10} {'Branch':>8}")
    print(f"  {'-'*50}")
    sorted_profiles = sorted(profiles.values(), key=lambda x: x["h_l1_given_l0"])
    for p in sorted_profiles:
        print(f"  {p['dataset']:<18} {p['n_l0']:>4} {p['n_l1']:>5} {p['h_l1_given_l0']:>10.3f} {p['branching_factor']:>8.1f}")


if __name__ == "__main__":
    main()
