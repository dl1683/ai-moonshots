"""
Predict-Before-Train Experiment: The Decisive Test
===================================================

Hypothesis (Hierarchical Sufficiency Principle):
  The steerability of a trained V5 embedding can be PREDICTED from the
  data distribution alone, before any model training occurs.

Protocol:
  1. From raw data, compute hierarchy complexity profile H(L1|L0), H(L0), etc.
  2. Predict steerability from these information-theoretic measures
  3. Train V5 and measure actual steerability
  4. Test whether prediction holds across datasets

If prediction correlates with actual steerability --> the structure is data-determined, not architecture-accidental.
This would be evidence for a fundamental law.

Key Metrics (from data alone):
  - H(L0): entropy of coarse labels (how hard is coarse classification?)
  - H(L1|L0): conditional entropy (how much info does fine add beyond coarse?)
  - Branching factor: |L1|/|L0|
  - Hierarchy depth ratio: H(L1|L0) / H(L0)
  - Category utility: avg info gain per hierarchy level
  - Successive refinement gap: I(X;L1) - I(X;L0) (mutual info difference)
"""

import sys
import os
import numpy as np
from collections import Counter
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset


def compute_entropy(labels):
    """Compute Shannon entropy of a label distribution."""
    counts = Counter(labels)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return -np.sum(probs * np.log2(probs + 1e-12))


def compute_conditional_entropy(fine_labels, coarse_labels):
    """Compute H(L1|L0) = sum_l0 P(l0) * H(L1|L0=l0)."""
    total = len(fine_labels)
    coarse_counts = Counter(coarse_labels)

    h_cond = 0.0
    for l0, count in coarse_counts.items():
        p_l0 = count / total
        # Get fine labels within this coarse group
        fine_in_group = [fl for fl, cl in zip(fine_labels, coarse_labels) if cl == l0]
        h_fine_given_l0 = compute_entropy(fine_in_group)
        h_cond += p_l0 * h_fine_given_l0

    return h_cond


def compute_hierarchy_profile(dataset_name, split="train", max_samples=10000):
    """Compute the information-theoretic hierarchy profile of a dataset."""
    data = load_hierarchical_dataset(dataset_name, split=split, max_samples=max_samples)

    l0_labels = [s.level0_label for s in data.samples]
    l1_labels = [s.level1_label for s in data.samples]

    n_samples = len(data.samples)
    n_l0 = len(set(l0_labels))
    n_l1 = len(set(l1_labels))

    # Information-theoretic measures
    h_l0 = compute_entropy(l0_labels)
    h_l1 = compute_entropy(l1_labels)
    h_l1_given_l0 = compute_conditional_entropy(l1_labels, l0_labels)

    # Mutual information between levels
    # I(L0; L1) = H(L1) - H(L1|L0)
    mi_levels = h_l1 - h_l1_given_l0

    # Branching factor
    branching = n_l1 / n_l0

    # Hierarchy depth ratio: how much fine adds beyond coarse
    depth_ratio = h_l1_given_l0 / (h_l0 + 1e-12)

    # Information concentration: fraction of total info at coarse level
    info_concentration = h_l0 / (h_l1 + 1e-12)

    # Average samples per fine class
    avg_samples_per_l1 = n_samples / n_l1

    # Category utility: average info gain from splitting coarse into fine
    # CU = H(L1) - H(L0) when L1 refines L0
    category_utility = h_l1 - h_l0

    # Coarse separability: how distinguishable are coarse classes?
    # Measured by max entropy minus actual entropy, normalized
    max_h_l0 = np.log2(n_l0)
    coarse_separability = h_l0 / max_h_l0 if max_h_l0 > 0 else 0

    # Fine granularity: how finely does L1 split within L0?
    max_h_l1_given_l0 = np.log2(branching) if branching > 1 else 0
    fine_granularity = h_l1_given_l0 / max_h_l1_given_l0 if max_h_l1_given_l0 > 0 else 0

    profile = {
        'dataset': dataset_name,
        'n_samples': n_samples,
        'n_l0': n_l0,
        'n_l1': n_l1,
        'branching_factor': branching,
        'h_l0': h_l0,
        'h_l1': h_l1,
        'h_l1_given_l0': h_l1_given_l0,
        'mi_levels': mi_levels,
        'depth_ratio': depth_ratio,
        'info_concentration': info_concentration,
        'category_utility': category_utility,
        'coarse_separability': coarse_separability,
        'fine_granularity': fine_granularity,
        'avg_samples_per_l1': avg_samples_per_l1,
    }

    return profile


def predict_steerability(profile):
    """
    Predict steerability from hierarchy profile alone.

    Theory: steerability should scale with:
    1. Hierarchy depth (more levels = more room for scale separation)
    2. Conditional entropy H(L1|L0) (more info at fine level = more to steer)
    3. Branching factor (deeper branching = more scale separation possible)

    Predicted steerability ~ depth_ratio * fine_granularity * log(branching)
    """
    # Primary predictor: how much "room" for steering
    depth = profile['depth_ratio']
    granularity = profile['fine_granularity']
    branching = profile['branching_factor']

    # Simple linear predictor (we'll fit this later)
    # For now, use an information-theoretic proxy:
    # Steerability ~ (info at fine level beyond coarse) / (total info)
    predicted = profile['h_l1_given_l0'] / (profile['h_l1'] + 1e-12)

    # Scale by log-branching (captures the structural depth)
    predicted *= np.log2(branching) / np.log2(max(branching, 2))

    return predicted


def main():
    datasets = ['yahoo', 'clinc', 'trec', 'newsgroups']

    print("=" * 80)
    print("  PREDICT-BEFORE-TRAIN: Hierarchy Complexity Profiles")
    print("=" * 80)

    profiles = {}
    for ds in datasets:
        print(f"\n  Computing profile for {ds}...")
        try:
            profile = compute_hierarchy_profile(ds)
            profiles[ds] = profile

            print(f"    Classes: {profile['n_l0']} L0 -> {profile['n_l1']} L1 (branching={profile['branching_factor']:.1f})")
            print(f"    H(L0)={profile['h_l0']:.3f} bits, H(L1)={profile['h_l1']:.3f} bits")
            print(f"    H(L1|L0)={profile['h_l1_given_l0']:.3f} bits (fine info beyond coarse)")
            print(f"    I(L0;L1)={profile['mi_levels']:.3f} bits (shared info between levels)")
            print(f"    Depth ratio={profile['depth_ratio']:.3f}")
            print(f"    Info concentration (coarse)={profile['info_concentration']:.3f}")
            print(f"    Fine granularity={profile['fine_granularity']:.3f}")
        except Exception as e:
            print(f"    FAILED: {e}")

    # Predictions
    print(f"\n{'='*80}")
    print(f"  PREDICTIONS vs ACTUAL STEERABILITY")
    print(f"{'='*80}")

    # Known actual steerability (from experiments)
    actual_steerability = {
        'yahoo': 0.011,     # Shallow hierarchy, weak effect
        'clinc': 0.054,     # Deep hierarchy, strong effect
        # trec: not measured yet in ablation format
        # newsgroups: not measured yet
    }

    print(f"\n  {'Dataset':<12} {'Branch':<8} {'H(L1|L0)':<10} {'DepthRatio':<12} {'Predicted':<12} {'Actual':<12} {'Match?'}")
    print(f"  {'-'*78}")

    for ds, profile in profiles.items():
        predicted = predict_steerability(profile)
        actual = actual_steerability.get(ds, None)
        match = ""
        if actual is not None:
            # Check if ordering is preserved
            match = "YES" if (predicted > 0.3) == (actual > 0.02) else "NO"

        print(f"  {ds:<12} {profile['branching_factor']:<8.1f} {profile['h_l1_given_l0']:<10.3f} "
              f"{profile['depth_ratio']:<12.3f} {predicted:<12.4f} "
              f"{actual if actual is not None else 'TBD':<12} {match}")

    # Correlation analysis
    datasets_with_actual = [ds for ds in profiles if ds in actual_steerability]
    if len(datasets_with_actual) >= 2:
        preds = [predict_steerability(profiles[ds]) for ds in datasets_with_actual]
        actuals = [actual_steerability[ds] for ds in datasets_with_actual]

        # Rank correlation (with only 2 points this is just sign agreement)
        rank_match = (preds[0] > preds[1]) == (actuals[0] > actuals[1])
        print(f"\n  Rank correlation (prediction vs actual): {'MATCH' if rank_match else 'MISMATCH'}")
        print(f"  Predicted ordering: {' > '.join([f'{ds}({predict_steerability(profiles[ds]):.4f})' for ds in sorted(datasets_with_actual, key=lambda d: -predict_steerability(profiles[d]))])}")
        print(f"  Actual ordering:    {' > '.join([f'{ds}({actual_steerability[ds]:.4f})' for ds in sorted(datasets_with_actual, key=lambda d: -actual_steerability[d])])}")

    # Save profiles
    results_dir = Path(__file__).parent.parent / "results"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in profiles.items()}
    with open(results_dir / "hierarchy_profiles.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Profiles saved to {results_dir / 'hierarchy_profiles.json'}")

    print(f"\n{'='*80}")
    print(f"  NEXT STEPS FOR LAW-LEVEL EVIDENCE")
    print(f"{'='*80}")
    print(f"""
  1. Run V5 steerability on ALL datasets (need TREC and Newsgroups ablation-style eval)
  2. Add more datasets: DBPedia, WOS, CIFAR-100-Coarse-Fine, iNaturalist
  3. Fit a proper prediction model: Steer = f(H(L1|L0), branching, n_l1, ...)
  4. Test cross-domain (text -> vision -> biology)
  5. If prediction holds -> LAW. If not -> find what's missing.
""")


if __name__ == "__main__":
    main()
