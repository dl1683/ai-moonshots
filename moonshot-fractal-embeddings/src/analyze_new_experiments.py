"""Analysis of new experiments: capacity sweep, backbone control, deep hierarchy.

Produces:
1. Capacity sweep: optimal dim per dataset, Goldilocks peak shift plot
2. Backbone control: V5-frozen vs flat-finetune comparison
3. Deep hierarchy: steerability vs H(L1|L0) with new HUPD data points
4. Updated scaling analysis combining all datasets

Run: python src/analyze_new_experiments.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"


def analyze_capacity_sweep():
    """Analyze capacity sweep Goldilocks results."""
    path = RESULTS_DIR / "capacity_sweep_goldilocks.json"
    if not path.exists():
        print("  No capacity sweep results yet")
        return None

    d = json.load(open(path))
    results = d['results']
    h_values = d['H_L1_L0']

    print("\n" + "=" * 70)
    print("CAPACITY SWEEP GOLDILOCKS ANALYSIS")
    print("=" * 70)

    dataset_peaks = {}

    for ds in sorted(results.keys()):
        dims = results[ds]
        h = h_values.get(ds, 0)
        print(f"\n  {ds} (H(L1|L0) = {h:.2f}):")

        dim_means = {}
        for dim_str in sorted(dims.keys(), key=int):
            runs = dims[dim_str]
            if not isinstance(runs, list) or not runs:
                continue
            steers = [r['steerability_score'] for r in runs]
            mean_s = np.mean(steers)
            std_s = np.std(steers)
            dim_means[int(dim_str)] = (mean_s, std_s, len(steers))
            print(f"    dim={dim_str:>3}: S = {mean_s:+.4f} +/- {std_s:.4f} (n={len(steers)})")

        if dim_means:
            best_dim = max(dim_means, key=lambda d: dim_means[d][0])
            dataset_peaks[ds] = {
                'H': h,
                'best_dim': best_dim,
                'best_S': dim_means[best_dim][0],
                'all_dims': {d: v[0] for d, v in dim_means.items()},
            }
            print(f"    -> Peak at dim={best_dim} (S={dim_means[best_dim][0]:+.4f})")

    # Test Goldilocks prediction: peak dim should correlate with H
    if len(dataset_peaks) >= 3:
        hs = [dataset_peaks[ds]['H'] for ds in sorted(dataset_peaks)]
        peaks = [dataset_peaks[ds]['best_dim'] for ds in sorted(dataset_peaks)]
        rho, p = sp_stats.spearmanr(hs, peaks)
        r, p_pearson = sp_stats.pearsonr(hs, peaks)

        print(f"\n  GOLDILOCKS PREDICTION:")
        print(f"    Peak dim vs H: Spearman rho={rho:.3f} (p={p:.4f})")
        print(f"    Peak dim vs H: Pearson r={r:.3f} (p={p_pearson:.4f})")

        if rho > 0.8 and p < 0.05:
            print(f"    *** PASS: Peak shifts with hierarchy complexity ***")
        elif rho > 0.5:
            print(f"    ** Positive trend, needs more data **")
        else:
            print(f"    -> No clear trend (rho={rho:.2f})")

    return dataset_peaks


def analyze_backbone_control():
    """Analyze backbone fine-tuning control experiment."""
    path = RESULTS_DIR / "backbone_finetune_control.json"
    if not path.exists():
        print("  No backbone control results yet")
        return None

    d = json.load(open(path))
    results = d['results']

    print("\n" + "=" * 70)
    print("BACKBONE FINE-TUNING CONTROL ANALYSIS")
    print("=" * 70)

    arm_dataset_steers = {}  # arm -> dataset -> [steers]

    for ds in sorted(results.keys()):
        arms = results[ds]
        print(f"\n  {ds}:")
        print(f"    {'Arm':<18} {'Mean':>8} {'Std':>8} {'N':>4}")
        print(f"    {'-'*40}")

        for arm in ['v5_frozen', 'mrl_frozen', 'flat_finetune', 'v5_finetune']:
            if arm not in arms or not arms[arm]:
                continue
            steers = [r['steerability_score'] for r in arms[arm]]

            if arm not in arm_dataset_steers:
                arm_dataset_steers[arm] = {}
            arm_dataset_steers[arm][ds] = steers

            mean_s = np.mean(steers)
            std_s = np.std(steers)
            print(f"    {arm:<18} {mean_s:>+8.4f} {std_s:>8.4f} {len(steers):>4}")

    # KEY TEST: V5-frozen vs flat-finetune
    print(f"\n  KEY COMPARISONS:")

    verdicts = []
    for ds in sorted(results.keys()):
        v5f = arm_dataset_steers.get('v5_frozen', {}).get(ds, [])
        flatft = arm_dataset_steers.get('flat_finetune', {}).get(ds, [])
        if len(v5f) >= 3 and len(flatft) >= 3:
            t, p = sp_stats.ttest_ind(v5f, flatft)
            gap = np.mean(v5f) - np.mean(flatft)
            d = gap / np.sqrt((np.std(v5f)**2 + np.std(flatft)**2) / 2) if np.std(v5f) + np.std(flatft) > 0 else 0
            print(f"    {ds}: V5-frozen vs flat-finetune: gap={gap:+.4f}, t={t:.2f}, p={p:.4f}, d={d:.2f}")

            if gap > 0.02 and p < 0.05:
                verdicts.append('PASS')
            elif gap > 0:
                verdicts.append('TREND')
            else:
                verdicts.append('FAIL')

    if verdicts:
        n_pass = verdicts.count('PASS')
        n_fail = verdicts.count('FAIL')
        print(f"\n    Overall: {n_pass} PASS, {verdicts.count('TREND')} TREND, {n_fail} FAIL")
        if n_pass > n_fail:
            print(f"    *** EXPERIMENT PASSES: Alignment > capacity ***")
        elif n_fail > n_pass:
            print(f"    !!! DEVASTATING: Capacity alone may suffice !!!")
        else:
            print(f"    ** Mixed results, need more data **")

    # Pooled analysis
    print(f"\n  POOLED ANALYSIS:")
    for arm in ['v5_frozen', 'mrl_frozen', 'flat_finetune', 'v5_finetune']:
        all_steers = []
        for ds_steers in arm_dataset_steers.get(arm, {}).values():
            all_steers.extend(ds_steers)
        if all_steers:
            print(f"    {arm:<18}: {np.mean(all_steers):+.4f} +/- {np.std(all_steers):.4f} (n={len(all_steers)})")

    return arm_dataset_steers


def analyze_deep_hierarchy():
    """Analyze deep hierarchy experiment results."""
    configs = {
        'hupd_sec_cls': {'H': 2.44, 'desc': 'Section->Class'},
        'hupd_sec_sub': {'H': 4.45, 'desc': 'Section->Subclass'},
        'hupd_cls_sub': {'H': None, 'desc': 'Class->Subclass'},
    }

    print("\n" + "=" * 70)
    print("DEEP HIERARCHY EXPERIMENT ANALYSIS")
    print("=" * 70)

    found_any = False
    new_points = []

    for cfg_name, cfg in configs.items():
        path = RESULTS_DIR / f"benchmark_bge-small_{cfg_name}.json"
        if not path.exists():
            print(f"\n  {cfg_name}: No results yet")
            continue

        found_any = True
        d = json.load(open(path))
        v5_data = d.get('v5', {})
        mrl_data = d.get('mrl', {})

        v5_steers = []
        mrl_steers = []
        for seed_key, v in v5_data.items():
            if isinstance(v, dict) and 'prefix_accuracy' in v:
                pa = v['prefix_accuracy']
                s = (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
                    (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))
                v5_steers.append(s)
        for seed_key, v in mrl_data.items():
            if isinstance(v, dict) and 'prefix_accuracy' in v:
                pa = v['prefix_accuracy']
                s = (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
                    (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))
                mrl_steers.append(s)

        print(f"\n  {cfg_name} ({cfg['desc']}, H~{cfg['H']}):")
        if v5_steers:
            print(f"    V5:  {np.mean(v5_steers):+.4f} +/- {np.std(v5_steers):.4f} (n={len(v5_steers)})")
        if mrl_steers:
            print(f"    MRL: {np.mean(mrl_steers):+.4f} +/- {np.std(mrl_steers):.4f} (n={len(mrl_steers)})")

        if v5_steers and mrl_steers:
            gap = np.mean(v5_steers) - np.mean(mrl_steers)
            if len(v5_steers) >= 2 and len(mrl_steers) >= 2:
                t, p = sp_stats.ttest_ind(v5_steers, mrl_steers)
                d_cohen = gap / np.sqrt((np.std(v5_steers)**2 + np.std(mrl_steers)**2) / 2) if np.std(v5_steers) + np.std(mrl_steers) > 0 else 0
                print(f"    Gap: {gap:+.4f}, t={t:.2f}, p={p:.4f}, d={d_cohen:.2f}")

            if cfg['H'] is not None:
                new_points.append({
                    'dataset': cfg_name,
                    'H': cfg['H'],
                    'v5_mean': float(np.mean(v5_steers)),
                    'mrl_mean': float(np.mean(mrl_steers)),
                    'gap': gap,
                })

    if not found_any:
        print("  No deep hierarchy results yet")

    return new_points


def combined_scaling_analysis(new_points=None):
    """Combine existing + new data for scaling analysis."""

    # Existing results from paper (Table 1)
    existing = [
        {'dataset': 'Yahoo', 'H': 1.23, 'v5_mean': 0.015, 'mrl_mean': 0.005},
        {'dataset': 'GoEmotions', 'H': 1.88, 'v5_mean': 0.020, 'mrl_mean': 0.006},
        {'dataset': 'Newsgroups', 'H': 1.88, 'v5_mean': 0.035, 'mrl_mean': 0.000},
        {'dataset': 'TREC', 'H': 2.21, 'v5_mean': 0.044, 'mrl_mean': -0.001},
        {'dataset': 'arXiv', 'H': 2.62, 'v5_mean': 0.027, 'mrl_mean': -0.001},
        {'dataset': 'DBPedia_Cl', 'H': 3.17, 'v5_mean': 0.120, 'mrl_mean': 0.008},
        {'dataset': 'CLINC', 'H': 3.90, 'v5_mean': 0.150, 'mrl_mean': 0.007},
        {'dataset': 'WOS', 'H': 5.05, 'v5_mean': 0.038, 'mrl_mean': 0.001},
    ]

    all_data = existing.copy()
    if new_points:
        all_data.extend(new_points)

    print("\n" + "=" * 70)
    print("COMBINED SCALING ANALYSIS")
    print("=" * 70)

    # V5 steerability vs H
    hs = [d['H'] for d in all_data]
    v5s = [d['v5_mean'] for d in all_data]
    gaps = [d['v5_mean'] - d['mrl_mean'] for d in all_data]

    rho_v5, p_v5 = sp_stats.spearmanr(hs, v5s)
    rho_gap, p_gap = sp_stats.spearmanr(hs, gaps)

    print(f"\n  V5 steerability vs H: rho={rho_v5:.3f} (p={p_v5:.4f})")
    print(f"  V5-MRL gap vs H: rho={rho_gap:.3f} (p={p_gap:.4f})")

    # With new data points highlighted
    for d in all_data:
        marker = " [NEW]" if d in (new_points or []) else ""
        print(f"    {d['dataset']:<15} H={d['H']:<5.2f} V5={d['v5_mean']:+.4f} "
              f"gap={d['v5_mean']-d['mrl_mean']:+.4f}{marker}")

    # Check if WOS outlier explanation holds
    if len(all_data) >= 8:
        # Without WOS
        no_wos = [d for d in all_data if d['dataset'] != 'WOS']
        hs_nw = [d['H'] for d in no_wos]
        v5s_nw = [d['v5_mean'] for d in no_wos]
        rho_nw, p_nw = sp_stats.spearmanr(hs_nw, v5s_nw)
        print(f"\n  Without WOS: rho={rho_nw:.3f} (p={p_nw:.4f})")

    return {
        'rho_v5': rho_v5, 'p_v5': p_v5,
        'rho_gap': rho_gap, 'p_gap': p_gap,
        'n_datasets': len(all_data),
    }


if __name__ == "__main__":
    peaks = analyze_capacity_sweep()
    backbone = analyze_backbone_control()
    deep_points = analyze_deep_hierarchy()
    scaling = combined_scaling_analysis(deep_points)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Capacity sweep: {'Analyzed' if peaks else 'Pending'}")
    print(f"  Backbone control: {'Analyzed' if backbone else 'Pending'}")
    print(f"  Deep hierarchy: {len(deep_points) if deep_points else 0} new data points")
    print(f"  Scaling rho (with new data): {scaling['rho_v5']:.3f}")
