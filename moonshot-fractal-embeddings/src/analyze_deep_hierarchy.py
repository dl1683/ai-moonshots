"""Analyze deep hierarchy experiment results.

Computes statistics for all completed deep hierarchy configs and generates
paper-ready summary table + scaling law validation against product predictor.

Usage: python src/analyze_deep_hierarchy.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Expected configs and their H(L1|L0)
DEEP_CONFIGS = {
    'hupd_sec_cls': {'description': 'HUPD Section->Class', 'H': 2.42, 'K0': 8, 'K1': 121},
    'hupd_sec_sub': {'description': 'HUPD Section->Subclass', 'H': 4.44, 'K0': 8, 'K1': 587},
    'hwv_l0_l2': {'description': 'HWV Root->L2', 'H': 4.09, 'K0': 10, 'K1': 253},
    'hwv_l0_l3': {'description': 'HWV Root->L3', 'H': 4.59, 'K0': 10, 'K1': 230},
}


def compute_steerability(prefix_data):
    """S = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)"""
    return (prefix_data['j1_l0'] - prefix_data['j4_l0']) + \
           (prefix_data['j4_l1'] - prefix_data['j1_l1'])


def load_config_results(config_name):
    """Load results for a deep hierarchy config."""
    bench_file = RESULTS_DIR / f"benchmark_bge-small_{config_name}.json"
    if not bench_file.exists():
        return None

    data = json.load(open(bench_file))

    # Parse structure
    v5_steers = []
    mrl_steers = []
    baseline = None
    seeds_found = []

    # Check structure: could be nested under 'v5'/'mrl' or different
    if 'v5' in data and 'mrl' in data:
        for seed_str, seed_data in data['v5'].items():
            if isinstance(seed_data, dict) and 'prefix_accuracy' in seed_data:
                S = compute_steerability(seed_data['prefix_accuracy'])
                v5_steers.append(S)
                seeds_found.append(int(seed_str))
                if baseline is None:
                    baseline = seed_data.get('baseline', {})

        for seed_str, seed_data in data['mrl'].items():
            if isinstance(seed_data, dict) and 'prefix_accuracy' in seed_data:
                S = compute_steerability(seed_data['prefix_accuracy'])
                mrl_steers.append(S)

    return {
        'v5_steers': np.array(v5_steers),
        'mrl_steers': np.array(mrl_steers),
        'baseline': baseline,
        'n_seeds_v5': len(v5_steers),
        'n_seeds_mrl': len(mrl_steers),
        'seeds': seeds_found,
    }


def main():
    print("=" * 70)
    print("DEEP HIERARCHY ANALYSIS")
    print("=" * 70)

    all_results = {}

    for config_name, config_info in DEEP_CONFIGS.items():
        results = load_config_results(config_name)
        if results is None:
            print(f"\n{config_info['description']}: NOT FOUND")
            continue

        all_results[config_name] = results

        print(f"\n{config_info['description']} ({config_name})")
        print(f"  H(L1|L0) = {config_info['H']:.2f} bits, K0={config_info['K0']}, K1={config_info['K1']}")

        if results['baseline']:
            bl = results['baseline']
            print(f"  Baseline: L0={bl.get('l0_accuracy', 'N/A')}, L1={bl.get('l1_accuracy', 'N/A')}")

        v5 = results['v5_steers']
        mrl = results['mrl_steers']

        print(f"  V5:  n={len(v5)}, S = {v5.mean():+.4f} +/- {v5.std():.4f}")
        print(f"       Seeds: {[f'{s:+.3f}' for s in v5]}")
        print(f"  MRL: n={len(mrl)}, S = {mrl.mean():+.4f} +/- {mrl.std():.4f}")
        print(f"       Seeds: {[f'{s:+.3f}' for s in mrl]}")

        if len(v5) >= 3 and len(mrl) >= 3:
            n = min(len(v5), len(mrl))
            t_stat, p_val = stats.ttest_rel(v5[:n], mrl[:n])
            diff = v5[:n] - mrl[:n]
            d = diff.mean() / diff.std() if diff.std() > 0 else float('inf')
            print(f"  Paired t-test: t={t_stat:.2f}, p={p_val:.4f}, d={d:.1f}")

    # Scaling law analysis with product predictor
    if len(all_results) >= 3:
        print("\n" + "=" * 70)
        print("SCALING LAW: Product Predictor for Deep Hierarchies")
        print("=" * 70)

        H_values = []
        S_values = []
        base_l1_values = []
        dataset_names = []

        for config_name, results in all_results.items():
            config_info = DEEP_CONFIGS[config_name]
            v5 = results['v5_steers']
            if len(v5) >= 3:
                bl = results['baseline']
                bl_l1 = bl.get('l1_accuracy', 0) if bl else 0
                H = config_info['H']
                S = v5.mean()

                H_values.append(H)
                S_values.append(S)
                base_l1_values.append(bl_l1)
                dataset_names.append(config_name)

                print(f"  {config_name}: H={H:.2f}, S={S:+.4f}, base_L1={bl_l1:.3f}, product={H*bl_l1:.3f}")

        if len(H_values) >= 3:
            H_arr = np.array(H_values)
            S_arr = np.array(S_values)
            bl_arr = np.array(base_l1_values)
            product = H_arr * bl_arr

            # Correlations
            rho_H, p_H = stats.spearmanr(H_arr, S_arr)
            rho_prod, p_prod = stats.spearmanr(product, S_arr)
            r_H, p_rH = stats.pearsonr(H_arr, S_arr)
            r_prod, p_rprod = stats.pearsonr(product, S_arr)

            print(f"\n  H alone:    Spearman rho={rho_H:.3f} (p={p_H:.3f}), Pearson r={r_H:.3f} (p={p_rH:.3f})")
            print(f"  Product:    Spearman rho={rho_prod:.3f} (p={p_prod:.3f}), Pearson r={r_prod:.3f} (p={p_rprod:.3f})")

    # Summary table for paper
    print("\n" + "=" * 70)
    print("PAPER TABLE (LaTeX)")
    print("=" * 70)
    print("\\begin{tabular}{llcccccc}")
    print("\\toprule")
    print("Dataset & $\\hlo$ & $K_0$ & $K_1$ & V5 $\\steer$ & MRL $\\steer$ & $p$ & $d$ \\\\")
    print("\\midrule")

    for config_name, config_info in DEEP_CONFIGS.items():
        if config_name not in all_results:
            continue
        results = all_results[config_name]
        v5 = results['v5_steers']
        mrl = results['mrl_steers']

        if len(v5) >= 3 and len(mrl) >= 3:
            n = min(len(v5), len(mrl))
            t_stat, p_val = stats.ttest_rel(v5[:n], mrl[:n])
            diff = v5[:n] - mrl[:n]
            d = diff.mean() / diff.std() if diff.std() > 0 else float('inf')

            desc = config_info['description'].replace('HUPD ', '').replace('HWV ', '')
            H = config_info['H']
            K0 = config_info['K0']
            K1 = config_info['K1']

            sig = ''
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'

            print(f"{desc} & {H:.2f} & {K0} & {K1} & "
                  f"${v5.mean():+.3f} \\pm {v5.std():.3f}$ & "
                  f"${mrl.mean():+.3f} \\pm {mrl.std():.3f}$ & "
                  f"{p_val:.3f}{sig} & {d:.1f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
