"""Deep Hierarchy Experiments: HUPD at multiple hierarchy depths.

Tests the core prediction: V5 steerability advantage grows with
hierarchy depth (conditional entropy).

Three HUPD configurations:
1. hupd_sec_cls: Section(8) -> Class(121), H~2.44 bits
2. hupd_sec_sub: Section(8) -> Subclass(587), H~4.45 bits
3. hupd_cls_sub: Class(121) -> Subclass(587), H varies

Each runs V5 + MRL with 5 seeds on bge-small.

Run: python src/run_deep_hierarchy_experiments.py [config_name|all]
"""

import sys
import os
import json
import torch
import gc
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_SEEDS = [42, 123, 456, 789, 1024]
MODEL_KEY = "bge-small"

# Deep hierarchy configurations
DEEP_CONFIGS = {
    'hupd_sec_cls': {
        'dataset_name': 'hupd_sec_cls',
        'description': 'HUPD Section(8) -> Class(121)',
        'expected_H': 2.44,
        'max_train': 15000,
        'max_test': 3000,
    },
    'hupd_sec_sub': {
        'dataset_name': 'hupd_sec_sub',
        'description': 'HUPD Section(8) -> Subclass(587)',
        'expected_H': 4.45,
        'max_train': 30000,  # More samples needed for 587 classes
        'max_test': 5000,
    },
    'hwv_l0_l2': {
        'dataset_name': 'hwv_l0_l2',
        'description': 'HWV Root(10) -> L2(381)',
        'expected_H': 4.36,
        'max_train': 8000,   # ~9K available
        'max_test': 2000,
    },
    'hwv_l0_l3': {
        'dataset_name': 'hwv_l0_l3',
        'description': 'HWV Root(10) -> L3(437)',
        'expected_H': 5.08,
        'max_train': 5000,   # ~6K available at depth 4+
        'max_test': 1500,
    },
}


def compute_steer(pa):
    return (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
           (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))


def run_config(config_name, config, seeds=None):
    """Run V5 + MRL for a single deep hierarchy config."""
    from fractal_v5 import run_v5_experiment
    from mrl_v5_baseline import run_mrl_experiment

    if seeds is None:
        seeds = ALL_SEEDS

    ds_name = config['dataset_name']
    bench_file = RESULTS_DIR / f"benchmark_{MODEL_KEY}_{ds_name}.json"

    # Load existing
    existing = {}
    if bench_file.exists():
        existing = json.load(open(bench_file))
    all_v5 = existing.get('v5', {})
    all_mrl = existing.get('mrl', {})

    print(f"\n{'=' * 60}")
    print(f"  {config['description']}")
    print(f"  Expected H(L1|L0) ~ {config['expected_H']} bits")
    print(f"  Train: {config['max_train']}, Test: {config['max_test']}")
    print(f"{'=' * 60}")

    for seed in seeds:
        seed_key = str(seed)

        # V5
        if seed_key in all_v5 and isinstance(all_v5[seed_key], dict) and 'prefix_accuracy' in all_v5[seed_key]:
            pa = all_v5[seed_key]['prefix_accuracy']
            steer = compute_steer(pa)
            print(f"  V5 {ds_name} seed={seed} -> CACHED, steer={steer:+.4f}")
        else:
            print(f"\n  V5 {ds_name} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                r = run_v5_experiment(
                    model_key=MODEL_KEY, dataset_name=ds_name,
                    stage1_epochs=5, stage2_epochs=0, batch_size=16,
                    device="cuda", seed=seed,
                    max_train_samples=config['max_train'],
                    max_test_samples=config['max_test'],
                )
                all_v5[seed_key] = r
                pa = r.get('prefix_accuracy', {})
                steer = compute_steer(pa)
                print(f"    V5 seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"    V5 ERROR: {e}")
                import traceback; traceback.print_exc()

        # MRL
        if seed_key in all_mrl and isinstance(all_mrl[seed_key], dict) and 'prefix_accuracy' in all_mrl[seed_key]:
            pa = all_mrl[seed_key]['prefix_accuracy']
            steer = compute_steer(pa)
            print(f"  MRL {ds_name} seed={seed} -> CACHED, steer={steer:+.4f}")
        else:
            print(f"\n  MRL {ds_name} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                r = run_mrl_experiment(
                    model_key=MODEL_KEY, dataset_name=ds_name,
                    stage1_epochs=5, stage2_epochs=0, batch_size=16,
                    device="cuda", seed=seed,
                    max_train_samples=config['max_train'],
                    max_test_samples=config['max_test'],
                )
                all_mrl[seed_key] = r
                pa = r.get('prefix_accuracy', {})
                steer = compute_steer(pa)
                print(f"    MRL seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"    MRL ERROR: {e}")
                import traceback; traceback.print_exc()

        # Incremental save
        out = {
            "model": MODEL_KEY, "dataset": ds_name,
            "config": config['description'],
            "seeds": sorted([int(k) for k in set(list(all_v5.keys()) + list(all_mrl.keys()))]),
            "timestamp": datetime.now().isoformat(),
            "v5": all_v5, "mrl": all_mrl,
        }
        with open(bench_file, 'w') as f:
            json.dump(out, f, indent=2,
                      default=lambda x: float(x) if hasattr(x, 'item') else x)

    # Summary
    v5_steers = [compute_steer(v['prefix_accuracy'])
                 for v in all_v5.values()
                 if isinstance(v, dict) and 'prefix_accuracy' in v]
    mrl_steers = [compute_steer(v['prefix_accuracy'])
                  for v in all_mrl.values()
                  if isinstance(v, dict) and 'prefix_accuracy' in v]

    if v5_steers:
        print(f"\n  {ds_name} V5 steer: {np.mean(v5_steers):+.4f} +/- {np.std(v5_steers):.4f} (n={len(v5_steers)})")
    if mrl_steers:
        print(f"  {ds_name} MRL steer: {np.mean(mrl_steers):+.4f} +/- {np.std(mrl_steers):.4f} (n={len(mrl_steers)})")

    if v5_steers and mrl_steers:
        from scipy import stats as sp_stats
        t, p = sp_stats.ttest_ind(v5_steers, mrl_steers)
        d = (np.mean(v5_steers) - np.mean(mrl_steers)) / np.sqrt(
            (np.std(v5_steers)**2 + np.std(mrl_steers)**2) / 2
        )
        print(f"  t={t:.2f}, p={p:.4f}, d={d:.2f}")

    print(f"  Saved to {bench_file}")
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'config': config_name,
        'v5_mean': float(np.mean(v5_steers)) if v5_steers else None,
        'mrl_mean': float(np.mean(mrl_steers)) if mrl_steers else None,
        'v5_std': float(np.std(v5_steers)) if v5_steers else None,
        'mrl_std': float(np.std(mrl_steers)) if mrl_steers else None,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("DEEP HIERARCHY EXPERIMENTS: HUPD with V5 vs MRL")
    print("=" * 70)

    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target == 'all':
            configs_to_run = list(DEEP_CONFIGS.keys())
        elif target in DEEP_CONFIGS:
            configs_to_run = [target]
        else:
            print(f"Unknown config: {target}. Available: {list(DEEP_CONFIGS.keys()) + ['all']}")
            sys.exit(1)
    else:
        configs_to_run = list(DEEP_CONFIGS.keys())

    all_summaries = []
    for cfg_name in configs_to_run:
        cfg = DEEP_CONFIGS[cfg_name]
        summary = run_config(cfg_name, cfg)
        all_summaries.append(summary)

    # Cross-config analysis
    print(f"\n{'=' * 70}")
    print("CROSS-CONFIG ANALYSIS: Steerability vs Hierarchy Depth")
    print(f"{'=' * 70}")
    for s in all_summaries:
        cfg = DEEP_CONFIGS[s['config']]
        print(f"  {s['config']:<20} H~{cfg['expected_H']:<6} "
              f"V5={s['v5_mean']:+.4f} MRL={s['mrl_mean']:+.4f} "
              f"gap={s['v5_mean']-s['mrl_mean']:+.4f}"
              if s['v5_mean'] is not None and s['mrl_mean'] is not None
              else f"  {s['config']}: incomplete")

    # Save summary
    summary_file = RESULTS_DIR / "deep_hierarchy_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'experiment': 'deep_hierarchy',
            'model': MODEL_KEY,
            'timestamp': datetime.now().isoformat(),
            'summaries': all_summaries,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
