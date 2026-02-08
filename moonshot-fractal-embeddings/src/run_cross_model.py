"""Cross-model replication: Run V5+MRL on CLINC and TREC with Qwen3-0.6B.

Critical for NeurIPS: proves the scaling law is architecture-invariant.
If steerability scales with H(L1|L0) on BOTH bge-small AND Qwen3-0.6B,
the law is universal, not architecture-specific.
"""

import sys
import os
import json
import torch
import gc
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from fractal_v5 import run_v5_experiment
from mrl_v5_baseline import run_mrl_experiment

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Datasets with highest expected contrast in steerability
DATASETS = ["clinc", "trec"]
SEEDS = [42, 123, 456]  # 3 seeds for publication-grade CIs


def compute_steerability(prefix_accuracy):
    """Steer = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)."""
    return (prefix_accuracy.get('j1_l0', 0) - prefix_accuracy.get('j4_l0', 0)) + \
           (prefix_accuracy.get('j4_l1', 0) - prefix_accuracy.get('j1_l1', 0))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_key = "qwen3-0.6b"
    all_results = {}

    for ds_name in DATASETS:
        print(f"\n{'='*70}")
        print(f"  CROSS-MODEL: {model_key} on {ds_name}")
        print(f"{'='*70}")

        # Load existing results to skip completed seeds
        out_path = RESULTS_DIR / f"crossmodel_{model_key}_{ds_name}.json"
        if out_path.exists():
            existing = json.load(open(out_path))
            ds_results = existing
            ds_results["seeds"] = SEEDS
        else:
            ds_results = {"dataset": ds_name, "model": model_key, "seeds": SEEDS}

        # V5
        v5_results = ds_results.get('v5', {})
        for seed in SEEDS:
            if str(seed) in v5_results:
                pa = v5_results[str(seed)].get('prefix_accuracy', {})
                steer = compute_steerability(pa)
                print(f"\n  V5 {ds_name} seed={seed} — CACHED, steer={steer:+.4f}")
                continue
            print(f"\n  V5 {ds_name} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                r = run_v5_experiment(
                    model_key=model_key,
                    dataset_name=ds_name,
                    stage1_epochs=5,
                    stage2_epochs=0,
                    batch_size=16,  # Smaller batch for Qwen3 (larger model)
                    device=device,
                    seed=seed,
                )
                v5_results[str(seed)] = r
                pa = r.get('prefix_accuracy', {})
                steer = compute_steerability(pa)
                print(f"  V5 seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"  V5 ERROR: {e}")
                import traceback
                traceback.print_exc()

        ds_results['v5'] = v5_results

        # MRL
        mrl_results = ds_results.get('mrl', {})
        for seed in SEEDS:
            if str(seed) in mrl_results:
                pa = mrl_results[str(seed)].get('prefix_accuracy', {})
                steer = compute_steerability(pa)
                print(f"\n  MRL {ds_name} seed={seed} — CACHED, steer={steer:+.4f}")
                continue
            print(f"\n  MRL {ds_name} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                r = run_mrl_experiment(
                    model_key=model_key,
                    dataset_name=ds_name,
                    stage1_epochs=5,
                    stage2_epochs=0,
                    batch_size=16,
                    device=device,
                    seed=seed,
                )
                mrl_results[str(seed)] = r
                pa = r.get('prefix_accuracy', {})
                steer = compute_steerability(pa)
                print(f"  MRL seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"  MRL ERROR: {e}")
                import traceback
                traceback.print_exc()

        ds_results['mrl'] = mrl_results
        all_results[ds_name] = ds_results

        # Save per-dataset (out_path already set above)
        with open(out_path, 'w') as f:
            json.dump(ds_results, f, indent=2,
                     default=lambda x: float(x) if hasattr(x, 'item') else x)
        print(f"  Saved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL REPLICATION SUMMARY ({model_key})")
    print(f"{'='*70}")

    profiles = json.load(open(RESULTS_DIR / "hierarchy_profiles.json"))

    print(f"\n  {'Dataset':<12} {'H(L1|L0)':<10} {'V5 Steer':<12} {'MRL Steer':<12} {'Gap'}")
    print(f"  {'-'*55}")

    for ds_name in DATASETS:
        h = profiles.get(ds_name, {}).get('h_l1_given_l0', 0)
        dr = all_results.get(ds_name, {})
        v5_steers = []
        mrl_steers = []
        for sd in dr.get('v5', {}).values():
            if isinstance(sd, dict) and 'prefix_accuracy' in sd:
                v5_steers.append(compute_steerability(sd['prefix_accuracy']))
        for sd in dr.get('mrl', {}).values():
            if isinstance(sd, dict) and 'prefix_accuracy' in sd:
                mrl_steers.append(compute_steerability(sd['prefix_accuracy']))

        v5m = np.mean(v5_steers) if v5_steers else float('nan')
        mrlm = np.mean(mrl_steers) if mrl_steers else float('nan')
        print(f"  {ds_name:<12} {h:<10.3f} {v5m:+.4f}      {mrlm:+.4f}      {v5m-mrlm:+.4f}")

    # Compare with bge-small
    print(f"\n  COMPARISON: bge-small vs {model_key}")
    print(f"  {'Dataset':<12} {'H':<8} {'bge V5':<10} {'Qwen V5':<10} {'bge MRL':<10} {'Qwen MRL'}")
    print(f"  {'-'*55}")

    bge_data = {}
    for ds_name in DATASETS:
        bench_file = RESULTS_DIR / f"benchmark_bge-small_{ds_name}.json"
        if bench_file.exists():
            bd = json.load(open(bench_file))
            v5s = []
            for sd in bd.get('v5', {}).values():
                if isinstance(sd, dict) and 'prefix_accuracy' in sd:
                    v5s.append(compute_steerability(sd['prefix_accuracy']))
            mrls = []
            for sd in bd.get('mrl', {}).values():
                if isinstance(sd, dict) and 'prefix_accuracy' in sd:
                    mrls.append(compute_steerability(sd['prefix_accuracy']))
            bge_data[ds_name] = {
                'v5': np.mean(v5s) if v5s else float('nan'),
                'mrl': np.mean(mrls) if mrls else float('nan'),
            }
        # Check ablation file for CLINC
        elif ds_name == 'clinc':
            abl_file = RESULTS_DIR / "ablation_steerability_bge-small_clinc.json"
            if abl_file.exists():
                ad = json.load(open(abl_file))
                steers = []
                for r in ad['results'].get('v5', []):
                    if 'prefix_results' in r:
                        pr = r['prefix_results']
                        pa = {'j1_l0': pr['j1']['l0'], 'j1_l1': pr['j1']['l1'],
                              'j4_l0': pr['j4']['l0'], 'j4_l1': pr['j4']['l1']}
                        steers.append(compute_steerability(pa))
                bge_data[ds_name] = {
                    'v5': np.mean(steers) if steers else float('nan'),
                    'mrl': float('nan'),
                }

    for ds_name in DATASETS:
        h = profiles.get(ds_name, {}).get('h_l1_given_l0', 0)
        bge = bge_data.get(ds_name, {})
        qwen = all_results.get(ds_name, {})
        bge_v5 = bge.get('v5', float('nan'))
        qwen_v5_steers = [compute_steerability(sd['prefix_accuracy'])
                          for sd in qwen.get('v5', {}).values()
                          if isinstance(sd, dict) and 'prefix_accuracy' in sd]
        qwen_v5 = np.mean(qwen_v5_steers) if qwen_v5_steers else float('nan')
        bge_mrl = bge.get('mrl', float('nan'))
        qwen_mrl_steers = [compute_steerability(sd['prefix_accuracy'])
                           for sd in qwen.get('mrl', {}).values()
                           if isinstance(sd, dict) and 'prefix_accuracy' in sd]
        qwen_mrl = np.mean(qwen_mrl_steers) if qwen_mrl_steers else float('nan')
        print(f"  {ds_name:<12} {h:<8.3f} {bge_v5:+.4f}    {qwen_v5:+.4f}    {bge_mrl:+.4f}    {qwen_mrl:+.4f}")


if __name__ == "__main__":
    main()
