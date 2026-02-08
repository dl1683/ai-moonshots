"""
Master Runner: All Remaining Experiments for NeurIPS 2026
==========================================================

Addresses Codex review (4/10 -> target 7+/10):
1. Downstream eval (CLINC, TREC) with retrieval + tree-distance + MI profile
2. Additional benchmarks (GoEmotions, arXiv) to get n=6 datasets for scaling law
3. All with 3 seeds for proper CIs

Run order (GPU-efficient):
1. downstream_eval on CLINC (3 seeds) — highest priority, largest effect
2. downstream_eval on TREC (3 seeds) — second priority
3. GoEmotions V5+MRL benchmark (3 seeds) — new dataset for scaling law
4. arXiv V5+MRL benchmark (3 seeds) — another new dataset
"""

import sys
import os
import json
import torch
import gc
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_downstream_evals():
    """Run downstream eval on key datasets."""
    from downstream_eval import run_downstream_eval

    for ds in ["clinc", "trec"]:
        print(f"\n{'#'*70}")
        print(f"# DOWNSTREAM EVAL: bge-small on {ds}")
        print(f"{'#'*70}")

        result = run_downstream_eval(
            model_key="bge-small",
            dataset_name=ds,
            seeds=(42, 123, 456),
            k=10,
        )

        torch.cuda.empty_cache()
        gc.collect()


def run_new_benchmarks():
    """Run V5+MRL on new datasets for scaling law."""
    from fractal_v5 import run_v5_experiment
    from mrl_v5_baseline import run_mrl_experiment

    datasets = ["goemotions", "arxiv"]
    seeds = [42, 123, 456]

    for ds in datasets:
        print(f"\n{'#'*70}")
        print(f"# BENCHMARK: bge-small on {ds}")
        print(f"{'#'*70}")

        all_v5 = {}
        all_mrl = {}

        for seed in seeds:
            # V5
            print(f"\n  V5 {ds} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                r = run_v5_experiment(
                    model_key="bge-small",
                    dataset_name=ds,
                    stage1_epochs=5,
                    stage2_epochs=0,
                    batch_size=32,
                    device="cuda",
                    seed=seed,
                )
                all_v5[str(seed)] = r
                pa = r.get('prefix_accuracy', {})
                steer = (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
                        (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))
                print(f"    V5 seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"    V5 ERROR: {e}")
                import traceback
                traceback.print_exc()

            # MRL
            print(f"\n  MRL {ds} seed={seed}")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                r = run_mrl_experiment(
                    model_key="bge-small",
                    dataset_name=ds,
                    stage1_epochs=5,
                    stage2_epochs=0,
                    batch_size=32,
                    device="cuda",
                    seed=seed,
                )
                all_mrl[str(seed)] = r
                pa = r.get('prefix_accuracy', {})
                steer = (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
                        (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))
                print(f"    MRL seed {seed}: steer={steer:+.4f}")
            except Exception as e:
                print(f"    MRL ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Save
        out = {
            "model": "bge-small",
            "dataset": ds,
            "v5": all_v5,
            "mrl": all_mrl,
        }
        out_path = RESULTS_DIR / f"benchmark_bge-small_{ds}.json"
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2,
                     default=lambda x: float(x) if hasattr(x, 'item') else x)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    print("="*70)
    print("MASTER RUNNER: All remaining experiments for NeurIPS 2026")
    print("="*70)

    # Phase 1: Downstream evaluations (highest priority)
    run_downstream_evals()

    # Phase 2: New dataset benchmarks
    run_new_benchmarks()

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
