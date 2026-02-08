"""
Paradigm-Shift Experiment Runner
=================================

Runs ALL experiments needed to move from 4/10 to 8+/10:

Phase 1: Prefix Surgery (CLINC, bge-small) — ~10 min
Phase 2: CIFAR-100 Vision (ResNet-18) — ~15 min
Phase 3: Downstream Eval (CLINC, bge-small, 3 seeds) — ~30 min
Phase 4: GoEmotions benchmark (bge-small, 3 seeds) — ~20 min

Total: ~1.5 hours on RTX 5090
"""

import sys
import os
import torch
import gc

sys.path.insert(0, os.path.dirname(__file__))


def phase1_prefix_surgery():
    """Run the paradigm-level prefix surgery experiment."""
    print("\n" + "#"*70)
    print("# PHASE 1: CAUSAL PREFIX SURGERY")
    print("#"*70)
    from prefix_surgery import run_prefix_surgery
    run_prefix_surgery(
        model_key="bge-small",
        dataset_name="clinc",
        seed=42,
        n_pairs=500,
    )
    torch.cuda.empty_cache()
    gc.collect()


def phase2_vision():
    """Run CIFAR-100 vision fractal experiment."""
    print("\n" + "#"*70)
    print("# PHASE 2: CIFAR-100 VISION FRACTAL")
    print("#"*70)
    from vision_fractal import run_vision_experiment
    run_vision_experiment(backbone="resnet18", epochs=15, seed=42)
    torch.cuda.empty_cache()
    gc.collect()


def phase3_downstream():
    """Run downstream eval on CLINC."""
    print("\n" + "#"*70)
    print("# PHASE 3: DOWNSTREAM EVALUATION (CLINC)")
    print("#"*70)
    from downstream_eval import run_downstream_eval
    run_downstream_eval(
        model_key="bge-small",
        dataset_name="clinc",
        seeds=(42, 123, 456),
        k=10,
    )
    torch.cuda.empty_cache()
    gc.collect()


def phase4_new_benchmarks():
    """Run GoEmotions benchmark for 5th dataset."""
    print("\n" + "#"*70)
    print("# PHASE 4: GoEmotions BENCHMARK")
    print("#"*70)
    from fractal_v5 import run_v5_experiment
    from mrl_v5_baseline import run_mrl_experiment
    import json
    import numpy as np
    from pathlib import Path

    RESULTS_DIR = Path(__file__).parent.parent / "results"
    ds = "goemotions"
    seeds = [42, 123, 456]
    all_v5, all_mrl = {}, {}

    for seed in seeds:
        print(f"\n  V5 {ds} seed={seed}")
        torch.cuda.empty_cache()
        gc.collect()
        try:
            r = run_v5_experiment(
                model_key="bge-small", dataset_name=ds,
                stage1_epochs=5, stage2_epochs=0, batch_size=32,
                device="cuda", seed=seed,
            )
            all_v5[str(seed)] = r
            pa = r.get('prefix_accuracy', {})
            steer = (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
                    (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))
            print(f"    V5 seed {seed}: steer={steer:+.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

        print(f"\n  MRL {ds} seed={seed}")
        torch.cuda.empty_cache()
        gc.collect()
        try:
            r = run_mrl_experiment(
                model_key="bge-small", dataset_name=ds,
                stage1_epochs=5, stage2_epochs=0, batch_size=32,
                device="cuda", seed=seed,
            )
            all_mrl[str(seed)] = r
            pa = r.get('prefix_accuracy', {})
            steer = (pa.get('j1_l0', 0) - pa.get('j4_l0', 0)) + \
                    (pa.get('j4_l1', 0) - pa.get('j1_l1', 0))
            print(f"    MRL seed {seed}: steer={steer:+.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    out_path = RESULTS_DIR / f"benchmark_bge-small_{ds}.json"
    with open(out_path, 'w') as f:
        json.dump({"model": "bge-small", "dataset": ds, "v5": all_v5, "mrl": all_mrl},
                 f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    print("="*70)
    print("PARADIGM-SHIFT EXPERIMENT RUNNER")
    print("="*70)

    phase1_prefix_surgery()
    phase2_vision()
    phase3_downstream()
    phase4_new_benchmarks()

    print("\n" + "="*70)
    print("ALL PARADIGM EXPERIMENTS COMPLETE!")
    print("="*70)
