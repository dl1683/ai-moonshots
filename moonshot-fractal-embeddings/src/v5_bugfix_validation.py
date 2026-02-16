"""V5 bugfix validation: run 3 datasets x 5 seeds to verify training bug fix."""
import json
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

DATASETS = ["clinc", "dbpedia_classes", "trec"]
SEEDS = [42, 123, 456, 789, 1337]
MODEL = "bge-small"

def main():
    from fractal_v5 import run_v5_experiment

    results_dir = Path(__file__).parent.parent / "results"
    output_file = results_dir / "v5_bugfix_validation.json"

    # Load existing results if resuming
    all_results = {}
    if output_file.exists():
        with open(output_file) as f:
            all_results = json.load(f)

    total = len(DATASETS) * len(SEEDS)
    done = 0

    for dataset in DATASETS:
        if dataset not in all_results:
            all_results[dataset] = {}

        for seed in SEEDS:
            key = str(seed)
            if key in all_results[dataset] and all_results[dataset][key].get("status") == "ok":
                done += 1
                print(f"[{done}/{total}] SKIP {dataset} seed={seed} (already done)")
                continue

            done += 1
            print(f"\n{'='*70}")
            print(f"[{done}/{total}] {MODEL} | {dataset} | seed={seed}")
            print(f"{'='*70}")

            t0 = time.time()
            try:
                result = run_v5_experiment(
                    model_key=MODEL,
                    dataset_name=dataset,
                    stage1_epochs=5,
                    stage2_epochs=0,
                    batch_size=32,
                    device="cuda",
                    seed=seed,
                )
                elapsed = time.time() - t0

                all_results[dataset][key] = {
                    "status": "ok",
                    "baseline_l0": result["baseline"]["l0_accuracy"],
                    "baseline_l1": result["baseline"]["l1_accuracy"],
                    "v5_l0": result["v5"]["l0_accuracy"],
                    "v5_l1": result["v5"]["l1_accuracy"],
                    "delta_l0": result["delta"]["l0"],
                    "delta_l1": result["delta"]["l1"],
                    "prefix_accuracy": result["prefix_accuracy"],
                    "steerability": (
                        result["prefix_accuracy"]["j4_l1"] - result["prefix_accuracy"]["j1_l1"]
                    ) + (
                        result["prefix_accuracy"]["j1_l0"] - result["prefix_accuracy"]["j4_l0"]
                    ),
                    "runtime_sec": elapsed,
                    "seed": seed,
                }

                print(f"  Delta L0: {result['delta']['l0']:+.4f}")
                print(f"  Delta L1: {result['delta']['l1']:+.4f}")
                print(f"  Steerability: {all_results[dataset][key]['steerability']:+.4f}")
                print(f"  Runtime: {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - t0
                all_results[dataset][key] = {
                    "status": "error",
                    "error": str(e),
                    "runtime_sec": elapsed,
                    "seed": seed,
                }
                print(f"  ERROR: {e}")

            # Save after each run (resume-safe)
            def convert(obj):
                import numpy as np
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, (float, int, str, bool, type(None))):
                    return obj
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, (list, tuple)):
                    return [convert(x) for x in obj]
                return str(obj)

            with open(output_file, 'w') as f:
                json.dump(convert(all_results), f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("BUGFIX VALIDATION SUMMARY")
    print(f"{'='*70}")

    for dataset in DATASETS:
        if dataset not in all_results:
            continue
        seeds_data = [v for v in all_results[dataset].values() if v.get("status") == "ok"]
        if not seeds_data:
            print(f"\n{dataset}: NO VALID RUNS")
            continue

        import numpy as np
        delta_l0s = [d["delta_l0"] for d in seeds_data]
        delta_l1s = [d["delta_l1"] for d in seeds_data]
        steers = [d["steerability"] for d in seeds_data]

        print(f"\n{dataset} ({len(seeds_data)} seeds):")
        print(f"  Delta L0: {np.mean(delta_l0s):+.4f} +/- {np.std(delta_l0s):.4f}")
        print(f"  Delta L1: {np.mean(delta_l1s):+.4f} +/- {np.std(delta_l1s):.4f}")
        print(f"  Steerability: {np.mean(steers):+.4f} +/- {np.std(steers):.4f}")

        # Compare to pre-bugfix if available
        pre_bugfix_file = results_dir / f"benchmark_bge-small_{dataset}.json"
        if pre_bugfix_file.exists():
            with open(pre_bugfix_file) as f:
                pre = json.load(f)
            if "steerability" in pre:
                print(f"  Pre-bugfix steerability: {pre['steerability']:+.4f}")
                print(f"  Change: {np.mean(steers) - pre['steerability']:+.4f}")

if __name__ == "__main__":
    main()
