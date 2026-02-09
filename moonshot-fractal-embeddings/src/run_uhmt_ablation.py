"""Run UHMT (Uniform Hierarchical Multi-Task) ablation on CLINC and TREC.

UHMT gives every prefix length identical 0.5*L0 + 0.5*L1 supervision.
This tests whether hierarchy AWARENESS alone produces steerability,
or if V5's hierarchy ALIGNMENT is required.

Prediction: UHMT steerability ~0 (near MRL/no_prefix levels).
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Force unbuffered output for background execution
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))

from ablation_steerability import (
    UniformHierarchicalTrainer,
    run_single_ablation,
)


def run_uhmt_experiments():
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {"dataset": "clinc", "seeds": [42, 123, 456, 789, 1024]},
        {"dataset": "trec", "seeds": [42, 123, 456]},
    ]

    for cfg in configs:
        dataset = cfg["dataset"]
        seeds = cfg["seeds"]

        print(f"\n{'='*70}")
        print(f"  UHMT ABLATION: {dataset} ({len(seeds)} seeds)")
        print(f"{'='*70}")

        all_results = []
        for seed in seeds:
            result = run_single_ablation(
                ablation_name="uhmt",
                trainer_class=UniformHierarchicalTrainer,
                model_key="bge-small",
                dataset_name=dataset,
                stage1_epochs=5,
                batch_size=32,
                seed=seed,
                device="cuda",
            )
            all_results.append(result)
            torch.cuda.empty_cache()

        # Compute summary stats
        steers = [r['steerability_score'] for r in all_results]
        sc = [r['short_coarse'] for r in all_results]
        ff = [r['full_fine'] for r in all_results]
        sg = [r['specialization_gap'] for r in all_results]

        print(f"\n{'='*70}")
        print(f"  UHMT SUMMARY: {dataset}")
        print(f"{'='*70}")
        print(f"  Steerability: {np.mean(steers):+.4f} +/- {np.std(steers, ddof=1):.4f}")
        print(f"  ShortCoarse:  {np.mean(sc):.4f} +/- {np.std(sc, ddof=1):.4f}")
        print(f"  FullFine:     {np.mean(ff):.4f} +/- {np.std(ff, ddof=1):.4f}")
        print(f"  SpecGap:      {np.mean(sg):+.4f} +/- {np.std(sg, ddof=1):.4f}")

        # Save
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        output = {
            "ablation": "uhmt",
            "model": "bge-small",
            "dataset": dataset,
            "seeds": seeds,
            "timestamp": datetime.now().isoformat(),
            "results": convert(all_results),
            "summary": {
                "steerability_mean": float(np.mean(steers)),
                "steerability_std": float(np.std(steers, ddof=1)),
                "short_coarse_mean": float(np.mean(sc)),
                "full_fine_mean": float(np.mean(ff)),
                "spec_gap_mean": float(np.mean(sg)),
            },
        }

        out_path = results_dir / f"uhmt_ablation_bge-small_{dataset}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    run_uhmt_experiments()
