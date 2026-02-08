"""
Run remaining ablations (inverted + no_prefix) on CLINC bge-small.
V5 control already done with 5 seeds. Hardcoded results below.
"""
import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from ablation_steerability import (
    InvertedV5Trainer, NoPrefixV5Trainer, V5Trainer,
    run_single_ablation, evaluate_prefix_steerability,
)
from hierarchical_datasets import load_hierarchical_dataset

# V5 control results (already completed from first run)
V5_RESULTS = [
    {'ablation': 'v5', 'seed': 42, 'steerability_score': 0.0480, 'specialization_gap': 0.0895, 'short_coarse': 0.9720, 'full_fine': 0.9395,
     'prefix_results': {'j1': {'l0': 0.9720, 'l1': 0.8825}, 'j2': {'l0': 0.9770, 'l1': 0.9355}, 'j3': {'l0': 0.9795, 'l1': 0.9430}, 'j4': {'l0': 0.9810, 'l1': 0.9395}},
     'coarse_gain': 0.9720-0.9810, 'fine_gain': 0.9395-0.8825},
    {'ablation': 'v5', 'seed': 123, 'steerability_score': 0.0530, 'specialization_gap': 0.0915, 'short_coarse': 0.9760, 'full_fine': 0.9460,
     'prefix_results': {'j1': {'l0': 0.9760, 'l1': 0.8845}, 'j2': {'l0': 0.9800, 'l1': 0.9355}, 'j3': {'l0': 0.9845, 'l1': 0.9480}, 'j4': {'l0': 0.9845, 'l1': 0.9460}},
     'coarse_gain': 0.9760-0.9845, 'fine_gain': 0.9460-0.8845},
    {'ablation': 'v5', 'seed': 456, 'steerability_score': 0.0525, 'specialization_gap': 0.0880, 'short_coarse': 0.9735, 'full_fine': 0.9465,
     'prefix_results': {'j1': {'l0': 0.9735, 'l1': 0.8855}, 'j2': {'l0': 0.9820, 'l1': 0.9320}, 'j3': {'l0': 0.9820, 'l1': 0.9450}, 'j4': {'l0': 0.9820, 'l1': 0.9465}},
     'coarse_gain': 0.9735-0.9820, 'fine_gain': 0.9465-0.8855},
    {'ablation': 'v5', 'seed': 789, 'steerability_score': 0.0580, 'specialization_gap': 0.0930, 'short_coarse': 0.9790, 'full_fine': 0.9475,
     'prefix_results': {'j1': {'l0': 0.9790, 'l1': 0.8860}, 'j2': {'l0': 0.9805, 'l1': 0.9425}, 'j3': {'l0': 0.9835, 'l1': 0.9470}, 'j4': {'l0': 0.9825, 'l1': 0.9475}},
     'coarse_gain': 0.9790-0.9825, 'fine_gain': 0.9475-0.8860},
    {'ablation': 'v5', 'seed': 1024, 'steerability_score': 0.0555, 'specialization_gap': 0.0895, 'short_coarse': 0.9750, 'full_fine': 0.9495,
     'prefix_results': {'j1': {'l0': 0.9750, 'l1': 0.8855}, 'j2': {'l0': 0.9825, 'l1': 0.9375}, 'j3': {'l0': 0.9845, 'l1': 0.9430}, 'j4': {'l0': 0.9835, 'l1': 0.9495}},
     'coarse_gain': 0.9750-0.9835, 'fine_gain': 0.9495-0.8855},
]

SEEDS = [42, 123, 456, 789, 1024]

print("="*70)
print("  CAUSAL ABLATION STUDY — INVERTED + NO_PREFIX")
print("  V5 control results loaded from completed run (5 seeds)")
print("="*70)

all_results = {'v5': V5_RESULTS}

# Run inverted and no_prefix
for ablation_name, trainer_class in [
    ('inverted', InvertedV5Trainer),
    ('no_prefix', NoPrefixV5Trainer),
]:
    all_results[ablation_name] = []
    for seed in SEEDS:
        result = run_single_ablation(
            ablation_name=ablation_name,
            trainer_class=trainer_class,
            model_key="bge-small",
            dataset_name="clinc",
            stage1_epochs=5,
            batch_size=32,
            seed=seed,
            device="cuda",
        )
        all_results[ablation_name].append(result)


# Summary
print("\n" + "=" * 70)
print("CAUSAL ABLATION SUMMARY (5 seeds each)")
print("=" * 70)
print(f"{'Ablation':<15} {'ShortCoarse':>12} {'FullFine':>10} {'SpecGap':>10} {'Steerability':>13}")
print("-" * 60)

for name, results in all_results.items():
    sc = np.mean([r['short_coarse'] for r in results])
    ff = np.mean([r['full_fine'] for r in results])
    sg = np.mean([r['specialization_gap'] for r in results])
    ss = np.mean([r['steerability_score'] for r in results])
    sc_std = np.std([r['short_coarse'] for r in results])
    ss_std = np.std([r['steerability_score'] for r in results])
    print(f"{name:<15} {sc:>8.4f}+/-{sc_std:.4f} {ff:>10.4f} {sg:>10.4f} {ss:>+9.4f}+/-{ss_std:.4f}")

# Codex pass/fail criteria
print()
print("CODEX PASS/FAIL CRITERIA:")
v5_ss = np.mean([r['steerability_score'] for r in all_results['v5']])
inv_ss = np.mean([r['steerability_score'] for r in all_results.get('inverted', [{'steerability_score': 0}])])
nop_ss = np.mean([r['steerability_score'] for r in all_results.get('no_prefix', [{'steerability_score': 0}])])

inv_pass = inv_ss < -0.05
print(f"  INVERTED: Steerability={inv_ss:+.4f}  Criterion: < -0.05  => {'PASS' if inv_pass else 'FAIL'}")

nop_pass = abs(nop_ss) <= 0.02
print(f"  NO_PREFIX: Steerability={nop_ss:+.4f}  Criterion: |x| <= 0.02  => {'PASS' if nop_pass else 'FAIL'}")

if inv_pass and nop_pass:
    print("\n  *** CAUSAL MECHANISM CONFIRMED ***")
elif inv_pass or nop_pass:
    print("\n  ** PARTIAL CAUSAL EVIDENCE **")
else:
    print("\n  * INCONCLUSIVE — need to investigate *")

# Save
results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

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
    'model': 'bge-small',
    'dataset': 'clinc',
    'seeds': SEEDS,
    'timestamp': datetime.now().isoformat(),
    'results': convert(all_results),
}
out_path = results_dir / "ablation_steerability_bge-small_clinc.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
