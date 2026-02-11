"""
Backbone Fine-Tuning Control Experiment (Existential Test)
==========================================================

THE QUESTION: If someone fully fine-tunes a backbone with flat supervision,
does emergent scale separation appear? If yes, our hierarchy-alignment
story weakens. If no, V5's alignment is necessary even with more capacity.

Four arms (equal compute):
1. V5-frozen:      Head-only, hierarchy-aligned prefix supervision (current approach)
2. MRL-frozen:     Head-only, flat MRL supervision (current baseline)
3. flat-finetune:  Head + backbone (last 4 layers), flat MRL supervision
4. V5-finetune:    Head + backbone (last 4 layers), hierarchy-aligned supervision

All arms: same epochs, same data, same evaluation.
Steerability measured on test set.

PASS: V5-frozen beats flat-finetune on steerability (alignment > capacity)
DEVASTATING: flat-finetune matches/exceeds V5-frozen (capacity alone suffices)

Datasets: CLINC, DBPedia_Classes, TREC (highest effect sizes from main results)
Seeds: 5 per arm per dataset
"""

import sys
import os
import json
import gc
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from mrl_v5_baseline import MRLTrainerV5
from ablation_steerability import evaluate_prefix_steerability

# Configuration
DATASETS = ['clinc', 'dbpedia_classes', 'trec']
SEEDS = [42, 123, 456, 789, 1024]
MODEL_KEY = "bge-small"
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 5  # Additional backbone fine-tuning epochs
UNFREEZE_LAYERS = 4
BATCH_SIZE = 16
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Arms configuration
ARMS = {
    'v5_frozen': {
        'trainer_class': 'V5Trainer',
        'stage2_epochs': 0,
        'description': 'V5 head-only (current approach)',
    },
    'mrl_frozen': {
        'trainer_class': 'MRLTrainerV5',
        'stage2_epochs': 0,
        'description': 'MRL head-only (current baseline)',
    },
    'flat_finetune': {
        'trainer_class': 'MRLTrainerV5',
        'stage2_epochs': STAGE2_EPOCHS,
        'description': 'MRL with backbone fine-tuning (flat + capacity)',
    },
    'v5_finetune': {
        'trainer_class': 'V5Trainer',
        'stage2_epochs': STAGE2_EPOCHS,
        'description': 'V5 with backbone fine-tuning (alignment + capacity)',
    },
}


def run_single_arm(arm_name, arm_config, dataset_name, seed, device="cuda"):
    """Run a single arm of the experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"ARM: {arm_name} | Dataset: {dataset_name} | Seed: {seed}")
    print(f"  {arm_config['description']}")
    print(f"  Stage2 epochs: {arm_config['stage2_epochs']}")
    print(f"{'='*60}")

    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15, seed=42)

    class TempDataset:
        def __init__(self, samples, level0_names, level1_names):
            self.samples = samples
            self.level0_names = level0_names
            self.level1_names = level1_names

    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_data_obj = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)
    config = MODELS[MODEL_KEY]

    # For MRL, num_l0_classes = num_l1_classes (both heads output L1 logits)
    is_mrl = arm_config['trainer_class'] == 'MRLTrainerV5'
    model_num_l0 = num_l1 if is_mrl else num_l0

    model = FractalModelV5(
        config=config,
        num_l0_classes=model_num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    # Select trainer class
    if arm_config['trainer_class'] == 'V5Trainer':
        trainer = V5Trainer(
            model=model,
            train_dataset=train_data_obj,
            val_dataset=val_data,
            device=device,
            stage1_epochs=STAGE1_EPOCHS,
            stage2_epochs=arm_config['stage2_epochs'],
            unfreeze_layers=UNFREEZE_LAYERS,
        )
    else:
        trainer = MRLTrainerV5(
            model=model,
            train_dataset=train_data_obj,
            val_dataset=val_data,
            device=device,
            stage1_epochs=STAGE1_EPOCHS,
            stage2_epochs=arm_config['stage2_epochs'],
            unfreeze_layers=UNFREEZE_LAYERS,
        )

    history = trainer.train(batch_size=BATCH_SIZE, patience=5)

    # Evaluate steerability on test set
    steerability = evaluate_prefix_steerability(model, test_data.samples)

    print(f"  Results:")
    for j in [1, 2, 3, 4]:
        jr = steerability['prefix_results'][f'j{j}']
        print(f"    j={j}: L0={jr['l0']:.4f}, L1={jr['l1']:.4f}")
    print(f"  Steerability: {steerability['steerability_score']:+.4f}")

    # Count trainable params for compute equalization check
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'arm': arm_name,
        'dataset': dataset_name,
        'seed': seed,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'stage1_epochs': STAGE1_EPOCHS,
        'stage2_epochs': arm_config['stage2_epochs'],
        **steerability,
    }


def convert(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    return obj


if __name__ == "__main__":
    out_path = RESULTS_DIR / "backbone_finetune_control.json"

    # Load existing results for incremental saving
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    all_results = existing.get('results', {})

    print("=" * 70)
    print("  BACKBONE FINE-TUNING CONTROL (Existential Test)")
    print("  Does flat fine-tuning produce emergent scale separation?")
    print(f"  Arms: {list(ARMS.keys())}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    total_runs = len(ARMS) * len(DATASETS) * len(SEEDS)
    completed = 0

    for ds in DATASETS:
        if ds not in all_results:
            all_results[ds] = {}

        for arm_name, arm_config in ARMS.items():
            if arm_name not in all_results[ds]:
                all_results[ds][arm_name] = []

            done_seeds = {r['seed'] for r in all_results[ds][arm_name]}

            for seed in SEEDS:
                completed += 1
                if seed in done_seeds:
                    cached = [r for r in all_results[ds][arm_name] if r['seed'] == seed][0]
                    print(f"  [{completed}/{total_runs}] {ds}/{arm_name}/s{seed} -> CACHED "
                          f"(S={cached['steerability_score']:+.4f})")
                    continue

                print(f"\n  [{completed}/{total_runs}] {ds}/{arm_name}/s{seed}...")
                try:
                    result = run_single_arm(arm_name, arm_config, ds, seed)
                    all_results[ds][arm_name].append(result)

                    # Incremental save
                    output = {
                        'experiment': 'backbone_finetune_control',
                        'model': MODEL_KEY,
                        'arms': {k: v['description'] for k, v in ARMS.items()},
                        'datasets': DATASETS,
                        'seeds': SEEDS,
                        'config': {
                            'stage1_epochs': STAGE1_EPOCHS,
                            'stage2_epochs': STAGE2_EPOCHS,
                            'unfreeze_layers': UNFREEZE_LAYERS,
                            'batch_size': BATCH_SIZE,
                        },
                        'timestamp': datetime.now().isoformat(),
                        'results': convert(all_results),
                    }
                    with open(out_path, 'w') as f:
                        json.dump(output, f, indent=2)

                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # ===== ANALYSIS =====
    print("\n" + "=" * 70)
    print("BACKBONE FINE-TUNING CONTROL ANALYSIS")
    print("=" * 70)

    from scipy import stats as sp_stats

    for ds in DATASETS:
        if ds not in all_results:
            continue

        print(f"\n  Dataset: {ds}")
        print(f"  {'Arm':<18} {'Steerability':>15} {'StdDev':>10} {'N':>5}")
        print("  " + "-" * 50)

        arm_steers = {}
        for arm_name in ARMS:
            if arm_name not in all_results[ds] or not all_results[ds][arm_name]:
                continue
            steers = [r['steerability_score'] for r in all_results[ds][arm_name]]
            mean_s = np.mean(steers)
            std_s = np.std(steers)
            arm_steers[arm_name] = steers
            print(f"  {arm_name:<18} {mean_s:>+11.4f}    {std_s:>8.4f}  {len(steers):>5}")

        # Key comparisons
        if 'v5_frozen' in arm_steers and 'flat_finetune' in arm_steers:
            v5f = arm_steers['v5_frozen']
            flatft = arm_steers['flat_finetune']
            t, p = sp_stats.ttest_ind(v5f, flatft)
            d = (np.mean(v5f) - np.mean(flatft)) / np.sqrt(
                (np.std(v5f)**2 + np.std(flatft)**2) / 2
            )
            gap = np.mean(v5f) - np.mean(flatft)
            print(f"\n  KEY TEST: V5-frozen vs flat-finetune")
            print(f"    Gap: {gap:+.4f}, t={t:.2f}, p={p:.4f}, d={d:.2f}")
            if gap > 0.03 and p < 0.05:
                print(f"    *** PASS: V5 alignment beats flat fine-tuning ***")
            elif gap > 0:
                print(f"    ** Positive but needs more power **")
            else:
                print(f"    !!! DEVASTATING: flat-finetune matches/exceeds V5 !!!")

        if 'v5_finetune' in arm_steers and 'v5_frozen' in arm_steers:
            v5ft = arm_steers['v5_finetune']
            v5f = arm_steers['v5_frozen']
            t, p = sp_stats.ttest_ind(v5ft, v5f)
            gap = np.mean(v5ft) - np.mean(v5f)
            print(f"\n  BONUS: V5-finetune vs V5-frozen")
            print(f"    Gap: {gap:+.4f}, t={t:.2f}, p={p:.4f}")
            if gap > 0.02:
                print(f"    -> Fine-tuning + alignment is even better (expected)")
            else:
                print(f"    -> Fine-tuning doesn't help much (head-only sufficient)")

    # Pooled analysis across datasets
    print(f"\n  POOLED ANALYSIS (across all datasets):")
    pooled = {}
    for arm_name in ARMS:
        all_steers = []
        for ds in DATASETS:
            if ds in all_results and arm_name in all_results[ds]:
                all_steers.extend([r['steerability_score'] for r in all_results[ds][arm_name]])
        if all_steers:
            pooled[arm_name] = all_steers
            print(f"    {arm_name:<18}: mean={np.mean(all_steers):+.4f} "
                  f"+/- {np.std(all_steers):.4f} (n={len(all_steers)})")

    if 'v5_frozen' in pooled and 'flat_finetune' in pooled:
        t, p = sp_stats.ttest_ind(pooled['v5_frozen'], pooled['flat_finetune'])
        gap = np.mean(pooled['v5_frozen']) - np.mean(pooled['flat_finetune'])
        print(f"\n    POOLED KEY TEST: V5-frozen vs flat-finetune")
        print(f"      Gap: {gap:+.4f}, t={t:.2f}, p={p:.4f}")

    # Final save
    output = {
        'experiment': 'backbone_finetune_control',
        'model': MODEL_KEY,
        'arms': {k: v['description'] for k, v in ARMS.items()},
        'datasets': DATASETS,
        'seeds': SEEDS,
        'config': {
            'stage1_epochs': STAGE1_EPOCHS,
            'stage2_epochs': STAGE2_EPOCHS,
            'unfreeze_layers': UNFREEZE_LAYERS,
            'batch_size': BATCH_SIZE,
        },
        'timestamp': datetime.now().isoformat(),
        'results': convert(all_results),
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
