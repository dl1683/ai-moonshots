"""
Prospective Validation: Test pre-registered predictions on unseen datasets.

Pre-registered predictions were saved to results/prospective_predictions_round2.json
BEFORE running any experiments on these datasets. This script:
1. Loads the pre-registered predictions
2. Runs V5 + MRL on each unseen dataset (3 seeds)
3. Computes actual steerability
4. Compares predicted vs actual
5. Reports prospective prediction accuracy

Usage: python -u src/run_prospective_validation.py
"""

import json
import sys
import gc
import numpy as np
import torch
from pathlib import Path
from scipy import stats as sp_stats
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


def compute_steerability(prefix_accuracy: dict) -> float:
    """S = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)"""
    return (prefix_accuracy.get('j1_l0', 0) - prefix_accuracy.get('j4_l0', 0)) + \
           (prefix_accuracy.get('j4_l1', 0) - prefix_accuracy.get('j1_l1', 0))


def run_single_experiment(dataset_name, model_key="bge-small", seed=42, device="cuda"):
    """Run V5 + MRL on a single dataset/seed. Returns steerability results."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from hierarchical_datasets import load_hierarchical_dataset
    from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
    from mrl_v5_baseline import MRLTrainerV5
    from multi_model_pipeline import MODELS

    # Load data
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

    class TempDataset:
        def __init__(self, samples, level0_names, level1_names):
            self.samples = samples
            self.level0_names = level0_names
            self.level1_names = level1_names

    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_data.samples = train_samples

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)
    config = MODELS[model_key]

    # ---- V5 ----
    v5_model = FractalModelV5(
        config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
        num_scales=4, scale_dim=64, device=device,
    ).to(device)
    v5_trainer = V5Trainer(
        model=v5_model, train_dataset=train_data, val_dataset=val_data,
        device=device, stage1_epochs=5, stage2_epochs=0,
    )
    v5_trainer.train(batch_size=32, patience=5)

    # Eval V5 prefix accuracy
    test_temp = TempDataset(test_data.samples, test_data.level0_names, test_data.level1_names)
    v5_trainer.val_dataset = test_temp
    v5_prefix = v5_trainer.evaluate_prefix_accuracy()
    v5_steer = compute_steerability(v5_prefix)

    del v5_model, v5_trainer
    torch.cuda.empty_cache(); gc.collect()

    # ---- MRL ----
    mrl_model = FractalModelV5(
        config=config, num_l0_classes=num_l1, num_l1_classes=num_l1,
        num_scales=4, scale_dim=64, device=device,
    ).to(device)
    mrl_trainer = MRLTrainerV5(
        model=mrl_model, train_dataset=train_data, val_dataset=val_data,
        device=device, stage1_epochs=5, stage2_epochs=0,
    )
    mrl_trainer.train(batch_size=32, patience=5)

    mrl_trainer.val_dataset = test_temp
    mrl_prefix = mrl_trainer.evaluate_prefix_accuracy()
    mrl_steer = compute_steerability(mrl_prefix)

    del mrl_model, mrl_trainer
    torch.cuda.empty_cache(); gc.collect()

    return {
        'v5_steer': float(v5_steer),
        'mrl_steer': float(mrl_steer),
        'gap': float(v5_steer - mrl_steer),
        'v5_prefix': {k: float(v) for k, v in v5_prefix.items()},
        'mrl_prefix': {k: float(v) for k, v in mrl_prefix.items()},
    }


def main():
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    SEEDS = [42, 123, 456]

    # Load pre-registered predictions
    pred_file = RESULTS_DIR / "prospective_predictions_round2.json"
    predictions = json.load(open(pred_file))
    print("=" * 70)
    print("  PROSPECTIVE VALIDATION: Running experiments on unseen datasets")
    print("=" * 70)
    print(f"  Predictions timestamp: {predictions['timestamp']}")
    print(f"  Calibration: R^2={predictions['calibration']['r_squared']:.4f}")

    datasets = list(predictions['predictions'].keys())
    all_results = {}

    for ds in datasets:
        pred = predictions['predictions'][ds]
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds}")
        print(f"  Prediction: S = {pred['predicted_S']:+.4f} "
              f"[{pred['prediction_interval_95'][0]:+.4f}, "
              f"{pred['prediction_interval_95'][1]:+.4f}]")
        print(f"{'='*70}")

        seed_results = []
        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            result = run_single_experiment(ds, seed=seed)
            seed_results.append(result)
            print(f"    V5 S = {result['v5_steer']:+.4f}, "
                  f"MRL S = {result['mrl_steer']:+.4f}, "
                  f"Gap = {result['gap']:+.4f}")

        # Aggregate
        v5_steers = [r['v5_steer'] for r in seed_results]
        mrl_steers = [r['mrl_steer'] for r in seed_results]
        gaps = [r['gap'] for r in seed_results]

        mean_s = np.mean(v5_steers)
        std_s = np.std(v5_steers, ddof=1)
        mean_gap = np.mean(gaps)

        residual = mean_s - pred['predicted_S']
        in_pi = (pred['prediction_interval_95'][0] <= mean_s <= pred['prediction_interval_95'][1])
        sign_correct = (mean_s > 0) == (pred['predicted_S'] > 0)

        print(f"\n  RESULT for {ds}:")
        print(f"    Predicted S: {pred['predicted_S']:+.4f}")
        print(f"    Actual V5 S: {mean_s:+.4f} +/- {std_s:.4f}")
        print(f"    Residual:    {residual:+.4f}")
        print(f"    In 95% PI:   {'YES' if in_pi else 'NO'}")
        print(f"    Sign correct: {'YES' if sign_correct else 'NO'}")

        all_results[ds] = {
            'predicted_S': pred['predicted_S'],
            'prediction_interval': pred['prediction_interval_95'],
            'actual_S_mean': float(mean_s),
            'actual_S_std': float(std_s),
            'actual_S_per_seed': v5_steers,
            'mrl_S_mean': float(np.mean(mrl_steers)),
            'gap_mean': float(mean_gap),
            'residual': float(residual),
            'in_prediction_interval': bool(in_pi),
            'sign_correct': bool(sign_correct),
            'seed_results': seed_results,
        }

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("  PROSPECTIVE VALIDATION SUMMARY")
    print(f"{'='*70}")

    n_in_pi = sum(1 for r in all_results.values() if r['in_prediction_interval'])
    n_sign = sum(1 for r in all_results.values() if r['sign_correct'])
    n_total = len(all_results)
    residuals = [r['residual'] for r in all_results.values()]
    rmse = np.sqrt(np.mean(np.array(residuals)**2))
    mae = np.mean(np.abs(residuals))

    # Correlation across all datasets (calibration + test)
    predicted = [r['predicted_S'] for r in all_results.values()]
    actual = [r['actual_S_mean'] for r in all_results.values()]
    if len(predicted) > 2:
        rho, p_rho = sp_stats.spearmanr(predicted, actual)
        r, p_r = sp_stats.pearsonr(predicted, actual)
    else:
        rho, p_rho, r, p_r = 0, 1, 0, 1

    print(f"\n  Datasets tested: {n_total}")
    print(f"  In 95% PI:      {n_in_pi}/{n_total} ({n_in_pi/n_total:.0%})")
    print(f"  Sign correct:   {n_sign}/{n_total} ({n_sign/n_total:.0%})")
    print(f"  RMSE:           {rmse:.4f}")
    print(f"  MAE:            {mae:.4f}")
    print(f"  Spearman rho:   {rho:.3f} (p={p_rho:.4f})")
    print(f"  Pearson r:      {r:.3f} (p={p_r:.4f})")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'type': 'prospective_validation_round2',
        'calibration': predictions['calibration'],
        'results': all_results,
        'summary': {
            'n_datasets': n_total,
            'n_in_pi': n_in_pi,
            'n_sign_correct': n_sign,
            'rmse': float(rmse),
            'mae': float(mae),
            'spearman_rho': float(rho),
            'spearman_p': float(p_rho),
            'pearson_r': float(r),
            'pearson_p': float(p_r),
        },
    }

    out_path = RESULTS_DIR / "prospective_validation_round2.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
