"""
Prospective Validation: Test pre-registered predictions on unseen datasets.

Pre-registered predictions were saved to results/prospective_predictions_round2.json
BEFORE running any experiments on these datasets. This script:
1. Loads the pre-registered predictions
2. Runs V5 + MRL on each unseen dataset (3 seeds)
3. Computes actual steerability
4. Compares predicted vs actual
5. Reports prospective prediction accuracy

OPTIMIZATIONS (v2):
- Dataset caching (load once per dataset, reuse across seeds)
- Vectorized k-NN (matrix multiply instead of per-sample loop)
- Partial result saving after each dataset
- Explicit stdout flush for Windows buffering

Usage: python -u src/run_prospective_validation.py
"""

import json
import sys
import os
import gc
import numpy as np
import torch
from pathlib import Path
from scipy import stats as sp_stats
from datetime import datetime

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, str(Path(__file__).parent))


def flush():
    """Force flush stdout/stderr for Windows."""
    sys.stdout.flush()
    sys.stderr.flush()


def compute_steerability(prefix_accuracy: dict) -> float:
    """S = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)"""
    return (prefix_accuracy.get('j1_l0', 0) - prefix_accuracy.get('j4_l0', 0)) + \
           (prefix_accuracy.get('j4_l1', 0) - prefix_accuracy.get('j1_l1', 0))


def fast_knn_accuracy(emb, labels, k=5):
    """Vectorized k-NN: matrix multiply + argpartition instead of per-sample loop."""
    # emb: [N, D], already L2-normalized
    sim_matrix = emb @ emb.T  # [N, N] cosine similarity
    np.fill_diagonal(sim_matrix, -np.inf)  # exclude self

    # Use argpartition for O(N) per query instead of O(N log N)
    n = len(emb)
    if k >= n - 1:
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    else:
        top_k_indices = np.argpartition(-sim_matrix, k, axis=1)[:, :k]

    # Vote
    neighbor_labels = labels[top_k_indices]  # [N, k]
    correct = 0
    for i in range(n):
        unique, counts = np.unique(neighbor_labels[i], return_counts=True)
        pred = unique[np.argmax(counts)]
        if pred == labels[i]:
            correct += 1
    return correct / n


def fast_evaluate_prefix_accuracy(model, test_samples, device="cuda"):
    """Faster prefix accuracy evaluation using vectorized k-NN."""
    model.eval()
    samples = test_samples[:min(500, len(test_samples))]
    texts = [s.text for s in samples]
    l0_labels = np.array([s.level0_label for s in samples])
    l1_labels = np.array([s.level1_label for s in samples])

    results = {}
    for j in [1, 2, 3, 4]:
        prefix_len = j if j < 4 else None
        with torch.no_grad():
            emb = model.encode(texts, batch_size=64, prefix_len=prefix_len).numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        results[f'j{j}_l0'] = fast_knn_accuracy(emb, l0_labels)
        results[f'j{j}_l1'] = fast_knn_accuracy(emb, l1_labels)

    return results


def run_single_experiment(train_data, test_data, num_l0, num_l1, model_key="bge-small", seed=42, device="cuda"):
    """Run V5 + MRL on pre-loaded data. Returns steerability results."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
    from mrl_v5_baseline import MRLTrainerV5
    from multi_model_pipeline import MODELS

    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

    class TempDataset:
        def __init__(self, samples, level0_names, level1_names):
            self.samples = samples
            self.level0_names = level0_names
            self.level1_names = level1_names

    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_ds = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)

    config = MODELS[model_key]

    # ---- V5 ----
    print("    [V5] Training...", end=" "); flush()
    v5_model = FractalModelV5(
        config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
        num_scales=4, scale_dim=64, device=device,
    ).to(device)
    v5_trainer = V5Trainer(
        model=v5_model, train_dataset=train_ds, val_dataset=val_data,
        device=device, stage1_epochs=5, stage2_epochs=0,
    )
    v5_trainer.train(batch_size=32, patience=5)

    print("Evaluating...", end=" "); flush()
    v5_prefix = fast_evaluate_prefix_accuracy(v5_model, test_data.samples, device)
    v5_steer = compute_steerability(v5_prefix)
    print(f"S={v5_steer:+.4f}"); flush()

    del v5_model, v5_trainer
    torch.cuda.empty_cache(); gc.collect()

    # ---- MRL ----
    print("    [MRL] Training...", end=" "); flush()
    mrl_model = FractalModelV5(
        config=config, num_l0_classes=num_l1, num_l1_classes=num_l1,
        num_scales=4, scale_dim=64, device=device,
    ).to(device)
    mrl_trainer = MRLTrainerV5(
        model=mrl_model, train_dataset=train_ds, val_dataset=val_data,
        device=device, stage1_epochs=5, stage2_epochs=0,
    )
    mrl_trainer.train(batch_size=32, patience=5)

    print("Evaluating...", end=" "); flush()
    mrl_prefix = fast_evaluate_prefix_accuracy(mrl_model, test_data.samples, device)
    mrl_steer = compute_steerability(mrl_prefix)
    print(f"S={mrl_steer:+.4f}"); flush()

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

    from hierarchical_datasets import load_hierarchical_dataset

    # Load pre-registered predictions
    pred_file = RESULTS_DIR / "prospective_predictions_round2.json"
    predictions = json.load(open(pred_file))
    print("=" * 70)
    print("  PROSPECTIVE VALIDATION v2 (optimized)")
    print("=" * 70)
    print(f"  Predictions timestamp: {predictions['timestamp']}")
    print(f"  Calibration: R^2={predictions['calibration']['r_squared']:.4f}")
    flush()

    datasets = list(predictions['predictions'].keys())
    all_results = {}

    # Check for partial results (resume support)
    partial_file = RESULTS_DIR / "prospective_validation_round2_partial.json"
    if partial_file.exists():
        partial = json.load(open(partial_file))
        all_results = partial.get('results', {})
        print(f"  Resuming: {len(all_results)} datasets already done")
        flush()

    for ds in datasets:
        if ds in all_results:
            print(f"\n  Skipping {ds} (already done)")
            flush()
            continue

        pred = predictions['predictions'][ds]
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds}")
        print(f"  Prediction: S = {pred['predicted_S']:+.4f} "
              f"[{pred['prediction_interval_95'][0]:+.4f}, "
              f"{pred['prediction_interval_95'][1]:+.4f}]")
        print(f"{'='*70}")
        flush()

        # Load dataset ONCE, reuse across seeds
        print(f"  Loading {ds}...", end=" "); flush()
        train_data = load_hierarchical_dataset(ds, split="train", max_samples=10000)
        test_data = load_hierarchical_dataset(ds, split="test", max_samples=2000)
        num_l0 = len(train_data.level0_names)
        num_l1 = len(train_data.level1_names)
        print(f"L0={num_l0}, L1={num_l1}"); flush()

        seed_results = []
        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---"); flush()
            result = run_single_experiment(
                train_data, test_data, num_l0, num_l1, seed=seed
            )
            seed_results.append(result)
            print(f"    V5 S = {result['v5_steer']:+.4f}, "
                  f"MRL S = {result['mrl_steer']:+.4f}, "
                  f"Gap = {result['gap']:+.4f}")
            flush()

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
        flush()

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

        # Save partial results after each dataset
        partial_output = {
            'timestamp': datetime.now().isoformat(),
            'type': 'prospective_validation_round2_partial',
            'results': all_results,
        }
        with open(partial_file, 'w') as f:
            json.dump(partial_output, f, indent=2,
                     default=lambda x: float(x) if hasattr(x, 'item') else x)
        print(f"  Partial results saved ({len(all_results)}/{len(datasets)} datasets)")
        flush()

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
    flush()

    # Save final
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
    flush()

    # Clean up partial file
    if partial_file.exists():
        partial_file.unlink()


if __name__ == "__main__":
    main()
