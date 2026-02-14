"""
Run a single V5 or MRL experiment and save results.

Usage:
  python -u src/run_single_experiment.py --dataset clinc --model bge-base --seed 42 --method v5
  python -u src/run_single_experiment.py --dataset clinc --model bge-base --seed 42 --method mrl
"""

import sys
import gc
import json
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


def compute_steerability(prefix_accuracy: dict) -> float:
    return (prefix_accuracy.get('j1_l0', 0) - prefix_accuracy.get('j4_l0', 0)) + \
           (prefix_accuracy.get('j4_l1', 0) - prefix_accuracy.get('j1_l1', 0))


def run_experiment(dataset_name, model_key, seed, method, device="cuda"):
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

    print(f"  Dataset: {dataset_name}, Model: {model_key}, Seed: {seed}, Method: {method}")
    print(f"  L0: {num_l0}, L1: {num_l1}")

    if method == "v5":
        model = FractalModelV5(
            config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
            num_scales=4, scale_dim=64, device=device,
        ).to(device)
        trainer = V5Trainer(
            model=model, train_dataset=train_data, val_dataset=val_data,
            device=device, stage1_epochs=5, stage2_epochs=0,
        )
    else:  # mrl
        model = FractalModelV5(
            config=config, num_l0_classes=num_l1, num_l1_classes=num_l1,
            num_scales=4, scale_dim=64, device=device,
        ).to(device)
        trainer = MRLTrainerV5(
            model=model, train_dataset=train_data, val_dataset=val_data,
            device=device, stage1_epochs=5, stage2_epochs=0,
        )

    trainer.train(batch_size=32, patience=5)

    test_temp = TempDataset(test_data.samples, test_data.level0_names, test_data.level1_names)
    trainer.val_dataset = test_temp
    prefix_acc = trainer.evaluate_prefix_accuracy()
    steer = compute_steerability(prefix_acc)

    print(f"  S = {steer:+.4f}")
    print(f"  Prefix acc: {prefix_acc}")

    result = {
        'dataset': dataset_name,
        'model': model_key,
        'seed': seed,
        'method': method,
        'steerability': float(steer),
        'prefix_accuracy': {k: float(v) for k, v in prefix_acc.items()},
        'timestamp': datetime.now().isoformat(),
    }

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="bge-small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["v5", "mrl"], required=True)
    args = parser.parse_args()

    result = run_experiment(args.dataset, args.model, args.seed, args.method)

    # Save result
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / f"{args.method}_{args.model}_{args.dataset}_seed{args.seed}.json"

    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"\n  Saved to {out_file}")


if __name__ == "__main__":
    main()
