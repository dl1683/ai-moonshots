"""Unseen-Leaf Generalization: Does V5's coarse prefix generalize to novel fine classes?

KEY EXPERIMENT: Train V5 and MRL with some L1 classes held out. At test time:
- For SEEN L1 classes: both V5 and MRL should work normally
- For UNSEEN L1 classes: V5's 64d prefix should STILL correctly classify L0
  (because the prefix encodes domain structure, not specific intent identity)

If V5's prefix L0 accuracy on unseen intents is high, it proves:
1. The prefix captures genuine coarse SEMANTIC structure
2. Not just a lookup table of training classes
3. The prefix representation transfers to novel fine classes

This is a generalization test that MRL cannot match: MRL's prefixes encode
the same information at lower fidelity, so they can't selectively preserve
coarse structure for unseen classes.

Run: python src/unseen_leaf_generalization.py [dataset] [holdout_frac]
"""

import sys
import os
import json
import numpy as np
import torch
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_holdout_split(dataset, holdout_frac=0.3, seed=42):
    """Hold out a fraction of L1 classes from each L0 group.

    Returns:
        train_samples: samples from SEEN L1 classes only
        test_seen: test samples from SEEN L1 classes
        test_unseen: test samples from UNSEEN L1 classes
        seen_l1: set of seen L1 class indices
        unseen_l1: set of unseen L1 class indices
    """
    rng = np.random.RandomState(seed)

    # Group L1 classes by L0
    l0_to_l1 = defaultdict(set)
    for s in dataset.samples:
        l0_to_l1[s.level0_label].add(s.level1_label)

    # For each L0 group, hold out some L1 classes
    seen_l1 = set()
    unseen_l1 = set()

    for l0, l1_classes in l0_to_l1.items():
        l1_list = sorted(l1_classes)
        n_holdout = max(1, int(len(l1_list) * holdout_frac))
        rng.shuffle(l1_list)
        unseen = set(l1_list[:n_holdout])
        seen = set(l1_list[n_holdout:])

        # Ensure at least 1 seen class per L0
        if len(seen) == 0 and len(unseen) > 1:
            seen.add(unseen.pop())

        seen_l1.update(seen)
        unseen_l1.update(unseen)

    # Split samples
    train_samples = [s for s in dataset.samples if s.level1_label in seen_l1]
    # For test, we need both seen and unseen
    # We'll use train_samples for both training and seen-test via split
    test_seen = []
    test_unseen = []

    return train_samples, seen_l1, unseen_l1


def run_experiment(dataset_name="clinc", holdout_frac=0.3, model_key="bge-small",
                   seeds=None, device="cuda"):
    """Run the unseen-leaf generalization experiment."""
    from fractal_v5 import (FractalModelV5, V5Trainer, MODELS as V5_MODELS,
                             split_train_val)
    from mrl_v5_baseline import MRLTrainerV5
    from hierarchical_datasets import load_hierarchical_dataset
    import random

    if seeds is None:
        seeds = [42, 123, 456]

    config = V5_MODELS[model_key]

    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}: {dataset_name}, holdout={holdout_frac:.0%}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load full data
        full_train = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
        full_test = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

        # Create holdout split
        train_samples, seen_l1, unseen_l1 = create_holdout_split(full_train, holdout_frac, seed)

        # Create datasets
        num_l0 = len(full_train.level0_names)
        num_l1 = len(full_train.level1_names)

        print(f"  L0 classes: {num_l0}, Total L1: {num_l1}")
        print(f"  Seen L1: {len(seen_l1)}, Unseen L1: {len(unseen_l1)}")
        print(f"  Training samples (seen only): {len(train_samples)}")

        # Split test into seen/unseen
        test_seen = [s for s in full_test.samples if s.level1_label in seen_l1]
        test_unseen = [s for s in full_test.samples if s.level1_label in unseen_l1]
        print(f"  Test seen: {len(test_seen)}, Test unseen: {len(test_unseen)}")

        if len(test_unseen) < 20:
            print("  Too few unseen test samples, skipping")
            continue

        # Create train/val split from seen-only data
        train_split, val_split = split_train_val(train_samples, val_ratio=0.15)

        class TempDataset:
            def __init__(self, samples, level0_names, level1_names):
                self.samples = samples
                self.level0_names = level0_names
                self.level1_names = level1_names

        train_data = TempDataset(train_split, full_train.level0_names, full_train.level1_names)
        val_data = TempDataset(val_split, full_train.level0_names, full_train.level1_names)

        seed_result = {'seed': seed, 'seen_l1': len(seen_l1), 'unseen_l1': len(unseen_l1)}

        for method in ["v5", "mrl"]:
            torch.cuda.empty_cache()
            gc.collect()

            print(f"\n  Training {method.upper()}...")

            model = FractalModelV5(
                config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
                num_scales=4, scale_dim=64, device=device,
            ).to(device)

            if method == "v5":
                trainer = V5Trainer(
                    model=model, train_dataset=train_data, val_dataset=val_data,
                    device=device, stage1_epochs=5, stage2_epochs=0, unfreeze_layers=4,
                )
            else:
                trainer = MRLTrainerV5(
                    model=model, train_dataset=train_data, val_dataset=val_data,
                    device=device, stage1_epochs=5, stage2_epochs=0, unfreeze_layers=4,
                )

            trainer.train(batch_size=16, patience=5)
            model.eval()

            # Evaluate on seen and unseen test sets
            for split_name, test_samples in [("seen", test_seen), ("unseen", test_unseen)]:
                if not test_samples:
                    continue

                texts = [s.text for s in test_samples]
                true_l0 = np.array([s.level0_label for s in test_samples])
                true_l1 = np.array([s.level1_label for s in test_samples])

                # Get embeddings at each prefix length
                for j in [1, 4]:
                    prefix_len = j if j < 4 else None
                    embs = model.encode(texts, batch_size=64,
                                       prefix_len=prefix_len).cpu().numpy()

                    # Normalize
                    norms = np.linalg.norm(embs, axis=1, keepdims=True)
                    embs = embs / np.maximum(norms, 1e-8)

                    # Get reference embeddings from training set
                    ref_texts = [s.text for s in train_data.samples]
                    ref_l0 = np.array([s.level0_label for s in train_data.samples])
                    ref_l1 = np.array([s.level1_label for s in train_data.samples])

                    ref_embs = model.encode(ref_texts, batch_size=64,
                                           prefix_len=prefix_len).cpu().numpy()
                    ref_norms = np.linalg.norm(ref_embs, axis=1, keepdims=True)
                    ref_embs = ref_embs / np.maximum(ref_norms, 1e-8)

                    # k-NN classification (k=5)
                    k = 5
                    sims = embs @ ref_embs.T
                    topk_idx = np.argsort(sims, axis=1)[:, -k:]

                    # L0 accuracy via majority vote
                    l0_preds = []
                    for i in range(len(texts)):
                        nn_l0 = ref_l0[topk_idx[i]]
                        counts = np.bincount(nn_l0, minlength=num_l0)
                        l0_preds.append(np.argmax(counts))
                    l0_acc = np.mean(np.array(l0_preds) == true_l0)

                    # L1 accuracy via majority vote
                    l1_preds = []
                    for i in range(len(texts)):
                        nn_l1 = ref_l1[topk_idx[i]]
                        counts = np.bincount(nn_l1, minlength=num_l1)
                        l1_preds.append(np.argmax(counts))
                    l1_acc = np.mean(np.array(l1_preds) == true_l1)

                    dim_str = "64d" if j == 1 else "256d"
                    key_prefix = f"{method}_{split_name}"
                    seed_result[f"{key_prefix}_l0_{dim_str}"] = float(l0_acc)
                    seed_result[f"{key_prefix}_l1_{dim_str}"] = float(l1_acc)

                    print(f"    {method.upper()} {split_name:>6} {dim_str}: "
                          f"L0={l0_acc:.3f}, L1={l1_acc:.3f}")

            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()

        # Compute generalization metrics
        v5_seen_l0_64 = seed_result.get('v5_seen_l0_64d', 0)
        v5_unseen_l0_64 = seed_result.get('v5_unseen_l0_64d', 0)
        mrl_seen_l0_64 = seed_result.get('mrl_seen_l0_64d', 0)
        mrl_unseen_l0_64 = seed_result.get('mrl_unseen_l0_64d', 0)

        seed_result['v5_l0_generalization_gap'] = v5_seen_l0_64 - v5_unseen_l0_64
        seed_result['mrl_l0_generalization_gap'] = mrl_seen_l0_64 - mrl_unseen_l0_64

        print(f"\n  GENERALIZATION SUMMARY (seed {seed}):")
        print(f"    V5  prefix L0: seen={v5_seen_l0_64:.3f}, unseen={v5_unseen_l0_64:.3f}, "
              f"gap={seed_result['v5_l0_generalization_gap']:+.3f}")
        print(f"    MRL prefix L0: seen={mrl_seen_l0_64:.3f}, unseen={mrl_unseen_l0_64:.3f}, "
              f"gap={seed_result['mrl_l0_generalization_gap']:+.3f}")

        all_results.append(seed_result)

    # Summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    for key_metric in ['v5_unseen_l0_64d', 'mrl_unseen_l0_64d',
                       'v5_seen_l0_64d', 'mrl_seen_l0_64d',
                       'v5_l0_generalization_gap', 'mrl_l0_generalization_gap']:
        vals = [r[key_metric] for r in all_results if key_metric in r]
        if vals:
            print(f"  {key_metric}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    # Key comparison: V5 vs MRL on unseen L0 accuracy at 64d
    v5_unseen = [r['v5_unseen_l0_64d'] for r in all_results if 'v5_unseen_l0_64d' in r]
    mrl_unseen = [r['mrl_unseen_l0_64d'] for r in all_results if 'mrl_unseen_l0_64d' in r]

    if v5_unseen and mrl_unseen:
        from scipy import stats as sp_stats
        t, p = sp_stats.ttest_ind(v5_unseen, mrl_unseen)
        d = (np.mean(v5_unseen) - np.mean(mrl_unseen)) / np.sqrt(
            (np.std(v5_unseen)**2 + np.std(mrl_unseen)**2) / 2
        ) if np.std(v5_unseen) + np.std(mrl_unseen) > 0 else 0
        print(f"\n  V5 vs MRL (unseen L0@64d): t={t:.2f}, p={p:.4f}, d={d:.2f}")
        print(f"  V5 unseen: {np.mean(v5_unseen):.3f}, MRL unseen: {np.mean(mrl_unseen):.3f}")

    # Save
    out = {
        'experiment': 'unseen_leaf_generalization',
        'dataset': dataset_name,
        'holdout_frac': holdout_frac,
        'model': model_key,
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
    }
    out_path = RESULTS_DIR / f"unseen_leaf_{dataset_name}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return all_results


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "clinc"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    run_experiment(ds, holdout_frac=frac)
