"""
CRITICAL EXPERIMENT: MRL baseline on Qwen3-0.6B
================================================
This determines whether V5's +5.36%/+6.47% gains over flat on Yahoo
are due to hierarchy-alignment (our claim) or just multi-scale training (MRL does that too).

If MRL ≈ V5: hierarchy-alignment doesn't matter, paper is weakened
If MRL << V5: hierarchy-alignment IS the key innovation, paper is strong

Uses identical setup to run_simple_multi_seed in v5_statistical_validation.py:
- Seeds: 0, 1, 2, 3, 4
- Train/val split: seed=42 (fixed)
- Stage1 epochs: 5
- Batch size: 24 (Qwen3-0.6B needs smaller batches)
"""

import sys
import os
import json
import gc
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS, load_model
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from mrl_v5_baseline import MRLTrainerV5


def knn_accuracy(embeddings, labels, k=5):
    """KNN accuracy (same as in v5_statistical_validation.py)."""
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    scores = cross_val_score(knn, embeddings, labels, cv=5, scoring='accuracy')
    return float(scores.mean())


def evaluate_knn_test(model, test_samples, max_samples=2000):
    """Evaluate on test set using KNN."""
    samples = test_samples[:min(max_samples, len(test_samples))]
    texts = [s.text for s in samples]
    l0_labels = np.array([s.level0_label for s in samples])
    l1_labels = np.array([s.level1_label for s in samples])

    embeddings = model.encode(texts, batch_size=24).numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Use cosine similarity - find nearest neighbors in test set itself
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -1)  # Exclude self

    k = 5
    l0_correct = 0
    l1_correct = 0

    for i in range(len(samples)):
        top_k = np.argsort(-sims[i])[:k]

        # Majority vote for L0
        l0_votes = l0_labels[top_k]
        l0_pred = np.bincount(l0_votes).argmax()
        if l0_pred == l0_labels[i]:
            l0_correct += 1

        # Majority vote for L1
        l1_votes = l1_labels[top_k]
        l1_pred = np.bincount(l1_votes).argmax()
        if l1_pred == l1_labels[i]:
            l1_correct += 1

    return l0_correct / len(samples), l1_correct / len(samples)


def run_mrl_multiseed(
    num_seeds=5,
    model_key="qwen3-0.6b",
    dataset_name="yahoo",
    stage1_epochs=5,
    batch_size=24,
    device="cuda",
):
    """Run MRL baseline with multiple seeds - mirrors run_simple_multi_seed exactly."""
    print("=" * 70)
    print(f"CRITICAL: MRL MULTI-SEED ON {model_key.upper()}")
    print("=" * 70)
    print(f"Seeds: {num_seeds}")
    print(f"This is the make-or-break experiment.")
    print(f"V5 results: L0=71.61±0.74%, L1=64.17±0.85%")
    print(f"Flat baseline: L0=66.25%, L1=57.70%")
    print()

    # Load data (same as V5 multi-seed)
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)
    config = MODELS[model_key]

    # Get baseline (same as V5 validation - unfinetuned model)
    print("Getting unfinetuned baseline...")
    base_model = load_model(model_key, use_fractal=False, device=device)
    base_l0, base_l1 = evaluate_knn_test(base_model, test_data.samples)
    print(f"  Baseline: L0={base_l0:.4f}, L1={base_l1:.4f}")
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # Run MRL with multiple seeds
    mrl_l0_results = []
    mrl_l1_results = []
    prefix_results = []

    for seed in range(num_seeds):
        print(f"\n{'='*50}")
        print(f"[Seed {seed+1}/{num_seeds}] (seed={seed})")
        print(f"{'='*50}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Fixed train/val split (same as V5)
        train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15, seed=42)

        class TempDataset:
            def __init__(self, samples, level0_names, level1_names):
                self.samples = samples
                self.level0_names = level0_names
                self.level1_names = level1_names

        val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
        train_data_seed = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)

        # Create MRL model (same arch as V5, but both heads use L1 classes)
        model = FractalModelV5(
            config=config,
            num_l0_classes=num_l1,  # KEY MRL DIFFERENCE: both heads use L1
            num_l1_classes=num_l1,
            num_scales=4,
            scale_dim=64,
            device=device,
        ).to(device)

        trainer = MRLTrainerV5(
            model=model,
            train_dataset=train_data_seed,
            val_dataset=val_data,
            device=device,
            stage1_epochs=stage1_epochs,
        )
        trainer.train(batch_size=batch_size, patience=5)

        # Evaluate full embedding (j=4)
        l0, l1 = evaluate_knn_test(model, test_data.samples)
        mrl_l0_results.append(l0)
        mrl_l1_results.append(l1)
        print(f"  MRL seed {seed}: L0={l0:.4f}, L1={l1:.4f}")

        # Also evaluate prefix lengths
        seed_prefix = {}
        for j in [1, 2, 3, 4]:
            pl = j if j < 4 else None
            texts = [s.text for s in test_data.samples[:2000]]
            l0_labels = np.array([s.level0_label for s in test_data.samples[:2000]])
            l1_labels = np.array([s.level1_label for s in test_data.samples[:2000]])

            emb = model.encode(texts, batch_size=24, prefix_len=pl).numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

            sims = emb @ emb.T
            np.fill_diagonal(sims, -1)

            l0_c = 0
            l1_c = 0
            for i in range(len(texts)):
                top5 = np.argsort(-sims[i])[:5]
                if np.bincount(l0_labels[top5]).argmax() == l0_labels[i]:
                    l0_c += 1
                if np.bincount(l1_labels[top5]).argmax() == l1_labels[i]:
                    l1_c += 1
            seed_prefix[f"j{j}"] = {
                "l0": l0_c / len(texts),
                "l1": l1_c / len(texts),
            }
            print(f"    j={j}: L0={l0_c/len(texts):.4f}, L1={l1_c/len(texts):.4f}")

        prefix_results.append(seed_prefix)

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    mrl_l0 = np.array(mrl_l0_results)
    mrl_l1 = np.array(mrl_l1_results)

    print("\n" + "=" * 70)
    print("CRITICAL RESULTS: MRL vs V5 on Qwen3-0.6B")
    print("=" * 70)
    print(f"{'Method':<15} {'L0 (mean±std)':<20} {'L1 (mean±std)':<20}")
    print("-" * 55)
    print(f"{'Flat':<15} {base_l0:.4f}              {base_l1:.4f}")
    print(f"{'MRL':<15} {mrl_l0.mean():.4f}±{mrl_l0.std():.4f}      {mrl_l1.mean():.4f}±{mrl_l1.std():.4f}")
    print(f"{'V5':<15} 0.7161±0.0074          0.6417±0.0085")
    print()
    print(f"MRL vs Flat:  L0={mrl_l0.mean()-base_l0:+.4f}, L1={mrl_l1.mean()-base_l1:+.4f}")
    print(f"V5  vs Flat:  L0=+0.0536, L1=+0.0647")
    print(f"V5  vs MRL:   L0={0.7161-mrl_l0.mean():+.4f}, L1={0.6417-mrl_l1.mean():+.4f}")
    print()

    if mrl_l0.mean() > 0.70 and mrl_l1.mean() > 0.63:
        print(">>> WARNING: MRL is close to V5. Hierarchy-alignment may not be the key factor.")
    elif mrl_l0.mean() < 0.69 and mrl_l1.mean() < 0.61:
        print(">>> GOOD NEWS: MRL significantly worse than V5. Hierarchy-alignment IS the key innovation!")
    else:
        print(">>> MIXED: MRL partially closes gap. Need deeper analysis.")

    # Save results
    results = {
        "experiment": "mrl_multiseed_critical",
        "model": model_key,
        "dataset": dataset_name,
        "num_seeds": num_seeds,
        "baseline": {"l0": base_l0, "l1": base_l1},
        "mrl_mean": {"l0": float(mrl_l0.mean()), "l1": float(mrl_l1.mean())},
        "mrl_std": {"l0": float(mrl_l0.std()), "l1": float(mrl_l1.std())},
        "v5_mean": {"l0": 0.7161, "l1": 0.6417},
        "v5_std": {"l0": 0.0074, "l1": 0.0085},
        "per_seed": [
            {
                "seed": i,
                "l0": float(mrl_l0_results[i]),
                "l1": float(mrl_l1_results[i]),
                "prefix": prefix_results[i],
            }
            for i in range(num_seeds)
        ],
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"mrl_multiseed_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_mrl_multiseed()
