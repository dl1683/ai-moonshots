"""
Causal Prefix Surgery: The Paradigm-Shifting Experiment
=========================================================

THE CLAIM: In fractal embeddings, prefix blocks carry distinct semantic levels.
The coarse prefix (j=1, 64d) encodes L0 identity.
The fine suffix (dims 65-256) encodes L1|L0 refinement.

THE EXPERIMENT:
1. Train V5 and MRL models on CLINC
2. Encode test samples
3. For each pair of samples from DIFFERENT L0 classes:
   - SWAP only the first 64d block
   - Measure: does the L0 classification change? (coarse transfer)
   - Measure: does the L1 identity within-L0 change? (fine preservation)
4. Prediction:
   - V5: swapping coarse prefix CHANGES L0 class but preserves within-L0 info
   - MRL: swapping prefix has MIXED effects (no clean separation)

This is causal prefix surgery — direct manipulation of semantic levels by
modifying specific embedding dimensions. If it works, it's definitive evidence
that fractal embeddings have semantically separated structure.
"""

import sys
import os
import json
import torch
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from mrl_v5_baseline import MRLTrainerV5
from multi_model_pipeline import MODELS

RESULTS_DIR = Path(__file__).parent.parent / "results"


def knn_classify(query_emb, ref_embs, ref_labels, k=5):
    """Classify a single query by k-NN majority vote."""
    sims = query_emb @ ref_embs.T
    top_k = np.argsort(-sims)[:k]
    labels = ref_labels[top_k]
    counts = Counter(labels.tolist())
    return max(counts, key=counts.get)


def run_prefix_surgery(
    model_key="bge-small",
    dataset_name="clinc",
    seed=42,
    stage1_epochs=5,
    batch_size=32,
    n_pairs=500,
    device="cuda",
):
    """
    Run the causal prefix surgery experiment.

    For each pair of samples from different L0 classes:
    1. Encode both samples with the trained model
    2. Create chimeric embeddings by swapping prefix blocks
    3. Classify chimeras and measure coarse transfer / fine preservation
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*70}")
    print(f"  CAUSAL PREFIX SURGERY: {model_key} on {dataset_name}")
    print(f"{'='*70}")

    # Load data
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

    class TempDataset:
        def __init__(self, samples, l0_names, l1_names):
            self.samples = samples
            self.level0_names = l0_names
            self.level1_names = l1_names

    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_data.samples = train_samples

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)
    config = MODELS[model_key]

    # Prepare test data
    test_texts = [s.text for s in test_data.samples]
    test_l0 = np.array([s.level0_label for s in test_data.samples])
    test_l1 = np.array([s.level1_label for s in test_data.samples])

    results = {}

    for method_name, TrainerCls, l0_cls in [
        ("v5", V5Trainer, num_l0),
        ("mrl", MRLTrainerV5, num_l1),
    ]:
        print(f"\n--- {method_name.upper()} ---")

        # Train model
        model = FractalModelV5(
            config=config,
            num_l0_classes=l0_cls,
            num_l1_classes=num_l1,
            num_scales=4,
            scale_dim=64,
            device=device,
        ).to(device)

        trainer = TrainerCls(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            device=device,
            stage1_epochs=stage1_epochs,
        )
        trainer.train(batch_size=batch_size, patience=5)

        # Encode all test samples at full length
        full_embs = model.encode(test_texts, batch_size=batch_size).numpy()
        full_embs = full_embs / np.linalg.norm(full_embs, axis=1, keepdims=True)

        # Reference embeddings for kNN classification
        ref_embs = full_embs.copy()
        ref_l0 = test_l0.copy()
        ref_l1 = test_l1.copy()

        # Generate pairs from DIFFERENT L0 classes
        l0_groups = {}
        for i, l0 in enumerate(test_l0):
            l0_groups.setdefault(int(l0), []).append(i)

        l0_classes = list(l0_groups.keys())
        pairs = []
        for _ in range(n_pairs * 5):  # oversample to get enough valid pairs
            c1, c2 = np.random.choice(l0_classes, 2, replace=False)
            i = np.random.choice(l0_groups[c1])
            j = np.random.choice(l0_groups[c2])
            if test_l0[i] != test_l0[j]:
                pairs.append((i, j))
            if len(pairs) >= n_pairs:
                break

        print(f"  Generated {len(pairs)} cross-L0 pairs")

        # Surgery: for each pair, swap the first 64d block
        coarse_transfer_count = 0
        fine_preserved_count = 0
        total_pairs = len(pairs)

        # Also track: original classification accuracy
        orig_l0_correct = 0
        chimera_l0_to_donor = 0
        chimera_l1_preserved = 0

        surgery_details = []

        for idx_a, idx_b in pairs:
            emb_a = full_embs[idx_a].copy()  # "recipient" — gets donor's coarse prefix
            emb_b = full_embs[idx_b].copy()  # "donor" — provides coarse prefix

            # Original classifications (kNN)
            orig_l0_a = knn_classify(emb_a, ref_embs, ref_l0, k=5)
            orig_l1_a = knn_classify(emb_a, ref_embs, ref_l1, k=5)

            # Create chimera: recipient's fine suffix + donor's coarse prefix
            chimera = emb_a.copy()
            chimera[:64] = emb_b[:64]  # Swap first 64 dims (coarse block)
            chimera = chimera / np.linalg.norm(chimera)  # Re-normalize

            # Chimera classifications
            chimera_l0 = knn_classify(chimera, ref_embs, ref_l0, k=5)
            chimera_l1 = knn_classify(chimera, ref_embs, ref_l1, k=5)

            # Did coarse classification change to donor's class?
            donor_l0 = int(test_l0[idx_b])
            recipient_l0 = int(test_l0[idx_a])
            recipient_l1 = int(test_l1[idx_a])

            coarse_transferred = (chimera_l0 == donor_l0)
            fine_preserved = (chimera_l1 == recipient_l1)

            if coarse_transferred:
                coarse_transfer_count += 1
            if fine_preserved:
                fine_preserved_count += 1
            if orig_l0_a == recipient_l0:
                orig_l0_correct += 1

            surgery_details.append({
                "recipient_l0": recipient_l0,
                "donor_l0": donor_l0,
                "recipient_l1": recipient_l1,
                "orig_l0_pred": int(orig_l0_a),
                "chimera_l0_pred": int(chimera_l0),
                "chimera_l1_pred": int(chimera_l1),
                "coarse_transferred": bool(coarse_transferred),
                "fine_preserved": bool(fine_preserved),
            })

        coarse_transfer_rate = coarse_transfer_count / total_pairs
        fine_preservation_rate = fine_preserved_count / total_pairs
        orig_l0_acc = orig_l0_correct / total_pairs

        # Surgery score: high coarse transfer + high fine preservation = good separation
        surgery_score = coarse_transfer_rate + fine_preservation_rate

        print(f"\n  SURGERY RESULTS ({method_name.upper()}):")
        print(f"    Original L0 accuracy:     {orig_l0_acc:.4f}")
        print(f"    Coarse transfer rate:     {coarse_transfer_rate:.4f}")
        print(f"    Fine preservation rate:   {fine_preservation_rate:.4f}")
        print(f"    Surgery score (sum):      {surgery_score:.4f}")
        print(f"    Ideal: coarse_transfer HIGH + fine_preserved HIGH")

        results[method_name] = {
            "coarse_transfer_rate": coarse_transfer_rate,
            "fine_preservation_rate": fine_preservation_rate,
            "surgery_score": surgery_score,
            "orig_l0_accuracy": orig_l0_acc,
            "n_pairs": total_pairs,
        }

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # Summary comparison
    print(f"\n{'='*70}")
    print("  PREFIX SURGERY COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} {'V5':>10} {'MRL':>10} {'Delta':>10}")
    print(f"  {'-'*60}")

    for metric in ["coarse_transfer_rate", "fine_preservation_rate", "surgery_score"]:
        v5_val = results["v5"][metric]
        mrl_val = results["mrl"][metric]
        print(f"  {metric:<30} {v5_val:>10.4f} {mrl_val:>10.4f} {v5_val-mrl_val:>+10.4f}")

    # Save
    out_path = RESULTS_DIR / f"prefix_surgery_{model_key}_{dataset_name}_seed{seed}.json"
    output = {
        "model": model_key,
        "dataset": dataset_name,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        **results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2,
                 default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bge-small")
    parser.add_argument("--dataset", default="clinc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pairs", type=int, default=500)
    args = parser.parse_args()

    run_prefix_surgery(
        model_key=args.model,
        dataset_name=args.dataset,
        seed=args.seed,
        n_pairs=args.n_pairs,
    )
