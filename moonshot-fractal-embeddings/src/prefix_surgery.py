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

        # Reference embeddings at DIFFERENT granularities
        ref_full = full_embs.copy()
        ref_prefix = full_embs[:, :64].copy()  # j=1: first 64d
        ref_prefix = ref_prefix / np.linalg.norm(ref_prefix, axis=1, keepdims=True)
        ref_suffix = full_embs[:, 64:].copy()  # dims 64-256
        ref_suffix = ref_suffix / np.linalg.norm(ref_suffix, axis=1, keepdims=True)
        ref_l0 = test_l0.copy()
        ref_l1 = test_l1.copy()

        # Information localization analysis (key differentiator!)
        # Measure: where is L0 info? where is L1 info?
        n_test = len(test_l0)
        loc_counters = {"prefix_l0": 0, "prefix_l1": 0, "suffix_l0": 0, "suffix_l1": 0}
        for i in range(n_test):
            p = ref_prefix[i]
            s = ref_suffix[i]
            if knn_classify(p, ref_prefix, ref_l0, k=5) == int(test_l0[i]):
                loc_counters["prefix_l0"] += 1
            if knn_classify(p, ref_prefix, ref_l1, k=5) == int(test_l1[i]):
                loc_counters["prefix_l1"] += 1
            if knn_classify(s, ref_suffix, ref_l0, k=5) == int(test_l0[i]):
                loc_counters["suffix_l0"] += 1
            if knn_classify(s, ref_suffix, ref_l1, k=5) == int(test_l1[i]):
                loc_counters["suffix_l1"] += 1
        localization = {k: v / n_test for k, v in loc_counters.items()}

        print(f"\n  INFORMATION LOCALIZATION ({method_name.upper()}):")
        print(f"    Prefix (64d) L0 acc: {localization['prefix_l0']:.4f}")
        print(f"    Prefix (64d) L1 acc: {localization['prefix_l1']:.4f}  <- KEY: V5 should be LOW")
        print(f"    Suffix (192d) L0 acc: {localization['suffix_l0']:.4f}")
        print(f"    Suffix (192d) L1 acc: {localization['suffix_l1']:.4f}")
        print(f"    L1 leakage into prefix: {localization['prefix_l1']:.4f}")

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
        # Measure at the RIGHT granularity:
        #   L0 classification: use ONLY prefix (64d) — where coarse info lives
        #   L1 classification: use ONLY suffix (192d) — where fine info lives
        #   Also measure on full 256d for comparison
        counters = {
            "prefix_l0_transfer": 0,  # chimera prefix -> donor L0?
            "suffix_l1_preserved": 0,  # chimera suffix -> recipient L1?
            "full_l0_transfer": 0,    # full 256d -> donor L0?
            "full_l1_preserved": 0,   # full 256d -> recipient L1?
            "orig_prefix_l0_correct": 0,
            "orig_suffix_l1_correct": 0,
            "orig_full_l0_correct": 0,
        }
        total_pairs = len(pairs)
        surgery_details = []

        for idx_a, idx_b in pairs:
            emb_a = full_embs[idx_a].copy()  # "recipient"
            emb_b = full_embs[idx_b].copy()  # "donor"

            donor_l0 = int(test_l0[idx_b])
            recipient_l0 = int(test_l0[idx_a])
            recipient_l1 = int(test_l1[idx_a])

            # Original prefix-only L0 accuracy
            a_prefix = emb_a[:64] / np.linalg.norm(emb_a[:64])
            orig_prefix_l0 = knn_classify(a_prefix, ref_prefix, ref_l0, k=5)
            if orig_prefix_l0 == recipient_l0:
                counters["orig_prefix_l0_correct"] += 1

            # Original suffix-only L1 accuracy
            a_suffix = emb_a[64:] / np.linalg.norm(emb_a[64:])
            orig_suffix_l1 = knn_classify(a_suffix, ref_suffix, ref_l1, k=5)
            if orig_suffix_l1 == recipient_l1:
                counters["orig_suffix_l1_correct"] += 1

            # Original full L0 accuracy
            orig_full_l0 = knn_classify(emb_a, ref_full, ref_l0, k=5)
            if orig_full_l0 == recipient_l0:
                counters["orig_full_l0_correct"] += 1

            # Create chimera: donor's prefix + recipient's suffix
            chimera = emb_a.copy()
            chimera[:64] = emb_b[:64]

            # Chimera prefix -> classify L0 with prefix-only kNN
            chi_prefix = chimera[:64] / np.linalg.norm(chimera[:64])
            chi_prefix_l0 = knn_classify(chi_prefix, ref_prefix, ref_l0, k=5)
            if chi_prefix_l0 == donor_l0:
                counters["prefix_l0_transfer"] += 1

            # Chimera suffix -> classify L1 with suffix-only kNN (unchanged!)
            chi_suffix = chimera[64:] / np.linalg.norm(chimera[64:])
            chi_suffix_l1 = knn_classify(chi_suffix, ref_suffix, ref_l1, k=5)
            if chi_suffix_l1 == recipient_l1:
                counters["suffix_l1_preserved"] += 1

            # Also full-embedding classification for comparison
            chimera_norm = chimera / np.linalg.norm(chimera)
            chi_full_l0 = knn_classify(chimera_norm, ref_full, ref_l0, k=5)
            chi_full_l1 = knn_classify(chimera_norm, ref_full, ref_l1, k=5)
            if chi_full_l0 == donor_l0:
                counters["full_l0_transfer"] += 1
            if chi_full_l1 == recipient_l1:
                counters["full_l1_preserved"] += 1

            surgery_details.append({
                "recipient_l0": recipient_l0,
                "donor_l0": donor_l0,
                "recipient_l1": recipient_l1,
                "chi_prefix_l0": int(chi_prefix_l0),
                "chi_suffix_l1": int(chi_suffix_l1),
                "chi_full_l0": int(chi_full_l0),
                "prefix_transferred": bool(chi_prefix_l0 == donor_l0),
                "suffix_preserved": bool(chi_suffix_l1 == recipient_l1),
            })

        rates = {k: v / total_pairs for k, v in counters.items()}

        # Surgery score: prefix L0 transfer + suffix L1 preservation
        surgery_score = rates["prefix_l0_transfer"] + rates["suffix_l1_preserved"]

        print(f"\n  SURGERY RESULTS ({method_name.upper()}):")
        print(f"    --- Prefix-only (64d) L0 classification ---")
        print(f"    Original L0 acc (prefix):   {rates['orig_prefix_l0_correct']:.4f}")
        print(f"    Chimera L0 -> donor (prefix): {rates['prefix_l0_transfer']:.4f}")
        print(f"    --- Suffix-only (192d) L1 classification ---")
        print(f"    Original L1 acc (suffix):   {rates['orig_suffix_l1_correct']:.4f}")
        print(f"    Chimera L1 preserved (suffix): {rates['suffix_l1_preserved']:.4f}")
        print(f"    --- Full embedding (256d) for comparison ---")
        print(f"    Original L0 acc (full):     {rates['orig_full_l0_correct']:.4f}")
        print(f"    Chimera L0 -> donor (full): {rates['full_l0_transfer']:.4f}")
        print(f"    Chimera L1 preserved (full): {rates['full_l1_preserved']:.4f}")
        print(f"    --- Surgery score (prefix_transfer + suffix_preserved) ---")
        print(f"    Surgery score:              {surgery_score:.4f}")

        results[method_name] = {
            **rates,
            "surgery_score": surgery_score,
            "localization": localization,
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

    for metric in ["prefix_l0_transfer", "suffix_l1_preserved", "surgery_score"]:
        v5_val = results["v5"][metric]
        mrl_val = results["mrl"][metric]
        print(f"  {metric:<30} {v5_val:>10.4f} {mrl_val:>10.4f} {v5_val-mrl_val:>+10.4f}")

    print(f"\n  {'='*70}")
    print("  INFORMATION LOCALIZATION COMPARISON (key result)")
    print(f"  {'='*70}")
    print(f"  {'Metric':<30} {'V5':>10} {'MRL':>10} {'Delta':>10}")
    print(f"  {'-'*60}")
    for metric in ["prefix_l0", "prefix_l1", "suffix_l0", "suffix_l1"]:
        v5_val = results["v5"]["localization"][metric]
        mrl_val = results["mrl"]["localization"][metric]
        marker = " <- KEY" if metric == "prefix_l1" else ""
        print(f"  {metric:<30} {v5_val:>10.4f} {mrl_val:>10.4f} {v5_val-mrl_val:>+10.4f}{marker}")

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
