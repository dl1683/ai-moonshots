"""
Causal Ablation Study for Steerability
=======================================

Tests whether V5's prefix specialization is CAUSED by hierarchy-aligned supervision
or is a trivial artifact of the easier L0 label space.

Ablations:
1. V5 (control): short→L0, full→L1  [already have results]
2. MRL (control): short→L1, full→L1  [already have results]
3. INVERTED: short→L1, full→L0       [if steerability REVERSES, proves causality]
4. NO_PREFIX: full→L1 only, no prefix path [tests if prefix supervision matters]
5. RANDOM_L0: short→shuffled_L0, full→L1 [tests if coherent L0 labels needed]

The key prediction:
- INVERTED should show j=1 as FINE specialist (high L1, low L0)
- NO_PREFIX should show flat steerability (like MRL or worse)
- RANDOM_L0 should show no steerability (random labels = no signal)
"""

import sys
import os
import json
import gc
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS, load_model
from fractal_v5 import (
    FractalModelV5, V5Trainer, ContrastiveDatasetV5,
    split_train_val, knn_accuracy
)


class InvertedV5Trainer(V5Trainer):
    """V5 Trainer with INVERTED supervision: short→L1, full→L0."""

    def _train_epoch(self, dataloader, optimizer, epoch: int, stage: int) -> Tuple[float, Dict]:
        total_loss = 0
        num_batches = 0
        use_amp = (stage == 2)

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = batch['anchor_ids'].shape[0]
            prefix_lengths = self.sample_prefix_length(batch_size).to(self.device)

            with autocast(enabled=use_amp):
                # ===== FULL PATH → L0 labels (INVERTED from V5) =====
                full_dropout_mask = self.create_full_dropout_mask(batch_size)
                anchor_full = self.model(
                    batch['anchor_ids'], batch['anchor_mask'],
                    block_dropout_mask=full_dropout_mask
                )
                # Use L0 positives for full path
                l0_pos_full = self.model(batch['l0_pos_ids'], batch['l0_pos_mask'])
                full_emb = anchor_full['full_embedding']
                l0_pos_emb = l0_pos_full['full_embedding']

                full_contrastive = self.contrastive_loss(full_emb, l0_pos_emb)
                full_margin = self.margin_loss(full_emb, batch['l0_labels'])
                # classify_leaf still outputs num_l1 logits, but we target L0
                # We need classify_top for L0 labels
                full_cls = F.cross_entropy(
                    self.model.fractal_head.classify_top(full_emb),
                    batch['l0_labels']
                )
                loss_full = full_contrastive + self.MARGIN_WEIGHT * full_margin + self.CLASS_WEIGHT * full_cls

                # ===== PREFIX PATH → L1 labels (INVERTED from V5) =====
                prefix_dropout_mask = self.create_block_dropout_mask(batch_size, prefix_lengths)
                anchor_prefix = self.model(
                    batch['anchor_ids'], batch['anchor_mask'],
                    block_dropout_mask=prefix_dropout_mask
                )
                # Use L1 positives for prefix path
                l1_pos_prefix = self.model(batch['l1_pos_ids'], batch['l1_pos_mask'])
                mode_prefix_len = prefix_lengths.mode().values.item()
                prefix_emb = self.model.fractal_head.get_prefix_embedding(
                    anchor_prefix['blocks'], mode_prefix_len
                )
                l1_pos_emb = l1_pos_prefix['full_embedding']

                prefix_contrastive = self.contrastive_loss(prefix_emb, l1_pos_emb)
                prefix_margin = self.margin_loss(prefix_emb, batch['l1_labels'])
                # classify_leaf for L1 labels
                prefix_cls = F.cross_entropy(
                    self.model.fractal_head.classify_leaf(prefix_emb),
                    batch['l1_labels']
                )
                loss_prefix = prefix_contrastive + self.MARGIN_WEIGHT * prefix_margin + self.CLASS_WEIGHT * prefix_cls

                loss = loss_full + self.PREFIX_WEIGHT * loss_prefix

            if torch.isnan(loss):
                continue

            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        # Evaluate
        val_metrics = self._evaluate(self.val_dataset.samples)
        return avg_loss, val_metrics


class NoPrefixV5Trainer(V5Trainer):
    """V5 Trainer with NO prefix supervision — full→L1 only."""

    def _train_epoch(self, dataloader, optimizer, epoch: int, stage: int) -> Tuple[float, Dict]:
        total_loss = 0
        num_batches = 0
        use_amp = (stage == 2)

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = batch['anchor_ids'].shape[0]

            with autocast(enabled=use_amp):
                # ===== FULL PATH ONLY → L1 labels =====
                full_dropout_mask = self.create_full_dropout_mask(batch_size)
                anchor_full = self.model(
                    batch['anchor_ids'], batch['anchor_mask'],
                    block_dropout_mask=full_dropout_mask
                )
                l1_pos_full = self.model(batch['l1_pos_ids'], batch['l1_pos_mask'])
                full_emb = anchor_full['full_embedding']
                l1_pos_emb = l1_pos_full['full_embedding']

                full_contrastive = self.contrastive_loss(full_emb, l1_pos_emb)
                full_margin = self.margin_loss(full_emb, batch['l1_labels'])
                full_cls = F.cross_entropy(
                    self.model.fractal_head.classify_leaf(full_emb),
                    batch['l1_labels']
                )
                loss = full_contrastive + self.MARGIN_WEIGHT * full_margin + self.CLASS_WEIGHT * full_cls

            if torch.isnan(loss):
                continue

            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        val_metrics = self._evaluate(self.val_dataset.samples)
        return avg_loss, val_metrics


def evaluate_prefix_steerability(model, test_samples, max_samples=2000):
    """Evaluate at all prefix lengths and compute steerability metrics."""
    samples = test_samples[:min(max_samples, len(test_samples))]
    texts = [s.text for s in samples]
    l0_labels = np.array([s.level0_label for s in samples])
    l1_labels = np.array([s.level1_label for s in samples])

    results = {}
    for j in [1, 2, 3, 4]:
        prefix_len = j if j < 4 else None
        emb = model.encode(texts, batch_size=32, prefix_len=prefix_len).numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        # KNN evaluation
        sims = emb @ emb.T
        np.fill_diagonal(sims, -1)

        k = 5
        l0_c = 0
        l1_c = 0
        for i in range(len(samples)):
            top5 = np.argsort(-sims[i])[:k]
            if np.bincount(l0_labels[top5]).argmax() == l0_labels[i]:
                l0_c += 1
            if np.bincount(l1_labels[top5]).argmax() == l1_labels[i]:
                l1_c += 1

        results[f'j{j}'] = {
            'l0': l0_c / len(samples),
            'l1': l1_c / len(samples),
        }

    # Steerability metrics (Codex-recommended suite)
    j1 = results['j1']
    j4 = results['j4']
    metrics = {
        'prefix_results': results,
        'short_coarse': j1['l0'],           # ShortCoarse = L0@j1
        'full_fine': j4['l1'],               # FullFine = L1@j4
        'specialization_gap': j1['l0'] - j1['l1'],  # SpecializationGap
        'coarse_gain': j1['l0'] - j4['l0'],  # CoarseGain
        'fine_gain': j4['l1'] - j1['l1'],    # FineGain
        'steerability_score': (j1['l0'] - j4['l0']) + (j4['l1'] - j1['l1']),
    }
    return metrics


def run_single_ablation(
    ablation_name,
    trainer_class,
    model_key="bge-small",
    dataset_name="clinc",
    num_l0_override=None,
    num_l1_override=None,
    stage1_epochs=5,
    batch_size=32,
    seed=42,
    device="cuda",
):
    """Run a single ablation variant."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_name} (seed={seed})")
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

    num_l0 = num_l0_override or len(train_data.level0_names)
    num_l1 = num_l1_override or len(train_data.level1_names)
    config = MODELS[model_key]

    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    trainer = trainer_class(
        model=model,
        train_dataset=train_data_obj,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
    )
    trainer.train(batch_size=batch_size, patience=5)

    # Evaluate steerability
    steerability = evaluate_prefix_steerability(model, test_data.samples)
    print(f"\n  Steerability results:")
    for j in [1, 2, 3, 4]:
        jr = steerability['prefix_results'][f'j{j}']
        print(f"    j={j}: L0={jr['l0']:.4f}, L1={jr['l1']:.4f}")
    print(f"  ShortCoarse (L0@j1):     {steerability['short_coarse']:.4f}")
    print(f"  FullFine (L1@j4):        {steerability['full_fine']:.4f}")
    print(f"  SpecializationGap:       {steerability['specialization_gap']:.4f}")
    print(f"  SteerabilityScore:       {steerability['steerability_score']:.4f}")

    # Final classification accuracy (j=4)
    j4 = steerability['prefix_results']['j4']
    print(f"  Final L0: {j4['l0']:.4f}, L1: {j4['l1']:.4f}")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'ablation': ablation_name,
        'seed': seed,
        **steerability,
    }


def run_all_ablations(
    model_key="bge-small",
    dataset_name="clinc",
    seeds=[42, 123, 456],
    stage1_epochs=5,
    device="cuda",
):
    """Run all ablation variants and compare."""
    all_results = {}

    ablations = {
        'v5': V5Trainer,
        'inverted': InvertedV5Trainer,
        'no_prefix': NoPrefixV5Trainer,
    }

    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)

    for ablation_name, trainer_class in ablations.items():
        all_results[ablation_name] = []
        for seed in seeds:
            result = run_single_ablation(
                ablation_name=ablation_name,
                trainer_class=trainer_class,
                model_key=model_key,
                dataset_name=dataset_name,
                stage1_epochs=stage1_epochs,
                seed=seed,
                device=device,
            )
            all_results[ablation_name].append(result)

    # Summary
    print("\n" + "=" * 70)
    print("CAUSAL ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Ablation':<15} {'ShortCoarse':>12} {'FullFine':>10} {'SpecGap':>10} {'Steerability':>13}")
    print("-" * 60)

    for name, results in all_results.items():
        sc = np.mean([r['short_coarse'] for r in results])
        ff = np.mean([r['full_fine'] for r in results])
        sg = np.mean([r['specialization_gap'] for r in results])
        ss = np.mean([r['steerability_score'] for r in results])
        print(f"{name:<15} {sc:>12.4f} {ff:>10.4f} {sg:>10.4f} {ss:>13.4f}")

    print()
    print("INTERPRETATION:")
    v5_ss = np.mean([r['steerability_score'] for r in all_results.get('v5', [])])
    inv_ss = np.mean([r['steerability_score'] for r in all_results.get('inverted', [])])
    nop_ss = np.mean([r['steerability_score'] for r in all_results.get('no_prefix', [])])

    if inv_ss < 0 and v5_ss > 0:
        print("  INVERTED shows REVERSED steerability => CAUSAL PROOF!")
        print("  Supervision direction determines specialization, not label complexity.")
    elif abs(inv_ss) < abs(v5_ss) * 0.5:
        print("  INVERTED shows reduced steerability => partial causality.")
    else:
        print("  INVERTED shows similar steerability => specialization may be trivial.")

    if abs(nop_ss) < abs(v5_ss) * 0.3:
        print("  NO_PREFIX shows low steerability => prefix supervision is essential.")

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
        'model': model_key,
        'dataset': dataset_name,
        'seeds': seeds,
        'timestamp': datetime.now().isoformat(),
        'results': convert(all_results),
    }
    out_path = results_dir / f"ablation_steerability_{model_key}_{dataset_name}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bge-small")
    parser.add_argument("--dataset", default="clinc")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    seeds = [42, 123, 456][:args.seeds]
    run_all_ablations(
        model_key=args.model,
        dataset_name=args.dataset,
        seeds=seeds,
        stage1_epochs=args.epochs,
    )
