"""
MRL (Matryoshka Representation Learning) Baseline for V5 Comparison
====================================================================

This is a FAIR baseline for comparing against Fractal V5's hierarchy-aligned
progressive prefix supervision. The key experimental question:

    "Does hierarchy-alignment matter, or does standard MRL (same loss at all
    prefix lengths) achieve similar multi-scale quality?"

Design (from Codex):
- Reuses FractalModelV5 but with num_l0_classes=num_l1_classes so both
  head_top and head_leaf output L1 (finest-grain) logits.
- MRL branch: for each prefix length j in {1,2,3,4}, apply the SAME loss
  targeting L1 labels (no hierarchy-specific supervision).
- Full branch: identical to V5 (InfoNCE + Margin + CE on L1 labels).
- Same hyperparameters, block dropout, dataset, and evaluation as V5.

The ONLY difference from V5:
- V5: prefix path uses L0 (coarse) labels => hierarchy-specific supervision
- MRL: prefix path uses L1 (fine) labels => same labels at every scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import gc
import copy
from pathlib import Path

from multi_model_pipeline import MODELS, ModelConfig
from fractal_v5 import (
    FractalModelV5,
    FractalHeadV5,
    ContrastiveDatasetV5,
    collate_fn_v5,
    split_train_val,
)

from torch.cuda.amp import autocast, GradScaler


class MRLTrainerV5:
    """
    Matryoshka Representation Learning trainer -- fair baseline for V5.

    Key difference from V5Trainer:
    - V5Trainer applies hierarchy-aligned losses: L0 labels on prefix, L1 on full.
    - MRLTrainerV5 applies L1 labels at EVERY prefix length. No hierarchy awareness.

    Everything else is identical: same architecture, same hyperparameters,
    same block dropout, same contrastive + margin + CE loss formulation.

    Training loop:
      L_full  = InfoNCE(z_full, z_pos) + 0.5*Margin(z_full, y1) + CE(head_top(z_full), y1)
      For j in {1,2,3,4} with weights [0.4, 0.3, 0.2, 0.1]:
        z_j = prefix_embedding(blocks, j)
        L_j = InfoNCE(z_j, z_pos) + 0.5*Margin(z_j, y1) + CE(head_top(z_j), y1)
      L_mrl = sum_j p_j * L_j
      L_total = L_full + 0.6 * L_mrl
    """

    # Hyperparameters -- identical to V5Trainer
    PREFIX_PROBS = [0.4, 0.3, 0.2, 0.1]  # Weights for j=1,2,3,4
    BLOCK_KEEP_PROBS = [0.95, 0.9, 0.8, 0.7]  # Block dropout for full path
    MRL_WEIGHT = 0.6  # Weight for L_mrl in total loss (matches V5 PREFIX_WEIGHT)
    MARGIN_WEIGHT = 0.5  # Weight for margin loss
    CLASS_WEIGHT = 1.0   # Weight for classification loss

    def __init__(
        self,
        model: FractalModelV5,
        train_dataset,
        val_dataset,
        device: str = "cuda",
        # Stage 1 params
        stage1_epochs: int = 5,
        stage1_lr: float = 1e-4,
        # Stage 2 params
        stage2_epochs: int = 0,
        stage2_lr_head: float = 2e-5,
        stage2_lr_backbone: float = 1e-6,
        unfreeze_layers: int = 4,
        # Other
        temperature: float = 0.07,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

        self.stage1_epochs = stage1_epochs
        self.stage1_lr = stage1_lr
        self.stage2_epochs = stage2_epochs
        self.stage2_lr_head = stage2_lr_head
        self.stage2_lr_backbone = stage2_lr_backbone
        self.unfreeze_layers = unfreeze_layers
        self.temperature = temperature

        # Mixed precision scaler for Stage 2
        self.scaler = GradScaler()

    def create_full_dropout_mask(self, batch_size: int) -> torch.Tensor:
        """
        Create block dropout mask for full path per design spec.
        keep_probs = [0.95, 0.9, 0.8, 0.7] for blocks 0,1,2,3
        """
        device = self.device
        mask = torch.ones(batch_size, self.model.num_scales, device=device)

        for block_idx, keep_prob in enumerate(self.BLOCK_KEEP_PROBS):
            drop_mask = torch.rand(batch_size, device=device) > keep_prob
            mask[drop_mask, block_idx] = 0.0

        return mask

    def contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        logits = anchor @ positive.T / self.temperature
        targets = torch.arange(len(anchor), device=anchor.device)
        return F.cross_entropy(logits, targets)

    def margin_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.2,
    ) -> torch.Tensor:
        """In-batch margin ranking loss."""
        batch_size = embeddings.shape[0]

        # Similarity matrix
        sims = embeddings @ embeddings.T

        # Same-class mask
        same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
        same_class.fill_diagonal_(False)

        # Different-class mask
        diff_class = ~same_class
        diff_class.fill_diagonal_(False)

        losses = []
        for i in range(batch_size):
            pos_mask = same_class[i]
            if not pos_mask.any():
                continue
            pos_sims = sims[i][pos_mask]

            neg_mask = diff_class[i]
            if not neg_mask.any():
                continue
            neg_sims = sims[i][neg_mask]

            for pos_sim in pos_sims:
                loss = F.relu(neg_sims - pos_sim + margin).mean()
                losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=embeddings.device)

    def train(self, batch_size: int = 32, patience: int = 5) -> List[Dict]:
        from torch.utils.data import DataLoader

        train_ds = ContrastiveDatasetV5(self.train_dataset.samples, self.model.tokenizer)

        def collate(batch):
            return collate_fn_v5(batch, self.model.tokenizer, self.device)

        dataloader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate, drop_last=True,
        )

        history = []
        global_best_score = -float('inf')
        global_best_state = None

        # ==================== STAGE 1: Head only ====================
        print(f"\n[MRL Stage 1] Training HEAD ONLY ({self.stage1_epochs} epochs)")
        self.model.freeze_backbone()

        optimizer = torch.optim.AdamW(
            self.model.fractal_head.parameters(),
            lr=self.stage1_lr,
            weight_decay=0.01,
        )

        for epoch in range(self.stage1_epochs):
            self.model.train()
            epoch_loss, epoch_metrics = self._train_epoch(dataloader, optimizer, epoch, stage=1)

            val_score = self._evaluate()
            history.append({
                'stage': 1, 'epoch': epoch + 1,
                'loss': epoch_loss,
                **epoch_metrics,
                **val_score,
            })

            print(f"  MRL Stage 1 Epoch {epoch+1}: loss={epoch_loss:.4f}, "
                  f"L0={val_score['l0_accuracy']:.4f}, L1={val_score['l1_accuracy']:.4f}")

            if val_score['l0_accuracy'] + val_score['l1_accuracy'] > global_best_score:
                global_best_score = val_score['l0_accuracy'] + val_score['l1_accuracy']
                global_best_state = copy.deepcopy(self.model.state_dict())

        # ==================== STAGE 2: Head + Backbone ====================
        if self.stage2_epochs > 0:
            print(f"\n[MRL Stage 2] Training HEAD + BACKBONE ({self.stage2_epochs} epochs)")
            self.model.unfreeze_last_n_layers(self.unfreeze_layers)

            head_params = list(self.model.fractal_head.parameters())
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]

            optimizer = torch.optim.AdamW([
                {'params': head_params, 'lr': self.stage2_lr_head},
                {'params': backbone_params, 'lr': self.stage2_lr_backbone},
            ], weight_decay=0.01)

            no_improve = 0

            for epoch in range(self.stage2_epochs):
                self.model.train()
                epoch_loss, epoch_metrics = self._train_epoch(dataloader, optimizer, epoch, stage=2)

                val_score = self._evaluate()
                history.append({
                    'stage': 2, 'epoch': epoch + 1,
                    'loss': epoch_loss,
                    **epoch_metrics,
                    **val_score,
                })

                print(f"  MRL Stage 2 Epoch {epoch+1}: loss={epoch_loss:.4f}, "
                      f"L0={val_score['l0_accuracy']:.4f}, L1={val_score['l1_accuracy']:.4f}")

                score = val_score['l0_accuracy'] + val_score['l1_accuracy']
                if score > global_best_score:
                    global_best_score = score
                    global_best_state = copy.deepcopy(self.model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  Early stopping at MRL Stage 2 Epoch {epoch + 1}")
                        break

        # Restore best
        if global_best_state:
            self.model.load_state_dict(global_best_state)
            print(f"\nRestored best MRL model (score={global_best_score:.4f})")

        return history

    def _train_epoch(self, dataloader, optimizer, epoch: int, stage: int) -> Tuple[float, Dict]:
        """
        MRL training epoch.

        Key difference from V5:
        - Instead of one sampled prefix length per batch, we compute losses
          at ALL prefix lengths j=1..4, weighted by PREFIX_PROBS.
        - ALL prefix losses target L1 (finest-grain) labels, NOT L0.
        """
        total_loss = 0
        total_loss_full = 0
        total_loss_mrl = 0
        num_batches = 0

        use_amp = (stage == 2)

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = batch['anchor_ids'].shape[0]

            with autocast(enabled=use_amp):
                # ===== FULL PATH (identical to V5) =====
                full_dropout_mask = self.create_full_dropout_mask(batch_size)

                anchor_full = self.model(
                    batch['anchor_ids'], batch['anchor_mask'],
                    block_dropout_mask=full_dropout_mask,
                )
                l1_pos_full = self.model(batch['l1_pos_ids'], batch['l1_pos_mask'])

                full_emb = anchor_full['full_embedding']
                l1_pos_emb = l1_pos_full['full_embedding']

                # L_full = InfoNCE + 0.5*Margin + CE(head_top, y1)
                # Note: head_top now outputs L1 logits since num_l0_classes=num_l1_classes
                full_contrastive = self.contrastive_loss(full_emb, l1_pos_emb)
                full_margin = self.margin_loss(full_emb, batch['l1_labels'])
                full_cls = F.cross_entropy(
                    self.model.fractal_head.classify_leaf(full_emb),
                    batch['l1_labels'],
                )

                loss_full = (
                    full_contrastive
                    + self.MARGIN_WEIGHT * full_margin
                    + self.CLASS_WEIGHT * full_cls
                )

                # ===== MRL PATH (different from V5) =====
                # Run anchor once WITHOUT block dropout to get all blocks
                anchor_mrl = self.model(batch['anchor_ids'], batch['anchor_mask'])
                # Use L1 positives for contrastive (MRL targets finest grain everywhere)
                l1_pos_mrl = self.model(batch['l1_pos_ids'], batch['l1_pos_mask'])
                l1_pos_emb_mrl = l1_pos_mrl['full_embedding']

                loss_mrl = torch.tensor(0.0, device=self.device)

                for j_idx, (j, p_j) in enumerate(
                    zip(range(1, self.model.num_scales + 1), self.PREFIX_PROBS)
                ):
                    # Get prefix embedding at length j
                    prefix_emb_j = self.model.fractal_head.get_prefix_embedding(
                        anchor_mrl['blocks'], j
                    )

                    # L_j = InfoNCE(z_j, z_pos) + 0.5*Margin(z_j, y1) + CE(head_top(z_j), y1)
                    # ALL targeting L1 labels
                    contrastive_j = self.contrastive_loss(prefix_emb_j, l1_pos_emb_mrl)
                    margin_j = self.margin_loss(prefix_emb_j, batch['l1_labels'])
                    cls_j = F.cross_entropy(
                        self.model.fractal_head.classify_top(prefix_emb_j),
                        batch['l1_labels'],
                    )

                    loss_j = (
                        contrastive_j
                        + self.MARGIN_WEIGHT * margin_j
                        + self.CLASS_WEIGHT * cls_j
                    )

                    loss_mrl = loss_mrl + p_j * loss_j

                # ===== TOTAL LOSS =====
                loss = loss_full + self.MRL_WEIGHT * loss_mrl

            # Skip if NaN
            if torch.isnan(loss):
                print(f"    Warning: NaN loss at batch {batch_idx}, skipping")
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
            total_loss_full += loss_full.item()
            total_loss_mrl += loss_mrl.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {
            'loss_full': total_loss_full / max(num_batches, 1),
            'loss_mrl': total_loss_mrl / max(num_batches, 1),
        }
        return avg_loss, metrics

    def _evaluate(self) -> Dict:
        """Evaluate on validation set (identical to V5)."""
        self.model.eval()
        samples = self.val_dataset.samples[:min(1000, len(self.val_dataset.samples))]
        texts = [s.text for s in samples]
        l0_labels = np.array([s.level0_label for s in samples])
        l1_labels = np.array([s.level1_label for s in samples])

        embeddings = self.model.encode(texts, batch_size=32).numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        def knn_accuracy(emb, labels, k=5):
            correct = 0
            for i in range(len(emb)):
                sims = emb @ emb[i]
                sims[i] = -float('inf')
                top_k = np.argsort(-sims)[:k]
                neighbor_labels = labels[top_k]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                pred = unique[np.argmax(counts)]
                if pred == labels[i]:
                    correct += 1
            return correct / len(emb)

        return {
            'l0_accuracy': knn_accuracy(embeddings, l0_labels),
            'l1_accuracy': knn_accuracy(embeddings, l1_labels),
        }

    def evaluate_prefix_accuracy(self) -> Dict:
        """
        Evaluate accuracy at each prefix length.
        Tests if MRL learns any scale separation without hierarchy alignment.
        """
        self.model.eval()
        samples = self.val_dataset.samples[:min(500, len(self.val_dataset.samples))]
        texts = [s.text for s in samples]
        l0_labels = np.array([s.level0_label for s in samples])
        l1_labels = np.array([s.level1_label for s in samples])

        results = {}

        def knn_accuracy(emb, labels, k=5):
            correct = 0
            for i in range(len(emb)):
                sims = emb @ emb[i]
                sims[i] = -float('inf')
                top_k = np.argsort(-sims)[:k]
                neighbor_labels = labels[top_k]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                pred = unique[np.argmax(counts)]
                if pred == labels[i]:
                    correct += 1
            return correct / len(emb)

        for j in [1, 2, 3, 4]:
            prefix_len = j if j < 4 else None  # j=4 means full
            emb = self.model.encode(texts, batch_size=32, prefix_len=prefix_len).numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

            results[f'j{j}_l0'] = knn_accuracy(emb, l0_labels)
            results[f'j{j}_l1'] = knn_accuracy(emb, l1_labels)

        return results


def run_mrl_experiment(
    model_key: str = "bge-small",
    dataset_name: str = "yahoo",
    stage1_epochs: int = 5,
    stage2_epochs: int = 0,
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Run MRL baseline experiment.

    Mirrors run_v5_experiment() from fractal_v5.py for fair comparison.
    The ONLY difference: num_l0_classes=num_l1_classes and MRL training loop.
    """
    import random
    import torch
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from hierarchical_datasets import load_hierarchical_dataset

    print("=" * 70)
    print(f"MRL BASELINE (V5-comparable): {model_key} on {dataset_name}")
    print("=" * 70)
    print("This is the Matryoshka baseline: same loss (L1) at all prefix lengths.")
    print("Comparison target: Fractal V5 which uses hierarchy-aligned losses.")
    print(f"Config: PREFIX_PROBS={MRLTrainerV5.PREFIX_PROBS}, "
          f"BLOCK_KEEP={MRLTrainerV5.BLOCK_KEEP_PROBS}")

    # Load data
    print("\n[1] Loading data...")
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

    print(f"  Train: {len(train_data.samples)}")
    print(f"  Val: {len(val_data.samples)}")
    print(f"  Test: {len(test_data.samples)}")

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)

    # Baseline (unfinetuned model)
    print("\n[2] Unfinetuned baseline...")
    from multi_model_pipeline import load_model

    base_model = load_model(model_key, use_fractal=False, device=device)

    def evaluate_knn(model, dataset, max_samples=2000):
        samples = dataset.samples[:min(max_samples, len(dataset.samples))]
        texts = [s.text for s in samples]
        l0_labels = np.array([s.level0_label for s in samples])
        l1_labels = np.array([s.level1_label for s in samples])

        embeddings = model.encode(texts, batch_size=32).numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        def knn_acc(emb, labels, k=5):
            correct = 0
            for i in range(len(emb)):
                sims = emb @ emb[i]
                sims[i] = -float('inf')
                top_k = np.argsort(-sims)[:k]
                neighbor_labels = labels[top_k]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                pred = unique[np.argmax(counts)]
                if pred == labels[i]:
                    correct += 1
            return correct / len(emb)

        return {
            'l0_accuracy': knn_acc(embeddings, l0_labels),
            'l1_accuracy': knn_acc(embeddings, l1_labels),
        }

    base_results = evaluate_knn(base_model, test_data)
    print(f"  L0: {base_results['l0_accuracy']:.4f}")
    print(f"  L1: {base_results['l1_accuracy']:.4f}")

    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # MRL Model
    # KEY: num_l0_classes=num_l1 so head_top also outputs L1-sized logits
    print(f"\n[3] Training MRL baseline...")
    config = MODELS[model_key]

    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l1,   # <-- THE KEY DIFFERENCE: both heads output L1 logits
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    trainer = MRLTrainerV5(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
        stage1_lr=1e-4,
        stage2_epochs=stage2_epochs,
        unfreeze_layers=4,
        temperature=0.07,
    )

    history = trainer.train(batch_size=batch_size, patience=5)

    # Final eval
    print("\n[4] Final evaluation on TEST set...")
    model.eval()
    mrl_results = evaluate_knn(model, test_data)
    print(f"  L0: {mrl_results['l0_accuracy']:.4f}")
    print(f"  L1: {mrl_results['l1_accuracy']:.4f}")

    # Evaluate at each prefix length
    print("\n[5] Prefix-length accuracy...")
    test_data_temp = TempDataset(
        test_data.samples, test_data.level0_names, test_data.level1_names,
    )
    trainer.val_dataset = test_data_temp
    prefix_results = trainer.evaluate_prefix_accuracy()

    print("  Prefix Length | L0 Acc | L1 Acc")
    print("  " + "-" * 35)
    for j in [1, 2, 3, 4]:
        l0 = prefix_results[f'j{j}_l0']
        l1 = prefix_results[f'j{j}_l1']
        print(f"  j={j} ({j*64}d)    | {l0:.4f} | {l1:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("MRL BASELINE SUMMARY")
    print("=" * 70)
    delta_l0 = mrl_results['l0_accuracy'] - base_results['l0_accuracy']
    delta_l1 = mrl_results['l1_accuracy'] - base_results['l1_accuracy']

    print(f"  {'Metric':<10} {'Baseline':<10} {'MRL':<10} {'Delta':<10}")
    print(f"  {'-'*40}")
    print(f"  {'L0':<10} {base_results['l0_accuracy']:<10.4f} "
          f"{mrl_results['l0_accuracy']:<10.4f} {delta_l0:+10.4f}")
    print(f"  {'L1':<10} {base_results['l1_accuracy']:<10.4f} "
          f"{mrl_results['l1_accuracy']:<10.4f} {delta_l1:+10.4f}")

    print(f"\n  MRL vs unfinetuned: L0={delta_l0:+.2%}, L1={delta_l1:+.2%}")
    print("  Compare these numbers against V5 results to measure")
    print("  the effect of hierarchy-alignment.")

    # Save results
    results = {
        'method': 'mrl_baseline',
        'model': model_key,
        'dataset': dataset_name,
        'baseline': base_results,
        'mrl': mrl_results,
        'delta': {'l0': delta_l0, 'l1': delta_l1},
        'prefix_accuracy': prefix_results,
        'history': history,
        'training_config': {
            'prefix_probs': MRLTrainerV5.PREFIX_PROBS,
            'block_keep_probs': MRLTrainerV5.BLOCK_KEEP_PROBS,
            'mrl_weight': MRLTrainerV5.MRL_WEIGHT,
            'margin_weight': MRLTrainerV5.MARGIN_WEIGHT,
            'class_weight': MRLTrainerV5.CLASS_WEIGHT,
            'stage1_epochs': stage1_epochs,
            'stage2_epochs': stage2_epochs,
            'batch_size': batch_size,
            'lr': 1e-4,
            'weight_decay': 0.01,
            'temperature': 0.07,
            'grad_clip': 1.0,
            'note': 'num_l0_classes set to num_l1_classes so head_top outputs L1 logits',
        },
    }

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"mrl_baseline_{model_key}_{dataset_name}.json"
    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MRL Baseline for Fractal V5 comparison"
    )
    parser.add_argument("--model", type=str, default="bge-small")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_mrl_experiment(
        model_key=args.model,
        dataset_name=args.dataset,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
    )
