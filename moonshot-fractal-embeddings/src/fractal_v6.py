"""
Fractal Embeddings V6 - V5 + Conditional Gradient-Reversal Leakage Adversary
==============================================================================

V6 = V5 architecture + train-only adversary that suppresses fine-grained
information (L1) from leaking into the coarse prefix (first 64d block).

Grounded in Theorem 3 (Steerability-Leakage Decomposition):
    S = H(L1|L0) - I(M1; L1|L0) - Delta
Maximizing steerability S requires minimizing I(M1; L1|L0).
The adversary estimates this mutual information and GRL pushes it toward zero.

Key properties:
- Inference-identical to V5 (adversary discarded after training)
- Same head architecture, same backbone, same output dimension
- Only ~30K additional train-time parameters
- Adversary: GRL(M1) concat L0_embed -> MLP(80->128->K1)

Designed by Codex (GPT-5.3, xhigh reasoning), Feb 13 2026.
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
from torch.autograd import Function

from fractal_v5 import (
    FractalHeadV5,
    FractalModelV5,
    V5Trainer,
    ContrastiveDatasetV5,
    collate_fn_v5,
    split_train_val,
)
from multi_model_pipeline import MODELS


# ============================================================================
# Gradient Reversal Layer (Ganin et al., 2016)
# ============================================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ============================================================================
# Leakage Adversary
# ============================================================================

class LeakageAdversary(nn.Module):
    """
    Conditional leakage adversary that predicts L1 from the prefix block M1,
    conditioned on L0. Applied only during training; discarded at inference.

    Input:  GRL(M1) [scale_dim=64] concat L0_embed [l0_embed_dim=16]
    Output: logits [num_l1], masked to valid L1 classes per L0
    """

    def __init__(
        self,
        scale_dim: int = 64,
        num_l0: int = 10,
        num_l1: int = 100,
        l0_embed_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=1.0)
        self.l0_embed = nn.Embedding(num_l0, l0_embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(scale_dim + l0_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_l1),
        )

        # L0 -> L1 compatibility mask (built from training data)
        self.register_buffer(
            "l0_l1_mask",
            torch.zeros(num_l0, num_l1, dtype=torch.bool),
        )
        self._mask_built = False

    def build_hierarchy_mask(self, l0_labels: torch.Tensor, l1_labels: torch.Tensor):
        """Build L0->L1 compatibility mask from training data."""
        if self._mask_built:
            return
        for l0, l1 in zip(l0_labels.cpu().tolist(), l1_labels.cpu().tolist()):
            self.l0_l1_mask[int(l0), int(l1)] = True
        self._mask_built = True

    def forward(
        self,
        m1: torch.Tensor,
        l0_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict L1 from GRL(M1) conditioned on L0.

        Args:
            m1: First block embedding [batch, scale_dim]
            l0_labels: Ground-truth L0 labels [batch]

        Returns:
            L1 logits [batch, num_l1], masked to valid L1s per L0
        """
        m1_rev = self.grl(m1)                    # [batch, scale_dim]
        l0_emb = self.l0_embed(l0_labels)         # [batch, l0_embed_dim]
        x = torch.cat([m1_rev, l0_emb], dim=-1)  # [batch, scale_dim + l0_embed_dim]
        logits = self.mlp(x)                       # [batch, num_l1]

        if self._mask_built:
            mask = self.l0_l1_mask[l0_labels]     # [batch, num_l1]
            logits = logits.masked_fill(~mask, -1e9)

        return logits

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# V6 Trainer
# ============================================================================

class V6Trainer(V5Trainer):
    """
    V6 Trainer = V5 Trainer + leakage adversary.

    Total loss: L_V6 = L_V5 + lambda_adv * CE(adversary, L1)
    where the adversary gets GRL(M1) so its gradient reverses into M1.

    The adversary forces the first block (prefix) to NOT encode fine-grained
    L1 information, improving steerability.
    """

    def __init__(
        self,
        model: FractalModelV5,
        train_dataset,
        val_dataset,
        device: str = "cuda",
        lambda_adv: float = 0.3,
        adv_warmup_epochs: int = 1,
        **kwargs,
    ):
        super().__init__(model, train_dataset, val_dataset, device, **kwargs)
        self.lambda_adv = lambda_adv
        self.adv_warmup_epochs = adv_warmup_epochs

        # Build adversary matching the model's dimensions
        head = model.fractal_head
        self.adversary = LeakageAdversary(
            scale_dim=head.scale_dim,
            num_l0=head.head_top[-1].out_features,   # num_l0
            num_l1=head.head_leaf[-1].out_features,   # num_l1
            l0_embed_dim=16,
            hidden_dim=128,
            dropout=0.1,
        ).to(device)

        print(f"  V6 adversary: {self.adversary.param_count:,} params, "
              f"lambda={lambda_adv}, warmup={adv_warmup_epochs} epochs")

    def _build_hierarchy_mask_from_dataset(self):
        """Build the L0->L1 mask from training data."""
        l0s = torch.tensor([s.level0_label for s in self.train_dataset.samples])
        l1s = torch.tensor([s.level1_label for s in self.train_dataset.samples])
        self.adversary.build_hierarchy_mask(l0s, l1s)

    def train(self, batch_size: int = 32, patience: int = 5) -> List[Dict]:
        from torch.utils.data import DataLoader

        # Build hierarchy mask
        self._build_hierarchy_mask_from_dataset()

        train_ds = ContrastiveDatasetV5(self.train_dataset.samples, self.model.tokenizer)

        def collate(batch):
            return collate_fn_v5(batch, self.model.tokenizer, self.device)

        dataloader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate, drop_last=True
        )

        history = []
        global_best_score = -float('inf')
        global_best_state = None

        total_epochs = self.stage1_epochs + self.stage2_epochs

        # ==================== STAGE 1: Head + Adversary only ====================
        print(f"\n[Stage 1] Training HEAD + ADVERSARY ({self.stage1_epochs} epochs)")
        self.model.freeze_backbone()

        optimizer = torch.optim.AdamW([
            {"params": self.model.fractal_head.parameters(), "lr": self.stage1_lr},
            {"params": self.adversary.parameters(), "lr": self.stage1_lr * 2},
        ], weight_decay=0.01)

        for epoch in range(self.stage1_epochs):
            self.model.train()
            self.adversary.train()

            # Update adversary schedule
            global_epoch = epoch
            self._update_adversary_schedule(global_epoch, total_epochs)

            epoch_loss, epoch_metrics = self._train_epoch_v6(
                dataloader, optimizer, epoch, stage=1
            )

            val_score = self._evaluate()
            history.append({
                'stage': 1, 'epoch': epoch + 1,
                'loss': epoch_loss,
                **epoch_metrics,
                **val_score
            })

            print(f"  Stage 1 Epoch {epoch+1}: loss={epoch_loss:.4f}, "
                  f"L0={val_score['l0_accuracy']:.4f}, L1={val_score['l1_accuracy']:.4f}, "
                  f"adv_loss={epoch_metrics.get('loss_adv', 0):.4f}")

            if val_score['l0_accuracy'] + val_score['l1_accuracy'] > global_best_score:
                global_best_score = val_score['l0_accuracy'] + val_score['l1_accuracy']
                global_best_state = copy.deepcopy(self.model.state_dict())

        # ==================== STAGE 2: Head + Backbone + Adversary ====================
        if self.stage2_epochs > 0:
            print(f"\n[Stage 2] Training HEAD + BACKBONE + ADVERSARY "
                  f"({self.stage2_epochs} epochs)")
            self.model.unfreeze_last_n_layers(self.unfreeze_layers)

            head_params = list(self.model.fractal_head.parameters())
            backbone_params = [p for p in self.model.backbone.parameters()
                             if p.requires_grad]

            optimizer = torch.optim.AdamW([
                {'params': head_params, 'lr': self.stage2_lr_head},
                {'params': backbone_params, 'lr': self.stage2_lr_backbone},
                {'params': self.adversary.parameters(),
                 'lr': self.stage2_lr_head * 2},
            ], weight_decay=0.01)

            no_improve = 0

            for epoch in range(self.stage2_epochs):
                self.model.train()
                self.adversary.train()

                global_epoch = self.stage1_epochs + epoch
                self._update_adversary_schedule(global_epoch, total_epochs)

                epoch_loss, epoch_metrics = self._train_epoch_v6(
                    dataloader, optimizer, epoch, stage=2
                )

                val_score = self._evaluate()
                history.append({
                    'stage': 2, 'epoch': epoch + 1,
                    'loss': epoch_loss,
                    **epoch_metrics,
                    **val_score
                })

                print(f"  Stage 2 Epoch {epoch+1}: loss={epoch_loss:.4f}, "
                      f"L0={val_score['l0_accuracy']:.4f}, "
                      f"L1={val_score['l1_accuracy']:.4f}, "
                      f"adv_loss={epoch_metrics.get('loss_adv', 0):.4f}")

                score = val_score['l0_accuracy'] + val_score['l1_accuracy']
                if score > global_best_score:
                    global_best_score = score
                    global_best_state = copy.deepcopy(self.model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  Early stopping at Stage 2 Epoch {epoch + 1}")
                        break

        # Restore best
        if global_best_state:
            self.model.load_state_dict(global_best_state)
            print(f"\nRestored best model (score={global_best_score:.4f})")

        return history

    def _update_adversary_schedule(self, global_epoch: int, total_epochs: int):
        """Update GRL alpha and effective lambda based on epoch."""
        # DANN-style GRL schedule
        p = global_epoch / max(total_epochs, 1)
        self.adversary.grl.alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        # Lambda warmup
        if global_epoch < self.adv_warmup_epochs:
            self._current_lambda = (
                self.lambda_adv * (global_epoch + 1) / self.adv_warmup_epochs
            )
        else:
            self._current_lambda = self.lambda_adv

    def _train_epoch_v6(
        self, dataloader, optimizer, epoch: int, stage: int
    ) -> Tuple[float, Dict]:
        """V6 training epoch: V5 losses + adversary loss."""
        from torch.cuda.amp import autocast

        total_loss = 0
        total_loss_full = 0
        total_loss_prefix = 0
        total_loss_adv = 0
        num_batches = 0

        use_amp = (stage == 2)
        current_lambda = getattr(self, '_current_lambda', self.lambda_adv)

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = batch['anchor_ids'].shape[0]

            prefix_lengths = self.sample_prefix_length(batch_size).to(self.device)

            with autocast(enabled=use_amp):
                # ===== FULL PATH (same as V5) =====
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
                loss_full = (full_contrastive +
                            self.MARGIN_WEIGHT * full_margin +
                            self.CLASS_WEIGHT * full_cls)

                # ===== PREFIX PATH (same as V5) =====
                prefix_dropout_mask = self.create_block_dropout_mask(
                    batch_size, prefix_lengths
                )
                anchor_prefix = self.model(
                    batch['anchor_ids'], batch['anchor_mask'],
                    block_dropout_mask=prefix_dropout_mask
                )
                l0_pos_prefix = self.model(
                    batch['l0_pos_ids'], batch['l0_pos_mask']
                )

                mode_prefix_len = prefix_lengths.mode().values.item()
                prefix_emb = self.model.fractal_head.get_prefix_embedding(
                    anchor_prefix['blocks'], mode_prefix_len
                )
                l0_pos_emb = l0_pos_prefix['full_embedding']

                prefix_contrastive = self.contrastive_loss(prefix_emb, l0_pos_emb)
                prefix_margin = self.margin_loss(prefix_emb, batch['l0_labels'])
                prefix_cls = F.cross_entropy(
                    self.model.fractal_head.classify_top(prefix_emb),
                    batch['l0_labels']
                )
                loss_prefix = (prefix_contrastive +
                              self.MARGIN_WEIGHT * prefix_margin +
                              self.CLASS_WEIGHT * prefix_cls)

                # ===== V6: ADVERSARY LOSS =====
                # Get first block from the FULL-path forward (no dropout on M1)
                m1 = anchor_full['blocks'][0]  # [batch, scale_dim]
                adv_logits = self.adversary(m1, batch['l0_labels'])
                loss_adv = F.cross_entropy(adv_logits, batch['l1_labels'])

                # ===== TOTAL LOSS =====
                loss = (loss_full +
                        self.PREFIX_WEIGHT * loss_prefix +
                        current_lambda * loss_adv)

            if torch.isnan(loss):
                print(f"    Warning: NaN loss at batch {batch_idx}, skipping")
                continue

            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) +
                    list(self.adversary.parameters()),
                    1.0
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) +
                    list(self.adversary.parameters()),
                    1.0
                )
                optimizer.step()

            total_loss += loss.item()
            total_loss_full += loss_full.item()
            total_loss_prefix += loss_prefix.item()
            total_loss_adv += loss_adv.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {
            'loss_full': total_loss_full / max(num_batches, 1),
            'loss_prefix': total_loss_prefix / max(num_batches, 1),
            'loss_adv': total_loss_adv / max(num_batches, 1),
        }
        return avg_loss, metrics


# ============================================================================
# Experiment runner
# ============================================================================

def run_v6_experiment(
    model_key: str = "bge-small",
    dataset_name: str = "yahoo",
    stage1_epochs: int = 5,
    stage2_epochs: int = 0,
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 42,
    max_train_samples: int = 10000,
    max_test_samples: int = 2000,
    lambda_adv: float = 0.3,
):
    """Run V6 experiment (V5 + adversary)."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from hierarchical_datasets import load_hierarchical_dataset

    print("=" * 70)
    print(f"FRACTAL V6: {model_key} on {dataset_name} (lambda_adv={lambda_adv})")
    print("=" * 70)
    print("V6 = V5 + conditional gradient-reversal leakage adversary")

    # Load data
    print("\n[1] Loading data...")
    train_data = load_hierarchical_dataset(
        dataset_name, split="train", max_samples=max_train_samples
    )
    test_data = load_hierarchical_dataset(
        dataset_name, split="test", max_samples=max_test_samples
    )

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

    # Baseline
    print("\n[2] Baseline...")
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

    # V6 Model (uses V5 architecture)
    print(f"\n[3] Training V6...")
    config = MODELS[model_key]

    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    trainer = V6Trainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        unfreeze_layers=4,
        lambda_adv=lambda_adv,
    )

    history = trainer.train(batch_size=batch_size, patience=5)

    # Final eval
    print("\n[4] Final evaluation on TEST set...")
    model.eval()
    v6_results = evaluate_knn(model, test_data)
    print(f"  L0: {v6_results['l0_accuracy']:.4f}")
    print(f"  L1: {v6_results['l1_accuracy']:.4f}")

    # Prefix accuracy
    print("\n[5] Prefix-length accuracy...")
    test_data_temp = TempDataset(
        test_data.samples, test_data.level0_names, test_data.level1_names
    )
    trainer.val_dataset = test_data_temp
    prefix_results = trainer.evaluate_prefix_accuracy()

    print("  Prefix Length | L0 Acc | L1 Acc")
    print("  " + "-" * 35)
    for j in [1, 2, 3, 4]:
        l0 = prefix_results[f'j{j}_l0']
        l1 = prefix_results[f'j{j}_l1']
        print(f"  j={j} ({j*64}d)    | {l0:.4f} | {l1:.4f}")

    # Steerability
    from external_baselines.common import compute_steer
    steerability = compute_steer(prefix_results)
    print(f"\n  Steerability S = {steerability:+.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    delta_l0 = v6_results['l0_accuracy'] - base_results['l0_accuracy']
    delta_l1 = v6_results['l1_accuracy'] - base_results['l1_accuracy']

    print(f"  {'Metric':<10} {'Baseline':<10} {'V6':<10} {'Delta':<10}")
    print(f"  {'-'*40}")
    print(f"  {'L0':<10} {base_results['l0_accuracy']:<10.4f} "
          f"{v6_results['l0_accuracy']:<10.4f} {delta_l0:+10.4f}")
    print(f"  {'L1':<10} {base_results['l1_accuracy']:<10.4f} "
          f"{v6_results['l1_accuracy']:<10.4f} {delta_l1:+10.4f}")

    # Save results
    results = {
        'model': model_key,
        'dataset': dataset_name,
        'method': 'v6',
        'lambda_adv': lambda_adv,
        'baseline': base_results,
        'v6': v6_results,
        'delta': {'l0': delta_l0, 'l1': delta_l1},
        'prefix_accuracy': prefix_results,
        'steerability': steerability,
        'history': history,
        'training_config': {
            'prefix_probs': V5Trainer.PREFIX_PROBS,
            'block_keep_probs': V5Trainer.BLOCK_KEEP_PROBS,
            'prefix_weight': V5Trainer.PREFIX_WEIGHT,
            'lambda_adv': lambda_adv,
        }
    }

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    results_path = (Path(__file__).parent.parent / "results" /
                    f"v6_{model_key}_{dataset_name}.json")
    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bge-small")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lambda-adv", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_v6_experiment(
        model_key=args.model,
        dataset_name=args.dataset,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        lambda_adv=args.lambda_adv,
        seed=args.seed,
    )
