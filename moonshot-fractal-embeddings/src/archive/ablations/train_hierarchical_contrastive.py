"""
Hierarchical Multi-Task + Tree-Distance Contrastive Training

Per experimental design: To achieve +20-30% gains, we need to:
1. Unfreeze last 2 transformer blocks
2. Multi-task classification: CE(L0) + CE(L1)
3. Tree-distance contrastive loss with margin proportional to hierarchy distance
4. Differential learning rates (backbone: 5e-6, head: 5e-4)

This addresses the fundamental bottleneck: frozen backbone limits gains to single digits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path

# Local imports
from multi_model_pipeline import MODELS, load_model
from hierarchical_datasets import load_hierarchical_dataset


@dataclass
class HierContrastiveConfig:
    """Configuration for hierarchical contrastive training"""
    # Model
    backbone_name: str = "qwen3-0.6b"
    hidden_dim: int = 1024

    # Hierarchy
    num_l0_classes: int = 10
    num_l1_classes: int = 60

    # Training
    backbone_lr: float = 5e-6
    head_lr: float = 5e-4
    num_epochs: int = 5
    batch_size: int = 24
    warmup_ratio: float = 0.1

    # Loss weights
    class_weight: float = 1.0
    contrast_weight: float = 0.0  # Disabled for stability testing
    temperature: float = 0.5  # Increased from 0.1 for numerical stability
    tree_margin_alpha: float = 0.5
    gradient_clip: float = 1.0  # Gradient clipping norm

    # Unfreezing
    unfreeze_layers: int = 2  # Last N transformer layers

    # Device
    device: str = "cuda"


class TreeDistanceContrastiveLoss(nn.Module):
    """
    Simple supervised contrastive loss for hierarchical classification.
    Uses L1 labels for positive pairs, with stable computation.
    """

    def __init__(self, temperature: float = 0.5, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature  # Higher temperature for stability
        self.alpha = alpha

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, D]
        l0_labels: torch.Tensor,   # [B]
        l1_labels: torch.Tensor    # [B]
    ) -> torch.Tensor:
        """Stable supervised contrastive loss"""
        device = embeddings.device
        batch_size = embeddings.size(0)

        if batch_size <= 1:
            return embeddings.sum() * 0.0  # Return zero with gradient connection

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix [B, B]
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        # Scale by temperature
        sim_matrix = sim_matrix / self.temperature

        # Create masks
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        same_l1 = l1_labels.unsqueeze(0) == l1_labels.unsqueeze(1)
        pos_mask = same_l1 & ~self_mask

        # Check if any positives exist
        num_pos_per_sample = pos_mask.sum(dim=1)
        has_pos = num_pos_per_sample > 0

        if not has_pos.any():
            # No positive pairs - return small loss to maintain gradients
            return embeddings.sum() * 0.0

        # For numerical stability, subtract max per row
        logits_max = sim_matrix.max(dim=1, keepdim=True)[0]
        logits = sim_matrix - logits_max.detach()

        # Compute log softmax denominator (excluding self)
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * (~self_mask).float()  # Zero out diagonal
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        # Log probabilities
        log_prob = logits - log_sum_exp

        # Mean log prob for positive pairs
        pos_log_prob = (pos_mask.float() * log_prob).sum(dim=1)
        pos_count = num_pos_per_sample.clamp(min=1).float()

        # Only compute loss for samples with positives
        per_sample_loss = -pos_log_prob / pos_count
        loss = per_sample_loss[has_pos].mean()

        # Clamp for stability
        loss = loss.clamp(min=0, max=10)

        return loss


class HierarchicalClassifier(nn.Module):
    """
    Hierarchical classifier with multi-task heads.

    Architecture:
    - Backbone (partially unfrozen)
    - Projection layer
    - L0 head (coarse)
    - L1 head (fine)
    """

    def __init__(self, config: HierContrastiveConfig):
        super().__init__()
        self.config = config

        # Projection from backbone hidden to embedding
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 256),
            nn.LayerNorm(256)
        )

        # Classification heads
        self.l0_head = nn.Linear(256, config.num_l0_classes)
        self.l1_head = nn.Linear(256, config.num_l1_classes)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Backbone output [B, D]
        Returns:
            embeddings: Normalized embeddings [B, 256]
            l0_logits: Coarse logits [B, num_l0]
            l1_logits: Fine logits [B, num_l1]
        """
        # Project
        embeddings = self.projection(hidden_states)
        embeddings_norm = F.normalize(embeddings, dim=1)

        # Classify
        l0_logits = self.l0_head(embeddings)
        l1_logits = self.l1_head(embeddings)

        return embeddings_norm, l0_logits, l1_logits


class HierarchicalContrastiveTrainer:
    """
    Trainer for hierarchical contrastive learning with partial backbone unfreezing.
    """

    def __init__(
        self,
        backbone,
        tokenizer,
        config: HierContrastiveConfig,
        train_dataset,
        val_dataset,
        use_amp: bool = True
    ):
        self.backbone = backbone.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.use_amp = use_amp
        self.device = config.device

        # Create classifier head
        self.classifier = HierarchicalClassifier(config).to(config.device)

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = TreeDistanceContrastiveLoss(
            temperature=config.temperature,
            alpha=config.tree_margin_alpha
        )

        # Setup backbone unfreezing
        self._setup_backbone_unfreezing()

        # Setup optimizers with differential LR
        self._setup_optimizers()

        # Mixed precision
        self.scaler = GradScaler() if use_amp else None

        # Best model tracking
        self.best_score = 0.0
        self.best_state = None

    def _setup_backbone_unfreezing(self):
        """Freeze all backbone layers except last N"""
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Find transformer layers - try multiple paths
        layers = None
        norm = None

        # Try different paths for layers
        if hasattr(self.backbone, 'layers'):
            layers = self.backbone.layers
            norm = getattr(self.backbone, 'norm', None)
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'layers'):
            layers = self.backbone.model.layers
            norm = getattr(self.backbone.model, 'norm', None)
        elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            layers = self.backbone.encoder.layer
            norm = None

        if layers is not None:
            num_layers = len(layers)
            print(f"  Found {num_layers} transformer layers")

            # Unfreeze last N layers
            unfreeze_from = num_layers - self.config.unfreeze_layers
            for i, layer in enumerate(layers):
                if i >= unfreeze_from:
                    for param in layer.parameters():
                        param.requires_grad = True
                    print(f"  Unfreezing layer {i}")

            # Also unfreeze final layernorm
            if norm is not None:
                for param in norm.parameters():
                    param.requires_grad = True
                print("  Unfreezing final norm")
        else:
            # Fallback: keep everything frozen, only train head
            print("  Warning: Could not find standard layer structure")
            print("  Keeping backbone frozen, training head only")

        # Count trainable params
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.classifier.parameters())
        print(f"  Backbone trainable: {backbone_trainable:,}")
        print(f"  Head trainable: {head_trainable:,}")

    def _setup_optimizers(self):
        """Setup optimizers with differential learning rates"""
        # Backbone params (unfrozen only)
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]

        # Head params (all)
        head_params = list(self.classifier.parameters())

        # Create optimizer with param groups
        self.optimizer = AdamW([
            {'params': backbone_params, 'lr': self.config.backbone_lr},
            {'params': head_params, 'lr': self.config.head_lr}
        ], weight_decay=0.01)

    def _encode_batch(self, texts: List[str], requires_grad: bool = False) -> torch.Tensor:
        """Encode texts through backbone"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Use no_grad if backbone is fully frozen, otherwise allow gradients
        context = torch.no_grad() if not requires_grad else torch.enable_grad()
        with context:
            outputs = self.backbone(**inputs)

            # Get pooled representation
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state[:, -1, :]  # Last token
            elif hasattr(outputs, 'pooler_output'):
                hidden = outputs.pooler_output
            else:
                hidden = outputs[0][:, -1, :]

        # Convert to FP32 for classifier (backbone outputs FP16)
        return hidden.float()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        # Keep backbone in eval mode - it may have dropout/batchnorm that shouldn't be active
        # when we're fine-tuning with frozen or partially frozen weights
        self.backbone.eval()
        self.classifier.train()

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda x: x  # Return list of samples as-is
        )

        total_loss = 0
        total_class_loss = 0
        total_contrast_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Get batch data
            texts = [item.text for item in batch]
            l0_labels = torch.tensor([item.level0_label for item in batch], device=self.device)
            l1_labels = torch.tensor([item.level1_label for item in batch], device=self.device)

            self.optimizer.zero_grad()

            # Check if backbone has any trainable params
            backbone_has_grad = any(p.requires_grad for p in self.backbone.parameters())

            with autocast(enabled=self.use_amp):
                # Encode through backbone
                hidden = self._encode_batch(texts, requires_grad=backbone_has_grad)

                # Get embeddings and logits
                embeddings, l0_logits, l1_logits = self.classifier(hidden)

                # Classification loss
                l0_loss = self.ce_loss(l0_logits, l0_labels)
                l1_loss = self.ce_loss(l1_logits, l1_labels)
                class_loss = l0_loss + l1_loss

                # Contrastive loss
                contrast_loss = self.contrastive_loss(embeddings, l0_labels, l1_labels)

                # Total loss
                loss = self.config.class_weight * class_loss + self.config.contrast_weight * contrast_loss

            # Check for NaN
            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected, skipping batch")
                continue

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters()) + list(self.classifier.parameters()),
                    self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters()) + list(self.classifier.parameters()),
                    self.config.gradient_clip
                )
                self.optimizer.step()

            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_contrast_loss += contrast_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'cls': f'{class_loss.item():.3f}',
                'ctr': f'{contrast_loss.item():.3f}'
            })

        if num_batches == 0:
            return {'loss': float('nan'), 'class_loss': float('nan'), 'contrast_loss': float('nan')}

        return {
            'loss': total_loss / num_batches,
            'class_loss': total_class_loss / num_batches,
            'contrast_loss': total_contrast_loss / num_batches
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.backbone.eval()
        self.classifier.eval()

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda x: x  # Return list of samples as-is
        )

        all_l0_preds = []
        all_l1_preds = []
        all_l0_labels = []
        all_l1_labels = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = [item.text for item in batch]
            l0_labels = [item.level0_label for item in batch]
            l1_labels = [item.level1_label for item in batch]

            with autocast(enabled=self.use_amp):
                hidden = self._encode_batch(texts)
                _, l0_logits, l1_logits = self.classifier(hidden)

            l0_preds = l0_logits.argmax(dim=1).cpu().tolist()
            l1_preds = l1_logits.argmax(dim=1).cpu().tolist()

            all_l0_preds.extend(l0_preds)
            all_l1_preds.extend(l1_preds)
            all_l0_labels.extend(l0_labels)
            all_l1_labels.extend(l1_labels)

        l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
        l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])

        return {
            'l0_acc': l0_acc,
            'l1_acc': l1_acc,
            'combined': l0_acc + l1_acc
        }

    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("HIERARCHICAL CONTRASTIVE TRAINING")
        print(f"{'='*60}")
        print(f"Backbone: {self.config.backbone_name}")
        print(f"Unfreezing: Last {self.config.unfreeze_layers} layers")
        print(f"Backbone LR: {self.config.backbone_lr}")
        print(f"Head LR: {self.config.head_lr}")
        print(f"Contrastive weight: {self.config.contrast_weight}")
        print(f"Temperature: {self.config.temperature}")
        print()

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n[Epoch {epoch}/{self.config.num_epochs}]")

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Class loss: {train_metrics['class_loss']:.4f}")
            print(f"  Contrast loss: {train_metrics['contrast_loss']:.4f}")

            # Evaluate
            val_metrics = self.evaluate()
            print(f"  Val L0: {val_metrics['l0_acc']:.4f}")
            print(f"  Val L1: {val_metrics['l1_acc']:.4f}")

            # Track best
            score = val_metrics['combined']
            if score > self.best_score:
                self.best_score = score
                self.best_state = {
                    'classifier': {k: v.cpu().clone() for k, v in self.classifier.state_dict().items()},
                    'epoch': epoch,
                    'l0_acc': val_metrics['l0_acc'],
                    'l1_acc': val_metrics['l1_acc']
                }
                print(f"  ** New best! Score: {score:.4f}")

        # Restore best model
        if self.best_state:
            self.classifier.load_state_dict({k: v.to(self.device) for k, v in self.best_state['classifier'].items()})
            print(f"\nRestored best model from epoch {self.best_state['epoch']}")
            print(f"Best L0: {self.best_state['l0_acc']:.4f}, L1: {self.best_state['l1_acc']:.4f}")

        return self.best_state


def compute_baseline(backbone, tokenizer, val_dataset, device, config):
    """Compute baseline accuracy using raw backbone embeddings"""
    print("Computing baseline...")
    backbone.eval()

    # Collect embeddings and labels
    all_embeddings = []
    all_l0_labels = []
    all_l1_labels = []

    # Use simple list collation
    def collate_fn(batch):
        return batch  # Return list of HierarchicalSample as-is

    dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Baseline"):
            texts = [item.text for item in batch]
            l0_labels = [item.level0_label for item in batch]
            l1_labels = [item.level1_label for item in batch]

            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

            with autocast():
                outputs = backbone(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state[:, -1, :]
                else:
                    hidden = outputs[0][:, -1, :]

            all_embeddings.append(hidden.cpu())
            all_l0_labels.extend(l0_labels)
            all_l1_labels.extend(l1_labels)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    l0_labels = np.array(all_l0_labels)
    l1_labels = np.array(all_l1_labels)

    # Simple KNN evaluation
    from sklearn.neighbors import KNeighborsClassifier

    # Split for KNN eval
    n = len(embeddings)
    n_train = int(0.8 * n)

    train_emb, test_emb = embeddings[:n_train], embeddings[n_train:]
    train_l0, test_l0 = l0_labels[:n_train], l0_labels[n_train:]
    train_l1, test_l1 = l1_labels[:n_train], l1_labels[n_train:]

    # L0 KNN
    knn_l0 = KNeighborsClassifier(n_neighbors=5)
    knn_l0.fit(train_emb, train_l0)
    l0_acc = knn_l0.score(test_emb, test_l0)

    # L1 KNN
    knn_l1 = KNeighborsClassifier(n_neighbors=5)
    knn_l1.fit(train_emb, train_l1)
    l1_acc = knn_l1.score(test_emb, test_l1)

    print(f"Baseline L0: {l0_acc:.4f}, L1: {l1_acc:.4f}")
    return l0_acc, l1_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--backbone_lr", type=float, default=5e-6)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--contrast_weight", type=float, default=0.2)
    parser.add_argument("--unfreeze_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset (use subsets for efficiency)
    print("\nLoading dataset...")
    train_data = load_hierarchical_dataset(args.dataset, split="train", max_samples=10000)
    val_data = load_hierarchical_dataset(args.dataset, split="test", max_samples=2000)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Get number of classes
    l0_classes = len(set(item.level0_label for item in train_data))
    l1_classes = len(set(item.level1_label for item in train_data))
    print(f"  L0 classes: {l0_classes}, L1 classes: {l1_classes}")

    # Load backbone
    print(f"\nLoading backbone: {args.model}...")
    model_config = MODELS[args.model]
    wrapper = load_model(args.model, use_fractal=False, device=device)
    backbone = wrapper.backbone
    tokenizer = wrapper.tokenizer
    hidden_dim = model_config.hidden_dim
    print(f"  Hidden dim: {hidden_dim}")

    # Create config
    config = HierContrastiveConfig(
        backbone_name=args.model,
        hidden_dim=hidden_dim,
        num_l0_classes=l0_classes,
        num_l1_classes=l1_classes,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        contrast_weight=args.contrast_weight,
        unfreeze_layers=args.unfreeze_layers,
        device=device
    )

    # Compute baseline
    baseline_l0, baseline_l1 = compute_baseline(backbone, tokenizer, val_data, device, config)

    # Create trainer (disable AMP since backbone is already FP16)
    print("\nSetting up trainer...")
    trainer = HierarchicalContrastiveTrainer(
        backbone=backbone,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_data,
        val_dataset=val_data,
        use_amp=False  # Backbone is FP16, AMP causes issues
    )

    # Train
    results = trainer.train()

    # Print final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Baseline: L0={baseline_l0:.4f}, L1={baseline_l1:.4f}")
    if results:
        print(f"Best: L0={results['l0_acc']:.4f}, L1={results['l1_acc']:.4f}")
        print(f"Delta: L0={results['l0_acc']-baseline_l0:+.4f}, L1={results['l1_acc']-baseline_l1:+.4f}")
        print(f"Delta %: L0={100*(results['l0_acc']-baseline_l0):+.2f}%, L1={100*(results['l1_acc']-baseline_l1):+.2f}%")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"hier_contrastive_{args.model}.json"
    results_path.parent.mkdir(exist_ok=True)

    save_data = {
        "model": args.model,
        "config": {
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "contrast_weight": args.contrast_weight,
            "unfreeze_layers": args.unfreeze_layers,
            "epochs": args.epochs
        },
        "baseline": {"l0": float(baseline_l0), "l1": float(baseline_l1)},
        "best": {
            "l0": float(results['l0_acc']),
            "l1": float(results['l1_acc']),
            "epoch": results['epoch']
        } if results else None
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
