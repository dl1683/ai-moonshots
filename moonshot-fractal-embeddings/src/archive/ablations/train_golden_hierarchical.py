"""
Golden Ratio Hierarchical Classification

Uses φ (golden ratio ≈ 1.618) to naturally balance:
- Loss weights between hierarchy levels
- Architecture dimensions
- Learning rate decay

Mathematical intuition: The golden ratio creates optimal balance between
competing objectives, appearing in natural hierarchies and fractal structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import math

# Local imports
from multi_model_pipeline import MODELS, load_model
from hierarchical_datasets import load_hierarchical_dataset

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618
PHI_INV = 1 / PHI  # ≈ 0.618


@dataclass
class GoldenConfig:
    """Configuration using golden ratio principles"""
    backbone_name: str = "qwen3-0.6b"
    hidden_dim: int = 1024
    num_l0_classes: int = 10
    num_l1_classes: int = 60

    # Training
    head_lr: float = 5e-4
    num_epochs: int = 10
    batch_size: int = 24

    # Golden ratio loss weights
    l0_weight: float = 1.0  # Coarse level
    l1_weight: float = PHI_INV  # ≈ 0.618 - Fine level gets less weight initially

    # Gradient clipping
    gradient_clip: float = 1.0

    device: str = "cuda"


class GoldenHierarchicalClassifier(nn.Module):
    """
    Hierarchical classifier with golden-ratio dimensions.

    Layer sizes follow Fibonacci-like pattern:
    1024 → 633 → 391 → 256 (each ÷ φ, rounded)
    """

    def __init__(self, config: GoldenConfig):
        super().__init__()
        self.config = config

        # Golden ratio dimension reduction
        d0 = config.hidden_dim  # 1024
        d1 = int(d0 / PHI)      # 633
        d2 = int(d1 / PHI)      # 391
        d3 = int(d2 / PHI)      # 241

        print(f"  Golden dimensions: {d0} -> {d1} -> {d2} -> {d3}")

        # Projection with golden-ratio dimensions
        self.projection = nn.Sequential(
            nn.Linear(d0, d1),
            nn.GELU(),
            nn.LayerNorm(d1),
            nn.Linear(d1, d2),
            nn.GELU(),
            nn.LayerNorm(d2),
        )

        # Classification heads
        self.l0_head = nn.Linear(d2, config.num_l0_classes)
        self.l1_head = nn.Linear(d2, config.num_l1_classes)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Backbone output [B, D]
        Returns:
            l0_logits: Coarse logits [B, num_l0]
            l1_logits: Fine logits [B, num_l1]
        """
        features = self.projection(hidden_states)
        l0_logits = self.l0_head(features)
        l1_logits = self.l1_head(features)
        return l0_logits, l1_logits


class GoldenLRScheduler:
    """
    Learning rate scheduler based on golden ratio.

    Decays LR by φ^(-1) at Fibonacci-spaced intervals.
    """

    def __init__(self, optimizer, base_lr: float, total_steps: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.current_step = 0

        # Generate Fibonacci checkpoints
        self.checkpoints = self._fibonacci_checkpoints(total_steps)
        self.decay_count = 0

    def _fibonacci_checkpoints(self, max_steps: int) -> List[int]:
        """Generate Fibonacci sequence up to max_steps"""
        fibs = [1, 2]
        while fibs[-1] < max_steps:
            fibs.append(fibs[-1] + fibs[-2])
        # Normalize to fit within total steps
        scale = max_steps / fibs[-1]
        return [int(f * scale) for f in fibs[:-1]]

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        # Check if we hit a Fibonacci checkpoint
        if self.checkpoints and self.current_step >= self.checkpoints[0]:
            self.checkpoints.pop(0)
            self.decay_count += 1

            # Decay by golden ratio inverse
            new_lr = self.base_lr * (PHI_INV ** self.decay_count)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class GoldenHierarchicalTrainer:
    """Trainer using golden ratio principles"""

    def __init__(
        self,
        backbone,
        tokenizer,
        config: GoldenConfig,
        train_dataset,
        val_dataset
    ):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config.device

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Create classifier with golden dimensions
        self.classifier = GoldenHierarchicalClassifier(config).to(self.device)

        # Loss with golden ratio weights
        self.ce_loss = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.classifier.parameters(),
            lr=config.head_lr,
            weight_decay=0.01
        )

        # Count params
        trainable = sum(p.numel() for p in self.classifier.parameters())
        print(f"  Trainable params: {trainable:,}")

        # Best tracking
        self.best_score = 0.0
        self.best_state = None

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode texts through frozen backbone"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.backbone(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state[:, -1, :]
            elif hasattr(outputs, 'pooler_output'):
                hidden = outputs.pooler_output
            else:
                hidden = outputs[0][:, -1, :]

        return hidden.float()

    def train_epoch(self, epoch: int, scheduler) -> Dict[str, float]:
        """Train for one epoch"""
        self.classifier.train()

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda x: x
        )

        total_loss = 0
        total_l0_loss = 0
        total_l1_loss = 0
        num_batches = 0

        # Golden ratio weight adjustment over training
        # Start favoring L0, gradually shift to L1
        progress = epoch / self.config.num_epochs
        l0_weight = self.config.l0_weight * (1 - progress * PHI_INV)
        l1_weight = self.config.l1_weight * (1 + progress * PHI_INV)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            texts = [item.text for item in batch]
            l0_labels = torch.tensor([item.level0_label for item in batch], device=self.device)
            l1_labels = torch.tensor([item.level1_label for item in batch], device=self.device)

            self.optimizer.zero_grad()

            hidden = self._encode_batch(texts)
            l0_logits, l1_logits = self.classifier(hidden)

            # Golden ratio weighted loss
            l0_loss = self.ce_loss(l0_logits, l0_labels)
            l1_loss = self.ce_loss(l1_logits, l1_labels)
            loss = l0_weight * l0_loss + l1_weight * l1_loss

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.classifier.parameters(),
                self.config.gradient_clip
            )
            self.optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_l0_loss += l0_loss.item()
            total_l1_loss += l1_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'lr': f'{scheduler.get_lr():.2e}'
            })

        if num_batches == 0:
            return {'loss': float('nan'), 'l0_loss': float('nan'), 'l1_loss': float('nan')}

        return {
            'loss': total_loss / num_batches,
            'l0_loss': total_l0_loss / num_batches,
            'l1_loss': total_l1_loss / num_batches
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.classifier.eval()

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )

        all_l0_preds, all_l1_preds = [], []
        all_l0_labels, all_l1_labels = [], []

        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = [item.text for item in batch]
            l0_labels = [item.level0_label for item in batch]
            l1_labels = [item.level1_label for item in batch]

            hidden = self._encode_batch(texts)
            l0_logits, l1_logits = self.classifier(hidden)

            all_l0_preds.extend(l0_logits.argmax(dim=1).cpu().tolist())
            all_l1_preds.extend(l1_logits.argmax(dim=1).cpu().tolist())
            all_l0_labels.extend(l0_labels)
            all_l1_labels.extend(l1_labels)

        l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
        l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])

        return {'l0_acc': l0_acc, 'l1_acc': l1_acc}

    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("GOLDEN RATIO HIERARCHICAL TRAINING")
        print(f"{'='*60}")
        print(f"phi (golden ratio): {PHI:.6f}")
        print(f"L0 base weight: {self.config.l0_weight:.3f}")
        print(f"L1 base weight: {self.config.l1_weight:.3f} (= 1/phi)")
        print(f"LR decay factor: {PHI_INV:.3f} (= 1/phi)")

        total_steps = len(self.train_dataset) // self.config.batch_size * self.config.num_epochs
        scheduler = GoldenLRScheduler(self.optimizer, self.config.head_lr, total_steps)

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n[Epoch {epoch}/{self.config.num_epochs}]")

            train_metrics = self.train_epoch(epoch, scheduler)
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  L0 loss: {train_metrics['l0_loss']:.4f}, L1 loss: {train_metrics['l1_loss']:.4f}")

            val_metrics = self.evaluate()
            print(f"  Val L0: {val_metrics['l0_acc']:.4f}")
            print(f"  Val L1: {val_metrics['l1_acc']:.4f}")

            score = val_metrics['l0_acc'] + val_metrics['l1_acc']
            if score > self.best_score:
                self.best_score = score
                self.best_state = {
                    'classifier': {k: v.cpu().clone() for k, v in self.classifier.state_dict().items()},
                    'epoch': epoch,
                    'l0_acc': val_metrics['l0_acc'],
                    'l1_acc': val_metrics['l1_acc']
                }
                print(f"  ** New best! Combined: {score:.4f}")

        if self.best_state:
            self.classifier.load_state_dict({k: v.to(self.device) for k, v in self.best_state['classifier'].items()})
            print(f"\nRestored best model from epoch {self.best_state['epoch']}")

        return self.best_state


def compute_baseline(backbone, tokenizer, val_dataset, device, config):
    """Compute baseline with kNN"""
    from sklearn.neighbors import KNeighborsClassifier

    print("Computing baseline...")
    backbone.eval()

    all_embeddings = []
    all_l0_labels = []
    all_l1_labels = []

    dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=lambda x: x)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Baseline"):
            texts = [item.text for item in batch]
            l0_labels = [item.level0_label for item in batch]
            l1_labels = [item.level1_label for item in batch]

            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = backbone(**inputs)

            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state[:, -1, :]
            else:
                hidden = outputs[0][:, -1, :]

            all_embeddings.append(hidden.cpu().float())
            all_l0_labels.extend(l0_labels)
            all_l1_labels.extend(l1_labels)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()

    # Use half for "training", half for "testing"
    mid = len(embeddings) // 2

    knn_l0 = KNeighborsClassifier(n_neighbors=5)
    knn_l0.fit(embeddings[:mid], all_l0_labels[:mid])
    l0_acc = knn_l0.score(embeddings[mid:], all_l0_labels[mid:])

    knn_l1 = KNeighborsClassifier(n_neighbors=5)
    knn_l1.fit(embeddings[:mid], all_l1_labels[:mid])
    l1_acc = knn_l1.score(embeddings[mid:], all_l1_labels[mid:])

    print(f"Baseline L0: {l0_acc:.4f}, L1: {l1_acc:.4f}")
    return l0_acc, l1_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\nLoading dataset...")
    train_data = load_hierarchical_dataset(args.dataset, split="train", max_samples=10000)
    val_data = load_hierarchical_dataset(args.dataset, split="test", max_samples=2000)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    l0_classes = len(set(item.level0_label for item in train_data))
    l1_classes = len(set(item.level1_label for item in train_data))
    print(f"  L0 classes: {l0_classes}, L1 classes: {l1_classes}")

    # Load backbone
    print(f"\nLoading backbone: {args.model}...")
    model_config = MODELS[args.model]
    wrapper = load_model(args.model, use_fractal=False, device=device)
    backbone = wrapper.backbone
    tokenizer = wrapper.tokenizer

    # Config
    config = GoldenConfig(
        backbone_name=args.model,
        hidden_dim=model_config.hidden_dim,
        num_l0_classes=l0_classes,
        num_l1_classes=l1_classes,
        head_lr=args.head_lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device
    )

    # Baseline
    baseline_l0, baseline_l1 = compute_baseline(backbone, tokenizer, val_data, device, config)

    # Train
    print("\nSetting up trainer...")
    trainer = GoldenHierarchicalTrainer(
        backbone=backbone,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_data,
        val_dataset=val_data
    )

    results = trainer.train()

    # Results
    print(f"\n{'='*60}")
    print("FINAL RESULTS (GOLDEN RATIO)")
    print(f"{'='*60}")
    print(f"Baseline: L0={baseline_l0:.4f}, L1={baseline_l1:.4f}")
    if results:
        print(f"Best: L0={results['l0_acc']:.4f}, L1={results['l1_acc']:.4f}")
        delta_l0 = results['l0_acc'] - baseline_l0
        delta_l1 = results['l1_acc'] - baseline_l1
        print(f"Delta: L0={delta_l0:+.4f}, L1={delta_l1:+.4f}")
        print(f"Delta %: L0={100*delta_l0:+.2f}%, L1={100*delta_l1:+.2f}%")

    # Save
    results_path = Path(__file__).parent.parent / "results" / f"golden_hier_{args.model}.json"
    results_path.parent.mkdir(exist_ok=True)

    save_data = {
        "model": args.model,
        "golden_ratio": PHI,
        "config": {
            "l0_weight": config.l0_weight,
            "l1_weight": config.l1_weight,
            "head_lr": args.head_lr,
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
