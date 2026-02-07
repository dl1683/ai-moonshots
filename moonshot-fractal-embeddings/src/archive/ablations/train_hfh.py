"""
Training script for Hyperbolic Fractal Head (HFH) - V6

Targets: +20-30% improvement over baseline on hierarchical classification
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Local imports
from hyperbolic_fractal_head import HyperbolicFractalHead, HFHConfig, HyperbolicFractalModel
from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class HFHTrainer:
    """Trainer for Hyperbolic Fractal Head."""

    def __init__(
        self,
        model: HyperbolicFractalModel,
        train_dataset,
        val_dataset,
        tokenizer,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        epochs: int = 10,
        batch_size: int = 24,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_amp = use_amp

        # Only optimize HFH parameters (backbone frozen)
        self.optimizer = torch.optim.AdamW(
            model.hfh.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
        )

        # Mixed precision
        self.scaler = GradScaler() if use_amp else None

        # Best model tracking
        self.best_score = 0.0
        self.best_state = None

    def collate_fn(self, batch):
        """Collate batch with tokenization."""
        texts = [item.text for item in batch]
        l0_labels = torch.tensor([item.level0_label for item in batch])
        l1_labels = torch.tensor([item.level1_label for item in batch])

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "l0_labels": l0_labels,
            "l1_labels": l1_labels,
        }

    def train_epoch(self, dataloader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_l0_loss = 0.0
        total_l1_loss = 0.0
        total_corr_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            l0_labels = batch["l0_labels"].to(self.device)
            l1_labels = batch["l1_labels"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(input_ids, attention_mask)

                # Compute loss
                losses = self.model.hfh.compute_loss(
                    outputs, l0_labels, l1_labels,
                    embeddings_for_corr_dim=outputs.get("embeddings", [None])[0]
                )

            # Backward pass
            if self.use_amp:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.hfh.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.hfh.parameters(), 1.0)
                self.optimizer.step()

            # Track losses
            total_loss += losses["total"].item()
            total_l0_loss += losses["l0"].item()
            total_l1_loss += losses["l1"].item()
            total_corr_loss += losses["corr_dim"].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.3f}",
                "l0": f"{losses['l0'].item():.3f}",
                "l1": f"{losses['l1'].item():.3f}",
            })

        return {
            "loss": total_loss / num_batches,
            "l0_loss": total_l0_loss / num_batches,
            "l1_loss": total_l1_loss / num_batches,
            "corr_loss": total_corr_loss / num_batches,
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """Evaluate model on validation set."""
        self.model.eval()

        all_l0_preds = []
        all_l1_preds = []
        all_l0_labels = []
        all_l1_labels = []
        all_embeddings = []
        curvatures = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            l0_labels = batch["l0_labels"]
            l1_labels = batch["l1_labels"]

            with autocast(enabled=self.use_amp):
                outputs = self.model(input_ids, attention_mask, return_embeddings=True)

            # Predictions
            l0_preds = outputs["l0_logits"].argmax(dim=-1).cpu()
            l1_preds = outputs["l1_logits"].argmax(dim=-1).cpu()

            all_l0_preds.append(l0_preds)
            all_l1_preds.append(l1_preds)
            all_l0_labels.append(l0_labels)
            all_l1_labels.append(l1_labels)

            # Store curvatures (from last batch)
            curvatures = outputs["scale_curvatures"]

            # Collect embeddings for analysis
            if outputs.get("embeddings"):
                # Concatenate all scale embeddings
                batch_emb = []
                for emb in outputs["embeddings"]:
                    if emb.dim() == 2:
                        batch_emb.append(emb.cpu())
                    else:
                        batch_emb.append(emb[..., 1:].cpu())  # Remove time for Lorentz
                all_embeddings.append(torch.cat(batch_emb, dim=-1))

        # Concatenate all
        all_l0_preds = torch.cat(all_l0_preds)
        all_l1_preds = torch.cat(all_l1_preds)
        all_l0_labels = torch.cat(all_l0_labels)
        all_l1_labels = torch.cat(all_l1_labels)

        # Accuracy
        l0_acc = (all_l0_preds == all_l0_labels).float().mean().item()
        l1_acc = (all_l1_preds == all_l1_labels).float().mean().item()

        # KNN evaluation (more robust)
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

            # Split for KNN
            n_train = int(len(all_embeddings) * 0.8)
            train_emb = all_embeddings[:n_train]
            test_emb = all_embeddings[n_train:]
            train_l0 = all_l0_labels[:n_train].numpy()
            test_l0 = all_l0_labels[n_train:].numpy()
            train_l1 = all_l1_labels[:n_train].numpy()
            test_l1 = all_l1_labels[n_train:].numpy()

            if len(test_emb) > 0:
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(train_emb, train_l0)
                l0_knn = accuracy_score(test_l0, knn.predict(test_emb))

                knn.fit(train_emb, train_l1)
                l1_knn = accuracy_score(test_l1, knn.predict(test_emb))
            else:
                l0_knn = l0_acc
                l1_knn = l1_acc
        else:
            l0_knn = l0_acc
            l1_knn = l1_acc

        return {
            "l0_acc": l0_acc,
            "l1_acc": l1_acc,
            "l0_knn": l0_knn,
            "l1_knn": l1_knn,
            "curvatures": curvatures,
        }

    def train(self) -> dict:
        """Full training loop."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

        history = []

        for epoch in range(self.epochs):
            print(f"\n[Epoch {epoch+1}/{self.epochs}]")

            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"  Train loss: {train_metrics['loss']:.4f}")

            # Evaluate
            val_metrics = self.evaluate(val_loader)
            print(f"  Val L0 acc: {val_metrics['l0_acc']:.4f}, L1 acc: {val_metrics['l1_acc']:.4f}")
            print(f"  Val L0 KNN: {val_metrics['l0_knn']:.4f}, L1 KNN: {val_metrics['l1_knn']:.4f}")
            print(f"  Curvatures: {[f'{c:.3f}' for c in val_metrics['curvatures']]}")

            # Learning rate step
            self.scheduler.step()

            # Track best model
            score = val_metrics["l0_acc"] + val_metrics["l1_acc"]
            if score > self.best_score:
                self.best_score = score
                self.best_state = {k: v.cpu().clone() for k, v in self.model.hfh.state_dict().items()}
                print(f"  New best! Score: {score:.4f}")

            history.append({
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
            })

        # Restore best model
        if self.best_state is not None:
            self.model.hfh.load_state_dict(self.best_state)
            print(f"\nRestored best model (score={self.best_score:.4f})")

        return history


def compute_baseline(
    backbone,
    test_dataset,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 24,
) -> dict:
    """Compute baseline accuracy using raw backbone embeddings."""
    print(f"\nComputing baseline...")

    backbone = backbone.to(device)
    backbone.eval()

    all_embeddings = []
    all_l0_labels = []
    all_l1_labels = []

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            "texts": [item.text for item in batch],
            "l0_labels": torch.tensor([item.level0_label for item in batch]),
            "l1_labels": torch.tensor([item.level1_label for item in batch]),
        },
        num_workers=0,
    )

    for batch in tqdm(dataloader, desc="Baseline embedding"):
        texts = batch["texts"]
        l0_labels = batch["l0_labels"]
        l1_labels = batch["l1_labels"]

        # Tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = backbone(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )

            # Mean pooling
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float().to(device)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

            all_embeddings.append(pooled.cpu())
            all_l0_labels.append(l0_labels)
            all_l1_labels.append(l1_labels)

    # Concatenate
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_l0_labels = torch.cat(all_l0_labels).numpy()
    all_l1_labels = torch.cat(all_l1_labels).numpy()

    # KNN evaluation
    n_train = int(len(all_embeddings) * 0.8)
    train_emb = all_embeddings[:n_train]
    test_emb = all_embeddings[n_train:]
    train_l0 = all_l0_labels[:n_train]
    test_l0 = all_l0_labels[n_train:]
    train_l1 = all_l1_labels[:n_train]
    test_l1 = all_l1_labels[n_train:]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_emb, train_l0)
    l0_baseline = accuracy_score(test_l0, knn.predict(test_emb))

    knn.fit(train_emb, train_l1)
    l1_baseline = accuracy_score(test_l1, knn.predict(test_emb))

    print(f"  Baseline L0: {l0_baseline:.4f}, L1: {l1_baseline:.4f}")

    return {
        "l0": l0_baseline,
        "l1": l1_baseline,
    }


def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_scales", type=int, default=4)
    parser.add_argument("--scale_dim", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("HYPERBOLIC FRACTAL HEAD (HFH) - V6 TRAINING")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    train_data = load_hierarchical_dataset(args.dataset, split="train", max_samples=10000)
    val_data = load_hierarchical_dataset(args.dataset, split="test", max_samples=2000)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Get number of classes
    num_l0 = len(set(item.level0_label for item in train_data))
    num_l1 = len(set(item.level1_label for item in train_data))
    print(f"  L0 classes: {num_l0}, L1 classes: {num_l1}")

    # Load backbone
    print(f"\nLoading backbone: {args.model}...")
    model_config = MODELS.get(args.model)
    if model_config is None:
        print(f"Unknown model: {args.model}")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.hf_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    backbone = AutoModel.from_pretrained(
        model_config.hf_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float32,
    )

    hidden_dim = backbone.config.hidden_size
    print(f"  Hidden dim: {hidden_dim}")

    # Compute baseline
    baseline = compute_baseline(backbone, val_data, tokenizer, device, args.batch_size)

    # Create HFH config
    hfh_config = HFHConfig(
        input_dim=hidden_dim,
        num_scales=args.num_scales,
        scale_dim=args.scale_dim,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        manifolds=("lorentz", "lorentz", "poincare", "poincare"),
        init_curvatures=(0.5, 1.0, 1.0, 2.0),
        hyperbolic_weight=1.0,
        euclidean_weight=0.3,
        corr_dim_weight=0.05,
        use_wavelet=True,
    )

    # Create model
    model = HyperbolicFractalModel(backbone, hfh_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable:,}")

    # Train
    trainer = HFHTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        tokenizer=tokenizer,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    history = trainer.train()

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=trainer.collate_fn,
        num_workers=0,
    )

    final_metrics = trainer.evaluate(val_loader)

    print(f"\nBaseline:")
    print(f"  L0: {baseline['l0']:.4f}")
    print(f"  L1: {baseline['l1']:.4f}")

    print(f"\nHFH (V6):")
    print(f"  L0: {final_metrics['l0_acc']:.4f} (delta: {final_metrics['l0_acc'] - baseline['l0']:+.4f})")
    print(f"  L1: {final_metrics['l1_acc']:.4f} (delta: {final_metrics['l1_acc'] - baseline['l1']:+.4f})")
    print(f"  L0 KNN: {final_metrics['l0_knn']:.4f}")
    print(f"  L1 KNN: {final_metrics['l1_knn']:.4f}")
    print(f"  Final curvatures: {[f'{c:.3f}' for c in final_metrics['curvatures']]}")

    # Compute improvement percentages
    l0_delta_pct = (final_metrics['l0_acc'] - baseline['l0']) * 100
    l1_delta_pct = (final_metrics['l1_acc'] - baseline['l1']) * 100

    print(f"\n*** IMPROVEMENT ***")
    print(f"  L0: {l0_delta_pct:+.2f}%")
    print(f"  L1: {l1_delta_pct:+.2f}%")

    # Save results
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "config": {
            "num_scales": args.num_scales,
            "scale_dim": args.scale_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        "baseline": baseline,
        "hfh": {
            "l0_acc": final_metrics["l0_acc"],
            "l1_acc": final_metrics["l1_acc"],
            "l0_knn": final_metrics["l0_knn"],
            "l1_knn": final_metrics["l1_knn"],
            "curvatures": final_metrics["curvatures"],
        },
        "delta": {
            "l0": final_metrics["l0_acc"] - baseline["l0"],
            "l1": final_metrics["l1_acc"] - baseline["l1"],
        },
        "delta_pct": {
            "l0": l0_delta_pct,
            "l1": l1_delta_pct,
        },
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = f"../results/hfh_v6_{args.model}_{args.dataset}_seed{args.seed}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
