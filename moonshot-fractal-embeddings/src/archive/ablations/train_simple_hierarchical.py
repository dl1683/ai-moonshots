"""
Simple Hierarchical Classification Training (No Frills)

Just multi-task CE loss with frozen backbone.
No contrastive loss, no AMP, no complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from multi_model_pipeline import MODELS, load_model
from hierarchical_datasets import load_hierarchical_dataset


class SimpleHierarchicalClassifier(nn.Module):
    """Simple 2-layer MLP for hierarchical classification"""

    def __init__(self, hidden_dim: int, num_l0: int, num_l1: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256)
        )
        self.l0_head = nn.Linear(256, num_l0)
        self.l1_head = nn.Linear(256, num_l1)

    def forward(self, x):
        features = self.projection(x)
        return self.l0_head(features), self.l1_head(features)


def encode_batch(backbone, tokenizer, texts, device):
    """Encode texts through frozen backbone"""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = backbone(**inputs)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state[:, -1, :]
        else:
            hidden = outputs[0][:, -1, :]

    return hidden.float()


def train_epoch(model, backbone, tokenizer, train_data, optimizer, device, batch_size):
    """Train for one epoch"""
    model.train()
    backbone.eval()

    dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: x
    )

    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        texts = [item.text for item in batch]
        l0_labels = torch.tensor([item.level0_label for item in batch], device=device)
        l1_labels = torch.tensor([item.level1_label for item in batch], device=device)

        optimizer.zero_grad()

        # Encode and classify
        hidden = encode_batch(backbone, tokenizer, texts, device)
        l0_logits, l1_logits = model(hidden)

        # Loss
        loss = ce_loss(l0_logits, l0_labels) + ce_loss(l1_logits, l1_labels)

        # Skip NaN (shouldn't happen but just in case)
        if torch.isnan(loss):
            print("NaN detected, skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model, backbone, tokenizer, val_data, device, batch_size):
    """Evaluate on validation set"""
    model.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_l0_preds, all_l1_preds = [], []
    all_l0_labels, all_l1_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]
        l1_labels = [item.level1_label for item in batch]

        hidden = encode_batch(backbone, tokenizer, texts, device)
        l0_logits, l1_logits = model(hidden)

        all_l0_preds.extend(l0_logits.argmax(dim=1).cpu().tolist())
        all_l1_preds.extend(l1_logits.argmax(dim=1).cpu().tolist())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
    l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])

    return l0_acc, l1_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
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

    l0_classes = len(set(item.level0_label for item in train_data))
    l1_classes = len(set(item.level1_label for item in train_data))
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"  L0: {l0_classes} classes, L1: {l1_classes} classes")

    # Load backbone
    print(f"\nLoading backbone: {args.model}...")
    model_config = MODELS[args.model]
    wrapper = load_model(args.model, use_fractal=False, device=device)
    backbone = wrapper.backbone
    tokenizer = wrapper.tokenizer

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    # Create classifier
    classifier = SimpleHierarchicalClassifier(
        hidden_dim=model_config.hidden_dim,
        num_l0=l0_classes,
        num_l1=l1_classes
    ).to(device)

    params = sum(p.numel() for p in classifier.parameters())
    print(f"  Classifier params: {params:,}")

    # Baseline
    print("\nComputing baseline...")
    baseline_l0, baseline_l1 = evaluate(classifier, backbone, tokenizer, val_data, device, args.batch_size)
    print(f"  Baseline L0: {baseline_l0:.4f}, L1: {baseline_l1:.4f}")

    # Optimizer
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)

    # Training
    print(f"\n{'='*60}")
    print("SIMPLE HIERARCHICAL TRAINING")
    print(f"{'='*60}")

    best_score = 0
    best_results = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss = train_epoch(classifier, backbone, tokenizer, train_data, optimizer, device, args.batch_size)
        print(f"  Train loss: {train_loss:.4f}")

        l0_acc, l1_acc = evaluate(classifier, backbone, tokenizer, val_data, device, args.batch_size)
        print(f"  Val L0: {l0_acc:.4f}, L1: {l1_acc:.4f}")

        score = l0_acc + l1_acc
        if score > best_score:
            best_score = score
            best_results = {'epoch': epoch, 'l0': l0_acc, 'l1': l1_acc}
            print(f"  ** New best!")

    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Baseline: L0={baseline_l0:.4f}, L1={baseline_l1:.4f}")
    if best_results:
        print(f"Best: L0={best_results['l0']:.4f}, L1={best_results['l1']:.4f}")
        delta_l0 = best_results['l0'] - baseline_l0
        delta_l1 = best_results['l1'] - baseline_l1
        print(f"Delta: L0={delta_l0:+.4f}, L1={delta_l1:+.4f}")
        print(f"Delta %: L0={100*delta_l0:+.2f}%, L1={100*delta_l1:+.2f}%")

    # Save
    results_path = Path(__file__).parent.parent / "results" / f"simple_hier_{args.model}.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "baseline": {"l0": float(baseline_l0), "l1": float(baseline_l1)},
            "best": {
                "l0": float(best_results['l0']),
                "l1": float(best_results['l1']),
                "epoch": best_results['epoch']
            } if best_results else None
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
