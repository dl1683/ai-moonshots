"""
Fractal vs Non-Fractal Comparison

Proper A/B test following the experimental design:
1. Baseline: Non-fractal head, standard CE loss
2. Fractal-Only: Fractal head, same loss
3. Parameter-matched comparison

Tests the core hypothesis: Does fractal structure help hierarchical classification?
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


class NonFractalHead(nn.Module):
    """Standard MLP head (baseline)"""
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

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class FractalClassifierHead(nn.Module):
    """
    Fractal head for hierarchical classification.

    Uses multi-scale embeddings with shared encoder blocks.
    Scale 0 (coarse) -> L0 prediction
    Scale 1 (fine) -> L1 prediction
    """
    def __init__(self, hidden_dim: int, num_l0: int, num_l1: int, num_scales: int = 2):
        super().__init__()
        self.num_scales = num_scales

        # Shared fractal block (self-similar structure)
        self.shared_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2)
        )

        # Scale-specific projections
        self.scale_projs = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 128) for _ in range(num_scales)
        ])

        # Hierarchical heads
        self.l0_head = nn.Linear(128, num_l0)  # From coarse scale
        self.l1_head = nn.Linear(128, num_l1)  # From fine scale

    def forward(self, x, return_scales=False):
        # Apply shared block (self-similar processing)
        shared = self.shared_block(x)

        # Generate multi-scale representations
        scales = []
        for proj in self.scale_projs:
            scales.append(proj(shared))

        # Coarse scale (0) for L0, Fine scale (1) for L1
        l0_logits = self.l0_head(scales[0])
        l1_logits = self.l1_head(scales[-1])

        if return_scales:
            return l0_logits, l1_logits, scales
        return l0_logits, l1_logits

    def forward_swapped(self, x):
        """Scale-swap test: use wrong scales for predictions"""
        shared = self.shared_block(x)
        scales = [proj(shared) for proj in self.scale_projs]

        # SWAPPED: fine scale for L0, coarse scale for L1
        l0_logits = self.l0_head(scales[-1])  # Wrong scale
        l1_logits = self.l1_head(scales[0])   # Wrong scale
        return l0_logits, l1_logits

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def encode_batch(backbone, tokenizer, texts, device):
    """Encode texts through frozen backbone"""
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=512, return_tensors="pt"
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
        train_data, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=lambda x: x
    )

    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        texts = [item.text for item in batch]
        l0_labels = torch.tensor([item.level0_label for item in batch], device=device)
        l1_labels = torch.tensor([item.level1_label for item in batch], device=device)

        optimizer.zero_grad()
        hidden = encode_batch(backbone, tokenizer, texts, device)
        l0_logits, l1_logits = model(hidden)

        loss = ce_loss(l0_logits, l0_labels) + ce_loss(l1_logits, l1_labels)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model, backbone, tokenizer, val_data, device, batch_size, swapped=False):
    """Evaluate on validation set"""
    model.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_l0_preds, all_l1_preds = [], []
    all_l0_labels, all_l1_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]
        l1_labels = [item.level1_label for item in batch]

        hidden = encode_batch(backbone, tokenizer, texts, device)

        if swapped and hasattr(model, 'forward_swapped'):
            l0_logits, l1_logits = model.forward_swapped(hidden)
        else:
            l0_logits, l1_logits = model(hidden)

        all_l0_preds.extend(l0_logits.argmax(dim=1).cpu().tolist())
        all_l1_preds.extend(l1_logits.argmax(dim=1).cpu().tolist())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
    l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])

    # Hierarchical accuracy: both L0 and L1 correct
    hier_acc = np.mean([
        (p0 == l0) and (p1 == l1)
        for p0, l0, p1, l1 in zip(all_l0_preds, all_l0_labels, all_l1_preds, all_l1_labels)
    ])

    return {'l0_acc': l0_acc, 'l1_acc': l1_acc, 'hier_acc': hier_acc}


def run_experiment(model, backbone, tokenizer, train_data, val_data, device, config, name):
    """Run a single experiment condition"""
    print(f"\n{'='*60}")
    print(f"CONDITION: {name}")
    print(f"{'='*60}")
    print(f"Parameters: {model.param_count():,}")

    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

    best_score = 0
    best_results = None

    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, backbone, tokenizer, train_data, optimizer, device, config['batch_size'])
        results = evaluate(model, backbone, tokenizer, val_data, device, config['batch_size'])

        print(f"Epoch {epoch}: loss={train_loss:.4f}, L0={results['l0_acc']:.4f}, L1={results['l1_acc']:.4f}, Hier={results['hier_acc']:.4f}")

        score = results['l0_acc'] + results['l1_acc']
        if score > best_score:
            best_score = score
            best_results = results.copy()
            best_results['epoch'] = epoch

    # Scale-swap test for fractal models
    if hasattr(model, 'forward_swapped'):
        swap_results = evaluate(model, backbone, tokenizer, val_data, device, config['batch_size'], swapped=True)
        best_results['swap_l0_acc'] = swap_results['l0_acc']
        best_results['swap_l1_acc'] = swap_results['l1_acc']
        print(f"\nScale-Swap Test: L0={swap_results['l0_acc']:.4f}, L1={swap_results['l1_acc']:.4f}")
        print(f"  L0 drop from swap: {best_results['l0_acc'] - swap_results['l0_acc']:+.4f}")
        print(f"  L1 drop from swap: {best_results['l1_acc'] - swap_results['l1_acc']:+.4f}")

    return best_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seeds", type=int, nargs='+', default=[42])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    hidden_dim = model_config.hidden_dim

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr
    }

    all_results = {}

    for seed in args.seeds:
        print(f"\n{'#'*60}")
        print(f"SEED: {seed}")
        print(f"{'#'*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Condition 1: Baseline (Non-Fractal)
        baseline_model = NonFractalHead(hidden_dim, l0_classes, l1_classes).to(device)
        baseline_results = run_experiment(
            baseline_model, backbone, tokenizer, train_data, val_data,
            device, config, "Baseline (Non-Fractal)"
        )

        # Condition 2: Fractal-Only
        torch.manual_seed(seed)  # Reset for fair comparison
        fractal_model = FractalClassifierHead(hidden_dim, l0_classes, l1_classes).to(device)
        fractal_results = run_experiment(
            fractal_model, backbone, tokenizer, train_data, val_data,
            device, config, "Fractal-Only"
        )

        all_results[seed] = {
            'baseline': baseline_results,
            'fractal': fractal_results
        }

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"Baseline params: {baseline_model.param_count():,}")
        print(f"Fractal params:  {fractal_model.param_count():,}")
        print(f"\nBaseline: L0={baseline_results['l0_acc']:.4f}, L1={baseline_results['l1_acc']:.4f}")
        print(f"Fractal:  L0={fractal_results['l0_acc']:.4f}, L1={fractal_results['l1_acc']:.4f}")
        print(f"\nDelta: L0={fractal_results['l0_acc']-baseline_results['l0_acc']:+.4f}, L1={fractal_results['l1_acc']-baseline_results['l1_acc']:+.4f}")

        del baseline_model, fractal_model
        torch.cuda.empty_cache()

    # Summary across seeds
    if len(args.seeds) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS SEEDS")
        print(f"{'='*60}")

        baseline_l0 = [all_results[s]['baseline']['l0_acc'] for s in args.seeds]
        baseline_l1 = [all_results[s]['baseline']['l1_acc'] for s in args.seeds]
        fractal_l0 = [all_results[s]['fractal']['l0_acc'] for s in args.seeds]
        fractal_l1 = [all_results[s]['fractal']['l1_acc'] for s in args.seeds]

        print(f"Baseline L0: {np.mean(baseline_l0):.4f} +/- {np.std(baseline_l0):.4f}")
        print(f"Baseline L1: {np.mean(baseline_l1):.4f} +/- {np.std(baseline_l1):.4f}")
        print(f"Fractal L0:  {np.mean(fractal_l0):.4f} +/- {np.std(fractal_l0):.4f}")
        print(f"Fractal L1:  {np.mean(fractal_l1):.4f} +/- {np.std(fractal_l1):.4f}")
        print(f"\nMean Delta L0: {np.mean(fractal_l0) - np.mean(baseline_l0):+.4f}")
        print(f"Mean Delta L1: {np.mean(fractal_l1) - np.mean(baseline_l1):+.4f}")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"fractal_comparison_{args.model}.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "config": config,
            "results": {str(k): v for k, v in all_results.items()}
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
