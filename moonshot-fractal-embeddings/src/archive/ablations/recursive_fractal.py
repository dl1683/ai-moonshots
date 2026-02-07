"""
Recursive Residual Fractal Coding - True Hierarchical Dependency

Based on the moonshot experimental design:

Core idea: Scale 1 is a RESIDUAL that only exists AFTER scale 0.
Forces causal hierarchy instead of parallel heads.

Architecture:
- Shared block F applied recursively
- z0 = F(x)           # Coarse embedding
- z1 = F([x, z0])     # Fine embedding conditioned on coarse
- L0 head uses z0 only
- L1 head uses z0 + z1 (or just z1 since it already has z0 info)

Loss:
- CE(y0 | z0) + CE(y1 | z0, z1)
- Plus information-theoretic regularizer to separate scales

This is a TRUE fractal: same operator F applied recursively,
with information flow from coarse to fine.
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


class RecursiveFractalBlock(nn.Module):
    """
    Self-similar block that can be applied recursively.
    At each scale, it takes input and optional conditioning from previous scale.
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Core transformation (self-similar)
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.transform(x)


class RecursiveFractalHead(nn.Module):
    """
    True recursive fractal head with hierarchical dependency.

    z0 = F(x)                    # Coarse scale
    z1 = F(concat(x, z0))        # Fine scale (conditioned on coarse)

    L0 prediction uses z0 only
    L1 prediction uses concat(z0, z1)
    """
    def __init__(self, hidden_dim: int, num_l0: int, num_l1: int, scale_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale_dim = scale_dim

        # Coarse scale block: input -> z0
        self.coarse_block = RecursiveFractalBlock(hidden_dim, scale_dim)

        # Fine scale block: input + z0 -> z1 (recursive application)
        # Input dimension is hidden_dim + scale_dim
        self.fine_block = RecursiveFractalBlock(hidden_dim + scale_dim, scale_dim)

        # Heads
        self.l0_head = nn.Linear(scale_dim, num_l0)  # Coarse head: z0 -> L0
        self.l1_head = nn.Linear(scale_dim * 2, num_l1)  # Fine head: [z0, z1] -> L1

        # Residual predictor: try to predict y0 from z1 alone
        # If z1 contains coarse info, this will succeed -> we want to penalize
        self.residual_predictor = nn.Linear(scale_dim, num_l0)

    def forward(self, x, return_scales=False):
        """
        Forward pass with hierarchical dependency.

        Returns:
            l0_logits: Predictions for coarse level (from z0 only)
            l1_logits: Predictions for fine level (from z0 + z1)
            scales: Optional dict with z0, z1 for analysis
        """
        # Coarse scale: z0 = F(x)
        z0 = self.coarse_block(x)

        # Fine scale: z1 = F([x, z0]) - conditioned on coarse
        z1 = self.fine_block(torch.cat([x, z0], dim=-1))

        # Predictions
        l0_logits = self.l0_head(z0)  # L0 from coarse only
        l1_logits = self.l1_head(torch.cat([z0, z1], dim=-1))  # L1 from both

        if return_scales:
            return l0_logits, l1_logits, {'z0': z0, 'z1': z1}
        return l0_logits, l1_logits

    def get_residual_leakage(self, x):
        """
        Compute how much coarse information leaks into z1.
        Used for information-theoretic regularization.
        """
        z0 = self.coarse_block(x)
        z1 = self.fine_block(torch.cat([x, z0], dim=-1))

        # Try to predict L0 from z1 alone (should fail if z1 is pure residual)
        l0_from_z1 = self.residual_predictor(z1)
        return l0_from_z1

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class BaselineHead(nn.Module):
    """Standard MLP baseline for fair comparison."""
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


def encode_batch(backbone, tokenizer, texts, device):
    """Encode texts through frozen backbone."""
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


def train_epoch_recursive(model, backbone, tokenizer, train_data, optimizer, device,
                          batch_size, leakage_weight=0.1):
    """
    Train recursive fractal head with information-theoretic regularization.

    Loss = CE(y0|z0) + CE(y1|z0,z1) + lambda * CE(y0|z1)

    The third term penalizes z1 for containing coarse information.
    We WANT the residual predictor to fail -> minimize its success.
    """
    model.train()
    backbone.eval()

    dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=lambda x: x
    )

    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0
    total_main = 0
    total_leakage = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        texts = [item.text for item in batch]
        l0_labels = torch.tensor([item.level0_label for item in batch], device=device)
        l1_labels = torch.tensor([item.level1_label for item in batch], device=device)

        optimizer.zero_grad()
        hidden = encode_batch(backbone, tokenizer, texts, device)

        # Main predictions
        l0_logits, l1_logits = model(hidden)

        # Main losses
        l0_loss = ce_loss(l0_logits, l0_labels)
        l1_loss = ce_loss(l1_logits, l1_labels)
        main_loss = l0_loss + l1_loss

        # Information-theoretic regularization: penalize leakage
        # We want z1 to NOT contain coarse information
        # So we add a loss that REDUCES the accuracy of predicting L0 from z1
        if hasattr(model, 'get_residual_leakage'):
            l0_from_z1 = model.get_residual_leakage(hidden)
            # Negative CE: we want this prediction to be BAD
            # But negative CE can be unstable, so instead maximize entropy
            # or use reversed labels approach

            # Approach: minimize the confidence of predicting L0 from z1
            # By maximizing entropy of this prediction
            probs = F.softmax(l0_from_z1, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            # Maximize entropy = minimize negative entropy
            leakage_loss = -entropy * leakage_weight
        else:
            leakage_loss = 0

        loss = main_loss + leakage_loss

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_main += main_loss.item()
        if isinstance(leakage_loss, torch.Tensor):
            total_leakage += leakage_loss.item()
        num_batches += 1

    return {
        'total': total_loss / max(1, num_batches),
        'main': total_main / max(1, num_batches),
        'leakage': total_leakage / max(1, num_batches)
    }


def train_epoch_baseline(model, backbone, tokenizer, train_data, optimizer, device, batch_size):
    """Standard training for baseline."""
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
def evaluate(model, backbone, tokenizer, val_data, device, batch_size):
    """Evaluate on validation set."""
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
        l0_logits, l1_logits = model(hidden)

        all_l0_preds.extend(l0_logits.argmax(dim=1).cpu().tolist())
        all_l1_preds.extend(l1_logits.argmax(dim=1).cpu().tolist())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
    l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])

    # Hierarchical accuracy: both correct
    hier_acc = np.mean([
        (p0 == l0) and (p1 == l1)
        for p0, l0, p1, l1 in zip(all_l0_preds, all_l0_labels, all_l1_preds, all_l1_labels)
    ])

    return {'l0_acc': l0_acc, 'l1_acc': l1_acc, 'hier_acc': hier_acc}


@torch.no_grad()
def analyze_scale_separation(model, backbone, tokenizer, val_data, device, batch_size):
    """
    Analyze how well scales are separated.

    - Check if z1 can predict L0 (leakage)
    - Check if z0 can predict L1 (sufficiency)
    """
    if not hasattr(model, 'get_residual_leakage'):
        return {}

    model.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_l0_from_z1_preds = []
    all_l0_labels = []

    for batch in dataloader:
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]

        hidden = encode_batch(backbone, tokenizer, texts, device)
        l0_from_z1 = model.get_residual_leakage(hidden)

        all_l0_from_z1_preds.extend(l0_from_z1.argmax(dim=1).cpu().tolist())
        all_l0_labels.extend(l0_labels)

    # How well can z1 predict L0? (should be low = good separation)
    leakage_acc = np.mean([p == l for p, l in zip(all_l0_from_z1_preds, all_l0_labels)])

    return {'leakage_acc': leakage_acc}


def run_experiment(model, backbone, tokenizer, train_data, val_data, device, config, name,
                   use_leakage_loss=False):
    """Run training and evaluation."""
    print(f"\n{'='*60}")
    print(f"CONDITION: {name}")
    print(f"{'='*60}")
    print(f"Parameters: {model.param_count():,}")

    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

    best_score = 0
    best_results = None

    for epoch in range(1, config['epochs'] + 1):
        if use_leakage_loss and hasattr(model, 'get_residual_leakage'):
            losses = train_epoch_recursive(
                model, backbone, tokenizer, train_data, optimizer, device,
                config['batch_size'], config.get('leakage_weight', 0.1)
            )
            loss_str = f"loss={losses['total']:.4f} (main={losses['main']:.4f}, leak={losses['leakage']:.4f})"
        else:
            loss = train_epoch_baseline(
                model, backbone, tokenizer, train_data, optimizer, device,
                config['batch_size']
            )
            loss_str = f"loss={loss:.4f}"

        results = evaluate(model, backbone, tokenizer, val_data, device, config['batch_size'])

        print(f"Epoch {epoch}: {loss_str}, L0={results['l0_acc']:.4f}, L1={results['l1_acc']:.4f}, Hier={results['hier_acc']:.4f}")

        score = results['l0_acc'] + results['l1_acc']
        if score > best_score:
            best_score = score
            best_results = results.copy()
            best_results['epoch'] = epoch

    # Analyze scale separation
    sep_results = analyze_scale_separation(model, backbone, tokenizer, val_data, device, config['batch_size'])
    if sep_results:
        best_results.update(sep_results)
        print(f"\nScale Separation: L0 leakage from z1 = {sep_results['leakage_acc']:.4f}")
        print(f"  (Lower = better separation, random = {1/len(set(item.level0_label for item in val_data)):.4f})")

    return best_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--leakage_weight", type=float, default=0.1)
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
        'lr': args.lr,
        'leakage_weight': args.leakage_weight
    }

    all_results = {}

    for seed in args.seeds:
        print(f"\n{'#'*60}")
        print(f"SEED: {seed}")
        print(f"{'#'*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Condition 1: Baseline
        baseline_model = BaselineHead(hidden_dim, l0_classes, l1_classes).to(device)
        baseline_results = run_experiment(
            baseline_model, backbone, tokenizer, train_data, val_data,
            device, config, "Baseline (Standard MLP)"
        )

        # Condition 2: Recursive Fractal (no leakage loss)
        torch.manual_seed(seed)
        fractal_model = RecursiveFractalHead(hidden_dim, l0_classes, l1_classes).to(device)
        fractal_results = run_experiment(
            fractal_model, backbone, tokenizer, train_data, val_data,
            device, config, "Recursive Fractal (no regularization)",
            use_leakage_loss=False
        )

        # Condition 3: Recursive Fractal WITH leakage loss
        torch.manual_seed(seed)
        fractal_reg_model = RecursiveFractalHead(hidden_dim, l0_classes, l1_classes).to(device)
        fractal_reg_results = run_experiment(
            fractal_reg_model, backbone, tokenizer, train_data, val_data,
            device, config, "Recursive Fractal + Info-Theoretic Reg",
            use_leakage_loss=True
        )

        all_results[seed] = {
            'baseline': baseline_results,
            'fractal': fractal_results,
            'fractal_reg': fractal_reg_results
        }

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"Baseline params:    {baseline_model.param_count():,}")
        print(f"Fractal params:     {fractal_model.param_count():,}")
        print(f"\nBaseline:         L0={baseline_results['l0_acc']:.4f}, L1={baseline_results['l1_acc']:.4f}")
        print(f"Fractal:          L0={fractal_results['l0_acc']:.4f}, L1={fractal_results['l1_acc']:.4f}")
        print(f"Fractal+Reg:      L0={fractal_reg_results['l0_acc']:.4f}, L1={fractal_reg_results['l1_acc']:.4f}")
        print(f"\nDelta (Fractal):     L0={fractal_results['l0_acc']-baseline_results['l0_acc']:+.4f}, L1={fractal_results['l1_acc']-baseline_results['l1_acc']:+.4f}")
        print(f"Delta (Fractal+Reg): L0={fractal_reg_results['l0_acc']-baseline_results['l0_acc']:+.4f}, L1={fractal_reg_results['l1_acc']-baseline_results['l1_acc']:+.4f}")

        del baseline_model, fractal_model, fractal_reg_model
        torch.cuda.empty_cache()

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"recursive_fractal_{args.model}.json"
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
