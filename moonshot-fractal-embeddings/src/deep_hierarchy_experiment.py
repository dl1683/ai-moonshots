"""
Deep Hierarchy Experiment
==========================

Test fractal embeddings on DEEP hierarchies where structure should matter.

Uses DBPedia which has:
- 6 super-categories (L0)
- 14 categories (L1)
- Natural semantic relationships

And Amazon Product Reviews for even deeper hierarchy.

Key hypothesis: Fractal embeddings should show advantage on DEEPER hierarchies
where flat classifiers struggle to capture multi-level structure.
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

from multi_model_pipeline import MODELS
from hierarchical_datasets import load_hierarchical_dataset


class FlatClassifier(nn.Module):
    """Standard flat classifier (baseline)."""

    def __init__(self, hidden_dim: int, num_classes: list):
        super().__init__()
        self.num_levels = len(num_classes)

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.heads = nn.ModuleList([
            nn.Linear(512, num_c) for num_c in num_classes
        ])

    def forward(self, x):
        shared = self.shared(x)
        return [head(shared) for head in self.heads]


class HierarchicalFractalClassifier(nn.Module):
    """
    Fractal classifier with recursive structure for deep hierarchies.

    Key insight: Each level builds on the previous level's representation.
    L0 -> L1 -> L2 -> ... with residual connections and shared processing.
    """

    def __init__(self, hidden_dim: int, num_classes: list, proj_dim: int = 256):
        super().__init__()
        self.num_levels = len(num_classes)
        self.proj_dim = proj_dim

        # Shared fractal block (applied recursively)
        self.fractal_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Level-specific projections
        self.level_projs = nn.ModuleList()
        self.level_heads = nn.ModuleList()

        for i, num_c in enumerate(num_classes):
            # Projection for this level (conditioned on previous)
            input_dim = hidden_dim // 2 + (proj_dim if i > 0 else 0)
            self.level_projs.append(nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.LayerNorm(proj_dim),
            ))
            self.level_heads.append(nn.Linear(proj_dim, num_c))

    def forward(self, x, return_embeddings=False):
        """
        Forward pass building hierarchy recursively.

        Level 0: fractal_block(x) -> proj_0 -> head_0
        Level 1: [fractal_block(x), proj_0] -> proj_1 -> head_1
        Level 2: [fractal_block(x), proj_1] -> proj_2 -> head_2
        ...
        """
        # Shared fractal processing
        shared = self.fractal_block(x)

        logits = []
        embeddings = []
        prev_emb = None

        for i in range(self.num_levels):
            # Concatenate with previous level embedding
            if prev_emb is not None:
                level_input = torch.cat([shared, prev_emb], dim=-1)
            else:
                level_input = shared

            # Project and classify
            level_emb = self.level_projs[i](level_input)
            level_emb = F.normalize(level_emb, dim=-1)
            level_logits = self.level_heads[i](level_emb)

            logits.append(level_logits)
            embeddings.append(level_emb)
            prev_emb = level_emb

        if return_embeddings:
            return logits, embeddings
        return logits


def encode_batch(backbone, tokenizer, texts, device):
    """Encode texts through backbone."""
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


def train_epoch(model, backbone, tokenizer, train_data, optimizer, device, batch_size, level_names):
    """Train for one epoch."""
    model.train()
    backbone.eval()

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        texts = [item.text for item in batch]

        # Get labels for all levels
        labels = []
        for level_name in level_names:
            level_labels = torch.tensor(
                [getattr(item, f"{level_name}_label") for item in batch],
                device=device
            )
            labels.append(level_labels)

        optimizer.zero_grad()
        hidden = encode_batch(backbone, tokenizer, texts, device)
        logits = model(hidden)

        # Loss for all levels
        loss = sum(ce_loss_fn(l, lab) for l, lab in zip(logits, labels))

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model, backbone, tokenizer, val_data, device, batch_size, level_names):
    """Evaluate on validation set."""
    model.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_preds = [[] for _ in level_names]
    all_labels = [[] for _ in level_names]

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        texts = [item.text for item in batch]

        for i, level_name in enumerate(level_names):
            level_labels = [getattr(item, f"{level_name}_label") for item in batch]
            all_labels[i].extend(level_labels)

        hidden = encode_batch(backbone, tokenizer, texts, device)
        logits = model(hidden)

        for i, l in enumerate(logits):
            all_preds[i].extend(l.argmax(dim=1).cpu().tolist())

    results = {}
    for i, level_name in enumerate(level_names):
        acc = np.mean([p == l for p, l in zip(all_preds[i], all_labels[i])])
        results[f'{level_name}_acc'] = acc

    # Hierarchical accuracy (all levels correct)
    hier_correct = [
        all(all_preds[i][j] == all_labels[i][j] for i in range(len(level_names)))
        for j in range(len(all_preds[0]))
    ]
    results['hier_acc'] = np.mean(hier_correct)

    return results


def run_comparison(dataset_name, backbone_name, device, epochs=10, batch_size=24):
    """Run flat vs fractal comparison on a dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    val_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)

    # Get number of classes at each level
    num_l0 = len(set(item.level0_label for item in train_data))
    num_l1 = len(set(item.level1_label for item in train_data))
    num_classes = [num_l0, num_l1]
    level_names = ['level0', 'level1']

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"  L0: {num_l0} classes, L1: {num_l1} classes")
    print(f"  Hierarchy depth: {len(num_classes)} levels")

    # Load backbone
    model_config = MODELS[backbone_name]
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_path, trust_remote_code=model_config.trust_remote_code)
    if model_config.pooling == "last":
        tokenizer.padding_side = "left"

    backbone = AutoModel.from_pretrained(
        model_config.hf_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16,
    ).to(device)

    for p in backbone.parameters():
        p.requires_grad = False

    hidden_dim = model_config.hidden_dim

    results = {}

    # Test Flat Classifier
    print("\n--- FLAT CLASSIFIER ---")
    flat_model = FlatClassifier(hidden_dim, num_classes).to(device)
    flat_optimizer = AdamW(flat_model.parameters(), lr=5e-4, weight_decay=0.01)

    best_flat = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(flat_model, backbone, tokenizer, train_data, flat_optimizer, device, batch_size, level_names)
        eval_results = evaluate(flat_model, backbone, tokenizer, val_data, device, batch_size, level_names)

        if best_flat is None or eval_results['level0_acc'] + eval_results['level1_acc'] > best_flat['level0_acc'] + best_flat['level1_acc']:
            best_flat = eval_results.copy()
            best_flat['epoch'] = epoch

        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: L0={eval_results['level0_acc']:.4f}, L1={eval_results['level1_acc']:.4f}")

    results['flat'] = best_flat
    print(f"  Best: L0={best_flat['level0_acc']:.4f}, L1={best_flat['level1_acc']:.4f}")

    del flat_model
    torch.cuda.empty_cache()

    # Test Fractal Classifier
    print("\n--- FRACTAL CLASSIFIER ---")
    fractal_model = HierarchicalFractalClassifier(hidden_dim, num_classes).to(device)
    fractal_optimizer = AdamW(fractal_model.parameters(), lr=5e-4, weight_decay=0.01)

    best_fractal = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(fractal_model, backbone, tokenizer, train_data, fractal_optimizer, device, batch_size, level_names)
        eval_results = evaluate(fractal_model, backbone, tokenizer, val_data, device, batch_size, level_names)

        if best_fractal is None or eval_results['level0_acc'] + eval_results['level1_acc'] > best_fractal['level0_acc'] + best_fractal['level1_acc']:
            best_fractal = eval_results.copy()
            best_fractal['epoch'] = epoch

        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: L0={eval_results['level0_acc']:.4f}, L1={eval_results['level1_acc']:.4f}")

    results['fractal'] = best_fractal
    print(f"  Best: L0={best_fractal['level0_acc']:.4f}, L1={best_fractal['level1_acc']:.4f}")

    # Comparison
    print("\n--- COMPARISON ---")
    delta_l0 = best_fractal['level0_acc'] - best_flat['level0_acc']
    delta_l1 = best_fractal['level1_acc'] - best_flat['level1_acc']
    print(f"  Flat:    L0={best_flat['level0_acc']:.4f}, L1={best_flat['level1_acc']:.4f}")
    print(f"  Fractal: L0={best_fractal['level0_acc']:.4f}, L1={best_fractal['level1_acc']:.4f}")
    print(f"  Delta:   L0={delta_l0:+.4f}, L1={delta_l1:+.4f}")

    results['delta'] = {'l0': delta_l0, 'l1': delta_l1}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test on multiple datasets with different hierarchy depths
    datasets = ["yahoo", "dbpedia", "agnews"]
    all_results = {}

    for dataset_name in datasets:
        try:
            results = run_comparison(dataset_name, args.model, device, args.epochs, args.batch_size)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Dataset':<12} {'Flat L0':<10} {'Flat L1':<10} {'Frac L0':<10} {'Frac L1':<10} {'Delta L0':<10} {'Delta L1':<10}")
    print("-" * 72)

    for dataset_name, results in all_results.items():
        if 'error' in results:
            print(f"{dataset_name:<12} ERROR: {results['error'][:40]}")
            continue

        flat = results['flat']
        frac = results['fractal']
        delta = results['delta']

        print(f"{dataset_name:<12} {flat['level0_acc']:<10.4f} {flat['level1_acc']:<10.4f} "
              f"{frac['level0_acc']:<10.4f} {frac['level1_acc']:<10.4f} "
              f"{delta['l0']:+<10.4f} {delta['l1']:+<10.4f}")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"deep_hierarchy_comparison_{args.model}.json"

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
