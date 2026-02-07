"""
Deep Scaling Test (Depths 5-7)
==============================

Extends rigorous scaling experiment to depths 6-7 with dynamically generated hierarchy.
Uses the same proven methodology but with deeper trees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import random

from multi_model_pipeline import MODELS


@dataclass
class HierarchicalSample:
    text: str
    labels: List[int]


class DeepHierarchyDataset(Dataset):
    """Dataset with dynamically generated deep hierarchy."""

    ROOTS = ["Science", "Arts", "Business", "Technology"]

    def __init__(self, depth: int, samples_per_leaf: int = 30, seed: int = 42):
        self.depth = depth
        random.seed(seed)
        np.random.seed(seed)

        # Build hierarchy tree
        self.paths = self._build_paths()
        self.level_labels = self._build_label_mappings()
        self.samples = self._generate_samples(samples_per_leaf)

    def _build_paths(self) -> List[List[str]]:
        """Build all paths through the hierarchy."""
        paths = [[r] for r in self.ROOTS]

        for level in range(1, self.depth):
            new_paths = []
            branching = 4 if level < 3 else 3  # Reduce branching at deeper levels
            for path in paths:
                for i in range(branching):
                    child = f"{path[-1][:3]}_L{level}_{i}"
                    new_paths.append(path + [child])
            paths = new_paths

        return paths

    def _build_label_mappings(self) -> List[dict]:
        level_labels = []
        for level in range(self.depth):
            unique = sorted(set(p[level] for p in self.paths))
            level_labels.append({lab: idx for idx, lab in enumerate(unique)})
        return level_labels

    def _generate_samples(self, samples_per_leaf: int) -> List[HierarchicalSample]:
        samples = []

        for path in self.paths:
            for _ in range(samples_per_leaf):
                # Generate text with path-specific keywords
                words = [path[0].lower(), path[-1].split('_')[0].lower()]

                # Add noise
                noise = ["the", "is", "a", "for", "study", "analysis", "research"]
                words.extend(random.sample(noise, min(4, len(noise))))

                # Add confounders from other paths
                other_path = random.choice(self.paths)
                if other_path != path:
                    words.append(other_path[-1].split('_')[0].lower())

                random.shuffle(words)
                text = " ".join(words)
                labels = [self.level_labels[i][path[i]] for i in range(self.depth)]

                samples.append(HierarchicalSample(text=text, labels=labels))

        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def num_classes_per_level(self) -> List[int]:
        return [len(m) for m in self.level_labels]


# ============== CLASSIFIERS ==============

class FlatClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: List[int]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleList([nn.Linear(512, nc) for nc in num_classes])

    def forward(self, x):
        shared = self.shared(x)
        return [head(shared) for head in self.heads]


class FractalClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: List[int], proj_dim: int = 256):
        super().__init__()
        self.num_levels = len(num_classes)
        self.proj_dim = proj_dim

        self.fractal_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        self.level_projs = nn.ModuleList()
        self.level_heads = nn.ModuleList()

        for i, nc in enumerate(num_classes):
            input_dim = hidden_dim // 2 + (proj_dim if i > 0 else 0)
            self.level_projs.append(nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.LayerNorm(proj_dim),
            ))
            self.level_heads.append(nn.Linear(proj_dim, nc))

    def forward(self, x):
        shared = self.fractal_block(x)
        logits = []
        prev_emb = None

        for i in range(self.num_levels):
            if prev_emb is not None:
                level_input = torch.cat([shared, prev_emb], dim=-1)
            else:
                level_input = shared

            level_emb = self.level_projs[i](level_input)
            level_emb = F.normalize(level_emb, dim=-1)
            level_logits = self.level_heads[i](level_emb)

            logits.append(level_logits)
            prev_emb = level_emb

        return logits


# ============== TRAINING ==============

def encode_batch(backbone, tokenizer, texts, device):
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = backbone(**inputs)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state[:, -1, :]
        else:
            hidden = outputs[0][:, -1, :]
    return hidden.float()


def train_epoch(model, backbone, tokenizer, dataset, optimizer, device, batch_size):
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    ce = nn.CrossEntropyLoss()
    total_loss = 0

    for batch in dataloader:
        texts = [item.text for item in batch]
        labels = [torch.tensor([item.labels[i] for item in batch], device=device)
                  for i in range(dataset.depth)]

        optimizer.zero_grad()
        hidden = encode_batch(backbone, tokenizer, texts, device)
        logits = model(hidden)

        loss = sum(ce(l, lab) for l, lab in zip(logits, labels))
        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, backbone, tokenizer, dataset, device, batch_size) -> Dict:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_preds = [[] for _ in range(dataset.depth)]
    all_labels = [[] for _ in range(dataset.depth)]

    for batch in dataloader:
        texts = [item.text for item in batch]
        for i in range(dataset.depth):
            all_labels[i].extend([item.labels[i] for item in batch])

        hidden = encode_batch(backbone, tokenizer, texts, device)
        logits = model(hidden)

        for i, l in enumerate(logits):
            all_preds[i].extend(l.argmax(dim=1).cpu().tolist())

    # Per-level accuracy
    results = {}
    for i in range(dataset.depth):
        acc = np.mean([p == l for p, l in zip(all_preds[i], all_labels[i])])
        results[f'l{i}_acc'] = float(acc)

    # Hierarchical accuracy
    n = len(all_preds[0])
    hier_correct = [all(all_preds[i][j] == all_labels[i][j] for i in range(dataset.depth)) for j in range(n)]
    results['hier_acc'] = float(np.mean(hier_correct))

    return results


def run_single_seed(depth, seed, backbone, tokenizer, device, epochs=6, batch_size=32):
    """Run one seed and return results."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data = DeepHierarchyDataset(depth=depth, samples_per_leaf=30, seed=seed)
    val_data = DeepHierarchyDataset(depth=depth, samples_per_leaf=10, seed=seed + 1000)

    num_classes = train_data.num_classes_per_level
    hidden_dim = backbone.config.hidden_size

    results = {'seed': seed, 'depth': depth}

    for name, ClsCls in [('flat', FlatClassifier), ('fractal', FractalClassifier)]:
        model = ClsCls(hidden_dim, num_classes).to(device)
        optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

        best = None
        for ep in range(1, epochs + 1):
            train_epoch(model, backbone, tokenizer, train_data, optimizer, device, batch_size)
            ev = evaluate(model, backbone, tokenizer, val_data, device, batch_size)
            if best is None or ev['hier_acc'] > best['hier_acc']:
                best = ev.copy()
                best['epoch'] = ep

        results[name] = best
        del model
        torch.cuda.empty_cache()

    return results


def compute_statistics(results_list, classifier, metric):
    """Compute mean and 95% CI."""
    values = [r[classifier][metric] for r in results_list if classifier in r and metric in r[classifier]]
    if not values:
        return {'mean': 0, 'ci_95': 0, 'values': []}

    mean = np.mean(values)
    if len(values) > 1:
        sem = np.std(values, ddof=1) / np.sqrt(len(values))
        ci_95 = 1.96 * sem
    else:
        ci_95 = 0

    return {'mean': float(mean), 'ci_95': float(ci_95), 'values': [float(v) for v in values]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument("--depths", type=int, nargs='+', default=[5, 6, 7])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Depths: {args.depths}")

    # Load backbone
    print(f"\nLoading backbone: {args.model}...")
    model_config = MODELS[args.model]

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_path, trust_remote_code=model_config.trust_remote_code)
    if model_config.pooling == "last":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(
        model_config.hf_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16,
    ).to(device)

    for p in backbone.parameters():
        p.requires_grad = False

    print("Model loaded successfully!")

    # Run experiments
    all_results = {}

    for depth in args.depths:
        print(f"\n{'='*60}")
        print(f"DEPTH: {depth}")
        print(f"{'='*60}")

        depth_results = []
        for seed in tqdm(args.seeds, desc=f"Seeds for depth {depth}"):
            result = run_single_seed(depth, seed, backbone, tokenizer, device, args.epochs, args.batch_size)
            depth_results.append(result)

        all_results[f'depth_{depth}'] = {
            'raw': depth_results,
            'statistics': {}
        }

        for clf in ['flat', 'fractal']:
            all_results[f'depth_{depth}']['statistics'][clf] = {}
            metrics = ['hier_acc'] + [f'l{i}_acc' for i in range(depth)]
            for metric in metrics:
                all_results[f'depth_{depth}']['statistics'][clf][metric] = compute_statistics(depth_results, clf, metric)

    # Print summary
    print("\n" + "="*80)
    print("DEEP SCALING SUMMARY (with 95% CI)")
    print("="*80)

    print(f"\n{'Depth':<8} {'Flat Hier Acc':<22} {'Fractal Hier Acc':<22} {'Delta':<15}")
    print("-" * 70)

    for depth in args.depths:
        stats = all_results[f'depth_{depth}']['statistics']
        flat = stats['flat']['hier_acc']
        frac = stats['fractal']['hier_acc']
        delta = frac['mean'] - flat['mean']

        print(f"{depth:<8} {flat['mean']:.4f} +/- {flat['ci_95']:.4f}       {frac['mean']:.4f} +/- {frac['ci_95']:.4f}       {delta:+.4f}")

    # Statistical significance
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE (paired t-test)")
    print("="*80)

    from scipy.stats import ttest_rel
    for depth in args.depths:
        stats = all_results[f'depth_{depth}']['statistics']
        frac_vals = stats['fractal']['hier_acc']['values']
        flat_vals = stats['flat']['hier_acc']['values']

        if len(frac_vals) > 1:
            t_stat, p_value = ttest_rel(frac_vals, flat_vals)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"Depth {depth}: t = {t_stat:.3f}, p = {p_value:.4f} {sig}")
        else:
            print(f"Depth {depth}: Insufficient seeds for statistical test")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"deep_scaling_{args.model}.json"
    with open(results_path, "w") as f:
        json.dump({
            'model': args.model,
            'depths': args.depths,
            'seeds': args.seeds,
            'results': all_results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
