"""
Ablation Experiments for Fractal Classifiers
==============================================

Ablation experiments:
1. Noisy hierarchy (corrupt parent labels)
2. Low-data regime (reduced training data)
3. Label imbalance (rare leaves)
4. Depth stress-test (go deeper than 5)
5. Error analysis (where does fractal win?)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
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
    clean_labels: List[int]  # Original labels before noise


class NoisyHierarchyDataset(Dataset):
    """Dataset with controllable label noise at each hierarchy level."""

    HIERARCHY = {
        "Science": ["Physics", "Chemistry", "Biology", "Mathematics"],
        "Arts": ["Music", "Painting", "Literature", "Film"],
        "Business": ["Finance", "Marketing", "Management", "Accounting"],
        "Technology": ["Software", "Hardware", "Networks", "AI"],
        "Physics": ["Quantum", "Classical", "Thermodynamics", "Optics"],
        "Chemistry": ["Organic", "Inorganic", "Biochemistry", "Physical"],
        "AI": ["MachineLearning", "NLP", "Vision", "Robotics"],
        "MachineLearning": ["Supervised", "Unsupervised", "Reinforcement", "Deep"],
    }

    def __init__(self, depth: int, noise_rate: float = 0.0, samples_per_leaf: int = 50, seed: int = 42):
        self.depth = depth
        self.noise_rate = noise_rate
        random.seed(seed)
        np.random.seed(seed)

        self.paths = self._build_paths(depth)
        self.level_labels = self._build_label_mappings()
        self.samples = self._generate_samples(samples_per_leaf)

    def _build_paths(self, depth: int) -> List[List[str]]:
        roots = ["Science", "Arts", "Business", "Technology"]
        paths = [[r] for r in roots]

        for level in range(1, depth):
            new_paths = []
            for path in paths:
                last_node = path[-1]
                if last_node in self.HIERARCHY:
                    for child in self.HIERARCHY[last_node]:
                        new_paths.append(path + [child])
                else:
                    n_children = 4 if level < 3 else 3
                    for i in range(n_children):
                        new_paths.append(path + [f"{last_node}_{level}_{i}"])
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
                # Generate clean labels
                clean_labels = [self.level_labels[i][path[i]] for i in range(self.depth)]

                # Apply noise to labels (randomly flip to different class at each level)
                noisy_labels = clean_labels.copy()
                for level in range(self.depth):
                    if random.random() < self.noise_rate:
                        # Flip to random different class
                        n_classes = len(self.level_labels[level])
                        new_label = random.randint(0, n_classes - 1)
                        while new_label == clean_labels[level] and n_classes > 1:
                            new_label = random.randint(0, n_classes - 1)
                        noisy_labels[level] = new_label

                # Generate text based on CLEAN path (signal matches clean, not noisy)
                text = self._generate_text(path)
                samples.append(HierarchicalSample(
                    text=text,
                    labels=noisy_labels,
                    clean_labels=clean_labels
                ))

        random.shuffle(samples)
        return samples

    def _generate_text(self, path: List[str]) -> str:
        words = [path[0].lower(), path[-1].lower().split('_')[0]]
        noise = ["study", "research", "analysis", "paper", "work", "topic", "area"]
        words.extend(random.sample(noise, 4))
        random.shuffle(words)
        return " ".join(words)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def num_classes_per_level(self) -> List[int]:
        return [len(m) for m in self.level_labels]


class ImbalancedHierarchyDataset(Dataset):
    """Dataset with long-tail label distribution."""

    HIERARCHY = {
        "Science": ["Physics", "Chemistry", "Biology", "Mathematics"],
        "Arts": ["Music", "Painting", "Literature", "Film"],
        "Business": ["Finance", "Marketing", "Management", "Accounting"],
        "Technology": ["Software", "Hardware", "Networks", "AI"],
    }

    def __init__(self, depth: int, imbalance_factor: float = 0.1, seed: int = 42):
        """
        imbalance_factor: Ratio of samples for rarest class vs most common.
        E.g., 0.1 means rarest class has 10% the samples of most common.
        """
        self.depth = depth
        self.imbalance_factor = imbalance_factor
        random.seed(seed)
        np.random.seed(seed)

        self.paths = self._build_paths(depth)
        self.level_labels = self._build_label_mappings()
        self.samples = self._generate_imbalanced_samples()

    def _build_paths(self, depth: int) -> List[List[str]]:
        roots = ["Science", "Arts", "Business", "Technology"]
        paths = [[r] for r in roots]

        for level in range(1, depth):
            new_paths = []
            for path in paths:
                last_node = path[-1]
                if last_node in self.HIERARCHY:
                    for child in self.HIERARCHY[last_node]:
                        new_paths.append(path + [child])
                else:
                    for i in range(4):
                        new_paths.append(path + [f"{last_node}_{level}_{i}"])
            paths = new_paths

        return paths

    def _build_label_mappings(self) -> List[dict]:
        level_labels = []
        for level in range(self.depth):
            unique = sorted(set(p[level] for p in self.paths))
            level_labels.append({lab: idx for idx, lab in enumerate(unique)})
        return level_labels

    def _generate_imbalanced_samples(self) -> List[HierarchicalSample]:
        samples = []
        n_paths = len(self.paths)

        # Create exponential distribution of samples per path
        # First path gets 100 samples, last gets 100 * imbalance_factor
        max_samples = 100
        min_samples = int(max_samples * self.imbalance_factor)

        for i, path in enumerate(self.paths):
            # Linear interpolation for sample count
            ratio = i / max(1, n_paths - 1)
            n_samples = int(max_samples - ratio * (max_samples - min_samples))

            for _ in range(n_samples):
                labels = [self.level_labels[j][path[j]] for j in range(self.depth)]
                text = " ".join([path[0].lower(), path[-1].lower().split('_')[0],
                                "study", "research", "work"])
                samples.append(HierarchicalSample(text=text, labels=labels, clean_labels=labels))

        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def num_classes_per_level(self) -> List[int]:
        return [len(m) for m in self.level_labels]


# ============== CLASSIFIERS (same as before) ==============

class FlatClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: List[int]):
        super().__init__()
        self.num_levels = len(num_classes)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleList([nn.Linear(512, num_c) for num_c in num_classes])

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

        for i, num_c in enumerate(num_classes):
            input_dim = hidden_dim // 2 + (proj_dim if i > 0 else 0)
            self.level_projs.append(nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.LayerNorm(proj_dim),
            ))
            self.level_heads.append(nn.Linear(proj_dim, num_c))

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

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, backbone, tokenizer, dataset, device, batch_size, use_clean_labels=False) -> Dict:
    """Evaluate with option to use clean labels (for noisy experiments)."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_preds = [[] for _ in range(dataset.depth)]
    all_labels = [[] for _ in range(dataset.depth)]

    for batch in dataloader:
        texts = [item.text for item in batch]
        for i in range(dataset.depth):
            if use_clean_labels:
                all_labels[i].extend([item.clean_labels[i] for item in batch])
            else:
                all_labels[i].extend([item.labels[i] for item in batch])

        hidden = encode_batch(backbone, tokenizer, texts, device)
        logits = model(hidden)

        for i, l in enumerate(logits):
            all_preds[i].extend(l.argmax(dim=1).cpu().tolist())

    results = {}
    for i in range(dataset.depth):
        acc = np.mean([p == l for p, l in zip(all_preds[i], all_labels[i])])
        results[f'l{i}_acc'] = float(acc)

    n = len(all_preds[0])
    hier_correct = [all(all_preds[i][j] == all_labels[i][j] for i in range(dataset.depth)) for j in range(n)]
    results['hier_acc'] = float(np.mean(hier_correct))

    return results


def run_noisy_experiment(backbone, tokenizer, device, depth=3, epochs=8, batch_size=32):
    """Test robustness to label noise."""
    print("\n" + "="*60)
    print(f"NOISY HIERARCHY EXPERIMENT (Depth {depth})")
    print("="*60)

    noise_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    results = {}

    for noise in noise_rates:
        print(f"\n--- Noise Rate: {noise*100:.0f}% ---")
        train_data = NoisyHierarchyDataset(depth=depth, noise_rate=noise, samples_per_leaf=50, seed=42)
        val_data = NoisyHierarchyDataset(depth=depth, noise_rate=0.0, samples_per_leaf=15, seed=43)

        num_classes = train_data.num_classes_per_level
        hidden_dim = backbone.config.hidden_size

        results[f'noise_{noise}'] = {}

        for name, ClsCls in [('flat', FlatClassifier), ('fractal', FractalClassifier)]:
            model = ClsCls(hidden_dim, num_classes).to(device)
            optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

            best = None
            for ep in range(1, epochs + 1):
                train_epoch(model, backbone, tokenizer, train_data, optimizer, device, batch_size)
                ev = evaluate(model, backbone, tokenizer, val_data, device, batch_size, use_clean_labels=True)
                if best is None or ev['hier_acc'] > best['hier_acc']:
                    best = ev.copy()

            results[f'noise_{noise}'][name] = best
            print(f"  {name}: hier_acc={best['hier_acc']:.4f}")

            del model
            torch.cuda.empty_cache()

    return results


def run_low_data_experiment(backbone, tokenizer, device, depth=3, epochs=8, batch_size=32):
    """Test performance with limited training data."""
    print("\n" + "="*60)
    print(f"LOW-DATA REGIME EXPERIMENT (Depth {depth})")
    print("="*60)

    data_fractions = [0.1, 0.25, 0.5, 1.0]
    results = {}

    # Generate full dataset
    full_train = NoisyHierarchyDataset(depth=depth, noise_rate=0.0, samples_per_leaf=100, seed=42)
    val_data = NoisyHierarchyDataset(depth=depth, noise_rate=0.0, samples_per_leaf=20, seed=43)

    for frac in data_fractions:
        print(f"\n--- Data Fraction: {frac*100:.0f}% ---")

        # Subsample training data
        n_samples = int(len(full_train) * frac)
        indices = random.sample(range(len(full_train)), n_samples)
        train_subset = Subset(full_train, indices)

        num_classes = full_train.num_classes_per_level
        hidden_dim = backbone.config.hidden_size

        results[f'frac_{frac}'] = {}

        for name, ClsCls in [('flat', FlatClassifier), ('fractal', FractalClassifier)]:
            model = ClsCls(hidden_dim, num_classes).to(device)
            optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

            # Create dataloader from subset
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                     drop_last=True, collate_fn=lambda x: [full_train.samples[i.item() if hasattr(i, 'item') else i] for i in x] if isinstance(x[0], int) else x)

            best = None
            for ep in range(1, epochs + 1):
                # Manual training loop for subset
                model.train()
                ce = nn.CrossEntropyLoss()
                for batch_indices in train_loader:
                    if isinstance(batch_indices[0], int):
                        batch = [full_train.samples[i] for i in batch_indices]
                    else:
                        batch = batch_indices

                    texts = [item.text for item in batch]
                    labels = [torch.tensor([item.labels[i] for item in batch], device=device)
                              for i in range(depth)]

                    optimizer.zero_grad()
                    hidden = encode_batch(backbone, tokenizer, texts, device)
                    logits = model(hidden)

                    loss = sum(ce(l, lab) for l, lab in zip(logits, labels))
                    if not torch.isnan(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                ev = evaluate(model, backbone, tokenizer, val_data, device, batch_size)
                if best is None or ev['hier_acc'] > best['hier_acc']:
                    best = ev.copy()

            results[f'frac_{frac}'][name] = best
            print(f"  {name}: hier_acc={best['hier_acc']:.4f}")

            del model
            torch.cuda.empty_cache()

    return results


def run_depth_stress_test(backbone, tokenizer, device, max_depth=7, epochs=6, batch_size=32):
    """Test very deep hierarchies (beyond 5 levels)."""
    print("\n" + "="*60)
    print(f"DEPTH STRESS TEST (up to {max_depth} levels)")
    print("="*60)

    results = {}

    for depth in range(2, max_depth + 1):
        print(f"\n--- Depth: {depth} ---")
        train_data = NoisyHierarchyDataset(depth=depth, noise_rate=0.0, samples_per_leaf=30, seed=42)
        val_data = NoisyHierarchyDataset(depth=depth, noise_rate=0.0, samples_per_leaf=10, seed=43)

        num_classes = train_data.num_classes_per_level
        hidden_dim = backbone.config.hidden_size

        print(f"  Classes per level: {num_classes}")
        print(f"  Train samples: {len(train_data)}, Val samples: {len(val_data)}")

        results[f'depth_{depth}'] = {}

        for name, ClsCls in [('flat', FlatClassifier), ('fractal', FractalClassifier)]:
            model = ClsCls(hidden_dim, num_classes).to(device)
            optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

            best = None
            for ep in range(1, epochs + 1):
                train_epoch(model, backbone, tokenizer, train_data, optimizer, device, batch_size)
                ev = evaluate(model, backbone, tokenizer, val_data, device, batch_size)
                if best is None or ev['hier_acc'] > best['hier_acc']:
                    best = ev.copy()

            results[f'depth_{depth}'][name] = best
            print(f"  {name}: hier_acc={best['hier_acc']:.4f}")

            del model
            torch.cuda.empty_cache()

        delta = results[f'depth_{depth}']['fractal']['hier_acc'] - results[f'depth_{depth}']['flat']['hier_acc']
        print(f"  Delta: {delta:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--experiment", type=str, choices=['noisy', 'lowdata', 'depth', 'all'], default='all')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load backbone
    print(f"\nLoading backbone: {args.model}...")
    model_config = MODELS[args.model]

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

    all_results = {}

    if args.experiment in ['noisy', 'all']:
        all_results['noisy'] = run_noisy_experiment(backbone, tokenizer, device)

    if args.experiment in ['lowdata', 'all']:
        all_results['lowdata'] = run_low_data_experiment(backbone, tokenizer, device)

    if args.experiment in ['depth', 'all']:
        all_results['depth_stress'] = run_depth_stress_test(backbone, tokenizer, device)

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"ablation_{args.model}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "="*80)
    print("ABLATION EXPERIMENT SUMMARY")
    print("="*80)

    if 'noisy' in all_results:
        print("\nNOISY HIERARCHY (0% to 50% label noise):")
        for noise_key, noise_results in all_results['noisy'].items():
            delta = noise_results['fractal']['hier_acc'] - noise_results['flat']['hier_acc']
            print(f"  {noise_key}: Flat={noise_results['flat']['hier_acc']:.4f}, Fractal={noise_results['fractal']['hier_acc']:.4f}, Δ={delta:+.4f}")

    if 'lowdata' in all_results:
        print("\nLOW-DATA REGIME (10% to 100% training data):")
        for frac_key, frac_results in all_results['lowdata'].items():
            delta = frac_results['fractal']['hier_acc'] - frac_results['flat']['hier_acc']
            print(f"  {frac_key}: Flat={frac_results['flat']['hier_acc']:.4f}, Fractal={frac_results['fractal']['hier_acc']:.4f}, Δ={delta:+.4f}")

    if 'depth_stress' in all_results:
        print("\nDEPTH STRESS TEST:")
        for depth_key, depth_results in all_results['depth_stress'].items():
            delta = depth_results['fractal']['hier_acc'] - depth_results['flat']['hier_acc']
            print(f"  {depth_key}: Flat={depth_results['flat']['hier_acc']:.4f}, Fractal={depth_results['fractal']['hier_acc']:.4f}, Δ={delta:+.4f}")


if __name__ == "__main__":
    main()
