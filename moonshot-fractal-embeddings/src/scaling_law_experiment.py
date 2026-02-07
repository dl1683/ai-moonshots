"""
Fractal Scaling Law Experiment
==============================

Test the hypothesis: Fractal embeddings show INCREASING advantage over flat
embeddings as hierarchy depth increases.

Approach:
- Create synthetic hierarchical datasets with varying depths (2, 3, 4, 5 levels)
- Each level has clear semantic structure
- Compare flat vs fractal classifiers across depths
- Plot the scaling law: advantage vs depth

If fractal shows linear or super-linear advantage growth with depth,
this validates the core hypothesis.
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
from typing import List, Tuple
import random

from multi_model_pipeline import MODELS


@dataclass
class HierarchicalSample:
    """Sample with labels at multiple hierarchy levels."""
    text: str
    labels: List[int]  # [l0, l1, l2, ...]


class SyntheticHierarchyDataset(Dataset):
    """
    Synthetic dataset with configurable hierarchy depth.

    Structure at each depth:
    - Level 0: Domain (Science, Arts, Business, Tech)
    - Level 1: Field (Physics, Chemistry, ... under Science)
    - Level 2: Subfield (Quantum, Classical, ... under Physics)
    - Level 3: Topic (Entanglement, Superposition, ... under Quantum)
    - Level 4: Concept (specific terms)

    Each sample is a phrase that contains keywords from each level,
    making classification progressively harder at deeper levels.
    """

    # Hierarchical structure - each level branches into children
    HIERARCHY = {
        # Level 0 -> Level 1 mapping
        "Science": ["Physics", "Chemistry", "Biology", "Mathematics"],
        "Arts": ["Music", "Painting", "Literature", "Film"],
        "Business": ["Finance", "Marketing", "Management", "Accounting"],
        "Technology": ["Software", "Hardware", "Networks", "AI"],

        # Level 1 -> Level 2 mapping
        "Physics": ["Quantum", "Classical", "Thermodynamics", "Optics"],
        "Chemistry": ["Organic", "Inorganic", "Biochemistry", "Physical"],
        "Biology": ["Genetics", "Ecology", "Anatomy", "Microbiology"],
        "Mathematics": ["Algebra", "Calculus", "Statistics", "Geometry"],

        "Music": ["Classical", "Jazz", "Rock", "Electronic"],
        "Painting": ["Renaissance", "Modern", "Impressionist", "Abstract"],
        "Literature": ["Poetry", "Fiction", "Drama", "Essays"],
        "Film": ["Documentary", "Action", "Comedy", "Drama"],

        "Finance": ["Investment", "Banking", "Insurance", "Trading"],
        "Marketing": ["Digital", "Brand", "Research", "Advertising"],
        "Management": ["Strategy", "Operations", "HR", "Leadership"],
        "Accounting": ["Audit", "Tax", "Forensic", "Cost"],

        "Software": ["Web", "Mobile", "Desktop", "Embedded"],
        "Hardware": ["Processors", "Memory", "Storage", "Displays"],
        "Networks": ["Internet", "Wireless", "Security", "Protocols"],
        "AI": ["MachineLearning", "NLP", "Vision", "Robotics"],

        # Level 2 -> Level 3 mapping
        "Quantum": ["Entanglement", "Superposition", "Tunneling", "Decoherence"],
        "Classical": ["Mechanics", "Electromagnetism", "Waves", "Fluids"],
        "Thermodynamics": ["Entropy", "Heat", "Equilibrium", "Phase"],
        "Optics": ["Refraction", "Diffraction", "Lasers", "Holography"],

        "Organic": ["Synthesis", "Reactions", "Polymers", "Spectroscopy"],
        "Inorganic": ["Coordination", "Catalysis", "Materials", "Bioinorganic"],
        "Biochemistry": ["Enzymes", "Metabolism", "Proteins", "DNA"],
        "Physical": ["Kinetics", "Electrochemistry", "Spectroscopy", "Surfaces"],

        # More Level 2 -> Level 3 mappings...
        "MachineLearning": ["Supervised", "Unsupervised", "Reinforcement", "Deep"],
        "NLP": ["Translation", "Sentiment", "QA", "Generation"],
        "Vision": ["Detection", "Recognition", "Segmentation", "Generation"],
        "Robotics": ["Navigation", "Manipulation", "Planning", "Control"],

        # Level 3 -> Level 4 mapping (leaf concepts)
        "Entanglement": ["Bell", "EPR", "Teleportation", "Swapping"],
        "Superposition": ["Qubit", "Coherence", "Interference", "Gates"],
        "Deep": ["CNN", "RNN", "Transformer", "GAN"],
        "Supervised": ["Classification", "Regression", "SVM", "Trees"],
        "Unsupervised": ["Clustering", "PCA", "Autoencoders", "GMM"],
    }

    # Keywords for generating text at each concept
    KEYWORDS = {
        "Science": ["research", "experiment", "hypothesis", "scientific"],
        "Arts": ["creative", "artistic", "aesthetic", "expression"],
        "Business": ["company", "profit", "market", "enterprise"],
        "Technology": ["digital", "innovation", "system", "computing"],

        "Physics": ["force", "energy", "matter", "motion"],
        "Chemistry": ["reaction", "molecule", "compound", "element"],
        "Biology": ["organism", "cell", "evolution", "life"],
        "Mathematics": ["proof", "theorem", "equation", "number"],

        "Quantum": ["quantum", "particle", "wave", "probability"],
        "Classical": ["newton", "motion", "velocity", "acceleration"],
        "MachineLearning": ["model", "training", "prediction", "algorithm"],
        "Deep": ["neural", "network", "layer", "backpropagation"],

        # Add more as needed
    }

    def __init__(self, depth: int, samples_per_leaf: int = 50, seed: int = 42):
        """
        Args:
            depth: Number of hierarchy levels (2-5)
            samples_per_leaf: Number of samples per leaf node
            seed: Random seed for reproducibility
        """
        self.depth = depth
        random.seed(seed)
        np.random.seed(seed)

        # Build hierarchy paths up to specified depth
        self.paths = self._build_paths(depth)

        # Create label mappings for each level
        self.level_labels = self._build_label_mappings()

        # Generate samples
        self.samples = self._generate_samples(samples_per_leaf)

    def _build_paths(self, depth: int) -> List[List[str]]:
        """Build all paths through hierarchy up to depth."""
        # Start with root nodes
        roots = ["Science", "Arts", "Business", "Technology"]
        paths = [[r] for r in roots]

        # Extend paths to desired depth
        for level in range(1, depth):
            new_paths = []
            for path in paths:
                last_node = path[-1]
                if last_node in self.HIERARCHY:
                    children = self.HIERARCHY[last_node]
                    for child in children:
                        new_paths.append(path + [child])
                else:
                    # If no children, repeat last node
                    new_paths.append(path + [last_node])
            paths = new_paths

        return paths

    def _build_label_mappings(self) -> List[dict]:
        """Build label->index mappings for each level."""
        level_labels = []
        for level in range(self.depth):
            unique_labels = sorted(set(path[level] for path in self.paths))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            level_labels.append(label_to_idx)
        return level_labels

    def _get_keywords(self, node: str) -> List[str]:
        """Get keywords for a node."""
        if node in self.KEYWORDS:
            return self.KEYWORDS[node]
        # Default keywords based on node name
        return [node.lower(), f"{node.lower()}s", f"about {node.lower()}"]

    def _generate_samples(self, samples_per_leaf: int) -> List[HierarchicalSample]:
        """Generate training samples with controlled difficulty."""
        samples = []

        # Build noise pool from OTHER hierarchy branches (confounders)
        all_nodes = set()
        for children in self.HIERARCHY.values():
            all_nodes.update(children)
        all_nodes.update(self.HIERARCHY.keys())

        for path in self.paths:
            # Get nodes NOT in this path (confounders)
            path_set = set(path)
            confounders = [n for n in all_nodes if n not in path_set]

            for _ in range(samples_per_leaf):
                # Only use keywords from DEEPEST level (makes upper levels harder)
                deepest_node = path[-1]
                target_keywords = self._get_keywords(deepest_node)

                # Sample 1-2 keywords (sparse signal)
                n_target = random.randint(1, 2)
                chosen_keywords = random.sample(target_keywords, min(n_target, len(target_keywords)))

                # Add LOTS of noise and confounders (makes it hard)
                noise_words = ["the", "a", "is", "about", "study", "analysis", "work", "research",
                              "paper", "article", "topic", "subject", "area", "field", "domain",
                              "concept", "theory", "method", "approach", "technique", "process",
                              "data", "result", "finding", "discussion", "conclusion", "application"]

                # Add confounding keywords from OTHER branches
                confounder_keywords = []
                if confounders:
                    # Pick 2-3 random confounders
                    n_conf = random.randint(2, 3)
                    conf_nodes = random.sample(confounders, min(n_conf, len(confounders)))
                    for cn in conf_nodes:
                        ck = self._get_keywords(cn)
                        confounder_keywords.extend(random.sample(ck, min(1, len(ck))))

                # Assemble with heavy noise ratio
                all_words = (
                    chosen_keywords +  # 1-2 signal words
                    random.sample(noise_words, random.randint(8, 12)) +  # 8-12 noise
                    confounder_keywords  # 2-3 confounding keywords
                )

                random.shuffle(all_words)
                text = " ".join(all_words)

                # Get labels for each level
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
        """Number of classes at each level."""
        return [len(mapping) for mapping in self.level_labels]


class FlatClassifier(nn.Module):
    """Standard flat classifier (baseline)."""

    def __init__(self, hidden_dim: int, num_classes: List[int]):
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

    def __init__(self, hidden_dim: int, num_classes: List[int], proj_dim: int = 256):
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

    def forward(self, x):
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
            prev_emb = level_emb

        return logits


def encode_batch(backbone, tokenizer, texts, device):
    """Encode texts through backbone."""
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = backbone(**inputs)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state[:, -1, :]
        else:
            hidden = outputs[0][:, -1, :]

    return hidden.float()


def train_epoch(model, backbone, tokenizer, dataset, optimizer, device, batch_size):
    """Train for one epoch."""
    model.train()
    backbone.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        texts = [item.text for item in batch]

        # Get labels for all levels
        labels = []
        for level in range(dataset.depth):
            level_labels = torch.tensor(
                [item.labels[level] for item in batch],
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
def evaluate(model, backbone, tokenizer, dataset, device, batch_size):
    """Evaluate on dataset."""
    model.eval()
    backbone.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_preds = [[] for _ in range(dataset.depth)]
    all_labels = [[] for _ in range(dataset.depth)]

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        texts = [item.text for item in batch]

        for level in range(dataset.depth):
            level_labels = [item.labels[level] for item in batch]
            all_labels[level].extend(level_labels)

        hidden = encode_batch(backbone, tokenizer, texts, device)
        logits = model(hidden)

        for level, l in enumerate(logits):
            all_preds[level].extend(l.argmax(dim=1).cpu().tolist())

    results = {}
    for level in range(dataset.depth):
        acc = np.mean([p == l for p, l in zip(all_preds[level], all_labels[level])])
        results[f'l{level}_acc'] = acc

    # Hierarchical accuracy (all levels correct)
    hier_correct = [
        all(all_preds[i][j] == all_labels[i][j] for i in range(dataset.depth))
        for j in range(len(all_preds[0]))
    ]
    results['hier_acc'] = np.mean(hier_correct)

    return results


def run_depth_comparison(depth: int, backbone, tokenizer, device, epochs=8, batch_size=32):
    """Run flat vs fractal comparison at a specific depth."""
    print(f"\n{'='*60}")
    print(f"DEPTH: {depth} levels")
    print(f"{'='*60}")

    # Create datasets
    train_data = SyntheticHierarchyDataset(depth=depth, samples_per_leaf=100, seed=42)
    val_data = SyntheticHierarchyDataset(depth=depth, samples_per_leaf=20, seed=123)

    num_classes = train_data.num_classes_per_level
    print(f"  Classes per level: {num_classes}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    hidden_dim = backbone.config.hidden_size
    results = {}

    # Test Flat Classifier
    print("\n--- FLAT CLASSIFIER ---")
    flat_model = FlatClassifier(hidden_dim, num_classes).to(device)
    flat_optimizer = AdamW(flat_model.parameters(), lr=5e-4, weight_decay=0.01)

    best_flat = None
    for epoch in range(1, epochs + 1):
        train_epoch(flat_model, backbone, tokenizer, train_data, flat_optimizer, device, batch_size)
        eval_results = evaluate(flat_model, backbone, tokenizer, val_data, device, batch_size)

        total_acc = sum(eval_results[f'l{i}_acc'] for i in range(depth))
        if best_flat is None or total_acc > sum(best_flat[f'l{i}_acc'] for i in range(depth)):
            best_flat = eval_results.copy()
            best_flat['epoch'] = epoch

        if epoch % 3 == 0:
            accs = [f"L{i}={eval_results[f'l{i}_acc']:.4f}" for i in range(depth)]
            print(f"  Epoch {epoch}: {', '.join(accs)}")

    results['flat'] = best_flat
    accs = [f"L{i}={best_flat[f'l{i}_acc']:.4f}" for i in range(depth)]
    print(f"  Best: {', '.join(accs)}, Hier={best_flat['hier_acc']:.4f}")

    del flat_model
    torch.cuda.empty_cache()

    # Test Fractal Classifier
    print("\n--- FRACTAL CLASSIFIER ---")
    fractal_model = HierarchicalFractalClassifier(hidden_dim, num_classes).to(device)
    fractal_optimizer = AdamW(fractal_model.parameters(), lr=5e-4, weight_decay=0.01)

    best_fractal = None
    for epoch in range(1, epochs + 1):
        train_epoch(fractal_model, backbone, tokenizer, train_data, fractal_optimizer, device, batch_size)
        eval_results = evaluate(fractal_model, backbone, tokenizer, val_data, device, batch_size)

        total_acc = sum(eval_results[f'l{i}_acc'] for i in range(depth))
        if best_fractal is None or total_acc > sum(best_fractal[f'l{i}_acc'] for i in range(depth)):
            best_fractal = eval_results.copy()
            best_fractal['epoch'] = epoch

        if epoch % 3 == 0:
            accs = [f"L{i}={eval_results[f'l{i}_acc']:.4f}" for i in range(depth)]
            print(f"  Epoch {epoch}: {', '.join(accs)}")

    results['fractal'] = best_fractal
    accs = [f"L{i}={best_fractal[f'l{i}_acc']:.4f}" for i in range(depth)]
    print(f"  Best: {', '.join(accs)}, Hier={best_fractal['hier_acc']:.4f}")

    # Comparison
    print("\n--- COMPARISON ---")
    flat_accs = [f"L{i}={best_flat[f'l{i}_acc']:.4f}" for i in range(depth)]
    frac_accs = [f"L{i}={best_fractal[f'l{i}_acc']:.4f}" for i in range(depth)]
    print(f"  Flat:    {', '.join(flat_accs)}, Hier={best_flat['hier_acc']:.4f}")
    print(f"  Fractal: {', '.join(frac_accs)}, Hier={best_fractal['hier_acc']:.4f}")

    # Compute deltas
    deltas = {f'l{i}': best_fractal[f'l{i}_acc'] - best_flat[f'l{i}_acc'] for i in range(depth)}
    deltas['hier'] = best_fractal['hier_acc'] - best_flat['hier_acc']
    results['delta'] = deltas

    delta_strs = [f"L{i}={deltas[f'l{i}']:+.4f}" for i in range(depth)]
    print(f"  Delta:   {', '.join(delta_strs)}, Hier={deltas['hier']:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_depth", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=5)
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

    # Test across different depths
    all_results = {}
    for depth in range(args.min_depth, args.max_depth + 1):
        results = run_depth_comparison(depth, backbone, tokenizer, device, args.epochs, args.batch_size)
        all_results[f'depth_{depth}'] = results

    # Summary with scaling law
    print("\n" + "="*80)
    print("SCALING LAW SUMMARY: Fractal Advantage vs Hierarchy Depth")
    print("="*80)
    print(f"{'Depth':<8} {'Flat Hier':<12} {'Frac Hier':<12} {'Delta Hier':<12} {'Advantage':<12}")
    print("-" * 56)

    scaling_data = []
    for depth in range(args.min_depth, args.max_depth + 1):
        key = f'depth_{depth}'
        if key not in all_results:
            continue

        flat_hier = all_results[key]['flat']['hier_acc']
        frac_hier = all_results[key]['fractal']['hier_acc']
        delta_hier = all_results[key]['delta']['hier']

        # Relative advantage (%)
        if flat_hier > 0:
            rel_advantage = (delta_hier / flat_hier) * 100
        else:
            rel_advantage = 0

        scaling_data.append({
            'depth': depth,
            'flat_hier': flat_hier,
            'frac_hier': frac_hier,
            'delta_hier': delta_hier,
            'rel_advantage': rel_advantage
        })

        print(f"{depth:<8} {flat_hier:<12.4f} {frac_hier:<12.4f} {delta_hier:+<12.4f} {rel_advantage:+.2f}%")

    # Save results
    all_results['scaling_law'] = scaling_data

    results_path = Path(__file__).parent.parent / "results" / f"scaling_law_{args.model}.json"

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

    # Print scaling insight
    if len(scaling_data) >= 2:
        first_adv = scaling_data[0]['rel_advantage']
        last_adv = scaling_data[-1]['rel_advantage']
        if last_adv > first_adv:
            print(f"\n{'*'*60}")
            print("SCALING LAW CONFIRMED!")
            print(f"Fractal advantage increases from {first_adv:+.2f}% to {last_adv:+.2f}%")
            print(f"as hierarchy depth increases from {args.min_depth} to {args.max_depth}!")
            print(f"{'*'*60}")
        else:
            print(f"\n{'*'*60}")
            print("Scaling law NOT confirmed - fractal advantage does not increase with depth")
            print(f"{'*'*60}")


if __name__ == "__main__":
    main()
