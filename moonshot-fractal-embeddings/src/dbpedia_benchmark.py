"""
DBPedia-14 Benchmark
====================

DBPedia-14 has 14 classes that can be grouped into higher-level ontologies:
- Person: Artist, Athlete, OfficeHolder
- Place: Building, NaturalPlace, Village, MeanOfTransportation
- Work: Film, WrittenWork, Album
- Organization: Company, EducationalInstitution
- Biology: Animal, Plant

This creates a 5 super-category, 14 sub-category hierarchy.
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
from datasets import load_dataset

from multi_model_pipeline import MODELS


# DBPedia hierarchy mapping
# Original class indices -> class names
CLASS_NAMES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
]

# Super-categories
HIERARCHY = {
    "Organization": ["Company", "EducationalInstitution"],
    "Person": ["Artist", "Athlete", "OfficeHolder"],
    "Place": ["MeanOfTransportation", "Building", "NaturalPlace", "Village"],
    "Biology": ["Animal", "Plant"],
    "Work": ["Album", "Film", "WrittenWork"],
}

SUPER_TO_IDX = {s: i for i, s in enumerate(HIERARCHY.keys())}
CLASS_TO_SUPER = {}
for super_cat, classes in HIERARCHY.items():
    for cls in classes:
        CLASS_TO_SUPER[cls] = super_cat


@dataclass
class HierarchicalSample:
    text: str
    labels: List[int]


class DBPediaDataset(Dataset):
    """DBPedia-14 with 2-level hierarchy."""

    def __init__(self, split: str = "train", max_samples: int = 10000, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        print(f"Loading DBPedia-14 ({split})...")
        ds = load_dataset("fancyzhx/dbpedia_14", split=split)

        self.samples = []
        for item in ds:
            if len(self.samples) >= max_samples:
                break

            text = item['content'][:512]
            sub_idx = item['label']  # 0-13

            class_name = CLASS_NAMES[sub_idx]
            super_cat = CLASS_TO_SUPER[class_name]
            super_idx = SUPER_TO_IDX[super_cat]

            self.samples.append(HierarchicalSample(
                text=text,
                labels=[super_idx, sub_idx]
            ))

        random.shuffle(self.samples)
        print(f"  Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def num_classes_per_level(self) -> List[int]:
        return [len(HIERARCHY), 14]

    @property
    def depth(self):
        return 2


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

    results = {}
    for i in range(dataset.depth):
        acc = np.mean([p == l for p, l in zip(all_preds[i], all_labels[i])])
        results[f'l{i}_acc'] = float(acc)

    n = len(all_preds[0])
    hier_correct = [all(all_preds[i][j] == all_labels[i][j] for i in range(dataset.depth)) for j in range(n)]
    results['hier_acc'] = float(np.mean(hier_correct))

    return results


def run_single_seed(seed, backbone, tokenizer, device, epochs=8, batch_size=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data = DBPediaDataset(split="train", max_samples=10000, seed=seed)
    val_data = DBPediaDataset(split="test", max_samples=2000, seed=seed + 1000)

    num_classes = train_data.num_classes_per_level
    hidden_dim = backbone.config.hidden_size

    results = {'seed': seed}

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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")

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

    print("Model loaded!")

    # Run experiments
    print("\n" + "="*60)
    print("DBPEDIA-14 BENCHMARK (2-level hierarchy)")
    print("5 super-categories (Org, Person, Place, Biology, Work)")
    print("14 sub-categories")
    print("="*60)

    all_results = []
    for seed in tqdm(args.seeds, desc="Seeds"):
        result = run_single_seed(seed, backbone, tokenizer, device, args.epochs, args.batch_size)
        all_results.append(result)

    # Compute statistics
    stats = {
        'flat': {},
        'fractal': {}
    }
    for clf in ['flat', 'fractal']:
        for metric in ['hier_acc', 'l0_acc', 'l1_acc']:
            stats[clf][metric] = compute_statistics(all_results, clf, metric)

    # Print summary
    print("\n" + "="*60)
    print("DBPEDIA-14 RESULTS")
    print("="*60)

    print(f"\n{'Classifier':<15} {'Super (L0)':<20} {'Sub (L1)':<20} {'Hier Acc':<20}")
    print("-"*75)

    for clf in ['flat', 'fractal']:
        l0 = stats[clf]['l0_acc']
        l1 = stats[clf]['l1_acc']
        hier = stats[clf]['hier_acc']
        print(f"{clf:<15} {l0['mean']*100:.2f}% +/- {l0['ci_95']*100:.2f}%   "
              f"{l1['mean']*100:.2f}% +/- {l1['ci_95']*100:.2f}%   "
              f"{hier['mean']*100:.2f}% +/- {hier['ci_95']*100:.2f}%")

    delta = stats['fractal']['hier_acc']['mean'] - stats['flat']['hier_acc']['mean']
    print(f"\nFractal advantage: {delta*100:+.2f}%")

    # Statistical test
    from scipy.stats import ttest_rel
    frac_vals = stats['fractal']['hier_acc']['values']
    flat_vals = stats['flat']['hier_acc']['values']
    if len(frac_vals) > 1:
        t_stat, p_value = ttest_rel(frac_vals, flat_vals)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"Paired t-test: t = {t_stat:.3f}, p = {p_value:.4f} {sig}")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"dbpedia_benchmark_{args.model}.json"
    with open(results_path, "w") as f:
        json.dump({
            'dataset': 'dbpedia_14',
            'hierarchy': '5 super-categories, 14 sub-categories',
            'model': args.model,
            'seeds': args.seeds,
            'statistics': stats,
            'raw_results': all_results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
