"""
Fast Hierarchy Randomization Experiment
========================================

Optimized version that pre-computes backbone embeddings ONCE,
then only trains the lightweight classifier heads.

This is ~10x faster than re-encoding texts for each run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import json
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import ttest_rel
from datasets import load_dataset

from multi_model_pipeline import MODELS


# ============== DATA STRUCTURES ==============

@dataclass
class ExperimentConfig:
    dataset: str = "20newsgroups"
    k_randomizations: int = 30
    s_seeds: int = 3
    base_seed: int = 1337


@dataclass
class TrainRunResult:
    seed: int
    metrics: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RandomizationResult:
    rand_id: int
    rand_seed: int
    mapping: Dict[int, int]
    group_sizes: Dict[int, int]
    train_runs: List[TrainRunResult]
    aggregate: Dict[str, float]


# ============== 20 NEWSGROUPS HIERARCHY ==============

HIERARCHY = {
    "comp": ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
             "comp.sys.mac.hardware", "comp.windows.x"],
    "rec": ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
    "sci": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
    "misc": ["misc.forsale"],
    "talk": ["talk.politics.misc", "talk.politics.guns", "talk.politics.mideast", "talk.religion.misc"],
    "soc": ["alt.atheism", "soc.religion.christian"],
}

SUPER_TO_IDX = {s: i for i, s in enumerate(HIERARCHY.keys())}
SUB_TO_SUPER = {}
SUB_TO_IDX = {}

idx = 0
for super_cat, subs in HIERARCHY.items():
    for sub in subs:
        SUB_TO_SUPER[sub] = super_cat
        SUB_TO_IDX[sub] = idx
        idx += 1

NUM_SUPER = len(HIERARCHY)
NUM_SUB = len(SUB_TO_IDX)


def extract_leaf_parent_mapping() -> Dict[int, int]:
    mapping = {}
    for sub_name, sub_idx in SUB_TO_IDX.items():
        super_name = SUB_TO_SUPER[sub_name]
        super_idx = SUPER_TO_IDX[super_name]
        mapping[sub_idx] = super_idx
    return mapping


def level_preserving_parent_shuffle(
    rng: np.random.Generator,
    leaf_to_parent: Dict[int, int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    parents = sorted(set(leaf_to_parent.values()))
    group_sizes = {}
    for parent in parents:
        group_sizes[parent] = sum(1 for p in leaf_to_parent.values() if p == parent)

    leaves = sorted(leaf_to_parent.keys())
    shuffled_leaves = leaves.copy()
    rng.shuffle(shuffled_leaves)

    new_mapping = {}
    i = 0
    for parent in parents:
        size = group_sizes[parent]
        for leaf in shuffled_leaves[i:i + size]:
            new_mapping[leaf] = parent
        i += size

    return new_mapping, group_sizes


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


# ============== PRE-COMPUTE EMBEDDINGS ==============

def load_and_encode_data(backbone, tokenizer, device, max_train=8000, max_test=2000):
    """Load data and pre-compute ALL embeddings once."""

    print("Loading 20 Newsgroups and pre-computing embeddings...")

    def encode_texts(texts, desc="Encoding"):
        embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True,
                               max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = backbone(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state[:, -1, :]
                else:
                    hidden = outputs[0][:, -1, :]
                embeddings.append(hidden.float().cpu())  # Convert to float32
        return torch.cat(embeddings, dim=0)

    # Load train data
    train_ds = load_dataset("SetFit/20_newsgroups", split="train")
    train_texts, train_sub_labels = [], []
    for item in train_ds:
        if len(train_texts) >= max_train:
            break
        label_text = item['label_text']
        if label_text in SUB_TO_IDX:
            train_texts.append(item['text'][:512])
            train_sub_labels.append(SUB_TO_IDX[label_text])

    # Load test data
    test_ds = load_dataset("SetFit/20_newsgroups", split="test")
    test_texts, test_sub_labels = [], []
    for item in test_ds:
        if len(test_texts) >= max_test:
            break
        label_text = item['label_text']
        if label_text in SUB_TO_IDX:
            test_texts.append(item['text'][:512])
            test_sub_labels.append(SUB_TO_IDX[label_text])

    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Encode all texts ONCE
    train_embeddings = encode_texts(train_texts, "Encoding train")
    test_embeddings = encode_texts(test_texts, "Encoding test")

    train_sub_labels = torch.tensor(train_sub_labels)
    test_sub_labels = torch.tensor(test_sub_labels)

    return {
        'train_emb': train_embeddings,
        'train_sub': train_sub_labels,
        'test_emb': test_embeddings,
        'test_sub': test_sub_labels,
        'hidden_dim': train_embeddings.shape[1],
    }


def get_labels_with_mapping(sub_labels: torch.Tensor, leaf_to_parent: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert sub-category labels to (super, sub) using the given mapping."""
    super_labels = torch.tensor([leaf_to_parent[int(s)] for s in sub_labels])
    return super_labels, sub_labels


# ============== FAST TRAINING (on pre-computed embeddings) ==============

def train_classifier_fast(
    model: nn.Module,
    train_emb: torch.Tensor,
    train_super: torch.Tensor,
    train_sub: torch.Tensor,
    test_emb: torch.Tensor,
    test_super: torch.Tensor,
    test_sub: torch.Tensor,
    device: str,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 5e-4,
) -> Dict[str, float]:
    """Train classifier on pre-computed embeddings (FAST)."""

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    # Create data loaders
    train_dataset = TensorDataset(train_emb, train_super, train_sub)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(test_emb, test_super, test_sub)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_result = None

    for epoch in range(epochs):
        # Train
        model.train()
        for emb, sup, sub in train_loader:
            emb, sup, sub = emb.to(device), sup.to(device), sub.to(device)
            optimizer.zero_grad()
            logits = model(emb)
            loss = ce(logits[0], sup) + ce(logits[1], sub)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Eval
        model.eval()
        all_preds_super, all_preds_sub = [], []
        all_labels_super, all_labels_sub = [], []

        with torch.no_grad():
            for emb, sup, sub in test_loader:
                emb = emb.to(device)
                logits = model(emb)
                all_preds_super.extend(logits[0].argmax(dim=1).cpu().tolist())
                all_preds_sub.extend(logits[1].argmax(dim=1).cpu().tolist())
                all_labels_super.extend(sup.tolist())
                all_labels_sub.extend(sub.tolist())

        l0_acc = np.mean([p == l for p, l in zip(all_preds_super, all_labels_super)])
        l1_acc = np.mean([p == l for p, l in zip(all_preds_sub, all_labels_sub)])
        hier_acc = np.mean([
            all_preds_super[i] == all_labels_super[i] and all_preds_sub[i] == all_labels_sub[i]
            for i in range(len(all_preds_super))
        ])

        result = {'l0_acc': float(l0_acc), 'l1_acc': float(l1_acc), 'hier_acc': float(hier_acc), 'epoch': epoch + 1}

        if best_result is None or result['hier_acc'] > best_result['hier_acc']:
            best_result = result

    return best_result


# ============== AGGREGATION ==============

def aggregate_train_runs(train_runs: List[TrainRunResult]) -> Dict[str, float]:
    if not train_runs:
        return {}
    metric_names = list(train_runs[0].metrics.keys())
    aggregate = {}
    for metric in metric_names:
        values = [run.metrics[metric] for run in train_runs]
        aggregate[f'{metric}.mean'] = float(np.mean(values))
        aggregate[f'{metric}.std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return aggregate


def summarize_results(
    randomizations: List[RandomizationResult],
    true_results: List[TrainRunResult],
    flat_results: List[TrainRunResult],
) -> Dict[str, Any]:
    summary = {}

    true_hier_accs = [r.metrics['hier_acc'] for r in true_results]
    flat_hier_accs = [r.metrics['hier_acc'] for r in flat_results]

    delta_true_per_seed = [t - f for t, f in zip(true_hier_accs, flat_hier_accs)]
    delta_true_mean = np.mean(delta_true_per_seed)
    delta_true_sem = np.std(delta_true_per_seed, ddof=1) / np.sqrt(len(delta_true_per_seed)) if len(delta_true_per_seed) > 1 else 0.0

    summary['delta_true'] = {
        'mean': float(delta_true_mean),
        'sem': float(delta_true_sem),
        'ci95_low': float(delta_true_mean - 1.96 * delta_true_sem),
        'ci95_high': float(delta_true_mean + 1.96 * delta_true_sem),
    }

    delta_rand_means = []
    flat_hier_mean = np.mean(flat_hier_accs)
    for rand_result in randomizations:
        rand_hier_mean = rand_result.aggregate.get('hier_acc.mean', 0)
        delta_rand_means.append(rand_hier_mean - flat_hier_mean)

    if delta_rand_means:
        delta_rand_exp_mean = np.mean(delta_rand_means)
        delta_rand_sem = np.std(delta_rand_means, ddof=1) / np.sqrt(len(delta_rand_means)) if len(delta_rand_means) > 1 else 0.0

        summary['delta_rand'] = {
            'exp_mean': float(delta_rand_exp_mean),
            'sem': float(delta_rand_sem),
            'ci95_low': float(delta_rand_exp_mean - 1.96 * delta_rand_sem),
            'ci95_high': float(delta_rand_exp_mean + 1.96 * delta_rand_sem),
            'values': [float(v) for v in delta_rand_means],
        }

        gap = delta_true_mean - delta_rand_exp_mean
        gap_se = np.sqrt(delta_true_sem**2 + delta_rand_sem**2)
        summary['gap'] = {
            'value': float(gap),
            'ci95_low': float(gap - 1.96 * gap_se),
            'ci95_high': float(gap + 1.96 * gap_se),
        }

        frac_le_zero = sum(1 for d in delta_rand_means if d <= 0) / len(delta_rand_means)
        frac_le_threshold = sum(1 for d in delta_rand_means if d <= 0.001) / len(delta_rand_means)
        summary['fractions'] = {
            'delta_rand_le_zero': float(frac_le_zero),
            'delta_rand_le_0.1pct': float(frac_le_threshold),
        }

        # Determine status
        if gap > 0 and summary['gap']['ci95_low'] > 0 and frac_le_threshold >= 0.80:
            summary['status'] = 'PASS'
        elif summary['gap']['ci95_low'] <= 0 and gap < 0.002:
            summary['status'] = 'FAIL'
        else:
            summary['status'] = 'INCONCLUSIVE'

    return summary


# ============== MAIN EXPERIMENT ==============

def run_fast_experiment(
    backbone, tokenizer, device,
    cfg: ExperimentConfig,
    epochs: int = 8,
    batch_size: int = 64,
):
    print("\n" + "="*70)
    print("FAST HIERARCHY RANDOMIZATION EXPERIMENT")
    print(f"K={cfg.k_randomizations} randomizations, S={cfg.s_seeds} seeds")
    print("="*70)

    # Pre-compute embeddings ONCE
    data = load_and_encode_data(backbone, tokenizer, device)
    hidden_dim = data['hidden_dim']
    num_classes = [NUM_SUPER, NUM_SUB]

    true_mapping = extract_leaf_parent_mapping()
    train_seeds = [cfg.base_seed + i * 111 for i in range(cfg.s_seeds)]

    # Get labels with true mapping
    train_super, train_sub = get_labels_with_mapping(data['train_sub'], true_mapping)
    test_super, test_sub = get_labels_with_mapping(data['test_sub'], true_mapping)

    # Checkpoint
    checkpoint_path = Path(__file__).parent.parent / "results" / "hierarchy_randomization_checkpoint.json"

    # ---- Flat baseline ----
    print("\n[1/3] Running FLAT baseline...")
    flat_results = []
    for seed in tqdm(train_seeds, desc="Flat"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = FlatClassifier(hidden_dim, num_classes)
        metrics = train_classifier_fast(
            model, data['train_emb'], train_super, train_sub,
            data['test_emb'], test_super, test_sub, device, epochs, batch_size
        )
        flat_results.append(TrainRunResult(seed=seed, metrics=metrics))
        del model
        torch.cuda.empty_cache()

    flat_mean = np.mean([r.metrics['hier_acc'] for r in flat_results])
    print(f"  Flat hier_acc: {flat_mean*100:.2f}%")

    # ---- Fractal + True ----
    print("\n[2/3] Running FRACTAL + TRUE hierarchy...")
    true_results = []
    for seed in tqdm(train_seeds, desc="Fractal+True"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = FractalClassifier(hidden_dim, num_classes)
        metrics = train_classifier_fast(
            model, data['train_emb'], train_super, train_sub,
            data['test_emb'], test_super, test_sub, device, epochs, batch_size
        )
        true_results.append(TrainRunResult(seed=seed, metrics=metrics))
        del model
        torch.cuda.empty_cache()

    true_mean = np.mean([r.metrics['hier_acc'] for r in true_results])
    delta_true = true_mean - flat_mean
    print(f"  Fractal+True hier_acc: {true_mean*100:.2f}%")
    print(f"  Delta_true: {delta_true*100:+.2f}%")

    # ---- Randomizations ----
    print(f"\n[3/3] Running FRACTAL + RANDOM hierarchy (K={cfg.k_randomizations})...")

    randomizations = []
    start_k = 0

    # Load checkpoint if exists
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            for rand_data in checkpoint.get('randomizations', []):
                train_runs = [TrainRunResult(seed=tr['seed'], metrics=tr['metrics'])
                              for tr in rand_data['train_runs']]
                randomizations.append(RandomizationResult(
                    rand_id=rand_data['rand_id'],
                    rand_seed=rand_data['rand_seed'],
                    mapping=rand_data['mapping'],
                    group_sizes=rand_data['group_sizes'],
                    train_runs=train_runs,
                    aggregate=rand_data['aggregate'],
                ))
            start_k = len(randomizations)
            if start_k > 0:
                print(f"  Resuming from checkpoint: {start_k} randomizations done")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")

    for k in tqdm(range(start_k, cfg.k_randomizations), desc="Randomizations", initial=start_k, total=cfg.k_randomizations):
        rand_seed = cfg.base_seed + 10000 + k * 7
        rng = np.random.default_rng(rand_seed)

        rand_mapping, group_sizes = level_preserving_parent_shuffle(rng, true_mapping)

        # Get labels with randomized mapping
        rand_train_super, _ = get_labels_with_mapping(data['train_sub'], rand_mapping)
        rand_test_super, _ = get_labels_with_mapping(data['test_sub'], rand_mapping)

        train_runs = []
        for seed in train_seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = FractalClassifier(hidden_dim, num_classes)
            metrics = train_classifier_fast(
                model, data['train_emb'], rand_train_super, data['train_sub'],
                data['test_emb'], rand_test_super, data['test_sub'], device, epochs, batch_size
            )
            train_runs.append(TrainRunResult(seed=seed, metrics=metrics))
            del model
            torch.cuda.empty_cache()

        aggregate = aggregate_train_runs(train_runs)

        randomizations.append(RandomizationResult(
            rand_id=k,
            rand_seed=rand_seed,
            mapping={int(mk): int(mv) for mk, mv in rand_mapping.items()},
            group_sizes={int(gk): int(gv) for gk, gv in group_sizes.items()},
            train_runs=train_runs,
            aggregate=aggregate,
        ))

        # Save checkpoint
        checkpoint_data = {
            'flat_results': [{'seed': r.seed, 'metrics': r.metrics} for r in flat_results],
            'true_results': [{'seed': r.seed, 'metrics': r.metrics} for r in true_results],
            'randomizations': [
                {
                    'rand_id': r.rand_id,
                    'rand_seed': r.rand_seed,
                    'mapping': r.mapping,
                    'group_sizes': r.group_sizes,
                    'train_runs': [{'seed': tr.seed, 'metrics': tr.metrics} for tr in r.train_runs],
                    'aggregate': r.aggregate,
                }
                for r in randomizations
            ],
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        if (k + 1) % 5 == 0:
            rand_mean = aggregate.get('hier_acc.mean', 0)
            print(f"  K={k+1}: rand_hier_acc={rand_mean*100:.2f}%")

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    summary = summarize_results(randomizations, true_results, flat_results)

    print(f"\nFlat baseline: {flat_mean*100:.2f}%")
    print(f"Fractal+True:  {true_mean*100:.2f}%")
    print(f"Delta_true:    {summary['delta_true']['mean']*100:+.2f}% [{summary['delta_true']['ci95_low']*100:.2f}%, {summary['delta_true']['ci95_high']*100:.2f}%]")

    if 'delta_rand' in summary:
        print(f"\nDelta_rand:    {summary['delta_rand']['exp_mean']*100:+.2f}% [{summary['delta_rand']['ci95_low']*100:.2f}%, {summary['delta_rand']['ci95_high']*100:.2f}%]")
        print(f"Gap:           {summary['gap']['value']*100:+.2f}% [{summary['gap']['ci95_low']*100:.2f}%, {summary['gap']['ci95_high']*100:.2f}%]")
        print(f"\nFraction Delta_rand <= 0:    {summary['fractions']['delta_rand_le_zero']*100:.1f}%")
        print(f"Fraction Delta_rand <= +0.1%: {summary['fractions']['delta_rand_le_0.1pct']*100:.1f}%")
        print(f"\nSTATUS: {summary['status']}")

    # Save final results
    output_path = Path(__file__).parent.parent / "results" / "hierarchy_randomization_fast.json"
    with open(output_path, 'w') as f:
        json.dump({
            'flat_results': [{'seed': r.seed, 'metrics': r.metrics} for r in flat_results],
            'true_results': [{'seed': r.seed, 'metrics': r.metrics} for r in true_results],
            'randomizations': [
                {
                    'rand_id': r.rand_id,
                    'rand_seed': r.rand_seed,
                    'mapping': r.mapping,
                    'group_sizes': r.group_sizes,
                    'train_runs': [{'seed': tr.seed, 'metrics': tr.metrics} for tr in r.train_runs],
                    'aggregate': r.aggregate,
                }
                for r in randomizations
            ],
            'summary': summary,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--s", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    cfg = ExperimentConfig(k_randomizations=args.k, s_seeds=args.s)
    run_fast_experiment(backbone, tokenizer, device, cfg, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
