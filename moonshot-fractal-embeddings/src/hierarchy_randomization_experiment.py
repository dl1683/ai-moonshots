"""
Hierarchy Randomization Experiment
===================================

Negative control experiment to prove that fractal classifier advantage
comes from CORRECT hierarchy structure, not just having ANY hierarchy.

If we randomize the hierarchy (wrong parent-child relationships),
the fractal advantage should disappear or reverse.

Implementation following the experimental design spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import random
import json
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ttest_rel, wilcoxon
from datasets import load_dataset

from multi_model_pipeline import MODELS


# ============== DATA STRUCTURES ==============

@dataclass
class ExperimentConfig:
    """Configuration for the hierarchy randomization experiment."""
    dataset: str = "20newsgroups"
    k_randomizations: int = 30
    s_seeds: int = 3
    base_seed: int = 1337
    scheme: str = "level_preserving_parent_shuffle"
    early_stop_metric: Optional[str] = None
    early_stop_ci_width: Optional[float] = None
    early_stop_min_k: int = 10
    max_failures: int = 3


@dataclass
class TrainRunResult:
    """Metrics and metadata from a single training run."""
    seed: int
    metrics: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RandomizationResult:
    """All results for one randomized hierarchy."""
    rand_id: int
    rand_seed: int
    mapping: Dict[int, int]  # leaf_idx -> parent_idx
    group_sizes: Dict[int, int]  # parent_idx -> count
    train_runs: List[TrainRunResult]
    aggregate: Dict[str, float]


@dataclass
class ExperimentResult:
    """Full experiment output and summary statistics."""
    dataset: str
    scheme: str
    k: int
    s: int
    true_hierarchy_results: List[TrainRunResult]  # Results with correct hierarchy
    flat_results: List[TrainRunResult]  # Flat baseline results
    randomizations: List[RandomizationResult]
    summary: Dict[str, Any]


# ============== 20 NEWSGROUPS HIERARCHY ==============

# Original hierarchy from real_benchmark.py
HIERARCHY = {
    "comp": ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
             "comp.sys.mac.hardware", "comp.windows.x"],
    "rec": ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
    "sci": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
    "misc": ["misc.forsale"],
    "talk": ["talk.politics.misc", "talk.politics.guns", "talk.politics.mideast", "talk.religion.misc"],
    "soc": ["alt.atheism", "soc.religion.christian"],
}

# Build base mappings
SUPER_TO_IDX = {s: i for i, s in enumerate(HIERARCHY.keys())}
IDX_TO_SUPER = {i: s for s, i in SUPER_TO_IDX.items()}

SUB_TO_SUPER = {}
SUB_TO_IDX = {}
IDX_TO_SUB = {}

idx = 0
for super_cat, subs in HIERARCHY.items():
    for sub in subs:
        SUB_TO_SUPER[sub] = super_cat
        SUB_TO_IDX[sub] = idx
        IDX_TO_SUB[idx] = sub
        idx += 1

NUM_SUPER = len(HIERARCHY)
NUM_SUB = len(SUB_TO_IDX)


# ============== CORE RANDOMIZATION FUNCTION ==============

def extract_leaf_parent_mapping() -> Dict[int, int]:
    """
    Return leaf_id -> parent_id for all leaves in the 20NG hierarchy.

    Returns:
        Dict mapping sub-category index to super-category index.
    """
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
    """
    Return a shuffled leaf->parent mapping that preserves each parent's leaf count.

    Algorithm:
    1. Build parents as deterministic ordered list of parent IDs
    2. Build group_sizes as {parent_id: count_of_leaves}
    3. Build leaves as deterministic ordered list of leaf IDs
    4. Shuffle leaves using RNG
    5. Reassign leaves to parents preserving group sizes

    Args:
        rng: Numpy random number generator instance.
        leaf_to_parent: Original mapping from leaf ids to parent ids.

    Returns:
        Tuple of (new_mapping, group_sizes)
    """
    # Step 1: Build deterministic ordered list of parents
    parents = sorted(set(leaf_to_parent.values()))

    # Step 2: Build group_sizes
    group_sizes = {}
    for parent in parents:
        group_sizes[parent] = sum(1 for p in leaf_to_parent.values() if p == parent)

    # Step 3: Build deterministic ordered list of leaves
    leaves = sorted(leaf_to_parent.keys())

    # Step 4: Shuffle leaves in-place
    shuffled_leaves = leaves.copy()
    rng.shuffle(shuffled_leaves)

    # Step 5: Reassign leaves to parents preserving group sizes
    new_mapping = {}
    i = 0
    for parent in parents:
        size = group_sizes[parent]
        for leaf in shuffled_leaves[i:i + size]:
            new_mapping[leaf] = parent
        i += size

    # Verify correctness
    assert len(new_mapping) == len(leaf_to_parent), "Mapping size mismatch"
    for parent in parents:
        actual_size = sum(1 for p in new_mapping.values() if p == parent)
        assert actual_size == group_sizes[parent], f"Group size mismatch for parent {parent}"

    return new_mapping, group_sizes


# ============== DATASET WITH CONFIGURABLE HIERARCHY ==============

@dataclass
class HierarchicalSample:
    text: str
    labels: List[int]  # [super_idx, sub_idx]


class NewsGroupsDatasetWithMapping(Dataset):
    """20 Newsgroups with configurable leaf->parent mapping."""

    def __init__(
        self,
        split: str = "train",
        max_samples: int = 5000,
        seed: int = 42,
        leaf_to_parent: Optional[Dict[int, int]] = None,
    ):
        random.seed(seed)
        np.random.seed(seed)

        # Use provided mapping or default
        self.leaf_to_parent = leaf_to_parent or extract_leaf_parent_mapping()

        ds = load_dataset("SetFit/20_newsgroups", split=split)

        self.samples = []
        for item in ds:
            if len(self.samples) >= max_samples:
                break

            text = item['text'][:512]
            label_text = item['label_text']

            # Find sub-category index
            if label_text not in SUB_TO_IDX:
                continue

            sub_idx = SUB_TO_IDX[label_text]

            # Use the (possibly randomized) mapping for super-category
            super_idx = self.leaf_to_parent[sub_idx]

            self.samples.append(HierarchicalSample(
                text=text,
                labels=[super_idx, sub_idx]
            ))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def num_classes_per_level(self) -> List[int]:
        return [NUM_SUPER, NUM_SUB]

    @property
    def depth(self):
        return 2


# ============== CLASSIFIERS (copied from real_benchmark.py) ==============

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


# ============== TRAINING UTILITIES ==============

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
def evaluate(model, backbone, tokenizer, dataset, device, batch_size) -> Dict[str, float]:
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


# ============== SINGLE RUN FUNCTION ==============

def run_single_condition(
    seed: int,
    backbone,
    tokenizer,
    device,
    classifier_type: str,  # 'flat' or 'fractal'
    leaf_to_parent: Optional[Dict[int, int]] = None,
    epochs: int = 8,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Run a single training condition.

    Args:
        seed: Random seed
        backbone: Pre-trained backbone model
        tokenizer: Tokenizer
        device: Device to use
        classifier_type: 'flat' or 'fractal'
        leaf_to_parent: Hierarchy mapping (None = use true hierarchy)
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        Dict of evaluation metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load data with specified hierarchy
    train_data = NewsGroupsDatasetWithMapping(
        split="train", max_samples=8000, seed=seed, leaf_to_parent=leaf_to_parent
    )
    val_data = NewsGroupsDatasetWithMapping(
        split="test", max_samples=2000, seed=seed + 1000, leaf_to_parent=leaf_to_parent
    )

    num_classes = train_data.num_classes_per_level
    hidden_dim = backbone.config.hidden_size

    # Create classifier
    if classifier_type == 'flat':
        model = FlatClassifier(hidden_dim, num_classes).to(device)
    else:
        model = FractalClassifier(hidden_dim, num_classes).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    best = None
    for ep in range(1, epochs + 1):
        train_epoch(model, backbone, tokenizer, train_data, optimizer, device, batch_size)
        ev = evaluate(model, backbone, tokenizer, val_data, device, batch_size)
        if best is None or ev['hier_acc'] > best['hier_acc']:
            best = ev.copy()
            best['epoch'] = ep

    del model
    torch.cuda.empty_cache()

    return best


# ============== AGGREGATION FUNCTIONS ==============

def aggregate_train_runs(train_runs: List[TrainRunResult]) -> Dict[str, float]:
    """
    Compute per-metric aggregates across S seeds for one randomization.

    Returns dict with keys: {metric}.mean, {metric}.std, {metric}.min, {metric}.max
    """
    if not train_runs:
        return {}

    # Get all metric names
    metric_names = list(train_runs[0].metrics.keys())

    aggregate = {}
    for metric in metric_names:
        values = [run.metrics[metric] for run in train_runs]
        aggregate[f'{metric}.mean'] = float(np.mean(values))
        aggregate[f'{metric}.std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate[f'{metric}.min'] = float(np.min(values))
        aggregate[f'{metric}.max'] = float(np.max(values))

    return aggregate


def summarize_randomizations(
    randomizations: List[RandomizationResult],
    true_results: List[TrainRunResult],
    flat_results: List[TrainRunResult],
) -> Dict[str, Any]:
    """
    Compute experiment-level statistics across K randomizations.

    Computes:
    - Delta_true = (Fractal + H_true) - Flat
    - Delta_rand[k] = (Fractal + H_rand[k]) - Flat
    - Delta_true - mean(Delta_rand) with CI
    - Fraction of randomizations where Delta_rand > 0 and >= Delta_true
    """
    summary = {}

    # Compute Delta_true per seed
    true_hier_accs = [r.metrics['hier_acc'] for r in true_results]
    flat_hier_accs = [r.metrics['hier_acc'] for r in flat_results]

    delta_true_per_seed = [t - f for t, f in zip(true_hier_accs, flat_hier_accs)]
    delta_true_mean = np.mean(delta_true_per_seed)
    delta_true_std = np.std(delta_true_per_seed, ddof=1) if len(delta_true_per_seed) > 1 else 0.0
    delta_true_sem = delta_true_std / np.sqrt(len(delta_true_per_seed)) if len(delta_true_per_seed) > 1 else 0.0

    summary['delta_true'] = {
        'mean': float(delta_true_mean),
        'std': float(delta_true_std),
        'sem': float(delta_true_sem),
        'ci95_low': float(delta_true_mean - 1.96 * delta_true_sem),
        'ci95_high': float(delta_true_mean + 1.96 * delta_true_sem),
        'values': [float(v) for v in delta_true_per_seed],
    }

    # Compute Delta_rand for each randomization (use mean across seeds)
    delta_rand_means = []
    for rand_result in randomizations:
        # Get hier_acc.mean from aggregate
        rand_hier_mean = rand_result.aggregate.get('hier_acc.mean', 0)
        flat_hier_mean = np.mean(flat_hier_accs)
        delta_rand_means.append(rand_hier_mean - flat_hier_mean)

    if delta_rand_means:
        delta_rand_exp_mean = np.mean(delta_rand_means)
        delta_rand_exp_std = np.std(delta_rand_means, ddof=1) if len(delta_rand_means) > 1 else 0.0
        delta_rand_sem = delta_rand_exp_std / np.sqrt(len(delta_rand_means)) if len(delta_rand_means) > 1 else 0.0

        summary['delta_rand'] = {
            'exp_mean': float(delta_rand_exp_mean),
            'exp_std': float(delta_rand_exp_std),
            'sem': float(delta_rand_sem),
            'ci95_low': float(delta_rand_exp_mean - 1.96 * delta_rand_sem),
            'ci95_high': float(delta_rand_exp_mean + 1.96 * delta_rand_sem),
            'values': [float(v) for v in delta_rand_means],
        }

        # Compute Delta_true - mean(Delta_rand)
        gap = delta_true_mean - delta_rand_exp_mean
        # Combined SE (simplified - assumes independence)
        gap_se = np.sqrt(delta_true_sem**2 + delta_rand_sem**2)

        summary['gap'] = {
            'value': float(gap),
            'se': float(gap_se),
            'ci95_low': float(gap - 1.96 * gap_se),
            'ci95_high': float(gap + 1.96 * gap_se),
        }

        # Fraction statistics
        frac_positive = sum(1 for d in delta_rand_means if d > 0) / len(delta_rand_means)
        frac_ge_true = sum(1 for d in delta_rand_means if d >= delta_true_mean) / len(delta_rand_means)
        frac_le_zero = sum(1 for d in delta_rand_means if d <= 0) / len(delta_rand_means)
        frac_le_threshold = sum(1 for d in delta_rand_means if d <= 0.001) / len(delta_rand_means)  # +0.1%

        summary['fractions'] = {
            'delta_rand_positive': float(frac_positive),
            'delta_rand_ge_delta_true': float(frac_ge_true),
            'delta_rand_le_zero': float(frac_le_zero),
            'delta_rand_le_0.1pct': float(frac_le_threshold),
        }

    return summary


def evaluate_stopping_criteria(summary: Dict[str, Any], cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Evaluate the stopping criteria based on experimental design specification.

    Returns dict with:
    - status: 'PASS', 'FAIL', or 'INCONCLUSIVE'
    - reasons: List of reasons for the status
    - needs_human_review: bool
    """
    result = {
        'status': 'INCONCLUSIVE',
        'reasons': [],
        'needs_human_review': False,
    }

    delta_true = summary.get('delta_true', {})
    delta_rand = summary.get('delta_rand', {})
    gap = summary.get('gap', {})
    fractions = summary.get('fractions', {})

    # Check for PASS conditions (all must hold)
    pass_conditions = []

    # 1. Delta_true > 0 and significant
    if delta_true.get('ci95_low', 0) > 0:
        pass_conditions.append("Delta_true CI excludes 0")
    else:
        result['reasons'].append("Delta_true CI includes 0")

    # 2. Gap > 0 with CI excluding 0
    if gap.get('ci95_low', 0) > 0:
        pass_conditions.append("Gap CI excludes 0")
    else:
        result['reasons'].append("Gap CI includes 0")

    # 3. At least 80% of Delta_rand <= 0 (or <= +0.1%)
    if fractions.get('delta_rand_le_0.1pct', 0) >= 0.80:
        pass_conditions.append(">=80% of Delta_rand <= +0.1%")
    else:
        result['reasons'].append(f"Only {fractions.get('delta_rand_le_0.1pct', 0)*100:.1f}% of Delta_rand <= +0.1%")

    # Check for FAIL conditions (any triggers fail)
    fail_conditions = []

    # 1. Gap CI includes 0 AND gap < 0.2%
    if gap.get('ci95_low', -999) <= 0 and gap.get('value', 999) < 0.002:
        fail_conditions.append("Gap CI includes 0 and gap < 0.2%")

    # 2. Delta_rand mean >= Delta_true - 0.1%
    if delta_rand.get('exp_mean', -999) >= delta_true.get('mean', 999) - 0.001:
        fail_conditions.append("Delta_rand mean >= Delta_true - 0.1%")

    # Determine status
    if len(pass_conditions) == 3:
        result['status'] = 'PASS'
        result['reasons'] = pass_conditions
    elif fail_conditions:
        result['status'] = 'FAIL'
        result['reasons'] = fail_conditions
    else:
        result['status'] = 'INCONCLUSIVE'

    # Check for human review triggers
    if delta_rand:
        if delta_rand.get('exp_std', 0) > 2 * abs(delta_rand.get('exp_mean', 0.001)):
            result['needs_human_review'] = True
            result['reasons'].append("High variance: std > 2*|mean|")

    return result


# ============== MAIN EXPERIMENT RUNNER ==============

def run_hierarchy_randomization_experiment(
    backbone,
    tokenizer,
    device,
    cfg: ExperimentConfig,
    epochs: int = 8,
    batch_size: int = 32,
) -> ExperimentResult:
    """
    Run the full hierarchy randomization experiment.

    Following the experimental design spec:
    1. Run Flat baseline for S seeds
    2. Run Fractal + H_true for S seeds
    3. Run Fractal + H_rand[k] for K randomizations x S seeds
    4. Compute statistics and evaluate stopping criteria
    """

    print("\n" + "="*70)
    print("HIERARCHY RANDOMIZATION EXPERIMENT")
    print(f"K={cfg.k_randomizations} randomizations, S={cfg.s_seeds} seeds")
    print("="*70)

    # Get true hierarchy
    true_mapping = extract_leaf_parent_mapping()

    # Seeds for training
    train_seeds = [cfg.base_seed + i * 111 for i in range(cfg.s_seeds)]

    # ---- Step 1: Run Flat baseline ----
    print("\n[1/3] Running FLAT baseline...")
    flat_results = []
    for seed in tqdm(train_seeds, desc="Flat seeds"):
        metrics = run_single_condition(
            seed, backbone, tokenizer, device,
            classifier_type='flat',
            leaf_to_parent=true_mapping,  # Flat uses true labels
            epochs=epochs, batch_size=batch_size,
        )
        flat_results.append(TrainRunResult(seed=seed, metrics=metrics))

    flat_mean = np.mean([r.metrics['hier_acc'] for r in flat_results])
    print(f"  Flat hier_acc: {flat_mean*100:.2f}%")

    # ---- Step 2: Run Fractal + H_true ----
    print("\n[2/3] Running FRACTAL + TRUE hierarchy...")
    true_results = []
    for seed in tqdm(train_seeds, desc="Fractal+True seeds"):
        metrics = run_single_condition(
            seed, backbone, tokenizer, device,
            classifier_type='fractal',
            leaf_to_parent=true_mapping,
            epochs=epochs, batch_size=batch_size,
        )
        true_results.append(TrainRunResult(seed=seed, metrics=metrics))

    true_mean = np.mean([r.metrics['hier_acc'] for r in true_results])
    delta_true = true_mean - flat_mean
    print(f"  Fractal+True hier_acc: {true_mean*100:.2f}%")
    print(f"  Delta_true: {delta_true*100:+.2f}%")

    # ---- Step 3: Run Fractal + H_rand for K randomizations ----
    print(f"\n[3/3] Running FRACTAL + RANDOM hierarchy (K={cfg.k_randomizations})...")

    randomizations = []
    failures = 0

    # Checkpoint file for incremental saving
    checkpoint_path = Path(__file__).parent.parent / "results" / "hierarchy_randomization_checkpoint.json"
    start_k = 0

    # Try to load existing checkpoint
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            # Restore previous randomizations
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
            print(f"  Resuming from checkpoint: {start_k} randomizations already done")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}, starting fresh")

    for k in tqdm(range(start_k, cfg.k_randomizations), desc="Randomizations", initial=start_k, total=cfg.k_randomizations):
        rand_seed = cfg.base_seed + 10000 + k * 7
        rng = np.random.default_rng(rand_seed)

        # Generate randomized hierarchy
        rand_mapping, group_sizes = level_preserving_parent_shuffle(rng, true_mapping)

        # Run for S seeds
        train_runs = []
        try:
            for seed in train_seeds:
                metrics = run_single_condition(
                    seed, backbone, tokenizer, device,
                    classifier_type='fractal',
                    leaf_to_parent=rand_mapping,
                    epochs=epochs, batch_size=batch_size,
                )
                train_runs.append(TrainRunResult(seed=seed, metrics=metrics))
        except Exception as e:
            print(f"  Randomization {k} failed: {e}")
            failures += 1
            if failures >= cfg.max_failures:
                print(f"  Max failures ({cfg.max_failures}) reached, stopping early")
                break
            continue

        # Aggregate results
        aggregate = aggregate_train_runs(train_runs)

        randomizations.append(RandomizationResult(
            rand_id=k,
            rand_seed=rand_seed,
            mapping={int(mk): int(mv) for mk, mv in rand_mapping.items()},
            group_sizes={int(gk): int(gv) for gk, gv in group_sizes.items()},
            train_runs=train_runs,
            aggregate=aggregate,
        ))

        # Save checkpoint after EVERY randomization
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
            'k_done': len(randomizations),
            'k_total': cfg.k_randomizations,
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Print progress every 5 randomizations
        if (k + 1) % 5 == 0:
            rand_mean = aggregate.get('hier_acc.mean', 0)
            print(f"  K={k+1}: rand_hier_acc={rand_mean*100:.2f}%")

    # ---- Compute Summary Statistics ----
    print("\n" + "-"*70)
    print("COMPUTING SUMMARY STATISTICS")
    print("-"*70)

    summary = summarize_randomizations(randomizations, true_results, flat_results)
    stopping = evaluate_stopping_criteria(summary, cfg)
    summary['stopping_criteria'] = stopping

    # Print summary
    print(f"\nDelta_true (Fractal+True - Flat): {summary['delta_true']['mean']*100:+.2f}%")
    print(f"  95% CI: [{summary['delta_true']['ci95_low']*100:.2f}%, {summary['delta_true']['ci95_high']*100:.2f}%]")

    if 'delta_rand' in summary:
        print(f"\nDelta_rand (Fractal+Rand - Flat): {summary['delta_rand']['exp_mean']*100:+.2f}%")
        print(f"  95% CI: [{summary['delta_rand']['ci95_low']*100:.2f}%, {summary['delta_rand']['ci95_high']*100:.2f}%]")

        print(f"\nGap (Delta_true - Delta_rand): {summary['gap']['value']*100:+.2f}%")
        print(f"  95% CI: [{summary['gap']['ci95_low']*100:.2f}%, {summary['gap']['ci95_high']*100:.2f}%]")

        print(f"\nFractions:")
        print(f"  Delta_rand > 0: {summary['fractions']['delta_rand_positive']*100:.1f}%")
        print(f"  Delta_rand <= 0: {summary['fractions']['delta_rand_le_zero']*100:.1f}%")
        print(f"  Delta_rand <= +0.1%: {summary['fractions']['delta_rand_le_0.1pct']*100:.1f}%")

    print(f"\n{'='*70}")
    print(f"STOPPING CRITERIA STATUS: {stopping['status']}")
    print(f"{'='*70}")
    for reason in stopping['reasons']:
        print(f"  - {reason}")
    if stopping['needs_human_review']:
        print("\n*** HUMAN REVIEW NEEDED ***")

    return ExperimentResult(
        dataset=cfg.dataset,
        scheme=cfg.scheme,
        k=len(randomizations),
        s=cfg.s_seeds,
        true_hierarchy_results=true_results,
        flat_results=flat_results,
        randomizations=randomizations,
        summary=summary,
    )


def save_results(result: ExperimentResult, output_path: Path):
    """Save experiment results to JSON."""

    # Convert to serializable format
    data = {
        'dataset': result.dataset,
        'scheme': result.scheme,
        'k': result.k,
        's': result.s,
        'true_hierarchy_results': [
            {'seed': r.seed, 'metrics': r.metrics} for r in result.true_hierarchy_results
        ],
        'flat_results': [
            {'seed': r.seed, 'metrics': r.metrics} for r in result.flat_results
        ],
        'randomizations': [
            {
                'rand_id': r.rand_id,
                'rand_seed': r.rand_seed,
                'mapping': r.mapping,
                'group_sizes': r.group_sizes,
                'train_runs': [{'seed': tr.seed, 'metrics': tr.metrics} for tr in r.train_runs],
                'aggregate': r.aggregate,
            }
            for r in result.randomizations
        ],
        'summary': result.summary,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


# ============== MAIN ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchy Randomization Experiment")
    parser.add_argument("--model", type=str, default="qwen3-0.6b", help="Backbone model")
    parser.add_argument("--k", type=int, default=30, help="Number of randomizations")
    parser.add_argument("--s", type=int, default=3, help="Number of seeds per condition")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--base_seed", type=int, default=1337, help="Base random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load backbone
    print(f"\nLoading backbone: {args.model}...")
    model_config = MODELS[args.model]

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.hf_path, trust_remote_code=model_config.trust_remote_code
    )
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

    # Create config
    cfg = ExperimentConfig(
        dataset="20newsgroups",
        k_randomizations=args.k,
        s_seeds=args.s,
        base_seed=args.base_seed,
    )

    # Run experiment
    result = run_hierarchy_randomization_experiment(
        backbone, tokenizer, device, cfg,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    # Save results
    output_path = Path(__file__).parent.parent / "results" / f"hierarchy_randomization_{args.model}_k{args.k}_s{args.s}.json"
    save_results(result, output_path)

    return result


if __name__ == "__main__":
    main()
