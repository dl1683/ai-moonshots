"""
Three-Level Hierarchy Experiment
================================

Demonstrates that fractal embeddings generalize beyond 2-level hierarchies
to monotonic 3-level semantic zoom.

Uses CLINC-150 with a constructed 3-level hierarchy:
  L0: 5 super-domains (Finance, Home_Life, Transport, Productivity, Social)
  L1: 10 domains (the original CLINC domains)
  L2: 150 intents (the original CLINC fine labels)

V5 training aligns 3 prefix levels:
  j=1 (64d)  -> L0 supervision (super-domain)
  j=2 (128d) -> mixed L0 + L1 supervision
  j=3 (192d) -> mixed L1 + L2 supervision
  j=4 (256d) -> L2 supervision (intent)

Expected result: monotonic semantic zoom at 3 granularity levels.
"""

import sys
import os
import json
import gc
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from copy import deepcopy

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset, HierarchicalSample
from multi_model_pipeline import MODELS
from fractal_v5 import FractalModelV5, split_train_val
from mrl_v5_baseline import MRLTrainerV5

# ---------------------------------------------------------------
# 3-level hierarchy construction
# ---------------------------------------------------------------

# Natural grouping of CLINC-150's 10 domains into 5 super-domains
SUPER_DOMAIN_MAP = {
    "banking": "Finance",
    "credit_cards": "Finance",
    "kitchen_and_dining": "Home_Life",
    "home": "Home_Life",
    "auto_and_commute": "Transport",
    "travel": "Transport",
    "utility": "Productivity",
    "work": "Productivity",
    "small_talk": "Social",
    "meta": "Social",
}

SUPER_DOMAIN_NAMES = ["Finance", "Home_Life", "Transport", "Productivity", "Social"]


def add_super_domain_labels(dataset):
    """Add L0 (super-domain) labels to CLINC samples. Shifts existing L0->L1, L1->L2."""
    domain_names = dataset.level0_names  # original 10 domains
    intent_names = dataset.level1_names  # original 150 intents

    super_to_idx = {name: i for i, name in enumerate(SUPER_DOMAIN_NAMES)}
    domain_to_super = {}
    for domain_name in domain_names:
        super_name = SUPER_DOMAIN_MAP.get(domain_name)
        if super_name is None:
            raise ValueError(f"Unknown domain: {domain_name}")
        domain_to_super[domain_names.index(domain_name)] = super_to_idx[super_name]

    new_samples = []
    for s in dataset.samples:
        super_label = domain_to_super[s.level0_label]
        new_samples.append(HierarchicalSample(
            text=s.text,
            level0_label=super_label,           # NEW L0: super-domain (5)
            level1_label=s.level0_label,         # NEW L1: domain (10)
            level2_label=s.level1_label,         # NEW L2: intent (150)
            level0_name=SUPER_DOMAIN_NAMES[super_label],
            level1_name=s.level0_name,           # original domain name
            level2_name=s.level1_name,           # original intent name
        ))

    dataset.samples = new_samples
    dataset.level0_names = SUPER_DOMAIN_NAMES[:]
    dataset.level1_names = domain_names[:]
    dataset.level2_names = intent_names[:]
    return dataset


# ---------------------------------------------------------------
# 3-level V5 Trainer
# ---------------------------------------------------------------

class ThreeLevelV5Trainer:
    """
    V5 trainer for 3-level hierarchies.

    j=1 (64d):  pure L0 (super-domain)
    j=2 (128d): 0.6*L0 + 0.4*L1 (domain)
    j=3 (192d): 0.3*L1 + 0.7*L2 (intent)
    j=4 (256d): pure L2 (intent)
    """

    def __init__(self, model, train_dataset, val_dataset, device="cuda",
                 epochs=5, lr=1e-4, prefix_weight=0.6):
        self.model = model
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.prefix_weight = prefix_weight

        # Classification heads:
        # model.fractal_head.head_top -> L0 (super-domain, 5 classes)
        # model.fractal_head.head_leaf -> L2 (intent, 150 classes)
        # New head_mid -> L1 (domain, 10 classes)
        self.num_l1 = len(train_dataset.level1_names)
        output_dim = model.embed_dim  # 256 (num_scales * scale_dim)
        self.head_mid = torch.nn.Sequential(
            torch.nn.LayerNorm(output_dim),
            torch.nn.Linear(output_dim, self.num_l1),
        ).to(device)

        # Optimizer over fractal head + mid head (backbone already frozen)
        self.optimizer = torch.optim.AdamW(
            list(model.fractal_head.parameters()) +
            list(self.head_mid.parameters()),
            lr=lr, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.ce_loss = torch.nn.CrossEntropyLoss()

        # Prefix sampling probs: [0.4, 0.3, 0.2, 0.1]
        self.prefix_probs = [0.4, 0.3, 0.2, 0.1]

    def _compute_loss(self, result, l0_labels, l1_labels, l2_labels, j):
        """Compute 3-level progressive loss for prefix j."""
        fh = self.model.fractal_head
        if j < 4:
            prefix_emb = fh.get_prefix_embedding(result['blocks'], j)
        else:
            prefix_emb = result['full_embedding']

        if j == 1:
            logits_l0 = fh.head_top(prefix_emb)
            return self.ce_loss(logits_l0, l0_labels)
        elif j == 2:
            logits_l0 = fh.head_top(prefix_emb)
            logits_l1 = self.head_mid(prefix_emb)
            return 0.6 * self.ce_loss(logits_l0, l0_labels) + 0.4 * self.ce_loss(logits_l1, l1_labels)
        elif j == 3:
            logits_l1 = self.head_mid(prefix_emb)
            logits_l2 = fh.head_leaf(prefix_emb)
            return 0.3 * self.ce_loss(logits_l1, l1_labels) + 0.7 * self.ce_loss(logits_l2, l2_labels)
        else:  # j == 4
            logits_l2 = fh.head_leaf(prefix_emb)
            return self.ce_loss(logits_l2, l2_labels)

    def _tokenize_texts(self, texts, max_len=512):
        """Tokenize texts for forward pass (preserves gradients unlike encode())."""
        max_len = min(self.model.config.max_seq_len, max_len)
        if self.model.config.prefix_query:
            texts = [self.model.config.prefix_query + t for t in texts]
        inputs = self.model.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        )
        return inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)

    def train(self, batch_size=16):
        """Train the 3-level V5 model."""
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

        texts = [s.text for s in self.train_data.samples]
        l0 = torch.tensor([s.level0_label for s in self.train_data.samples], device=self.device)
        l1 = torch.tensor([s.level1_label for s in self.train_data.samples], device=self.device)
        l2 = torch.tensor([s.level2_label for s in self.train_data.samples], device=self.device)

        n = len(texts)
        best_score = -1
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            self.head_mid.train()
            total_loss = 0
            num_batches = 0
            perm = np.random.permutation(n)

            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch_texts = [texts[i] for i in idx]
                batch_l0 = l0[idx]
                batch_l1 = l1[idx]
                batch_l2 = l2[idx]

                self.optimizer.zero_grad()

                with autocast(enabled=True):
                    # Tokenize and forward (preserves gradients)
                    input_ids, attention_mask = self._tokenize_texts(batch_texts)
                    result = self.model(input_ids, attention_mask)

                    # Full-embedding L2 loss
                    full_emb = result['full_embedding']
                    full_loss = self.ce_loss(
                        self.model.fractal_head.head_leaf(full_emb), batch_l2
                    )

                    # Sample prefix j
                    j = np.random.choice([1, 2, 3, 4], p=self.prefix_probs)
                    prefix_loss = self._compute_loss(result, batch_l0, batch_l1, batch_l2, j)

                    loss = full_loss + self.prefix_weight * prefix_loss

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.fractal_head.parameters()) + list(self.head_mid.parameters()), 1.0
                )
                scaler.step(self.optimizer)
                scaler.update()
                total_loss += loss.item()
                num_batches += 1

            self.scheduler.step()

            # Evaluate on val set
            val_l0, val_l1, val_l2 = self._evaluate()
            score = val_l0 + val_l1 + val_l2
            print(f"  Epoch {epoch+1}: loss={total_loss/max(num_batches,1):.4f}, "
                  f"L0={val_l0:.4f}, L1={val_l1:.4f}, L2={val_l2:.4f}")

            if score > best_score:
                best_score = score
                best_state = {
                    'model': deepcopy(self.model.state_dict()),
                    'head_mid': deepcopy(self.head_mid.state_dict()),
                }

        if best_state is not None:
            self.model.load_state_dict(best_state['model'])
            self.head_mid.load_state_dict(best_state['head_mid'])
        print(f"  Best val score: {best_score:.4f}")

    def _evaluate(self):
        """Evaluate classification accuracy at j=4 for all 3 levels."""
        self.model.eval()
        self.head_mid.eval()

        texts = [s.text for s in self.val_data.samples]
        l0_true = np.array([s.level0_label for s in self.val_data.samples])
        l1_true = np.array([s.level1_label for s in self.val_data.samples])
        l2_true = np.array([s.level2_label for s in self.val_data.samples])

        with torch.no_grad():
            # encode() is fine for eval (no_grad is intended)
            emb = self.model.encode(texts, batch_size=32)
            if isinstance(emb, np.ndarray):
                emb = torch.tensor(emb, device=self.device, dtype=torch.float32)
            emb = emb.to(self.device)

            l0_logits = self.model.fractal_head.head_top(emb)
            l1_logits = self.head_mid(emb)
            l2_logits = self.model.fractal_head.head_leaf(emb)

            l0_pred = l0_logits.argmax(dim=1).cpu().numpy()
            l1_pred = l1_logits.argmax(dim=1).cpu().numpy()
            l2_pred = l2_logits.argmax(dim=1).cpu().numpy()

        return (
            (l0_pred == l0_true).mean(),
            (l1_pred == l1_true).mean(),
            (l2_pred == l2_true).mean(),
        )


# ---------------------------------------------------------------
# Evaluation: 3-level steerability via k-NN
# ---------------------------------------------------------------

def evaluate_3level_steerability(model, test_data, head_mid, device="cuda"):
    """
    Evaluate k-NN classification accuracy at each prefix length j=1..4
    for all 3 hierarchy levels.
    """
    texts = [s.text for s in test_data.samples]
    l0_true = np.array([s.level0_label for s in test_data.samples])
    l1_true = np.array([s.level1_label for s in test_data.samples])
    l2_true = np.array([s.level2_label for s in test_data.samples])

    results = {}
    model.eval()

    for j in [1, 2, 3, 4]:
        prefix_len = j if j < 4 else None
        with torch.no_grad():
            emb = model.encode(texts, batch_size=32, prefix_len=prefix_len)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-9)

        # k-NN with k=5
        k = 5
        sims = emb @ emb.T
        np.fill_diagonal(sims, -1)

        n = len(texts)
        l0_correct = 0
        l1_correct = 0
        l2_correct = 0

        for i in range(n):
            top_k = np.argsort(-sims[i])[:k]
            # L0
            labels_k = l0_true[top_k]
            vals, counts = np.unique(labels_k, return_counts=True)
            pred = vals[np.argmax(counts)]
            if pred == l0_true[i]:
                l0_correct += 1
            # L1
            labels_k = l1_true[top_k]
            vals, counts = np.unique(labels_k, return_counts=True)
            pred = vals[np.argmax(counts)]
            if pred == l1_true[i]:
                l1_correct += 1
            # L2
            labels_k = l2_true[top_k]
            vals, counts = np.unique(labels_k, return_counts=True)
            pred = vals[np.argmax(counts)]
            if pred == l2_true[i]:
                l2_correct += 1

        results[j] = {
            'l0': l0_correct / n,
            'l1': l1_correct / n,
            'l2': l2_correct / n,
        }
        print(f"    j={j} ({j*64}d): L0={results[j]['l0']:.4f}, "
              f"L1={results[j]['l1']:.4f}, L2={results[j]['l2']:.4f}")

    # Compute steerability metrics
    s_02 = (results[1]['l0'] - results[4]['l0']) + (results[4]['l2'] - results[1]['l2'])
    s_01 = (results[1]['l0'] - results[2]['l0']) + (results[1]['l1'] - results[1]['l1'])
    s_12 = (results[2]['l1'] - results[4]['l1']) + (results[4]['l2'] - results[2]['l2'])

    # Better metric: monotonic zoom score
    # For each adjacent pair, check if coarser level decreases and finer level increases
    mono_score = 0
    # j1->j2: L0 should decrease, L1 should increase
    if results[1]['l0'] >= results[2]['l0'] and results[2]['l1'] >= results[1]['l1']:
        mono_score += 1
    # j2->j4: L1 should decrease, L2 should increase
    if results[2]['l1'] >= results[4]['l1'] and results[4]['l2'] >= results[2]['l2']:
        mono_score += 1

    return {
        'prefix_results': results,
        'steerability_02': s_02,
        'steerability_01': s_01,
        'steerability_12': s_12,
        'monotonic_zoom_score': mono_score,  # 0, 1, or 2
    }


# ---------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------

def run_three_level_experiment(
    model_key="bge-small",
    seeds=(42, 123, 456),
    device="cuda",
):
    print("=" * 70)
    print("THREE-LEVEL HIERARCHY EXPERIMENT")
    print("L0: 5 super-domains -> L1: 10 domains -> L2: 150 intents")
    print("=" * 70)

    RESULTS_DIR = Path(__file__).parent.parent / "results"
    config = MODELS[model_key]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load and augment CLINC with 3-level hierarchy
        print("[1] Loading CLINC with 3-level hierarchy...")
        train_data = load_hierarchical_dataset("clinc", split="train", max_samples=10000)
        test_data = load_hierarchical_dataset("clinc", split="test", max_samples=2000)

        train_data = add_super_domain_labels(train_data)
        test_data = add_super_domain_labels(test_data)

        train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

        class TempDS:
            def __init__(self, samples, l0n, l1n, l2n):
                self.samples = samples
                self.level0_names = l0n
                self.level1_names = l1n
                self.level2_names = l2n

        val_data = TempDS(val_samples, train_data.level0_names,
                          train_data.level1_names, train_data.level2_names)
        train_data.samples = train_samples

        num_l0 = len(train_data.level0_names)   # 5
        num_l1 = len(train_data.level1_names)   # 10
        num_l2 = len(train_data.level2_names)   # 150

        print(f"  Hierarchy: {num_l0} L0 -> {num_l1} L1 -> {num_l2} L2")

        # Train 3-level V5
        print("\n[2] Training 3-Level V5...")
        v5_model = FractalModelV5(
            config=config,
            num_l0_classes=num_l0,    # 5 super-domains
            num_l1_classes=num_l2,    # 150 intents (head_bot)
            num_scales=4, scale_dim=64, device=device,
        ).to(device)

        trainer = ThreeLevelV5Trainer(
            model=v5_model, train_dataset=train_data,
            val_dataset=val_data, device=device, epochs=5,
        )
        trainer.train(batch_size=16)

        # Evaluate 3-level steerability
        print("\n[3] Evaluating V5 3-level steerability...")
        v5_results = evaluate_3level_steerability(
            v5_model, test_data, trainer.head_mid, device
        )
        print(f"  V5 Steerability S_02: {v5_results['steerability_02']:+.4f}")
        print(f"  V5 Monotonic zoom: {v5_results['monotonic_zoom_score']}/2")

        # Train MRL baseline (all prefixes -> L2 only)
        print("\n[4] Training MRL baseline...")
        mrl_model = FractalModelV5(
            config=config,
            num_l0_classes=num_l2,
            num_l1_classes=num_l2,
            num_scales=4, scale_dim=64, device=device,
        ).to(device)

        # MRL baseline: all prefixes trained on L2 (finest level)
        mrl_train_samples = []
        for s in train_data.samples:
            mrl_train_samples.append(HierarchicalSample(
                text=s.text,
                level0_label=s.level2_label,
                level1_label=s.level2_label,
                level2_label=s.level2_label,
                level0_name=s.level2_name,
                level1_name=s.level2_name,
                level2_name=s.level2_name,
            ))
        mrl_val_samples = []
        for s in val_data.samples:
            mrl_val_samples.append(HierarchicalSample(
                text=s.text,
                level0_label=s.level2_label,
                level1_label=s.level2_label,
                level2_label=s.level2_label,
                level0_name=s.level2_name,
                level1_name=s.level2_name,
                level2_name=s.level2_name,
            ))

        mrl_td = TempDS(mrl_train_samples, train_data.level2_names,
                        train_data.level2_names, train_data.level2_names)
        mrl_vd = TempDS(mrl_val_samples, val_data.level2_names,
                        val_data.level2_names, val_data.level2_names)

        mrl_trainer = MRLTrainerV5(
            model=mrl_model, train_dataset=mrl_td,
            val_dataset=mrl_vd, device=device, stage1_epochs=5,
        )
        mrl_trainer.train(batch_size=16, patience=5)

        # Evaluate MRL
        print("\n[5] Evaluating MRL 3-level steerability...")
        mrl_results = evaluate_3level_steerability(
            mrl_model, test_data, None, device
        )
        print(f"  MRL Steerability S_02: {mrl_results['steerability_02']:+.4f}")
        print(f"  MRL Monotonic zoom: {mrl_results['monotonic_zoom_score']}/2")

        all_results.append({
            'seed': seed,
            'v5': v5_results,
            'mrl': mrl_results,
        })

        del v5_model, mrl_model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("  THREE-LEVEL EXPERIMENT SUMMARY")
    print("=" * 70)

    for method in ['v5', 'mrl']:
        print(f"\n  {method.upper()}:")
        s02_vals = [r[method]['steerability_02'] for r in all_results]
        mono_vals = [r[method]['monotonic_zoom_score'] for r in all_results]
        print(f"    S_02: {np.mean(s02_vals):+.4f} +/- {np.std(s02_vals):.4f}")
        print(f"    Monotonic zoom: {np.mean(mono_vals):.1f}/2")

        # Average prefix results
        for j in [1, 2, 3, 4]:
            l0s = [r[method]['prefix_results'][j]['l0'] for r in all_results]
            l1s = [r[method]['prefix_results'][j]['l1'] for r in all_results]
            l2s = [r[method]['prefix_results'][j]['l2'] for r in all_results]
            print(f"    j={j} ({j*64}d): L0={np.mean(l0s):.4f}, "
                  f"L1={np.mean(l1s):.4f}, L2={np.mean(l2s):.4f}")

    # Save
    out_path = RESULTS_DIR / "three_level_clinc.json"
    with open(out_path, 'w') as f:
        json.dump({
            'experiment': 'three_level_hierarchy',
            'dataset': 'clinc_3level',
            'hierarchy': '5 super-domains -> 10 domains -> 150 intents',
            'model': model_key,
            'seeds': list(seeds),
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_three_level_experiment()
