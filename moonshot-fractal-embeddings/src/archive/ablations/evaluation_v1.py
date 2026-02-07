"""
Fractal Embeddings Evaluation
=============================

Tests to determine if fractal structure improves embeddings:

1. Semantic Similarity: Do fractal embeddings better capture meaning?
2. Retrieval: Can we retrieve relevant documents better?
3. Hierarchical Structure: Do different scales capture different granularities?
4. Efficiency: Can we use fewer dimensions and still perform well? (Matryoshka property)
5. Ablations: Does the fractal structure matter, or is it just more parameters?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

from fractal_embeddings import (
    FractalEmbeddingModel,
    FractalEmbeddingConfig,
    FractalEmbeddingTrainer,
)


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

class HierarchicalDataGenerator:
    """
    Generates data with known hierarchical structure.

    Creates a taxonomy:
    - Level 0: Super-categories (e.g., Animals, Vehicles)
    - Level 1: Categories (e.g., Mammals, Birds, Cars, Planes)
    - Level 2: Instances (e.g., Dog, Cat, Eagle, Sparrow)

    If fractal embeddings work, they should:
    - Scale 0 (coarse): Group super-categories
    - Scale 1 (medium): Group categories
    - Scale 2+ (fine): Distinguish instances
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        num_super_categories: int = 4,
        categories_per_super: int = 4,
        instances_per_category: int = 8,
        seq_len: int = 16,
    ):
        self.vocab_size = vocab_size
        self.num_super_categories = num_super_categories
        self.categories_per_super = categories_per_super
        self.instances_per_category = instances_per_category
        self.seq_len = seq_len

        # Build hierarchy
        self._build_hierarchy()

    def _build_hierarchy(self):
        """Build vocabulary with hierarchical structure."""
        # Reserve special tokens
        self.pad_token = 0
        self.cls_token = 1

        # Assign vocabulary ranges to hierarchy levels
        vocab_per_super = (self.vocab_size - 2) // self.num_super_categories

        self.hierarchy = {}  # instance_id -> (super_cat, cat, instance)
        self.super_to_vocab = {}  # super_cat -> vocab range
        self.cat_to_vocab = {}  # (super_cat, cat) -> vocab range

        token_id = 2
        for super_cat in range(self.num_super_categories):
            super_start = token_id
            vocab_per_cat = vocab_per_super // self.categories_per_super

            for cat in range(self.categories_per_super):
                cat_start = token_id
                vocab_per_instance = vocab_per_cat // self.instances_per_category

                for instance in range(self.instances_per_category):
                    instance_id = (super_cat, cat, instance)
                    instance_vocab = list(range(token_id, token_id + vocab_per_instance))
                    self.hierarchy[instance_id] = instance_vocab
                    token_id += vocab_per_instance

                self.cat_to_vocab[(super_cat, cat)] = list(range(cat_start, token_id))

            self.super_to_vocab[super_cat] = list(range(super_start, token_id))

        self.all_instances = list(self.hierarchy.keys())

    def generate_sample(self, instance_id: Tuple[int, int, int]) -> torch.Tensor:
        """Generate a sample sequence for an instance."""
        vocab = self.hierarchy[instance_id]
        tokens = [self.cls_token] + [random.choice(vocab) for _ in range(self.seq_len - 1)]
        return torch.tensor(tokens)

    def generate_batch(
        self,
        batch_size: int,
        same_super_prob: float = 0.3,
        same_cat_prob: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate a batch with controlled similarity structure.

        Returns:
            input_ids: (B, T)
            labels: Instance IDs for contrastive learning
            metadata: Hierarchy information
        """
        samples = []
        labels = []
        metadata = {'super_cat': [], 'cat': [], 'instance': []}

        for _ in range(batch_size):
            instance_id = random.choice(self.all_instances)
            super_cat, cat, instance = instance_id

            sample = self.generate_sample(instance_id)
            samples.append(sample)
            labels.append(self.all_instances.index(instance_id))
            metadata['super_cat'].append(super_cat)
            metadata['cat'].append(cat)
            metadata['instance'].append(instance)

        input_ids = torch.stack(samples)
        labels = torch.tensor(labels)

        return input_ids, labels, metadata

    def generate_similarity_pairs(
        self,
        num_pairs: int,
        relationship: str = 'same_instance',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate pairs with known relationships.

        relationship: 'same_instance', 'same_category', 'same_super', 'different'
        """
        samples_a = []
        samples_b = []
        similarities = []

        for _ in range(num_pairs):
            if relationship == 'same_instance':
                instance = random.choice(self.all_instances)
                a = self.generate_sample(instance)
                b = self.generate_sample(instance)
                sim = 1.0

            elif relationship == 'same_category':
                super_cat = random.randint(0, self.num_super_categories - 1)
                cat = random.randint(0, self.categories_per_super - 1)
                inst_a = random.randint(0, self.instances_per_category - 1)
                inst_b = random.randint(0, self.instances_per_category - 1)
                while inst_b == inst_a:
                    inst_b = random.randint(0, self.instances_per_category - 1)
                a = self.generate_sample((super_cat, cat, inst_a))
                b = self.generate_sample((super_cat, cat, inst_b))
                sim = 0.7

            elif relationship == 'same_super':
                super_cat = random.randint(0, self.num_super_categories - 1)
                cat_a = random.randint(0, self.categories_per_super - 1)
                cat_b = random.randint(0, self.categories_per_super - 1)
                while cat_b == cat_a:
                    cat_b = random.randint(0, self.categories_per_super - 1)
                inst_a = random.randint(0, self.instances_per_category - 1)
                inst_b = random.randint(0, self.instances_per_category - 1)
                a = self.generate_sample((super_cat, cat_a, inst_a))
                b = self.generate_sample((super_cat, cat_b, inst_b))
                sim = 0.4

            else:  # different
                instance_a = random.choice(self.all_instances)
                instance_b = random.choice(self.all_instances)
                while instance_b[0] == instance_a[0]:  # different super-category
                    instance_b = random.choice(self.all_instances)
                a = self.generate_sample(instance_a)
                b = self.generate_sample(instance_b)
                sim = 0.1

            samples_a.append(a)
            samples_b.append(b)
            similarities.append(sim)

        return (
            torch.stack(samples_a),
            torch.stack(samples_b),
            torch.tensor(similarities)
        )


# =============================================================================
# BASELINE MODEL (Non-Fractal)
# =============================================================================

class BaselineEmbeddingModel(nn.Module):
    """
    Standard embedding model without fractal structure.

    Uses same total parameters but without multi-scale extraction.
    This is our control to verify that fractal structure matters.
    """

    def __init__(self, config: FractalEmbeddingConfig):
        super().__init__()
        self.config = config

        # Match fractal model's embedding dimension
        self.total_dim = config.total_embed_dim

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02
        )

        # Stack of transformer layers (not shared)
        # Use same number as fractal scales for fair comparison
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
            )
            for _ in range(config.num_scales)
        ])

        # Project to final dimension
        self.pooling = config.pooling
        self.out_proj = nn.Linear(config.hidden_dim, self.total_dim)
        self.norm = nn.LayerNorm(self.total_dim)

        if self.pooling == "attention":
            self.attn_weights = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_scales: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape

        x = self.token_embed(input_ids) + self.pos_embed[:, :T, :]

        # Pass through all layers
        for layer in self.layers:
            x = layer(x)

        # Pool
        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == "attention":
            weights = self.attn_weights(x).squeeze(-1)
            if attention_mask is not None:
                weights = weights.masked_fill(~attention_mask.bool(), float('-inf'))
            weights = F.softmax(weights, dim=-1)
            pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = x.mean(dim=1)

        embedding = self.norm(self.out_proj(pooled))

        result = {'embedding': embedding}

        if return_all_scales:
            # Split embedding into "pseudo-scales" for comparison
            scale_dim = self.config.scale_dim
            result['scale_embeddings'] = [
                embedding[:, i*scale_dim:(i+1)*scale_dim]
                for i in range(self.config.num_scales)
            ]

        return result


# =============================================================================
# EVALUATION METRICS
# =============================================================================

class EmbeddingEvaluator:
    """Comprehensive evaluation of embedding quality."""

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def semantic_similarity_test(
        self,
        model: nn.Module,
        data_gen: HierarchicalDataGenerator,
        num_pairs: int = 100,
    ) -> Dict[str, float]:
        """
        Test if model captures semantic similarity at different levels.

        Expected: same_instance > same_category > same_super > different
        """
        model.eval()
        results = {}

        for relationship in ['same_instance', 'same_category', 'same_super', 'different']:
            a, b, expected_sim = data_gen.generate_similarity_pairs(num_pairs, relationship)
            a, b = a.to(self.device), b.to(self.device)

            with torch.no_grad():
                emb_a = model(a)['embedding']
                emb_b = model(b)['embedding']

                # Cosine similarity
                emb_a = F.normalize(emb_a, dim=-1)
                emb_b = F.normalize(emb_b, dim=-1)
                sim = (emb_a * emb_b).sum(dim=-1)

            results[f'sim_{relationship}'] = sim.mean().item()

        # Check ordering
        results['ordering_correct'] = (
            results['sim_same_instance'] >
            results['sim_same_category'] >
            results['sim_same_super'] >
            results['sim_different']
        )

        # Separation score (how well does it separate levels?)
        results['separation'] = (
            (results['sim_same_instance'] - results['sim_same_category']) +
            (results['sim_same_category'] - results['sim_same_super']) +
            (results['sim_same_super'] - results['sim_different'])
        ) / 3

        return results

    def retrieval_test(
        self,
        model: nn.Module,
        data_gen: HierarchicalDataGenerator,
        num_queries: int = 50,
        corpus_size: int = 500,
    ) -> Dict[str, float]:
        """
        Test retrieval performance.

        For each query, can we retrieve items from same category/super-category?
        """
        model.eval()

        # Build corpus
        corpus_ids, corpus_labels, corpus_meta = data_gen.generate_batch(corpus_size)
        corpus_ids = corpus_ids.to(self.device)

        with torch.no_grad():
            corpus_emb = model(corpus_ids)['embedding']
            corpus_emb = F.normalize(corpus_emb, dim=-1)

        # Generate queries and evaluate
        results = defaultdict(list)

        for _ in range(num_queries):
            # Pick a random instance as query
            query_instance = random.choice(data_gen.all_instances)
            query_sample = data_gen.generate_sample(query_instance).unsqueeze(0).to(self.device)

            with torch.no_grad():
                query_emb = model(query_sample)['embedding']
                query_emb = F.normalize(query_emb, dim=-1)

            # Compute similarities to corpus
            sims = (query_emb @ corpus_emb.T).squeeze(0)

            # Get top-k
            top_k = 10
            _, top_indices = sims.topk(top_k)

            # Check retrieval quality
            query_super, query_cat, _ = query_instance

            same_cat_count = 0
            same_super_count = 0

            for idx in top_indices.cpu().numpy():
                ret_super = corpus_meta['super_cat'][idx]
                ret_cat = corpus_meta['cat'][idx]

                if ret_super == query_super and ret_cat == query_cat:
                    same_cat_count += 1
                elif ret_super == query_super:
                    same_super_count += 1

            results['recall_same_cat@10'].append(same_cat_count / top_k)
            results['recall_same_super@10'].append((same_cat_count + same_super_count) / top_k)

        return {k: np.mean(v) for k, v in results.items()}

    def scale_analysis(
        self,
        model: FractalEmbeddingModel,
        data_gen: HierarchicalDataGenerator,
        num_samples: int = 200,
    ) -> Dict[str, float]:
        """
        Analyze what each scale captures.

        Hypothesis: Coarser scales should better distinguish super-categories,
        finer scales should better distinguish instances.
        """
        model.eval()

        # Generate samples
        input_ids, labels, meta = data_gen.generate_batch(num_samples)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            result = model(input_ids, return_all_scales=True)
            scale_embeddings = result['scale_embeddings']

        results = {}

        for scale_idx, scale_emb in enumerate(scale_embeddings):
            scale_emb = F.normalize(scale_emb, dim=-1).cpu()

            # Measure clustering quality at each hierarchy level
            # Using average intra-class vs inter-class similarity

            # Super-category clustering
            super_sims_intra = []
            super_sims_inter = []

            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    sim = (scale_emb[i] * scale_emb[j]).sum().item()
                    if meta['super_cat'][i] == meta['super_cat'][j]:
                        super_sims_intra.append(sim)
                    else:
                        super_sims_inter.append(sim)

            if super_sims_intra and super_sims_inter:
                results[f'scale_{scale_idx}_super_separation'] = (
                    np.mean(super_sims_intra) - np.mean(super_sims_inter)
                )

            # Category clustering
            cat_sims_intra = []
            cat_sims_inter = []

            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    sim = (scale_emb[i] * scale_emb[j]).sum().item()
                    if (meta['super_cat'][i] == meta['super_cat'][j] and
                        meta['cat'][i] == meta['cat'][j]):
                        cat_sims_intra.append(sim)
                    elif meta['super_cat'][i] == meta['super_cat'][j]:
                        cat_sims_inter.append(sim)

            if cat_sims_intra and cat_sims_inter:
                results[f'scale_{scale_idx}_cat_separation'] = (
                    np.mean(cat_sims_intra) - np.mean(cat_sims_inter)
                )

        return results

    def matryoshka_test(
        self,
        model: FractalEmbeddingModel,
        data_gen: HierarchicalDataGenerator,
        num_pairs: int = 100,
    ) -> Dict[str, float]:
        """
        Test Matryoshka property: Can we use fewer dimensions and still perform?

        This tests the nested embedding quality at different sizes.
        """
        model.eval()

        # Generate pairs
        a, b, _ = data_gen.generate_similarity_pairs(num_pairs, 'same_category')
        c, d, _ = data_gen.generate_similarity_pairs(num_pairs, 'different')
        a, b, c, d = a.to(self.device), b.to(self.device), c.to(self.device), d.to(self.device)

        with torch.no_grad():
            result_a = model(a, return_all_scales=True)
            result_b = model(b, return_all_scales=True)
            result_c = model(c, return_all_scales=True)
            result_d = model(d, return_all_scales=True)

        results = {}

        # Test at each nested size
        for i, (nested_a, nested_b, nested_c, nested_d) in enumerate(zip(
            result_a['nested_embeddings'],
            result_b['nested_embeddings'],
            result_c['nested_embeddings'],
            result_d['nested_embeddings'],
        )):
            nested_a = F.normalize(nested_a, dim=-1)
            nested_b = F.normalize(nested_b, dim=-1)
            nested_c = F.normalize(nested_c, dim=-1)
            nested_d = F.normalize(nested_d, dim=-1)

            # Same category similarity
            same_sim = (nested_a * nested_b).sum(dim=-1).mean().item()

            # Different similarity
            diff_sim = (nested_c * nested_d).sum(dim=-1).mean().item()

            # Discrimination
            results[f'nested_{i+1}_discrimination'] = same_sim - diff_sim
            results[f'nested_{i+1}_same_sim'] = same_sim
            results[f'nested_{i+1}_diff_sim'] = diff_sim

        return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_full_evaluation(
    num_epochs: int = 20,
    batch_size: int = 64,
    device: str = None,
):
    """
    Full comparison: Fractal vs Baseline embeddings.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("FRACTAL EMBEDDINGS EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")

    # Configuration
    config = FractalEmbeddingConfig(
        vocab_size=2000,
        hidden_dim=128,
        num_heads=4,
        num_scales=4,
        scale_dim=32,
        max_seq_len=32,
    )

    print(f"\nConfiguration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num scales: {config.num_scales}")
    print(f"  Scale dim: {config.scale_dim}")
    print(f"  Total embed dim: {config.total_embed_dim}")

    # Create models
    fractal_model = FractalEmbeddingModel(config).to(device)
    baseline_model = BaselineEmbeddingModel(config).to(device)

    fractal_params = sum(p.numel() for p in fractal_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())

    print(f"\nModel parameters:")
    print(f"  Fractal: {fractal_params:,}")
    print(f"  Baseline: {baseline_params:,}")

    # Data generator
    data_gen = HierarchicalDataGenerator(
        vocab_size=config.vocab_size,
        num_super_categories=4,
        categories_per_super=4,
        instances_per_category=8,
        seq_len=config.max_seq_len,
    )

    print(f"\nHierarchy:")
    print(f"  {data_gen.num_super_categories} super-categories")
    print(f"  {data_gen.categories_per_super} categories each")
    print(f"  {data_gen.instances_per_category} instances each")
    print(f"  Total instances: {len(data_gen.all_instances)}")

    # Trainers
    fractal_trainer = FractalEmbeddingTrainer(fractal_model, lr=1e-3)
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-3)

    # Evaluator
    evaluator = EmbeddingEvaluator(device)

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(num_epochs):
        fractal_model.train()
        baseline_model.train()

        epoch_losses = {'fractal': [], 'baseline': []}

        for _ in range(20):  # batches per epoch
            input_ids, labels, _ = data_gen.generate_batch(batch_size)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = torch.ones_like(input_ids)

            # Train fractal model
            metrics = fractal_trainer.train_step(input_ids, attention_mask, labels)
            epoch_losses['fractal'].append(metrics['total_loss'])

            # Train baseline model
            baseline_optimizer.zero_grad()
            baseline_result = baseline_model(input_ids, attention_mask, return_all_scales=True)

            # Simple contrastive loss for baseline
            embeddings = baseline_result['embedding']
            embeddings = F.normalize(embeddings, dim=-1)
            sim_matrix = embeddings @ embeddings.T / 0.05
            loss = F.cross_entropy(sim_matrix, torch.arange(batch_size, device=device))
            loss.backward()
            baseline_optimizer.step()
            epoch_losses['baseline'].append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Fractal loss={np.mean(epoch_losses['fractal']):.4f}, "
                  f"Baseline loss={np.mean(epoch_losses['baseline']):.4f}")

    # Evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    results = {'fractal': {}, 'baseline': {}}

    # 1. Semantic Similarity
    print("\n1. SEMANTIC SIMILARITY TEST")
    print("-" * 40)

    for name, model in [('fractal', fractal_model), ('baseline', baseline_model)]:
        sim_results = evaluator.semantic_similarity_test(model, data_gen)
        results[name]['similarity'] = sim_results

        print(f"\n{name.upper()}:")
        print(f"  Same instance:  {sim_results['sim_same_instance']:.4f}")
        print(f"  Same category:  {sim_results['sim_same_category']:.4f}")
        print(f"  Same super-cat: {sim_results['sim_same_super']:.4f}")
        print(f"  Different:      {sim_results['sim_different']:.4f}")
        print(f"  Ordering correct: {sim_results['ordering_correct']}")
        print(f"  Separation score: {sim_results['separation']:.4f}")

    # 2. Retrieval
    print("\n2. RETRIEVAL TEST")
    print("-" * 40)

    for name, model in [('fractal', fractal_model), ('baseline', baseline_model)]:
        ret_results = evaluator.retrieval_test(model, data_gen)
        results[name]['retrieval'] = ret_results

        print(f"\n{name.upper()}:")
        for k, v in ret_results.items():
            print(f"  {k}: {v:.4f}")

    # 3. Scale Analysis (Fractal only)
    print("\n3. SCALE ANALYSIS (Fractal model)")
    print("-" * 40)

    scale_results = evaluator.scale_analysis(fractal_model, data_gen)
    results['fractal']['scale_analysis'] = scale_results

    print("\nSuper-category separation by scale:")
    for i in range(config.num_scales):
        key = f'scale_{i}_super_separation'
        if key in scale_results:
            print(f"  Scale {i}: {scale_results[key]:.4f}")

    print("\nCategory separation by scale:")
    for i in range(config.num_scales):
        key = f'scale_{i}_cat_separation'
        if key in scale_results:
            print(f"  Scale {i}: {scale_results[key]:.4f}")

    # 4. Matryoshka Test
    print("\n4. MATRYOSHKA TEST (Fractal model)")
    print("-" * 40)

    mat_results = evaluator.matryoshka_test(fractal_model, data_gen)
    results['fractal']['matryoshka'] = mat_results

    print("\nDiscrimination by nested size:")
    for i in range(config.num_scales):
        key = f'nested_{i+1}_discrimination'
        if key in mat_results:
            dims = config.scale_dim * (i + 1)
            print(f"  {i+1} scale(s) ({dims:3d} dims): {mat_results[key]:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    fractal_sep = results['fractal']['similarity']['separation']
    baseline_sep = results['baseline']['similarity']['separation']
    fractal_ret = results['fractal']['retrieval']['recall_same_cat@10']
    baseline_ret = results['baseline']['retrieval']['recall_same_cat@10']

    print(f"\nSeparation Score: Fractal={fractal_sep:.4f}, Baseline={baseline_sep:.4f}")
    print(f"  Fractal {'better' if fractal_sep > baseline_sep else 'worse'} by {abs(fractal_sep - baseline_sep):.4f}")

    print(f"\nRetrieval (same-cat@10): Fractal={fractal_ret:.4f}, Baseline={baseline_ret:.4f}")
    print(f"  Fractal {'better' if fractal_ret > baseline_ret else 'worse'} by {abs(fractal_ret - baseline_ret):.4f}")

    # Key insight
    if 'scale_analysis' in results['fractal']:
        scale_res = results['fractal']['scale_analysis']
        super_seps = [scale_res.get(f'scale_{i}_super_separation', 0) for i in range(config.num_scales)]
        cat_seps = [scale_res.get(f'scale_{i}_cat_separation', 0) for i in range(config.num_scales)]

        if super_seps and cat_seps:
            print("\nKey Fractal Insight:")
            best_super_scale = np.argmax(super_seps)
            best_cat_scale = np.argmax(cat_seps)
            print(f"  Best scale for super-categories: Scale {best_super_scale}")
            print(f"  Best scale for categories: Scale {best_cat_scale}")

    return results


if __name__ == '__main__':
    results = run_full_evaluation(num_epochs=20, batch_size=64)
