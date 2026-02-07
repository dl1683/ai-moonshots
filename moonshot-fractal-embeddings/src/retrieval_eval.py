"""
Hierarchical Retrieval Evaluation
===================================

Evaluates whether fractal embeddings enable "steerable" retrieval:
- Short prefixes (j=1,2) → better at coarse-level retrieval
- Full embeddings (j=4) → better at fine-level retrieval

This is the key differentiator: MRL uses same loss everywhere,
so prefixes should NOT specialize. V5 uses hierarchy-aligned losses,
so short prefixes SHOULD specialize for coarse retrieval.

Key metric: SteerabilityScore = CoarseGain + FineGain
  CoarseGain = Recall_L0@10(j=1) - Recall_L0@10(j=4)
  FineGain   = Recall_L1@10(j=4) - Recall_L1@10(j=1)

Success: V5 SteerabilityScore > MRL SteerabilityScore
"""

import numpy as np
import torch
import gc
import json
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def encode_with_prefix(model, texts, prefix_len=None, batch_size=32):
    """Encode texts using a trained fractal model, optionally at a prefix length."""
    emb = model.encode(texts, batch_size=batch_size, prefix_len=prefix_len).numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def hierarchical_relevance(query_l0, query_l1, doc_l0, doc_l1):
    """
    Compute hierarchical relevance:
      2 = same L1 (exact match)
      1 = same L0, different L1 (coarse match)
      0 = different L0
    """
    if query_l1 == doc_l1:
        return 2
    elif query_l0 == doc_l0:
        return 1
    else:
        return 0


def recall_at_k(query_embs, doc_embs, query_labels, doc_labels, k=10, level="l0"):
    """
    Compute Recall@k for a given hierarchy level.
    For each query, check if at least one relevant doc appears in top-k.
    """
    n_queries = len(query_embs)
    hits = 0

    # Compute all similarities at once
    sims = query_embs @ doc_embs.T  # (n_queries, n_docs)

    for i in range(n_queries):
        top_k_indices = np.argsort(-sims[i])[:k]
        query_label = query_labels[i]
        retrieved_labels = doc_labels[top_k_indices]

        if np.any(retrieved_labels == query_label):
            hits += 1

    return hits / n_queries


def ndcg_hierarchical(query_embs, doc_embs, query_l0, query_l1, doc_l0, doc_l1, k=10):
    """
    Compute NDCG@k with hierarchical relevance (rel=2 for same L1, 1 for same L0).
    """
    n_queries = len(query_embs)
    sims = query_embs @ doc_embs.T

    ndcg_scores = []
    for i in range(n_queries):
        top_k_indices = np.argsort(-sims[i])[:k]

        # Compute gains
        gains = []
        for idx in top_k_indices:
            rel = hierarchical_relevance(query_l0[i], query_l1[i], doc_l0[idx], doc_l1[idx])
            gains.append(rel)

        # DCG
        dcg = sum(g / np.log2(pos + 2) for pos, g in enumerate(gains))

        # Ideal DCG: sort all docs by relevance
        all_rels = []
        for idx in range(len(doc_l0)):
            rel = hierarchical_relevance(query_l0[i], query_l1[i], doc_l0[idx], doc_l1[idx])
            all_rels.append(rel)
        all_rels_sorted = sorted(all_rels, reverse=True)[:k]
        idcg = sum(g / np.log2(pos + 2) for pos, g in enumerate(all_rels_sorted))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return float(np.mean(ndcg_scores))


def mrr_at_k(query_embs, doc_embs, query_labels, doc_labels, k=10):
    """Mean Reciprocal Rank for L1 labels."""
    n_queries = len(query_embs)
    sims = query_embs @ doc_embs.T

    rr_sum = 0
    for i in range(n_queries):
        top_k_indices = np.argsort(-sims[i])[:k]
        for rank, idx in enumerate(top_k_indices, 1):
            if doc_labels[idx] == query_labels[i]:
                rr_sum += 1.0 / rank
                break

    return rr_sum / n_queries


def evaluate_retrieval_at_prefix(model, query_samples, doc_samples, prefix_len=None, k=10):
    """
    Evaluate retrieval at a specific prefix length.
    Returns dict with Recall_L0@k, Recall_L1@k, NDCG_hier@k, MRR_L1.
    """
    query_texts = [s.text for s in query_samples]
    doc_texts = [s.text for s in doc_samples]

    query_l0 = np.array([s.level0_label for s in query_samples])
    query_l1 = np.array([s.level1_label for s in query_samples])
    doc_l0 = np.array([s.level0_label for s in doc_samples])
    doc_l1 = np.array([s.level1_label for s in doc_samples])

    query_embs = encode_with_prefix(model, query_texts, prefix_len=prefix_len)
    doc_embs = encode_with_prefix(model, doc_texts, prefix_len=prefix_len)

    return {
        f"recall_l0@{k}": recall_at_k(query_embs, doc_embs, query_l0, doc_l0, k=k, level="l0"),
        f"recall_l1@{k}": recall_at_k(query_embs, doc_embs, query_l1, doc_l1, k=k, level="l1"),
        f"ndcg_hier@{k}": ndcg_hierarchical(query_embs, doc_embs, query_l0, query_l1, doc_l0, doc_l1, k=k),
        f"mrr_l1": mrr_at_k(query_embs, doc_embs, query_l1, doc_l1, k=k),
    }


def compute_steerability(prefix_results, k=10):
    """
    Compute SteerabilityScore from prefix-level retrieval results.

    SteerabilityScore = CoarseGain + FineGain
      CoarseGain = Recall_L0@k(j=1) - Recall_L0@k(j=4)   [short prefix better for coarse]
      FineGain   = Recall_L1@k(j=4) - Recall_L1@k(j=1)   [full embedding better for fine]
    """
    r_l0_j1 = prefix_results.get("j1", {}).get(f"recall_l0@{k}", 0)
    r_l0_j4 = prefix_results.get("j4", {}).get(f"recall_l0@{k}", 0)
    r_l1_j1 = prefix_results.get("j1", {}).get(f"recall_l1@{k}", 0)
    r_l1_j4 = prefix_results.get("j4", {}).get(f"recall_l1@{k}", 0)

    coarse_gain = r_l0_j1 - r_l0_j4
    fine_gain = r_l1_j4 - r_l1_j1

    return {
        "coarse_gain": coarse_gain,
        "fine_gain": fine_gain,
        "steerability_score": coarse_gain + fine_gain,
        "recall_l0_j1": r_l0_j1,
        "recall_l0_j4": r_l0_j4,
        "recall_l1_j1": r_l1_j1,
        "recall_l1_j4": r_l1_j4,
    }


def run_retrieval_eval(
    model_key="bge-small",
    dataset_name="yahoo",
    stage1_epochs=5,
    batch_size=32,
    seed=42,
    k=10,
    max_docs=5000,
    max_queries=500,
    device="cuda",
):
    """
    Full retrieval evaluation: train V5 + MRL, evaluate at each prefix length.
    """
    from hierarchical_datasets import load_hierarchical_dataset
    from fractal_v5 import FractalModelV5, V5Trainer, ContrastiveDatasetV5, split_train_val
    from mrl_v5_baseline import MRLTrainerV5
    from multi_model_pipeline import MODELS, load_model

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*70}")
    print(f"RETRIEVAL EVALUATION: {model_key} on {dataset_name}")
    print(f"{'='*70}")

    # Load data
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_docs + max_queries)

    train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

    class TempDataset:
        def __init__(self, samples, level0_names, level1_names):
            self.samples = samples
            self.level0_names = level0_names
            self.level1_names = level1_names

    val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
    train_data.samples = train_samples

    num_l0 = len(train_data.level0_names)
    num_l1 = len(train_data.level1_names)

    # Split test into docs and queries
    test_samples = test_data.samples
    np.random.shuffle(test_samples)
    query_samples = test_samples[:max_queries]
    doc_samples = test_samples[max_queries : max_queries + max_docs]

    if len(doc_samples) < 100:
        # Not enough data for separate queries/docs; use all as both
        query_samples = test_samples[:max_queries]
        doc_samples = test_samples  # queries search through all docs (self included)

    print(f"  Docs: {len(doc_samples)}, Queries: {len(query_samples)}")

    config = MODELS[model_key]
    all_results = {}

    # =================== TRAIN AND EVAL V5 ===================
    print(f"\n--- Training V5 ---")
    v5_model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    v5_trainer = V5Trainer(
        model=v5_model,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
    )
    v5_trainer.train(batch_size=batch_size, patience=5)

    print("  Evaluating V5 retrieval at each prefix length...")
    v5_prefix_results = {}
    for j in [1, 2, 3, 4]:
        prefix_len = j if j < 4 else None
        metrics = evaluate_retrieval_at_prefix(
            v5_model, query_samples, doc_samples, prefix_len=prefix_len, k=k
        )
        v5_prefix_results[f"j{j}"] = metrics
        print(f"    j={j}: R_L0@{k}={metrics[f'recall_l0@{k}']:.4f}, "
              f"R_L1@{k}={metrics[f'recall_l1@{k}']:.4f}, "
              f"NDCG={metrics[f'ndcg_hier@{k}']:.4f}")

    v5_steerability = compute_steerability(v5_prefix_results, k=k)
    all_results["v5"] = {
        "prefix_results": v5_prefix_results,
        "steerability": v5_steerability,
    }

    del v5_model, v5_trainer
    torch.cuda.empty_cache()
    gc.collect()

    # =================== TRAIN AND EVAL MRL ===================
    print(f"\n--- Training MRL ---")
    mrl_model = FractalModelV5(
        config=config,
        num_l0_classes=num_l1,  # KEY: both heads output L1 logits
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    mrl_trainer = MRLTrainerV5(
        model=mrl_model,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
    )
    mrl_trainer.train(batch_size=batch_size, patience=5)

    print("  Evaluating MRL retrieval at each prefix length...")
    mrl_prefix_results = {}
    for j in [1, 2, 3, 4]:
        prefix_len = j if j < 4 else None
        metrics = evaluate_retrieval_at_prefix(
            mrl_model, query_samples, doc_samples, prefix_len=prefix_len, k=k
        )
        mrl_prefix_results[f"j{j}"] = metrics
        print(f"    j={j}: R_L0@{k}={metrics[f'recall_l0@{k}']:.4f}, "
              f"R_L1@{k}={metrics[f'recall_l1@{k}']:.4f}, "
              f"NDCG={metrics[f'ndcg_hier@{k}']:.4f}")

    mrl_steerability = compute_steerability(mrl_prefix_results, k=k)
    all_results["mrl"] = {
        "prefix_results": mrl_prefix_results,
        "steerability": mrl_steerability,
    }

    del mrl_model, mrl_trainer
    torch.cuda.empty_cache()
    gc.collect()

    # =================== SUMMARY ===================
    print(f"\n{'='*70}")
    print("RETRIEVAL STEERABILITY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'V5':>10} {'MRL':>10} {'Delta':>10}")
    print("-" * 55)
    for metric in ["coarse_gain", "fine_gain", "steerability_score"]:
        v5_val = v5_steerability[metric]
        mrl_val = mrl_steerability[metric]
        delta = v5_val - mrl_val
        print(f"{metric:<25} {v5_val:>10.4f} {mrl_val:>10.4f} {delta:>+10.4f}")

    print(f"\n{'Prefix':<10} {'V5 R_L0':>10} {'MRL R_L0':>10} {'V5 R_L1':>10} {'MRL R_L1':>10}")
    print("-" * 50)
    for j in [1, 2, 3, 4]:
        v5_l0 = v5_prefix_results[f"j{j}"][f"recall_l0@{k}"]
        mrl_l0 = mrl_prefix_results[f"j{j}"][f"recall_l0@{k}"]
        v5_l1 = v5_prefix_results[f"j{j}"][f"recall_l1@{k}"]
        mrl_l1 = mrl_prefix_results[f"j{j}"][f"recall_l1@{k}"]
        print(f"j={j:<7} {v5_l0:>10.4f} {mrl_l0:>10.4f} {v5_l1:>10.4f} {mrl_l1:>10.4f}")

    winner = "V5" if v5_steerability["steerability_score"] > mrl_steerability["steerability_score"] else "MRL"
    print(f"\nSteerability winner: {winner}")

    # Save
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"retrieval_{model_key}_{dataset_name}_seed{seed}.json"
    output = {
        "model": model_key,
        "dataset": dataset_name,
        "seed": seed,
        "k": k,
        "timestamp": datetime.now().isoformat(),
        **convert(all_results),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical retrieval evaluation")
    parser.add_argument("--model", type=str, default="bge-small")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    run_retrieval_eval(
        model_key=args.model,
        dataset_name=args.dataset,
        stage1_epochs=args.epochs,
        seed=args.seed,
        k=args.k,
    )
