"""
Retrieval Benchmark for Fractal Embeddings (V5) vs MRL
=======================================================

Measures Recall@k and MRR at each prefix length j=1..4 (64d..256d)
at BOTH L0 (coarse) and L1 (fine) granularity.

Protocol:
1. Train V5 and MRL models (3 seeds each) on train split
2. Split test data into queries (50%) and documents (50%)
3. At each prefix length j:
   - Compute cosine similarity between each query and all documents
   - Rank documents by similarity
   - Recall@k (k=1,5,10,20): fraction of queries where at least one of
     the top-k docs shares the query's L0/L1 label
   - MRR: mean reciprocal rank of the first doc sharing the query's label

Expected findings:
- V5 at j=1 (64d): HIGH L0 Recall (coarse specialization), moderate L1 Recall
- V5 at j=4 (256d): HIGH L1 Recall (fine retrieval recovered)
- MRL at j=1 (64d): moderate L0 AND L1 Recall (no specialization)
- MRL at j=4 (256d): similar L1 Recall to V5 at 256d
"""

import sys
import os
import json
import gc
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from mrl_v5_baseline import MRLTrainerV5


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def compute_embeddings_all_j(model, texts: List[str], batch_size: int = 32) -> Dict[int, np.ndarray]:
    """Compute L2-normalised embeddings at every prefix length j=1..4."""
    embeddings = {}
    for j in [1, 2, 3, 4]:
        prefix_len = j if j < 4 else None
        emb = model.encode(texts, batch_size=batch_size, prefix_len=prefix_len).numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        emb = emb / norms
        embeddings[j] = emb
    return embeddings


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def retrieval_metrics(
    query_embs: np.ndarray,
    doc_embs: np.ndarray,
    query_labels: np.ndarray,
    doc_labels: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """
    Compute Recall@k and MRR for a single label level.

    Recall@k = fraction of queries where *at least one* of the top-k docs
               shares the query's label.
    MRR      = mean over queries of 1/rank, where rank is the position of the
               first doc that shares the query's label.
    """
    n_queries = query_embs.shape[0]
    n_docs = doc_embs.shape[0]

    # Similarity matrix  [n_queries, n_docs]
    sims = query_embs @ doc_embs.T  # cosine (already L2-normalised)

    # Argsort descending for each query
    ranked_indices = np.argsort(-sims, axis=1)  # [n_queries, n_docs]

    recall = {k: 0 for k in ks}
    reciprocal_ranks = []

    for i in range(n_queries):
        q_label = query_labels[i]
        ranked_doc_labels = doc_labels[ranked_indices[i]]
        matches = (ranked_doc_labels == q_label)

        # Recall@k
        for k in ks:
            if matches[:k].any():
                recall[k] += 1

        # MRR -- first match position (0-indexed)
        match_positions = np.where(matches)[0]
        if len(match_positions) > 0:
            reciprocal_ranks.append(1.0 / (match_positions[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall[k] / n_queries
    metrics["mrr"] = float(np.mean(reciprocal_ranks))
    return metrics


def evaluate_retrieval(
    query_embs_by_j: Dict[int, np.ndarray],
    doc_embs_by_j: Dict[int, np.ndarray],
    query_l0: np.ndarray,
    query_l1: np.ndarray,
    doc_l0: np.ndarray,
    doc_l1: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict:
    """
    Evaluate retrieval metrics at every prefix length j=1..4 for both L0 and L1.
    Returns a nested dict: results[j]["L0"] / results[j]["L1"] = {recall@k, mrr}
    """
    results = {}
    for j in [1, 2, 3, 4]:
        q_emb = query_embs_by_j[j]
        d_emb = doc_embs_by_j[j]

        l0_metrics = retrieval_metrics(q_emb, d_emb, query_l0, doc_l0, ks)
        l1_metrics = retrieval_metrics(q_emb, d_emb, query_l1, doc_l1, ks)

        results[j] = {"L0": l0_metrics, "L1": l1_metrics}
    return results


# ---------------------------------------------------------------------------
# Training helpers (mirror adaptive_retrieval.py patterns)
# ---------------------------------------------------------------------------

class TempDataset:
    """Lightweight wrapper used for val/test datasets."""
    def __init__(self, samples, l0_names, l1_names):
        self.samples = samples
        self.level0_names = l0_names
        self.level1_names = l1_names


def train_v5_model(config, train_data, val_data, num_l0, num_l1, device, stage1_epochs=5):
    """Train a V5 model and return it (eval mode)."""
    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l0,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    trainer = V5Trainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=0,  # head-only (matches existing experiments)
    )
    trainer.train(batch_size=16, patience=5)
    model.eval()
    return model


def train_mrl_model(config, train_data, val_data, num_l1, device, stage1_epochs=5):
    """Train an MRL model and return it (eval mode)."""
    # MRL: num_l0_classes = num_l1 so head_top also outputs L1 logits
    model = FractalModelV5(
        config=config,
        num_l0_classes=num_l1,
        num_l1_classes=num_l1,
        num_scales=4,
        scale_dim=64,
        device=device,
    ).to(device)

    trainer = MRLTrainerV5(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=0,
    )
    trainer.train(batch_size=16, patience=5)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_retrieval_table(results: Dict, method: str, dataset: str):
    """Print a nicely formatted table of retrieval results for one method/seed."""
    print(f"\n  {method.upper()} on {dataset}")
    header = (
        f"  {'j':>3} {'dims':>5}  |"
        f"  {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR':>6}  |"
        f"  {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR':>6}"
    )
    print(f"  {'':>9}  | {'L0 (coarse)':^38} | {'L1 (fine)':^38}")
    print(header)
    print("  " + "-" * len(header.strip()))

    for j in [1, 2, 3, 4]:
        l0 = results[j]["L0"]
        l1 = results[j]["L1"]
        print(
            f"  {j:>3} {j*64:>4}d  |"
            f"  {l0['recall@1']:6.3f} {l0['recall@5']:6.3f}"
            f" {l0['recall@10']:6.3f} {l0['recall@20']:6.3f}"
            f" {l0['mrr']:6.3f}  |"
            f"  {l1['recall@1']:6.3f} {l1['recall@5']:6.3f}"
            f" {l1['recall@10']:6.3f} {l1['recall@20']:6.3f}"
            f" {l1['mrr']:6.3f}"
        )


def print_comparison_table(all_results: Dict, dataset: str, seeds: List[int]):
    """Print averaged comparison across seeds: V5 vs MRL, per j."""
    print(f"\n{'='*80}")
    print(f"  RETRIEVAL BENCHMARK COMPARISON ({dataset.upper()}) -- averaged over {len(seeds)} seeds")
    print(f"{'='*80}")

    for level in ["L0", "L1"]:
        print(f"\n  --- {level} ---")
        header = f"  {'j':>3} {'dims':>5}  | {'V5 R@1':>7} {'V5 R@10':>8} {'V5 MRR':>7}  | {'MRL R@1':>8} {'MRL R@10':>9} {'MRL MRR':>8}  | {'dR@1':>6} {'dR@10':>7} {'dMRR':>6}"
        print(header)
        print("  " + "-" * len(header.strip()))

        for j in [1, 2, 3, 4]:
            v5_vals = {m: [] for m in ["recall@1", "recall@10", "mrr"]}
            mrl_vals = {m: [] for m in ["recall@1", "recall@10", "mrr"]}

            for seed in seeds:
                sk = str(seed)
                for m in v5_vals:
                    v5_vals[m].append(all_results[sk]["v5"][j][level][m])
                    mrl_vals[m].append(all_results[sk]["mrl"][j][level][m])

            v5_mean = {m: np.mean(v5_vals[m]) for m in v5_vals}
            mrl_mean = {m: np.mean(mrl_vals[m]) for m in mrl_vals}
            delta = {m: v5_mean[m] - mrl_mean[m] for m in v5_vals}

            print(
                f"  {j:>3} {j*64:>4}d  |"
                f"  {v5_mean['recall@1']:7.3f} {v5_mean['recall@10']:8.3f} {v5_mean['mrr']:7.3f}  |"
                f"  {mrl_mean['recall@1']:8.3f} {mrl_mean['recall@10']:9.3f} {mrl_mean['mrr']:8.3f}  |"
                f"  {delta['recall@1']:+6.3f} {delta['recall@10']:+7.3f} {delta['mrr']:+6.3f}"
            )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_retrieval_benchmark(
    model_key: str = "bge-small",
    dataset_name: str = "clinc",
    seeds: Tuple[int, ...] = (42, 123, 456),
    stage1_epochs: int = 5,
    device: str = "cuda",
    max_train_samples: int = 10000,
    max_test_samples: int = 2000,
    ks: Tuple[int, ...] = (1, 5, 10, 20),
):
    """Run the full retrieval benchmark: V5 vs MRL across seeds and prefix lengths."""
    print("=" * 80)
    print(f"RETRIEVAL BENCHMARK: {model_key} on {dataset_name}")
    print(f"Seeds: {seeds}  |  Stage-1 epochs: {stage1_epochs}  |  ks: {ks}")
    print("=" * 80)

    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    config = MODELS[model_key]

    # We store per-seed retrieval dicts keyed by string seed
    all_seed_results: Dict[str, Dict] = {}

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        # Seed everything
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ------------------------------------------------------------------
        # 1. Load data
        # ------------------------------------------------------------------
        print("[1] Loading data...")
        train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=max_train_samples)
        test_data  = load_hierarchical_dataset(dataset_name, split="test",  max_samples=max_test_samples)

        train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)
        val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
        train_data.samples = train_samples

        num_l0 = len(train_data.level0_names)
        num_l1 = len(train_data.level1_names)
        print(f"  Train={len(train_data.samples)}, Val={len(val_data.samples)}, Test={len(test_data.samples)}")
        print(f"  Hierarchy: {num_l0} L0 -> {num_l1} L1")

        # ------------------------------------------------------------------
        # 2. Train V5
        # ------------------------------------------------------------------
        print("\n[2] Training V5...")
        v5_model = train_v5_model(config, train_data, val_data, num_l0, num_l1, device, stage1_epochs)

        # ------------------------------------------------------------------
        # 3. Train MRL
        # ------------------------------------------------------------------
        print("\n[3] Training MRL...")
        mrl_model = train_mrl_model(config, train_data, val_data, num_l1, device, stage1_epochs)

        # ------------------------------------------------------------------
        # 4. Compute embeddings on test set
        # ------------------------------------------------------------------
        print("\n[4] Computing embeddings...")
        test_texts = [s.text for s in test_data.samples]
        test_l0 = np.array([s.level0_label for s in test_data.samples])
        test_l1 = np.array([s.level1_label for s in test_data.samples])

        v5_embs  = compute_embeddings_all_j(v5_model, test_texts, batch_size=32)
        mrl_embs = compute_embeddings_all_j(mrl_model, test_texts, batch_size=32)

        # Free GPU memory before the numpy-heavy retrieval step
        del v5_model, mrl_model
        torch.cuda.empty_cache()
        gc.collect()

        # ------------------------------------------------------------------
        # 5. Split test into queries (50%) and docs (50%)
        # ------------------------------------------------------------------
        n = len(test_texts)
        perm = np.random.permutation(n)
        q_idx = perm[: n // 2]
        d_idx = perm[n // 2 :]

        query_l0 = test_l0[q_idx]
        query_l1 = test_l1[q_idx]
        doc_l0   = test_l0[d_idx]
        doc_l1   = test_l1[d_idx]

        # ------------------------------------------------------------------
        # 6. Evaluate retrieval
        # ------------------------------------------------------------------
        print("\n[5] Evaluating retrieval...")

        v5_query_embs  = {j: v5_embs[j][q_idx]  for j in [1, 2, 3, 4]}
        v5_doc_embs    = {j: v5_embs[j][d_idx]   for j in [1, 2, 3, 4]}
        mrl_query_embs = {j: mrl_embs[j][q_idx]  for j in [1, 2, 3, 4]}
        mrl_doc_embs   = {j: mrl_embs[j][d_idx]  for j in [1, 2, 3, 4]}

        v5_results = evaluate_retrieval(
            v5_query_embs, v5_doc_embs, query_l0, query_l1, doc_l0, doc_l1, ks,
        )
        mrl_results = evaluate_retrieval(
            mrl_query_embs, mrl_doc_embs, query_l0, query_l1, doc_l0, doc_l1, ks,
        )

        # Pretty print
        print_retrieval_table(v5_results, "V5", dataset_name)
        print_retrieval_table(mrl_results, "MRL", dataset_name)

        # Store
        # Convert int keys to str for JSON safety later
        def to_serialisable(d):
            out = {}
            for k, v in d.items():
                key = str(k) if isinstance(k, int) else k
                if isinstance(v, dict):
                    out[key] = to_serialisable(v)
                elif isinstance(v, (np.floating, np.integer)):
                    out[key] = float(v)
                elif isinstance(v, np.ndarray):
                    out[key] = v.tolist()
                else:
                    out[key] = v
            return out

        all_seed_results[str(seed)] = {
            "v5":  to_serialisable(v5_results),
            "mrl": to_serialisable(mrl_results),
        }

        # Cleanup arrays
        del v5_embs, mrl_embs
        gc.collect()

    # ------------------------------------------------------------------
    # 7. Aggregated comparison
    # ------------------------------------------------------------------
    print_comparison_table(all_seed_results, dataset_name, list(seeds))

    # ------------------------------------------------------------------
    # 8. Steerability summary (V5 L0@j1 vs L0@j4 delta compared to MRL)
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  STEERABILITY CHECK (L0 Recall@10 at j=1 minus j=4)")
    print(f"{'='*80}")

    for method in ["v5", "mrl"]:
        deltas = []
        for seed in seeds:
            sk = str(seed)
            r1  = all_seed_results[sk][method]["1"]["L0"]["recall@10"]
            r4  = all_seed_results[sk][method]["4"]["L0"]["recall@10"]
            deltas.append(r1 - r4)
        mean_d = np.mean(deltas)
        std_d  = np.std(deltas)
        print(f"  {method.upper():>4}: mean(L0_R@10_j1 - L0_R@10_j4) = {mean_d:+.4f} +/- {std_d:.4f}")

    print("\n  Positive delta = short prefix better for coarse retrieval (steerability)")
    print("  Expectation: V5 positive, MRL ~zero")

    # ------------------------------------------------------------------
    # 9. Save JSON
    # ------------------------------------------------------------------
    output = {
        "experiment": "retrieval_benchmark",
        "model": model_key,
        "dataset": dataset_name,
        "seeds": list(seeds),
        "ks": list(ks),
        "stage1_epochs": stage1_epochs,
        "max_train_samples": max_train_samples,
        "max_test_samples": max_test_samples,
        "timestamp": datetime.now().isoformat(),
        "results_by_seed": all_seed_results,
    }

    out_path = RESULTS_DIR / f"retrieval_benchmark_{model_key}_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval benchmark: V5 vs MRL")
    parser.add_argument("--model",  type=str, default="bge-small")
    parser.add_argument("--dataset", type=str, default="clinc")
    parser.add_argument("--seeds",  type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-test",  type=int, default=2000)
    args = parser.parse_args()

    run_retrieval_benchmark(
        model_key=args.model,
        dataset_name=args.dataset,
        seeds=tuple(args.seeds),
        stage1_epochs=args.stage1_epochs,
        device=args.device,
        max_train_samples=args.max_train,
        max_test_samples=args.max_test,
    )
