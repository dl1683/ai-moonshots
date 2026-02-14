"""
Adaptive Cascade Retrieval with Fractal Embeddings
===================================================

Demonstrates that fractal (hierarchy-aligned) embeddings enable
compute-adaptive retrieval that changes the economics of vector search.

Key insight: V5's scale separation means the 64d prefix captures coarse
category structure, so cascade routing can skip full-resolution scoring
for most queries while maintaining retrieval quality.

Pipeline:
  1. Train V5 and MRL models on hierarchical data
  2. Extract block embeddings at all prefix lengths (64d, 128d, 192d, 256d)
  3. Build FAISS indices at each dimension
  4. Cascade retrieval: coarse filter -> confidence check -> fine re-rank
  5. Report: recall@k, latency, effective dims, storage savings

Comparison methods:
  - Full-256d: Always use full embeddings (baseline quality)
  - Flat-64d: Always use 64d prefix (fast but lower quality)
  - Cascade: 64d coarse search -> uncertain -> 256d re-rank (our method)
  - Two-stage: 64d first-pass + 256d re-rank all (standard industry pattern)
"""

import sys
import os
import time
import json
import gc
import numpy as np
import torch
import faiss
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval configuration."""
    method: str
    recall_at_1_l1: float
    recall_at_5_l1: float
    recall_at_10_l1: float
    recall_at_1_l0: float
    recall_at_5_l0: float
    recall_at_10_l0: float
    ndcg_at_10_l0: float
    mean_latency_ms: float
    p95_latency_ms: float
    effective_dims: float
    storage_bytes_per_doc: int
    n_queries: int


def extract_raw_block_embeddings(model, texts: List[str], batch_size: int = 64):
    """
    Extract raw block embeddings (NOT zero-padded) for efficient retrieval.
    Returns dict: prefix_len -> [N, prefix_len*64] float32 array, L2-normalized.
    """
    model.eval()
    all_blocks = {j: [] for j in range(1, model.num_scales + 1)}

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if model.config.prefix_query:
                batch = [model.config.prefix_query + t for t in batch]

            inputs = model.tokenizer(
                batch, padding=True, truncation=True,
                max_length=min(model.config.max_seq_len, 512),
                return_tensors="pt"
            ).to(model.device)

            result = model.forward(inputs['input_ids'], inputs['attention_mask'])
            blocks = result['blocks']

            for j in range(1, model.num_scales + 1):
                prefix = torch.cat(blocks[:j], dim=-1)
                prefix = torch.nn.functional.normalize(prefix, dim=-1)
                all_blocks[j].append(prefix.cpu().numpy())

    return {j: np.concatenate(v, axis=0).astype(np.float32)
            for j, v in all_blocks.items()}


def build_index(embs: np.ndarray) -> faiss.Index:
    """Build flat inner-product FAISS index (cosine on L2-normed vectors)."""
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


def recall_at_k(retrieved: np.ndarray, labels: np.ndarray,
                query_labels: np.ndarray, k: int) -> float:
    """Fraction of queries with at least one correct label in top-k."""
    hits = 0
    for i in range(len(query_labels)):
        topk = retrieved[i, :k]
        topk = topk[topk >= 0]  # ignore padding
        if len(topk) > 0 and np.any(labels[topk] == query_labels[i]):
            hits += 1
    return hits / len(query_labels)


def ndcg_at_k(retrieved: np.ndarray, labels: np.ndarray,
              query_labels: np.ndarray, k: int) -> float:
    """Compute mean NDCG@k with binary relevance."""
    ndcgs = []
    for i in range(len(query_labels)):
        topk = retrieved[i, :k]
        topk = topk[topk >= 0]
        gains = (labels[topk] == query_labels[i]).astype(float)
        dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
        # ideal: all relevant docs first
        n_rel = min(np.sum(labels == query_labels[i]) - 1, k)  # -1 exclude self
        idcg = np.sum(1.0 / np.log2(np.arange(2, int(n_rel) + 2))) if n_rel > 0 else 1.0
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))


def flat_search(index: faiss.Index, queries: np.ndarray,
                self_mask: np.ndarray, k: int = 10):
    """Search index, excluding self-matches. Returns ids, scores, latencies."""
    all_ids = []
    all_scores = []
    latencies = []

    for i in range(len(queries)):
        q = queries[i:i+1]
        t0 = time.perf_counter()
        scores, ids = index.search(q, k + 5)  # extra to handle self-removal
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        # Remove self-match
        mask = ids[0] != self_mask[i]
        filtered_ids = ids[0][mask][:k]
        filtered_scores = scores[0][mask][:k]

        # Pad if needed
        if len(filtered_ids) < k:
            pad = np.full(k - len(filtered_ids), -1, dtype=np.int64)
            filtered_ids = np.concatenate([filtered_ids, pad])
            pad_s = np.full(k - len(filtered_scores), -1.0, dtype=np.float32)
            filtered_scores = np.concatenate([filtered_scores, pad_s])

        all_ids.append(filtered_ids)
        all_scores.append(filtered_scores)

    return np.array(all_ids), np.array(all_scores), np.array(latencies)


def cascade_search(coarse_index: faiss.Index, coarse_embs: np.ndarray,
                   fine_embs: np.ndarray, self_indices: np.ndarray,
                   threshold: float = 0.85, coarse_k: int = 100,
                   final_k: int = 10):
    """
    Cascade retrieval:
    1. Search with coarse (64d) -> top coarse_k candidates
    2. If top-1 score > threshold -> accept (no re-ranking needed)
    3. Else re-rank candidates with fine (256d) embeddings
    """
    n = len(coarse_embs)
    coarse_dim = coarse_embs.shape[1]
    fine_dim = fine_embs.shape[1]

    all_ids = np.full((n, final_k), -1, dtype=np.int64)
    eff_dims = np.zeros(n)
    latencies = np.zeros(n)
    escalated = np.zeros(n, dtype=bool)

    for i in range(n):
        t0 = time.perf_counter()

        q_coarse = coarse_embs[i:i+1]
        scores, ids = coarse_index.search(q_coarse, coarse_k + 5)
        scores, ids = scores[0], ids[0]

        # Remove self
        mask = ids != self_indices[i]
        scores = scores[mask][:coarse_k]
        ids = ids[mask][:coarse_k]

        if len(scores) > 0 and scores[0] >= threshold:
            # Confident: accept coarse ranking
            result = ids[:final_k]
            eff_dims[i] = coarse_dim
        else:
            # Re-rank with fine embeddings
            q_fine = fine_embs[i]
            candidate_fine = fine_embs[ids]
            fine_scores = candidate_fine @ q_fine
            reranked = np.argsort(-fine_scores)[:final_k]
            result = ids[reranked]
            eff_dims[i] = coarse_dim + fine_dim
            escalated[i] = True

        t1 = time.perf_counter()
        latencies[i] = (t1 - t0) * 1000
        n_result = min(len(result), final_k)
        all_ids[i, :n_result] = result[:n_result]

    return all_ids, eff_dims, latencies, escalated


def run_retrieval_benchmark(
    dataset_name: str = "dbpedia_classes",
    model_key: str = "bge-small",
    seed: int = 42,
    max_samples: int = 5000,
    cascade_thresholds: Optional[List[float]] = None,
    device: str = "cuda",
):
    """
    Full retrieval benchmark: V5 vs MRL at multiple resolutions + cascade.
    Uses leave-one-out retrieval on test set.
    """
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cascade_thresholds is None:
        cascade_thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    from hierarchical_datasets import load_hierarchical_dataset
    from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
    from mrl_v5_baseline import MRLTrainerV5
    from multi_model_pipeline import MODELS

    print("=" * 70)
    print(f"ADAPTIVE CASCADE RETRIEVAL: {model_key} on {dataset_name}")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_samples)

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
    print(f"  Train: {len(train_data.samples)}, Val: {len(val_data.samples)}, "
          f"Test: {len(test_data.samples)}")
    print(f"  L0: {num_l0} classes, L1: {num_l1} classes")

    texts = [s.text for s in test_data.samples]
    l0_labels = np.array([s.level0_label for s in test_data.samples])
    l1_labels = np.array([s.level1_label for s in test_data.samples])
    n = len(texts)
    self_indices = np.arange(n)

    config = MODELS[model_key]
    all_results = []

    # ================================================================
    # Train and evaluate V5
    # ================================================================
    print("\n[2] Training V5...")
    v5_model = FractalModelV5(
        config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
        num_scales=4, scale_dim=64, device=device,
    ).to(device)
    v5_trainer = V5Trainer(
        model=v5_model, train_dataset=train_data, val_dataset=val_data,
        device=device, stage1_epochs=5, stage2_epochs=0,
    )
    v5_trainer.train(batch_size=32, patience=5)

    print("  Extracting V5 block embeddings...")
    v5_embs = extract_raw_block_embeddings(v5_model, texts)
    del v5_model, v5_trainer
    torch.cuda.empty_cache(); gc.collect()

    # ================================================================
    # Train and evaluate MRL
    # ================================================================
    print("\n[3] Training MRL...")
    mrl_model = FractalModelV5(
        config=config, num_l0_classes=num_l1, num_l1_classes=num_l1,
        num_scales=4, scale_dim=64, device=device,
    ).to(device)
    mrl_trainer = MRLTrainerV5(
        model=mrl_model, train_dataset=train_data, val_dataset=val_data,
        device=device, stage1_epochs=5, stage2_epochs=0,
    )
    mrl_trainer.train(batch_size=32, patience=5)

    print("  Extracting MRL block embeddings...")
    mrl_embs = extract_raw_block_embeddings(mrl_model, texts)
    del mrl_model, mrl_trainer
    torch.cuda.empty_cache(); gc.collect()

    # ================================================================
    # Flat retrieval at each prefix length
    # ================================================================
    print("\n[4] Flat retrieval at each dimension...")
    print(f"  {'Method':<16} {'R@1(L1)':<10} {'R@5(L1)':<10} {'R@10(L1)':<10} "
          f"{'R@1(L0)':<10} {'nDCG@10':<10} {'Latency':<10}")
    print("  " + "-" * 76)

    for label, embs_dict in [("V5", v5_embs), ("MRL", mrl_embs)]:
        for j in [1, 2, 3, 4]:
            dim = j * 64
            embs = embs_dict[j]
            index = build_index(embs)
            ids, scores, lats = flat_search(index, embs, self_indices, k=10)

            r = RetrievalMetrics(
                method=f"{label}-{dim}d",
                recall_at_1_l1=recall_at_k(ids, l1_labels, l1_labels, 1),
                recall_at_5_l1=recall_at_k(ids, l1_labels, l1_labels, 5),
                recall_at_10_l1=recall_at_k(ids, l1_labels, l1_labels, 10),
                recall_at_1_l0=recall_at_k(ids, l0_labels, l0_labels, 1),
                recall_at_5_l0=recall_at_k(ids, l0_labels, l0_labels, 5),
                recall_at_10_l0=recall_at_k(ids, l0_labels, l0_labels, 10),
                ndcg_at_10_l0=ndcg_at_k(ids, l0_labels, l0_labels, 10),
                mean_latency_ms=float(np.mean(lats)),
                p95_latency_ms=float(np.percentile(lats, 95)),
                effective_dims=dim,
                storage_bytes_per_doc=dim * 4,
                n_queries=n,
            )
            all_results.append(r)
            print(f"  {r.method:<16} {r.recall_at_1_l1:<10.4f} {r.recall_at_5_l1:<10.4f} "
                  f"{r.recall_at_10_l1:<10.4f} {r.recall_at_1_l0:<10.4f} "
                  f"{r.ndcg_at_10_l0:<10.4f} {r.mean_latency_ms:<10.3f}")

    # ================================================================
    # Cascade retrieval
    # ================================================================
    print(f"\n[5] Cascade retrieval (64d -> 256d re-rank)...")
    print(f"  {'Method':<28} {'R@1(L1)':<10} {'R@10(L1)':<10} {'R@1(L0)':<10} "
          f"{'EffDims':<10} {'Escal%':<8} {'Lat(ms)':<10}")
    print("  " + "-" * 86)

    for label, embs_dict in [("V5", v5_embs), ("MRL", mrl_embs)]:
        coarse = embs_dict[1]
        fine = embs_dict[4]
        coarse_idx = build_index(coarse)

        for t in cascade_thresholds:
            ids, eff, lats, esc = cascade_search(
                coarse_idx, coarse, fine, self_indices,
                threshold=t, coarse_k=100, final_k=10,
            )
            frac_esc = float(np.mean(esc))

            r = RetrievalMetrics(
                method=f"{label}-cascade-t{t:.2f}",
                recall_at_1_l1=recall_at_k(ids, l1_labels, l1_labels, 1),
                recall_at_5_l1=recall_at_k(ids, l1_labels, l1_labels, 5),
                recall_at_10_l1=recall_at_k(ids, l1_labels, l1_labels, 10),
                recall_at_1_l0=recall_at_k(ids, l0_labels, l0_labels, 1),
                recall_at_5_l0=recall_at_k(ids, l0_labels, l0_labels, 5),
                recall_at_10_l0=recall_at_k(ids, l0_labels, l0_labels, 10),
                ndcg_at_10_l0=ndcg_at_k(ids, l0_labels, l0_labels, 10),
                mean_latency_ms=float(np.mean(lats)),
                p95_latency_ms=float(np.percentile(lats, 95)),
                effective_dims=float(np.mean(eff)),
                storage_bytes_per_doc=(64 + 256) * 4,
                n_queries=n,
            )
            all_results.append(r)
            print(f"  {r.method:<28} {r.recall_at_1_l1:<10.4f} {r.recall_at_10_l1:<10.4f} "
                  f"{r.recall_at_1_l0:<10.4f} {r.effective_dims:<10.0f} "
                  f"{frac_esc:<8.1%} {r.mean_latency_ms:<10.3f}")

    # ================================================================
    # Two-stage baseline (standard industry pattern)
    # ================================================================
    print(f"\n[6] Two-stage baseline (64d filter -> 256d re-rank all)...")
    for label, embs_dict in [("V5", v5_embs), ("MRL", mrl_embs)]:
        coarse = embs_dict[1]
        fine = embs_dict[4]
        coarse_idx = build_index(coarse)

        # Always re-rank (threshold = infinity)
        ids, eff, lats, esc = cascade_search(
            coarse_idx, coarse, fine, self_indices,
            threshold=2.0, coarse_k=100, final_k=10,  # threshold>1 = always re-rank
        )

        r = RetrievalMetrics(
            method=f"{label}-twostage",
            recall_at_1_l1=recall_at_k(ids, l1_labels, l1_labels, 1),
            recall_at_5_l1=recall_at_k(ids, l1_labels, l1_labels, 5),
            recall_at_10_l1=recall_at_k(ids, l1_labels, l1_labels, 10),
            recall_at_1_l0=recall_at_k(ids, l0_labels, l0_labels, 1),
            recall_at_5_l0=recall_at_k(ids, l0_labels, l0_labels, 5),
            recall_at_10_l0=recall_at_k(ids, l0_labels, l0_labels, 10),
            ndcg_at_10_l0=ndcg_at_k(ids, l0_labels, l0_labels, 10),
            mean_latency_ms=float(np.mean(lats)),
            p95_latency_ms=float(np.percentile(lats, 95)),
            effective_dims=float(np.mean(eff)),
            storage_bytes_per_doc=(64 + 256) * 4,
            n_queries=n,
        )
        all_results.append(r)
        print(f"  {r.method:<28} R@1={r.recall_at_1_l1:.4f}, R@10={r.recall_at_10_l1:.4f}, "
              f"R@1_L0={r.recall_at_1_l0:.4f}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    v5_full = next(r for r in all_results if r.method == "V5-256d")
    mrl_full = next(r for r in all_results if r.method == "MRL-256d")
    v5_64 = next(r for r in all_results if r.method == "V5-64d")
    mrl_64 = next(r for r in all_results if r.method == "MRL-64d")

    print(f"\n  At 64d (4x storage/compute savings):")
    print(f"    V5:  R@1={v5_64.recall_at_1_l1:.4f}, R@10={v5_64.recall_at_10_l1:.4f}, "
          f"R@1_L0={v5_64.recall_at_1_l0:.4f}")
    print(f"    MRL: R@1={mrl_64.recall_at_1_l1:.4f}, R@10={mrl_64.recall_at_10_l1:.4f}, "
          f"R@1_L0={mrl_64.recall_at_1_l0:.4f}")
    adv = v5_64.recall_at_10_l1 - mrl_64.recall_at_10_l1
    print(f"    V5 advantage: R@10 = {adv:+.4f}")

    print(f"\n  At 256d (full resolution):")
    print(f"    V5:  R@1={v5_full.recall_at_1_l1:.4f}, R@10={v5_full.recall_at_10_l1:.4f}")
    print(f"    MRL: R@1={mrl_full.recall_at_1_l1:.4f}, R@10={mrl_full.recall_at_10_l1:.4f}")

    # Find best cascade that matches 95% of full recall
    v5_cascades = sorted(
        [r for r in all_results if r.method.startswith("V5-cascade")],
        key=lambda x: x.effective_dims
    )
    target = v5_full.recall_at_10_l1 * 0.95
    best = None
    for r in v5_cascades:
        if r.recall_at_10_l1 >= target:
            best = r
            break

    if best:
        savings = 1.0 - best.effective_dims / 256
        print(f"\n  Best V5 cascade (>= 95% of full R@10):")
        print(f"    {best.method}: R@10={best.recall_at_10_l1:.4f} "
              f"(target >= {target:.4f})")
        print(f"    Effective dims: {best.effective_dims:.0f} ({savings:.0%} savings)")
        print(f"    Latency: {best.mean_latency_ms:.3f}ms "
              f"(vs full {v5_full.mean_latency_ms:.3f}ms)")

    # ================================================================
    # Save results
    # ================================================================
    output = {
        "dataset": dataset_name,
        "model": model_key,
        "seed": seed,
        "n_corpus": n,
        "n_l0": int(num_l0),
        "n_l1": int(num_l1),
        "timestamp": datetime.now().isoformat(),
        "methods": [asdict(r) for r in all_results],
    }

    results_dir = Path(__file__).parent.parent / "results" / "retrieval"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"cascade_{dataset_name}_seed{seed}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return output


def generate_retrieval_figure(results_dir: str = None):
    """Generate publication-quality figure for cascade retrieval."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results" / "retrieval"
    results_dir = Path(results_dir)

    files = sorted(results_dir.glob("cascade_*.json"))
    if not files:
        print("No retrieval results found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {"V5": "#2196F3", "MRL": "#FF5722"}
    markers = {"V5": "o", "MRL": "s"}

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        ds = data["dataset"]
        methods = data["methods"]

        # Panel 1: R@10(L1) vs Effective Dims
        ax = axes[0]
        for prefix in ["V5", "MRL"]:
            flat_pts = [(m["effective_dims"], m["recall_at_10_l1"])
                       for m in methods
                       if m["method"].startswith(f"{prefix}-") and "cascade" not in m["method"]
                       and "twostage" not in m["method"]]
            cascade_pts = [(m["effective_dims"], m["recall_at_10_l1"])
                          for m in methods if m["method"].startswith(f"{prefix}-cascade")]
            if flat_pts:
                x, y = zip(*sorted(flat_pts))
                ax.plot(x, y, f'{markers[prefix]}-', color=colors[prefix],
                       label=f"{prefix} flat", linewidth=2, markersize=8)
            if cascade_pts:
                x, y = zip(*sorted(cascade_pts))
                ax.plot(x, y, f'{markers[prefix]}--', color=colors[prefix],
                       alpha=0.5, markersize=5, label=f"{prefix} cascade")

        ax.set_xlabel("Effective Dimensions", fontsize=12)
        ax.set_ylabel("Recall@10 (L1)", fontsize=12)
        ax.set_title(f"Quality vs Compute ({ds})", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: R@1(L0) vs Dims (coarse retrieval quality)
        ax = axes[1]
        for prefix in ["V5", "MRL"]:
            pts = [(m["effective_dims"], m["recall_at_1_l0"])
                   for m in methods
                   if m["method"].startswith(f"{prefix}-") and "cascade" not in m["method"]
                   and "twostage" not in m["method"]]
            if pts:
                x, y = zip(*sorted(pts))
                ax.plot(x, y, f'{markers[prefix]}-', color=colors[prefix],
                       label=prefix, linewidth=2, markersize=8)

        ax.set_xlabel("Embedding Dimensions", fontsize=12)
        ax.set_ylabel("Recall@1 (L0 -- Coarse)", fontsize=12)
        ax.set_title(f"Coarse Category Retrieval ({ds})", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 3: Escalation rate vs R@10
        ax = axes[2]
        for prefix in ["V5", "MRL"]:
            pts = []
            for m in methods:
                if m["method"].startswith(f"{prefix}-cascade"):
                    esc_rate = (m["effective_dims"] - 64) / (256 - 64)
                    pts.append((esc_rate, m["recall_at_10_l1"]))
            if pts:
                x, y = zip(*sorted(pts))
                ax.plot(x, y, f'{markers[prefix]}-', color=colors[prefix],
                       label=prefix, linewidth=2, markersize=6)

        ax.set_xlabel("Escalation Rate", fontsize=12)
        ax.set_ylabel("Recall@10 (L1)", fontsize=12)
        ax.set_title(f"Cascade Efficiency ({ds})", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_dir = results_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        plt.savefig(fig_dir / f"fig_cascade_retrieval.{ext}",
                   dpi=150, bbox_inches='tight')
    print(f"  Figure saved to {fig_dir / 'fig_cascade_retrieval.pdf'}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dbpedia_classes")
    parser.add_argument("--model", default="bge-small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()

    if args.figure_only:
        generate_retrieval_figure()
    else:
        run_retrieval_benchmark(
            dataset_name=args.dataset,
            model_key=args.model,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        generate_retrieval_figure()
