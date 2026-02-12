"""
Adaptive Semantic-Zoom RAG Demo
================================

THE KILLER APPLICATION: One fractal embedding provides runtime-controllable
semantic granularity for retrieval, replacing multi-model stacks.

Strategies compared:
  1. fixed_64d  -- always use 64d prefix (fast, coarse)
  2. fixed_256d -- always use 256d full embedding (slow, precise)
  3. two_stage  -- 64d first-pass + 256d rerank top-100
  4. adaptive_v5  -- progressive resolution with V5 (score-based confidence)
  5. adaptive_mrl -- progressive resolution with MRL (score-based confidence)

Metrics:
  - nDCG@10 with graded relevance (2=same L1, 1=same L0 diff L1, 0=other)
  - Recall@k (k=1,5,10) for L1 labels
  - MAP for L1 labels
  - Wall-clock latency per query (ms)
  - FLOPs estimate per query
  - Average embedding dimensions used

Produces:
  - results/adaptive_rag_demo.json
  - results/figures/paper/fig_adaptive_pareto.png
"""

import sys
import os
import json
import gc
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val, MODELS
from mrl_v5_baseline import MRLTrainerV5

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "paper"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Retrieval metric helpers
# ---------------------------------------------------------------------------

def graded_relevance(query_l0, query_l1, doc_l0s, doc_l1s):
    """Compute graded relevance: 2=same L1, 1=same L0 diff L1, 0=other."""
    rels = np.zeros(len(doc_l0s), dtype=np.float64)
    same_l1 = doc_l1s == query_l1
    same_l0 = doc_l0s == query_l0
    rels[same_l0 & ~same_l1] = 1.0
    rels[same_l1] = 2.0
    return rels


def ndcg_at_k(relevances, k=10):
    """Compute nDCG@k from a relevance array (already in ranked order)."""
    relevances = relevances[:k]
    gains = 2.0 ** relevances - 1.0
    discounts = np.log2(np.arange(len(gains)) + 2.0)
    dcg = np.sum(gains / discounts)
    # Ideal DCG
    ideal_rels = np.sort(relevances)[::-1]
    ideal_gains = 2.0 ** ideal_rels - 1.0
    idcg = np.sum(ideal_gains / discounts[:len(ideal_gains)])
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def recall_at_k(ranked_l1s, query_l1, total_relevant, k):
    """Recall@k: fraction of relevant docs found in top-k."""
    if total_relevant == 0:
        return 0.0
    found = np.sum(ranked_l1s[:k] == query_l1)
    return float(found / total_relevant)


def average_precision(ranked_l1s, query_l1):
    """Average precision for a single query."""
    relevant = (ranked_l1s == query_l1).astype(np.float64)
    if relevant.sum() == 0:
        return 0.0
    cumsum = np.cumsum(relevant)
    precision_at_k = cumsum / (np.arange(len(relevant)) + 1.0)
    return float(np.sum(precision_at_k * relevant) / relevant.sum())


def compute_retrieval_metrics(
    ranked_indices: np.ndarray,
    query_l0: int, query_l1: int,
    doc_l0s: np.ndarray, doc_l1s: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute all retrieval metrics for a single query."""
    ranked_l0s = doc_l0s[ranked_indices]
    ranked_l1s = doc_l1s[ranked_indices]
    rels = graded_relevance(query_l0, query_l1, ranked_l0s, ranked_l1s)
    total_relevant = np.sum(doc_l1s == query_l1)

    metrics = {}
    metrics['ndcg@10'] = ndcg_at_k(rels, k=10)
    for k in ks:
        metrics[f'recall@{k}'] = recall_at_k(ranked_l1s, query_l1, total_relevant, k)
    metrics['ap'] = average_precision(ranked_l1s, query_l1)
    return metrics


# ---------------------------------------------------------------------------
# Embedding + retrieval primitives
# ---------------------------------------------------------------------------

def compute_embeddings(model, texts, batch_size=64):
    """Compute normalized embeddings at all 4 prefix lengths."""
    model.eval()
    embeddings = {}
    for j in [1, 2, 3, 4]:
        prefix_len = j if j < 4 else None
        emb = model.encode(texts, batch_size=batch_size, prefix_len=prefix_len)
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        # Physically truncate to j*64 dimensions
        emb = emb[:, :j * 64]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-9)
        embeddings[j] = emb
    return embeddings


def retrieve_ranked(query_emb, doc_embs):
    """Return ranked doc indices and similarities by cosine similarity."""
    sims = doc_embs @ query_emb
    ranked = np.argsort(-sims)
    return ranked, sims[ranked]


def score_confidence(sims_sorted, mode='gap'):
    """Score-based confidence (no label leakage).

    gap: difference between top-1 and top-2 similarity
    margin: top-1 similarity value (higher = more confident)
    """
    if mode == 'gap':
        if len(sims_sorted) < 2:
            return float(sims_sorted[0]) if len(sims_sorted) > 0 else 0.0
        return float(sims_sorted[0] - sims_sorted[1])
    elif mode == 'margin':
        return float(sims_sorted[0]) if len(sims_sorted) > 0 else 0.0
    else:
        raise ValueError(f"Unknown confidence mode: {mode}")


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def fixed_retrieval_eval(
    query_embs, doc_embs, query_l0s, query_l1s, doc_l0s, doc_l1s,
    dim_label: int,
):
    """Fixed-dimension retrieval with proper metrics."""
    n_queries = len(query_l0s)
    all_metrics = []
    n_docs = len(doc_l0s)
    flops_per_query = 2 * n_docs * dim_label

    # Warmup
    for i in range(min(50, n_queries)):
        _ = doc_embs @ query_embs[i]

    latencies = []
    for i in range(n_queries):
        t0 = time.perf_counter_ns()
        ranked, sims = retrieve_ranked(query_embs[i], doc_embs)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)  # ms

        m = compute_retrieval_metrics(ranked, query_l0s[i], query_l1s[i], doc_l0s, doc_l1s)
        all_metrics.append(m)

    agg = _aggregate_metrics(all_metrics)
    agg['avg_dims'] = float(dim_label)
    agg['flops_per_query'] = float(flops_per_query)
    agg['latency_mean_ms'] = float(np.mean(latencies))
    agg['latency_p50_ms'] = float(np.median(latencies))
    agg['latency_p95_ms'] = float(np.percentile(latencies, 95))
    return agg


def twostage_retrieval_eval(
    query_embs_j1, query_embs_j4,
    doc_embs_j1, doc_embs_j4,
    query_l0s, query_l1s, doc_l0s, doc_l1s,
    first_pass_k=100, top_k=10,
):
    """Two-stage: 64d first-pass + 256d rerank."""
    n_queries = len(query_l0s)
    n_docs = len(doc_l0s)
    all_metrics = []
    flops_per_query = 2 * (n_docs * 64 + first_pass_k * 256)

    # Warmup
    for i in range(min(50, n_queries)):
        _ = doc_embs_j1 @ query_embs_j1[i]

    latencies = []
    for i in range(n_queries):
        t0 = time.perf_counter_ns()
        # Stage 1: coarse retrieval
        sims_coarse = doc_embs_j1 @ query_embs_j1[i]
        top_k_coarse = np.argsort(-sims_coarse)[:first_pass_k]
        # Stage 2: fine rerank
        candidate_embs = doc_embs_j4[top_k_coarse]
        sims_fine = candidate_embs @ query_embs_j4[i]
        reranked = top_k_coarse[np.argsort(-sims_fine)]
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)

        m = compute_retrieval_metrics(reranked, query_l0s[i], query_l1s[i], doc_l0s, doc_l1s)
        all_metrics.append(m)

    agg = _aggregate_metrics(all_metrics)
    agg['avg_dims'] = float((n_docs * 64 + first_pass_k * 256) / n_docs)
    agg['flops_per_query'] = float(flops_per_query)
    agg['latency_mean_ms'] = float(np.mean(latencies))
    agg['latency_p50_ms'] = float(np.median(latencies))
    agg['latency_p95_ms'] = float(np.percentile(latencies, 95))
    agg['first_pass_k'] = first_pass_k
    return agg


def adaptive_retrieval_eval(
    query_embs_by_j, doc_embs_by_j,
    query_l0s, query_l1s, doc_l0s, doc_l1s,
    confidence_threshold: float,
    confidence_mode: str = 'gap',
):
    """Adaptive progressive retrieval with score-based confidence."""
    n_queries = len(query_l0s)
    n_docs = len(doc_l0s)
    all_metrics = []
    dims_used = []
    cumulative_dims = []
    resolution_dist = {1: 0, 2: 0, 3: 0, 4: 0}

    # Warmup
    for i in range(min(50, n_queries)):
        _ = doc_embs_by_j[1] @ query_embs_by_j[1][i]

    latencies = []
    for i in range(n_queries):
        t0 = time.perf_counter_ns()
        final_ranked = None
        terminal_j = 4
        cum_dim = 0

        for j in [1, 2, 3, 4]:
            q_emb = query_embs_by_j[j][i]
            d_embs = doc_embs_by_j[j]
            ranked, sims_ranked = retrieve_ranked(q_emb, d_embs)
            cum_dim += j * 64

            conf = score_confidence(sims_ranked, mode=confidence_mode)
            final_ranked = ranked

            if conf >= confidence_threshold or j == 4:
                terminal_j = j
                break

        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)

        dims_used.append(terminal_j * 64)
        cumulative_dims.append(cum_dim)
        resolution_dist[terminal_j] += 1

        m = compute_retrieval_metrics(final_ranked, query_l0s[i], query_l1s[i], doc_l0s, doc_l1s)
        all_metrics.append(m)

    agg = _aggregate_metrics(all_metrics)
    agg['avg_terminal_dim'] = float(np.mean(dims_used))
    agg['avg_computed_dim'] = float(np.mean(cumulative_dims))
    agg['flops_per_query'] = float(2 * n_docs * np.mean(cumulative_dims))
    agg['dim_savings_pct'] = float((256 - np.mean(dims_used)) / 256 * 100)
    agg['latency_mean_ms'] = float(np.mean(latencies))
    agg['latency_p50_ms'] = float(np.median(latencies))
    agg['latency_p95_ms'] = float(np.percentile(latencies, 95))
    agg['resolution_distribution'] = {
        f'j{j} ({j*64}d)': resolution_dist[j] / n_queries for j in [1, 2, 3, 4]
    }
    agg['confidence_threshold'] = confidence_threshold
    return agg


def _aggregate_metrics(metric_list):
    """Average a list of per-query metric dicts."""
    keys = metric_list[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in metric_list]
        agg[k] = float(np.mean(vals))
    return agg


# ---------------------------------------------------------------------------
# Calibrated thresholds from validation data
# ---------------------------------------------------------------------------

def calibrate_thresholds(
    val_embs_by_j, val_l0s, val_l1s,
    doc_embs_by_j, doc_l0s, doc_l1s,
    confidence_mode='gap',
    quantiles=(0.50, 0.65, 0.80, 0.90, 0.95),
):
    """Compute confidence thresholds from validation score gaps at j=1."""
    n_val = len(val_l0s)
    scores = []
    for i in range(n_val):
        q_emb = val_embs_by_j[1][i]
        d_embs = doc_embs_by_j[1]
        sims = d_embs @ q_emb
        sims_sorted = np.sort(sims)[::-1]
        conf = score_confidence(sims_sorted, mode=confidence_mode)
        scores.append(conf)
    scores = np.array(scores)
    thresholds = [float(np.percentile(scores, q * 100)) for q in quantiles]
    return thresholds


# ---------------------------------------------------------------------------
# Stratified query/doc split
# ---------------------------------------------------------------------------

def stratified_split(n_samples, l1_labels, ratio=0.5, seed=42):
    """Split indices into query/doc ensuring every L1 label has docs."""
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(l1_labels)
    q_idx_list = []
    d_idx_list = []

    for label in unique_labels:
        label_indices = np.where(l1_labels == label)[0]
        rng.shuffle(label_indices)
        n_q = max(1, int(len(label_indices) * ratio))
        n_d = len(label_indices) - n_q
        if n_d < 1:
            # Need at least 1 doc per label
            n_q = len(label_indices) - 1
            n_d = 1
        q_idx_list.extend(label_indices[:n_q])
        d_idx_list.extend(label_indices[n_q:])

    return np.array(q_idx_list), np.array(d_idx_list)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_rag_demo(
    datasets=('clinc', 'dbpedia_classes', 'trec'),
    model_key='bge-small',
    seeds=(42, 123, 456),
    device='cuda',
    confidence_mode='gap',
    max_train=10000,
    max_test=2000,
):
    """Run the full adaptive RAG demo across datasets and seeds."""
    print("=" * 70)
    print("ADAPTIVE SEMANTIC-ZOOM RAG DEMO")
    print("=" * 70)

    results = {
        'config': {
            'model': model_key,
            'datasets': list(datasets),
            'seeds': list(seeds),
            'confidence_mode': confidence_mode,
            'max_train': max_train,
            'max_test': max_test,
        },
        'timestamp': datetime.now().isoformat(),
        'datasets': {},
    }

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*60}")

        ds_results = {'per_seed': {}, 'aggregated': {}}

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Load data
            print("  [1/5] Loading data...")
            train_data = load_hierarchical_dataset(ds_name, split='train', max_samples=max_train)
            test_data = load_hierarchical_dataset(ds_name, split='test', max_samples=max_test)

            train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

            class TempDataset:
                def __init__(self, samples, l0_names, l1_names):
                    self.samples = samples
                    self.level0_names = l0_names
                    self.level1_names = l1_names

            val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
            train_data_trimmed = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)

            num_l0 = len(train_data.level0_names)
            num_l1 = len(train_data.level1_names)
            config = MODELS[model_key]

            # Train V5
            print("  [2/5] Training V5...")
            v5_model = FractalModelV5(
                config=config, num_l0_classes=num_l0, num_l1_classes=num_l1,
                num_scales=4, scale_dim=64, device=device,
            ).to(device)
            v5_trainer = V5Trainer(
                model=v5_model, train_dataset=train_data_trimmed, val_dataset=val_data,
                device=device, stage1_epochs=5, stage2_epochs=0,
            )
            v5_trainer.train(batch_size=16, patience=5)

            # Train MRL
            print("  [3/5] Training MRL...")
            mrl_model = FractalModelV5(
                config=config, num_l0_classes=num_l1, num_l1_classes=num_l1,
                num_scales=4, scale_dim=64, device=device,
            ).to(device)
            mrl_trainer = MRLTrainerV5(
                model=mrl_model, train_dataset=train_data_trimmed, val_dataset=val_data,
                device=device, stage1_epochs=5, stage2_epochs=0,
            )
            mrl_trainer.train(batch_size=16, patience=5)

            # Compute embeddings
            print("  [4/5] Computing embeddings...")
            test_texts = [s.text for s in test_data.samples]
            test_l0 = np.array([s.level0_label for s in test_data.samples])
            test_l1 = np.array([s.level1_label for s in test_data.samples])

            v5_embs = compute_embeddings(v5_model, test_texts)
            mrl_embs = compute_embeddings(mrl_model, test_texts)

            # Also compute val embeddings for threshold calibration
            val_texts = [s.text for s in val_data.samples]
            val_l0 = np.array([s.level0_label for s in val_data.samples])
            val_l1 = np.array([s.level1_label for s in val_data.samples])
            v5_val_embs = compute_embeddings(v5_model, val_texts)
            mrl_val_embs = compute_embeddings(mrl_model, val_texts)

            # Stratified query/doc split
            q_idx, d_idx = stratified_split(len(test_texts), test_l1, ratio=0.5, seed=seed)

            # Calibrate thresholds from validation data
            # Use val as queries, doc portion of test as corpus for calibration
            v5_thresholds = calibrate_thresholds(
                {j: v5_val_embs[j] for j in [1, 2, 3, 4]},
                val_l0, val_l1,
                {j: v5_embs[j][d_idx] for j in [1, 2, 3, 4]},
                test_l0[d_idx], test_l1[d_idx],
                confidence_mode=confidence_mode,
            )
            mrl_thresholds = calibrate_thresholds(
                {j: mrl_val_embs[j] for j in [1, 2, 3, 4]},
                val_l0, val_l1,
                {j: mrl_embs[j][d_idx] for j in [1, 2, 3, 4]},
                test_l0[d_idx], test_l1[d_idx],
                confidence_mode=confidence_mode,
            )

            print("  [5/5] Evaluating strategies...")
            seed_results = {}

            # ---- Fixed baselines ----
            for method_name, embs in [('v5', v5_embs), ('mrl', mrl_embs)]:
                for j in [1, 4]:
                    label = f'{method_name}_fixed_{j*64}d'
                    r = fixed_retrieval_eval(
                        embs[j][q_idx], embs[j][d_idx],
                        test_l0[q_idx], test_l1[q_idx],
                        test_l0[d_idx], test_l1[d_idx],
                        dim_label=j * 64,
                    )
                    seed_results[label] = r
                    print(f"    {label}: nDCG@10={r['ndcg@10']:.4f}, "
                          f"R@10={r['recall@10']:.4f}, MAP={r['ap']:.4f}, "
                          f"lat={r['latency_mean_ms']:.3f}ms")

            # ---- Two-stage ----
            for method_name, embs in [('v5', v5_embs), ('mrl', mrl_embs)]:
                label = f'{method_name}_twostage'
                r = twostage_retrieval_eval(
                    embs[1][q_idx], embs[4][q_idx],
                    embs[1][d_idx], embs[4][d_idx],
                    test_l0[q_idx], test_l1[q_idx],
                    test_l0[d_idx], test_l1[d_idx],
                    first_pass_k=100, top_k=10,
                )
                seed_results[label] = r
                print(f"    {label}: nDCG@10={r['ndcg@10']:.4f}, "
                      f"R@10={r['recall@10']:.4f}, MAP={r['ap']:.4f}, "
                      f"lat={r['latency_mean_ms']:.3f}ms")

            # ---- Adaptive ----
            for method_name, embs, thresholds in [
                ('v5', v5_embs, v5_thresholds),
                ('mrl', mrl_embs, mrl_thresholds),
            ]:
                q_embs_by_j = {j: embs[j][q_idx] for j in [1, 2, 3, 4]}
                d_embs_by_j = {j: embs[j][d_idx] for j in [1, 2, 3, 4]}

                for ti, thresh in enumerate(thresholds):
                    label = f'{method_name}_adaptive_q{ti}'
                    r = adaptive_retrieval_eval(
                        q_embs_by_j, d_embs_by_j,
                        test_l0[q_idx], test_l1[q_idx],
                        test_l0[d_idx], test_l1[d_idx],
                        confidence_threshold=thresh,
                        confidence_mode=confidence_mode,
                    )
                    seed_results[label] = r
                    print(f"    {label} (t={thresh:.4f}): nDCG@10={r['ndcg@10']:.4f}, "
                          f"R@10={r['recall@10']:.4f}, "
                          f"avg_dim={r['avg_terminal_dim']:.0f}d, "
                          f"savings={r['dim_savings_pct']:.1f}%, "
                          f"lat={r['latency_mean_ms']:.3f}ms")

            ds_results['per_seed'][str(seed)] = seed_results

            # Cleanup
            del v5_model, mrl_model, v5_trainer, mrl_trainer
            torch.cuda.empty_cache()
            gc.collect()

        # Aggregate across seeds
        ds_results['aggregated'] = aggregate_across_seeds(ds_results['per_seed'])
        ds_results['metadata'] = {
            'num_l0': num_l0,
            'num_l1': num_l1,
            'n_queries': len(q_idx),
            'n_docs': len(d_idx),
        }
        results['datasets'][ds_name] = ds_results

        # Print aggregated summary
        print(f"\n  AGGREGATED ({ds_name}):")
        agg = ds_results['aggregated']
        for label in sorted(agg.keys()):
            a = agg[label]
            lat_str = f"lat={a.get('latency_mean_ms_mean', 0):.3f}ms" if 'latency_mean_ms_mean' in a else ""
            print(f"    {label}: nDCG@10={a.get('ndcg@10_mean', 0):.4f}+-{a.get('ndcg@10_std', 0):.4f}, "
                  f"R@10={a.get('recall@10_mean', 0):.4f}, "
                  f"MAP={a.get('ap_mean', 0):.4f}, {lat_str}")

    # Compute Pareto frontiers
    results['global_summary'] = compute_global_summary(results)

    # Save
    out_path = RESULTS_DIR / "adaptive_rag_demo.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")

    # Generate figure
    plot_pareto_frontier(results)

    return results


def aggregate_across_seeds(per_seed_results):
    """Aggregate metrics across seeds (mean + std)."""
    seeds = list(per_seed_results.keys())
    if not seeds:
        return {}

    all_labels = list(per_seed_results[seeds[0]].keys())
    aggregated = {}

    for label in all_labels:
        agg = {}
        # Collect numeric keys
        seed_data = [per_seed_results[s][label] for s in seeds if label in per_seed_results[s]]
        if not seed_data:
            continue
        for key in seed_data[0]:
            if isinstance(seed_data[0][key], (int, float)):
                vals = [d[key] for d in seed_data]
                agg[f'{key}_mean'] = float(np.mean(vals))
                agg[f'{key}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        aggregated[label] = agg

    return aggregated


def compute_global_summary(results):
    """Compute Pareto frontier across all datasets."""
    summary = {}
    for ds_name, ds_data in results['datasets'].items():
        agg = ds_data['aggregated']
        pareto_points = []
        for label, metrics in agg.items():
            ndcg = metrics.get('ndcg@10_mean', 0)
            lat = metrics.get('latency_mean_ms_mean', float('inf'))
            flops = metrics.get('flops_per_query_mean', float('inf'))
            pareto_points.append({
                'label': label,
                'ndcg@10': ndcg,
                'latency_ms': lat,
                'flops': flops,
            })
        # Find Pareto-optimal (max nDCG, min latency)
        pareto_optimal = []
        for i, p in enumerate(pareto_points):
            dominated = False
            for j, q in enumerate(pareto_points):
                if i == j:
                    continue
                if q['ndcg@10'] >= p['ndcg@10'] and q['latency_ms'] <= p['latency_ms']:
                    if q['ndcg@10'] > p['ndcg@10'] or q['latency_ms'] < p['latency_ms']:
                        dominated = True
                        break
            if not dominated:
                pareto_optimal.append(p['label'])

        summary[ds_name] = {
            'all_points': pareto_points,
            'pareto_optimal': pareto_optimal,
        }
    return summary


# ---------------------------------------------------------------------------
# Pareto figure
# ---------------------------------------------------------------------------

def plot_pareto_frontier(results):
    """Generate publication-quality Pareto frontier figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'legend.fontsize': 8, 'figure.dpi': 150,
    })

    datasets = list(results['datasets'].keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5), squeeze=False)

    ds_titles = {
        'clinc': 'CLINC (H=3.90)',
        'dbpedia_classes': 'DBPedia Classes (H=3.17)',
        'trec': 'TREC (H=2.21)',
    }

    for di, ds_name in enumerate(datasets):
        ax = axes[0, di]
        agg = results['datasets'][ds_name]['aggregated']
        pareto_labels = results['global_summary'][ds_name]['pareto_optimal']

        for label, metrics in agg.items():
            ndcg = metrics.get('ndcg@10_mean', 0)
            lat = metrics.get('latency_mean_ms_mean', 0)

            # Determine style
            if 'v5_adaptive' in label:
                color, marker, zorder = '#1f77b4', 'o', 3
            elif 'mrl_adaptive' in label:
                color, marker, zorder = '#ff7f0e', 'o', 3
            elif 'twostage' in label:
                color, marker, zorder = '#2ca02c', '^', 4
            elif 'fixed_64' in label:
                color, marker, zorder = '#7f7f7f', 's', 2
            elif 'fixed_256' in label:
                color, marker, zorder = '#7f7f7f', 'D', 2
            else:
                color, marker, zorder = '#bcbd22', 'x', 1

            edgecolor = 'black' if label in pareto_labels else 'none'
            linewidth = 1.5 if label in pareto_labels else 0.5

            ax.scatter(lat, ndcg, c=color, marker=marker, s=60, zorder=zorder,
                       edgecolors=edgecolor, linewidths=linewidth)

        # Connect adaptive points with lines
        for method, color in [('v5_adaptive', '#1f77b4'), ('mrl_adaptive', '#ff7f0e')]:
            pts = [(metrics.get('latency_mean_ms_mean', 0), metrics.get('ndcg@10_mean', 0))
                   for label, metrics in agg.items() if method in label]
            if pts:
                pts.sort()
                xs, ys = zip(*pts)
                method_label = 'V5 adaptive' if 'v5' in method else 'MRL adaptive'
                ax.plot(xs, ys, '-', color=color, alpha=0.5, linewidth=1.5, label=method_label)

        # Add fixed/twostage to legend
        ax.scatter([], [], c='#7f7f7f', marker='s', label='Fixed 64d')
        ax.scatter([], [], c='#7f7f7f', marker='D', label='Fixed 256d')
        ax.scatter([], [], c='#2ca02c', marker='^', label='Two-stage')

        ax.set_xlabel('Latency (ms/query)')
        ax.set_ylabel('nDCG@10')
        ax.set_title(ds_titles.get(ds_name, ds_name))
        ax.legend(loc='lower right')

    fig.suptitle('Adaptive Retrieval: Quality vs Cost Pareto', fontsize=13, y=1.02)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        out = FIGURES_DIR / f"fig_adaptive_pareto.{ext}"
        plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved to {FIGURES_DIR / 'fig_adaptive_pareto.png'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive RAG Demo")
    parser.add_argument("--datasets", nargs="+",
                        default=["clinc", "dbpedia_classes", "trec"])
    parser.add_argument("--model", default="bge-small")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed only")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]

    run_rag_demo(
        datasets=tuple(args.datasets),
        model_key=args.model,
        seeds=tuple(args.seeds),
        device=args.device,
    )
