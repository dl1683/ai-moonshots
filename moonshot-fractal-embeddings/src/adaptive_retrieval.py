"""
Adaptive Retrieval with Fractal Embeddings
===========================================

THE KILLER EXPERIMENT: One fractal embedding replaces multi-model stacks.

At query time:
1. Start with 64d prefix → coarse filtering (fast, cheap)
2. If uncertain, extend to 128d → medium resolution
3. If still uncertain, extend to 256d → full resolution

Compare against:
- Fixed 64d (always fast, always coarse)
- Fixed 256d (always slow, always precise)
- Two-stage pipeline: 64d first-pass + 256d reranker (typical industry pattern)

Success criterion: Fractal achieves same quality as fixed-256d at much lower
average dimension, OR better quality at same average dimension.

This shows practical utility and changes the paradigm from
"one embedding = one operating point" to
"one embedding = progressive code with runtime-controllable semantics."
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS, load_model
from fractal_v5 import FractalModelV5, V5Trainer, ContrastiveDatasetV5, split_train_val
from mrl_v5_baseline import MRLBaselineModel, MRLTrainer


def compute_embeddings(model, texts, batch_size=32):
    """Compute embeddings at all prefix lengths."""
    embeddings = {}
    for j in [1, 2, 3, 4]:
        prefix_len = j if j < 4 else None
        emb = model.encode(texts, batch_size=batch_size, prefix_len=prefix_len).numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        embeddings[j] = emb
    return embeddings


def knn_predict(query_emb, doc_embs, doc_labels, k=5):
    """Return kNN prediction and confidence (max vote fraction)."""
    sims = doc_embs @ query_emb
    top_k_idx = np.argsort(-sims)[:k]
    top_k_labels = doc_labels[top_k_idx]
    unique, counts = np.unique(top_k_labels, return_counts=True)
    pred = unique[np.argmax(counts)]
    confidence = counts.max() / k
    return pred, confidence


def fixed_retrieval(query_embs, doc_embs, doc_l0, doc_l1, k=5):
    """Standard fixed-dimension retrieval."""
    n_queries = len(query_embs)
    l0_correct = 0
    l1_correct = 0

    for i in range(n_queries):
        pred_l0, _ = knn_predict(query_embs[i], doc_embs, doc_l0, k)
        pred_l1, _ = knn_predict(query_embs[i], doc_embs, doc_l1, k)
        if pred_l0 == doc_l0[i]:  # Placeholder: query labels should be separate
            l0_correct += 1
        if pred_l1 == doc_l1[i]:
            l1_correct += 1

    return {
        'l0_accuracy': l0_correct / n_queries,
        'l1_accuracy': l1_correct / n_queries,
    }


def adaptive_retrieval(
    query_embs_by_j: Dict[int, np.ndarray],
    doc_embs_by_j: Dict[int, np.ndarray],
    query_l0: np.ndarray,
    query_l1: np.ndarray,
    doc_l0: np.ndarray,
    doc_l1: np.ndarray,
    confidence_threshold: float = 0.8,
    k: int = 5,
) -> Dict:
    """
    Adaptive progressive retrieval.

    For each query:
    1. Start at j=1 (64d). Get kNN prediction + confidence.
    2. If confidence >= threshold, accept prediction.
    3. If not, extend to j=2, j=3, j=4 until confident or max reached.

    Tracks: accuracy, average dimensions used, dimension savings.
    """
    n_queries = len(query_l0)
    l0_correct = 0
    l1_correct = 0
    dims_used = []
    resolution_distribution = {1: 0, 2: 0, 3: 0, 4: 0}

    for i in range(n_queries):
        final_pred_l0 = None
        final_pred_l1 = None
        used_j = 4  # Default to full if nothing is confident

        for j in [1, 2, 3, 4]:
            q_emb = query_embs_by_j[j][i]
            d_embs = doc_embs_by_j[j]

            pred_l0, conf_l0 = knn_predict(q_emb, d_embs, doc_l0, k)
            pred_l1, conf_l1 = knn_predict(q_emb, d_embs, doc_l1, k)

            # Use L0 confidence for early stopping (coarse is what we trust at short prefix)
            # At j=1, we trust L0; at j=4, we trust L1
            if j == 1:
                confidence = conf_l0  # Trust coarse prediction at short prefix
            elif j == 4:
                confidence = conf_l1  # At full resolution, trust fine prediction
            else:
                confidence = (conf_l0 + conf_l1) / 2  # Mixed

            final_pred_l0 = pred_l0
            final_pred_l1 = pred_l1

            if confidence >= confidence_threshold:
                used_j = j
                break

        dims_used.append(used_j * 64)
        resolution_distribution[used_j] += 1

        if final_pred_l0 == query_l0[i]:
            l0_correct += 1
        if final_pred_l1 == query_l1[i]:
            l1_correct += 1

    avg_dims = np.mean(dims_used)
    return {
        'l0_accuracy': l0_correct / n_queries,
        'l1_accuracy': l1_correct / n_queries,
        'avg_dims_used': float(avg_dims),
        'dim_savings_pct': float((256 - avg_dims) / 256 * 100),
        'resolution_distribution': {
            f'j{j} ({j*64}d)': resolution_distribution[j] / n_queries
            for j in [1, 2, 3, 4]
        },
    }


def twostage_retrieval(
    query_embs_j1: np.ndarray,
    query_embs_j4: np.ndarray,
    doc_embs_j1: np.ndarray,
    doc_embs_j4: np.ndarray,
    query_l0: np.ndarray,
    query_l1: np.ndarray,
    doc_l0: np.ndarray,
    doc_l1: np.ndarray,
    first_pass_k: int = 50,
    rerank_k: int = 5,
) -> Dict:
    """
    Two-stage pipeline: 64d first-pass retrieval + 256d reranking.
    This is the standard industry pattern we're trying to replace.
    """
    n_queries = len(query_l0)
    l0_correct = 0
    l1_correct = 0

    for i in range(n_queries):
        # Stage 1: Coarse retrieval with 64d
        sims_coarse = doc_embs_j1 @ query_embs_j1[i]
        top_k_coarse = np.argsort(-sims_coarse)[:first_pass_k]

        # Stage 2: Rerank top-k with 256d
        candidate_embs = doc_embs_j4[top_k_coarse]
        sims_fine = candidate_embs @ query_embs_j4[i]
        top_k_fine = top_k_coarse[np.argsort(-sims_fine)[:rerank_k]]

        # Predict from reranked top-k
        pred_l0_labels = doc_l0[top_k_fine]
        pred_l1_labels = doc_l1[top_k_fine]
        unique_l0, counts_l0 = np.unique(pred_l0_labels, return_counts=True)
        unique_l1, counts_l1 = np.unique(pred_l1_labels, return_counts=True)
        pred_l0 = unique_l0[np.argmax(counts_l0)]
        pred_l1 = unique_l1[np.argmax(counts_l1)]

        if pred_l0 == query_l0[i]:
            l0_correct += 1
        if pred_l1 == query_l1[i]:
            l1_correct += 1

    # Average dims: 50 docs * 64d first pass + 5 docs * 256d rerank
    # Normalized: (50*64 + 5*256) / 50 = 89.6d effective per doc comparison
    avg_effective_dims = (first_pass_k * 64 + rerank_k * 256) / first_pass_k

    return {
        'l0_accuracy': l0_correct / n_queries,
        'l1_accuracy': l1_correct / n_queries,
        'avg_effective_dims': float(avg_effective_dims),
        'first_pass_k': first_pass_k,
        'rerank_k': rerank_k,
    }


def run_adaptive_retrieval(
    model_key: str = "bge-small",
    dataset_name: str = "clinc",
    seeds: Tuple[int, ...] = (42, 123, 456),
    confidence_thresholds: Tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0),
    device: str = "cuda",
):
    """Run the full adaptive retrieval comparison."""
    print("=" * 70)
    print(f"ADAPTIVE RETRIEVAL EXPERIMENT: {model_key} on {dataset_name}")
    print("=" * 70)

    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = {
        'model': model_key,
        'dataset': dataset_name,
        'seeds': list(seeds),
        'timestamp': datetime.now().isoformat(),
        'results': {},
    }

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load data
        print("[1] Loading data...")
        train_samples = load_hierarchical_dataset(dataset_name, split='train')
        test_samples = load_hierarchical_dataset(dataset_name, split='test')
        train_samples, val_samples = split_train_val(train_samples, val_frac=0.15, seed=seed)

        # Load model and train V5
        print("[2] Training V5...")
        model_config = MODELS[model_key]
        tokenizer, backbone = load_model(model_key)
        hidden_dim = model_config['hidden_dim']

        v5_model = FractalModelV5(
            backbone=backbone, tokenizer=tokenizer,
            hidden_dim=hidden_dim, output_dim=256,
            num_l0_classes=len(set(s.level0_label for s in train_samples)),
            num_l1_classes=len(set(s.level1_label for s in train_samples)),
            num_scales=4, device=device,
        )

        v5_trainer = V5Trainer(
            model=v5_model, train_samples=train_samples, val_samples=val_samples,
            device=device, batch_size=16, lr=1e-4,
        )
        v5_trainer.train(stage1_epochs=5, stage2_epochs=0)

        # Train MRL
        print("[3] Training MRL...")
        tokenizer2, backbone2 = load_model(model_key)
        mrl_model = MRLBaselineModel(
            backbone=backbone2, tokenizer=tokenizer2,
            hidden_dim=hidden_dim, output_dim=256,
            num_l0_classes=len(set(s.level0_label for s in train_samples)),
            num_l1_classes=len(set(s.level1_label for s in train_samples)),
            num_scales=4, device=device,
        )

        mrl_trainer = MRLTrainer(
            model=mrl_model, train_samples=train_samples, val_samples=val_samples,
            device=device, batch_size=16, lr=1e-4,
        )
        mrl_trainer.train(stage1_epochs=5, stage2_epochs=0)

        # Compute embeddings at all prefix lengths
        print("[4] Computing embeddings...")
        test_texts = [s.text for s in test_samples]
        test_l0 = np.array([s.level0_label for s in test_samples])
        test_l1 = np.array([s.level1_label for s in test_samples])

        v5_embs = compute_embeddings(v5_model, test_texts)
        mrl_embs = compute_embeddings(mrl_model, test_texts)

        # Split into queries and docs (50/50)
        n = len(test_texts)
        perm = np.random.permutation(n)
        q_idx = perm[:n // 2]
        d_idx = perm[n // 2:]

        seed_results = {
            'v5': {'fixed': {}, 'adaptive': {}, 'twostage': {}},
            'mrl': {'fixed': {}, 'adaptive': {}, 'twostage': {}},
        }

        for method_name, embs in [('v5', v5_embs), ('mrl', mrl_embs)]:
            print(f"\n[5] Evaluating {method_name.upper()}...")

            # Fixed-dimension baselines
            for j in [1, 4]:
                result = fixed_retrieval(
                    embs[j][q_idx], embs[j][d_idx],
                    test_l0[d_idx], test_l1[d_idx], k=5,
                )
                # Need to fix: use query labels for accuracy
                # Recompute with leave-one-out style
                l0_correct = 0
                l1_correct = 0
                for qi in q_idx:
                    pred_l0, _ = knn_predict(embs[j][qi], embs[j][d_idx], test_l0[d_idx], k=5)
                    pred_l1, _ = knn_predict(embs[j][qi], embs[j][d_idx], test_l1[d_idx], k=5)
                    if pred_l0 == test_l0[qi]:
                        l0_correct += 1
                    if pred_l1 == test_l1[qi]:
                        l1_correct += 1
                seed_results[method_name]['fixed'][f'j{j}'] = {
                    'l0_accuracy': l0_correct / len(q_idx),
                    'l1_accuracy': l1_correct / len(q_idx),
                    'dims': j * 64,
                }
                print(f"  Fixed j={j} ({j*64}d): L0={l0_correct/len(q_idx):.4f}, L1={l1_correct/len(q_idx):.4f}")

            # Adaptive retrieval at different thresholds
            for threshold in confidence_thresholds:
                q_embs_by_j = {j: embs[j][q_idx] for j in [1, 2, 3, 4]}
                d_embs_by_j = {j: embs[j][d_idx] for j in [1, 2, 3, 4]}

                result = adaptive_retrieval(
                    q_embs_by_j, d_embs_by_j,
                    test_l0[q_idx], test_l1[q_idx],
                    test_l0[d_idx], test_l1[d_idx],
                    confidence_threshold=threshold, k=5,
                )
                seed_results[method_name]['adaptive'][f'thresh_{threshold}'] = result
                print(f"  Adaptive (t={threshold}): L0={result['l0_accuracy']:.4f}, "
                      f"L1={result['l1_accuracy']:.4f}, "
                      f"avg_dims={result['avg_dims_used']:.0f}d, "
                      f"savings={result['dim_savings_pct']:.1f}%")

            # Two-stage pipeline
            result = twostage_retrieval(
                embs[1][q_idx], embs[4][q_idx],
                embs[1][d_idx], embs[4][d_idx],
                test_l0[q_idx], test_l1[q_idx],
                test_l0[d_idx], test_l1[d_idx],
                first_pass_k=50, rerank_k=5,
            )
            seed_results[method_name]['twostage'] = result
            print(f"  Two-stage (50->5): L0={result['l0_accuracy']:.4f}, "
                  f"L1={result['l1_accuracy']:.4f}, "
                  f"eff_dims={result['avg_effective_dims']:.0f}d")

        all_results['results'][str(seed)] = seed_results

        # Clean up GPU
        del v5_model, mrl_model, v5_trainer, mrl_trainer
        torch.cuda.empty_cache()
        import gc; gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("ADAPTIVE RETRIEVAL SUMMARY")
    print("=" * 70)

    for method in ['v5', 'mrl']:
        print(f"\n  {method.upper()}:")
        for seed in seeds:
            sr = all_results['results'][str(seed)][method]
            fixed_64 = sr['fixed']['j1']
            fixed_256 = sr['fixed']['j4']
            best_adaptive = None
            best_adaptive_key = None
            for key, val in sr['adaptive'].items():
                if best_adaptive is None or val['l1_accuracy'] > best_adaptive['l1_accuracy']:
                    best_adaptive = val
                    best_adaptive_key = key
            print(f"    Seed {seed}:")
            print(f"      Fixed 64d:  L0={fixed_64['l0_accuracy']:.4f}, L1={fixed_64['l1_accuracy']:.4f}")
            print(f"      Fixed 256d: L0={fixed_256['l0_accuracy']:.4f}, L1={fixed_256['l1_accuracy']:.4f}")
            print(f"      Best adaptive ({best_adaptive_key}): L0={best_adaptive['l0_accuracy']:.4f}, "
                  f"L1={best_adaptive['l1_accuracy']:.4f}, avg_dims={best_adaptive['avg_dims_used']:.0f}d")
            print(f"      Two-stage:  L0={sr['twostage']['l0_accuracy']:.4f}, "
                  f"L1={sr['twostage']['l1_accuracy']:.4f}")

    # Save results
    out_path = RESULTS_DIR / f"adaptive_retrieval_{model_key}_{dataset_name}.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_adaptive_retrieval(
        model_key="bge-small",
        dataset_name="clinc",
        seeds=(42, 123, 456),
    )
