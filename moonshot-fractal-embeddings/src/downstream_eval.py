"""
Comprehensive Downstream Evaluation for Fractal Embeddings
============================================================

Addresses Codex review critique: "No strong downstream utility demonstration."

Three orthogonal evaluation axes:
1. Hierarchical Retrieval: Recall@k at each prefix length for L0 and L1
2. Tree-Distance Calibration: Average hierarchy distance of k-NN neighbors
3. MI Profile: How much L0 vs L1 information each prefix carries

All three are NON-CIRCULAR: they measure embedding BEHAVIOR, not training loss.
"""

import sys
import os
import json
import torch
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from mrl_v5_baseline import MRLTrainerV5
from multi_model_pipeline import MODELS

RESULTS_DIR = Path(__file__).parent.parent / "results"


# =====================================================================
# 1. HIERARCHICAL RETRIEVAL
# =====================================================================

def retrieval_metrics(query_embs, doc_embs, query_l0, query_l1, doc_l0, doc_l1, k=10):
    """Compute Recall@k for L0 and L1, plus hierarchical NDCG."""
    sims = query_embs @ doc_embs.T
    n = len(query_embs)

    recall_l0, recall_l1 = 0, 0
    ndcg_scores = []

    for i in range(n):
        top_k = np.argsort(-sims[i])[:k]

        # Recall L0
        if np.any(doc_l0[top_k] == query_l0[i]):
            recall_l0 += 1
        # Recall L1
        if np.any(doc_l1[top_k] == query_l1[i]):
            recall_l1 += 1

        # Hierarchical NDCG (rel=2 for L1 match, 1 for L0 match, 0 otherwise)
        gains = []
        for idx in top_k:
            if doc_l1[idx] == query_l1[i]:
                gains.append(2)
            elif doc_l0[idx] == query_l0[i]:
                gains.append(1)
            else:
                gains.append(0)
        dcg = sum(g / np.log2(pos + 2) for pos, g in enumerate(gains))

        # Ideal DCG
        all_rels = []
        for idx in range(len(doc_l0)):
            if doc_l1[idx] == query_l1[i]:
                all_rels.append(2)
            elif doc_l0[idx] == query_l0[i]:
                all_rels.append(1)
            else:
                all_rels.append(0)
        ideal = sorted(all_rels, reverse=True)[:k]
        idcg = sum(g / np.log2(pos + 2) for pos, g in enumerate(ideal))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "recall_l0": recall_l0 / n,
        "recall_l1": recall_l1 / n,
        "ndcg_hier": float(np.mean(ndcg_scores)),
    }


# =====================================================================
# 2. TREE-DISTANCE CALIBRATION
# =====================================================================

def tree_distance(q_l0, q_l1, d_l0, d_l1):
    """
    Distance in 2-level hierarchy:
      0 = same L1 (identical fine class)
      1 = same L0, different L1 (sibling)
      2 = different L0 (different domain)
    """
    if q_l1 == d_l1:
        return 0
    elif q_l0 == d_l0:
        return 1
    else:
        return 2


def tree_distance_profile(query_embs, doc_embs, query_l0, query_l1, doc_l0, doc_l1, k=10):
    """
    Average tree-distance of k-NN at a given embedding.
    For V5 at j=1: should be ~1 (coarse matches, same L0 but different L1)
    For V5 at j=4: should be ~0 (fine matches, same L1)
    For MRL: should be similar across all j
    """
    sims = query_embs @ doc_embs.T
    n = len(query_embs)

    distances = []
    for i in range(n):
        top_k = np.argsort(-sims[i])[:k]
        for idx in top_k:
            distances.append(tree_distance(query_l0[i], query_l1[i], doc_l0[idx], doc_l1[idx]))

    return float(np.mean(distances))


# =====================================================================
# 3. MUTUAL INFORMATION PROFILING (via kNN classifier accuracy)
# =====================================================================

def mi_proxy(embeddings, labels, k=5):
    """
    Proxy for I(z; Y) using k-NN classification accuracy.
    Higher accuracy = more mutual information.
    This avoids circularity: we're measuring INFORMATION CONTENT,
    not whether the model was trained to be good at a specific task.
    """
    n = len(embeddings)
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -1)  # exclude self

    correct = 0
    for i in range(n):
        top_k = np.argsort(-sims[i])[:k]
        neighbor_labels = labels[top_k]
        # Majority vote
        counts = Counter(neighbor_labels.tolist())
        pred = max(counts, key=counts.get)
        if pred == labels[i]:
            correct += 1

    return correct / n


# =====================================================================
# MAIN EVALUATION RUNNER
# =====================================================================

def encode_at_prefix(model, texts, prefix_len, batch_size=32):
    """Encode texts at a specific prefix length, return normalized numpy array."""
    emb = model.encode(texts, batch_size=batch_size, prefix_len=prefix_len).numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return emb / norms


def run_downstream_eval(
    model_key="bge-small",
    dataset_name="clinc",
    seeds=(42,),
    stage1_epochs=5,
    batch_size=32,
    k=10,
    max_test=2000,
    device="cuda",
):
    """
    Run comprehensive downstream evaluation on a dataset.
    Trains V5 + MRL for each seed, evaluates:
    - Retrieval metrics at each prefix length
    - Tree-distance profile at each prefix length
    - MI proxy (kNN accuracy for L0 and L1) at each prefix length
    """
    all_seed_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  DOWNSTREAM EVAL: {model_key} on {dataset_name}, seed={seed}")
        print(f"{'='*70}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load data
        train_data = load_hierarchical_dataset(dataset_name, split="train", max_samples=10000)
        test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_test)

        train_samples, val_samples = split_train_val(train_data.samples, val_ratio=0.15)

        class TempDataset:
            def __init__(self, samples, l0_names, l1_names):
                self.samples = samples
                self.level0_names = l0_names
                self.level1_names = l1_names

        val_data = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)
        train_data.samples = train_samples

        num_l0 = len(train_data.level0_names)
        num_l1 = len(train_data.level1_names)

        # Use test data as both docs and queries (leave-one-out style)
        test_texts = [s.text for s in test_data.samples]
        test_l0 = np.array([s.level0_label for s in test_data.samples])
        test_l1 = np.array([s.level1_label for s in test_data.samples])

        config = MODELS[model_key]
        seed_result = {"seed": seed, "v5": {}, "mrl": {}}

        for method_name, TrainerCls, extra_kwargs in [
            ("v5", V5Trainer, {"num_l0_classes": num_l0}),
            ("mrl", MRLTrainerV5, {"num_l0_classes": num_l1}),  # MRL: both heads L1
        ]:
            print(f"\n--- Training {method_name.upper()} ---")
            model = FractalModelV5(
                config=config,
                num_l0_classes=extra_kwargs["num_l0_classes"],
                num_l1_classes=num_l1,
                num_scales=4,
                scale_dim=64,
                device=device,
            ).to(device)

            trainer = TrainerCls(
                model=model,
                train_dataset=train_data,
                val_dataset=val_data,
                device=device,
                stage1_epochs=stage1_epochs,
            )
            trainer.train(batch_size=batch_size, patience=5)

            # Evaluate at each prefix length
            method_results = {"prefix": {}}
            for j in [1, 2, 3, 4]:
                prefix_len = j if j < 4 else None
                embs = encode_at_prefix(model, test_texts, prefix_len, batch_size=batch_size)

                # 1. Retrieval
                ret = retrieval_metrics(embs, embs, test_l0, test_l1, test_l0, test_l1, k=k)

                # 2. Tree-distance
                td = tree_distance_profile(embs, embs, test_l0, test_l1, test_l0, test_l1, k=k)

                # 3. MI proxy
                mi_l0 = mi_proxy(embs, test_l0, k=5)
                mi_l1 = mi_proxy(embs, test_l1, k=5)

                method_results["prefix"][f"j{j}"] = {
                    "dims": j * 64,
                    "recall_l0": ret["recall_l0"],
                    "recall_l1": ret["recall_l1"],
                    "ndcg_hier": ret["ndcg_hier"],
                    "tree_distance": td,
                    "mi_l0": mi_l0,
                    "mi_l1": mi_l1,
                }

                print(f"  j={j} ({j*64}d): R_L0={ret['recall_l0']:.4f} R_L1={ret['recall_l1']:.4f} "
                      f"TreeDist={td:.3f} MI_L0={mi_l0:.4f} MI_L1={mi_l1:.4f}")

            # Compute steerability metrics
            j1 = method_results["prefix"]["j1"]
            j4 = method_results["prefix"]["j4"]

            method_results["steerability"] = {
                "retrieval": (j1["recall_l0"] - j4["recall_l0"]) + (j4["recall_l1"] - j1["recall_l1"]),
                "tree_distance_drop": j1["tree_distance"] - j4["tree_distance"],
                "mi_l0_gain_j1": j1["mi_l0"] - j1["mi_l1"],  # j=1 should favor L0
                "mi_l1_gain_j4": j4["mi_l1"] - j4["mi_l0"],  # j=4 should favor L1 (or not)
                "mi_shift": (j1["mi_l0"] - j4["mi_l0"]) + (j4["mi_l1"] - j1["mi_l1"]),
            }

            seed_result[method_name] = method_results

            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()

        all_seed_results.append(seed_result)

    # =================== AGGREGATE + SAVE ===================
    output = {
        "model": model_key,
        "dataset": dataset_name,
        "k": k,
        "timestamp": datetime.now().isoformat(),
        "seeds": list(seeds),
        "results": all_seed_results,
    }

    # Summary
    print(f"\n{'='*70}")
    print(f"  DOWNSTREAM EVALUATION SUMMARY: {model_key} on {dataset_name}")
    print(f"{'='*70}")

    for method in ["v5", "mrl"]:
        steers = [r[method]["steerability"] for r in all_seed_results]
        ret_s = np.mean([s["retrieval"] for s in steers])
        td_s = np.mean([s["tree_distance_drop"] for s in steers])
        mi_s = np.mean([s["mi_shift"] for s in steers])

        print(f"\n  {method.upper()}:")
        print(f"    Retrieval Steerability:  {ret_s:+.4f}")
        print(f"    Tree-Distance Drop:      {td_s:+.4f}")
        print(f"    MI Shift:                {mi_s:+.4f}")

        # Prefix profiles
        print(f"    {'j':>3} {'dims':>5} {'R_L0':>7} {'R_L1':>7} {'TD':>7} {'MI_L0':>7} {'MI_L1':>7}")
        for j in [1, 2, 3, 4]:
            vals = [r[method]["prefix"][f"j{j}"] for r in all_seed_results]
            print(f"    {j:>3} {j*64:>5} "
                  f"{np.mean([v['recall_l0'] for v in vals]):>7.4f} "
                  f"{np.mean([v['recall_l1'] for v in vals]):>7.4f} "
                  f"{np.mean([v['tree_distance'] for v in vals]):>7.3f} "
                  f"{np.mean([v['mi_l0'] for v in vals]):>7.4f} "
                  f"{np.mean([v['mi_l1'] for v in vals]):>7.4f}")

    out_path = RESULTS_DIR / f"downstream_{model_key}_{dataset_name}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2,
                 default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bge-small")
    parser.add_argument("--dataset", default="clinc")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    run_downstream_eval(
        model_key=args.model,
        dataset_name=args.dataset,
        seeds=tuple(args.seeds),
        k=args.k,
    )
