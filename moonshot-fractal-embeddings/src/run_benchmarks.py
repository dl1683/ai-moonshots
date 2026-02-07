"""
Comprehensive Benchmark Runner: V5 vs MRL vs Flat Baseline
============================================================

Runs all three methods on all datasets with multi-seed validation.
Produces publication-ready comparison tables.

Methods:
  - Flat: Unfinetuned base embedding model (KNN baseline)
  - MRL:  Matryoshka Representation Learning (same loss at all prefix lengths)
  - V5:   Fractal V5 with hierarchy-aligned progressive prefix supervision

Datasets: Yahoo Answers, 20 Newsgroups, CLINC150, TREC, DBPedia, AG News
"""

import json
import time
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


def evaluate_flat_baseline(model_key, dataset_name, device="cuda", max_test=2000):
    """Evaluate unfinetuned model with KNN — the flat baseline."""
    from multi_model_pipeline import load_model
    from hierarchical_datasets import load_hierarchical_dataset

    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_test)
    model = load_model(model_key, use_fractal=False, device=device)

    samples = test_data.samples[:max_test]
    texts = [s.text for s in samples]
    l0_labels = np.array([s.level0_label for s in samples])
    l1_labels = np.array([s.level1_label for s in samples])

    embeddings = model.encode(texts, batch_size=32).numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def knn_acc(emb, labels, k=5):
        correct = 0
        for i in range(len(emb)):
            sims = emb @ emb[i]
            sims[i] = -float("inf")
            top_k = np.argsort(-sims)[:k]
            neighbor_labels = labels[top_k]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            pred = unique[np.argmax(counts)]
            if pred == labels[i]:
                correct += 1
        return correct / len(emb)

    results = {
        "l0_accuracy": knn_acc(embeddings, l0_labels),
        "l1_accuracy": knn_acc(embeddings, l1_labels),
    }

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def run_single_benchmark(
    method,
    model_key,
    dataset_name,
    seed,
    stage1_epochs=5,
    batch_size=32,
    device="cuda",
):
    """Run a single method/dataset/seed combination."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if method == "flat":
        return evaluate_flat_baseline(model_key, dataset_name, device=device)

    elif method == "v5":
        from fractal_v5 import run_v5_experiment

        results = run_v5_experiment(
            model_key=model_key,
            dataset_name=dataset_name,
            stage1_epochs=stage1_epochs,
            stage2_epochs=0,
            batch_size=batch_size,
            device=device,
        )
        return {
            "l0_accuracy": results["v5"]["l0_accuracy"],
            "l1_accuracy": results["v5"]["l1_accuracy"],
            "prefix_accuracy": results.get("prefix_accuracy", {}),
        }

    elif method == "mrl":
        from mrl_v5_baseline import run_mrl_experiment

        results = run_mrl_experiment(
            model_key=model_key,
            dataset_name=dataset_name,
            stage1_epochs=stage1_epochs,
            stage2_epochs=0,
            batch_size=batch_size,
            device=device,
        )
        return {
            "l0_accuracy": results["mrl"]["l0_accuracy"],
            "l1_accuracy": results["mrl"]["l1_accuracy"],
            "prefix_accuracy": results.get("prefix_accuracy", {}),
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def run_full_benchmark(
    model_key="bge-small",
    datasets=None,
    methods=None,
    seeds=None,
    stage1_epochs=5,
    batch_size=32,
    device="cuda",
):
    """
    Run full benchmark suite across all datasets, methods, and seeds.
    """
    if datasets is None:
        datasets = ["yahoo", "newsgroups", "clinc", "trec", "dbpedia"]
    if methods is None:
        methods = ["flat", "mrl", "v5"]
    if seeds is None:
        seeds = [42, 123, 456]

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    start_time = time.time()

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*70}")

        all_results[ds_name] = {}

        for method in methods:
            print(f"\n--- Method: {method} ---")
            all_results[ds_name][method] = {"seeds": {}}

            for seed in seeds:
                print(f"\n  Seed {seed}:")
                try:
                    result = run_single_benchmark(
                        method=method,
                        model_key=model_key,
                        dataset_name=ds_name,
                        seed=seed,
                        stage1_epochs=stage1_epochs,
                        batch_size=batch_size,
                        device=device,
                    )
                    all_results[ds_name][method]["seeds"][str(seed)] = result
                    print(f"    L0: {result['l0_accuracy']:.4f}, L1: {result['l1_accuracy']:.4f}")

                    # Flat baseline is deterministic, so only run once
                    if method == "flat":
                        for remaining_seed in seeds[1:]:
                            all_results[ds_name][method]["seeds"][str(remaining_seed)] = result
                        break

                except Exception as e:
                    print(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[ds_name][method]["seeds"][str(seed)] = {
                        "error": str(e),
                        "l0_accuracy": 0.0,
                        "l1_accuracy": 0.0,
                    }

                # Cleanup between seeds
                torch.cuda.empty_cache()
                gc.collect()

            # Compute mean/std across seeds
            l0_vals = [
                v["l0_accuracy"]
                for v in all_results[ds_name][method]["seeds"].values()
                if "error" not in v
            ]
            l1_vals = [
                v["l1_accuracy"]
                for v in all_results[ds_name][method]["seeds"].values()
                if "error" not in v
            ]
            if l0_vals:
                all_results[ds_name][method]["l0_mean"] = float(np.mean(l0_vals))
                all_results[ds_name][method]["l0_std"] = float(np.std(l0_vals))
                all_results[ds_name][method]["l1_mean"] = float(np.mean(l1_vals))
                all_results[ds_name][method]["l1_std"] = float(np.std(l1_vals))

    elapsed = time.time() - start_time

    # Print summary table
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {model_key} | Seeds: {seeds} | Epochs: {stage1_epochs}")
    print(f"Total time: {elapsed/60:.1f} minutes\n")

    # L0 table
    print(f"{'Dataset':<15}", end="")
    for m in methods:
        print(f"  {m+' L0':>14}", end="")
    print()
    print("-" * (15 + 16 * len(methods)))
    for ds_name in datasets:
        print(f"{ds_name:<15}", end="")
        for m in methods:
            data = all_results[ds_name].get(m, {})
            mean = data.get("l0_mean", 0)
            std = data.get("l0_std", 0)
            print(f"  {mean*100:>6.2f}±{std*100:<5.2f}", end="")
        print()

    print()

    # L1 table
    print(f"{'Dataset':<15}", end="")
    for m in methods:
        print(f"  {m+' L1':>14}", end="")
    print()
    print("-" * (15 + 16 * len(methods)))
    for ds_name in datasets:
        print(f"{ds_name:<15}", end="")
        for m in methods:
            data = all_results[ds_name].get(m, {})
            mean = data.get("l1_mean", 0)
            std = data.get("l1_std", 0)
            print(f"  {mean*100:>6.2f}±{std*100:<5.2f}", end="")
        print()

    # V5 vs MRL delta
    if "v5" in methods and "mrl" in methods:
        print(f"\n{'='*80}")
        print("V5 vs MRL (hierarchy-aligned vs standard)")
        print(f"{'='*80}")
        print(f"{'Dataset':<15}  {'L0 delta':>10}  {'L1 delta':>10}  {'V5 wins?':>10}")
        print("-" * 50)
        for ds_name in datasets:
            v5 = all_results[ds_name].get("v5", {})
            mrl = all_results[ds_name].get("mrl", {})
            dl0 = v5.get("l0_mean", 0) - mrl.get("l0_mean", 0)
            dl1 = v5.get("l1_mean", 0) - mrl.get("l1_mean", 0)
            wins = "YES" if (dl0 > 0 and dl1 > 0) else ("MIXED" if dl0 > 0 or dl1 > 0 else "NO")
            print(f"{ds_name:<15}  {dl0*100:>+10.2f}  {dl1*100:>+10.2f}  {wins:>10}")

    # Save
    output = {
        "model": model_key,
        "datasets": datasets,
        "methods": methods,
        "seeds": seeds,
        "stage1_epochs": stage1_epochs,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = results_dir / f"benchmark_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(convert(output), f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    parser.add_argument("--model", type=str, default="bge-small")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["yahoo", "newsgroups", "clinc", "trec", "dbpedia"],
    )
    parser.add_argument("--methods", nargs="+", default=["flat", "mrl", "v5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_full_benchmark(
        model_key=args.model,
        datasets=args.datasets,
        methods=args.methods,
        seeds=args.seeds,
        stage1_epochs=args.epochs,
        batch_size=args.batch_size,
    )
