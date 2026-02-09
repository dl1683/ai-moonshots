"""
FAISS Latency Benchmark + Noisy Hierarchy Sensitivity Test

Part 1: FAISS latency at different dimensionalities (64d, 128d, 192d, 256d)
         to ground the Pareto analysis in real hardware measurements.

Part 2: Noisy hierarchy sensitivity - test V5 steerability when L0 labels
         are randomly corrupted at 10%, 20%, 30% rates.
"""

import sys, os, json, time
import numpy as np
import faiss
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

ROOT = Path(__file__).resolve().parent.parent


def run_faiss_latency_benchmark():
    """Measure actual query latency at different embedding dimensionalities using FAISS."""
    print("=" * 70)
    print("FAISS LATENCY BENCHMARK")
    print("=" * 70)

    # Load CLINC benchmark data to get realistic embeddings
    benchmark_file = ROOT / "results" / "benchmark_bge-small_clinc.json"

    # We'll use synthetic data of realistic size if benchmark not available
    # CLINC has ~10K train, 2K test
    n_database = 10000
    n_queries = 2000
    dims = [64, 128, 192, 256]
    n_warmup = 100
    n_trials = 5

    results = {}

    for d in dims:
        print(f"\n--- Dimensionality: {d}d ---")

        # Generate random unit-norm vectors (realistic for normalized embeddings)
        np.random.seed(42)
        database = np.random.randn(n_database, d).astype(np.float32)
        # Normalize to unit vectors (cosine similarity via IP)
        norms = np.linalg.norm(database, axis=1, keepdims=True)
        database = database / norms

        queries = np.random.randn(n_queries, d).astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / norms

        # Build FAISS index (flat IP = cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(d)

        # Measure index build time
        t0 = time.perf_counter()
        index.add(database)
        build_time = time.perf_counter() - t0

        # Warmup
        for _ in range(n_warmup):
            index.search(queries[:1], 5)

        # Measure single-query latency (microseconds)
        single_latencies = []
        for i in range(min(500, n_queries)):
            t0 = time.perf_counter()
            index.search(queries[i:i+1], 5)
            single_latencies.append((time.perf_counter() - t0) * 1e6)  # microseconds

        # Measure batch query throughput
        batch_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            index.search(queries, 5)
            batch_times.append(time.perf_counter() - t0)

        mean_single_us = np.mean(single_latencies)
        p50_single_us = np.percentile(single_latencies, 50)
        p99_single_us = np.percentile(single_latencies, 99)
        mean_batch_s = np.mean(batch_times)
        qps = n_queries / mean_batch_s

        # Memory: d * n_database * 4 bytes (float32)
        memory_mb = d * n_database * 4 / (1024 * 1024)

        results[d] = {
            "dim": d,
            "n_database": n_database,
            "n_queries": n_queries,
            "build_time_ms": build_time * 1000,
            "single_query_mean_us": mean_single_us,
            "single_query_p50_us": p50_single_us,
            "single_query_p99_us": p99_single_us,
            "batch_time_s": mean_batch_s,
            "throughput_qps": qps,
            "memory_mb": memory_mb,
        }

        print(f"  Build: {build_time*1000:.1f} ms")
        print(f"  Single query: {mean_single_us:.0f} us (p50={p50_single_us:.0f}, p99={p99_single_us:.0f})")
        print(f"  Batch ({n_queries} queries): {mean_batch_s*1000:.1f} ms ({qps:.0f} QPS)")
        print(f"  Memory: {memory_mb:.1f} MB")

    # Summary comparison
    print("\n" + "=" * 70)
    print("LATENCY SUMMARY")
    print("=" * 70)
    print(f"{'Dim':>6} | {'Mean (us)':>10} | {'P50 (us)':>10} | {'P99 (us)':>10} | {'QPS':>8} | {'Memory':>8} | {'Speedup':>8}")
    print("-" * 80)
    base = results[256]["single_query_mean_us"]
    for d in dims:
        r = results[d]
        speedup = base / r["single_query_mean_us"]
        print(f"{d:>5}d | {r['single_query_mean_us']:>10.0f} | {r['single_query_p50_us']:>10.0f} | {r['single_query_p99_us']:>10.0f} | {r['throughput_qps']:>8.0f} | {r['memory_mb']:>6.1f} MB | {speedup:>7.2f}x")

    # Also test with HNSW (approximate) for larger-scale realism
    print("\n" + "=" * 70)
    print("HNSW INDEX (Approximate NN) - More realistic for production")
    print("=" * 70)

    n_database_large = 100000
    hnsw_results = {}

    for d in dims:
        print(f"\n--- HNSW {d}d (n={n_database_large}) ---")
        np.random.seed(42)
        database = np.random.randn(n_database_large, d).astype(np.float32)
        norms = np.linalg.norm(database, axis=1, keepdims=True)
        database = database / norms

        queries = np.random.randn(n_queries, d).astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / norms

        # HNSW index
        index = faiss.IndexHNSWFlat(d, 32)  # M=32 connections
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16

        t0 = time.perf_counter()
        index.add(database)
        build_time = time.perf_counter() - t0

        # Warmup
        for _ in range(50):
            index.search(queries[:1], 5)

        # Measure
        single_latencies = []
        for i in range(min(500, n_queries)):
            t0 = time.perf_counter()
            index.search(queries[i:i+1], 5)
            single_latencies.append((time.perf_counter() - t0) * 1e6)

        batch_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            index.search(queries, 5)
            batch_times.append(time.perf_counter() - t0)

        hnsw_results[d] = {
            "dim": d,
            "n_database": n_database_large,
            "build_time_s": build_time,
            "single_query_mean_us": np.mean(single_latencies),
            "single_query_p50_us": np.percentile(single_latencies, 50),
            "batch_time_s": np.mean(batch_times),
            "throughput_qps": n_queries / np.mean(batch_times),
        }

        print(f"  Build: {build_time:.1f} s")
        print(f"  Single query: {np.mean(single_latencies):.0f} us (p50={np.percentile(single_latencies, 50):.0f})")
        print(f"  Batch: {np.mean(batch_times)*1000:.1f} ms ({n_queries/np.mean(batch_times):.0f} QPS)")

    print("\n" + "=" * 70)
    print("HNSW LATENCY SUMMARY (n=100K)")
    print("=" * 70)
    print(f"{'Dim':>6} | {'Mean (us)':>10} | {'QPS':>8} | {'Speedup':>8}")
    print("-" * 50)
    base = hnsw_results[256]["single_query_mean_us"]
    for d in dims:
        r = hnsw_results[d]
        speedup = base / r["single_query_mean_us"]
        print(f"{d:>5}d | {r['single_query_mean_us']:>10.0f} | {r['throughput_qps']:>8.0f} | {speedup:>7.2f}x")

    return {"flat": results, "hnsw": hnsw_results}


def run_noisy_hierarchy_test():
    """Test V5 steerability when L0 hierarchy labels are corrupted."""
    print("\n\n" + "=" * 70)
    print("NOISY HIERARCHY SENSITIVITY TEST")
    print("=" * 70)

    from hierarchical_datasets import load_clinc150
    from fractal_v5 import FractalV5Trainer, FractalConfig
    from mrl_v5_baseline import MRLBaselineTrainer
    from copy import copy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_rates = [0.0, 0.10, 0.20, 0.30, 0.50]
    seeds = [42, 123, 456]

    results = {}

    for noise_rate in noise_rates:
        print(f"\n{'='*50}")
        print(f"  NOISE RATE: {noise_rate*100:.0f}%")
        print(f"{'='*50}")

        seed_results = []

        for seed in seeds:
            print(f"\n  Seed {seed}...")
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Load CLINC
            train_data = load_clinc150(split="train")
            val_data = load_clinc150(split="test")

            # Corrupt L0 labels in training data
            if noise_rate > 0:
                num_l0 = len(set(s.level0_label for s in train_data.samples))
                n_corrupt = int(len(train_data.samples) * noise_rate)
                corrupt_indices = np.random.choice(len(train_data.samples), n_corrupt, replace=False)

                for idx in corrupt_indices:
                    s = train_data.samples[idx]
                    # Assign random wrong L0 label
                    wrong_labels = [l for l in range(num_l0) if l != s.level0_label]
                    new_l0 = np.random.choice(wrong_labels)
                    # Create modified sample
                    new_s = copy(s)
                    new_s.level0_label = int(new_l0)
                    new_s.level0_name = train_data.level0_names[int(new_l0)]
                    train_data.samples[idx] = new_s

            num_l0 = len(set(s.level0_label for s in train_data.samples))
            num_l1 = len(set(s.level1_label for s in train_data.samples))

            # Train V5
            config = FractalConfig(
                model_name="BAAI/bge-small-en-v1.5",
                output_dim=256,
                num_top_classes=num_l0,
                num_bot_classes=num_l1,
                epochs=5,
                batch_size=16,
                lr=1e-4,
            )

            v5_trainer = FractalV5Trainer(config, device=device)
            v5_trainer.train(train_data, val_data)

            # Evaluate steerability on CLEAN val data
            clean_val = load_clinc150(split="test")
            v5_accs = {}
            for j in [1, 2, 3, 4]:
                prefix_len = j * 64
                l0_acc, l1_acc = v5_trainer.evaluate_knn(clean_val, prefix_dim=prefix_len)
                v5_accs[j] = {"l0": l0_acc, "l1": l1_acc}

            v5_steer = (v5_accs[1]["l0"] - v5_accs[4]["l0"]) + (v5_accs[4]["l1"] - v5_accs[1]["l1"])

            print(f"    V5 steer = {v5_steer:+.4f}")
            print(f"    V5 L0@j1={v5_accs[1]['l0']:.3f}, L0@j4={v5_accs[4]['l0']:.3f}")
            print(f"    V5 L1@j1={v5_accs[1]['l1']:.3f}, L1@j4={v5_accs[4]['l1']:.3f}")

            seed_results.append({
                "seed": seed,
                "noise_rate": noise_rate,
                "v5_steer": v5_steer,
                "v5_accs": v5_accs,
            })

            # Clean up GPU memory
            del v5_trainer
            torch.cuda.empty_cache()

        steers = [r["v5_steer"] for r in seed_results]
        mean_steer = np.mean(steers)
        std_steer = np.std(steers, ddof=1)

        results[noise_rate] = {
            "noise_rate": noise_rate,
            "mean_steer": mean_steer,
            "std_steer": std_steer,
            "seeds": seed_results,
        }

        print(f"\n  Noise {noise_rate*100:.0f}%: V5 Steer = {mean_steer:+.4f} +/- {std_steer:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("NOISY HIERARCHY SUMMARY (CLINC, 3 seeds)")
    print("=" * 70)
    print(f"{'Noise %':>8} | {'V5 Steer':>12} | {'Retention':>10}")
    print("-" * 40)
    base_steer = results[0.0]["mean_steer"]
    for nr in noise_rates:
        r = results[nr]
        retention = r["mean_steer"] / base_steer * 100 if base_steer != 0 else 0
        print(f"{nr*100:>7.0f}% | {r['mean_steer']:>+.4f} +/- {r['std_steer']:.4f} | {retention:>8.1f}%")

    return results


if __name__ == "__main__":
    all_results = {}

    # Part 1: FAISS Latency
    latency_results = run_faiss_latency_benchmark()
    all_results["latency"] = latency_results

    # Part 2: Noisy Hierarchy
    noisy_results = run_noisy_hierarchy_test()
    all_results["noisy_hierarchy"] = {str(k): v for k, v in noisy_results.items()}

    # Save results
    out_path = ROOT / "results" / "latency_and_robustness.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nResults saved to {out_path}")
