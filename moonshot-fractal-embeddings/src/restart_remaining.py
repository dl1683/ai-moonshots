"""
Restart remaining experiments after hibernate CUDA crash.

What's done:
  - Yahoo bge-small: 3 seeds x 3 methods ✓
  - CLINC bge-small: 3 seeds x 3 methods ✓
  - CLINC ablations: running separately (ablation_steerability.py)
  - Qwen3-0.6B MRL seed 0: ✓
  - TREC bge-small MRL seeds 42, 123 ✓ (from stdout, JSON has last seed only)

What needs to run:
  1. TREC bge-small: MRL seed 456, V5 3 seeds
  2. DBPedia bge-small: flat + MRL + V5, 3 seeds each
  3. Newsgroups bge-small: flat + MRL + V5, 3 seeds each
  4. Qwen3-0.6B MRL: seeds 1-4

This script handles items 1-3 (bge-small). Qwen3 is separate.
"""

import sys
import os
import json
import gc
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import load_hierarchical_dataset
from fractal_v5 import run_v5_experiment
from mrl_v5_baseline import run_mrl_experiment
from multi_model_pipeline import MODELS, load_model


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_flat_eval(model_key, dataset_name, device="cuda"):
    """Evaluate unfinetuned baseline using raw backbone embeddings."""
    from transformers import AutoModel, AutoTokenizer

    test_data = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    config = MODELS[model_key]

    # Load backbone directly (no fractal head)
    backbone = AutoModel.from_pretrained(config.hf_path, trust_remote_code=config.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_path, trust_remote_code=config.trust_remote_code)
    backbone.eval().to(device)

    texts = [s.text for s in test_data.samples]
    l0_labels = np.array([s.level0_label for s in test_data.samples])
    l1_labels = np.array([s.level1_label for s in test_data.samples])

    # Encode
    all_embs = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch_texts = texts[i:i+bs]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = backbone(**enc)
            emb = out.last_hidden_state[:, 0].cpu().numpy()
            all_embs.append(emb)

    emb = np.concatenate(all_embs)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    sims = emb @ emb.T
    np.fill_diagonal(sims, -1)

    k = 5
    l0_c = l1_c = 0
    for i in range(len(texts)):
        top5 = np.argsort(-sims[i])[:k]
        if np.bincount(l0_labels[top5]).argmax() == l0_labels[i]:
            l0_c += 1
        if np.bincount(l1_labels[top5]).argmax() == l1_labels[i]:
            l1_c += 1

    l0_acc = l0_c / len(texts)
    l1_acc = l1_c / len(texts)

    del backbone, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return {"l0_accuracy": l0_acc, "l1_accuracy": l1_acc}


def run_dataset_benchmark(dataset_name, model_key="bge-small", seeds=[42, 123, 456], device="cuda"):
    """Run full benchmark for a single dataset."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {dataset_name} on {model_key}")
    print(f"{'='*70}")

    results = {"dataset": dataset_name, "model": model_key, "seeds": seeds}

    # Flat baseline (same for all seeds)
    print(f"\n--- Flat baseline ---")
    flat = run_flat_eval(model_key, dataset_name, device)
    results["flat"] = flat
    print(f"  Flat: L0={flat['l0_accuracy']:.4f}, L1={flat['l1_accuracy']:.4f}")

    # MRL
    print(f"\n--- MRL ({len(seeds)} seeds) ---")
    mrl_results = {}
    for seed in seeds:
        print(f"\n  [MRL seed={seed}]")
        try:
            r = run_mrl_experiment(
                model_key=model_key,
                dataset_name=dataset_name,
                stage1_epochs=5,
                seed=seed,
                device=device,
            )
            mrl_results[seed] = r
            print(f"  L0={r['mrl']['l0_accuracy']:.4f}, L1={r['mrl']['l1_accuracy']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            mrl_results[seed] = {"error": str(e)}

    results["mrl"] = mrl_results

    # V5
    print(f"\n--- V5 ({len(seeds)} seeds) ---")
    v5_results = {}
    for seed in seeds:
        print(f"\n  [V5 seed={seed}]")
        try:
            r = run_v5_experiment(
                model_key=model_key,
                dataset_name=dataset_name,
                stage1_epochs=5,
                seed=seed,
                device=device,
            )
            v5_results[seed] = r
            print(f"  L0={r['v5']['l0_accuracy']:.4f}, L1={r['v5']['l1_accuracy']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            v5_results[seed] = {"error": str(e)}

    results["v5"] = v5_results

    # Save
    out_path = RESULTS_DIR / f"benchmark_{model_key}_{dataset_name}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["trec", "dbpedia", "newsgroups"])
    parser.add_argument("--model", default="bge-small")
    args = parser.parse_args()

    all_results = {}
    for ds in args.datasets:
        try:
            r = run_dataset_benchmark(ds, model_key=args.model)
            all_results[ds] = r
        except Exception as e:
            print(f"\nFATAL ERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ds] = {"error": str(e)}

    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    for ds, r in all_results.items():
        if "error" in r:
            print(f"  {ds}: FAILED - {r['error']}")
            continue
        flat = r.get("flat", {})
        print(f"\n  {ds}:")
        print(f"    Flat: L0={flat.get('l0_accuracy', 'N/A'):.4f}, L1={flat.get('l1_accuracy', 'N/A'):.4f}")
        for method in ["mrl", "v5"]:
            method_results = r.get(method, {})
            accs = []
            for seed, sr in method_results.items():
                if isinstance(sr, dict) and "error" not in sr:
                    method_data = sr.get(method if method != "mrl" else "mrl", sr.get("v5", {}))
                    if "l0_accuracy" in method_data:
                        accs.append((method_data["l0_accuracy"], method_data["l1_accuracy"]))
            if accs:
                l0s = [a[0] for a in accs]
                l1s = [a[1] for a in accs]
                print(f"    {method.upper()}: L0={np.mean(l0s):.4f}+/-{np.std(l0s):.4f}, L1={np.mean(l1s):.4f}+/-{np.std(l1s):.4f}")
