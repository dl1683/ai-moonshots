#!/usr/bin/env python
"""
cti_checkpoint_sweep.py

CTI Phase 1 Fast Path: Measure per-layer representation quality across
Pythia training checkpoints.

The key claim to test: representation quality Q(l, C, N) follows a universal
law where the optimal layer shifts with log(compute/capacity).

Universal law form (Codex design):
  logit(Q) = b_d + alpha * log(C/N^gamma) - beta * (x - (mu_0 + mu_1 * log(C/N^gamma)))^2

where x = l/L is normalized depth, C = 6*N*T is compute, N = model params.

Fast path: Pythia-160M and Pythia-1B, 12 checkpoints, 2 datasets.
Full path adds: Pythia-410M, all 4 datasets, then 1.4B holdout.

Usage:
    python -u src/cti_checkpoint_sweep.py --fast         # 160M + 1B, 2 datasets
    python -u src/cti_checkpoint_sweep.py --fit           # 160M + 410M + 1B, 4 datasets
    python -u src/cti_checkpoint_sweep.py --holdout       # 1.4B holdout test
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset

# Tokens per step for Pythia: batch_size=1024, seq_len=2048
TOKENS_PER_STEP = 1024 * 2048  # 2,097,152

# Fast checkpoint set (12 points, log-spaced)
FAST_CHECKPOINTS = [0, 1000, 2000, 4000, 8000, 16000, 32000,
                    48000, 64000, 96000, 128000, 143000]

PYTHIA_MODELS = {
    "pythia-160m": {
        "hf_path": "EleutherAI/pythia-160m",
        "N": 162_322_944,  # ~162M params
        "num_layers": 12,
        "hidden_dim": 768,
    },
    "pythia-410m": {
        "hf_path": "EleutherAI/pythia-410m",
        "N": 405_334_016,  # ~405M params
        "num_layers": 24,
        "hidden_dim": 1024,
    },
    "pythia-1b": {
        "hf_path": "EleutherAI/pythia-1b",
        "N": 1_011_781_632,  # ~1B params
        "num_layers": 16,
        "hidden_dim": 2048,
    },
    "pythia-1.4b": {
        "hf_path": "EleutherAI/pythia-1.4b",
        "N": 1_414_647_808,  # ~1.4B params
        "num_layers": 24,
        "hidden_dim": 2048,
    },
}


def load_model_at_step(model_key: str, step: int, device: str = "cuda"):
    """Load Pythia model at a specific training step."""
    from transformers import AutoModel, AutoTokenizer

    info = PYTHIA_MODELS[model_key]
    revision = f"step{step}" if step > 0 else "step0"

    tokenizer = AutoTokenizer.from_pretrained(
        info["hf_path"], revision=revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        info["hf_path"],
        revision=revision,
        torch_dtype=torch.float16,
    )
    model = model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def extract_layer_reps(model, tokenizer, texts, device="cuda",
                       batch_size=32, max_seq_len=256):
    """Extract L2-normalized representations from all layers."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = texts[i * batch_size:(i + 1) * batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=max_seq_len, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, return_dict=True)

        if out.hidden_states is None:
            raise RuntimeError("No hidden states returned")

        mask = enc["attention_mask"]

        for li, hs in enumerate(out.hidden_states):
            hs_f = hs.float()
            # Last-token pooling for Pythia (decoder)
            seq_lens = mask.sum(dim=1) - 1
            pooled = hs_f[torch.arange(hs_f.size(0)), seq_lens]
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            if li not in all_hidden:
                all_hidden[li] = []
            all_hidden[li].append(pooled.cpu().numpy())

    return {li: np.concatenate(arrs, axis=0)
            for li, arrs in sorted(all_hidden.items())}


def knn_accuracy(embs, labels, k=20):
    """kNN accuracy with cosine similarity."""
    n = len(embs)
    if n < k + 1:
        return 0.0

    correct = 0
    chunk = 500
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = embs[start:end] @ embs.T
        for i in range(end - start):
            sims[i, start + i] = -float("inf")
        topk = np.argpartition(-sims, k, axis=1)[:, :k]
        for i in range(end - start):
            votes = labels[topk[i]]
            pred = np.bincount(votes).argmax()
            if pred == labels[start + i]:
                correct += 1

    return correct / n


def evaluate_checkpoint(model_key, step, datasets, device="cuda"):
    """Evaluate one model checkpoint across all layers and datasets."""
    info = PYTHIA_MODELS[model_key]
    T = step * TOKENS_PER_STEP
    C = 6 * info["N"] * T  # FLOPs

    print(f"\n  {model_key} step={step} (C={C:.2e} FLOPs)")

    model, tokenizer = load_model_at_step(model_key, step, device)

    result = {
        "model": model_key,
        "step": step,
        "tokens_seen": T,
        "C_flops": C,
        "N_params": info["N"],
        "num_layers": info["num_layers"],
        "hidden_dim": info["hidden_dim"],
        "datasets": {},
    }

    for ds_name, ds_data in datasets.items():
        texts = ds_data["texts"]
        l0_labels = ds_data["l0_labels"]
        l1_labels = ds_data["l1_labels"]

        # Extract hidden states at all layers
        layer_reps = extract_layer_reps(model, tokenizer, texts, device,
                                        batch_size=32 if info["N"] < 5e8 else 16)

        layer_results = {}
        L = info["num_layers"]
        for li in sorted(layer_reps.keys()):
            x = li / L  # normalized depth
            reps = layer_reps[li]

            knn_l0 = knn_accuracy(reps, l0_labels, k=20)
            knn_l1 = knn_accuracy(reps, l1_labels, k=20)

            layer_results[li] = {
                "layer": li,
                "x": round(x, 4),
                "knn_l0": round(knn_l0, 4),
                "knn_l1": round(knn_l1, 4),
            }

        # Find optimal layer
        best_l1 = max(layer_results.values(), key=lambda r: r["knn_l1"])
        best_layer = best_l1["layer"]

        result["datasets"][ds_name] = {
            "layers": layer_results,
            "best_layer": best_layer,
            "best_x": best_l1["x"],
            "best_knn_l1": best_l1["knn_l1"],
            "final_knn_l1": layer_results[max(layer_results.keys())]["knn_l1"],
            "n_samples": len(texts),
        }

        print(f"    {ds_name}: best_layer={best_layer}/{L} (x={best_l1['x']:.2f}) "
              f"best_knn_l1={best_l1['knn_l1']:.3f} "
              f"final_knn_l1={layer_results[max(layer_results.keys())]['knn_l1']:.3f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def load_eval_datasets(dataset_names, max_samples=2000):
    """Load and cache evaluation datasets."""
    datasets = {}
    for name in dataset_names:
        print(f"Loading {name}...")
        ds = load_hierarchical_dataset(name, split="test", max_samples=max_samples)
        texts = [s.text for s in ds.samples]
        l0 = np.array([s.level0_label for s in ds.samples])
        l1 = np.array([s.level1_label for s in ds.samples])
        datasets[name] = {"texts": texts, "l0_labels": l0, "l1_labels": l1}
        print(f"  {name}: {len(texts)} samples, {len(set(l0))} L0, {len(set(l1))} L1")
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                       help="Fast path: 160M + 1B, 2 datasets")
    parser.add_argument("--fit", action="store_true",
                       help="Fit path: 160M + 410M + 1B, 4 datasets")
    parser.add_argument("--holdout", action="store_true",
                       help="Holdout: 1.4B only")
    parser.add_argument("--models", type=str, default=None,
                       help="Comma-separated model keys")
    parser.add_argument("--datasets", type=str, default=None,
                       help="Comma-separated dataset names")
    parser.add_argument("--steps", type=str, default=None,
                       help="Comma-separated checkpoint steps")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine configuration
    if args.fast:
        model_keys = ["pythia-160m", "pythia-1b"]
        ds_names = ["agnews", "trec"]
        steps = FAST_CHECKPOINTS
    elif args.fit:
        model_keys = ["pythia-160m", "pythia-410m", "pythia-1b"]
        ds_names = ["clinc", "dbpedia_classes", "agnews", "trec"]
        steps = FAST_CHECKPOINTS
    elif args.holdout:
        model_keys = ["pythia-1.4b"]
        ds_names = ["clinc", "dbpedia_classes", "agnews", "trec"]
        steps = FAST_CHECKPOINTS
    else:
        model_keys = args.models.split(",") if args.models else ["pythia-160m"]
        ds_names = args.datasets.split(",") if args.datasets else ["agnews", "trec"]
        steps = [int(s) for s in args.steps.split(",")] if args.steps else FAST_CHECKPOINTS

    print("=" * 70)
    print("  CTI Phase 1: Checkpoint Quality Sweep")
    print("=" * 70)
    print(f"Models: {model_keys}")
    print(f"Datasets: {ds_names}")
    print(f"Checkpoints: {steps}")
    print(f"Total evaluations: {len(model_keys)} x {len(steps)} x {len(ds_names)} = "
          f"{len(model_keys) * len(steps) * len(ds_names)}")

    # Load datasets once
    datasets = load_eval_datasets(ds_names)

    # Run sweep
    all_results = []
    t_start = time.time()

    for model_key in model_keys:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_key}")
        print(f"{'='*70}")

        for step in steps:
            try:
                result = evaluate_checkpoint(model_key, step, datasets, device)
                all_results.append(result)

                # Incremental save
                out_path = RESULTS_DIR / "cti_checkpoint_sweep.json"
                with open(out_path, "w") as f:
                    json.dump({
                        "experiment": "CTI Phase 1: Checkpoint Quality Sweep",
                        "models": model_keys,
                        "datasets": ds_names,
                        "checkpoints": steps,
                        "results": all_results,
                        "elapsed_sec": time.time() - t_start,
                    }, f, indent=2,
                    default=lambda o: float(o) if hasattr(o, "item") else str(o))
            except Exception as e:
                print(f"  ERROR at {model_key} step={step}: {e}")
                all_results.append({
                    "model": model_key,
                    "step": step,
                    "error": str(e),
                })

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  COMPLETE: {len(all_results)} checkpoints in {elapsed:.0f}s")
    print(f"{'='*70}")

    # Final save
    out_path = RESULTS_DIR / "cti_checkpoint_sweep.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "CTI Phase 1: Checkpoint Quality Sweep",
            "models": model_keys,
            "datasets": ds_names,
            "checkpoints": steps,
            "results": all_results,
            "elapsed_sec": elapsed,
        }, f, indent=2,
        default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
