#!/usr/bin/env python
"""
cti_knn_sweep.py

Fast kNN-based layer-wise representation quality sweep for CTI.
Instead of training V5 heads (~8min/point), this extracts hidden states
at each layer and evaluates kNN accuracy on hierarchical labels (~15s/point).

Usage:
    python -u src/cti_knn_sweep.py --models bge-small,bge-base,bge-large
    python -u src/cti_knn_sweep.py --models pythia-160m,pythia-410m --k 5
    python -u src/cti_knn_sweep.py --all-encoder  # Run all encoder models
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset


# ── Model loading ─────────────────────────────────────────────────────

def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get model config from multi_model_pipeline or define inline."""
    try:
        from multi_model_pipeline import MODELS
        if model_key in MODELS:
            cfg = MODELS[model_key]
            return {
                "name": cfg.name,
                "hf_path": cfg.hf_path,
                "hidden_dim": cfg.hidden_dim,
                "pooling": cfg.pooling,
                "trust_remote_code": getattr(cfg, "trust_remote_code", False),
            }
    except ImportError:
        pass

    # Fallback for models not in pipeline
    EXTRA_MODELS = {
        "pythia-160m": {
            "name": "Pythia-160M", "hf_path": "EleutherAI/pythia-160m",
            "hidden_dim": 768, "pooling": "last", "trust_remote_code": False,
        },
        "pythia-2.8b": {
            "name": "Pythia-2.8B", "hf_path": "EleutherAI/pythia-2.8b",
            "hidden_dim": 2560, "pooling": "last", "trust_remote_code": False,
        },
        "mamba-130m": {
            "name": "Mamba-130M", "hf_path": "state-spaces/mamba-130m-hf",
            "hidden_dim": 768, "pooling": "last", "trust_remote_code": True,
        },
        "mamba-370m": {
            "name": "Mamba-370M", "hf_path": "state-spaces/mamba-370m-hf",
            "hidden_dim": 1024, "pooling": "last", "trust_remote_code": True,
        },
        "mamba-1.4b": {
            "name": "Mamba-1.4B", "hf_path": "state-spaces/mamba-1.4b-hf",
            "hidden_dim": 2048, "pooling": "last", "trust_remote_code": True,
        },
    }
    if model_key in EXTRA_MODELS:
        return EXTRA_MODELS[model_key]
    raise ValueError(f"Unknown model: {model_key}")


def load_backbone(model_key: str, device: str = "cuda"):
    """Load model backbone and tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    info = get_model_info(model_key)
    print(f"Loading {info['name']}...")

    tokenizer = AutoTokenizer.from_pretrained(
        info["hf_path"],
        trust_remote_code=info.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        info["hf_path"],
        trust_remote_code=info.get("trust_remote_code", False),
        dtype=torch.float16,
    )
    model = model.to(device).eval()

    num_layers = None
    cfg = getattr(model, "config", None)
    if cfg and hasattr(cfg, "num_hidden_layers"):
        num_layers = cfg.num_hidden_layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        num_layers = len(model.encoder.layer)
    elif hasattr(model, "layers"):
        num_layers = len(model.layers)

    print(f"  Hidden dim: {info['hidden_dim']}, Layers: {num_layers}, Pooling: {info['pooling']}")
    return model, tokenizer, info, num_layers


# ── Representation extraction ─────────────────────────────────────────

@torch.no_grad()
def extract_all_layer_representations(
    model,
    tokenizer,
    texts: List[str],
    pooling: str = "cls",
    batch_size: int = 64,
    device: str = "cuda",
    max_seq_len: int = 512,
) -> Dict[int, np.ndarray]:
    """Extract pooled representations from ALL hidden layers.

    Returns: {layer_idx: (N, D) numpy array} where layer 0 = embedding layer.
    """
    all_hidden = {}  # layer -> list of (B, D) arrays
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        ).to(device)

        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states  # tuple of (B, T, D) tensors

        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states")

        mask = enc["attention_mask"]  # (B, T)

        for layer_idx, hs in enumerate(hidden_states):
            hs_f = hs.float()

            if pooling == "cls":
                pooled = hs_f[:, 0, :]
            elif pooling == "last":
                # Last non-padding token
                seq_lens = mask.sum(dim=1) - 1  # (B,)
                pooled = hs_f[torch.arange(hs_f.size(0)), seq_lens]
            elif pooling == "mean":
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (hs_f * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = hs_f[:, 0, :]

            # L2 normalize
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            if layer_idx not in all_hidden:
                all_hidden[layer_idx] = []
            all_hidden[layer_idx].append(pooled.cpu().numpy())

    # Concatenate
    result = {}
    for layer_idx in sorted(all_hidden.keys()):
        result[layer_idx] = np.concatenate(all_hidden[layer_idx], axis=0)

    return result


# ── kNN evaluation ────────────────────────────────────────────────────

def knn_accuracy(embeddings: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Fast kNN accuracy using batch cosine similarity."""
    n = len(embeddings)
    if n == 0:
        return 0.0

    # Process in chunks to avoid OOM
    chunk_size = 500
    correct = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = embeddings[start:end]  # (C, D)
        sims = chunk @ embeddings.T  # (C, N)

        # Zero out self-similarity
        for i in range(end - start):
            sims[i, start + i] = -float('inf')

        top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]

        for i in range(end - start):
            neighbor_labels = labels[top_k_idx[i]]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            pred = unique[np.argmax(counts)]
            if pred == labels[start + i]:
                correct += 1

    return correct / n


# ── Main sweep ────────────────────────────────────────────────────────

def sweep_model(
    model_key: str,
    datasets: List[str],
    k: int = 5,
    max_samples: int = 2000,
    device: str = "cuda",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Run kNN sweep for one model across all datasets and all layers."""

    model, tokenizer, info, num_layers = load_backbone(model_key, device)
    results = {}

    for dataset_name in datasets:
        print(f"\n  Dataset: {dataset_name}")
        t0 = time.time()

        try:
            data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_samples)
        except Exception as e:
            print(f"    SKIP: {e}")
            results[dataset_name] = {"status": "error", "error": str(e)}
            continue

        texts = [s.text for s in data.samples]
        l0_labels = np.array([s.level0_label for s in data.samples])
        l1_labels = np.array([s.level1_label for s in data.samples])
        n_l0 = len(data.level0_names)
        n_l1 = len(data.level1_names)

        print(f"    Samples: {len(texts)}, L0 classes: {n_l0}, L1 classes: {n_l1}")

        # Extract all layer representations
        layer_reps = extract_all_layer_representations(
            model, tokenizer, texts,
            pooling=info["pooling"],
            batch_size=batch_size,
            device=device,
        )

        # Evaluate kNN at each layer
        layer_results = {}
        best_l1 = 0.0
        best_l0 = 0.0

        for layer_idx in sorted(layer_reps.keys()):
            emb = layer_reps[layer_idx]
            acc_l0 = knn_accuracy(emb, l0_labels, k=k)
            acc_l1 = knn_accuracy(emb, l1_labels, k=k)
            best_l0 = max(best_l0, acc_l0)
            best_l1 = max(best_l1, acc_l1)

            C_rel = layer_idx / num_layers if num_layers else layer_idx
            layer_results[layer_idx] = {
                "layer": layer_idx,
                "C_relative": float(C_rel),
                "C_absolute": float(layer_idx),
                "knn_l0": float(acc_l0),
                "knn_l1": float(acc_l1),
            }

            print(f"    L{layer_idx:>2}: L0={acc_l0:.4f}, L1={acc_l1:.4f}, C={C_rel:.3f}")

        # Compute distortion for each layer
        for layer_idx in layer_results:
            lr = layer_results[layer_idx]
            lr["D_l1"] = 1.0 - lr["knn_l1"] / max(best_l1, 1e-8)
            lr["D_l0"] = 1.0 - lr["knn_l0"] / max(best_l0, 1e-8)

        elapsed = time.time() - t0
        results[dataset_name] = {
            "status": "ok",
            "n_samples": len(texts),
            "n_l0": n_l0,
            "n_l1": n_l1,
            "num_layers_evaluated": len(layer_results),
            "best_l0": float(best_l0),
            "best_l1": float(best_l1),
            "layers": layer_results,
            "runtime_sec": elapsed,
        }
        print(f"    Done in {elapsed:.1f}s (best L0={best_l0:.4f}, L1={best_l1:.4f})")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "model": model_key,
        "model_name": info["name"],
        "hidden_dim": info["hidden_dim"],
        "num_layers": num_layers,
        "pooling": info["pooling"],
        "k": k,
        "datasets": results,
    }


def main():
    parser = argparse.ArgumentParser(description="CTI kNN layer-wise sweep")
    parser.add_argument("--models", type=str, default="bge-small,bge-base",
                        help="Comma-separated model keys")
    parser.add_argument("--datasets", type=str,
                        default="clinc,dbpedia_classes,trec,yahoo,20newsgroups",
                        help="Comma-separated dataset names")
    parser.add_argument("--k", type=int, default=5, help="kNN k value")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    output_path = Path(args.output) if args.output else RESULTS_DIR / "cti_knn_sweep.json"

    # Load existing results for resume
    all_results = {}
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)

    print(f"CTI kNN Sweep")
    print(f"=============")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"k={args.k}, max_samples={args.max_samples}")
    print()

    for model_key in models:
        if model_key in all_results and all_results[model_key].get("status") == "complete":
            print(f"\nSKIP {model_key} (already complete)")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        try:
            result = sweep_model(
                model_key=model_key,
                datasets=datasets,
                k=args.k,
                max_samples=args.max_samples,
                device=args.device,
                batch_size=args.batch_size,
            )
            result["status"] = "complete"
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            all_results[model_key] = result

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {"status": "error", "error": str(e)}

        # Save after each model (resume-safe)
        def convert(obj):
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(convert(all_results), f, indent=2)
        print(f"\nSaved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    for model_key in models:
        if model_key not in all_results:
            continue
        r = all_results[model_key]
        if r.get("status") != "complete":
            print(f"  {model_key}: {r.get('status', 'unknown')}")
            continue
        for ds_name, ds in r.get("datasets", {}).items():
            if ds.get("status") != "ok":
                continue
            layers = ds.get("layers", {})
            if not layers:
                continue
            # Find layer with best L1
            best_layer = max(layers.values(), key=lambda x: x.get("knn_l1", 0))
            print(f"  {model_key}|{ds_name}: best L1={best_layer['knn_l1']:.4f} at L{best_layer['layer']}")


if __name__ == "__main__":
    main()
