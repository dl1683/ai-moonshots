#!/usr/bin/env python
"""
CTI New Family Extraction: add 3 architecturally distinct decoder families.

Adds:
  - Gemma-2-2B (Google): GQA, RMSNorm, logit soft-capping
  - Phi-2 (Microsoft, 2.7B): synthetic data training, partial attention
  - Qwen2.5-1.5B (Alibaba): GQA, SwiGLU, multilingual

Each model evaluated on all 4 CTI datasets at all layers.
Results saved in same format as cti_multi_family.json for easy merging.

Usage:
    python -u src/cti_new_families.py
    python -u src/cti_new_families.py --models gemma-2-2b,phi-2
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset

# New model families - architecturally distinct from existing 5
MODELS = {
    # Gemma-2 (Google): GQA, RMSNorm, logit soft-capping, sliding window
    "gemma-2-2b": {
        "hf_path": "google/gemma-2-2b",
        "family": "gemma2",
        "tokens_trained": 2_000_000_000_000,  # 2T tokens
        "num_layers": 26,
        "hidden_dim": 2304,
        "trust_remote_code": False,
    },
    # Phi-2 (Microsoft): trained largely on synthetic data, different data recipe
    "phi-2": {
        "hf_path": "microsoft/phi-2",
        "family": "phi",
        "tokens_trained": 1_400_000_000_000,  # 1.4T tokens
        "num_layers": 32,
        "hidden_dim": 2560,
        "trust_remote_code": True,
    },
    # Qwen2.5 (Alibaba): multilingual training, GQA, SwiGLU
    "qwen2.5-0.5b": {
        "hf_path": "Qwen/Qwen2.5-0.5B",
        "family": "qwen2.5",
        "tokens_trained": 18_000_000_000_000,  # 18T tokens
        "num_layers": 24,
        "hidden_dim": 896,
        "trust_remote_code": True,
    },
    "qwen2.5-1.5b": {
        "hf_path": "Qwen/Qwen2.5-1.5B",
        "family": "qwen2.5",
        "tokens_trained": 18_000_000_000_000,  # 18T tokens
        "num_layers": 28,
        "hidden_dim": 1536,
        "trust_remote_code": True,
    },
}

DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


@torch.no_grad()
def extract_layer_reps(model, tokenizer, texts, device="cuda",
                       batch_size=16, max_seq_len=256):
    """Extract L2-normalized last-token reps from all layers."""
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
            # Pool at last non-padding token
            seq_lens = mask.sum(dim=1) - 1
            pooled = hs_f[torch.arange(hs_f.size(0)), seq_lens]
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            if li not in all_hidden:
                all_hidden[li] = []
            all_hidden[li].append(pooled.cpu().numpy())

    return {li: np.concatenate(arrs, axis=0)
            for li, arrs in sorted(all_hidden.items())}


def knn_accuracy(embs, labels, k=20):
    """kNN accuracy with chunked computation."""
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


def load_eval_datasets(dataset_names, max_samples=2000):
    """Load datasets for evaluation."""
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
    parser.add_argument("--models", type=str, default=None,
                       help="Comma-separated model keys to run (default: all)")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds_names = ["clinc", "dbpedia_classes", "agnews", "trec"]
    datasets = load_eval_datasets(ds_names)

    # Select models
    if args.models:
        model_keys = [k.strip() for k in args.models.split(",")]
    else:
        model_keys = list(MODELS.keys())

    results = []
    out_path = RESULTS_DIR / "cti_new_families.json"

    # Load existing results if resuming
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        results = existing.get("results", [])
        done_models = {r["model"] for r in results if "error" not in r}
        model_keys = [k for k in model_keys if k not in done_models]
        print(f"Resuming: {len(done_models)} done, {len(model_keys)} remaining")

    print("=" * 70)
    print("  CTI New Family Depth Profile Extraction")
    print("=" * 70)

    for model_key in model_keys:
        info = MODELS[model_key]
        print(f"\n--- {model_key} ({info['family']}) ---")
        t0 = time.time()

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                info["hf_path"],
                trust_remote_code=info.get("trust_remote_code", False),
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModel.from_pretrained(
                info["hf_path"],
                torch_dtype=torch.float16,
                trust_remote_code=info.get("trust_remote_code", False),
            )
            model = model.to(device).eval()
            n_params = sum(p.numel() for p in model.parameters())
            L = info["num_layers"]
            C = 6 * n_params * info["tokens_trained"]

            print(f"  Params: {n_params:,}, Layers: {L}, C_flops: {C:.2e}")

            result = {
                "model": model_key,
                "family": info["family"],
                "step": -1,  # final checkpoint
                "tokens_seen": info["tokens_trained"],
                "C_flops": C,
                "N_params": n_params,
                "num_layers": L,
                "hidden_dim": info["hidden_dim"],
                "datasets": {},
            }

            # Adjust batch size based on model size
            if n_params < 5e8:
                bs = 32
            elif n_params < 1e9:
                bs = 16
            elif n_params < 2e9:
                bs = 8
            else:
                bs = 4

            for ds_name, ds_data in datasets.items():
                texts = ds_data["texts"]
                l0_labels = ds_data["l0_labels"]
                l1_labels = ds_data["l1_labels"]

                layer_reps = extract_layer_reps(model, tokenizer, texts, device,
                                                batch_size=bs)

                layer_results = {}
                for li in sorted(layer_reps.keys()):
                    x = li / L
                    reps = layer_reps[li]
                    knn_l0 = knn_accuracy(reps, l0_labels, k=20)
                    knn_l1 = knn_accuracy(reps, l1_labels, k=20)
                    layer_results[li] = {
                        "layer": li,
                        "x": round(x, 4),
                        "knn_l0": round(knn_l0, 4),
                        "knn_l1": round(knn_l1, 4),
                    }

                best_l1 = max(layer_results.values(), key=lambda r: r["knn_l1"])
                result["datasets"][ds_name] = {
                    "layers": layer_results,
                    "best_layer": best_l1["layer"],
                    "best_x": best_l1["x"],
                    "best_knn_l1": best_l1["knn_l1"],
                    "final_knn_l1": layer_results[max(layer_results.keys())]["knn_l1"],
                    "n_samples": len(texts),
                }

                print(f"    {ds_name}: best=L{best_l1['layer']}/{L} "
                      f"(knn_l1={best_l1['knn_l1']:.3f}, "
                      f"final={layer_results[max(layer_results.keys())]['knn_l1']:.3f})")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results.append(result)
            print(f"  Done in {time.time()-t0:.0f}s")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"model": model_key, "family": info["family"], "error": str(e)})

        # Save incrementally
        out = {
            "experiment": "CTI New Family Depth Profiles",
            "families": sorted(set(r.get("family", "?") for r in results)),
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2,
                      default=lambda x: int(x) if isinstance(x, (np.integer,))
                      else float(x) if isinstance(x, (np.floating,)) else x)

    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
