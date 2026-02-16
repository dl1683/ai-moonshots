#!/usr/bin/env python
"""
CTI Multi-Family Validation: evaluate depth profiles at final checkpoint
for multiple model families and sizes.

Since most families lack training checkpoints, we test whether our depth
law (fitted on Pythia checkpoints) predicts quality profiles for other
architectures at their final training state.

Families:
  - Cerebras-GPT: 256M, 590M, 1.3B (GPT-3 style, Chinchilla-optimal)
  - OPT: 350M, 1.3B (Meta, decoder-only)
  - GPT-2: 124M, 355M, 774M (OpenAI, original transformer decoder)

Each model evaluated on all 4 datasets at all layers.
"""

from __future__ import annotations

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

# Model specs: family, hf_path, approx N_params, num_layers, hidden_dim, tokens_trained
MODELS = {
    # Cerebras-GPT: Chinchilla-optimal (20 tokens/param)
    "cerebras-gpt-256m": {
        "hf_path": "cerebras/Cerebras-GPT-256M",
        "family": "cerebras-gpt",
        "tokens_trained": 5_100_000_000,  # 256M * 20
        "num_layers": 14,
        "hidden_dim": 1088,
    },
    "cerebras-gpt-590m": {
        "hf_path": "cerebras/Cerebras-GPT-590M",
        "family": "cerebras-gpt",
        "tokens_trained": 11_800_000_000,  # 590M * 20
        "num_layers": 18,
        "hidden_dim": 1536,
    },
    "cerebras-gpt-1.3b": {
        "hf_path": "cerebras/Cerebras-GPT-1.3B",
        "family": "cerebras-gpt",
        "tokens_trained": 26_000_000_000,  # 1.3B * 20
        "num_layers": 24,
        "hidden_dim": 2048,
    },
    # OPT (Meta)
    "opt-350m": {
        "hf_path": "facebook/opt-350m",
        "family": "opt",
        "tokens_trained": 300_000_000_000,  # 300B tokens
        "num_layers": 24,
        "hidden_dim": 512,
    },
    "opt-1.3b": {
        "hf_path": "facebook/opt-1.3b",
        "family": "opt",
        "tokens_trained": 300_000_000_000,  # 300B tokens
        "num_layers": 24,
        "hidden_dim": 2048,
    },
    # GPT-2 (OpenAI)
    "gpt2-small": {
        "hf_path": "gpt2",
        "family": "gpt2",
        "tokens_trained": 40_000_000_000,  # ~40B tokens (WebText)
        "num_layers": 12,
        "hidden_dim": 768,
    },
    "gpt2-medium": {
        "hf_path": "gpt2-medium",
        "family": "gpt2",
        "tokens_trained": 40_000_000_000,
        "num_layers": 24,
        "hidden_dim": 1024,
    },
    "gpt2-large": {
        "hf_path": "gpt2-large",
        "family": "gpt2",
        "tokens_trained": 40_000_000_000,
        "num_layers": 36,
        "hidden_dim": 1280,
    },
}

DS_CLASSES = {"clinc": 150, "dbpedia_classes": 68, "agnews": 18, "trec": 42}


@torch.no_grad()
def extract_layer_reps(model, tokenizer, texts, device="cuda",
                       batch_size=16, max_seq_len=256):
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
            seq_lens = mask.sum(dim=1) - 1
            pooled = hs_f[torch.arange(hs_f.size(0)), seq_lens]
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            if li not in all_hidden:
                all_hidden[li] = []
            all_hidden[li].append(pooled.cpu().numpy())

    return {li: np.concatenate(arrs, axis=0)
            for li, arrs in sorted(all_hidden.items())}


def knn_accuracy(embs, labels, k=20):
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
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_names = ["clinc", "dbpedia_classes", "agnews", "trec"]
    datasets = load_eval_datasets(ds_names)

    results = []
    out_path = RESULTS_DIR / "cti_multi_family.json"

    print("=" * 70)
    print("  CTI Multi-Family Depth Profile Evaluation")
    print("=" * 70)

    for model_key, info in MODELS.items():
        print(f"\n--- {model_key} ({info['family']}) ---")
        t0 = time.time()

        try:
            tokenizer = AutoTokenizer.from_pretrained(info["hf_path"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModel.from_pretrained(
                info["hf_path"],
                torch_dtype=torch.float16,
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

            bs = 32 if n_params < 5e8 else (16 if n_params < 1e9 else 8)

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
                      f"(knn_l1={best_l1['knn_l1']:.3f}, final={layer_results[max(layer_results.keys())]['knn_l1']:.3f})")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results.append(result)
            print(f"  Done in {time.time()-t0:.0f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"model": model_key, "family": info["family"], "error": str(e)})

        # Save incrementally
        out = {
            "experiment": "CTI Multi-Family Depth Profiles",
            "families": sorted(set(r.get("family", "?") for r in results)),
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer,)) else float(x) if isinstance(x, (np.floating,)) else x)

    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
