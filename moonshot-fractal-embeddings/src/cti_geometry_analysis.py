#!/usr/bin/env python
"""
cti_geometry_analysis.py

Mechanistic geometry analysis across transformer depth.
Measures per-layer:
  - Anisotropy (average cosine similarity between random pairs)
  - Effective rank (from singular value spectrum)
  - Intrinsic dimensionality (MLE estimator)
  - Spectral concentration (fraction of variance in top-k SVs)

Tests if geometric property changes align with quality phase transitions.

Usage:
    python -u src/cti_geometry_analysis.py --models bge-small,bge-large --datasets clinc
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset


def compute_anisotropy(reps: np.ndarray, n_pairs: int = 1000) -> float:
    """Average cosine similarity between random pairs of representations."""
    n = len(reps)
    if n < 2:
        return 0.0

    # Normalize
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    reps_norm = reps / norms

    # Random pairs
    idx1 = np.random.randint(0, n, n_pairs)
    idx2 = np.random.randint(0, n, n_pairs)
    # Avoid self-pairs
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    cos_sims = np.sum(reps_norm[idx1] * reps_norm[idx2], axis=1)
    return float(np.mean(cos_sims))


def compute_effective_rank(reps: np.ndarray) -> float:
    """Effective rank from singular value spectrum.
    eff_rank = exp(H(p)) where p_i = sigma_i / sum(sigma), H = Shannon entropy.
    """
    try:
        _, s, _ = np.linalg.svd(reps, full_matrices=False)
        s = s[s > 1e-10]
        if len(s) == 0:
            return 0.0
        p = s / s.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        return float(np.exp(entropy))
    except Exception:
        return 0.0


def compute_spectral_concentration(reps: np.ndarray, top_k: int = 10) -> float:
    """Fraction of total variance explained by top-k singular values."""
    try:
        _, s, _ = np.linalg.svd(reps, full_matrices=False)
        total = np.sum(s ** 2)
        top = np.sum(s[:top_k] ** 2)
        return float(top / max(total, 1e-12))
    except Exception:
        return 0.0


def compute_intrinsic_dim_mle(reps: np.ndarray, k: int = 5) -> float:
    """MLE intrinsic dimensionality estimator (Levina-Bickel)."""
    from scipy.spatial.distance import cdist

    n = min(len(reps), 500)  # Subsample for speed
    if n < k + 1:
        return 0.0

    idx = np.random.choice(len(reps), n, replace=False)
    sub = reps[idx]

    dists = cdist(sub, sub, metric="euclidean")
    np.fill_diagonal(dists, np.inf)

    dims = []
    for i in range(n):
        sorted_d = np.sort(dists[i])
        knn_dists = sorted_d[:k]
        if knn_dists[-1] > 0:
            # MLE estimator
            log_ratios = np.log(knn_dists[-1] / np.clip(knn_dists[:-1], 1e-10, None))
            if np.sum(log_ratios) > 0:
                dim_est = (k - 1) / np.sum(log_ratios)
                dims.append(dim_est)

    return float(np.mean(dims)) if dims else 0.0


def extract_and_analyze(model_key: str, dataset_name: str,
                        max_samples: int = 1000, device: str = "cuda") -> Dict:
    """Extract representations at all layers and compute geometric properties."""
    from cti_knn_sweep import load_backbone

    # Load data
    data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_samples)
    texts = [s.text for s in data.samples]

    # Load model
    model, tokenizer, info, num_layers = load_backbone(model_key, device)
    pooling = info.get("pooling", "cls")

    print(f"  Model: {model_key}, Layers: {num_layers}, Samples: {len(texts)}")

    # Extract representations at all layers
    all_layer_reps = {i: [] for i in range(num_layers + 1)}
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                          max_length=128, return_tensors="pt").to(device)

            outputs = model(**enc, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states

            attn_mask = enc["attention_mask"]

            for layer_idx, hs in enumerate(hidden_states):
                if pooling == "cls":
                    pooled = hs[:, 0, :]
                elif pooling == "last":
                    # Last non-padding token
                    seq_lens = attn_mask.sum(dim=1) - 1
                    pooled = hs[torch.arange(hs.size(0)), seq_lens]
                else:  # mean
                    mask_expanded = attn_mask.unsqueeze(-1).float()
                    pooled = (hs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

                all_layer_reps[layer_idx].append(pooled.cpu().numpy())

    # Concatenate
    for layer_idx in all_layer_reps:
        all_layer_reps[layer_idx] = np.concatenate(all_layer_reps[layer_idx], axis=0)

    # Analyze geometry per layer
    results = {}
    for layer_idx in sorted(all_layer_reps.keys()):
        reps = all_layer_reps[layer_idx].astype(np.float32)
        s_relative = layer_idx / num_layers

        aniso = compute_anisotropy(reps)
        eff_rank = compute_effective_rank(reps)
        spec_conc = compute_spectral_concentration(reps, top_k=10)
        intr_dim = compute_intrinsic_dim_mle(reps, k=5)

        results[layer_idx] = {
            "layer": layer_idx,
            "s_relative": float(s_relative),
            "anisotropy": aniso,
            "effective_rank": eff_rank,
            "spectral_concentration_top10": spec_conc,
            "intrinsic_dim": intr_dim,
        }

        print(f"    L{layer_idx:2d} (s={s_relative:.3f}): aniso={aniso:.4f}  eff_rank={eff_rank:.1f}  "
              f"spec_conc={spec_conc:.4f}  intr_dim={intr_dim:.1f}")

    # Cleanup
    del model, all_layer_reps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_key,
        "dataset": dataset_name,
        "num_layers": num_layers,
        "n_samples": len(texts),
        "layers": results,
    }


def correlate_geometry_with_quality(geometry: Dict, atlas_path: str) -> Dict:
    """Correlate geometric properties with kNN quality from atlas."""
    with open(atlas_path) as f:
        atlas = json.load(f)

    correlations = {}

    for model_key, model_geom in geometry.items():
        for ds_name, ds_geom in model_geom.items():
            # Get quality curve from atlas
            if model_key not in atlas or "datasets" not in atlas[model_key]:
                continue
            if ds_name not in atlas[model_key]["datasets"]:
                continue

            atlas_layers = atlas[model_key]["datasets"][ds_name].get("layers", {})
            if not atlas_layers:
                continue

            # Match layers
            layers = sorted(ds_geom["layers"].keys())
            quality = []
            anisotropy = []
            eff_rank = []
            spec_conc = []
            intr_dim = []

            for layer_idx in layers:
                layer_str = str(layer_idx)
                if layer_str in atlas_layers:
                    quality.append(atlas_layers[layer_str]["knn_l1"])
                    anisotropy.append(ds_geom["layers"][layer_idx]["anisotropy"])
                    eff_rank.append(ds_geom["layers"][layer_idx]["effective_rank"])
                    spec_conc.append(ds_geom["layers"][layer_idx]["spectral_concentration_top10"])
                    intr_dim.append(ds_geom["layers"][layer_idx]["intrinsic_dim"])

            if len(quality) < 5:
                continue

            quality = np.array(quality)
            key = f"{model_key}|{ds_name}"
            correlations[key] = {}

            for name, values in [("anisotropy", anisotropy), ("effective_rank", eff_rank),
                                 ("spectral_concentration", spec_conc), ("intrinsic_dim", intr_dim)]:
                values = np.array(values)
                rho, p = stats.spearmanr(quality, values)
                correlations[key][name] = {"rho": float(rho), "p": float(p)}

    return correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="bge-small,bge-large,pythia-410m")
    parser.add_argument("--datasets", default="clinc,dbpedia_classes")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output", default=str(RESULTS_DIR / "cti_geometry_analysis.json"))
    parser.add_argument("--atlas", default=str(RESULTS_DIR / "cti_atlas_fit.json"))
    args = parser.parse_args()

    models = args.models.split(",")
    datasets = args.datasets.split(",")

    print("=" * 70)
    print("CTI GEOMETRY ANALYSIS")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")

    all_results = {}

    for model_key in models:
        all_results[model_key] = {}
        for ds_name in datasets:
            print(f"\n--- {model_key} x {ds_name} ---")
            try:
                result = extract_and_analyze(model_key, ds_name, args.max_samples)
                all_results[model_key][ds_name] = result
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[model_key][ds_name] = {"error": str(e)}

    # Correlate with quality
    print("\n" + "=" * 70)
    print("CORRELATION: Geometry vs Quality")
    print("=" * 70)

    correlations = correlate_geometry_with_quality(all_results, args.atlas)
    for key, corrs in correlations.items():
        print(f"\n  {key}:")
        for metric, vals in corrs.items():
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else "ns"
            print(f"    {metric:25s}: rho={vals['rho']:+.3f}  p={vals['p']:.4f} {sig}")

    # Save
    output = {
        "geometry": {
            m: {
                d: {
                    "num_layers": v.get("num_layers"),
                    "layers": {
                        str(k): vals for k, vals in v.get("layers", {}).items()
                    }
                } if "error" not in v else {"error": v["error"]}
                for d, v in ds.items()
            }
            for m, ds in all_results.items()
        },
        "quality_correlations": correlations,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
