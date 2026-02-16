#!/usr/bin/env python
"""
cgp_alignment_uniformity.py

CGP Week 1 completion: Measure alignment AND uniformity across objectives.

Wang & Isola (2020) showed that good representations need BOTH:
  - Alignment: same-class samples should be close (low alignment loss)
  - Uniformity: representations should be spread on the hypersphere (low uniformity loss)

Our previous CGP experiment showed contrastive training improved uniformity (isotropy)
but didn't improve kNN quality. Hypothesis: contrastive improved uniformity but hurt
alignment, explaining the quality plateau.

This script:
  1. Re-trains Pythia-160M with each objective (contrastive, mlm, lm) + baseline
  2. At each layer, computes alignment, uniformity, and kNN quality
  3. Tests if alignment + uniformity together predict quality better than either alone
  4. Builds the structural equation: Objective -> (Alignment, Uniformity) -> Quality

Usage:
    python -u src/cgp_alignment_uniformity.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset

# Reuse training functions from the CGP pipeline
from cgp_controlled_finetune import (
    MODEL_KEY, HF_PATH, HIDDEN_DIM, NUM_LAYERS,
    LORA_R, LORA_ALPHA, BATCH_SIZE, MAX_SEQ_LEN, NUM_STEPS, LR,
    DATASET_NAME, EVAL_DATASETS,
    apply_lora, TextDataset,
    train_contrastive, train_lm, train_mlm,
    pool_last_token,
)


# ── Alignment & Uniformity Metrics ──────────────────────────────────

def compute_alignment(reps: np.ndarray, labels: np.ndarray, alpha: float = 2.0) -> float:
    """
    Alignment loss (Wang & Isola 2020):
    L_align = E_{(x,y)~p_pos} [||f(x) - f(y)||^alpha]

    Where p_pos = positive pairs (same class).
    Lower = better (same-class items are closer).
    Uses L2-normalized representations.
    """
    # Normalize
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    reps_norm = reps / norms

    # Subsample if too many to avoid O(n^2) blow-up
    n = len(reps_norm)
    if n > 2000:
        idx = np.random.choice(n, 2000, replace=False)
        reps_norm = reps_norm[idx]
        labels = labels[idx]
        n = 2000

    # Find positive pairs (same class)
    total_dist = 0.0
    n_pairs = 0

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        class_reps = reps_norm[mask]
        nc = len(class_reps)
        if nc < 2:
            continue

        # Sample pairs within class (limit to avoid O(nc^2))
        max_pairs_per_class = min(500, nc * (nc - 1) // 2)
        if nc <= 32:
            # Compute all pairs
            for i in range(nc):
                for j in range(i + 1, nc):
                    diff = class_reps[i] - class_reps[j]
                    total_dist += np.sum(diff ** 2) ** (alpha / 2)
                    n_pairs += 1
        else:
            # Random sample pairs
            for _ in range(max_pairs_per_class):
                i, j = np.random.choice(nc, 2, replace=False)
                diff = class_reps[i] - class_reps[j]
                total_dist += np.sum(diff ** 2) ** (alpha / 2)
                n_pairs += 1

    if n_pairs == 0:
        return float('nan')

    return total_dist / n_pairs


def compute_uniformity(reps: np.ndarray, t: float = 2.0) -> float:
    """
    Uniformity loss (Wang & Isola 2020):
    L_uniform = log E_{(x,y)~p_data} [e^{-t * ||f(x) - f(y)||^2}]

    Lower (more negative) = better (more uniform on hypersphere).
    Uses L2-normalized representations.
    """
    # Normalize
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    reps_norm = reps / norms

    # Subsample if needed
    n = len(reps_norm)
    if n > 1000:
        idx = np.random.choice(n, 1000, replace=False)
        reps_norm = reps_norm[idx]
        n = 1000

    # Compute pairwise squared distances efficiently
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y = 2 - 2*x.y (for unit vectors)
    sim_matrix = reps_norm @ reps_norm.T
    sq_dists = 2.0 - 2.0 * sim_matrix

    # Only use upper triangle (exclude self-pairs)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    sq_dists_pairs = sq_dists[mask]

    # Compute uniformity
    exp_vals = np.exp(-t * sq_dists_pairs)
    uniformity = np.log(np.mean(exp_vals) + 1e-10)

    return float(uniformity)


def compute_class_separation(reps: np.ndarray, labels: np.ndarray) -> float:
    """
    Ratio of inter-class to intra-class distance.
    Higher = better class separation.
    """
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    reps_norm = reps / norms

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float('nan')

    # Compute class centroids
    centroids = []
    for lbl in unique_labels:
        mask = labels == lbl
        centroid = reps_norm[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Inter-class distance: mean pairwise distance between centroids
    nc = len(centroids)
    inter_dists = []
    for i in range(nc):
        for j in range(i + 1, nc):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    inter_dist = np.mean(inter_dists) if inter_dists else 0.0

    # Intra-class distance: mean distance to centroid
    intra_dists = []
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        class_reps = reps_norm[mask]
        dists = np.linalg.norm(class_reps - centroids[idx], axis=1)
        intra_dists.extend(dists.tolist())
    intra_dist = np.mean(intra_dists) if intra_dists else 1e-8

    return float(inter_dist / max(intra_dist, 1e-8))


# ── Full Measurement Pipeline ───────────────────────────────────────

def measure_full_layer_metrics(model, tokenizer, dataset_name, device,
                                max_samples=800) -> Dict:
    """
    At each layer, measure:
    - Alignment (same-class L0 and L1 distance)
    - Uniformity
    - Class separation ratio
    - kNN accuracy (L0, L1)
    - Geometry (anisotropy, effective rank)
    """
    from cti_geometry_analysis import (
        compute_anisotropy,
        compute_effective_rank,
    )
    from cti_knn_sweep import knn_accuracy

    data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_samples)
    texts = [s.text for s in data.samples]
    l0_labels = np.array([s.level0_label for s in data.samples])
    l1_labels = np.array([s.level1_label for s in data.samples])

    # Extract representations at all layers
    model.eval()
    all_layer_reps = {}
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)

            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

            attn_mask = enc["attention_mask"]
            for layer_idx, hs in enumerate(outputs.hidden_states):
                seq_lens = attn_mask.sum(dim=1) - 1
                pooled = hs[torch.arange(hs.size(0), device=device), seq_lens]
                if layer_idx not in all_layer_reps:
                    all_layer_reps[layer_idx] = []
                all_layer_reps[layer_idx].append(pooled.cpu().float().numpy())

    results = {}
    for layer_idx in sorted(all_layer_reps.keys()):
        reps = np.concatenate(all_layer_reps[layer_idx], axis=0).astype(np.float32)

        # Alignment (same-class distance)
        align_l0 = compute_alignment(reps, l0_labels)
        align_l1 = compute_alignment(reps, l1_labels)

        # Uniformity
        uniformity = compute_uniformity(reps)

        # Class separation
        sep_l0 = compute_class_separation(reps, l0_labels)
        sep_l1 = compute_class_separation(reps, l1_labels)

        # kNN accuracy
        knn_l0 = knn_accuracy(reps, l0_labels, k=5)
        knn_l1 = knn_accuracy(reps, l1_labels, k=5)

        # Geometry
        anisotropy = compute_anisotropy(reps)
        eff_rank = compute_effective_rank(reps)

        results[layer_idx] = {
            "layer": layer_idx,
            "s_relative": layer_idx / NUM_LAYERS,
            "alignment_l0": float(align_l0),
            "alignment_l1": float(align_l1),
            "uniformity": float(uniformity),
            "class_sep_l0": float(sep_l0),
            "class_sep_l1": float(sep_l1),
            "knn_l0": float(knn_l0),
            "knn_l1": float(knn_l1),
            "anisotropy": float(anisotropy),
            "effective_rank": float(eff_rank),
        }

    return results


# ── Structural Model Analysis ───────────────────────────────────────

def analyze_structural_model(all_results: Dict) -> Dict:
    """
    Test: Does Quality = f(Alignment, Uniformity)?

    Fit multivariate regression:
      knn_l0 ~ alignment_l0 + uniformity
      knn_l1 ~ alignment_l1 + uniformity

    Compare R^2 of:
      - Uniformity alone
      - Alignment alone
      - Both together
      - Anisotropy alone (baseline geometric predictor)
    """
    from sklearn.linear_model import LinearRegression

    # Collect all (layer, objective) data points
    rows = []
    for obj_name, obj_data in all_results.items():
        for ds_name, ds_data in obj_data.get("metrics", {}).items():
            for layer_key, layer_data in ds_data.items():
                row = {
                    "objective": obj_name,
                    "dataset": ds_name,
                    "layer": layer_data["layer"],
                    **{k: v for k, v in layer_data.items()
                       if k not in ("layer", "s_relative")}
                }
                rows.append(row)

    if not rows:
        return {"error": "no data"}

    # Convert to arrays
    align_l0 = np.array([r["alignment_l0"] for r in rows])
    align_l1 = np.array([r["alignment_l1"] for r in rows])
    uniformity = np.array([r["uniformity"] for r in rows])
    anisotropy = np.array([r["anisotropy"] for r in rows])
    eff_rank = np.array([r["effective_rank"] for r in rows])
    knn_l0 = np.array([r["knn_l0"] for r in rows])
    knn_l1 = np.array([r["knn_l1"] for r in rows])
    sep_l0 = np.array([r["class_sep_l0"] for r in rows])
    sep_l1 = np.array([r["class_sep_l1"] for r in rows])

    # Filter out NaN
    valid = (
        np.isfinite(align_l0) & np.isfinite(align_l1) &
        np.isfinite(uniformity) & np.isfinite(anisotropy) &
        np.isfinite(knn_l0) & np.isfinite(knn_l1)
    )

    results = {}

    for target_name, target, align_arr, sep_arr in [
        ("knn_l0", knn_l0, align_l0, sep_l0),
        ("knn_l1", knn_l1, align_l1, sep_l1),
    ]:
        v = valid
        y = target[v]
        n = len(y)

        if n < 10:
            results[target_name] = {"error": f"too few valid points ({n})"}
            continue

        # 1. Alignment alone
        X_align = -align_arr[v].reshape(-1, 1)  # Negate: lower alignment loss = better
        reg_a = LinearRegression().fit(X_align, y)
        r2_align = reg_a.score(X_align, y)

        # 2. Uniformity alone
        X_unif = -uniformity[v].reshape(-1, 1)  # Negate: lower = better
        reg_u = LinearRegression().fit(X_unif, y)
        r2_unif = reg_u.score(X_unif, y)

        # 3. Both together
        X_both = np.column_stack([-align_arr[v], -uniformity[v]])
        reg_both = LinearRegression().fit(X_both, y)
        r2_both = reg_both.score(X_both, y)

        # 4. Anisotropy alone (previous baseline)
        X_aniso = anisotropy[v].reshape(-1, 1)
        reg_aniso = LinearRegression().fit(X_aniso, y)
        r2_aniso = reg_aniso.score(X_aniso, y)

        # 5. Class separation alone
        X_sep = sep_arr[v].reshape(-1, 1)
        reg_sep = LinearRegression().fit(X_sep, y)
        r2_sep = reg_sep.score(X_sep, y)

        # 6. Everything combined
        X_all = np.column_stack([
            -align_arr[v], -uniformity[v], anisotropy[v],
            eff_rank[v], sep_arr[v]
        ])
        reg_all = LinearRegression().fit(X_all, y)
        r2_all = reg_all.score(X_all, y)

        # Correlation matrix
        corr_align = float(np.corrcoef(-align_arr[v], y)[0, 1])
        corr_unif = float(np.corrcoef(-uniformity[v], y)[0, 1])
        corr_aniso = float(np.corrcoef(anisotropy[v], y)[0, 1])
        corr_sep = float(np.corrcoef(sep_arr[v], y)[0, 1])

        results[target_name] = {
            "n_points": int(n),
            "r2_alignment_only": float(r2_align),
            "r2_uniformity_only": float(r2_unif),
            "r2_both": float(r2_both),
            "r2_anisotropy_only": float(r2_aniso),
            "r2_class_separation_only": float(r2_sep),
            "r2_all_features": float(r2_all),
            "r2_improvement_both_vs_best_single": float(
                r2_both - max(r2_align, r2_unif)
            ),
            "corr_alignment": corr_align,
            "corr_uniformity": corr_unif,
            "corr_anisotropy": corr_aniso,
            "corr_class_separation": corr_sep,
            "coefs_both": {
                "alignment": float(reg_both.coef_[0]),
                "uniformity": float(reg_both.coef_[1]),
                "intercept": float(reg_both.intercept_),
            },
        }

    # Per-objective analysis: how did each objective change alignment & uniformity?
    per_objective = {}
    baseline_metrics = all_results.get("baseline", {}).get("metrics", {})

    for obj_name, obj_data in all_results.items():
        if obj_name == "baseline":
            continue

        obj_metrics = obj_data.get("metrics", {})
        deltas = {}

        for ds_name in obj_metrics:
            if ds_name not in baseline_metrics:
                continue

            ds_deltas = {}
            for layer_key in obj_metrics[ds_name]:
                if layer_key not in baseline_metrics[ds_name]:
                    continue

                obj_layer = obj_metrics[ds_name][layer_key]
                base_layer = baseline_metrics[ds_name][layer_key]

                ds_deltas[layer_key] = {
                    "delta_alignment_l0": obj_layer["alignment_l0"] - base_layer["alignment_l0"],
                    "delta_alignment_l1": obj_layer["alignment_l1"] - base_layer["alignment_l1"],
                    "delta_uniformity": obj_layer["uniformity"] - base_layer["uniformity"],
                    "delta_knn_l0": obj_layer["knn_l0"] - base_layer["knn_l0"],
                    "delta_knn_l1": obj_layer["knn_l1"] - base_layer["knn_l1"],
                    "delta_class_sep_l0": obj_layer["class_sep_l0"] - base_layer["class_sep_l0"],
                    "delta_class_sep_l1": obj_layer["class_sep_l1"] - base_layer["class_sep_l1"],
                }

            deltas[ds_name] = ds_deltas

        # Average deltas across datasets and layers
        all_d_align_l0 = []
        all_d_unif = []
        all_d_knn_l1 = []
        all_d_sep_l1 = []

        for ds_name, ds_deltas in deltas.items():
            for layer_key, d in ds_deltas.items():
                if np.isfinite(d["delta_alignment_l1"]):
                    all_d_align_l0.append(d["delta_alignment_l0"])
                    all_d_unif.append(d["delta_uniformity"])
                    all_d_knn_l1.append(d["delta_knn_l1"])
                    all_d_sep_l1.append(d["delta_class_sep_l1"])

        per_objective[obj_name] = {
            "mean_delta_alignment_l0": float(np.mean(all_d_align_l0)) if all_d_align_l0 else None,
            "mean_delta_uniformity": float(np.mean(all_d_unif)) if all_d_unif else None,
            "mean_delta_knn_l1": float(np.mean(all_d_knn_l1)) if all_d_knn_l1 else None,
            "mean_delta_class_sep_l1": float(np.mean(all_d_sep_l1)) if all_d_sep_l1 else None,
            "deltas": deltas,
        }

    results["per_objective_deltas"] = per_objective

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    objectives = ["baseline", "contrastive", "mlm", "lm"]
    # Skip classification - it collapsed in previous run

    all_results = {}

    for obj in objectives:
        print(f"\n{'='*70}")
        print(f"  OBJECTIVE: {obj.upper()}")
        print(f"{'='*70}")

        # Load fresh model each time
        print(f"  Loading {HF_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            HF_PATH, torch_dtype=torch.float16
        ).to(device)

        if obj != "baseline":
            model = apply_lora(base_model, LORA_R, LORA_ALPHA)

            # Load training data
            data = load_hierarchical_dataset(DATASET_NAME, split="train", max_samples=5000)
            texts = [s.text for s in data.samples]
            l1_labels = [s.level1_label for s in data.samples]

            # Train
            t0 = time.time()
            if obj == "contrastive":
                avg_loss = train_contrastive(model, tokenizer, texts, device, NUM_STEPS)
            elif obj == "mlm":
                avg_loss = train_mlm(model, tokenizer, texts, device, NUM_STEPS)
            elif obj == "lm":
                avg_loss = train_lm(model, tokenizer, texts, device, NUM_STEPS)
            train_time = time.time() - t0
            print(f"  Trained in {train_time:.1f}s, avg_loss={avg_loss:.4f}")
        else:
            model = base_model
            avg_loss = None
            train_time = 0

        # Measure full metrics on each eval dataset
        metrics = {}
        for ds_name in EVAL_DATASETS:
            print(f"  Measuring {ds_name}...")
            try:
                ds_metrics = measure_full_layer_metrics(
                    model, tokenizer, ds_name, device, max_samples=800
                )
                metrics[ds_name] = ds_metrics

                # Print summary for best layer
                best = max(ds_metrics.values(), key=lambda x: x.get("knn_l1", 0))
                print(f"    Best L1={best['knn_l1']:.3f} at L{best['layer']}, "
                      f"align_l1={best['alignment_l1']:.4f}, "
                      f"unif={best['uniformity']:.4f}, "
                      f"sep={best['class_sep_l1']:.3f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                metrics[ds_name] = {"error": str(e)}

        all_results[obj] = {
            "objective": obj,
            "avg_loss": float(avg_loss) if avg_loss is not None else None,
            "train_time_sec": float(train_time),
            "metrics": metrics,
        }

        # Cleanup
        del model, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Structural Model Analysis ────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STRUCTURAL MODEL ANALYSIS")
    print(f"{'='*70}")

    analysis = analyze_structural_model(all_results)

    for target in ["knn_l0", "knn_l1"]:
        if target in analysis and "error" not in analysis[target]:
            a = analysis[target]
            print(f"\n  {target} prediction (n={a['n_points']}):")
            print(f"    R2 alignment only:    {a['r2_alignment_only']:.4f}")
            print(f"    R2 uniformity only:   {a['r2_uniformity_only']:.4f}")
            print(f"    R2 both:              {a['r2_both']:.4f}")
            print(f"    R2 anisotropy only:   {a['r2_anisotropy_only']:.4f}")
            print(f"    R2 class separation:  {a['r2_class_separation_only']:.4f}")
            print(f"    R2 all features:      {a['r2_all_features']:.4f}")
            print(f"    Improvement (both vs best single): {a['r2_improvement_both_vs_best_single']:.4f}")
            print(f"    Correlations:")
            print(f"      alignment:  {a['corr_alignment']:.4f}")
            print(f"      uniformity: {a['corr_uniformity']:.4f}")
            print(f"      anisotropy: {a['corr_anisotropy']:.4f}")
            print(f"      class_sep:  {a['corr_class_separation']:.4f}")

    # Per-objective summary
    print(f"\n  Per-objective changes from baseline:")
    for obj_name, obj_delta in analysis.get("per_objective_deltas", {}).items():
        print(f"\n    {obj_name}:")
        print(f"      Mean delta alignment L0: {obj_delta.get('mean_delta_alignment_l0', 'N/A')}")
        print(f"      Mean delta uniformity:   {obj_delta.get('mean_delta_uniformity', 'N/A')}")
        print(f"      Mean delta kNN L1:       {obj_delta.get('mean_delta_knn_l1', 'N/A')}")
        print(f"      Mean delta class sep L1: {obj_delta.get('mean_delta_class_sep_l1', 'N/A')}")

    # ── Save Everything ──────────────────────────────────────────────
    output = {
        "experiment": "CGP Week 1: Alignment-Uniformity Structural Model",
        "model": MODEL_KEY,
        "objectives": objectives,
        "conditions": all_results,
        "structural_model": analysis,
    }

    output_path = RESULTS_DIR / "cgp_alignment_uniformity.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
