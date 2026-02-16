#!/usr/bin/env python
"""
cgp_week2_control_study.py

CGP Week 2: Orthogonal Control Study (Codex-designed)

Tests whether class separation is a PROGRAMMABLE control knob for quality.

Design:
  - Primary objectives: contrastive, lm (the two that showed different geometry effects)
  - Add class-separation regularizer: L_sep = intra_class / (inter_class + eps)
    lambda_sep sweep: [0, 0.1, 0.3, 1.0]
  - Add uniformity regularizer: L_uni = log E[exp(-t||x-y||^2)]
    lambda_uni sweep: [0, 0.3]
  - 3 seeds per condition
  - Local geometry metrics: hubness, local margin, neighborhood entropy

Success criteria (Codex):
  - Monotonic increase in class_sep with lambda_sep
  - Significant ATE (average treatment effect) on L0 quality
  - L1 predictability R2 > 0.25 with expanded features

Usage:
    python -u src/cgp_week2_control_study.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from hierarchical_datasets import load_hierarchical_dataset
from cgp_controlled_finetune import (
    MODEL_KEY, HF_PATH, HIDDEN_DIM, NUM_LAYERS,
    LORA_R, LORA_ALPHA, BATCH_SIZE, MAX_SEQ_LEN, NUM_STEPS, LR,
    DATASET_NAME,
    apply_lora, TextDataset, LabeledDataset,
    pool_last_token,
)

# ── Configuration ────────────────────────────────────────────────────

LAMBDA_SEP_VALUES = [0.0, 0.1, 0.3, 1.0]
LAMBDA_UNI_VALUES = [0.0, 0.3]
PRIMARY_OBJECTIVES = ["contrastive", "lm"]
SEEDS = [42, 123, 456]
EVAL_DATASETS = ["clinc", "dbpedia_classes"]  # Focused for speed
STEPS = 500
EVAL_SAMPLES = 800


# ── Regularizers ─────────────────────────────────────────────────────

def class_separation_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Fisher-like class separation regularizer.
    L_sep = mean_intra_dist / (mean_inter_dist + eps)
    Lower = better separation. Minimizing this pushes classes apart.
    """
    # Normalize embeddings (cast to float32 for numerical stability)
    emb = F.normalize(embeddings.float(), dim=-1)
    unique_labels = labels.unique()

    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Compute centroids
    centroids = []
    intra_dists = []

    for lbl in unique_labels:
        mask = labels == lbl
        class_emb = emb[mask]
        if len(class_emb) < 2:
            continue
        centroid = class_emb.mean(dim=0)
        centroids.append(centroid)
        # Intra-class: mean distance to centroid
        dists = torch.norm(class_emb - centroid.unsqueeze(0), dim=-1)
        intra_dists.append(dists.mean())

    if len(centroids) < 2 or len(intra_dists) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    centroids = torch.stack(centroids)
    mean_intra = torch.stack(intra_dists).mean()

    # Inter-class: mean pairwise centroid distance
    n_c = len(centroids)
    inter_dists = []
    for i in range(n_c):
        for j in range(i + 1, n_c):
            inter_dists.append(torch.norm(centroids[i] - centroids[j]))
    mean_inter = torch.stack(inter_dists).mean()

    # Ratio (lower = better separation)
    return mean_intra / (mean_inter + 1e-6)


def uniformity_loss(embeddings: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity loss (Wang & Isola 2020).
    L_uni = log E[exp(-t * ||x-y||^2)]
    More negative = more uniform.
    """
    emb = F.normalize(embeddings.float(), dim=-1)
    sq_dists = torch.cdist(emb, emb, p=2).pow(2)

    # Upper triangle mask
    n = emb.size(0)
    mask = torch.triu(torch.ones(n, n, device=emb.device, dtype=torch.bool), diagonal=1)
    sq_dists_pairs = sq_dists[mask]

    return torch.log(torch.exp(-t * sq_dists_pairs).mean() + 1e-10)


# ── Training with Regularization ─────────────────────────────────────

def train_with_regularizers(
    model, tokenizer, texts, labels, device,
    primary_objective: str,
    lambda_sep: float = 0.0,
    lambda_uni: float = 0.0,
    num_steps: int = 500,
):
    """
    Train with primary objective + optional regularizers.
    Returns avg_loss, avg_sep_loss, avg_uni_loss.
    """
    if primary_objective == "contrastive":
        dataset = TextDataset(texts, tokenizer)
    else:
        dataset = TextDataset(texts, tokenizer)

    # Also need labeled data for sep regularizer
    labeled_dataset = LabeledDataset(texts, labels, tokenizer) if lambda_sep > 0 else None

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    labeled_loader = None
    if labeled_dataset:
        labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        labeled_iter = iter(labeled_loader)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    model.train()
    step = 0
    total_primary = 0
    total_sep = 0
    total_uni = 0

    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # ── Primary Objective ────────────────────────────────
            if primary_objective == "contrastive":
                out1 = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                out2 = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                z1 = pool_last_token(out1.hidden_states[-1], batch["attention_mask"])
                z2 = pool_last_token(out2.hidden_states[-1], batch["attention_mask"])
                z1 = F.normalize(z1, dim=-1)
                z2 = F.normalize(z2, dim=-1)
                sim = z1 @ z2.T / 0.05
                target = torch.arange(z1.size(0), device=device)
                primary_loss = F.cross_entropy(sim, target)
                last_hidden = out1.hidden_states[-1]
                attn_mask = batch["attention_mask"]

            elif primary_objective == "lm":
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                lm_labels = input_ids.clone()
                lm_labels[attention_mask == 0] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=lm_labels,
                    output_hidden_states=True,
                    return_dict=True,
                )
                primary_loss = outputs.loss
                last_hidden = outputs.hidden_states[-1]
                attn_mask = attention_mask

            loss = primary_loss
            total_primary += primary_loss.item()

            # ── Class Separation Regularizer ─────────────────────
            if lambda_sep > 0 and labeled_loader is not None:
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    labeled_batch = next(labeled_iter)

                lb_ids = labeled_batch["input_ids"].to(device)
                lb_mask = labeled_batch["attention_mask"].to(device)
                lb_labels = labeled_batch["labels"].to(device)

                lb_out = model(
                    input_ids=lb_ids,
                    attention_mask=lb_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                lb_emb = pool_last_token(lb_out.hidden_states[-1], lb_mask)
                sep_loss = class_separation_loss(lb_emb, lb_labels)
                loss = loss + lambda_sep * sep_loss
                total_sep += sep_loss.item()

            # ── Uniformity Regularizer ───────────────────────────
            if lambda_uni > 0:
                emb = pool_last_token(last_hidden, attn_mask)
                uni_loss = uniformity_loss(emb)
                # We want to MINIMIZE uniformity (more negative = better)
                # So we add it directly (it's already negative)
                loss = loss + lambda_uni * uni_loss
                total_uni += uni_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 200 == 0:
                print(f"    Step {step}/{num_steps}: "
                      f"primary={total_primary/step:.4f} "
                      f"sep={total_sep/step:.4f} "
                      f"uni={total_uni/step:.4f}")

    return (
        total_primary / max(step, 1),
        total_sep / max(step, 1),
        total_uni / max(step, 1),
    )


# ── Local Geometry Metrics ───────────────────────────────────────────

def compute_local_metrics(reps: np.ndarray, labels: np.ndarray, k: int = 10) -> Dict:
    """
    Compute local geometry metrics:
    - Hubness: skewness of k-occurrence distribution
    - Local margin: distance to nearest different-class - nearest same-class
    - Neighborhood entropy: entropy of class distribution in k-NN
    """
    from scipy.spatial.distance import cdist

    # Normalize
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    reps_norm = reps / np.maximum(norms, 1e-8)

    n = len(reps_norm)
    if n > 2000:
        idx = np.random.choice(n, 2000, replace=False)
        reps_norm = reps_norm[idx]
        labels = labels[idx]
        n = 2000

    # Cosine similarity matrix
    sim_matrix = reps_norm @ reps_norm.T
    np.fill_diagonal(sim_matrix, -np.inf)

    # k-NN indices
    knn_indices = np.argsort(-sim_matrix, axis=1)[:, :k]

    # 1. Hubness: how often each point appears as k-NN of others
    knn_counts = np.zeros(n)
    for i in range(n):
        for j in knn_indices[i]:
            knn_counts[j] += 1
    hubness_skewness = float(
        ((knn_counts - knn_counts.mean()) ** 3).mean() /
        (knn_counts.std() ** 3 + 1e-10)
    )

    # 2. Local margin: nearest same-class distance - nearest diff-class distance
    local_margins = []
    for i in range(n):
        sims = sim_matrix[i].copy()
        same_mask = labels == labels[i]
        same_mask[i] = False
        diff_mask = ~same_mask
        diff_mask[i] = False

        if same_mask.any() and diff_mask.any():
            nearest_same = sims[same_mask].max()
            nearest_diff = sims[diff_mask].max()
            # Positive margin = same class is closer (good)
            local_margins.append(nearest_same - nearest_diff)

    mean_margin = float(np.mean(local_margins)) if local_margins else 0.0

    # 3. Neighborhood entropy
    entropies = []
    for i in range(n):
        neighbor_labels = labels[knn_indices[i]]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    mean_entropy = float(np.mean(entropies))

    return {
        "hubness_skewness": hubness_skewness,
        "mean_local_margin": mean_margin,
        "mean_neighborhood_entropy": mean_entropy,
    }


# ── Full Measurement ─────────────────────────────────────────────────

def measure_condition(model, tokenizer, device, dataset_name, max_samples=800) -> Dict:
    """Measure all metrics for one condition on one dataset."""
    from cti_geometry_analysis import compute_anisotropy, compute_effective_rank
    from cti_knn_sweep import knn_accuracy
    from cgp_alignment_uniformity import (
        compute_alignment, compute_uniformity, compute_class_separation,
    )

    data = load_hierarchical_dataset(dataset_name, split="test", max_samples=max_samples)
    texts = [s.text for s in data.samples]
    l0_labels = np.array([s.level0_label for s in data.samples])
    l1_labels = np.array([s.level1_label for s in data.samples])

    model.eval()
    all_layer_reps = {}

    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch_texts = texts[i:i + 32]
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

    # Focus on last 3 layers + layer 0 for speed
    key_layers = [0, NUM_LAYERS - 2, NUM_LAYERS - 1, NUM_LAYERS]
    key_layers = [l for l in key_layers if l in all_layer_reps]

    results = {}
    for layer_idx in key_layers:
        reps = np.concatenate(all_layer_reps[layer_idx], axis=0).astype(np.float32)

        # Global metrics
        knn_l0 = knn_accuracy(reps, l0_labels, k=5)
        knn_l1 = knn_accuracy(reps, l1_labels, k=5)
        align_l0 = compute_alignment(reps, l0_labels)
        align_l1 = compute_alignment(reps, l1_labels)
        unif = compute_uniformity(reps)
        sep_l0 = compute_class_separation(reps, l0_labels)
        sep_l1 = compute_class_separation(reps, l1_labels)
        aniso = compute_anisotropy(reps)
        eff_rank = compute_effective_rank(reps)

        # Local metrics (on L1 labels for fine-grained)
        local = compute_local_metrics(reps, l1_labels, k=10)

        results[layer_idx] = {
            "layer": layer_idx,
            "knn_l0": float(knn_l0),
            "knn_l1": float(knn_l1),
            "alignment_l0": float(align_l0),
            "alignment_l1": float(align_l1),
            "uniformity": float(unif),
            "class_sep_l0": float(sep_l0),
            "class_sep_l1": float(sep_l1),
            "anisotropy": float(aniso),
            "effective_rank": float(eff_rank),
            **local,
        }

    return results


# ── Main Experiment Loop ─────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Objectives: {PRIMARY_OBJECTIVES}")
    print(f"Lambda_sep: {LAMBDA_SEP_VALUES}")
    print(f"Lambda_uni: {LAMBDA_UNI_VALUES}")
    print(f"Seeds: {SEEDS}")

    # Pre-load training data
    train_data = load_hierarchical_dataset(DATASET_NAME, split="train", max_samples=5000)
    texts = [s.text for s in train_data.samples]
    l0_labels = [s.level0_label for s in train_data.samples]
    l1_labels = [s.level1_label for s in train_data.samples]

    # Generate all conditions
    conditions = []
    # Baseline (no training)
    conditions.append(("baseline", 0.0, 0.0, 0))

    for obj, lam_sep, lam_uni, seed in product(
        PRIMARY_OBJECTIVES, LAMBDA_SEP_VALUES, LAMBDA_UNI_VALUES, SEEDS
    ):
        conditions.append((obj, lam_sep, lam_uni, seed))

    print(f"Total conditions: {len(conditions)}")

    all_results = []

    for idx, (obj, lam_sep, lam_uni, seed) in enumerate(conditions):
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(conditions)}] obj={obj} lam_sep={lam_sep} lam_uni={lam_uni} seed={seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            HF_PATH, torch_dtype=torch.float16
        ).to(device)

        if obj == "baseline":
            model = base_model
            avg_primary = avg_sep = avg_uni = 0.0
            train_time = 0.0
        else:
            model = apply_lora(base_model, LORA_R, LORA_ALPHA)
            t0 = time.time()
            avg_primary, avg_sep, avg_uni = train_with_regularizers(
                model, tokenizer, texts, l1_labels, device,
                primary_objective=obj,
                lambda_sep=lam_sep,
                lambda_uni=lam_uni,
                num_steps=STEPS,
            )
            train_time = time.time() - t0
            print(f"  Trained in {train_time:.1f}s")

        # Evaluate on focused datasets
        eval_results = {}
        for ds_name in EVAL_DATASETS:
            try:
                metrics = measure_condition(model, tokenizer, device, ds_name, EVAL_SAMPLES)
                eval_results[ds_name] = metrics

                # Quick summary: best last layer
                last_key = max(metrics.keys())
                m = metrics[last_key]
                print(f"    {ds_name} L{last_key}: knn_l0={m['knn_l0']:.3f} "
                      f"knn_l1={m['knn_l1']:.3f} sep_l1={m['class_sep_l1']:.3f} "
                      f"margin={m['mean_local_margin']:.4f}")
            except Exception as e:
                print(f"    {ds_name}: ERROR - {e}")
                eval_results[ds_name] = {"error": str(e)}

        result = {
            "objective": obj,
            "lambda_sep": lam_sep,
            "lambda_uni": lam_uni,
            "seed": seed,
            "avg_primary_loss": float(avg_primary),
            "avg_sep_loss": float(avg_sep),
            "avg_uni_loss": float(avg_uni),
            "train_time_sec": float(train_time),
            "metrics": eval_results,
        }
        all_results.append(result)

        # Cleanup
        del model, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save raw results FIRST (before analysis, to avoid data loss) ──
    output = {
        "experiment": "CGP Week 2: Orthogonal Control Study",
        "model": MODEL_KEY,
        "design": {
            "objectives": PRIMARY_OBJECTIVES,
            "lambda_sep_values": LAMBDA_SEP_VALUES,
            "lambda_uni_values": LAMBDA_UNI_VALUES,
            "seeds": SEEDS,
            "steps": STEPS,
            "eval_datasets": EVAL_DATASETS,
        },
        "conditions": all_results,
    }

    output_path = RESULTS_DIR / "cgp_week2_control_study.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nRaw results saved to {output_path}")

    # ── Analysis ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ANALYSIS: Causal Mediation Tests")
    print(f"{'='*60}")

    try:
        analysis = analyze_control_study(all_results)

        # Print key results
        for test_name, test_result in analysis.items():
            if isinstance(test_result, dict):
                print(f"\n  {test_name}:")
                for k, v in test_result.items():
                    if isinstance(v, (float, int, bool)):
                        print(f"    {k}: {v}")

        # Re-save with analysis
        output["analysis"] = analysis
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2,
                      default=lambda o: float(o) if hasattr(o, 'item') else str(o))
        print(f"\nFull results (with analysis) saved to {output_path}")
    except Exception as e:
        print(f"\n  WARNING: Analysis failed: {e}")
        print("  Raw results were already saved. Run cgp_week2_analysis.py instead.")


def analyze_control_study(results: List[Dict]) -> Dict:
    """
    Analyze the control study for causal effects.
    Tests:
    1. Monotonicity: does class_sep increase with lambda_sep?
    2. Dose-response: does quality improve with class_sep?
    3. Causal mediation: lambda_sep -> class_sep -> quality
    4. L1 prediction with local metrics
    """
    from sklearn.linear_model import LinearRegression

    # Collect data points (last layer only)
    rows = []
    for r in results:
        for ds_name, ds_metrics in r.get("metrics", {}).items():
            if isinstance(ds_metrics, dict) and "error" not in ds_metrics:
                # Get last layer
                last_key = max(k for k in ds_metrics.keys() if isinstance(k, int))
                m = ds_metrics[last_key]
                rows.append({
                    "objective": r["objective"],
                    "lambda_sep": r["lambda_sep"],
                    "lambda_uni": r["lambda_uni"],
                    "seed": r["seed"],
                    "dataset": ds_name,
                    **m,
                })

    if not rows:
        return {"error": "no data"}

    analysis = {}

    # 1. Monotonicity test: does class_sep increase with lambda_sep?
    for obj in PRIMARY_OBJECTIVES:
        obj_rows = [r for r in rows if r["objective"] == obj and r["lambda_uni"] == 0.0]
        if not obj_rows:
            continue

        lam_sep_vals = sorted(set(r["lambda_sep"] for r in obj_rows))
        mean_sep_by_lambda = {}
        mean_knn_l0_by_lambda = {}
        mean_knn_l1_by_lambda = {}

        for lam in lam_sep_vals:
            lam_rows = [r for r in obj_rows if r["lambda_sep"] == lam]
            seps = [r["class_sep_l1"] for r in lam_rows if np.isfinite(r.get("class_sep_l1", 0))]
            knn_l0s = [r["knn_l0"] for r in lam_rows]
            knn_l1s = [r["knn_l1"] for r in lam_rows]

            mean_sep_by_lambda[lam] = float(np.mean(seps)) if seps else None
            mean_knn_l0_by_lambda[lam] = float(np.mean(knn_l0s))
            mean_knn_l1_by_lambda[lam] = float(np.mean(knn_l1s))

        # Check monotonicity
        sep_values = [mean_sep_by_lambda[l] for l in lam_sep_vals if mean_sep_by_lambda.get(l) is not None]
        is_monotonic = all(sep_values[i] <= sep_values[i+1] for i in range(len(sep_values)-1))

        analysis[f"monotonicity_{obj}"] = {
            "lambda_sep_values": lam_sep_vals,
            "mean_class_sep_l1": mean_sep_by_lambda,
            "mean_knn_l0": mean_knn_l0_by_lambda,
            "mean_knn_l1": mean_knn_l1_by_lambda,
            "is_monotonic": bool(is_monotonic),
        }

    # 2. Overall dose-response: class_sep -> quality
    all_sep = np.array([r["class_sep_l1"] for r in rows if r["objective"] != "baseline"])
    all_knn_l0 = np.array([r["knn_l0"] for r in rows if r["objective"] != "baseline"])
    all_knn_l1 = np.array([r["knn_l1"] for r in rows if r["objective"] != "baseline"])

    valid = np.isfinite(all_sep) & np.isfinite(all_knn_l0)
    if valid.sum() > 5:
        r_l0, p_l0 = stats.spearmanr(all_sep[valid], all_knn_l0[valid])
        r_l1, p_l1 = stats.spearmanr(all_sep[valid], all_knn_l1[valid])
        analysis["dose_response"] = {
            "spearman_sep_vs_knn_l0": float(r_l0),
            "p_sep_vs_knn_l0": float(p_l0),
            "spearman_sep_vs_knn_l1": float(r_l1),
            "p_sep_vs_knn_l1": float(p_l1),
            "n_points": int(valid.sum()),
        }

    # 3. L1 prediction with expanded features (local metrics)
    feature_names = [
        "class_sep_l1", "alignment_l1", "uniformity", "anisotropy",
        "effective_rank", "hubness_skewness", "mean_local_margin",
        "mean_neighborhood_entropy",
    ]

    non_baseline = [r for r in rows if r["objective"] != "baseline"]
    if len(non_baseline) > 10:
        X_full = []
        y_l1 = []
        for r in non_baseline:
            features = [r.get(f, 0.0) for f in feature_names]
            if all(np.isfinite(f) for f in features):
                X_full.append(features)
                y_l1.append(r["knn_l1"])

        if len(X_full) > 10:
            X_full = np.array(X_full)
            y_l1 = np.array(y_l1)

            # R2 with all features
            reg = LinearRegression().fit(X_full, y_l1)
            r2_all = reg.score(X_full, y_l1)

            # R2 with just class_sep
            reg_sep = LinearRegression().fit(X_full[:, :1], y_l1)
            r2_sep = reg_sep.score(X_full[:, :1], y_l1)

            # R2 with local metrics only
            local_idx = [feature_names.index(f) for f in
                        ["hubness_skewness", "mean_local_margin", "mean_neighborhood_entropy"]]
            X_local = X_full[:, local_idx]
            reg_local = LinearRegression().fit(X_local, y_l1)
            r2_local = reg_local.score(X_local, y_l1)

            # Feature importances (coefficients * std)
            feature_importance = {}
            for i, fname in enumerate(feature_names):
                std = X_full[:, i].std()
                feature_importance[fname] = float(abs(reg.coef_[i]) * std)

            analysis["l1_prediction"] = {
                "n_points": len(y_l1),
                "r2_all_features": float(r2_all),
                "r2_class_sep_only": float(r2_sep),
                "r2_local_metrics_only": float(r2_local),
                "feature_importance": feature_importance,
                "feature_names": feature_names,
                "meets_success_criterion": bool(r2_all > 0.25),
            }

    # 4. Causal mediation: lambda_sep -> class_sep -> quality
    # Average treatment effect of lambda_sep on quality
    for obj in PRIMARY_OBJECTIVES:
        obj_rows = [r for r in rows if r["objective"] == obj and r["lambda_uni"] == 0.0]
        baseline_rows = [r for r in obj_rows if r["lambda_sep"] == 0.0]
        treated_rows = [r for r in obj_rows if r["lambda_sep"] > 0.0]

        if baseline_rows and treated_rows:
            baseline_l0 = np.mean([r["knn_l0"] for r in baseline_rows])
            treated_l0 = np.mean([r["knn_l0"] for r in treated_rows])
            baseline_l1 = np.mean([r["knn_l1"] for r in baseline_rows])
            treated_l1 = np.mean([r["knn_l1"] for r in treated_rows])

            ate_l0 = treated_l0 - baseline_l0
            ate_l1 = treated_l1 - baseline_l1

            # Test significance
            b_vals = [r["knn_l0"] for r in baseline_rows]
            t_vals = [r["knn_l0"] for r in treated_rows]
            if len(b_vals) >= 3 and len(t_vals) >= 3:
                t_stat, p_val = stats.ttest_ind(t_vals, b_vals)
            else:
                t_stat, p_val = 0.0, 1.0

            analysis[f"ate_{obj}"] = {
                "ate_l0": float(ate_l0),
                "ate_l1": float(ate_l1),
                "baseline_mean_l0": float(baseline_l0),
                "treated_mean_l0": float(treated_l0),
                "p_value_l0": float(p_val),
                "n_baseline": len(baseline_rows),
                "n_treated": len(treated_rows),
            }

    return analysis


if __name__ == "__main__":
    main()
