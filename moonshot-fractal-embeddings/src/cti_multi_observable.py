#!/usr/bin/env python
"""
MULTI-OBSERVABLE phase transition validation.

Reuses the proven ResidualScaler from cti_residual_surgery.py.
Measures 4 independent observables at each residual alpha:

1. kNN accuracy (supervised quality)
2. Intrinsic dimensionality (participation ratio)
3. Effective rank (nuclear/spectral norm ratio)
4. Alignment-Uniformity (Wang & Isola, 2020)

If all observables transition at the same alpha*, this proves
the phase transition is genuine (not an artifact of kNN).
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import (
    load_model, ResidualScaler, find_residual_layers
)
from hierarchical_datasets import load_hierarchical_dataset


def extract_representations(model, tokenizer, texts, alpha, device="cuda", batch_size=32):
    """Extract all layer representations with given residual alpha."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch_texts = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**enc, output_hidden_states=True, return_dict=True)

            hidden_states = outputs.hidden_states
            mask = enc.get("attention_mask", torch.ones(enc["input_ids"].shape, device=device))

            for layer_idx, hs in enumerate(hidden_states):
                hs_f = hs.float()
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (hs_f * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                # L2 normalize
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if layer_idx not in all_hidden:
                    all_hidden[layer_idx] = []
                all_hidden[layer_idx].append(pooled.cpu().numpy())

    # Concatenate
    reps = {}
    for layer_idx in sorted(all_hidden.keys()):
        reps[layer_idx] = np.concatenate(all_hidden[layer_idx], axis=0)
    return reps


def compute_knn(X, labels, k=5):
    """Compute kNN accuracy with train/test split."""
    n = len(labels)
    n_train = int(0.7 * n)
    try:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(X[:n_train], labels[:n_train])
        return float(knn.score(X[n_train:], labels[n_train:]))
    except Exception:
        return 0.0


def compute_intrinsic_dim(X):
    """Participation ratio: (sum lambda)^2 / sum(lambda^2)."""
    X_c = X - X.mean(axis=0)
    try:
        _, S, _ = np.linalg.svd(X_c, full_matrices=False)
        evals = S ** 2 / max(X.shape[0] - 1, 1)
        pr = float((evals.sum() ** 2) / (evals ** 2).sum())
        return pr
    except Exception:
        return 0.0


def compute_effective_rank(X):
    """Effective rank: sum(S) / max(S)."""
    X_c = X - X.mean(axis=0)
    try:
        _, S, _ = np.linalg.svd(X_c, full_matrices=False)
        return float(S.sum() / S.max()) if S.max() > 0 else 0.0
    except Exception:
        return 0.0


def compute_alignment_uniformity(X, labels, n_pairs=5000):
    """Alignment and uniformity metrics (Wang & Isola, 2020)."""
    rng = np.random.RandomState(42)

    # Alignment: mean squared distance between same-class pairs
    unique = np.unique(labels)
    align_sum = 0
    align_count = 0
    for lbl in unique[:50]:
        mask = labels == lbl
        class_reps = X[mask]
        if len(class_reps) >= 2:
            n_sample = min(50, len(class_reps))
            for _ in range(n_sample):
                i, j = rng.choice(len(class_reps), 2, replace=False)
                align_sum += np.sum((class_reps[i] - class_reps[j]) ** 2)
                align_count += 1

    alignment = align_sum / max(align_count, 1)

    # Uniformity: log E[exp(-2*||x-y||^2)]
    n = min(n_pairs, X.shape[0] * (X.shape[0] - 1) // 2)
    unif_sum = 0
    for _ in range(n):
        i, j = rng.choice(X.shape[0], 2, replace=False)
        dist_sq = np.sum((X[i] - X[j]) ** 2)
        unif_sum += np.exp(-2 * dist_sq)
    uniformity = float(np.log(unif_sum / max(n, 1) + 1e-10))

    return float(alignment), float(uniformity)


def compute_all_observables(reps, labels, n_layers):
    """Compute all observables at each layer."""
    results = {}
    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue

        knn = compute_knn(X, labels)
        pr = compute_intrinsic_dim(X)
        er = compute_effective_rank(X)
        alignment, uniformity = compute_alignment_uniformity(X, labels)

        results[layer_idx] = {
            "x": layer_idx / max(n_layers, 1),
            "knn_acc": knn,
            "intrinsic_dim": pr,
            "effective_rank": er,
            "alignment": alignment,
            "uniformity": uniformity,
        }

    return results


def compute_summary_stats(obs_by_layer, key, n_layers):
    """Compute mean, peak, and profile fit for one observable."""
    xs = []
    vals = []
    for layer_idx in sorted(obs_by_layer.keys()):
        xs.append(obs_by_layer[layer_idx]["x"])
        vals.append(obs_by_layer[layer_idx][key])

    xs = np.array(xs)
    vals = np.array(vals)

    mean_val = float(np.mean(vals))
    peak_val = float(np.max(vals))
    peak_layer = int(np.argmax(vals))

    return {
        "mean": mean_val,
        "peak": peak_val,
        "peak_layer": peak_layer,
    }


def main():
    print("=" * 70)
    print("MULTI-OBSERVABLE PHASE TRANSITION VALIDATION")
    print("=" * 70)

    model_id = "Qwen/Qwen3-0.6B"
    dataset_name = "clinc"
    # Focus on the transition region with denser sampling
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, n_layers, n_params = load_model(model_id, device)

    # Load dataset
    ds = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])
    n_classes = len(ds.level1_names)

    print(f"\nModel: {model_id} ({n_layers} layers)")
    print(f"Dataset: {dataset_name} ({len(texts)} samples, {n_classes} classes)")
    print(f"Alphas: {alphas}")

    observable_keys = ["knn_acc", "intrinsic_dim", "effective_rank", "alignment", "uniformity"]
    # Accumulate mean values across alphas for transition analysis
    means_by_alpha = {key: [] for key in observable_keys}
    all_results = {}

    for alpha in alphas:
        print(f"\n--- alpha = {alpha:.2f} ---")
        t0 = time.time()

        reps = extract_representations(model, tokenizer, texts, alpha, device)
        obs = compute_all_observables(reps, labels, n_layers)

        elapsed = time.time() - t0
        n_valid = len(obs)
        print(f"  Computed {n_valid} layers in {elapsed:.1f}s")

        # Summary for each observable
        for key in observable_keys:
            summary = compute_summary_stats(obs, key, n_layers)
            means_by_alpha[key].append(summary["mean"])
            print(f"  {key:20s}: mean={summary['mean']:.4f}, peak={summary['peak']:.4f} (L{summary['peak_layer']})")

        all_results[str(alpha)] = {
            "profile": {
                str(k): v for k, v in obs.items()
            },
        }
        sys.stdout.flush()

    # ============================================================
    # TRANSITION ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("MULTI-OBSERVABLE TRANSITION ANALYSIS")
    print(f"{'='*70}")

    alphas_arr = np.array(alphas)
    transition_points = {}

    for key in observable_keys:
        values = np.array(means_by_alpha[key])

        v_min, v_max = values.min(), values.max()
        v_range = v_max - v_min
        if v_range < 1e-10:
            transition_points[key] = None
            print(f"  {key:25s}: NO VARIATION")
            continue

        v_norm = (values - v_min) / v_range

        # Find alpha_50 (midpoint of transition)
        alpha_50 = None
        # Try increasing
        for i in range(len(alphas_arr) - 1):
            if v_norm[i] <= 0.5 and v_norm[i + 1] > 0.5:
                frac = (0.5 - v_norm[i]) / (v_norm[i + 1] - v_norm[i])
                alpha_50 = alphas_arr[i] + frac * (alphas_arr[i + 1] - alphas_arr[i])
                break
        # Try decreasing
        if alpha_50 is None:
            for i in range(len(alphas_arr) - 1):
                if v_norm[i] >= 0.5 and v_norm[i + 1] < 0.5:
                    frac = (0.5 - v_norm[i]) / (v_norm[i + 1] - v_norm[i])
                    alpha_50 = alphas_arr[i] + frac * (alphas_arr[i + 1] - alphas_arr[i])
                    break

        transition_points[key] = float(alpha_50) if alpha_50 is not None else None
        a50_str = f"{alpha_50:.3f}" if alpha_50 is not None else "N/A"
        direction = "incr" if values[-1] > values[0] else "decr"
        print(f"  {key:25s}: alpha_50 = {a50_str:>8}  range = {v_range:.4f}  ({direction})")

    # Consistency check
    valid = {k: v for k, v in transition_points.items() if v is not None}
    if len(valid) >= 2:
        vals = list(valid.values())
        mean_a = np.mean(vals)
        std_a = np.std(vals)
        cv = std_a / mean_a if mean_a > 0 else float("inf")
        print(f"\n  N observables with transition: {len(valid)}")
        print(f"  Mean alpha* = {mean_a:.3f} +/- {std_a:.3f}")
        print(f"  Coefficient of variation = {100 * cv:.1f}%")
        if cv < 0.15:
            print("  CONSISTENT: All observables transition at the same alpha*!")
        elif cv < 0.30:
            print("  PARTIALLY CONSISTENT: Same general region")
        else:
            print("  INCONSISTENT: Different observables show different transitions")

        # Compare with kNN-only alpha* from previous analysis
        print(f"\n  Reference (from cti_residual_dense.json): alpha* ~ 0.87")
        for k, v in valid.items():
            print(f"    {k}: {v:.3f}  (delta = {abs(v - 0.87):.3f})")

    # Save
    out = {
        "model_id": model_id,
        "dataset": dataset_name,
        "num_layers": n_layers,
        "n_params": n_params,
        "alphas": alphas,
        "observables": {
            key: means_by_alpha[key] for key in observable_keys
        },
        "transition_points": {
            k: v for k, v in transition_points.items()
        },
        "n_consistent": len(valid),
        "per_alpha": all_results,
    }

    out_path = RESULTS_DIR / "cti_multi_observable.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
