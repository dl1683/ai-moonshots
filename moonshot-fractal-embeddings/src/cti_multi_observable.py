#!/usr/bin/env python
"""
MULTI-OBSERVABLE phase transition validation.

A genuine phase transition must show consistent transition across INDEPENDENT
observables. We measure 4 observables at each residual alpha for one model:

1. kNN accuracy (existing) -- supervised quality
2. Intrinsic dimensionality (participation ratio) -- geometric complexity
3. Effective rank (nuclear norm / spectral norm) -- rank structure
4. Alignment-uniformity -- representation quality (Wang & Isola, 2020)

If all 4 observables undergo a transition at the same alpha*, this is
strong evidence for a genuine phase transition, not an artifact of
a single metric.
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_model_and_data(model_id, dataset_name="clinc"):
    """Load model and extract class labels."""
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Load dataset
    if dataset_name == "clinc":
        ds = load_dataset("clinc_oos", "plus", split="test")
        texts = ds["text"]
        labels = ds["intent"]
    elif dataset_name == "trec":
        ds = load_dataset("CogComp/trec", split="test")
        texts = ds["text"]
        labels = ds["coarse_label"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Subsample for speed
    n = min(2000, len(texts))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(texts), n, replace=False)
    texts = [texts[i] for i in idx]
    labels = np.array([labels[i] for i in idx])

    return model, tokenizer, device, texts, labels


def extract_layer_representations(model, tokenizer, device, texts, alpha, batch_size=32):
    """Extract representations at each layer with residual strength alpha."""
    n_layers = model.config.num_hidden_layers

    # Pre-tokenize
    all_hidden = [[] for _ in range(n_layers + 1)]

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            # Get embeddings
            if hasattr(model, "embed_tokens"):
                h = model.embed_tokens(inputs["input_ids"])
            elif hasattr(model, "embeddings"):
                h = model.embeddings(inputs["input_ids"])
            elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                h = model.model.embed_tokens(inputs["input_ids"])
            else:
                outputs = model(**inputs, output_hidden_states=True)
                for layer_idx in range(n_layers + 1):
                    hs = outputs.hidden_states[layer_idx]
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)
                    all_hidden[layer_idx].append(pooled.cpu().float().numpy())
                continue

            # Manual forward pass with alpha scaling
            mask = inputs.get("attention_mask", None)

            # Store embedding layer
            pooled = mean_pool(h, inputs["attention_mask"])
            all_hidden[0].append(pooled.cpu().float().numpy())

            # Pass through each layer
            for layer_idx in range(n_layers):
                layer = model.layers[layer_idx] if hasattr(model, "layers") else (
                    model.model.layers[layer_idx] if hasattr(model, "model") else None
                )
                if layer is None:
                    break

                # Standard transformer layer: h_new = h + f(h)
                # Modified: h_new = alpha * h + f(h)
                h_in = h.clone()

                # Forward through the layer to get the output
                # Most layers expect (hidden_states, attention_mask, position_ids)
                try:
                    layer_out = layer(h_in, attention_mask=mask)
                    if isinstance(layer_out, tuple):
                        h_out = layer_out[0]
                    else:
                        h_out = layer_out
                except Exception:
                    # Fallback: just use the layer normally
                    try:
                        layer_out = layer(h_in)
                        h_out = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                    except Exception:
                        break

                # Apply alpha scaling to residual
                # h_out = alpha * h_in + nonlinear_branch
                # But the layer already computed h_out = h_in + nonlinear_branch
                # So: nonlinear_branch = h_out - h_in
                # Modified: h_new = alpha * h_in + nonlinear_branch
                nonlinear = h_out - h_in
                h = alpha * h_in + nonlinear

                pooled = mean_pool(h, inputs["attention_mask"])
                all_hidden[layer_idx + 1].append(pooled.cpu().float().numpy())

    # Concatenate
    reps = {}
    for layer_idx in range(n_layers + 1):
        if all_hidden[layer_idx]:
            reps[layer_idx] = np.concatenate(all_hidden[layer_idx], axis=0)
    return reps


def mean_pool(h, attention_mask):
    """Mean pool with attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def compute_observables(reps, labels, n_layers):
    """Compute all observables for a set of layer representations."""
    results = {}
    skip_layers = set()  # Track layers without data

    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            skip_layers.add(layer_idx)
            continue

        # 1. kNN accuracy (k=5)
        from sklearn.model_selection import cross_val_score
        try:
            knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
            # Simple train/test split
            n = len(labels)
            n_train = int(0.7 * n)
            knn.fit(X[:n_train], labels[:n_train])
            knn_acc = knn.score(X[n_train:], labels[n_train:])
        except Exception:
            knn_acc = 0.0

        # 2. Intrinsic dimensionality (participation ratio)
        try:
            X_centered = X - X.mean(axis=0)
            # Use SVD for numerical stability
            _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
            eigenvalues = S ** 2 / (X.shape[0] - 1)
            # Participation ratio: (sum lambda)^2 / sum(lambda^2)
            pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
            # Normalize by dimension
            pr_normalized = pr / X.shape[1]
        except Exception:
            pr = 0.0
            pr_normalized = 0.0

        # 3. Effective rank (Vershynin definition)
        try:
            # nuclear_norm / spectral_norm = sum(S) / max(S)
            eff_rank = S.sum() / S.max() if S.max() > 0 else 0
            eff_rank_normalized = eff_rank / X.shape[1]
        except Exception:
            eff_rank = 0.0
            eff_rank_normalized = 0.0

        # 4. Alignment and Uniformity (Wang & Isola, 2020)
        try:
            # Normalize to unit sphere
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

            # Alignment: E[||f(x) - f(x+)||^2] for same-class pairs
            unique_labels = np.unique(labels)
            alignment_sum = 0
            alignment_count = 0
            for lbl in unique_labels[:50]:  # Cap at 50 classes for speed
                class_mask = labels == lbl
                class_reps = X_norm[class_mask]
                if len(class_reps) >= 2:
                    # Sample pairs
                    n_pairs = min(100, len(class_reps) * (len(class_reps) - 1) // 2)
                    for _ in range(n_pairs):
                        i, j = np.random.choice(len(class_reps), 2, replace=False)
                        alignment_sum += np.sum((class_reps[i] - class_reps[j]) ** 2)
                        alignment_count += 1
            alignment = alignment_sum / max(alignment_count, 1)

            # Uniformity: log E[e^{-2||f(x) - f(y)||^2}] for random pairs
            n_unif = min(5000, X_norm.shape[0] * (X_norm.shape[0] - 1) // 2)
            unif_sum = 0
            for _ in range(n_unif):
                i, j = np.random.choice(X_norm.shape[0], 2, replace=False)
                dist_sq = np.sum((X_norm[i] - X_norm[j]) ** 2)
                unif_sum += np.exp(-2 * dist_sq)
            uniformity = np.log(unif_sum / max(n_unif, 1) + 1e-10)
        except Exception:
            alignment = 0.0
            uniformity = 0.0

        results[layer_idx] = {
            "x": layer_idx / max(n_layers, 1),
            "knn_acc": float(knn_acc),
            "participation_ratio": float(pr),
            "pr_normalized": float(pr_normalized),
            "effective_rank": float(eff_rank),
            "eff_rank_normalized": float(eff_rank_normalized),
            "alignment": float(alignment),
            "uniformity": float(uniformity),
        }

    return results


def fit_gaussian_profile(layer_results, observable_key):
    """Fit Gaussian-in-logit to a depth profile for any observable."""
    from scipy.optimize import curve_fit
    from scipy.special import expit, logit as sp_logit

    xs = []
    ys = []
    for layer_idx in sorted(layer_results.keys(), key=int):
        r = layer_results[layer_idx]
        xs.append(r["x"])
        ys.append(r[observable_key])

    xs = np.array(xs)
    ys = np.array(ys)

    if len(xs) < 4:
        return {"beta": 0, "mu": 0, "b0": 0, "r2": 0}

    # Normalize to (0, 1) for logit
    y_min, y_max = ys.min(), ys.max()
    if y_max - y_min < 1e-10:
        return {"beta": 0, "mu": 0.5, "b0": 0, "r2": 0}

    ys_norm = np.clip((ys - y_min) / (y_max - y_min), 0.01, 0.99)

    try:
        def bell(x, b0, beta, mu):
            return expit(b0 - beta * (x - mu) ** 2)

        p0 = [0, 2, 0.5]
        bounds = ([-10, 0, -0.5], [10, 50, 1.5])
        popt, _ = curve_fit(bell, xs, ys_norm, p0=p0, bounds=bounds, maxfev=5000)
        y_pred = bell(xs, *popt)
        ss_res = np.sum((ys_norm - y_pred) ** 2)
        ss_tot = np.sum((ys_norm - ys_norm.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"beta": float(popt[1]), "mu": float(popt[2]), "b0": float(popt[0]), "r2": float(r2)}
    except Exception:
        return {"beta": 0, "mu": 0.5, "b0": 0, "r2": 0}


def main():
    print("=" * 70)
    print("MULTI-OBSERVABLE PHASE TRANSITION VALIDATION")
    print("4 independent observables at each residual alpha")
    print("=" * 70)

    model_id = "Qwen/Qwen3-0.6B"
    dataset = "clinc"
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    model, tokenizer, device, texts, labels = load_model_and_data(model_id, dataset)
    n_layers = model.config.num_hidden_layers

    print(f"\nModel: {model_id} ({n_layers} layers)")
    print(f"Dataset: {dataset} ({len(texts)} samples)")
    print(f"Alphas: {alphas}")
    print(f"Observables: kNN, participation_ratio, effective_rank, alignment/uniformity")

    all_results = {}
    observables_by_alpha = {
        "knn_acc": [],
        "participation_ratio": [],
        "effective_rank": [],
        "alignment": [],
        "uniformity": [],
    }

    for alpha in alphas:
        print(f"\n--- alpha = {alpha:.2f} ---")
        t0 = time.time()

        reps = extract_layer_representations(model, tokenizer, device, texts, alpha)
        obs = compute_observables(reps, labels, n_layers)
        elapsed = time.time() - t0
        print(f"  Computed {len(obs)} layers in {elapsed:.1f}s")
        sys.stdout.flush()

        # Fit profiles and extract summary stats
        for key in ["knn_acc", "participation_ratio", "effective_rank"]:
            fit = fit_gaussian_profile(obs, key)
            print(f"  {key}: beta={fit['beta']:.3f}, mu={fit['mu']:.3f}, R2={fit['r2']:.3f}")

        # Store mean across layers (as summary statistic)
        layer_vals = {key: [] for key in observables_by_alpha}
        for layer_idx in sorted(obs.keys()):
            for key in observables_by_alpha:
                layer_vals[key].append(obs[layer_idx][key])

        for key in observables_by_alpha:
            if layer_vals[key]:
                observables_by_alpha[key].append(np.mean(layer_vals[key]))
            else:
                observables_by_alpha[key].append(np.nan)

        all_results[str(alpha)] = {
            "profile": obs,
            "fits": {
                key: fit_gaussian_profile(obs, key)
                for key in ["knn_acc", "participation_ratio", "effective_rank"]
            },
        }

    # ============================================================
    # TRANSITION ANALYSIS: Do all observables agree on alpha*?
    # ============================================================
    print(f"\n{'='*70}")
    print("MULTI-OBSERVABLE TRANSITION ANALYSIS")
    print(f"{'='*70}")

    alphas_arr = np.array(alphas)
    transition_points = {}

    for key, values in observables_by_alpha.items():
        values = np.array(values)
        if np.all(np.isnan(values)):
            continue

        # Normalize
        v_min, v_max = np.nanmin(values), np.nanmax(values)
        if v_max - v_min < 1e-10:
            continue
        v_norm = (values - v_min) / (v_max - v_min)

        # Find alpha_50
        alpha_50 = None
        for i in range(len(alphas_arr) - 1):
            if not np.isnan(v_norm[i]) and not np.isnan(v_norm[i + 1]):
                if v_norm[i] <= 0.5 and v_norm[i + 1] > 0.5:
                    frac = (0.5 - v_norm[i]) / (v_norm[i + 1] - v_norm[i])
                    alpha_50 = alphas_arr[i] + frac * (alphas_arr[i + 1] - alphas_arr[i])
                    break
        if alpha_50 is None and not np.all(np.isnan(v_norm)):
            # Check if decreasing
            for i in range(len(alphas_arr) - 1):
                if not np.isnan(v_norm[i]) and not np.isnan(v_norm[i + 1]):
                    if v_norm[i] >= 0.5 and v_norm[i + 1] < 0.5:
                        frac = (0.5 - v_norm[i]) / (v_norm[i + 1] - v_norm[i])
                        alpha_50 = alphas_arr[i] + frac * (alphas_arr[i + 1] - alphas_arr[i])
                        break

        transition_points[key] = alpha_50
        print(f"  {key:25s}: alpha_50 = {f'{alpha_50:.3f}' if alpha_50 is not None else 'N/A':>8}  "
              f"range = {v_max - v_min:.4f}")

    # Check consistency
    valid_transitions = {k: v for k, v in transition_points.items() if v is not None}
    if len(valid_transitions) >= 2:
        values = list(valid_transitions.values())
        mean_alpha = np.mean(values)
        std_alpha = np.std(values)
        print(f"\n  Mean alpha* across observables: {mean_alpha:.3f} +/- {std_alpha:.3f}")
        print(f"  Coefficient of variation: {100 * std_alpha / mean_alpha:.1f}%")
        if std_alpha / mean_alpha < 0.15:
            print("  CONSISTENT: All observables transition at the same alpha*!")
        else:
            print("  MIXED: Some observables disagree on transition point")

    # Save
    out = {
        "model_id": model_id,
        "dataset": dataset,
        "num_layers": n_layers,
        "alphas": alphas,
        "observables_summary": {
            key: [float(v) if not np.isnan(v) else None for v in vals]
            for key, vals in observables_by_alpha.items()
        },
        "transition_points": {
            k: float(v) if v is not None else None
            for k, v in transition_points.items()
        },
        "per_alpha_results": all_results,
    }

    out_path = RESULTS_DIR / "cti_multi_observable.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
