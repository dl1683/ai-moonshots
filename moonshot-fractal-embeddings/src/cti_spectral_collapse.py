#!/usr/bin/env python
"""
SPECTRAL MARGIN-TO-NOISE PHASE TRANSITION: Universal kappa collapse.

Core hypothesis: quality onset occurs at a UNIVERSAL critical spectral
margin-to-noise ratio kappa_c, regardless of model, alpha, or intervention.

kappa = trace(S_B) / trace(S_W)
  S_B = between-class scatter (discriminative signal)
  S_W = within-class scatter (noise)

If all (model x alpha x intervention) curves collapse onto a single
kappa_c threshold when plotted as kNN vs kappa, this is a candidate
universal law of representation quality.

Pre-registered criterion:
  All kNN transitions occur at kappa_c within 20% relative range.

All models from MODEL_DIRECTORY.md.
"""

import json
import sys
import time
import gc
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model, ResidualScaler
from hierarchical_datasets import load_hierarchical_dataset


def extract_all_layer_reps(model, tokenizer, texts, alpha, device="cuda", batch_size=32):
    """Extract all layer representations with residual scaling."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**enc, output_hidden_states=True, return_dict=True)
            mask = enc.get("attention_mask",
                           torch.ones(enc["input_ids"].shape, device=device))
            for idx, hs in enumerate(outputs.hidden_states):
                hs_f = hs.float()
                m = mask.unsqueeze(-1).float()
                pooled = (hs_f * m).sum(1) / m.sum(1).clamp(min=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if idx not in all_hidden:
                    all_hidden[idx] = []
                all_hidden[idx].append(pooled.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in all_hidden.items()}


def compute_kappa(X, labels):
    """Compute spectral margin-to-noise ratio.

    kappa = trace(S_B) / trace(S_W)
    where S_B = between-class scatter, S_W = within-class scatter.

    Returns NaN-safe float (never inf).
    """
    try:
        if np.isnan(X).any():
            return 0.0

        unique_labels = np.unique(labels)
        grand_mean = X.mean(0)

        trace_sb = 0.0
        trace_sw = 0.0

        for lbl in unique_labels:
            mask = labels == lbl
            n_k = mask.sum()
            if n_k < 2:
                continue
            X_k = X[mask]
            mu_k = X_k.mean(0)
            trace_sb += n_k * np.sum((mu_k - grand_mean) ** 2)
            trace_sw += np.sum((X_k - mu_k) ** 2)

        if trace_sw < 1e-12:
            # Cap at a large but finite value
            return 100.0 if trace_sb > 0 else 0.0

        kappa = trace_sb / trace_sw
        # Cap to avoid inf contaminating means
        return float(min(kappa, 100.0))
    except Exception:
        return 0.0


def compute_knn(X, labels, n_train_frac=0.7):
    """kNN accuracy."""
    n = len(labels)
    n_train = int(n_train_frac * n)
    if n_train < 5 or n - n_train < 5:
        return 0.0
    try:
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X[:n_train], labels[:n_train])
        return float(knn.score(X[n_train:], labels[n_train:]))
    except Exception:
        return 0.0


def compute_id(X):
    """Participation ratio."""
    try:
        Xc = X - X.mean(0)
        if np.isnan(Xc).any() or np.std(Xc) < 1e-10:
            return 1.0
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        ev = S ** 2 / max(X.shape[0] - 1, 1)
        denom = (ev ** 2).sum()
        if denom < 1e-20:
            return 1.0
        return float((ev.sum() ** 2) / denom)
    except np.linalg.LinAlgError:
        return 1.0


def compute_mean_obs(reps, labels):
    """Compute mean kNN, kappa, and ID across all layers."""
    knn_vals, kappa_vals, id_vals = [], [], []

    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue
        knn_vals.append(compute_knn(X, labels))
        kappa_vals.append(compute_kappa(X, labels))
        id_vals.append(compute_id(X))

    # Filter out any remaining NaN/inf
    kappa_clean = [k for k in kappa_vals if np.isfinite(k)]
    return {
        "knn_acc": float(np.mean(knn_vals)) if knn_vals else 0,
        "kappa": float(np.mean(kappa_clean)) if kappa_clean else 0,
        "intrinsic_dim": float(np.mean(id_vals)) if id_vals else 0,
    }


def main():
    print("=" * 70)
    print("SPECTRAL MARGIN-TO-NOISE PHASE TRANSITION")
    print("Universal kappa collapse across models")
    print("=" * 70)

    # All models from MODEL_DIRECTORY.md
    models_to_test = [
        "Qwen/Qwen2-0.5B",
        "HuggingFaceTB/SmolLM2-360M",
        "EleutherAI/pythia-410m",
        "Qwen/Qwen3-0.6B",
    ]
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_hierarchical_dataset("clinc", split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])

    # Collect all (kappa, kNN) points
    all_points = []  # (model, alpha, kappa, knn)
    all_model_results = {}

    for model_id in models_to_test:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        try:
            model, tokenizer, n_layers, n_params = load_model(model_id, device)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        model_results = {}
        for alpha in alphas:
            print(f"  alpha={alpha:.2f}", end="", flush=True)
            t0 = time.time()
            reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)
            obs = compute_mean_obs(reps, labels)
            elapsed = time.time() - t0
            print(f"  kNN={obs['knn_acc']:.3f}  kappa={obs['kappa']:.4f}  "
                  f"ID={obs['intrinsic_dim']:.1f}  ({elapsed:.1f}s)")
            model_results[str(alpha)] = obs
            all_points.append({
                "model": model_id,
                "alpha": alpha,
                "kappa": obs["kappa"],
                "knn": obs["knn_acc"],
                "id": obs["intrinsic_dim"],
            })
            sys.stdout.flush()

        all_model_results[model_id] = model_results

        # Free GPU
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # COLLAPSE ANALYSIS: kNN vs kappa
    # ============================================================
    print(f"\n{'='*70}")
    print("COLLAPSE ANALYSIS: kNN vs kappa")
    print(f"{'='*70}")

    # Sort all points by kappa
    all_points.sort(key=lambda p: p["kappa"])

    print(f"\n  {'Model':>35} {'alpha':>6} {'kappa':>10} {'kNN':>8}")
    print(f"  {'-'*65}")
    for p in all_points:
        short = p["model"].split("/")[-1]
        print(f"  {short:>35} {p['alpha']:>6.2f} {p['kappa']:>10.4f} {p['knn']:>8.3f}")

    # Find kappa at which kNN transitions for each model
    print(f"\n  PER-MODEL kappa AT kNN TRANSITION:")
    kappa_at_transitions = []

    for model_id in models_to_test:
        model_pts = [p for p in all_points if p["model"] == model_id]
        if not model_pts:
            continue

        # Sort by alpha for transition finding
        model_pts_alpha = sorted(model_pts, key=lambda p: p["alpha"])
        knn_vals = [p["knn"] for p in model_pts_alpha]
        kappa_vals = [p["kappa"] for p in model_pts_alpha]

        knn_arr = np.array(knn_vals)
        knn_min, knn_max = knn_arr.min(), knn_arr.max()
        if knn_max - knn_min < 0.01:
            print(f"  {model_id}: no kNN transition")
            continue

        knn_norm = (knn_arr - knn_min) / (knn_max - knn_min)
        for i in range(len(knn_norm) - 1):
            if knn_norm[i] <= 0.5 and knn_norm[i + 1] > 0.5:
                frac = (0.5 - knn_norm[i]) / (knn_norm[i + 1] - knn_norm[i])
                kappa_trans = kappa_vals[i] + frac * (kappa_vals[i + 1] - kappa_vals[i])
                kappa_at_transitions.append({
                    "model": model_id,
                    "kappa_at_transition": float(kappa_trans),
                })
                short = model_id.split("/")[-1]
                print(f"  {short:>35}: kappa_c = {kappa_trans:.4f}")
                break

    if len(kappa_at_transitions) >= 2:
        kappas = np.array([k["kappa_at_transition"] for k in kappa_at_transitions])
        mean_kappa = np.mean(kappas)
        std_kappa = np.std(kappas)
        cv = std_kappa / mean_kappa if mean_kappa > 0 else float("inf")
        rel_range = (kappas.max() - kappas.min()) / mean_kappa if mean_kappa > 0 else float("inf")

        print(f"\n  UNIVERSALITY TEST:")
        print(f"    N models: {len(kappas)}")
        print(f"    Mean kappa_c: {mean_kappa:.4f}")
        print(f"    Std kappa_c: {std_kappa:.4f}")
        print(f"    CV: {cv:.2f}")
        print(f"    Relative range: {rel_range:.2f}")
        print(f"    Pre-registered: rel range < 0.20")

        if rel_range < 0.20:
            print(f"    UNIVERSAL COLLAPSE: YES (rel range {rel_range:.2f} < 0.20)")
            collapse = True
        else:
            print(f"    UNIVERSAL COLLAPSE: NO (rel range {rel_range:.2f} >= 0.20)")
            collapse = False
    else:
        print(f"\n  Insufficient models with transitions for universality test")
        collapse = False

    # ============================================================
    # Correlation: kappa vs kNN across ALL points
    # ============================================================
    print(f"\n{'='*70}")
    print("GLOBAL CORRELATION: kappa vs kNN")
    print(f"{'='*70}")

    from scipy.stats import spearmanr, pearsonr
    all_kappas = np.array([p["kappa"] for p in all_points])
    all_knns = np.array([p["knn"] for p in all_points])

    rho, p_rho = spearmanr(all_kappas, all_knns)
    r, p_r = pearsonr(all_kappas, all_knns)
    print(f"\n  Spearman rho = {rho:.4f} (p = {p_rho:.6f})")
    print(f"  Pearson r = {r:.4f} (p = {p_r:.6f})")

    if rho > 0.8 and p_rho < 0.001:
        print(f"  STRONG monotonic relationship: kappa determines kNN")
    elif rho > 0.5:
        print(f"  MODERATE relationship")
    else:
        print(f"  WEAK relationship")

    # ============================================================
    # Compare: kappa vs ID as predictors of kNN
    # ============================================================
    print(f"\n{'='*70}")
    print("PREDICTOR COMPARISON: kappa vs ID for kNN")
    print(f"{'='*70}")

    all_ids = np.array([p["id"] for p in all_points])
    rho_id, p_id = spearmanr(all_ids, all_knns)
    print(f"\n  kNN ~ kappa: rho = {rho:.4f} (p = {p_rho:.6f})")
    print(f"  kNN ~ ID:    rho = {rho_id:.4f} (p = {p_id:.6f})")
    print(f"  kappa is {'BETTER' if abs(rho) > abs(rho_id) else 'WORSE'} predictor of kNN than ID")

    # Save
    out = {
        "experiment": "spectral_margin_noise_phase_transition",
        "models": models_to_test,
        "alphas": alphas,
        "all_points": all_points,
        "model_results": all_model_results,
        "kappa_at_transitions": kappa_at_transitions,
        "global_correlation": {
            "knn_vs_kappa": {"rho": float(rho), "p": float(p_rho)},
            "knn_vs_id": {"rho": float(rho_id), "p": float(p_id)},
        },
    }
    if len(kappa_at_transitions) >= 2:
        out["universality"] = {
            "mean_kappa_c": float(mean_kappa),
            "std_kappa_c": float(std_kappa),
            "cv": float(cv),
            "relative_range": float(rel_range),
            "collapse": collapse,
        }

    out_path = RESULTS_DIR / "cti_spectral_collapse.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
