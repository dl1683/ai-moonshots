#!/usr/bin/env python
"""
CAUSAL INTERVENTION do(ID): Does manipulating intrinsic dimensionality
causally shift quality onset?

Core test of the information bottleneck interpretation:
- If compression MUST precede quality, then:
  - Forcing ID DOWN (compression) should bring quality onset EARLIER
  - Forcing ID UP (expansion) should push quality onset LATER

Method:
1. At each alpha, extract layer representations
2. Apply PCA projection to reduce ID (do(ID-))
3. Add structured noise to increase ID (do(ID+))
4. Measure kNN quality on modified representations
5. Compare quality curves with and without intervention

Pre-registered criterion:
  - do(ID-) should shift kNN transition alpha LEFT by > 0.02
  - do(ID+) should shift kNN transition alpha RIGHT by > 0.02
  - If neither: bottleneck interpretation is epiphenomenal

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


def compute_id(X):
    """Compute intrinsic dimensionality via participation ratio."""
    try:
        Xc = X - X.mean(0)
        # Check for degenerate representations (all same or NaN)
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


def compress_representations(X, target_frac=0.5):
    """Reduce ID by projecting onto top-k principal components.

    target_frac: keep enough components to explain this fraction of variance.
    This reduces ID while preserving the most informative directions.
    """
    try:
        Xc = X - X.mean(0)
        if np.isnan(Xc).any() or np.std(Xc) < 1e-10:
            return X.copy(), 0
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        ev = S ** 2
        total = ev.sum()
        if total < 1e-20:
            return X.copy(), 0
        cumvar = np.cumsum(ev) / total
        k = max(1, int(np.searchsorted(cumvar, target_frac) + 1))
        X_proj = Xc @ Vt[:k].T @ Vt[:k] + X.mean(0)
        norms = np.linalg.norm(X_proj, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return X_proj / norms, k
    except np.linalg.LinAlgError:
        return X.copy(), 0


def expand_representations(X, noise_scale=0.3, rng=None):
    """Increase ID by adding noise in underutilized dimensions.

    Adds Gaussian noise scaled to activate low-variance directions,
    effectively increasing the participation ratio.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    try:
        Xc = X - X.mean(0)
        if np.isnan(Xc).any() or np.std(Xc) < 1e-10:
            return X.copy()
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        weights = 1.0 / (S + 1e-8)
        weights = weights / weights.max()
        noise = rng.randn(X.shape[0], len(S))
        noise = noise * weights[None, :] * noise_scale
        X_noisy = X + noise @ Vt
        norms = np.linalg.norm(X_noisy, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return X_noisy / norms
    except np.linalg.LinAlgError:
        return X.copy()


def compute_knn(X, labels, n_train_frac=0.7):
    """Compute kNN accuracy."""
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


def compute_mean_obs(reps, labels):
    """Compute mean kNN and ID across all layers."""
    knn_vals, id_vals = [], []
    n_train = int(0.7 * len(labels))

    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue
        knn_vals.append(compute_knn(X, labels))
        id_vals.append(compute_id(X))

    return {
        "knn_acc": float(np.mean(knn_vals)) if knn_vals else 0,
        "intrinsic_dim": float(np.mean(id_vals)) if id_vals else 0,
    }


def intervene_on_reps(reps, labels, intervention, **kwargs):
    """Apply intervention to all layer representations and measure observables."""
    knn_vals, id_vals = [], []

    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue

        if intervention == "compress":
            X_mod, _ = compress_representations(X, **kwargs)
        elif intervention == "expand":
            X_mod = expand_representations(X, **kwargs)
        else:
            X_mod = X

        knn_vals.append(compute_knn(X_mod, labels))
        id_vals.append(compute_id(X_mod))

    return {
        "knn_acc": float(np.mean(knn_vals)) if knn_vals else 0,
        "intrinsic_dim": float(np.mean(id_vals)) if id_vals else 0,
    }


def find_knn_transition(alphas, knn_values):
    """Find alpha where kNN crosses 50% of its range."""
    alphas = np.array(alphas)
    knn = np.array(knn_values)
    knn_min, knn_max = knn.min(), knn.max()
    if knn_max - knn_min < 0.01:
        return None
    knn_norm = (knn - knn_min) / (knn_max - knn_min)
    for i in range(len(alphas) - 1):
        if knn_norm[i] <= 0.5 and knn_norm[i + 1] > 0.5:
            frac = (0.5 - knn_norm[i]) / (knn_norm[i + 1] - knn_norm[i])
            return float(alphas[i] + frac * (alphas[i + 1] - alphas[i]))
    return None


def main():
    print("=" * 70)
    print("CAUSAL INTERVENTION do(ID)")
    print("Does manipulating intrinsic dimensionality shift quality onset?")
    print("=" * 70)

    # Use Qwen2-0.5B (clearest gap: ID peak=0.70, kNN=0.86, gap=0.160)
    model_id = "Qwen/Qwen2-0.5B"  # From MODEL_DIRECTORY.md
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fine-grained alpha sweep through the gap region
    alphas = [0.0, 0.3, 0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

    # Load dataset
    ds = load_hierarchical_dataset("clinc", split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])

    # Load model
    model, tokenizer, n_layers, n_params = load_model(model_id, device)

    # Three conditions: natural, do(ID-), do(ID+)
    conditions = {
        "natural": {},
        "compress_50pct": {"intervention": "compress", "target_frac": 0.5},
        "compress_30pct": {"intervention": "compress", "target_frac": 0.3},
        "expand_0.3": {"intervention": "expand", "noise_scale": 0.3},
        "expand_0.5": {"intervention": "expand", "noise_scale": 0.5},
    }

    all_results = {}
    rng = np.random.RandomState(42)

    for alpha in alphas:
        print(f"\nalpha={alpha:.2f}", flush=True)
        t0 = time.time()

        # Extract representations once per alpha
        reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)

        # Natural (no intervention)
        natural_obs = compute_mean_obs(reps, labels)
        elapsed = time.time() - t0
        print(f"  natural:      kNN={natural_obs['knn_acc']:.3f}  ID={natural_obs['intrinsic_dim']:.1f}  ({elapsed:.1f}s)")

        result = {"natural": natural_obs}

        # Apply interventions
        for cond_name, cond_params in conditions.items():
            if cond_name == "natural":
                continue
            t1 = time.time()
            if "noise_scale" in cond_params:
                obs = intervene_on_reps(reps, labels, cond_params["intervention"],
                                        noise_scale=cond_params["noise_scale"], rng=rng)
            elif "target_frac" in cond_params:
                obs = intervene_on_reps(reps, labels, cond_params["intervention"],
                                        target_frac=cond_params["target_frac"])
            else:
                obs = natural_obs
            elapsed = time.time() - t1
            print(f"  {cond_name:15s}: kNN={obs['knn_acc']:.3f}  ID={obs['intrinsic_dim']:.1f}  ({elapsed:.1f}s)")
            result[cond_name] = obs

        all_results[str(alpha)] = result
        sys.stdout.flush()

    # ============================================================
    # ANALYSIS: Does intervention shift kNN transition?
    # ============================================================
    print(f"\n{'='*70}")
    print("CAUSAL ANALYSIS: kNN TRANSITION SHIFT")
    print(f"{'='*70}")

    alpha_list = [float(a) for a in alphas]
    transitions = {}

    for cond_name in conditions:
        knn_vals = [all_results[str(a)].get(cond_name, all_results[str(a)]["natural"])["knn_acc"]
                    for a in alphas]
        id_vals = [all_results[str(a)].get(cond_name, all_results[str(a)]["natural"])["intrinsic_dim"]
                   for a in alphas]

        alpha_knn = find_knn_transition(alpha_list, knn_vals)
        id_peak_idx = np.argmax(id_vals)
        alpha_id_peak = alpha_list[id_peak_idx]

        transitions[cond_name] = {
            "alpha_knn": alpha_knn,
            "alpha_id_peak": alpha_id_peak,
            "peak_id": float(id_vals[id_peak_idx]),
            "knn_at_1": float(knn_vals[-1]),
        }

        gap = (alpha_knn - alpha_id_peak) if alpha_knn is not None else None
        print(f"\n  {cond_name:15s}: ID peak={alpha_id_peak:.2f} (ID={id_vals[id_peak_idx]:.1f}), "
              f"kNN trans={alpha_knn}, gap={gap}")

    # ============================================================
    # KEY TEST: Did compression shift kNN onset LEFT?
    # ============================================================
    print(f"\n{'='*70}")
    print("PRE-REGISTERED TEST: DOES do(ID) SHIFT kNN TRANSITION?")
    print(f"{'='*70}")

    nat_trans = transitions["natural"]["alpha_knn"]
    if nat_trans is None:
        print("\n  INCONCLUSIVE: No natural kNN transition found")
    else:
        print(f"\n  Natural kNN transition: alpha = {nat_trans:.3f}")

        for cond_name in ["compress_50pct", "compress_30pct", "expand_0.3", "expand_0.5"]:
            cond_trans = transitions.get(cond_name, {}).get("alpha_knn")
            if cond_trans is not None:
                shift = cond_trans - nat_trans
                direction = "LEFT (earlier)" if shift < 0 else "RIGHT (later)"
                sig = abs(shift) > 0.02
                print(f"  {cond_name:15s}: alpha = {cond_trans:.3f}  shift = {shift:+.3f}  "
                      f"{direction}  {'SIGNIFICANT' if sig else 'not significant'}")

                # Check if direction matches prediction
                if "compress" in cond_name:
                    correct = shift < -0.02
                    print(f"    Prediction (compress -> LEFT): {'CONFIRMED' if correct else 'FAILED'}")
                elif "expand" in cond_name:
                    correct = shift > 0.02
                    print(f"    Prediction (expand -> RIGHT): {'CONFIRMED' if correct else 'FAILED'}")
            else:
                print(f"  {cond_name:15s}: No transition found")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    compress_shifts = []
    expand_shifts = []
    for cond_name, t in transitions.items():
        if t["alpha_knn"] is None or nat_trans is None:
            continue
        shift = t["alpha_knn"] - nat_trans
        if "compress" in cond_name:
            compress_shifts.append(shift)
        elif "expand" in cond_name:
            expand_shifts.append(shift)

    if compress_shifts and expand_shifts:
        mean_compress = np.mean(compress_shifts)
        mean_expand = np.mean(expand_shifts)
        print(f"\n  Mean compress shift: {mean_compress:+.3f}")
        print(f"  Mean expand shift:   {mean_expand:+.3f}")

        if mean_compress < -0.02 and mean_expand > 0.02:
            print("\n  CAUSAL MECHANISM CONFIRMED:")
            print("    Compression causally accelerates quality onset")
            print("    Expansion causally delays quality onset")
            print("    -> Information bottleneck interpretation is CAUSAL, not epiphenomenal")
        elif mean_compress < -0.02:
            print("\n  PARTIAL: Compression shifts onset left, but expansion effect unclear")
        elif mean_expand > 0.02:
            print("\n  PARTIAL: Expansion shifts onset right, but compression effect unclear")
        else:
            print("\n  EPIPHENOMENAL: ID manipulation does not causally shift quality onset")
            print("    -> Two-step pattern may be a measurement artifact")

    # Save results
    out = {
        "experiment": "causal_do_ID",
        "model_id": model_id,
        "alphas": alphas,
        "preregistered_criterion": "compress shifts kNN LEFT by >0.02, expand shifts RIGHT by >0.02",
        "results": all_results,
        "transitions": transitions,
    }
    if compress_shifts:
        out["mean_compress_shift"] = float(np.mean(compress_shifts))
    if expand_shifts:
        out["mean_expand_shift"] = float(np.mean(expand_shifts))

    out_path = RESULTS_DIR / "cti_causal_do_id.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
