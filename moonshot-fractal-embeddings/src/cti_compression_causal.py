#!/usr/bin/env python
"""
CAUSAL INTERVENTION: Does compression cause quality improvement?

The two-step finding shows:
  Step 1 (alpha~0.7): ID peaks (complexity maximum)
  Step 2 (alpha~0.87): kNN rises (quality emergence)
  Gap = 0.17: compression precedes quality

Codex concern: Is this a measurement artifact (different response curves)
or a genuine causal relationship (compression CAUSES quality)?

CAUSAL TEST: At each alpha, artificially compress representations via PCA
to different target dimensions. If IB is correct:
  - At alpha in the gap (0.75-0.85): compression should HELP quality
    because representations are over-complex but not yet organized
  - At alpha=1.0: compression should HURT quality
    because representations are already optimally compressed
  - The OPTIMAL compression level should track the ID profile

Pre-registered criterion:
  kNN(compressed) > kNN(original) for at least one PCA rank
  at alpha in [0.7, 0.85] (gap region)
"""

import json
import sys
import time
import gc
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model, ResidualScaler
from hierarchical_datasets import load_hierarchical_dataset


def extract_all_layer_reps(model, tokenizer, texts, alpha, device="cuda", batch_size=32):
    """Extract per-layer representations with residual scaling."""
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


def knn_accuracy(X, labels, n_train_frac=0.7):
    """Compute kNN accuracy with train/test split."""
    n = len(labels)
    n_train = int(n_train_frac * n)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X[:n_train], labels[:n_train])
    return knn.score(X[n_train:], labels[n_train:])


def intrinsic_dim(X):
    """Participation ratio as intrinsic dimensionality."""
    Xc = X - X.mean(0)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2 / max(X.shape[0] - 1, 1)
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def compress_and_evaluate(X, labels, target_dims, n_train_frac=0.7):
    """Compress representations to various PCA dimensions, measure kNN."""
    n = len(labels)
    n_train = int(n_train_frac * n)
    results = {}

    # Original (no compression)
    acc_orig = knn_accuracy(X, labels, n_train_frac)
    id_orig = intrinsic_dim(X)
    results["original"] = {"knn": acc_orig, "id": id_orig, "dim": X.shape[1]}

    for d in target_dims:
        if d >= X.shape[1]:
            continue
        pca = PCA(n_components=d)
        X_compressed = pca.fit_transform(X)
        # Re-normalize for cosine kNN
        norms = np.linalg.norm(X_compressed, axis=1, keepdims=True)
        X_compressed = X_compressed / np.clip(norms, 1e-8, None)
        acc = knn_accuracy(X_compressed, labels, n_train_frac)
        id_c = intrinsic_dim(X_compressed)
        var_explained = float(pca.explained_variance_ratio_.sum())
        results[d] = {
            "knn": acc,
            "id": id_c,
            "dim": d,
            "var_explained": var_explained,
            "knn_delta": acc - acc_orig,
        }

    return results


def main():
    print("=" * 70)
    print("CAUSAL INTERVENTION: Compression -> Quality")
    print("Does artificial compression improve quality in the gap region?")
    print("=" * 70)

    model_id = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Alphas: before peak, at peak, in gap, at quality, past quality
    alphas = [0.0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    # PCA target dimensions: aggressive to mild compression
    target_dims = [4, 8, 16, 32, 64, 128, 256]

    # Load data
    ds = load_hierarchical_dataset("clinc", split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])
    print(f"Dataset: CLINC, {len(texts)} samples, {len(np.unique(labels))} classes")

    # Load model
    model, tokenizer, n_layers, n_params = load_model(model_id, device)

    all_results = {}

    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"ALPHA = {alpha:.2f}")
        print(f"{'='*70}")

        t0 = time.time()
        reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)
        elapsed = time.time() - t0
        print(f"  Extracted {len(reps)} layers in {elapsed:.1f}s")

        # Use MEAN across all layers (same as multi-observable analysis)
        # Also test best single layer
        n_layers_actual = len(reps)

        # Strategy 1: Mean-pooled across layers
        X_mean = np.mean([reps[i] for i in sorted(reps.keys())], axis=0)
        print(f"\n  --- Mean-pooled across {n_layers_actual} layers ---")
        comp_results_mean = compress_and_evaluate(X_mean, labels, target_dims)

        orig = comp_results_mean["original"]
        print(f"  Original: kNN={orig['knn']:.3f}, ID={orig['id']:.1f}, dim={orig['dim']}")

        best_delta = -999
        best_dim = None
        for d in target_dims:
            if d in comp_results_mean:
                r = comp_results_mean[d]
                marker = " ***" if r["knn_delta"] > 0.01 else ""
                print(f"  PCA-{d:>3}: kNN={r['knn']:.3f} (delta={r['knn_delta']:+.3f}), "
                      f"ID={r['id']:.1f}, var={r['var_explained']:.3f}{marker}")
                if r["knn_delta"] > best_delta:
                    best_delta = r["knn_delta"]
                    best_dim = d

        if best_delta > 0:
            print(f"  COMPRESSION HELPS: best at PCA-{best_dim} (delta={best_delta:+.3f})")
        else:
            print(f"  Compression does NOT help (best delta={best_delta:+.3f})")

        # Strategy 2: Best single layer (layer with highest kNN)
        best_layer_acc = -1
        best_layer_idx = 0
        for layer_idx in sorted(reps.keys()):
            acc = knn_accuracy(reps[layer_idx], labels)
            if acc > best_layer_acc:
                best_layer_acc = acc
                best_layer_idx = layer_idx

        print(f"\n  --- Best single layer: L{best_layer_idx} (kNN={best_layer_acc:.3f}) ---")
        X_best = reps[best_layer_idx]
        comp_results_best = compress_and_evaluate(X_best, labels, target_dims)

        best_delta_single = -999
        best_dim_single = None
        for d in target_dims:
            if d in comp_results_best:
                r = comp_results_best[d]
                marker = " ***" if r["knn_delta"] > 0.01 else ""
                print(f"  PCA-{d:>3}: kNN={r['knn']:.3f} (delta={r['knn_delta']:+.3f}){marker}")
                if r["knn_delta"] > best_delta_single:
                    best_delta_single = r["knn_delta"]
                    best_dim_single = d

        all_results[str(alpha)] = {
            "alpha": alpha,
            "mean_pooled": {
                "original_knn": orig["knn"],
                "original_id": orig["id"],
                "compression": {str(d): comp_results_mean[d] for d in target_dims
                               if d in comp_results_mean},
                "best_delta": best_delta,
                "best_dim": best_dim,
                "compression_helps": best_delta > 0,
            },
            "best_layer": {
                "layer": best_layer_idx,
                "original_knn": best_layer_acc,
                "compression": {str(d): comp_results_best[d] for d in target_dims
                               if d in comp_results_best},
                "best_delta": best_delta_single,
                "best_dim": best_dim_single,
                "compression_helps": best_delta_single > 0,
            },
        }
        sys.stdout.flush()

    # ============================================================
    # CAUSAL ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("CAUSAL ANALYSIS: Compression benefit across alpha")
    print(f"{'='*70}")

    print(f"\n{'alpha':>6} {'orig_kNN':>9} {'best_PCA':>9} {'delta':>8} {'helps':>6}")
    print("-" * 45)

    gap_helps = False
    tail_helps = False
    for alpha in alphas:
        r = all_results[str(alpha)]["mean_pooled"]
        print(f"{alpha:>6.2f} {r['original_knn']:>9.3f} "
              f"{r['original_knn']+r['best_delta']:>9.3f} "
              f"{r['best_delta']:>+8.3f} {'YES' if r['compression_helps'] else 'no':>6}")
        if 0.7 <= alpha <= 0.85 and r["compression_helps"]:
            gap_helps = True
        if alpha >= 0.95 and r["compression_helps"]:
            tail_helps = True

    print(f"\n  Pre-registered test:")
    print(f"    Compression helps in gap [0.7, 0.85]: {'YES' if gap_helps else 'NO'}")
    print(f"    Compression helps at alpha>=0.95:     {'YES' if tail_helps else 'NO'}")

    if gap_helps and not tail_helps:
        print(f"    STRONG IB SUPPORT: compression helps in gap but not at convergence")
    elif gap_helps and tail_helps:
        print(f"    WEAK IB: compression helps everywhere (may be artifact)")
    elif not gap_helps:
        print(f"    IB NOT SUPPORTED: compression doesn't help in gap")

    # Save
    out = {
        "experiment": "compression_causal_intervention",
        "model": model_id,
        "dataset": "clinc",
        "alphas": alphas,
        "target_dims": target_dims,
        "results": all_results,
        "causal_test": {
            "gap_compression_helps": gap_helps,
            "tail_compression_helps": tail_helps,
            "ib_supported": gap_helps and not tail_helps,
        },
    }
    out_path = RESULTS_DIR / "cti_compression_causal.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
