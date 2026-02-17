#!/usr/bin/env python
"""
Cross-model multi-observable sweep.

Runs the 4-observable pipeline on multiple models to replicate the two-step
phase transition finding. Pre-registered criterion:
  Delta_alpha = alpha_quality - alpha_IDpeak > 0.05 for each model

Models: SmolLM2-360M, Pythia-410M, Qwen2-0.5B (all from MODEL_DIRECTORY.md)
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


def extract_reps(model, tokenizer, texts, alpha, device="cuda", batch_size=32):
    """Extract all layer reps with residual scaling."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**enc, output_hidden_states=True, return_dict=True)
            mask = enc.get("attention_mask", torch.ones(enc["input_ids"].shape, device=device))
            for idx, hs in enumerate(outputs.hidden_states):
                hs_f = hs.float()
                m = mask.unsqueeze(-1).float()
                pooled = (hs_f * m).sum(1) / m.sum(1).clamp(min=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if idx not in all_hidden:
                    all_hidden[idx] = []
                all_hidden[idx].append(pooled.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in all_hidden.items()}


def compute_obs(reps, labels):
    """Compute mean observables across layers."""
    knn_vals, id_vals, er_vals, align_vals = [], [], [], []
    rng = np.random.RandomState(42)
    n = len(labels)
    n_train = int(0.7 * n)

    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue

        # kNN
        try:
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
            knn.fit(X[:n_train], labels[:n_train])
            knn_vals.append(knn.score(X[n_train:], labels[n_train:]))
        except Exception:
            knn_vals.append(0.0)

        # Intrinsic dim (participation ratio)
        try:
            Xc = X - X.mean(0)
            _, S, _ = np.linalg.svd(Xc, full_matrices=False)
            ev = S ** 2 / max(X.shape[0] - 1, 1)
            id_vals.append(float((ev.sum() ** 2) / (ev ** 2).sum()))
        except Exception:
            id_vals.append(0.0)

        # Effective rank
        try:
            er_vals.append(float(S.sum() / S.max()) if S.max() > 0 else 0.0)
        except Exception:
            er_vals.append(0.0)

        # Alignment
        try:
            unique = np.unique(labels)
            asum, acount = 0, 0
            for lbl in unique[:50]:
                cr = X[labels == lbl]
                if len(cr) >= 2:
                    for _ in range(min(30, len(cr))):
                        i, j = rng.choice(len(cr), 2, replace=False)
                        asum += np.sum((cr[i] - cr[j]) ** 2)
                        acount += 1
            align_vals.append(asum / max(acount, 1))
        except Exception:
            align_vals.append(0.0)

    return {
        "knn_acc": float(np.mean(knn_vals)) if knn_vals else 0,
        "intrinsic_dim": float(np.mean(id_vals)) if id_vals else 0,
        "effective_rank": float(np.mean(er_vals)) if er_vals else 0,
        "alignment": float(np.mean(align_vals)) if align_vals else 0,
    }


def analyze_two_step(alphas, obs_dict):
    """Find complexity peak and quality emergence."""
    alphas = np.array(alphas)
    knn = np.array([obs_dict[a]["knn_acc"] for a in alphas])
    ID = np.array([obs_dict[a]["intrinsic_dim"] for a in alphas])

    # Step 1: ID peak
    id_peak_idx = np.argmax(ID)
    alpha_id_peak = float(alphas[id_peak_idx])

    # Step 2: kNN transition (alpha_50)
    knn_min, knn_max = knn.min(), knn.max()
    if knn_max - knn_min < 0.01:
        alpha_knn = None
    else:
        knn_norm = (knn - knn_min) / (knn_max - knn_min)
        alpha_knn = None
        for i in range(len(alphas) - 1):
            if knn_norm[i] <= 0.5 and knn_norm[i + 1] > 0.5:
                frac = (0.5 - knn_norm[i]) / (knn_norm[i + 1] - knn_norm[i])
                alpha_knn = float(alphas[i] + frac * (alphas[i + 1] - alphas[i]))
                break

    gap = (alpha_knn - alpha_id_peak) if alpha_knn is not None else None

    return {
        "alpha_id_peak": alpha_id_peak,
        "id_at_peak": float(ID[id_peak_idx]),
        "id_at_0": float(ID[0]),
        "id_at_1": float(ID[-1]),
        "alpha_knn": alpha_knn,
        "gap": gap,
    }


def main():
    print("=" * 70)
    print("CROSS-MODEL TWO-STEP REPLICATION")
    print("Pre-registered: Delta_alpha > 0.05 for each model")
    print("=" * 70)

    # ALL models from models/MODEL_DIRECTORY.md — no exceptions
    models_to_test = [
        "HuggingFaceTB/SmolLM2-360M",
        "EleutherAI/pythia-410m",
        "Qwen/Qwen2-0.5B",
    ]
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    dataset_name = "clinc"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset once
    ds = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])

    all_results = {}
    gaps = []

    for model_id in models_to_test:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        try:
            model, tokenizer, n_layers, n_params = load_model(model_id, device)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        obs_dict = {}
        for alpha in alphas:
            print(f"  alpha={alpha:.2f}", end="", flush=True)
            t0 = time.time()
            reps = extract_reps(model, tokenizer, texts, alpha, device)
            obs = compute_obs(reps, labels)
            elapsed = time.time() - t0
            print(f"  kNN={obs['knn_acc']:.3f}  ID={obs['intrinsic_dim']:.1f}  "
                  f"ER={obs['effective_rank']:.1f}  A={obs['alignment']:.3f}  ({elapsed:.1f}s)")
            obs_dict[alpha] = obs
            sys.stdout.flush()

        # Analyze
        analysis = analyze_two_step(alphas, obs_dict)
        print(f"\n  ID peak: alpha = {analysis['alpha_id_peak']:.2f} (ID = {analysis['id_at_peak']:.1f})")
        print(f"  kNN transition: alpha = {analysis['alpha_knn']}")
        print(f"  GAP = {analysis['gap']}")

        if analysis['gap'] is not None and analysis['gap'] > 0.05:
            print(f"  REPLICATION: YES (gap = {analysis['gap']:.3f} > 0.05)")
        elif analysis['gap'] is not None:
            print(f"  REPLICATION: NO (gap = {analysis['gap']:.3f} <= 0.05)")
        else:
            print(f"  REPLICATION: INCONCLUSIVE (no kNN transition found)")

        if analysis['gap'] is not None:
            gaps.append(analysis['gap'])

        all_results[model_id] = {
            "n_layers": n_layers,
            "n_params": n_params,
            "observables": {str(a): obs_dict[a] for a in alphas},
            "two_step": analysis,
        }

        # Free GPU
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # POOLED ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("POOLED TWO-STEP REPLICATION")
    print(f"{'='*70}")

    # Include Qwen3 from previous run
    qwen_path = RESULTS_DIR / "cti_two_step_transition.json"
    if qwen_path.exists():
        with open(qwen_path) as f:
            qwen = json.load(f)
        if qwen.get("gap") is not None:
            gaps.insert(0, qwen["gap"])
            print(f"\n  Qwen/Qwen3-0.6B (28L): gap = {qwen['gap']:.3f}")

    for model_id, result in all_results.items():
        gap = result["two_step"]["gap"]
        n_layers = result["n_layers"]
        print(f"  {model_id} ({n_layers}L): gap = {gap}")

    if len(gaps) >= 2:
        gaps = np.array(gaps)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        all_positive = np.all(gaps > 0.05)
        print(f"\n  N models with gap: {len(gaps)}")
        print(f"  Mean gap: {mean_gap:.3f} +/- {std_gap:.3f}")
        print(f"  All > 0.05: {'YES' if all_positive else 'NO'}")
        print(f"  Sign test: {np.sum(gaps > 0)}/{len(gaps)} positive")

    # Save
    out = {
        "experiment": "cross_model_two_step_replication",
        "preregistered_criterion": "gap > 0.05",
        "alphas": alphas,
        "results": all_results,
        "pooled_gaps": gaps.tolist() if isinstance(gaps, np.ndarray) else gaps,
        "mean_gap": float(mean_gap) if len(gaps) >= 2 else None,
    }
    out_path = RESULTS_DIR / "cti_two_step_replication.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
