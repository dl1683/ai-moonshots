#!/usr/bin/env python
"""
CAUSAL CONTROL: SPECTRAL WHITENING INTERVENTION GUIDED BY THEORY

If low eta (anisotropy) hurts kNN quality, then POST-HOC whitening
representations should improve it. This tests:

1. Does whitening improve kNN quality? (causal test of eta's role)
2. Is the improvement PREDICTED by the current eta?
   (low eta → more benefit from whitening)
3. Does the theory predict the optimal whitening strength?
4. Does dist_ratio (not just kNN) improve with whitening?
   (rules out tautological correlation)
5. Linear probe accuracy as independent verification

If whitening helps where theory predicts, we have a practical
intervention guided by theoretical understanding.

All models from MODEL_DIRECTORY.md.
"""

import json
import sys
import gc
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model, ResidualScaler
from hierarchical_datasets import load_hierarchical_dataset


def sigmoid(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def extract_reps(model, tokenizer, texts, alpha=1.0, device="cuda", batch_size=32):
    """Extract penultimate layer representations."""
    all_reps = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, return_dict=True)
            mask = enc.get("attention_mask",
                           torch.ones(enc["input_ids"].shape, device=device))
            hs = out.hidden_states[-2].float()
            m = mask.unsqueeze(-1).float()
            pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            all_reps.append(pooled.cpu().numpy())

    return np.concatenate(all_reps, axis=0)


def partial_whiten(X, strength=1.0, n_components=None):
    """Apply partial spectral whitening.

    strength=0: no whitening (original)
    strength=1: full whitening (all eigenvalues equalized)
    In between: partial whitening (eigenvalues partially equalized)
    """
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Keep only significant components (threshold at 1e-6 * max)
    threshold = S[0] * 1e-6
    keep = S > threshold
    S = S[keep]
    Vt = Vt[keep]
    U = U[:, keep]

    if n_components is not None:
        n_components = min(n_components, len(S))
        S = S[:n_components]
        Vt = Vt[:n_components]
        U = U[:, :n_components]

    # Partial whitening: scale eigenvalues toward uniform
    # scaling[i] = S[i]^(-strength) means:
    #   strength=0: no change
    #   strength=1: full whitening (divide by S)
    scaling = S ** (-strength)

    # Apply: project, scale, reconstruct
    # X_centered @ Vt.T gives projection coefficients (divided by S via SVD)
    # Actually, U * S gives the projection coefficients
    coeffs = U * S  # n x k
    coeffs_whitened = coeffs * scaling  # scale each component

    X_whitened = coeffs_whitened @ Vt
    X_whitened = X_whitened + mean

    # Replace any NaN/Inf
    X_whitened = np.nan_to_num(X_whitened, nan=0.0, posinf=0.0, neginf=0.0)

    # Re-normalize
    norms = np.linalg.norm(X_whitened, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_whitened = X_whitened / norms

    return X_whitened


def compute_metrics(X, labels):
    """Compute kappa, eta, kNN, dist_ratio, linear probe accuracy."""
    classes = np.unique(labels)
    K = len(classes)
    d = X.shape[1]
    grand_mean = X.mean(axis=0)

    # kappa and eta
    S_W_trace = 0.0
    S_B_trace = 0.0
    within_sq_frob = 0.0

    for c in classes:
        X_c = X[labels == c]
        n_c = len(X_c)
        class_mean = X_c.mean(axis=0)
        diff = X_c - class_mean
        S_W_trace += np.sum(diff ** 2)
        mean_diff = class_mean - grand_mean
        S_B_trace += n_c * np.sum(mean_diff ** 2)

        if n_c > 1:
            centered = diff / np.sqrt(n_c)
            gram = centered.T @ centered
            within_sq_frob += np.sum(gram ** 2)

    n = len(X)
    kappa = S_B_trace / max(S_W_trace, 1e-10)
    trace_sw = S_W_trace / n
    trace_sw2 = within_sq_frob / n
    eta = trace_sw ** 2 / max(d * trace_sw2, 1e-10)

    # kNN
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, labels)
    _, indices = knn.kneighbors(X)
    correct = sum(1 for i in range(len(X)) if labels[indices[i, 1]] == labels[i])
    knn_acc = correct / len(X)
    q = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)

    # dist_ratio (subsample for speed)
    n_sample = min(500, len(X))
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(len(X), n_sample, replace=False)
    same_dists = []
    diff_dists = []
    for i in idx_sample:
        dists = np.sum((X - X[i]) ** 2, axis=1)
        dists[i] = np.inf
        same_mask = labels == labels[i]
        diff_mask = labels != labels[i]
        same_mask[i] = False
        if same_mask.any():
            same_dists.append(np.min(dists[same_mask]))
        if diff_mask.any():
            diff_dists.append(np.min(dists[diff_mask]))

    dist_ratio = float(np.mean(diff_dists) / max(np.mean(same_dists), 1e-10))

    # Linear probe (5-fold CV)
    try:
        lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                 multi_class='multinomial', n_jobs=-1)
        lr_scores = cross_val_score(lr, X, labels, cv=5, scoring='accuracy')
        lr_acc = float(lr_scores.mean())
    except Exception:
        lr_acc = 0.0

    return {
        "kappa": float(kappa),
        "eta": float(eta),
        "knn": float(knn_acc),
        "q": float(q),
        "dist_ratio": float(dist_ratio),
        "lr_acc": float(lr_acc),
        "K": int(K),
    }


def main():
    print("=" * 70)
    print("CAUSAL CONTROL: SPECTRAL WHITENING INTERVENTION")
    print("=" * 70)

    # Load dataset
    ds_name = "clinc"
    print(f"\nLoading {ds_name}...")
    ds = load_hierarchical_dataset(ds_name, split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels_raw = np.array([s.level1_label for s in ds.samples])
    unique_labels = np.unique(labels_raw)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels_raw])
    K = len(unique_labels)
    print(f"  {len(texts)} samples, K={K}")

    # Test models
    model_configs = [
        ("EleutherAI/pythia-160m", "pythia-160m"),
        ("EleutherAI/pythia-410m", "pythia-410m"),
        ("Qwen/Qwen2-0.5B", "Qwen2-0.5B"),
    ]

    alphas = [0.3, 0.5, 0.7, 1.0]
    whitening_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_experiment_results = []

    for model_id, model_name in model_configs:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        model, tokenizer, _, _ = load_model(model_id, device="cuda")

        for alpha in alphas:
            print(f"\n  alpha={alpha}:")
            X = extract_reps(model, tokenizer, texts, alpha=alpha)

            # Check for NaN in raw reps
            if np.any(np.isnan(X)):
                print(f"    WARNING: NaN in raw reps, replacing with 0")
                X = np.nan_to_num(X, nan=0.0)
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                X = X / norms

            for ws in whitening_strengths:
                if ws == 0:
                    X_proc = X.copy()
                else:
                    X_proc = partial_whiten(X, strength=ws)

                # Safety check
                if np.any(np.isnan(X_proc)) or np.any(np.isinf(X_proc)):
                    print(f"    w={ws:.2f}: NaN/Inf after whitening, skipping")
                    continue

                metrics = compute_metrics(X_proc, labels)

                result = {
                    "model": model_name,
                    "alpha": alpha,
                    "whitening": ws,
                    **metrics,
                }
                all_experiment_results.append(result)

                tag = " <-- raw" if ws == 0 else ""
                print(f"    w={ws:.2f}: kNN={metrics['knn']:.4f}, "
                      f"LR={metrics['lr_acc']:.4f}, eta={metrics['eta']:.6f}, "
                      f"dr={metrics['dist_ratio']:.4f}{tag}")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("ANALYSIS: DOES WHITENING IMPROVE QUALITY?")
    print(f"{'='*70}")

    # For each (model, alpha), find optimal whitening strength
    improvements = []
    for model_name in set(r["model"] for r in all_experiment_results):
        for alpha in alphas:
            subset = [r for r in all_experiment_results
                      if r["model"] == model_name and r["alpha"] == alpha]
            if not subset:
                continue

            raw = [r for r in subset if r["whitening"] == 0.0]
            if not raw:
                continue
            raw = raw[0]

            best_knn = max(subset, key=lambda r: r["knn"])
            best_lr = max(subset, key=lambda r: r["lr_acc"])

            knn_gain = best_knn["knn"] - raw["knn"]
            lr_gain = best_lr["lr_acc"] - raw["lr_acc"]

            improvements.append({
                "model": model_name,
                "alpha": alpha,
                "raw_eta": raw["eta"],
                "raw_knn": raw["knn"],
                "best_knn": best_knn["knn"],
                "best_knn_w": best_knn["whitening"],
                "knn_gain": knn_gain,
                "raw_lr": raw["lr_acc"],
                "best_lr": best_lr["lr_acc"],
                "best_lr_w": best_lr["whitening"],
                "lr_gain": lr_gain,
            })

            print(f"  {model_name} a={alpha}: "
                  f"kNN +{knn_gain:+.4f} (w={best_knn['whitening']:.2f}), "
                  f"LR +{lr_gain:+.4f} (w={best_lr['whitening']:.2f}), "
                  f"raw_eta={raw['eta']:.6f}")

    # Test: does low eta predict more benefit from whitening?
    print(f"\n{'='*70}")
    print("CORRELATION: raw_eta vs whitening benefit")
    print(f"{'='*70}")

    raw_etas = np.array([r["raw_eta"] for r in improvements])
    knn_gains = np.array([r["knn_gain"] for r in improvements])
    lr_gains = np.array([r["lr_gain"] for r in improvements])

    if len(raw_etas) >= 5:
        rho_knn, p_knn = spearmanr(raw_etas, knn_gains)
        rho_lr, p_lr = spearmanr(raw_etas, lr_gains)
        print(f"  eta vs kNN_gain: rho={rho_knn:.4f} (p={p_knn:.4f})")
        print(f"  eta vs LR_gain: rho={rho_lr:.4f} (p={p_lr:.4f})")
        print(f"  Theory predicts: low eta -> more benefit (negative correlation)")
    else:
        rho_knn = rho_lr = 0.0

    # Test: does dist_ratio predict linear probe accuracy?
    print(f"\n{'='*70}")
    print("NON-TAUTOLOGICAL TEST: dist_ratio predicts linear probe?")
    print(f"{'='*70}")

    drs = np.array([r["dist_ratio"] for r in all_experiment_results])
    lr_accs = np.array([r["lr_acc"] for r in all_experiment_results])
    knn_accs = np.array([r["knn"] for r in all_experiment_results])
    kappas = np.array([r["kappa"] for r in all_experiment_results])

    rho_dr_lr, _ = spearmanr(drs, lr_accs)
    rho_kappa_lr, _ = spearmanr(kappas, lr_accs)
    rho_dr_knn, _ = spearmanr(drs, knn_accs)
    rho_kappa_knn, _ = spearmanr(kappas, knn_accs)

    print(f"  dist_ratio -> LR_acc: rho={rho_dr_lr:.4f}")
    print(f"  kappa -> LR_acc: rho={rho_kappa_lr:.4f}")
    print(f"  dist_ratio -> kNN_acc: rho={rho_dr_knn:.4f}")
    print(f"  kappa -> kNN_acc: rho={rho_kappa_knn:.4f}")
    print(f"  dist_ratio beats kappa for LR: {'YES' if abs(rho_dr_lr) > abs(rho_kappa_lr) else 'NO'}")

    # ============================================================
    # SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    n_improved_knn = sum(1 for r in improvements if r["knn_gain"] > 0)
    n_improved_lr = sum(1 for r in improvements if r["lr_gain"] > 0)
    n_total = len(improvements)

    checks = [
        ("Whitening improves kNN in >50% of cases",
         n_improved_knn > n_total / 2,
         f"{n_improved_knn}/{n_total}"),
        ("Whitening improves LR in >50% of cases",
         n_improved_lr > n_total / 2,
         f"{n_improved_lr}/{n_total}"),
        ("Low eta predicts more kNN benefit (rho < -0.3)",
         rho_knn < -0.3,
         f"rho={rho_knn:.4f}"),
        ("dist_ratio predicts LR (non-tautological, rho > 0.8)",
         rho_dr_lr > 0.8,
         f"rho={rho_dr_lr:.4f}"),
        ("dist_ratio beats kappa for LR prediction",
         abs(rho_dr_lr) > abs(rho_kappa_lr),
         f"|{rho_dr_lr:.4f}| vs |{rho_kappa_lr:.4f}|"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "whitening_intervention",
        "dataset": ds_name,
        "K": K,
        "n_configs": len(all_experiment_results),
        "all_results": all_experiment_results,
        "improvements": improvements,
        "correlations": {
            "eta_vs_knn_gain": float(rho_knn),
            "eta_vs_lr_gain": float(rho_lr),
            "dist_ratio_vs_lr": float(rho_dr_lr),
            "kappa_vs_lr": float(rho_kappa_lr),
            "dist_ratio_vs_knn": float(rho_dr_knn),
            "kappa_vs_knn": float(rho_kappa_knn),
        },
        "scorecard": {
            "passes": passes, "total": len(checks),
            "details": [{"criterion": c, "passed": bool(p), "value": v}
                        for c, p, v in checks],
        },
    }

    out_path = RESULTS_DIR / "cti_whitening_intervention.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
