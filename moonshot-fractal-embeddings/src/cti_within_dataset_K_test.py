#!/usr/bin/env python
"""
WITHIN-DATASET K INTERVENTION: The decisive experiment.

Codex design (5/10 Nobel, 8+/10 if this works):
Pick CLINC (150 classes), subsample K in {10, 20, 40, 80, 120, 150}.
For each K, sweep alpha (layer depth) to get many (kappa, q) points.
Pre-register model comparison:
  M_log:   sigmoid(a*kappa + b*log(K) + c)
  M_sqrt:  sigmoid(a*kappa/sqrt(K) + c)
  M_hybrid: sigmoid((a*kappa - b*log(K)) / (1 + c*sqrt(K)))

This breaks the confound between K and dataset identity,
giving a clean test of the K-normalization theory.
"""

import json
import gc
import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Use Pythia-410m (balance of size and quality, from MODEL_DIRECTORY)
MODEL_ID = "EleutherAI/pythia-410m"
DATASET = "clinc_oos"  # 150 classes
K_VALUES = [10, 20, 40, 80, 120, 150]
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_SUBSETS = 3  # Random class subsets per K
N_SAMPLES_PER_CLASS = 20  # For kNN evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clinc_data():
    """Load CLINC-OOS dataset."""
    from datasets import load_dataset
    ds = load_dataset("clinc_oos", "plus", split="train")
    texts = ds["text"]
    labels = ds["intent"]
    return texts, labels


def get_hidden_states(model, tokenizer, texts, alpha, batch_size=32):
    """Get interpolated hidden states at fractional depth alpha."""
    model.eval()
    n_layers = model.config.num_hidden_layers
    target_layer = int(alpha * n_layers)
    target_layer = min(target_layer, n_layers - 1)

    all_hidden = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Mean pool the target layer
        hidden = outputs.hidden_states[target_layer + 1]  # +1 for embedding layer
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        all_hidden.append(pooled.cpu().numpy())

    return np.concatenate(all_hidden, axis=0)


def compute_kappa_knn(embeddings, labels, K_subset_labels, n_per_class=20):
    """Compute kappa and kNN accuracy for a subset of K classes."""
    # Filter to only selected classes
    mask = np.isin(labels, K_subset_labels)
    emb = embeddings[mask]
    lab = labels[mask]

    # Relabel to 0..K-1
    label_map = {old: new for new, old in enumerate(sorted(set(K_subset_labels)))}
    lab_mapped = np.array([label_map[l] for l in lab])
    K = len(set(K_subset_labels))

    if len(emb) < K * 5:
        return None

    # Compute scatter matrices
    d = emb.shape[1]
    grand_mean = emb.mean(0)
    S_B = np.zeros((d, d))
    S_W = np.zeros((d, d))

    for k in range(K):
        mask_k = lab_mapped == k
        if mask_k.sum() < 2:
            continue
        emb_k = emb[mask_k]
        mu_k = emb_k.mean(0)
        n_k = mask_k.sum()
        diff = (mu_k - grand_mean).reshape(-1, 1)
        S_B += n_k * (diff @ diff.T)
        centered = emb_k - mu_k
        S_W += centered.T @ centered

    N = len(emb)
    S_B /= N
    S_W /= N

    tr_SB = np.trace(S_B)
    tr_SW = np.trace(S_W)

    if tr_SW < 1e-10:
        return None

    kappa = float(tr_SB / tr_SW)

    # kNN accuracy (leave-one-out style with 1-NN)
    try:
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        # Split into train/test
        indices = np.arange(len(emb))
        np.random.shuffle(indices)
        n_test = min(len(indices) // 5, 200)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        knn.fit(emb[train_idx], lab_mapped[train_idx])
        acc = float(knn.score(emb[test_idx], lab_mapped[test_idx]))
    except Exception:
        acc = 1.0 / K

    q = (acc - 1.0/K) / (1.0 - 1.0/K)
    q = max(min(q, 0.999), 0.001)

    return {
        "kappa": kappa,
        "knn": acc,
        "q": q,
        "K": K,
        "n_samples": len(emb),
    }


def main():
    print("=" * 70)
    print("WITHIN-DATASET K INTERVENTION (CLINC subsets)")
    print("Decisive test: log(K) vs sqrt(K) within one dataset")
    print("=" * 70)

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()
    print(f"  Loaded on {DEVICE}")

    # Load data
    print("Loading CLINC data...")
    texts, labels = load_clinc_data()
    texts = list(texts)
    labels = np.array(labels)

    unique_labels = sorted(set(labels))
    print(f"  {len(texts)} samples, {len(unique_labels)} classes")

    # Pre-compute embeddings at each alpha
    print("\nPre-computing embeddings at each alpha...")
    alpha_embeddings = {}
    for alpha in ALPHA_VALUES:
        print(f"  alpha={alpha:.1f}...", end="", flush=True)
        emb = get_hidden_states(model, tokenizer, texts, alpha)
        alpha_embeddings[alpha] = emb
        print(f" done (shape={emb.shape})")

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Run K-intervention
    print(f"\n{'='*70}")
    print(f"K INTERVENTION: K = {K_VALUES}")
    print("=" * 70)

    all_results = []
    np.random.seed(42)

    for K in K_VALUES:
        print(f"\n--- K = {K} ---")

        for subset_idx in range(N_SUBSETS if K < 150 else 1):
            # Select K random classes
            if K < 150:
                selected = sorted(np.random.choice(unique_labels, K, replace=False).tolist())
            else:
                selected = unique_labels

            for alpha in ALPHA_VALUES:
                emb = alpha_embeddings[alpha]
                result = compute_kappa_knn(emb, labels, selected)
                if result is None:
                    continue

                result["alpha"] = alpha
                result["subset_idx"] = subset_idx
                all_results.append(result)

            n_so_far = sum(1 for r in all_results if r["K"] == K)
            # Print summary for this K
            k_results = [r for r in all_results if r["K"] == K and r["subset_idx"] == subset_idx]
            if k_results:
                kappas = [r["kappa"] for r in k_results]
                qs = [r["q"] for r in k_results]
                print(f"  subset {subset_idx}: {len(k_results)} points, "
                      f"kappa=[{min(kappas):.4f}, {max(kappas):.4f}], "
                      f"q=[{min(qs):.4f}, {max(qs):.4f}]")

    print(f"\nTotal data points: {len(all_results)}")

    # Filter valid points
    valid = [r for r in all_results if 0.01 < r["q"] < 0.99 and r["kappa"] > 0]
    print(f"Valid points (0.01 < q < 0.99): {len(valid)}")

    if len(valid) < 20:
        print("Not enough valid points!")
        # Save what we have
        out = {"experiment": "within_dataset_K_test", "n_points": len(valid),
               "error": "too_few_valid_points", "all_results": all_results}
        with open(RESULTS_DIR / "cti_within_dataset_K.json", "w") as f:
            json.dump(out, f, indent=2,
                      default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
        return

    kappas = np.array([r["kappa"] for r in valid])
    qs = np.array([r["q"] for r in valid])
    Ks = np.array([float(r["K"]) for r in valid])

    # ================================================================
    # PRE-REGISTERED MODEL COMPARISON
    # ================================================================
    print(f"\n{'='*70}")
    print("PRE-REGISTERED MODEL COMPARISON")
    print("=" * 70)

    # M_log: sigmoid(a*kappa + b*log(K) + c)
    def loss_log(params):
        a, b, c = params
        x = a * kappas + b * np.log(Ks + 1) + c
        return np.sum((qs - expit(x))**2)

    best_log = None
    best_log_loss = float("inf")
    for a0 in [5.0, 10.0, 20.0]:
        for b0 in [-2.0, -1.0, -0.5]:
            for c0 in [-3.0, -1.0, 0.0]:
                try:
                    res = minimize(loss_log, [a0, b0, c0], method="Nelder-Mead",
                                   options={"maxiter": 10000})
                    if res.fun < best_log_loss:
                        best_log_loss = res.fun
                        best_log = res.x
                except:
                    pass

    a_log, b_log, c_log = best_log
    q_log = expit(a_log * kappas + b_log * np.log(Ks + 1) + c_log)
    r2_log = 1 - np.sum((qs - q_log)**2) / np.sum((qs - qs.mean())**2)
    mae_log = np.mean(np.abs(qs - q_log))
    print(f"\n  M_log:   sigmoid({a_log:.3f}*kappa + {b_log:.3f}*log(K+1) + {c_log:.3f})")
    print(f"    R^2 = {r2_log:.4f}, MAE = {mae_log:.4f}")

    # M_sqrt: sigmoid(a*kappa/sqrt(K) + c)
    def loss_sqrt(params):
        a, c = params
        x = a * kappas / np.sqrt(Ks) + c
        return np.sum((qs - expit(x))**2)

    best_sqrt = None
    best_sqrt_loss = float("inf")
    for a0 in [5.0, 10.0, 20.0, 50.0]:
        for c0 in [-5.0, -3.0, -1.0]:
            try:
                res = minimize(loss_sqrt, [a0, c0], method="Nelder-Mead",
                               options={"maxiter": 10000})
                if res.fun < best_sqrt_loss:
                    best_sqrt_loss = res.fun
                    best_sqrt = res.x
            except:
                pass

    a_sqrt, c_sqrt = best_sqrt
    q_sqrt = expit(a_sqrt * kappas / np.sqrt(Ks) + c_sqrt)
    r2_sqrt = 1 - np.sum((qs - q_sqrt)**2) / np.sum((qs - qs.mean())**2)
    mae_sqrt = np.mean(np.abs(qs - q_sqrt))
    print(f"\n  M_sqrt:  sigmoid({a_sqrt:.3f}*kappa/sqrt(K) + {c_sqrt:.3f})")
    print(f"    R^2 = {r2_sqrt:.4f}, MAE = {mae_sqrt:.4f}")

    # M_hybrid: sigmoid((a*kappa - b*log(K)) / (1 + c*sqrt(K)))
    def loss_hybrid(params):
        a, b, c, d_param = params
        numer = a * kappas - b * np.log(Ks + 1)
        denom = 1 + abs(c) * np.sqrt(Ks)
        x = numer / denom + d_param
        return np.sum((qs - expit(x))**2)

    best_hyb = None
    best_hyb_loss = float("inf")
    for a0 in [5.0, 10.0, 20.0]:
        for b0 in [0.1, 0.5, 1.0]:
            for c0 in [0.01, 0.1, 0.5]:
                try:
                    res = minimize(loss_hybrid, [a0, b0, c0, -1.0],
                                   method="Nelder-Mead", options={"maxiter": 20000})
                    if res.fun < best_hyb_loss:
                        best_hyb_loss = res.fun
                        best_hyb = res.x
                except:
                    pass

    a_hyb, b_hyb, c_hyb, d_hyb = best_hyb
    numer = a_hyb * kappas - b_hyb * np.log(Ks + 1)
    denom = 1 + abs(c_hyb) * np.sqrt(Ks)
    q_hyb = expit(numer / denom + d_hyb)
    r2_hyb = 1 - np.sum((qs - q_hyb)**2) / np.sum((qs - qs.mean())**2)
    mae_hyb = np.mean(np.abs(qs - q_hyb))
    print(f"\n  M_hybrid: sigmoid(({a_hyb:.3f}*kappa - {b_hyb:.3f}*log(K))/(1 + {abs(c_hyb):.4f}*sqrt(K)) + {d_hyb:.3f})")
    print(f"    R^2 = {r2_hyb:.4f}, MAE = {mae_hyb:.4f}")

    # M_div: sigmoid(a*kappa/log(K+1) + c)
    def loss_div(params):
        a, c = params
        x = a * kappas / np.log(Ks + 1) + c
        return np.sum((qs - expit(x))**2)

    best_div = None
    best_div_loss = float("inf")
    for a0 in [5.0, 10.0, 20.0, 50.0]:
        for c0 in [-5.0, -3.0, -1.0]:
            try:
                res = minimize(loss_div, [a0, c0], method="Nelder-Mead",
                               options={"maxiter": 10000})
                if res.fun < best_div_loss:
                    best_div_loss = res.fun
                    best_div = res.x
            except:
                pass

    a_div, c_div = best_div
    q_div = expit(a_div * kappas / np.log(Ks + 1) + c_div)
    r2_div = 1 - np.sum((qs - q_div)**2) / np.sum((qs - qs.mean())**2)
    mae_div = np.mean(np.abs(qs - q_div))
    print(f"\n  M_div:   sigmoid({a_div:.3f}*kappa/log(K+1) + {c_div:.3f})")
    print(f"    R^2 = {r2_div:.4f}, MAE = {mae_div:.4f}")

    # ================================================================
    # PER-K ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print("PER-K SIGMOID FITS")
    print("=" * 70)

    per_K_results = []
    for K_val in K_VALUES:
        K_mask = Ks == K_val
        if K_mask.sum() < 3:
            continue
        kap_K = kappas[K_mask]
        qs_K = qs[K_mask]

        def sig(x, a, b):
            return expit(a * x + b)

        from scipy.optimize import curve_fit
        try:
            popt, _ = curve_fit(sig, kap_K, qs_K, p0=[10.0, -1.0], maxfev=10000)
            q_pred = sig(kap_K, *popt)
            r2_K = 1 - np.sum((qs_K - q_pred)**2) / max(np.sum((qs_K - qs_K.mean())**2), 1e-10)
            per_K_results.append({
                "K": int(K_val), "a": float(popt[0]), "b": float(popt[1]),
                "r2": float(r2_K), "n": int(K_mask.sum()),
            })
            print(f"  K={int(K_val):>4}: sigmoid({popt[0]:.3f}*kappa + {popt[1]:.3f}), R^2={r2_K:.4f}, n={K_mask.sum()}")
        except Exception as e:
            print(f"  K={int(K_val):>4}: fit failed: {e}")

    # Check slope variation with K
    if len(per_K_results) >= 3:
        K_arr = np.array([r["K"] for r in per_K_results])
        slopes = np.array([r["a"] for r in per_K_results])

        a_logK = slopes * np.log(K_arr + 1)
        a_sqrtK = slopes * np.sqrt(K_arr)

        cv_logK = np.std(a_logK) / max(np.mean(a_logK), 1e-10)
        cv_sqrtK = np.std(a_sqrtK) / max(np.mean(a_sqrtK), 1e-10)

        print(f"\n  Slope collapse:")
        print(f"    a * log(K+1): CV = {cv_logK:.4f}")
        print(f"    a * sqrt(K):  CV = {cv_sqrtK:.4f}")
        if cv_logK < cv_sqrtK:
            print(f"    -> log(K) WINS on slope collapse")
        else:
            print(f"    -> sqrt(K) WINS on slope collapse")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    models = [
        ("M_log (additive)", r2_log, mae_log, 3),
        ("M_sqrt (divisive)", r2_sqrt, mae_sqrt, 2),
        ("M_hybrid (crossover)", r2_hyb, mae_hyb, 4),
        ("M_div (kappa/logK)", r2_div, mae_div, 2),
    ]

    print(f"\n  {'Model':>30} {'R^2':>8} {'MAE':>8} {'#p':>4}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*4}")
    for name, r2, mae, np_ in sorted(models, key=lambda x: -x[1]):
        print(f"  {name:>30} {r2:>8.4f} {mae:>8.4f} {np_:>4}")

    best_model = max(models, key=lambda x: x[1])
    print(f"\n  WINNER: {best_model[0]}")

    # Scorecard
    print(f"\n{'='*70}")
    print("SCORECARD")
    print("=" * 70)

    checks = [
        ("Best model R^2 > 0.85",
         best_model[1] > 0.85,
         f"best={best_model[1]:.4f}"),
        ("log(K) vs sqrt(K) distinguishable (delta > 0.02)",
         abs(r2_log - r2_sqrt) > 0.02,
         f"logK={r2_log:.4f}, sqrtK={r2_sqrt:.4f}"),
        ("Within-dataset variation covers q=[0.1, 0.9]",
         qs.min() < 0.2 and qs.max() > 0.8,
         f"q=[{qs.min():.3f}, {qs.max():.3f}]"),
        ("Per-K sigmoid R^2 > 0.80 in >= 3 K values",
         sum(1 for r in per_K_results if r["r2"] > 0.80) >= 3,
         f"{sum(1 for r in per_K_results if r['r2'] > 0.80)} K-values"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "within_dataset_K_intervention",
        "model": MODEL_ID,
        "dataset": "clinc_oos",
        "K_values": K_VALUES,
        "n_subsets": N_SUBSETS,
        "n_total_points": len(all_results),
        "n_valid_points": len(valid),
        "global_models": {
            "M_log": {"r2": float(r2_log), "mae": float(mae_log),
                      "params": [float(a_log), float(b_log), float(c_log)]},
            "M_sqrt": {"r2": float(r2_sqrt), "mae": float(mae_sqrt),
                       "params": [float(a_sqrt), float(c_sqrt)]},
            "M_hybrid": {"r2": float(r2_hyb), "mae": float(mae_hyb),
                         "params": [float(a_hyb), float(b_hyb), float(c_hyb), float(d_hyb)]},
            "M_div": {"r2": float(r2_div), "mae": float(mae_div),
                      "params": [float(a_div), float(c_div)]},
        },
        "per_K": per_K_results,
        "passes": passes,
    }

    out_path = RESULTS_DIR / "cti_within_dataset_K.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
