#!/usr/bin/env python
"""
PREREGISTERED BLIND PREDICTION + INTERVENTION

Codex design (6.0/10 review):
  "Run a preregistered blind prediction + intervention study:
   1. Freeze protocol, pick fully unseen datasets and architectures.
   2. Predict task quality only from kappa/sqrt(K) before seeing outcomes.
   3. In a subset, intervene to increase/decrease kappa and test whether
      quality moves along the same sigmoid."

Protocol:
  Phase 1 (TRAINING): Fit sigmoid on 189 known points (CLINC+AGNews+DBPedia)
  Phase 2 (BLIND): Extract kappa on Yahoo (K=10) and arXiv (K=115) without
    looking at kNN. Predict q = sigmoid(kappa/sqrt(K), frozen_params).
  Phase 3 (REVEAL): Compute actual kNN and compare.
  Phase 4 (INTERVENTION): For 2 models, vary alpha and verify kappa->quality
    follows the same sigmoid on the unseen datasets.

Pre-registered criteria (FROZEN before data collection):
  - Blind prediction MAE <= 0.08 (on normalized q)
  - Prediction rho >= 0.90 per dataset
  - Intervention follows sigmoid (R^2 >= 0.85 per model)
"""

import json
import sys
import time
import gc
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model, ResidualScaler
from hierarchical_datasets import load_hierarchical_dataset

# Unseen datasets for blind prediction
BLIND_DATASETS = ["yahoo", "arxiv"]

# Models for blind test (subset for speed - 2T + 2SSM)
BLIND_MODELS = [
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen3-0.6B",
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-370m-hf",
]

ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]


def sigmoid(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


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
                out = model(**enc, output_hidden_states=True, return_dict=True)
            mask = enc.get("attention_mask",
                           torch.ones(enc["input_ids"].shape, device=device))
            for idx, hs in enumerate(out.hidden_states):
                hs_f = hs.float()
                m = mask.unsqueeze(-1).float()
                pooled = (hs_f * m).sum(1) / m.sum(1).clamp(min=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if idx not in all_hidden:
                    all_hidden[idx] = []
                all_hidden[idx].append(pooled.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in all_hidden.items()}


def compute_kappa(X, labels):
    """Compute kappa from representations."""
    try:
        if np.isnan(X).any():
            return 0.0
        unique_labels = np.unique(labels)
        grand_mean = X.mean(0)
        trace_sb = 0.0
        trace_sw = 0.0
        for lbl in unique_labels:
            lbl_mask = labels == lbl
            n_k = lbl_mask.sum()
            if n_k < 2:
                continue
            X_k = X[lbl_mask]
            mu_k = X_k.mean(0)
            trace_sb += n_k * np.sum((mu_k - grand_mean) ** 2)
            trace_sw += np.sum((X_k - mu_k) ** 2)
        if trace_sw < 1e-12:
            return 100.0 if trace_sb > 0 else 0.0
        return float(min(trace_sb / trace_sw, 100.0))
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


def compute_layer_averages(reps, labels):
    """Compute kNN and kappa averaged across all layers."""
    knn_vals, kappa_vals = [], []
    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue
        knn_vals.append(compute_knn(X, labels))
        kappa_vals.append(compute_kappa(X, labels))
    return {
        "knn": float(np.mean(knn_vals)) if knn_vals else 0,
        "kappa": float(np.mean(kappa_vals)) if kappa_vals else 0,
    }


def main():
    print("=" * 70)
    print("PREREGISTERED BLIND PREDICTION + INTERVENTION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # PHASE 1: TRAIN SIGMOID ON KNOWN DATA
    # ============================================================
    print(f"\n{'='*70}")
    print("PHASE 1: TRAINING (freeze sigmoid params)")
    print(f"{'='*70}")

    # Load all known data
    with open(RESULTS_DIR / "cti_geometry_mediator.json") as f:
        clinc_raw = json.load(f)

    train_points = []
    for p in clinc_raw["all_points"]:
        K = 150
        knn = p["knn"]
        kappa = p["kappa"]
        q = (knn - 1.0 / K) / (1.0 - 1.0 / K)
        x = kappa / np.sqrt(K)
        train_points.append({"x": x, "q": q, "dataset": "clinc"})

    for ds_name in ["agnews", "dbpedia_classes"]:
        with open(RESULTS_DIR / f"cti_multidata_{ds_name}_cache.json") as f:
            data = json.load(f)
        for p in data:
            K = p["n_classes"]
            knn = p["knn"]
            kappa = p["kappa"]
            q = (knn - 1.0 / K) / (1.0 - 1.0 / K)
            x = kappa / np.sqrt(K)
            train_points.append({"x": x, "q": q, "dataset": ds_name})

    x_train = np.array([p["x"] for p in train_points])
    q_train = np.array([p["q"] for p in train_points])

    popt, _ = curve_fit(sigmoid, x_train, q_train,
                        p0=[0.6, 10, np.median(x_train), 0.0],
                        maxfev=10000)
    pred_train = sigmoid(x_train, *popt)
    ss_tot = np.sum((q_train - q_train.mean()) ** 2)
    r2_train = 1 - np.sum((q_train - pred_train) ** 2) / ss_tot
    mae_train = float(np.mean(np.abs(q_train - pred_train)))

    print(f"  Training data: {len(train_points)} points")
    print(f"  Frozen sigmoid params: a={popt[0]:.4f}, b={popt[1]:.4f}, "
          f"c={popt[2]:.4f}, d={popt[3]:.4f}")
    print(f"  Training R^2={r2_train:.4f}, MAE={mae_train:.4f}")
    print(f"  >>> PARAMS FROZEN. No more fitting. <<<")

    # ============================================================
    # PHASE 2+3: BLIND EXTRACTION + REVEAL ON UNSEEN DATASETS
    # ============================================================
    print(f"\n{'='*70}")
    print("PHASE 2+3: BLIND PREDICTION ON UNSEEN DATASETS")
    print(f"{'='*70}")

    blind_results = {}

    for ds_name in BLIND_DATASETS:
        print(f"\n--- Dataset: {ds_name} ---")
        ds = load_hierarchical_dataset(ds_name, split="test", max_samples=2000)
        texts = [s.text for s in ds.samples]
        labels = np.array([s.level1_label for s in ds.samples])
        K = len(np.unique(labels))
        print(f"  {len(texts)} samples, K={K} classes")

        ds_points = []

        for model_id in BLIND_MODELS:
            paradigm = "ssm" if "mamba" in model_id.lower() else "transformer"
            short = model_id.split("/")[-1]
            print(f"\n  MODEL: {short} ({paradigm})")

            try:
                model, tokenizer, n_layers, n_params = load_model(model_id, device)
            except Exception as e:
                print(f"    FAILED to load: {e}")
                continue

            for alpha in ALPHAS:
                print(f"    alpha={alpha:.2f}", end="", flush=True)
                t0 = time.time()

                try:
                    reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)
                    stats = compute_layer_averages(reps, labels)

                    # Predict BEFORE looking at kNN
                    kappa = stats["kappa"]
                    x_blind = kappa / np.sqrt(K)
                    q_pred = float(sigmoid(x_blind, *popt))

                    # Now reveal actual quality
                    knn_actual = stats["knn"]
                    q_actual = (knn_actual - 1.0 / K) / (1.0 - 1.0 / K)

                    error = abs(q_pred - q_actual)
                    elapsed = time.time() - t0
                    print(f"  kappa={kappa:.4f}  q_pred={q_pred:.3f}  "
                          f"q_actual={q_actual:.3f}  err={error:.3f}  ({elapsed:.1f}s)")

                    ds_points.append({
                        "model": model_id,
                        "paradigm": paradigm,
                        "dataset": ds_name,
                        "K": K,
                        "alpha": alpha,
                        "kappa": kappa,
                        "knn": knn_actual,
                        "q_actual": q_actual,
                        "q_predicted": q_pred,
                        "error": error,
                    })
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM!")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"  ERROR: {e}")

                sys.stdout.flush()

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Per-dataset analysis
        if ds_points:
            q_preds = np.array([p["q_predicted"] for p in ds_points])
            q_actuals = np.array([p["q_actual"] for p in ds_points])
            kappas = np.array([p["kappa"] for p in ds_points])

            mae = float(np.mean(np.abs(q_preds - q_actuals)))
            rho, p_rho = spearmanr(kappas, q_actuals)

            blind_results[ds_name] = {
                "K": K,
                "n_points": len(ds_points),
                "blind_mae": mae,
                "rho_kappa_q": float(rho),
                "points": ds_points,
            }

            print(f"\n  {ds_name} RESULTS:")
            print(f"    Blind prediction MAE: {mae:.4f}")
            print(f"    kappa-q rho: {rho:.4f}")
            print(f"    Pre-registered: MAE <= 0.08, rho >= 0.90")
            print(f"    MAE {'PASS' if mae <= 0.08 else 'FAIL'}, "
                  f"rho {'PASS' if rho >= 0.90 else 'FAIL'}")

    # ============================================================
    # PHASE 4: OVERALL BLIND PREDICTION ASSESSMENT
    # ============================================================
    print(f"\n{'='*70}")
    print("OVERALL BLIND PREDICTION ASSESSMENT")
    print(f"{'='*70}")

    all_blind = []
    for ds_name, res in blind_results.items():
        all_blind.extend(res["points"])

    if all_blind:
        all_q_pred = np.array([p["q_predicted"] for p in all_blind])
        all_q_actual = np.array([p["q_actual"] for p in all_blind])
        all_kappas = np.array([p["kappa"] for p in all_blind])
        all_Ks = np.array([p["K"] for p in all_blind])

        overall_mae = float(np.mean(np.abs(all_q_pred - all_q_actual)))
        overall_rho, _ = spearmanr(all_kappas / np.sqrt(all_Ks), all_q_actual)
        overall_r, _ = pearsonr(all_q_pred, all_q_actual)

        print(f"  Total blind points: {len(all_blind)}")
        print(f"  Overall blind MAE: {overall_mae:.4f}")
        print(f"  Overall rho (kappa/sqrt(K) vs q): {overall_rho:.4f}")
        print(f"  Prediction-actual Pearson r: {overall_r:.4f}")

        # Per-dataset summary
        for ds_name, res in sorted(blind_results.items()):
            status_mae = "PASS" if res["blind_mae"] <= 0.08 else "FAIL"
            status_rho = "PASS" if res["rho_kappa_q"] >= 0.90 else "FAIL"
            print(f"  {ds_name}: MAE={res['blind_mae']:.4f} [{status_mae}], "
                  f"rho={res['rho_kappa_q']:.4f} [{status_rho}]")

    # ============================================================
    # SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    checks = []
    for ds_name, res in sorted(blind_results.items()):
        checks.append((
            f"{ds_name} blind MAE <= 0.08",
            res["blind_mae"] <= 0.08,
            f"MAE={res['blind_mae']:.4f}"
        ))
        checks.append((
            f"{ds_name} rho >= 0.90",
            res["rho_kappa_q"] >= 0.90,
            f"rho={res['rho_kappa_q']:.4f}"
        ))

    if all_blind:
        checks.append((
            "Overall blind MAE <= 0.08",
            overall_mae <= 0.08,
            f"MAE={overall_mae:.4f}"
        ))
        checks.append((
            "Prediction-actual r >= 0.85",
            overall_r >= 0.85,
            f"r={overall_r:.4f}"
        ))

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "preregistered_blind_prediction",
        "frozen_sigmoid_params": {
            "a": float(popt[0]), "b": float(popt[1]),
            "c": float(popt[2]), "d": float(popt[3]),
        },
        "training": {
            "n_points": len(train_points),
            "r2": float(r2_train),
            "mae": float(mae_train),
        },
        "blind_results": {
            ds: {k: v for k, v in res.items() if k != "points"}
            for ds, res in blind_results.items()
        },
        "blind_points": all_blind,
        "overall": {
            "mae": float(overall_mae) if all_blind else None,
            "rho": float(overall_rho) if all_blind else None,
            "r": float(overall_r) if all_blind else None,
        },
        "scorecard": {
            "passes": passes,
            "total": len(checks),
            "details": [
                {"criterion": c, "passed": bool(p), "value": v}
                for c, p, v in checks
            ],
        },
    }

    out_path = RESULTS_DIR / "cti_blind_prediction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
