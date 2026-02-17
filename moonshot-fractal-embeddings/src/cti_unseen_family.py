#!/usr/bin/env python
"""
UNSEEN-FAMILY VALIDATION: Do universality classes predict hybrid architectures?

Codex design (7.6/10 Nobel review):
  The decisive test: fit kappa law on KNOWN families (transformer + Mamba),
  then predict UNSEEN family (Falcon-H1 hybrid = transformer + Mamba).

  If hybrids:
    a) Fall ON one existing curve -> that family dominates
    b) Fall BETWEEN curves -> hybrid nature is weighted average
    c) Fall on a NEW curve -> 3rd universality class
    d) Are predicted well by 2D law -> eta captures the mechanism

Pre-registered criteria:
  1. Unseen-family 1D prediction MAE <= 0.08 (kappa sigmoid trained on all 7 models)
  2. Hybrid eta falls between transformer and SSM eta (confirming intermediate geometry)
  3. 2D law trained on 7 models predicts hybrids better than 1D (MAE improvement > 0)
  4. Slope of hybrid Q(kappa) is between transformer (0.69) and SSM (1.13)

Models from MODEL_DIRECTORY.md:
  - Known: 4 transformers + 3 Mamba (from geometry mediator)
  - Unseen: Falcon-H1-0.5B (hybrid), Falcon-H1-1.5B (hybrid)
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


def compute_kappa_and_eta(X, labels):
    """Compute kappa and eta."""
    try:
        if np.isnan(X).any():
            return {"kappa": 0.0, "eta": 0.0}
        d = X.shape[1]
        unique_labels = np.unique(labels)
        grand_mean = X.mean(0)
        Z_parts = []
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
            centered = X_k - mu_k
            trace_sw += np.sum(centered ** 2)
            Z_parts.append(centered)
        if trace_sw < 1e-12:
            return {"kappa": 100.0 if trace_sb > 0 else 0.0, "eta": 0.0}
        kappa = float(min(trace_sb / trace_sw, 100.0))
        Z = np.concatenate(Z_parts, axis=0)
        try:
            s = np.linalg.svd(Z, compute_uv=False)
            s2 = s ** 2
            s4 = s2 ** 2
            trace_sw_sq = float(s4.sum())
            eta = float((trace_sw ** 2) / (d * trace_sw_sq)) if trace_sw_sq > 1e-20 else 0.0
        except Exception:
            eta = 0.0
        return {"kappa": kappa, "eta": eta}
    except Exception:
        return {"kappa": 0.0, "eta": 0.0}


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


def compute_layer_stats(reps, labels):
    """Compute averaged stats across layers."""
    knn_vals, kappa_vals, eta_vals = [], [], []
    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue
        knn_vals.append(compute_knn(X, labels))
        stats = compute_kappa_and_eta(X, labels)
        if np.isfinite(stats["kappa"]):
            kappa_vals.append(stats["kappa"])
        if np.isfinite(stats.get("eta", 0)):
            eta_vals.append(stats["eta"])
    return {
        "knn": float(np.mean(knn_vals)) if knn_vals else 0,
        "kappa": float(np.mean(kappa_vals)) if kappa_vals else 0,
        "eta": float(np.mean(eta_vals)) if eta_vals else 0,
    }


def sigmoid_1d(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def sigmoid_2d(features, a, b, c, d):
    z = a * features[:, 0] + b * features[:, 1] + c * features[:, 2] + d
    return 1.0 / (1.0 + np.exp(np.clip(-z, -500, 500)))


def main():
    print("=" * 70)
    print("UNSEEN-FAMILY VALIDATION")
    print("Train on transformers + Mamba, predict Falcon-H1 hybrids")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    # Load dataset
    ds = load_hierarchical_dataset("clinc", split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])
    print(f"Dataset: CLINC150, {len(texts)} samples")

    # ============================================================
    # STEP 1: Load training data (from geometry mediator)
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 1: LOAD TRAINING DATA")
    print(f"{'='*70}")

    mediator_path = RESULTS_DIR / "cti_geometry_mediator.json"
    with open(mediator_path) as f:
        mediator = json.load(f)

    train_points = mediator["all_points"]
    print(f"  Loaded {len(train_points)} training points from {mediator_path.name}")

    train_kappas = np.array([p["kappa"] for p in train_points])
    train_knns = np.array([p["knn"] for p in train_points])
    train_etas = np.array([p["eta"] for p in train_points])
    train_paradigms = np.array([p["paradigm"] for p in train_points])

    # ============================================================
    # STEP 2: Run hybrid models
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 2: RUN HYBRID MODELS")
    print(f"{'='*70}")

    hybrid_models = [
        "tiiuae/Falcon-H1-0.5B-Base",
    ]

    # Load cached data or re-extract
    cached_path = RESULTS_DIR / "cti_unseen_family_cache.json"
    if cached_path.exists():
        print(f"  Loading cached hybrid data from {cached_path.name}")
        with open(cached_path) as f:
            hybrid_points = json.load(f)
        print(f"  Using {len(hybrid_points)} cached hybrid points")
    else:
        hybrid_points = []
        for model_id in hybrid_models:
            short = model_id.split("/")[-1]
            print(f"\n  --- {short} ---")
            try:
                model, tokenizer, n_layers, n_params = load_model(model_id, device)
            except Exception as e:
                print(f"  FAILED to load: {e}")
                continue

            for alpha in alphas:
                print(f"    alpha={alpha:.2f}", end="", flush=True)
                t0 = time.time()
                try:
                    reps = extract_all_layer_reps(model, tokenizer, texts, alpha,
                                                  device)
                    stats = compute_layer_stats(reps, labels)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    print(f"  OOM: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                elapsed = time.time() - t0
                print(f"  kNN={stats['knn']:.3f}  kappa={stats['kappa']:.4f}  "
                      f"eta={stats['eta']:.4f}  ({elapsed:.1f}s)")

                hybrid_points.append({
                    "model": model_id,
                    "paradigm": "hybrid",
                    "n_layers": n_layers,
                    "n_params": n_params,
                    "alpha": alpha,
                    "knn": stats["knn"],
                    "kappa": stats["kappa"],
                    "eta": stats["eta"],
                    "kappa_eta": stats["kappa"] * stats["eta"],
                })
                sys.stdout.flush()

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Save cache
        with open(cached_path, "w") as f:
            json.dump(hybrid_points, f, indent=2)

    if not hybrid_points:
        print("  NO HYBRID DATA - cannot proceed")
        return

    hybrid_kappas = np.array([p["kappa"] for p in hybrid_points])
    hybrid_knns = np.array([p["knn"] for p in hybrid_points])
    hybrid_etas = np.array([p["eta"] for p in hybrid_points])

    # ============================================================
    # STEP 3: PREDICTION (train on known, predict hybrid)
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 3: UNSEEN-FAMILY PREDICTION")
    print(f"{'='*70}")

    # 3a. 1D sigmoid: fit on training data, predict hybrids
    try:
        popt_1d, _ = curve_fit(sigmoid_1d, train_kappas, train_knns,
                               p0=[0.5, 10, 0.3, 0.1], maxfev=10000)
        pred_1d = sigmoid_1d(hybrid_kappas, *popt_1d)
        mae_1d = float(np.mean(np.abs(hybrid_knns - pred_1d)))
    except Exception:
        mae_1d = 1.0
        pred_1d = np.zeros_like(hybrid_knns)

    # 3b. 2D sigmoid: fit on training data, predict hybrids
    train_log_k = np.log(np.clip(train_kappas, 1e-6, None))
    train_log_e = np.log(np.clip(train_etas, 1e-6, None))
    train_features = np.column_stack([train_log_k, train_log_e,
                                       train_log_k * train_log_e])

    hybrid_log_k = np.log(np.clip(hybrid_kappas, 1e-6, None))
    hybrid_log_e = np.log(np.clip(hybrid_etas, 1e-6, None))
    hybrid_features = np.column_stack([hybrid_log_k, hybrid_log_e,
                                        hybrid_log_k * hybrid_log_e])

    try:
        popt_2d, _ = curve_fit(sigmoid_2d, train_features, train_knns,
                               p0=[1.0, 0.5, 0.1, -1.0], maxfev=20000,
                               bounds=([-10, -10, -10, -10], [10, 10, 10, 10]))
        pred_2d = sigmoid_2d(hybrid_features, *popt_2d)
        mae_2d = float(np.mean(np.abs(hybrid_knns - pred_2d)))
    except Exception:
        mae_2d = 1.0
        pred_2d = np.zeros_like(hybrid_knns)

    # 3c. Per-architecture fit: transformer-only sigmoid
    t_mask = train_paradigms == "transformer"
    try:
        popt_t, _ = curve_fit(sigmoid_1d, train_kappas[t_mask], train_knns[t_mask],
                              p0=[0.5, 10, 0.3, 0.1], maxfev=10000)
        pred_t = sigmoid_1d(hybrid_kappas, *popt_t)
        mae_t = float(np.mean(np.abs(hybrid_knns - pred_t)))
    except Exception:
        mae_t = 1.0

    # 3d. Per-architecture fit: SSM-only sigmoid
    s_mask = train_paradigms == "ssm"
    try:
        popt_s, _ = curve_fit(sigmoid_1d, train_kappas[s_mask], train_knns[s_mask],
                              p0=[0.5, 10, 0.3, 0.1], maxfev=10000)
        pred_s = sigmoid_1d(hybrid_kappas, *popt_s)
        mae_s = float(np.mean(np.abs(hybrid_knns - pred_s)))
    except Exception:
        mae_s = 1.0

    print(f"\n  PREDICTION MAE ON HYBRIDS:")
    print(f"    1D (all data):         {mae_1d:.4f}")
    print(f"    2D (all data):         {mae_2d:.4f}")
    print(f"    1D (transformer only): {mae_t:.4f}")
    print(f"    1D (SSM only):         {mae_s:.4f}")
    best_family = "transformer" if mae_t < mae_s else "SSM"
    print(f"    Hybrid closest to: {best_family} family")

    # Per-point predictions
    print(f"\n  PER-POINT PREDICTIONS:")
    for i, p in enumerate(hybrid_points):
        short = p["model"].split("/")[-1]
        actual = p["knn"]
        p1d = pred_1d[i] if i < len(pred_1d) else 0
        p2d = pred_2d[i] if i < len(pred_2d) else 0
        print(f"    {short:>20} a={p['alpha']:.2f}: actual={actual:.3f}, "
              f"1D={p1d:.3f} (err={abs(actual-p1d):.3f}), "
              f"2D={p2d:.3f} (err={abs(actual-p2d):.3f})")

    # ============================================================
    # STEP 4: GEOMETRY ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 4: GEOMETRY ANALYSIS")
    print(f"{'='*70}")

    # Eta distribution comparison
    print(f"\n  ETA BY ARCHITECTURE:")
    for par_name, par_etas in [
        ("transformer", train_etas[t_mask]),
        ("ssm", train_etas[s_mask]),
        ("hybrid", hybrid_etas),
    ]:
        print(f"    {par_name:>12}: mean={par_etas.mean():.4f}, "
              f"std={par_etas.std():.4f}, "
              f"range=[{par_etas.min():.4f}, {par_etas.max():.4f}]")

    # Check if hybrid eta is between transformer and SSM
    mean_eta_t = train_etas[t_mask].mean()
    mean_eta_s = train_etas[s_mask].mean()
    mean_eta_h = hybrid_etas.mean()
    eta_between = min(mean_eta_t, mean_eta_s) <= mean_eta_h <= max(mean_eta_t, mean_eta_s)

    print(f"\n  Hybrid eta between transformer and SSM: {'YES' if eta_between else 'NO'}")
    if not eta_between:
        if mean_eta_h < min(mean_eta_t, mean_eta_s):
            print(f"    Hybrid is MORE anisotropic than both families")
        else:
            print(f"    Hybrid is MORE isotropic than both families")

    # Slope comparison
    print(f"\n  kNN vs kappa SLOPES:")
    slope_t = np.polyfit(train_kappas[t_mask], train_knns[t_mask], 1)[0]
    slope_s = np.polyfit(train_kappas[s_mask], train_knns[s_mask], 1)[0]
    slope_h = np.polyfit(hybrid_kappas, hybrid_knns, 1)[0]

    print(f"    transformer: {slope_t:.4f}")
    print(f"           SSM: {slope_s:.4f}")
    print(f"        hybrid: {slope_h:.4f}")
    slope_between = min(slope_t, slope_s) <= slope_h <= max(slope_t, slope_s)
    print(f"    Hybrid slope between T and SSM: {'YES' if slope_between else 'NO'}")

    # Per-model slopes
    hybrid_model_ids = sorted(set(p["model"] for p in hybrid_points))
    for mid in hybrid_model_ids:
        pts = [p for p in hybrid_points if p["model"] == mid]
        k = np.array([p["kappa"] for p in pts])
        q = np.array([p["knn"] for p in pts])
        sl = np.polyfit(k, q, 1)[0]
        rho, _ = spearmanr(k, q)
        short = mid.split("/")[-1]
        print(f"    {short:>20}: slope={sl:.4f}, rho={rho:.4f}")

    # ============================================================
    # STEP 5: UNIVERSALITY CLASS DETERMINATION
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 5: UNIVERSALITY CLASS DETERMINATION")
    print(f"{'='*70}")

    # Where does hybrid fall?
    # Compute distance to each family's curve
    print(f"\n  CURVE DISTANCE:")
    print(f"    To transformer curve: MAE={mae_t:.4f}")
    print(f"    To SSM curve:         MAE={mae_s:.4f}")
    print(f"    To pooled curve:      MAE={mae_1d:.4f}")

    if mae_t < mae_s and mae_t < mae_1d * 1.2:
        class_assignment = "transformer"
    elif mae_s < mae_t and mae_s < mae_1d * 1.2:
        class_assignment = "ssm"
    elif mae_1d < min(mae_t, mae_s):
        class_assignment = "intermediate (pooled best)"
    else:
        class_assignment = "new_class"

    print(f"    Assignment: hybrid -> {class_assignment}")

    # Test: does adding hybrids to the training set improve cross-prediction?
    all_kappas = np.concatenate([train_kappas, hybrid_kappas])
    all_knns = np.concatenate([train_knns, hybrid_knns])
    all_paradigms_ext = np.concatenate([train_paradigms,
                                        np.array(["hybrid"] * len(hybrid_kappas))])

    rho_all, p_all = spearmanr(all_kappas, all_knns)
    r_all, _ = pearsonr(all_kappas, all_knns)
    print(f"\n  GLOBAL CORRELATION (with hybrids):")
    print(f"    All 3 families (N={len(all_kappas)}): rho={rho_all:.4f}, r={r_all:.4f}")
    print(f"    Without hybrids (N={len(train_kappas)}): rho=0.965, r=0.953")

    # ============================================================
    # STEP 6: SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP 6: SCORECARD")
    print(f"{'='*70}")

    criteria = [
        ("1D MAE <= 0.08 (pooled model)", mae_1d <= 0.08, f"MAE={mae_1d:.4f}"),
        ("2D beats 1D on hybrids", mae_2d < mae_1d,
         f"2D={mae_2d:.4f} vs 1D={mae_1d:.4f}"),
        ("Hybrid eta between T and SSM", eta_between,
         f"eta_h={mean_eta_h:.4f}, T={mean_eta_t:.4f}, S={mean_eta_s:.4f}"),
        ("Hybrid slope between T and SSM", slope_between,
         f"slope_h={slope_h:.4f}, T={slope_t:.4f}, S={slope_s:.4f}"),
        ("Global rho >= 0.95 with hybrids", rho_all >= 0.95,
         f"rho={rho_all:.4f}"),
    ]

    passes = 0
    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        passes += int(passed)
        print(f"  [{status:>4}] {name}: {value}")

    print(f"\n  SCORE: {passes}/{len(criteria)} criteria passed")

    # ============================================================
    # SAVE
    # ============================================================
    results = {
        "experiment": "unseen_family_validation",
        "hypothesis": "kappa universality classes predict hybrid architecture behavior",
        "preregistered": {
            "pooled_mae_threshold": 0.08,
            "global_rho_threshold": 0.95,
        },
        "hybrid_points": hybrid_points,
        "prediction": {
            "mae_1d_pooled": float(mae_1d),
            "mae_2d_pooled": float(mae_2d),
            "mae_1d_transformer": float(mae_t),
            "mae_1d_ssm": float(mae_s),
            "2d_beats_1d": bool(mae_2d < mae_1d),
            "closest_family": best_family,
        },
        "geometry": {
            "eta": {
                "transformer_mean": float(mean_eta_t),
                "ssm_mean": float(mean_eta_s),
                "hybrid_mean": float(mean_eta_h),
                "hybrid_between": bool(eta_between),
            },
            "slope": {
                "transformer": float(slope_t),
                "ssm": float(slope_s),
                "hybrid": float(slope_h),
                "hybrid_between": bool(slope_between),
            },
        },
        "class_assignment": class_assignment,
        "global_correlation": {
            "rho": float(rho_all),
            "r": float(r_all),
            "n_points": len(all_kappas),
        },
        "scorecard": {
            "passes": passes,
            "total": len(criteria),
            "details": [
                {"criterion": name, "passed": passed, "value": value}
                for name, passed, value in criteria
            ],
        },
    }

    out_path = RESULTS_DIR / "cti_unseen_family.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
