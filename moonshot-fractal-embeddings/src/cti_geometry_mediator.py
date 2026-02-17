#!/usr/bin/env python
"""
ARCHITECTURE GEOMETRY MEDIATOR COLLAPSE

Codex design (8.1/10 Nobel review):
  Same kappa gives different kNN because kappa ignores within-class spectral
  geometry. Define geometry-efficiency factor:
    eta = (tr(S_W))^2 / (d * tr(S_W^2))
  This is the isotropy/effective-rank fraction of within-class scatter.
  When within-class scatter is isotropic (all eigenvalues equal), eta=1.
  When it's collapsed onto few dimensions, eta -> 0.

  Hypothesis: q = sigmoid(a * kappa * eta + b) collapses the
  architecture-dependent slopes observed in critical_universality experiment.

Pre-registered criteria:
  - Slope ratio (SSM/transformer) closes to [0.9, 1.1] (from 1.6x baseline)
  - Architecture dummy non-significant (F-test p > 0.05)
  - Pooled R^2 >= 0.97 for kNN ~ sigmoid(kappa * eta)
  - Leave-one-model-out MAE <= 0.05

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
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr, f as f_dist

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
    """Compute kappa and eta from representations.

    kappa = trace(S_B) / trace(S_W)  -- Fisher discriminant ratio
    eta = (trace(S_W))^2 / (d * trace(S_W^2))  -- within-class isotropy

    eta measures how isotropic the within-class scatter is.
    When all eigenvalues of S_W are equal, eta = 1 (perfect isotropy).
    When scatter is concentrated on few dimensions, eta -> 0.

    Computed efficiently via SVD:
      Z = centered within-class data matrix (n x d)
      S_W = Z^T Z
      tr(S_W) = sum(s^2) where s = singular values of Z
      tr(S_W^2) = sum(s^4)
    """
    try:
        if np.isnan(X).any():
            return {"kappa": 0.0, "eta": 0.0, "trace_sb": 0.0,
                    "trace_sw": 0.0, "trace_sw_sq": 0.0, "d": X.shape[1]}

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
            return {"kappa": 100.0 if trace_sb > 0 else 0.0, "eta": 0.0,
                    "trace_sb": float(trace_sb), "trace_sw": float(trace_sw),
                    "trace_sw_sq": 0.0, "d": d}

        kappa = float(min(trace_sb / trace_sw, 100.0))

        # Compute eta via SVD of Z
        Z = np.concatenate(Z_parts, axis=0)  # (n, d)
        try:
            s = np.linalg.svd(Z, compute_uv=False)
            s2 = s ** 2
            s4 = s2 ** 2
            trace_sw_sq = float(s4.sum())

            if trace_sw_sq < 1e-20:
                eta = 0.0
            else:
                eta = float((trace_sw ** 2) / (d * trace_sw_sq))
        except np.linalg.LinAlgError:
            eta = 0.0
            trace_sw_sq = 0.0

        return {
            "kappa": kappa,
            "eta": eta,
            "trace_sb": float(trace_sb),
            "trace_sw": float(trace_sw),
            "trace_sw_sq": trace_sw_sq,
            "d": d,
        }
    except Exception:
        return {"kappa": 0.0, "eta": 0.0, "trace_sb": 0.0,
                "trace_sw": 0.0, "trace_sw_sq": 0.0, "d": 0}


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
    """Compute kNN, kappa, and eta averaged across all layers."""
    knn_vals, kappa_vals, eta_vals = [], [], []
    per_layer = []

    for layer_idx in sorted(reps.keys()):
        X = reps[layer_idx]
        if X.shape[0] < 20:
            continue
        knn_val = compute_knn(X, labels)
        stats = compute_kappa_and_eta(X, labels)
        knn_vals.append(knn_val)
        if np.isfinite(stats["kappa"]):
            kappa_vals.append(stats["kappa"])
        if np.isfinite(stats.get("eta", 0)):
            eta_vals.append(stats["eta"])
        per_layer.append({
            "layer": layer_idx,
            "knn": knn_val,
            "kappa": stats["kappa"],
            "eta": stats["eta"],
        })

    return {
        "knn": float(np.mean(knn_vals)) if knn_vals else 0,
        "kappa": float(np.mean(kappa_vals)) if kappa_vals else 0,
        "eta": float(np.mean(eta_vals)) if eta_vals else 0,
        "per_layer": per_layer,
    }


def sigmoid(x, a, b, c, d):
    """Generalized sigmoid."""
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def main():
    print("=" * 70)
    print("ARCHITECTURE GEOMETRY MEDIATOR COLLAPSE")
    print("Does eta = tr(S_W)^2 / (d * tr(S_W^2)) explain the slope gap?")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    # Load dataset once
    ds = load_hierarchical_dataset("clinc", split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])
    print(f"Dataset: CLINC150, {len(texts)} samples, "
          f"{len(np.unique(labels))} classes")

    # All 7 models (from MODEL_DIRECTORY.md)
    all_models = [
        "HuggingFaceTB/SmolLM2-360M",
        "EleutherAI/pythia-410m",
        "Qwen/Qwen2-0.5B",
        "Qwen/Qwen3-0.6B",
        "state-spaces/mamba-130m-hf",
        "state-spaces/mamba-370m-hf",
        "state-spaces/mamba-790m-hf",
    ]

    all_points = []

    for model_id in all_models:
        paradigm = "ssm" if "mamba" in model_id.lower() else "transformer"
        short = model_id.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short} ({paradigm})")
        print(f"{'='*70}")

        try:
            model, tokenizer, n_layers, n_params = load_model(model_id, device)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        for alpha in alphas:
            print(f"  alpha={alpha:.2f}", end="", flush=True)
            t0 = time.time()
            reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)
            stats = compute_layer_stats(reps, labels)
            elapsed = time.time() - t0
            print(f"  kNN={stats['knn']:.3f}  kappa={stats['kappa']:.4f}  "
                  f"eta={stats['eta']:.4f}  ({elapsed:.1f}s)")

            all_points.append({
                "model": model_id,
                "paradigm": paradigm,
                "n_layers": n_layers,
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

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("ANALYSIS: GEOMETRY MEDIATOR COLLAPSE")
    print(f"{'='*70}")

    kappas = np.array([p["kappa"] for p in all_points])
    knns = np.array([p["knn"] for p in all_points])
    etas = np.array([p["eta"] for p in all_points])
    kappa_etas = np.array([p["kappa_eta"] for p in all_points])
    paradigms = np.array([p["paradigm"] for p in all_points])

    ss_tot = np.sum((knns - knns.mean()) ** 2)

    # ---- 1. BASELINE: kappa vs kNN slopes by architecture ----
    print(f"\n--- 1. BASELINE: kappa vs kNN by architecture ---")
    for par in ["transformer", "ssm"]:
        mask = paradigms == par
        rho, p = spearmanr(kappas[mask], knns[mask])
        r, pr = pearsonr(kappas[mask], knns[mask])
        slope = np.polyfit(kappas[mask], knns[mask], 1)[0]
        print(f"  {par:>12}: slope={slope:.4f}, rho={rho:.4f}, "
              f"r={r:.4f} (N={mask.sum()})")

    slope_t = np.polyfit(
        kappas[paradigms == "transformer"],
        knns[paradigms == "transformer"], 1)[0]
    slope_s = np.polyfit(
        kappas[paradigms == "ssm"],
        knns[paradigms == "ssm"], 1)[0]
    ratio_baseline = slope_s / slope_t if abs(slope_t) > 1e-10 else float("inf")
    print(f"  Slope ratio (SSM/transformer): {ratio_baseline:.4f}")

    # ---- 2. MEDIATOR: kappa*eta vs kNN slopes ----
    print(f"\n--- 2. MEDIATOR: kappa*eta vs kNN by architecture ---")
    for par in ["transformer", "ssm"]:
        mask = paradigms == par
        rho, p = spearmanr(kappa_etas[mask], knns[mask])
        r, pr = pearsonr(kappa_etas[mask], knns[mask])
        slope = np.polyfit(kappa_etas[mask], knns[mask], 1)[0]
        print(f"  {par:>12}: slope={slope:.4f}, rho={rho:.4f}, "
              f"r={r:.4f} (N={mask.sum()})")

    slope_t2 = np.polyfit(
        kappa_etas[paradigms == "transformer"],
        knns[paradigms == "transformer"], 1)[0]
    slope_s2 = np.polyfit(
        kappa_etas[paradigms == "ssm"],
        knns[paradigms == "ssm"], 1)[0]
    ratio_mediator = (slope_s2 / slope_t2
                      if abs(slope_t2) > 1e-10 else float("inf"))
    print(f"  Slope ratio (SSM/transformer): {ratio_mediator:.4f}")
    print(f"  Pre-registered: ratio in [0.9, 1.1]")
    if 0.9 <= ratio_mediator <= 1.1:
        print(f"  SLOPES COLLAPSED: YES")
    else:
        print(f"  SLOPES COLLAPSED: NO")

    # ---- 3. ARCHITECTURE DUMMY REGRESSION ----
    print(f"\n--- 3. ARCHITECTURE DUMMY REGRESSION ---")
    is_ssm = (paradigms == "ssm").astype(float)

    # Full model: kNN = a * kappa_eta + b * is_ssm + c
    X_full = np.column_stack([kappa_etas, is_ssm, np.ones(len(kappa_etas))])
    beta_full = np.linalg.lstsq(X_full, knns, rcond=None)[0]
    pred_full = X_full @ beta_full
    ss_res_full = np.sum((knns - pred_full) ** 2)
    r2_full = 1 - ss_res_full / ss_tot

    # Reduced model: kNN = a * kappa_eta + c
    X_red = np.column_stack([kappa_etas, np.ones(len(kappa_etas))])
    beta_red = np.linalg.lstsq(X_red, knns, rcond=None)[0]
    pred_red = X_red @ beta_red
    ss_res_red = np.sum((knns - pred_red) ** 2)
    r2_red = 1 - ss_res_red / ss_tot

    # F-test for architecture dummy significance
    n = len(knns)
    p_full_params = 3
    p_red_params = 2
    denom = ss_res_full / (n - p_full_params)
    if denom > 0:
        f_stat = ((ss_res_red - ss_res_full) /
                  (p_full_params - p_red_params)) / denom
        p_dummy = 1 - f_dist.cdf(f_stat, p_full_params - p_red_params,
                                  n - p_full_params)
    else:
        f_stat = 0.0
        p_dummy = 1.0

    print(f"  R^2 with architecture dummy:    {r2_full:.4f}")
    print(f"  R^2 without architecture dummy: {r2_red:.4f}")
    print(f"  Architecture dummy coefficient: {beta_full[1]:.4f}")
    print(f"  F-test: F={f_stat:.4f}, p={p_dummy:.6f}")
    print(f"  Pre-registered: p > 0.05 (dummy non-significant)")
    if p_dummy > 0.05:
        print(f"  ARCHITECTURE IRRELEVANT: YES (p={p_dummy:.4f})")
    else:
        print(f"  ARCHITECTURE STILL MATTERS: p={p_dummy:.6f}")

    # Also test with just kappa (no eta) for comparison
    X_kappa_full = np.column_stack([kappas, is_ssm, np.ones(len(kappas))])
    beta_kf = np.linalg.lstsq(X_kappa_full, knns, rcond=None)[0]
    pred_kf = X_kappa_full @ beta_kf
    ss_res_kf = np.sum((knns - pred_kf) ** 2)

    X_kappa_red = np.column_stack([kappas, np.ones(len(kappas))])
    beta_kr = np.linalg.lstsq(X_kappa_red, knns, rcond=None)[0]
    pred_kr = X_kappa_red @ beta_kr
    ss_res_kr = np.sum((knns - pred_kr) ** 2)

    f_kappa = ((ss_res_kr - ss_res_kf) /
               (p_full_params - p_red_params)) / (ss_res_kf / (n - p_full_params))
    p_kappa_dummy = 1 - f_dist.cdf(f_kappa, p_full_params - p_red_params,
                                    n - p_full_params)

    print(f"\n  COMPARISON (kappa alone):")
    print(f"    Arch dummy F={f_kappa:.4f}, p={p_kappa_dummy:.6f}")
    print(f"    eta reduces dummy significance: "
          f"{'YES' if p_dummy > p_kappa_dummy else 'NO'}")

    # ---- 4. SIGMOID FIT ----
    print(f"\n--- 4. POOLED SIGMOID FIT ---")
    try:
        popt_k, _ = curve_fit(sigmoid, kappas, knns,
                              p0=[0.5, 10, 0.3, 0.1], maxfev=10000)
        pred_k = sigmoid(kappas, *popt_k)
        r2_kappa = 1 - np.sum((knns - pred_k)**2) / ss_tot
    except Exception:
        r2_kappa = 0.0

    try:
        popt_ke, _ = curve_fit(sigmoid, kappa_etas, knns,
                               p0=[0.5, 10, 0.1, 0.1], maxfev=10000)
        pred_ke = sigmoid(kappa_etas, *popt_ke)
        r2_kappa_eta = 1 - np.sum((knns - pred_ke)**2) / ss_tot
    except Exception:
        r2_kappa_eta = 0.0

    print(f"  kNN ~ sigmoid(kappa):      R^2 = {r2_kappa:.4f}")
    print(f"  kNN ~ sigmoid(kappa*eta):  R^2 = {r2_kappa_eta:.4f}")
    print(f"  Pre-registered: R^2 >= 0.97")
    if r2_kappa_eta >= 0.97:
        print(f"  POOLED R^2 PASS: YES")
    else:
        print(f"  POOLED R^2 PASS: NO")

    # ---- 5. LEAVE-ONE-MODEL-OUT ----
    print(f"\n--- 5. LEAVE-ONE-MODEL-OUT CROSS-VALIDATION ---")
    models_list = sorted(set(p["model"] for p in all_points))
    lomo_errors = []

    for held_out in models_list:
        train_mask = np.array([p["model"] != held_out for p in all_points])
        test_mask = ~train_mask

        ke_train = kappa_etas[train_mask]
        knn_train = knns[train_mask]
        ke_test = kappa_etas[test_mask]
        knn_test = knns[test_mask]

        try:
            popt_cv, _ = curve_fit(sigmoid, ke_train, knn_train,
                                   p0=[0.5, 10, 0.1, 0.1], maxfev=10000)
            pred_cv = sigmoid(ke_test, *popt_cv)
            mae = float(np.mean(np.abs(knn_test - pred_cv)))
        except Exception:
            mae = 1.0

        short = held_out.split("/")[-1]
        par = "ssm" if "mamba" in held_out.lower() else "transformer"
        print(f"  Hold out {short:>20} ({par:>4}): MAE = {mae:.4f}")
        lomo_errors.append({"model": held_out, "paradigm": par, "mae": mae})

    mean_lomo_mae = np.mean([e["mae"] for e in lomo_errors])
    print(f"\n  Mean LOMO MAE: {mean_lomo_mae:.4f}")
    print(f"  Pre-registered: MAE <= 0.05")
    if mean_lomo_mae <= 0.05:
        print(f"  LOMO PASS: YES")
    else:
        print(f"  LOMO PASS: NO")

    # ---- 6. ETA DISTRIBUTION ----
    print(f"\n--- 6. ETA DISTRIBUTION BY ARCHITECTURE ---")
    for par in ["transformer", "ssm"]:
        mask = paradigms == par
        e = etas[mask]
        print(f"  {par:>12}: mean={e.mean():.4f}, std={e.std():.4f}, "
              f"range=[{e.min():.4f}, {e.max():.4f}]")

    # ---- 7. ALTERNATIVE POWER LAWS ----
    print(f"\n--- 7. ALTERNATIVE MEDIATORS: kappa * eta^p ---")
    best_p = None
    best_deviation = float("inf")
    power_results = []

    for p_val in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        if p_val == 0.0:
            composite = kappas.copy()
        else:
            composite = kappas * (etas ** p_val)

        s_t = np.polyfit(
            composite[paradigms == "transformer"],
            knns[paradigms == "transformer"], 1)[0]
        s_s = np.polyfit(
            composite[paradigms == "ssm"],
            knns[paradigms == "ssm"], 1)[0]
        ratio = s_s / s_t if abs(s_t) > 1e-10 else float("inf")
        deviation = abs(ratio - 1.0)

        # Also compute pooled R^2 for sigmoid fit
        try:
            popt_p, _ = curve_fit(sigmoid, composite, knns,
                                  p0=[0.5, 10, 0.1, 0.1], maxfev=10000)
            pred_p = sigmoid(composite, *popt_p)
            r2_p = 1 - np.sum((knns - pred_p)**2) / ss_tot
        except Exception:
            r2_p = 0.0

        if deviation < best_deviation:
            best_deviation = deviation
            best_p = p_val

        status = " <-- BEST" if deviation == best_deviation else ""
        print(f"  p={p_val:.2f}: ratio={ratio:.4f}, "
              f"sigmoid R^2={r2_p:.4f}{status}")
        power_results.append({
            "p": p_val, "ratio": float(ratio),
            "sigmoid_r2": float(r2_p)
        })

    print(f"\n  Best power: p={best_p}, deviation from 1.0 = {best_deviation:.4f}")

    # ---- 8. GLOBAL CORRELATION COMPARISON ----
    print(f"\n--- 8. GLOBAL CORRELATIONS ---")
    rho_k, p_k = spearmanr(kappas, knns)
    rho_ke, p_ke = spearmanr(kappa_etas, knns)
    rho_e, p_e = spearmanr(etas, knns)
    r_k, _ = pearsonr(kappas, knns)
    r_ke, _ = pearsonr(kappa_etas, knns)
    r_e, _ = pearsonr(etas, knns)

    print(f"  kappa:      Spearman={rho_k:.4f}, Pearson={r_k:.4f}")
    print(f"  kappa*eta:  Spearman={rho_ke:.4f}, Pearson={r_ke:.4f}")
    print(f"  eta alone:  Spearman={rho_e:.4f}, Pearson={r_e:.4f}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results = {
        "experiment": "architecture_geometry_mediator_collapse",
        "hypothesis": "q = sigmoid(kappa * eta) collapses architecture slopes",
        "eta_definition": "eta = tr(S_W)^2 / (d * tr(S_W^2))",
        "preregistered": {
            "slope_ratio_range": [0.9, 1.1],
            "dummy_p_threshold": 0.05,
            "pooled_r2_threshold": 0.97,
            "lomo_mae_threshold": 0.05,
        },
        "all_points": all_points,
        "baseline": {
            "slope_transformer": float(slope_t),
            "slope_ssm": float(slope_s),
            "ratio": float(ratio_baseline),
        },
        "mediator": {
            "slope_transformer": float(slope_t2),
            "slope_ssm": float(slope_s2),
            "ratio": float(ratio_mediator),
            "ratio_in_range": bool(0.9 <= ratio_mediator <= 1.1),
        },
        "dummy_test": {
            "with_eta": {
                "r2_with_dummy": float(r2_full),
                "r2_without_dummy": float(r2_red),
                "dummy_coeff": float(beta_full[1]),
                "f_stat": float(f_stat),
                "p_value": float(p_dummy),
                "non_significant": bool(p_dummy > 0.05),
            },
            "kappa_only": {
                "f_stat": float(f_kappa),
                "p_value": float(p_kappa_dummy),
            },
            "eta_reduces_significance": bool(p_dummy > p_kappa_dummy),
        },
        "sigmoid_fit": {
            "r2_kappa": float(r2_kappa),
            "r2_kappa_eta": float(r2_kappa_eta),
            "improvement": float(r2_kappa_eta - r2_kappa),
        },
        "lomo": {
            "per_model": lomo_errors,
            "mean_mae": float(mean_lomo_mae),
            "pass": bool(mean_lomo_mae <= 0.05),
        },
        "eta_distribution": {
            par: {
                "mean": float(etas[paradigms == par].mean()),
                "std": float(etas[paradigms == par].std()),
                "min": float(etas[paradigms == par].min()),
                "max": float(etas[paradigms == par].max()),
            } for par in ["transformer", "ssm"]
        },
        "power_sweep": power_results,
        "best_power": {
            "p": float(best_p),
            "deviation": float(best_deviation),
        },
        "global_correlations": {
            "kappa": {"spearman": float(rho_k), "pearson": float(r_k)},
            "kappa_eta": {"spearman": float(rho_ke), "pearson": float(r_ke)},
            "eta": {"spearman": float(rho_e), "pearson": float(r_e)},
        },
        "n_points": len(all_points),
    }

    out_path = RESULTS_DIR / "cti_geometry_mediator.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
