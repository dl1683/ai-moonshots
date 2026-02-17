#!/usr/bin/env python
"""
2D LAW OF REPRESENTATION QUALITY

Codex design (post-7.3/10 review):
  The 1D law kNN ~ sigmoid(kappa) has architecture-dependent slopes.
  eta = tr(S_W)^2 / (d * tr(S_W^2)) captures within-class isotropy
  and explains WHY architectures differ (SSMs 2x more isotropic).

  The 2D law combines both:
    kNN = sigmoid(a*log(kappa) + b*log(eta) + c*log(kappa)*log(eta) + d)

  This allows non-multiplicative interactions between discriminability
  (kappa) and isotropy (eta).

Pre-registered criteria:
  1. LOMO (leave-one-model-out) MAE <= 0.05
  2. 2D model beats kappa-only sigmoid on OOD error
  3. Architecture dummy becomes non-significant (p > 0.05) after 2D terms
  4. LOAO (leave-one-architecture-out): train on transformers, predict SSMs
     and vice versa -- cross-paradigm MAE <= 0.10

Uses saved data from cti_geometry_mediator.json (63 points, 7 models).
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr, pearsonr, f as f_dist

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def sigmoid_1d(x, a, b, c, d_param):
    """Standard 1D sigmoid."""
    return d_param + (a - d_param) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def sigmoid_2d(features, a, b, c, d_param):
    """2D sigmoid: sigmoid(a*f1 + b*f2 + c*f1*f2 + d).

    features: (N, 3) array of [log_kappa, log_eta, log_kappa*log_eta]
    Output: sigmoid(a*log_k + b*log_e + c*log_k*log_e + d)
    """
    z = a * features[:, 0] + b * features[:, 1] + c * features[:, 2] + d_param
    return 1.0 / (1.0 + np.exp(np.clip(-z, -500, 500)))


def fit_sigmoid_2d(features, knns):
    """Fit 2D sigmoid and return predictions + params."""
    try:
        popt, _ = curve_fit(
            sigmoid_2d, features, knns,
            p0=[1.0, 0.5, 0.1, -1.0], maxfev=20000,
            bounds=([-10, -10, -10, -10], [10, 10, 10, 10])
        )
        pred = sigmoid_2d(features, *popt)
        return popt, pred
    except Exception:
        return None, None


def fit_sigmoid_1d(x, y):
    """Fit 1D sigmoid and return predictions + params."""
    try:
        popt, _ = curve_fit(
            sigmoid_1d, x, y,
            p0=[0.5, 10, 0.3, 0.1], maxfev=10000
        )
        pred = sigmoid_1d(x, *popt)
        return popt, pred
    except Exception:
        return None, None


def main():
    print("=" * 70)
    print("2D LAW OF REPRESENTATION QUALITY")
    print("kNN = sigmoid(a*log(kappa) + b*log(eta) + c*log(kappa)*log(eta) + d)")
    print("=" * 70)

    # Load saved data
    data_path = RESULTS_DIR / "cti_geometry_mediator.json"
    with open(data_path) as f:
        data = json.load(f)

    all_points = data["all_points"]
    print(f"Loaded {len(all_points)} data points from {data_path.name}")

    # Extract arrays
    kappas = np.array([p["kappa"] for p in all_points])
    knns = np.array([p["knn"] for p in all_points])
    etas = np.array([p["eta"] for p in all_points])
    paradigms = np.array([p["paradigm"] for p in all_points])
    models = np.array([p["model"] for p in all_points])

    # Ensure positive values for log
    kappas_safe = np.clip(kappas, 1e-6, None)
    etas_safe = np.clip(etas, 1e-6, None)

    log_kappas = np.log(kappas_safe)
    log_etas = np.log(etas_safe)
    interaction = log_kappas * log_etas

    features_2d = np.column_stack([log_kappas, log_etas, interaction])

    ss_tot = np.sum((knns - knns.mean()) ** 2)

    # ============================================================
    # 1. FULL-DATA FIT COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("1. FULL-DATA FIT COMPARISON")
    print(f"{'='*70}")

    # 1a. Baseline: kNN ~ sigmoid(kappa)
    popt_1d, pred_1d = fit_sigmoid_1d(kappas, knns)
    if pred_1d is not None:
        r2_1d = 1 - np.sum((knns - pred_1d)**2) / ss_tot
        mae_1d = np.mean(np.abs(knns - pred_1d))
    else:
        r2_1d = 0.0
        mae_1d = 1.0
    print(f"  kNN ~ sigmoid(kappa):     R^2={r2_1d:.4f}, MAE={mae_1d:.4f}")

    # 1b. 2D: kNN ~ sigmoid(a*log(k) + b*log(e) + c*log(k)*log(e) + d)
    popt_2d, pred_2d = fit_sigmoid_2d(features_2d, knns)
    if pred_2d is not None:
        r2_2d = 1 - np.sum((knns - pred_2d)**2) / ss_tot
        mae_2d = np.mean(np.abs(knns - pred_2d))
    else:
        r2_2d = 0.0
        mae_2d = 1.0
    print(f"  kNN ~ sigmoid(2D):        R^2={r2_2d:.4f}, MAE={mae_2d:.4f}")

    if popt_2d is not None:
        print(f"  Coefficients: a={popt_2d[0]:.4f} (log_kappa), "
              f"b={popt_2d[1]:.4f} (log_eta), "
              f"c={popt_2d[2]:.4f} (interaction), "
              f"d={popt_2d[3]:.4f} (bias)")

    # 1c. Per-architecture residuals
    print(f"\n  Per-architecture residuals (2D model):")
    for par in ["transformer", "ssm"]:
        mask = paradigms == par
        if pred_2d is not None:
            resid = knns[mask] - pred_2d[mask]
            print(f"    {par:>12}: mean_resid={resid.mean():.4f}, "
                  f"std={resid.std():.4f}, MAE={np.abs(resid).mean():.4f}")

    # ============================================================
    # 2. ARCHITECTURE DUMMY TEST (2D model)
    # ============================================================
    print(f"\n{'='*70}")
    print("2. ARCHITECTURE DUMMY TEST")
    print(f"{'='*70}")

    is_ssm = (paradigms == "ssm").astype(float)

    # Test multiple models with and without architecture dummy
    # Model A: kNN ~ linear(log_k, log_e, log_k*log_e)
    X_2d = np.column_stack([log_kappas, log_etas, interaction, np.ones(len(knns))])
    beta_2d = np.linalg.lstsq(X_2d, knns, rcond=None)[0]
    pred_lin_2d = X_2d @ beta_2d
    ss_res_2d = np.sum((knns - pred_lin_2d) ** 2)
    r2_lin_2d = 1 - ss_res_2d / ss_tot

    # Model B: kNN ~ linear(log_k, log_e, log_k*log_e, is_ssm)
    X_2d_arch = np.column_stack([log_kappas, log_etas, interaction,
                                  is_ssm, np.ones(len(knns))])
    beta_2d_arch = np.linalg.lstsq(X_2d_arch, knns, rcond=None)[0]
    pred_lin_2d_arch = X_2d_arch @ beta_2d_arch
    ss_res_2d_arch = np.sum((knns - pred_lin_2d_arch) ** 2)
    r2_lin_2d_arch = 1 - ss_res_2d_arch / ss_tot

    # Model C: kNN ~ linear(log_k, is_ssm)  -- 1D with dummy
    X_1d_arch = np.column_stack([log_kappas, is_ssm, np.ones(len(knns))])
    beta_1d_arch = np.linalg.lstsq(X_1d_arch, knns, rcond=None)[0]
    pred_1d_arch = X_1d_arch @ beta_1d_arch
    ss_res_1d_arch = np.sum((knns - pred_1d_arch) ** 2)

    # F-test: does adding is_ssm to 2D model help?
    n = len(knns)
    p_full = 5  # log_k, log_e, interaction, is_ssm, intercept
    p_red = 4   # log_k, log_e, interaction, intercept
    denom = ss_res_2d_arch / (n - p_full)
    if denom > 0:
        f_stat = ((ss_res_2d - ss_res_2d_arch) /
                  (p_full - p_red)) / denom
        p_dummy = 1 - f_dist.cdf(f_stat, p_full - p_red, n - p_full)
    else:
        f_stat = 0.0
        p_dummy = 1.0

    # Also F-test for 1D + dummy
    denom_1d = ss_res_1d_arch / (n - 3)
    X_1d_only = np.column_stack([log_kappas, np.ones(len(knns))])
    beta_1d_only = np.linalg.lstsq(X_1d_only, knns, rcond=None)[0]
    ss_res_1d_only = np.sum((knns - X_1d_only @ beta_1d_only) ** 2)
    f_1d = ((ss_res_1d_only - ss_res_1d_arch) / 1) / denom_1d if denom_1d > 0 else 0
    p_1d_dummy = 1 - f_dist.cdf(f_1d, 1, n - 3) if denom_1d > 0 else 1.0

    print(f"  LINEAR MODELS:")
    print(f"    kNN ~ log_k + log_e + interaction:       R^2={r2_lin_2d:.4f}")
    print(f"    kNN ~ log_k + log_e + interaction + ssm: R^2={r2_lin_2d_arch:.4f}")
    print(f"")
    print(f"  ARCHITECTURE DUMMY F-TESTS:")
    print(f"    1D (log_k only): arch F={f_1d:.4f}, p={p_1d_dummy:.6f}")
    print(f"    2D (+ log_e, interaction): arch F={f_stat:.4f}, p={p_dummy:.6f}")
    print(f"    Dummy coefficient in 2D+arch: {beta_2d_arch[3]:.4f}")
    print(f"    Pre-registered: 2D dummy p > 0.05")
    if p_dummy > 0.05:
        print(f"    ARCHITECTURE ABSORBED BY 2D: YES")
    else:
        print(f"    ARCHITECTURE STILL SIGNIFICANT: p={p_dummy:.6f}")
    print(f"    Improvement: p went from {p_1d_dummy:.6f} (1D) to "
          f"{p_dummy:.6f} (2D) -- "
          f"{'reduced' if p_dummy > p_1d_dummy else 'increased'}")

    # ============================================================
    # 3. LEAVE-ONE-MODEL-OUT (LOMO)
    # ============================================================
    print(f"\n{'='*70}")
    print("3. LEAVE-ONE-MODEL-OUT (LOMO)")
    print(f"{'='*70}")

    unique_models = sorted(set(p["model"] for p in all_points))

    lomo_1d = []
    lomo_2d = []

    for held_out in unique_models:
        train_mask = models != held_out
        test_mask = ~train_mask

        # 1D baseline
        popt_cv1, _ = fit_sigmoid_1d(kappas[train_mask], knns[train_mask])
        if popt_cv1 is not None:
            pred_cv1 = sigmoid_1d(kappas[test_mask], *popt_cv1)
            mae_cv1 = float(np.mean(np.abs(knns[test_mask] - pred_cv1)))
        else:
            mae_cv1 = 1.0

        # 2D model
        popt_cv2, _ = fit_sigmoid_2d(features_2d[train_mask], knns[train_mask])
        if popt_cv2 is not None:
            pred_cv2 = sigmoid_2d(features_2d[test_mask], *popt_cv2)
            mae_cv2 = float(np.mean(np.abs(knns[test_mask] - pred_cv2)))
        else:
            mae_cv2 = 1.0

        short = held_out.split("/")[-1]
        par = "ssm" if "mamba" in held_out.lower() else "trans"
        print(f"  {short:>20} ({par}): "
              f"1D MAE={mae_cv1:.4f}, 2D MAE={mae_cv2:.4f}, "
              f"{'2D WINS' if mae_cv2 < mae_cv1 else '1D WINS'}")

        lomo_1d.append({"model": held_out, "mae": mae_cv1})
        lomo_2d.append({"model": held_out, "mae": mae_cv2})

    mean_lomo_1d = np.mean([e["mae"] for e in lomo_1d])
    mean_lomo_2d = np.mean([e["mae"] for e in lomo_2d])
    wins_2d = sum(1 for a, b in zip(lomo_1d, lomo_2d) if b["mae"] < a["mae"])

    print(f"\n  Mean LOMO MAE (1D): {mean_lomo_1d:.4f}")
    print(f"  Mean LOMO MAE (2D): {mean_lomo_2d:.4f}")
    print(f"  2D wins: {wins_2d}/{len(unique_models)}")
    print(f"  Pre-registered: 2D LOMO MAE <= 0.05")
    if mean_lomo_2d <= 0.05:
        print(f"  LOMO PASS: YES")
    else:
        print(f"  LOMO PASS: NO (MAE={mean_lomo_2d:.4f})")

    # ============================================================
    # 4. LEAVE-ONE-ARCHITECTURE-OUT (LOAO)
    # ============================================================
    print(f"\n{'='*70}")
    print("4. LEAVE-ONE-ARCHITECTURE-OUT (LOAO)")
    print(f"{'='*70}")

    loao_results = {}
    for held_arch in ["transformer", "ssm"]:
        train_mask = paradigms != held_arch
        test_mask = ~train_mask

        # 1D
        popt_a1, _ = fit_sigmoid_1d(kappas[train_mask], knns[train_mask])
        if popt_a1 is not None:
            pred_a1 = sigmoid_1d(kappas[test_mask], *popt_a1)
            mae_a1 = float(np.mean(np.abs(knns[test_mask] - pred_a1)))
        else:
            mae_a1 = 1.0

        # 2D
        popt_a2, _ = fit_sigmoid_2d(features_2d[train_mask], knns[train_mask])
        if popt_a2 is not None:
            pred_a2 = sigmoid_2d(features_2d[test_mask], *popt_a2)
            mae_a2 = float(np.mean(np.abs(knns[test_mask] - pred_a2)))
        else:
            mae_a2 = 1.0

        train_arch = "ssm" if held_arch == "transformer" else "transformer"
        print(f"  Train on {train_arch:>12}, predict {held_arch:>12}:")
        print(f"    1D MAE = {mae_a1:.4f}")
        print(f"    2D MAE = {mae_a2:.4f}")
        print(f"    {'2D WINS' if mae_a2 < mae_a1 else '1D WINS'}")
        loao_results[held_arch] = {
            "mae_1d": mae_a1, "mae_2d": mae_a2,
            "2d_wins": bool(mae_a2 < mae_a1),
        }

    mean_loao_1d = np.mean([v["mae_1d"] for v in loao_results.values()])
    mean_loao_2d = np.mean([v["mae_2d"] for v in loao_results.values()])
    print(f"\n  Mean LOAO MAE (1D): {mean_loao_1d:.4f}")
    print(f"  Mean LOAO MAE (2D): {mean_loao_2d:.4f}")
    print(f"  Pre-registered: LOAO MAE <= 0.10")
    if mean_loao_2d <= 0.10:
        print(f"  LOAO PASS: YES")
    else:
        print(f"  LOAO PASS: NO")

    # ============================================================
    # 5. ABLATION: Which 2D terms matter?
    # ============================================================
    print(f"\n{'='*70}")
    print("5. ABLATION: Which 2D terms matter?")
    print(f"{'='*70}")

    # Try different feature subsets
    ablation_configs = {
        "log_kappa only": log_kappas.reshape(-1, 1),
        "log_eta only": log_etas.reshape(-1, 1),
        "log_kappa + log_eta": np.column_stack([log_kappas, log_etas]),
        "log_kappa + interaction": np.column_stack([log_kappas, interaction]),
        "log_eta + interaction": np.column_stack([log_etas, interaction]),
        "full 2D (all 3)": features_2d,
    }

    for name, feats in ablation_configs.items():
        X_ab = np.column_stack([feats, np.ones(len(knns))])
        beta_ab = np.linalg.lstsq(X_ab, knns, rcond=None)[0]
        pred_ab = X_ab @ beta_ab
        r2_ab = 1 - np.sum((knns - pred_ab)**2) / ss_tot

        # LOMO
        maes_ab = []
        for held_out in unique_models:
            train_m = models != held_out
            test_m = ~train_m
            X_tr = np.column_stack([feats[train_m], np.ones(train_m.sum())])
            X_te = np.column_stack([feats[test_m], np.ones(test_m.sum())])
            b_ab = np.linalg.lstsq(X_tr, knns[train_m], rcond=None)[0]
            p_ab = X_te @ b_ab
            maes_ab.append(np.mean(np.abs(knns[test_m] - p_ab)))

        mean_mae_ab = np.mean(maes_ab)
        print(f"  {name:>30}: R^2={r2_ab:.4f}, LOMO MAE={mean_mae_ab:.4f}")

    # ============================================================
    # 6. NONLINEAR ALTERNATIVES
    # ============================================================
    print(f"\n{'='*70}")
    print("6. NONLINEAR ALTERNATIVES")
    print(f"{'='*70}")

    # 6a. GP-like: kNN = a * kappa^b * eta^c + d
    def power_law(X, a, b, c, d_param):
        k, e = X
        return a * (k ** b) * (e ** c) + d_param

    try:
        popt_pl, _ = curve_fit(
            power_law, (kappas_safe, etas_safe), knns,
            p0=[1.0, 1.0, 0.5, 0.0], maxfev=20000
        )
        pred_pl = power_law((kappas_safe, etas_safe), *popt_pl)
        r2_pl = 1 - np.sum((knns - pred_pl)**2) / ss_tot
        mae_pl = np.mean(np.abs(knns - pred_pl))
        print(f"  Power law: kNN = {popt_pl[0]:.4f} * kappa^{popt_pl[1]:.4f} * "
              f"eta^{popt_pl[2]:.4f} + {popt_pl[3]:.4f}")
        print(f"    R^2={r2_pl:.4f}, MAE={mae_pl:.4f}")
    except Exception as e:
        print(f"  Power law fit failed: {e}")
        r2_pl = 0.0
        mae_pl = 1.0
        popt_pl = None

    # 6b. Additive sigmoid: kNN = sigmoid(a*kappa + b*eta + c)
    def additive_sigmoid(X, a, b, c):
        k, e = X
        z = a * k + b * e + c
        return 1.0 / (1.0 + np.exp(np.clip(-z, -500, 500)))

    try:
        popt_as, _ = curve_fit(
            additive_sigmoid, (kappas, etas), knns,
            p0=[5.0, 1.0, -2.0], maxfev=20000
        )
        pred_as = additive_sigmoid((kappas, etas), *popt_as)
        r2_as = 1 - np.sum((knns - pred_as)**2) / ss_tot
        mae_as = np.mean(np.abs(knns - pred_as))
        print(f"  Additive sigmoid: sigmoid({popt_as[0]:.4f}*k + "
              f"{popt_as[1]:.4f}*e + {popt_as[2]:.4f})")
        print(f"    R^2={r2_as:.4f}, MAE={mae_as:.4f}")
    except Exception as e:
        print(f"  Additive sigmoid fit failed: {e}")
        r2_as = 0.0

    # ============================================================
    # 7. SUMMARY SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("7. SUMMARY SCORECARD")
    print(f"{'='*70}")

    criteria = [
        ("LOMO MAE <= 0.05", mean_lomo_2d <= 0.05, f"{mean_lomo_2d:.4f}"),
        ("2D beats 1D on LOMO", mean_lomo_2d < mean_lomo_1d,
         f"2D={mean_lomo_2d:.4f} vs 1D={mean_lomo_1d:.4f}"),
        ("Arch dummy p > 0.05", p_dummy > 0.05, f"p={p_dummy:.6f}"),
        ("LOAO MAE <= 0.10", mean_loao_2d <= 0.10, f"{mean_loao_2d:.4f}"),
        ("2D R^2 > 1D R^2", r2_2d > r2_1d,
         f"2D={r2_2d:.4f} vs 1D={r2_1d:.4f}"),
    ]

    passes = 0
    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        passes += int(passed)
        print(f"  [{status:>4}] {name}: {value}")

    print(f"\n  SCORE: {passes}/{len(criteria)} criteria passed")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results = {
        "experiment": "2d_law_representation_quality",
        "hypothesis": "kNN = sigmoid(a*log(kappa) + b*log(eta) + c*log(kappa)*log(eta) + d)",
        "preregistered": {
            "lomo_mae_threshold": 0.05,
            "loao_mae_threshold": 0.10,
            "arch_dummy_p_threshold": 0.05,
        },
        "full_data": {
            "r2_1d": float(r2_1d),
            "r2_2d": float(r2_2d),
            "mae_1d": float(mae_1d),
            "mae_2d": float(mae_2d),
            "coefficients_2d": [float(x) for x in popt_2d] if popt_2d is not None else None,
        },
        "dummy_test": {
            "1d_f_stat": float(f_1d),
            "1d_p_value": float(p_1d_dummy),
            "2d_f_stat": float(f_stat),
            "2d_p_value": float(p_dummy),
            "2d_dummy_coeff": float(beta_2d_arch[3]),
            "arch_absorbed": bool(p_dummy > 0.05),
        },
        "lomo": {
            "1d": lomo_1d,
            "2d": lomo_2d,
            "mean_mae_1d": float(mean_lomo_1d),
            "mean_mae_2d": float(mean_lomo_2d),
            "2d_wins": int(wins_2d),
            "total": len(unique_models),
        },
        "loao": loao_results,
        "scorecard": {
            "passes": passes,
            "total": len(criteria),
            "details": [
                {"criterion": name, "passed": passed, "value": value}
                for name, passed, value in criteria
            ],
        },
        "n_points": len(all_points),
    }

    out_path = RESULTS_DIR / "cti_2d_law.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
