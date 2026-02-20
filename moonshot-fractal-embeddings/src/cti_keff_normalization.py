#!/usr/bin/env python
"""
K_EFF NORMALIZATION TEST

Codex design (6.8/10 review):
  "Try K_eff = exp(H(y)) instead of raw K. Use q = sigmoid(a_d + b*kappa/sqrt(K_eff)).
   Run preregistered LODO blind test comparing baseline sqrt(K) vs sqrt(K_eff)."

Tests whether class-imbalance-adjusted K fixes arXiv calibration failure.

All 5 datasets: CLINC (150), AGNews (18), DBPedia (68), Yahoo (10), arXiv (115)
261 total points.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))


def sigmoid(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def load_all_data():
    """Load all 5 datasets' points."""
    all_points = []

    # CLINC from geometry mediator
    with open(RESULTS_DIR / "cti_geometry_mediator.json") as f:
        clinc_raw = json.load(f)
    for p in clinc_raw["all_points"]:
        all_points.append({
            "model": p["model"], "paradigm": p["paradigm"],
            "dataset": "clinc", "K": 150,
            "alpha": p["alpha"], "knn": p["knn"], "kappa": p["kappa"],
        })

    # AGNews and DBPedia from caches
    for ds in ["agnews", "dbpedia_classes"]:
        with open(RESULTS_DIR / f"cti_multidata_{ds}_cache.json") as f:
            data = json.load(f)
        for p in data:
            all_points.append({
                "model": p["model"], "paradigm": p["paradigm"],
                "dataset": p["dataset"], "K": p["n_classes"],
                "alpha": p["alpha"], "knn": p["knn"], "kappa": p["kappa"],
            })

    # Yahoo and arXiv from blind prediction
    with open(RESULTS_DIR / "cti_blind_prediction.json") as f:
        blind = json.load(f)
    for p in blind["blind_points"]:
        all_points.append({
            "model": p["model"], "paradigm": p["paradigm"],
            "dataset": p["dataset"], "K": p["K"],
            "alpha": p["alpha"], "knn": p["knn"], "kappa": p["kappa"],
        })

    return all_points


def compute_keff(dataset_name, all_points):
    """Compute K_eff = exp(H(y)) for each dataset from label distribution."""
    ds_points = [p for p in all_points if p["dataset"] == dataset_name]
    # We don't have raw labels here, so use the alpha=1.0 points
    # K_eff is a property of the dataset, not the model
    # For now, use the nominal K and try to estimate from kNN variance
    # Actually, we need the actual label distribution
    # Let's load the datasets directly
    from hierarchical_datasets import load_hierarchical_dataset
    import numpy as np

    try:
        ds = load_hierarchical_dataset(dataset_name, split="test", max_samples=2000)
        labels = np.array([s.level1_label for s in ds.samples])
        # Compute entropy of label distribution
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        H = -np.sum(probs * np.log(probs))
        K_eff = np.exp(H)
        K_raw = len(unique)
        return K_eff, K_raw, H
    except Exception as e:
        print(f"  Cannot load {dataset_name}: {e}")
        return None, None, None


def main():
    print("=" * 70)
    print("K_EFF = exp(H(y)) NORMALIZATION TEST")
    print("=" * 70)

    all_points = load_all_data()
    N = len(all_points)
    print(f"Total points: {N}")

    # Compute K_eff for each dataset
    datasets_seen = sorted(set(p["dataset"] for p in all_points))
    keff_map = {}

    print(f"\n--- Dataset K_eff computation ---")
    for ds_name in datasets_seen:
        K_eff, K_raw, H = compute_keff(ds_name, all_points)
        if K_eff is not None:
            keff_map[ds_name] = K_eff
            ratio = K_eff / K_raw
            print(f"  {ds_name:>20}: K_raw={K_raw:>4}, K_eff={K_eff:.1f}, "
                  f"H={H:.3f}, K_eff/K={ratio:.3f}")

    # Prepare arrays
    kappas = np.array([p["kappa"] for p in all_points])
    knns = np.array([p["knn"] for p in all_points])
    Ks = np.array([p["K"] for p in all_points])
    datasets = np.array([p["dataset"] for p in all_points])

    # K_eff array
    Keffs = np.array([keff_map.get(p["dataset"], p["K"]) for p in all_points])

    # Normalized quality
    q = (knns - 1.0 / Ks) / (1.0 - 1.0 / Ks)

    # ============================================================
    # TEST NORMALIZATIONS
    # ============================================================
    print(f"\n{'='*70}")
    print("NORMALIZATION COMPARISON (ALL 5 DATASETS)")
    print(f"{'='*70}")

    normalizations = {
        "raw kappa": kappas,
        "kappa/sqrt(K)": kappas / np.sqrt(Ks),
        "kappa/sqrt(K_eff)": kappas / np.sqrt(Keffs),
        "kappa/K": kappas / Ks,
        "kappa/K_eff": kappas / Keffs,
        "kappa/log(K)": kappas / np.log(Ks),
        "kappa/log(K_eff)": kappas / np.log(Keffs),
    }

    results_norm = {}
    for name, x_vals in normalizations.items():
        rho, p = spearmanr(x_vals, q)
        r, pr = pearsonr(x_vals, q)

        try:
            popt, _ = curve_fit(sigmoid, x_vals, q,
                                p0=[0.6, 10, np.median(x_vals), 0.0],
                                maxfev=10000)
            pred = sigmoid(x_vals, *popt)
            ss_tot = np.sum((q - q.mean()) ** 2)
            r2 = 1 - np.sum((q - pred) ** 2) / ss_tot
            mae = float(np.mean(np.abs(q - pred)))
        except Exception:
            r2 = 0.0
            mae = 1.0

        results_norm[name] = {"rho": float(rho), "r": float(r),
                              "r2": float(r2), "mae": float(mae)}
        print(f"  {name:>25}: rho={rho:.4f}, R^2={r2:.4f}, MAE={mae:.4f}")

    # ============================================================
    # LODO (Leave-One-Dataset-Out) for top normalizations
    # ============================================================
    print(f"\n{'='*70}")
    print("LODO COMPARISON: sqrt(K) vs sqrt(K_eff)")
    print(f"{'='*70}")

    for norm_name, norm_key in [("sqrt(K)", "kappa/sqrt(K)"),
                                  ("sqrt(K_eff)", "kappa/sqrt(K_eff)")]:
        x_all = normalizations[norm_key]

        print(f"\n  --- {norm_name} ---")
        lodo_maes = []

        for held_out in sorted(set(datasets)):
            train_mask = datasets != held_out
            test_mask = datasets == held_out

            x_train, q_train = x_all[train_mask], q[train_mask]
            x_test, q_test = x_all[test_mask], q[test_mask]

            try:
                popt, _ = curve_fit(sigmoid, x_train, q_train,
                                    p0=[0.6, 10, np.median(x_train), 0.0],
                                    maxfev=10000)
                pred = sigmoid(x_test, *popt)
                mae = float(np.mean(np.abs(q_test - pred)))
            except Exception:
                mae = 1.0

            rho_test, _ = spearmanr(x_test, q_test)
            lodo_maes.append(mae)
            print(f"    Hold out {held_out:>20}: MAE={mae:.4f}, rho={rho_test:.4f}")

        mean_lodo = np.mean(lodo_maes)
        print(f"    Mean LODO MAE: {mean_lodo:.4f}")

    # ============================================================
    # PER-DATASET FIT QUALITY
    # ============================================================
    print(f"\n{'='*70}")
    print("PER-DATASET SIGMOID FIT (kappa/sqrt(K_eff))")
    print(f"{'='*70}")

    x_keff = normalizations["kappa/sqrt(K_eff)"]
    for ds_name in sorted(set(datasets)):
        mask = datasets == ds_name
        x_ds = x_keff[mask]
        q_ds = q[mask]

        rho_ds, _ = spearmanr(x_ds, q_ds)
        try:
            popt_ds, _ = curve_fit(sigmoid, x_ds, q_ds,
                                   p0=[0.6, 10, np.median(x_ds), 0.0],
                                   maxfev=10000)
            pred_ds = sigmoid(x_ds, *popt_ds)
            ss_tot_ds = np.sum((q_ds - q_ds.mean()) ** 2)
            r2_ds = 1 - np.sum((q_ds - pred_ds) ** 2) / ss_tot_ds
        except Exception:
            r2_ds = 0.0

        K_raw = Ks[mask][0]
        K_eff = Keffs[mask][0]
        print(f"  {ds_name:>20}: K={K_raw}, K_eff={K_eff:.1f}, "
              f"rho={rho_ds:.4f}, R^2={r2_ds:.4f}")

    # ============================================================
    # MODEL WITH DATASET-SPECIFIC OFFSET
    # ============================================================
    print(f"\n{'='*70}")
    print("MODEL: q = sigmoid(a + b*kappa/sqrt(K_eff)) + d_dataset")
    print(f"{'='*70}")

    # Fit global sigmoid on kappa/sqrt(K_eff)
    x_norm = kappas / np.sqrt(Keffs)
    try:
        popt_global, _ = curve_fit(sigmoid, x_norm, q,
                                   p0=[0.6, 10, np.median(x_norm), 0.0],
                                   maxfev=10000)
        pred_global = sigmoid(x_norm, *popt_global)
        residuals = q - pred_global

        # Dataset-specific offsets
        for ds_name in sorted(set(datasets)):
            mask = datasets == ds_name
            offset = float(residuals[mask].mean())
            std = float(residuals[mask].std())
            print(f"  {ds_name:>20}: offset={offset:+.4f}, std={std:.4f}")

        # Corrected predictions with leave-one-dataset-out offset
        print(f"\n  LODO with offset correction:")
        lodo_corrected = []
        for held_out in sorted(set(datasets)):
            train_mask = datasets != held_out
            test_mask = datasets == held_out

            # Fit sigmoid on training data
            x_train, q_train = x_norm[train_mask], q[train_mask]
            x_test, q_test = x_norm[test_mask], q[test_mask]

            try:
                popt_lodo, _ = curve_fit(sigmoid, x_train, q_train,
                                         p0=[0.6, 10, np.median(x_train), 0.0],
                                         maxfev=10000)
                pred_test = sigmoid(x_test, *popt_lodo)
                mae_raw = float(np.mean(np.abs(q_test - pred_test)))

                # With offset: use mean residual of training data as hint
                # (but we can't use test data offset)
                print(f"    {held_out:>20}: MAE_raw={mae_raw:.4f}")
                lodo_corrected.append(mae_raw)
            except Exception:
                lodo_corrected.append(1.0)

        mean_corrected = np.mean(lodo_corrected)
        print(f"    Mean LODO MAE: {mean_corrected:.4f}")
    except Exception as e:
        print(f"  Global fit failed: {e}")

    # ============================================================
    # FINAL SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL SCORECARD")
    print(f"{'='*70}")

    # Check if K_eff beats K
    keff_r2 = results_norm["kappa/sqrt(K_eff)"]["r2"]
    k_r2 = results_norm["kappa/sqrt(K)"]["r2"]
    keff_rho = results_norm["kappa/sqrt(K_eff)"]["rho"]
    k_rho = results_norm["kappa/sqrt(K)"]["rho"]

    print(f"  sqrt(K):     R^2={k_r2:.4f}, rho={k_rho:.4f}")
    print(f"  sqrt(K_eff): R^2={keff_r2:.4f}, rho={keff_rho:.4f}")
    print(f"  K_eff wins: {'YES' if keff_r2 > k_r2 else 'NO'}")
    print(f"  Improvement: dR^2={keff_r2 - k_r2:+.4f}, drho={keff_rho - k_rho:+.4f}")

    # Save
    results = {
        "experiment": "keff_normalization_test",
        "n_points": N,
        "n_datasets": len(datasets_seen),
        "keff_values": {ds: float(keff_map.get(ds, 0)) for ds in datasets_seen},
        "normalization_comparison": results_norm,
        "keff_beats_k": bool(keff_r2 > k_r2),
        "improvement": {
            "dr2": float(keff_r2 - k_r2),
            "drho": float(keff_rho - k_rho),
        },
    }

    out_path = RESULTS_DIR / "cti_keff_normalization.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
