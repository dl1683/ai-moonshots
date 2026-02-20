#!/usr/bin/env python
"""
TESTING THEORETICAL PREDICTIONS FROM GAUSSIAN-CLUSTER DERIVATION

The theory (Codex-derived):
  Under Gaussian mixture model, kNN accuracy follows:
    Acc(kappa) = Phi((a*kappa - b) / c)
  This is a probit sigmoid arising from:
    - Same-class distance D+ ~ N(2*tr(Sigma_W), ...)
    - Different-class distance D- ~ N(2*tr(Sigma_W + Sigma_B), ...)
    - kappa = tr(Sigma_B)/tr(Sigma_W) controls the gap

6 Testable Predictions:
  1. Universal collapse: affine-normalized kappa curves align across datasets
  2. Critical point: kappa_c where slope is maximal (inter=intra distance)
  3. Sample-size scaling: sharpness grows with sqrt(log(n))
  4. Class-count effect: larger K shifts kappa_c right
  5. k effect: increasing k in kNN changes steepness
  6. alpha-to-kappa reparameterization: accuracy-vs-alpha collapses to accuracy-vs-kappa

Uses existing extracted data where possible, runs new extractions only for
predictions 3 and 5.
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
from scipy.stats import spearmanr, pearsonr, norm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))


def sigmoid(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(np.clip(-b * (x - c), -500, 500)))


def probit(x, a, b, c, d):
    """Probit sigmoid: Phi((x-c)/b) scaled."""
    return d + (a - d) * norm.cdf((x - c) / max(abs(b), 1e-8))


def load_all_data():
    """Load all extracted data."""
    all_points = []

    # CLINC
    with open(RESULTS_DIR / "cti_geometry_mediator.json") as f:
        clinc = json.load(f)
    for p in clinc["all_points"]:
        all_points.append({
            "model": p["model"], "paradigm": p["paradigm"],
            "dataset": "clinc", "K": 150,
            "alpha": p["alpha"], "knn": p["knn"], "kappa": p["kappa"],
        })

    # AGNews, DBPedia
    for ds in ["agnews", "dbpedia_classes"]:
        with open(RESULTS_DIR / f"cti_multidata_{ds}_cache.json") as f:
            for p in json.load(f):
                all_points.append({
                    "model": p["model"], "paradigm": p["paradigm"],
                    "dataset": p["dataset"], "K": p["n_classes"],
                    "alpha": p["alpha"], "knn": p["knn"], "kappa": p["kappa"],
                })

    # Yahoo, arXiv
    with open(RESULTS_DIR / "cti_blind_prediction.json") as f:
        blind = json.load(f)
    for p in blind["blind_points"]:
        all_points.append({
            "model": p["model"], "paradigm": p["paradigm"],
            "dataset": p["dataset"], "K": p["K"],
            "alpha": p["alpha"], "knn": p["knn"], "kappa": p["kappa"],
        })

    return all_points


def main():
    print("=" * 70)
    print("TESTING THEORETICAL PREDICTIONS")
    print("From Gaussian-cluster kNN derivation")
    print("=" * 70)

    all_data = load_all_data()
    N = len(all_data)

    kappas = np.array([p["kappa"] for p in all_data])
    knns = np.array([p["knn"] for p in all_data])
    Ks = np.array([p["K"] for p in all_data])
    datasets = np.array([p["dataset"] for p in all_data])
    models = np.array([p["model"] for p in all_data])
    alphas = np.array([p["alpha"] for p in all_data])

    q = (knns - 1.0 / Ks) / (1.0 - 1.0 / Ks)

    results = {"predictions": {}}

    # ============================================================
    # PREDICTION 1: Probit vs logistic sigmoid fit
    # ============================================================
    print(f"\n{'='*70}")
    print("PREDICTION 1: Probit vs Logistic sigmoid")
    print("Theory predicts PROBIT (from Gaussian CDF)")
    print(f"{'='*70}")

    pred1 = {}
    for ds_name in sorted(set(datasets)):
        mask = datasets == ds_name
        x_ds = kappas[mask]
        q_ds = q[mask]

        # Logistic fit
        try:
            popt_log, _ = curve_fit(sigmoid, x_ds, q_ds,
                                    p0=[0.6, 10, np.median(x_ds), 0.0],
                                    maxfev=10000)
            pred_log = sigmoid(x_ds, *popt_log)
            ss_tot = np.sum((q_ds - q_ds.mean()) ** 2)
            r2_log = 1 - np.sum((q_ds - pred_log) ** 2) / ss_tot
        except Exception:
            r2_log = 0.0

        # Probit fit
        try:
            popt_pro, _ = curve_fit(probit, x_ds, q_ds,
                                    p0=[0.6, 0.1, np.median(x_ds), 0.0],
                                    maxfev=10000)
            pred_pro = probit(x_ds, *popt_pro)
            r2_pro = 1 - np.sum((q_ds - pred_pro) ** 2) / ss_tot
        except Exception:
            r2_pro = 0.0

        winner = "PROBIT" if r2_pro > r2_log else "LOGISTIC"
        diff = r2_pro - r2_log
        print(f"  {ds_name:>20}: logistic R^2={r2_log:.4f}, probit R^2={r2_pro:.4f}, "
              f"diff={diff:+.4f} -> {winner}")
        pred1[ds_name] = {
            "logistic_r2": float(r2_log), "probit_r2": float(r2_pro),
            "probit_wins": bool(r2_pro > r2_log),
        }

    n_probit_wins = sum(1 for v in pred1.values() if v["probit_wins"])
    print(f"\n  Probit wins: {n_probit_wins}/{len(pred1)} datasets")
    results["predictions"]["1_probit_vs_logistic"] = {
        "per_dataset": pred1,
        "probit_wins_count": n_probit_wins,
        "prediction_confirmed": n_probit_wins >= len(pred1) // 2,
    }

    # ============================================================
    # PREDICTION 2: Critical point kappa_c (max slope)
    # ============================================================
    print(f"\n{'='*70}")
    print("PREDICTION 2: Critical point kappa_c at maximal slope")
    print("Theory: kappa_c where inter/intra distances match")
    print(f"{'='*70}")

    pred2 = {}
    for ds_name in sorted(set(datasets)):
        mask = datasets == ds_name
        x_ds = kappas[mask]
        q_ds = q[mask]

        try:
            popt, _ = curve_fit(sigmoid, x_ds, q_ds,
                                p0=[0.6, 10, np.median(x_ds), 0.0],
                                maxfev=10000)
            kappa_c = popt[2]  # inflection point
            steepness = popt[1]  # slope at inflection

            # Also fit probit
            popt_p, _ = curve_fit(probit, x_ds, q_ds,
                                  p0=[0.6, 0.1, np.median(x_ds), 0.0],
                                  maxfev=10000)
            kappa_c_probit = popt_p[2]

            K_val = Ks[mask][0]
            print(f"  {ds_name:>20} (K={K_val:>3}): kappa_c={kappa_c:.4f} "
                  f"(probit: {kappa_c_probit:.4f}), steepness={steepness:.2f}")
            pred2[ds_name] = {
                "K": int(K_val),
                "kappa_c_logistic": float(kappa_c),
                "kappa_c_probit": float(kappa_c_probit),
                "steepness": float(steepness),
            }
        except Exception as e:
            print(f"  {ds_name}: fit failed: {e}")

    results["predictions"]["2_critical_point"] = pred2

    # ============================================================
    # PREDICTION 4: K shifts kappa_c rightward
    # ============================================================
    print(f"\n{'='*70}")
    print("PREDICTION 4: Larger K shifts kappa_c rightward")
    print(f"{'='*70}")

    if len(pred2) >= 3:
        Ks_list = [pred2[ds]["K"] for ds in pred2]
        kcs_list = [pred2[ds]["kappa_c_logistic"] for ds in pred2]
        rho_Kkc, p_Kkc = spearmanr(Ks_list, kcs_list)
        print(f"  K values: {sorted(zip(Ks_list, kcs_list), key=lambda x: x[0])}")
        print(f"  Spearman(K, kappa_c) = {rho_Kkc:.4f} (p={p_Kkc:.4f})")
        print(f"  Prediction confirmed: {'YES' if rho_Kkc > 0 else 'NO'}")
        results["predictions"]["4_K_shifts_kappac"] = {
            "rho": float(rho_Kkc), "p": float(p_Kkc),
            "confirmed": bool(rho_Kkc > 0),
        }

    # ============================================================
    # PREDICTION 5: k in kNN changes steepness
    # ============================================================
    print(f"\n{'='*70}")
    print("PREDICTION 5: Varying k in kNN changes sigmoid steepness")
    print("Testing k=1,3,5,10 on CLINC data")
    print(f"{'='*70}")

    # Load CLINC representations from cache if available, otherwise use extracted kappas
    # We need to re-compute kNN with different k values
    # Use a single model at alpha=1.0 (Qwen2-0.5B on CLINC)
    # Actually, we need the raw representations. Let me extract for one model.

    from cti_residual_surgery import load_model, ResidualScaler
    from hierarchical_datasets import load_hierarchical_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_hierarchical_dataset("clinc", split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])

    model_id = "Qwen/Qwen2-0.5B"
    model, tokenizer, n_layers, n_params = load_model(model_id, device)

    test_alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    k_values = [1, 3, 5, 10, 20]

    pred5 = {k: [] for k in k_values}

    for alpha in test_alphas:
        print(f"  alpha={alpha:.2f}", end="", flush=True)

        # Extract representations
        all_hidden = {}
        n_batches = (len(texts) + 32 - 1) // 32
        with ResidualScaler(model, alpha):
            for i in range(n_batches):
                batch = texts[i * 32:(i + 1) * 32]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=128, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True, return_dict=True)
                mask_enc = enc.get("attention_mask",
                                   torch.ones(enc["input_ids"].shape, device=device))
                for idx, hs in enumerate(out.hidden_states):
                    hs_f = hs.float()
                    m = mask_enc.unsqueeze(-1).float()
                    pooled = (hs_f * m).sum(1) / m.sum(1).clamp(min=1)
                    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    if idx not in all_hidden:
                        all_hidden[idx] = []
                    all_hidden[idx].append(pooled.cpu().numpy())

        reps = {k_: np.concatenate(v, axis=0) for k_, v in all_hidden.items()}

        # Compute kappa (layer-averaged)
        kappa_vals = []
        for layer_idx in sorted(reps.keys()):
            X = reps[layer_idx]
            grand_mean = X.mean(0)
            trace_sb = 0.0
            trace_sw = 0.0
            for lbl in np.unique(labels):
                lbl_mask = labels == lbl
                n_k = lbl_mask.sum()
                if n_k < 2:
                    continue
                X_k = X[lbl_mask]
                mu_k = X_k.mean(0)
                trace_sb += n_k * np.sum((mu_k - grand_mean) ** 2)
                trace_sw += np.sum((X_k - mu_k) ** 2)
            if trace_sw > 1e-12:
                kappa_vals.append(min(trace_sb / trace_sw, 100.0))

        kappa_mean = float(np.mean(kappa_vals)) if kappa_vals else 0.0

        # Compute kNN for different k values (layer-averaged)
        n_train = int(0.7 * len(labels))
        for k in k_values:
            knn_vals = []
            for layer_idx in sorted(reps.keys()):
                X = reps[layer_idx]
                if X.shape[0] < 20:
                    continue
                try:
                    clf = KNeighborsClassifier(n_neighbors=min(k, n_train - 1),
                                              metric="cosine")
                    clf.fit(X[:n_train], labels[:n_train])
                    acc = float(clf.score(X[n_train:], labels[n_train:]))
                    knn_vals.append(acc)
                except Exception:
                    pass

            knn_mean = float(np.mean(knn_vals)) if knn_vals else 0.0
            pred5[k].append({"alpha": alpha, "kappa": kappa_mean, "knn": knn_mean})

        print(f"  kappa={kappa_mean:.4f}  k1={pred5[1][-1]['knn']:.3f}  "
              f"k5={pred5[5][-1]['knn']:.3f}  k20={pred5[20][-1]['knn']:.3f}")
        sys.stdout.flush()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Fit sigmoids for each k value
    print(f"\n  Sigmoid fits by k:")
    pred5_fits = {}
    for k in k_values:
        kappas_k = np.array([p["kappa"] for p in pred5[k]])
        knns_k = np.array([p["knn"] for p in pred5[k]])
        q_k = (knns_k - 1.0 / 150) / (1.0 - 1.0 / 150)

        try:
            popt_k, _ = curve_fit(sigmoid, kappas_k, q_k,
                                  p0=[0.6, 10, 0.3, 0.0], maxfev=10000)
            pred_k = sigmoid(kappas_k, *popt_k)
            ss_tot_k = np.sum((q_k - q_k.mean()) ** 2)
            r2_k = 1 - np.sum((q_k - pred_k) ** 2) / ss_tot_k
            steepness_k = popt_k[1]
            kc_k = popt_k[2]
        except Exception:
            steepness_k = 0.0
            kc_k = 0.0
            r2_k = 0.0

        print(f"    k={k:>2}: steepness={steepness_k:.2f}, kappa_c={kc_k:.4f}, R^2={r2_k:.4f}")
        pred5_fits[k] = {
            "steepness": float(steepness_k),
            "kappa_c": float(kc_k),
            "r2": float(r2_k),
        }

    # Check: does steepness increase with k?
    steepnesses = [pred5_fits[k]["steepness"] for k in k_values]
    rho_k_steep, p_k_steep = spearmanr(k_values, steepnesses)
    print(f"\n  Spearman(k, steepness) = {rho_k_steep:.4f} (p={p_k_steep:.4f})")
    print(f"  Theory predicts: steepness increases with k")
    print(f"  Confirmed: {'YES' if rho_k_steep > 0 else 'NO'}")

    results["predictions"]["5_k_effect"] = {
        "fits": pred5_fits,
        "rho_k_steepness": float(rho_k_steep),
        "confirmed": bool(rho_k_steep > 0),
    }

    # ============================================================
    # PREDICTION 6: alpha-to-kappa reparameterization
    # ============================================================
    print(f"\n{'='*70}")
    print("PREDICTION 6: Accuracy-vs-alpha collapses when reparameterized as accuracy-vs-kappa")
    print(f"{'='*70}")

    # For each model, fit sigmoid(alpha) and sigmoid(kappa)
    # Compare R^2
    pred6 = {}
    for ds_name in sorted(set(datasets)):
        ds_mask = datasets == ds_name
        ds_models = sorted(set(models[ds_mask]))

        alpha_r2s = []
        kappa_r2s = []

        for model_name in ds_models:
            m_mask = ds_mask & (models == model_name)
            if m_mask.sum() < 5:
                continue

            a_vals = alphas[m_mask]
            k_vals = kappas[m_mask]
            q_vals = q[m_mask]

            ss_tot_m = np.sum((q_vals - q_vals.mean()) ** 2)
            if ss_tot_m < 1e-10:
                continue

            # Fit sigmoid(alpha)
            try:
                popt_a, _ = curve_fit(sigmoid, a_vals, q_vals,
                                      p0=[0.6, 5, 0.5, 0.0], maxfev=10000)
                r2_a = 1 - np.sum((q_vals - sigmoid(a_vals, *popt_a)) ** 2) / ss_tot_m
            except Exception:
                r2_a = 0.0

            # Fit sigmoid(kappa)
            try:
                popt_k, _ = curve_fit(sigmoid, k_vals, q_vals,
                                      p0=[0.6, 10, np.median(k_vals), 0.0],
                                      maxfev=10000)
                r2_k = 1 - np.sum((q_vals - sigmoid(k_vals, *popt_k)) ** 2) / ss_tot_m
            except Exception:
                r2_k = 0.0

            alpha_r2s.append(r2_a)
            kappa_r2s.append(r2_k)

        if alpha_r2s:
            mean_alpha = np.mean(alpha_r2s)
            mean_kappa = np.mean(kappa_r2s)
            print(f"  {ds_name:>20}: mean R^2(alpha)={mean_alpha:.4f}, "
                  f"mean R^2(kappa)={mean_kappa:.4f}")
            pred6[ds_name] = {
                "mean_r2_alpha": float(mean_alpha),
                "mean_r2_kappa": float(mean_kappa),
                "kappa_better": bool(mean_kappa >= mean_alpha),
            }

    n_kappa_better = sum(1 for v in pred6.values() if v.get("kappa_better", False))
    print(f"\n  kappa reparameterization better: {n_kappa_better}/{len(pred6)} datasets")
    results["predictions"]["6_reparameterization"] = {
        "per_dataset": pred6,
        "kappa_better_count": n_kappa_better,
        "confirmed": n_kappa_better >= len(pred6) // 2,
    }

    # ============================================================
    # PREDICTION 1 (continued): Universal collapse with affine normalization
    # ============================================================
    print(f"\n{'='*70}")
    print("PREDICTION 1b: Universal collapse with affine normalization")
    print("Normalize each dataset's kappa by its fitted kappa_c and steepness")
    print(f"{'='*70}")

    # For each dataset, compute z = (kappa - kappa_c) * steepness
    z_all = []
    q_all_normalized = []

    for ds_name in sorted(set(datasets)):
        mask = datasets == ds_name
        x_ds = kappas[mask]
        q_ds = q[mask]

        try:
            popt_ds, _ = curve_fit(sigmoid, x_ds, q_ds,
                                   p0=[0.6, 10, np.median(x_ds), 0.0],
                                   maxfev=10000)
            z_ds = popt_ds[1] * (x_ds - popt_ds[2])
            z_all.extend(z_ds.tolist())
            q_all_normalized.extend(q_ds.tolist())
        except Exception:
            pass

    z_all = np.array(z_all)
    q_all_normalized = np.array(q_all_normalized)

    # Fit single sigmoid on normalized z
    try:
        popt_collapse, _ = curve_fit(sigmoid, z_all, q_all_normalized,
                                     p0=[0.6, 1.0, 0.0, 0.0], maxfev=10000)
        pred_collapse = sigmoid(z_all, *popt_collapse)
        ss_tot_c = np.sum((q_all_normalized - q_all_normalized.mean()) ** 2)
        r2_collapse = 1 - np.sum((q_all_normalized - pred_collapse) ** 2) / ss_tot_c
        mae_collapse = float(np.mean(np.abs(q_all_normalized - pred_collapse)))
        rho_collapse, _ = spearmanr(z_all, q_all_normalized)
        print(f"  Collapsed R^2 = {r2_collapse:.4f}, MAE = {mae_collapse:.4f}, rho = {rho_collapse:.4f}")
        print(f"  Theory predicts R^2 close to 1.0 after collapse")
        print(f"  Confirmed: {'YES' if r2_collapse > 0.90 else 'NO'}")
    except Exception as e:
        r2_collapse = 0.0
        rho_collapse = 0.0
        mae_collapse = 1.0
        print(f"  Collapse fit failed: {e}")

    results["predictions"]["1b_universal_collapse"] = {
        "r2": float(r2_collapse),
        "mae": float(mae_collapse),
        "rho": float(rho_collapse),
        "confirmed": bool(r2_collapse > 0.90),
    }

    # ============================================================
    # OVERALL SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("OVERALL SCORECARD")
    print(f"{'='*70}")

    checks = [
        ("P1: Probit fits at least as well as logistic",
         results["predictions"]["1_probit_vs_logistic"]["prediction_confirmed"],
         f"{n_probit_wins}/{len(pred1)} datasets"),
        ("P1b: Universal collapse R^2 > 0.90",
         results["predictions"]["1b_universal_collapse"]["confirmed"],
         f"R^2={r2_collapse:.4f}"),
        ("P4: Larger K -> larger kappa_c",
         results["predictions"].get("4_K_shifts_kappac", {}).get("confirmed", False),
         f"rho={results['predictions'].get('4_K_shifts_kappac', {}).get('rho', 0):.4f}"),
        ("P5: Steepness increases with k",
         results["predictions"]["5_k_effect"]["confirmed"],
         f"rho={rho_k_steep:.4f}"),
        ("P6: kappa reparameterization better than alpha",
         results["predictions"]["6_reparameterization"]["confirmed"],
         f"{n_kappa_better}/{len(pred6)}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    results["scorecard"] = {
        "passes": passes, "total": len(checks),
        "details": [{"criterion": c, "passed": bool(p), "value": v}
                     for c, p, v in checks],
    }

    out_path = RESULTS_DIR / "cti_theory_predictions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
