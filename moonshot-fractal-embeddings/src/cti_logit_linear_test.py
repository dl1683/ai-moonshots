#!/usr/bin/env python
"""
LOGIT-LINEAR TEST: Validate the corrected theoretical form

From Codex theory (Gumbel EVT derivation):
  logit(q) = A*kappa - B*log(K) + C

This is the fundamental prediction. If it holds, then:
  q = sigmoid(A*kappa - B*log(K) + C)
  = sigmoid(A*(kappa - (B/A)*log(K)) + C)

Setting kappa_c = (B/A)*log(K), the transition point scales with log(K).

Compare against:
1. logit(q) = A*kappa - B*sqrt(K) + C  (old theory)
2. logit(q) = A*kappa/log(K) + C  (simplified collapse form)
3. logit(q) = A*kappa - B*log(K) + C  (full EVT form)

Also test on REAL data (not just synthetic) from existing CTI results.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.special import expit, logit as sp_logit
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def safe_logit(q, eps=0.001):
    """Logit with clipping to avoid infinities."""
    q_clip = np.clip(q, eps, 1.0 - eps)
    return sp_logit(q_clip)


def simulate_knn(d, K, m, kappa_target, n_test=2000):
    """Simulate 1-NN for K-class isotropic Gaussians."""
    sigma2 = 1.0
    delta2 = kappa_target * K * d / max(K - 1, 1)

    if K - 1 <= d:
        V = np.eye(K, min(K-1, d))
        V = V - V.mean(0)
        norms = np.sqrt((V ** 2).sum(1, keepdims=True))
        norms[norms < 1e-10] = 1.0
        V = V / norms * np.sqrt((K - 1) / K)
        means = np.zeros((K, d))
        means[:, :min(K-1, d)] = V * np.sqrt(delta2)
    else:
        means = np.random.randn(K, d)
        means = means - means.mean(0)
        norms = np.sqrt((means ** 2).sum(1, keepdims=True))
        norms[norms < 1e-10] = 1.0
        means = means / norms * np.sqrt(delta2) * np.sqrt((K - 1) / K)

    grand_mean = means.mean(0)
    tr_sb = sum(np.sum((means[k] - grand_mean)**2) for k in range(K)) / K
    tr_sw = d * sigma2
    actual_kappa = tr_sb / tr_sw

    train_labels = np.repeat(np.arange(K), m)
    train_X = np.zeros((K * m, d))
    for k in range(K):
        train_X[k*m:(k+1)*m] = means[k] + np.random.randn(m, d)

    test_labels = np.random.randint(0, K, n_test)
    test_X = np.zeros((n_test, d))
    for i in range(n_test):
        test_X[i] = means[test_labels[i]] + np.random.randn(d)

    correct = 0
    bs = 500
    for s in range(0, n_test, bs):
        e = min(s + bs, n_test)
        diff = test_X[s:e, None, :] - train_X[None, :, :]
        dists = (diff ** 2).sum(2)
        nn = dists.argmin(1)
        correct += (train_labels[nn] == test_labels[s:e]).sum()

    acc = correct / n_test
    q = (acc - 1.0/K) / (1.0 - 1.0/K)
    return acc, q, actual_kappa


def main():
    print("=" * 70)
    print("LOGIT-LINEAR TEST: logit(q) = A*kappa - B*log(K) + C")
    print("Fundamental prediction from Gumbel EVT derivation")
    print("=" * 70)

    # Phase 1: Synthetic validation
    print("\n--- PHASE 1: SYNTHETIC (Gaussian clusters) ---\n")

    d = 300
    m = 40
    K_values = [2, 5, 10, 20, 50, 100, 200]
    kappa_range = np.linspace(0.02, 0.5, 12)

    all_data = []
    for K in K_values:
        print(f"  K={K}...", end="", flush=True)
        for kappa_t in kappa_range:
            if kappa_t * K > d * 0.8:
                continue
            acc, q, actual_kappa = simulate_knn(d, K, m, kappa_t, n_test=2000)
            all_data.append({"K": K, "kappa": actual_kappa, "q": q, "acc": acc})
            sys.stdout.write(".")
            sys.stdout.flush()
        print(" done")

    kappas = np.array([p["kappa"] for p in all_data])
    qs = np.array([p["q"] for p in all_data])
    Ks = np.array([p["K"] for p in all_data])
    logKs = np.log(Ks)
    sqrtKs = np.sqrt(Ks)

    # Filter out q ~ 0 or q ~ 1 for logit
    mask = (qs > 0.005) & (qs < 0.995)
    kappas_f = kappas[mask]
    qs_f = qs[mask]
    Ks_f = Ks[mask]
    logKs_f = np.log(Ks_f)
    sqrtKs_f = np.sqrt(Ks_f)
    logit_q = safe_logit(qs_f)

    print(f"\n  {len(all_data)} total, {mask.sum()} in logit range")

    # MODEL 1: logit(q) = A*kappa - B*log(K) + C  (EVT prediction)
    X1 = np.column_stack([kappas_f, logKs_f, np.ones(len(kappas_f))])
    beta1, _, _, _ = np.linalg.lstsq(X1, logit_q, rcond=None)
    pred1 = X1 @ beta1
    r2_1 = 1 - ((logit_q - pred1)**2).sum() / ((logit_q - logit_q.mean())**2).sum()

    # MODEL 2: logit(q) = A*kappa - B*sqrt(K) + C  (old theory)
    X2 = np.column_stack([kappas_f, sqrtKs_f, np.ones(len(kappas_f))])
    beta2, _, _, _ = np.linalg.lstsq(X2, logit_q, rcond=None)
    pred2 = X2 @ beta2
    r2_2 = 1 - ((logit_q - pred2)**2).sum() / ((logit_q - logit_q.mean())**2).sum()

    # MODEL 3: logit(q) = A*kappa/log(K) + C  (simplified collapse)
    X3 = np.column_stack([kappas_f / logKs_f, np.ones(len(kappas_f))])
    beta3, _, _, _ = np.linalg.lstsq(X3, logit_q, rcond=None)
    pred3 = X3 @ beta3
    r2_3 = 1 - ((logit_q - pred3)**2).sum() / ((logit_q - logit_q.mean())**2).sum()

    # MODEL 4: logit(q) = A*kappa/sqrt(K) + C  (old collapse)
    X4 = np.column_stack([kappas_f / sqrtKs_f, np.ones(len(kappas_f))])
    beta4, _, _, _ = np.linalg.lstsq(X4, logit_q, rcond=None)
    pred4 = X4 @ beta4
    r2_4 = 1 - ((logit_q - pred4)**2).sum() / ((logit_q - logit_q.mean())**2).sum()

    # MODEL 5: logit(q) = A*kappa + C  (no K normalization)
    X5 = np.column_stack([kappas_f, np.ones(len(kappas_f))])
    beta5, _, _, _ = np.linalg.lstsq(X5, logit_q, rcond=None)
    pred5 = X5 @ beta5
    r2_5 = 1 - ((logit_q - pred5)**2).sum() / ((logit_q - logit_q.mean())**2).sum()

    print(f"\n  --- LOGIT-SPACE LINEAR MODEL COMPARISON ---")
    print(f"  Model 1: logit(q) = A*kappa - B*log(K) + C    R^2 = {r2_1:.4f}  [EVT]")
    print(f"           A={beta1[0]:.4f}, B={-beta1[1]:.4f}, C={beta1[2]:.4f}")
    print(f"  Model 2: logit(q) = A*kappa - B*sqrt(K) + C   R^2 = {r2_2:.4f}  [old CLT]")
    print(f"           A={beta2[0]:.4f}, B={-beta2[1]:.4f}, C={beta2[2]:.4f}")
    print(f"  Model 3: logit(q) = A*kappa/log(K) + C        R^2 = {r2_3:.4f}  [collapse]")
    print(f"           A={beta3[0]:.4f}, C={beta3[1]:.4f}")
    print(f"  Model 4: logit(q) = A*kappa/sqrt(K) + C       R^2 = {r2_4:.4f}  [old collapse]")
    print(f"           A={beta4[0]:.4f}, C={beta4[1]:.4f}")
    print(f"  Model 5: logit(q) = A*kappa + C               R^2 = {r2_5:.4f}  [no K]")
    print(f"           A={beta5[0]:.4f}, C={beta5[1]:.4f}")

    # Check B coefficient: theory predicts B ~ 1 (Gumbel location shift)
    print(f"\n  EVT prediction: B should be ~ 1.0")
    print(f"  Observed B = {-beta1[1]:.4f}")

    # Phase 2: Real CTI data
    print(f"\n{'='*70}")
    print("PHASE 2: REAL CTI DATA")
    print(f"{'='*70}")

    # Load all available CTI data
    real_points = []

    # CLINC from geometry mediator
    try:
        with open(RESULTS_DIR / "cti_geometry_mediator.json") as f:
            clinc_raw = json.load(f)
        for p in clinc_raw["all_points"]:
            real_points.append({
                "dataset": "clinc", "K": 150,
                "model": p["model"], "alpha": p["alpha"],
                "knn": p["knn"], "kappa": p["kappa"],
            })
    except FileNotFoundError:
        print("  (skipping clinc - file not found)")

    # AGNews and DBPedia
    for ds in ["agnews", "dbpedia_classes"]:
        try:
            with open(RESULTS_DIR / f"cti_multidata_{ds}_cache.json") as f:
                data = json.load(f)
            for p in data:
                real_points.append({
                    "dataset": p["dataset"], "K": p["n_classes"],
                    "model": p["model"], "alpha": p["alpha"],
                    "knn": p["knn"], "kappa": p["kappa"],
                })
        except FileNotFoundError:
            print(f"  (skipping {ds} - file not found)")

    # Yahoo and arXiv from blind prediction
    try:
        with open(RESULTS_DIR / "cti_blind_prediction.json") as f:
            blind = json.load(f)
        for p in blind["blind_points"]:
            real_points.append({
                "dataset": p["dataset"], "K": p["K"],
                "model": p["model"], "alpha": p["alpha"],
                "knn": p["knn"], "kappa": p["kappa"],
            })
    except FileNotFoundError:
        print("  (skipping blind - file not found)")

    if real_points:
        print(f"  Total real points: {len(real_points)}")
        print(f"  Datasets: {sorted(set(p['dataset'] for p in real_points))}")
        print(f"  K values: {sorted(set(p['K'] for p in real_points))}")

        r_kappas = np.array([p["kappa"] for p in real_points])
        r_knns = np.array([p["knn"] for p in real_points])
        r_Ks = np.array([p["K"] for p in real_points])
        r_qs = (r_knns - 1.0/r_Ks) / (1.0 - 1.0/r_Ks)
        r_logKs = np.log(r_Ks)
        r_sqrtKs = np.sqrt(r_Ks)

        # Filter for logit
        mask_r = (r_qs > 0.005) & (r_qs < 0.995)
        rk = r_kappas[mask_r]
        rq = r_qs[mask_r]
        rK = r_Ks[mask_r]
        r_lK = np.log(rK)
        r_sK = np.sqrt(rK)
        r_logit = safe_logit(rq)

        print(f"  In logit range: {mask_r.sum()}")

        # Same models on real data
        X1r = np.column_stack([rk, r_lK, np.ones(len(rk))])
        b1r, _, _, _ = np.linalg.lstsq(X1r, r_logit, rcond=None)
        p1r = X1r @ b1r
        r2_1r = 1 - ((r_logit - p1r)**2).sum() / ((r_logit - r_logit.mean())**2).sum()

        X2r = np.column_stack([rk, r_sK, np.ones(len(rk))])
        b2r, _, _, _ = np.linalg.lstsq(X2r, r_logit, rcond=None)
        p2r = X2r @ b2r
        r2_2r = 1 - ((r_logit - p2r)**2).sum() / ((r_logit - r_logit.mean())**2).sum()

        X3r = np.column_stack([rk / r_lK, np.ones(len(rk))])
        b3r, _, _, _ = np.linalg.lstsq(X3r, r_logit, rcond=None)
        p3r = X3r @ b3r
        r2_3r = 1 - ((r_logit - p3r)**2).sum() / ((r_logit - r_logit.mean())**2).sum()

        X4r = np.column_stack([rk / r_sK, np.ones(len(rk))])
        b4r, _, _, _ = np.linalg.lstsq(X4r, r_logit, rcond=None)
        p4r = X4r @ b4r
        r2_4r = 1 - ((r_logit - p4r)**2).sum() / ((r_logit - r_logit.mean())**2).sum()

        print(f"\n  --- REAL DATA: LOGIT-SPACE MODEL COMPARISON ---")
        print(f"  Model 1: logit(q) = A*kappa - B*log(K) + C    R^2 = {r2_1r:.4f}  [EVT]")
        print(f"           A={b1r[0]:.4f}, B={-b1r[1]:.4f}, C={b1r[2]:.4f}")
        print(f"  Model 2: logit(q) = A*kappa - B*sqrt(K) + C   R^2 = {r2_2r:.4f}  [old CLT]")
        print(f"  Model 3: logit(q) = A*kappa/log(K) + C        R^2 = {r2_3r:.4f}  [collapse]")
        print(f"  Model 4: logit(q) = A*kappa/sqrt(K) + C       R^2 = {r2_4r:.4f}  [old collapse]")

        # Per-dataset logit fits
        print(f"\n  --- PER-DATASET FITS ---")
        for ds in sorted(set(p["dataset"] for p in real_points)):
            ds_mask = np.array([p["dataset"] == ds for p in real_points])
            combined = ds_mask & mask_r
            if combined.sum() < 5:
                continue
            dk = r_kappas[combined]
            dq = safe_logit(r_qs[combined])
            Xd = np.column_stack([dk, np.ones(len(dk))])
            bd, _, _, _ = np.linalg.lstsq(Xd, dq, rcond=None)
            pd = Xd @ bd
            r2_d = 1 - ((dq - pd)**2).sum() / ((dq - dq.mean())**2).sum()
            K_ds = r_Ks[combined][0]
            print(f"    {ds:>15} (K={K_ds:>3}): logit(q) = {bd[0]:.2f}*kappa + {bd[1]:.2f}, R^2={r2_d:.4f}")

    # Scorecard
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    checks = [
        ("EVT logit-linear (Model 1) best on synthetic",
         r2_1 == max(r2_1, r2_2, r2_3, r2_4, r2_5),
         f"R^2={r2_1:.4f} vs sqrt:{r2_2:.4f}, collapse:{r2_3:.4f}"),
        ("log(K) beats sqrt(K) on synthetic",
         r2_1 > r2_2,
         f"log:{r2_1:.4f} vs sqrt:{r2_2:.4f}"),
        ("EVT B coefficient ~ 1.0 (within [0.5, 2.0])",
         0.5 <= -beta1[1] <= 2.0,
         f"B={-beta1[1]:.4f}"),
        ("EVT logit-linear best on real data",
         r2_1r == max(r2_1r, r2_2r, r2_3r, r2_4r) if real_points else False,
         f"R^2={r2_1r:.4f}" if real_points else "no data"),
        ("log(K) beats sqrt(K) on real data",
         r2_1r > r2_2r if real_points else False,
         f"log:{r2_1r:.4f} vs sqrt:{r2_2r:.4f}" if real_points else "no data"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}: {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "logit_linear_test",
        "theory": "logit(q) = A*kappa - B*log(K) + C from Gumbel EVT",
        "synthetic": {
            "model1_evt_logK": {"r2": float(r2_1), "A": float(beta1[0]),
                                "B": float(-beta1[1]), "C": float(beta1[2])},
            "model2_clt_sqrtK": {"r2": float(r2_2)},
            "model3_collapse_logK": {"r2": float(r2_3)},
            "model4_collapse_sqrtK": {"r2": float(r2_4)},
            "model5_no_K": {"r2": float(r2_5)},
        },
        "real_data": {
            "model1_evt_logK": {"r2": float(r2_1r), "A": float(b1r[0]),
                                "B": float(-b1r[1]), "C": float(b1r[2])},
            "model2_clt_sqrtK": {"r2": float(r2_2r)},
            "model3_collapse_logK": {"r2": float(r2_3r)},
            "model4_collapse_sqrtK": {"r2": float(r2_4r)},
        } if real_points else {},
        "scorecard": {"passes": passes, "total": len(checks)},
    }

    out_path = RESULTS_DIR / "cti_logit_linear.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
