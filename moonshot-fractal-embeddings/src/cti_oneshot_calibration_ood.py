"""
Random-intercept calibration test for CTI OOD prediction.

Design:
- 3 new architectures (SmolLM2, phi-2, gemma-3) x 2 datasets x 4 layers = 24 pts
- slope alpha=1.477, beta=-0.309 fixed (from 12-arch LOAO)
- For each architecture, estimate C_d from 1,2,3,4 probe points (LOO-CV)
- Report MAE vs n_probe: shows how few calibration points are needed

Key hypothesis: 1 probe point per architecture collapses MAE from ~1.0 to <0.1
"""
import json
import os
import numpy as np
from scipy.stats import pearsonr
from scipy.special import logit as scipy_logit
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA_DS = 1.477
BETA_DS = -0.309

MODELS = ["SmolLM2-1.7B", "phi-2", "gemma-3-1b"]
DATASETS = ["banking77", "amazon_massive"]


def load_all_ood_points():
    pts = []
    for model_short in MODELS:
        for ds_name in DATASETS:
            cache_path = os.path.join(RESULTS_DIR,
                                      f"kappa_near_cache_{ds_name}_{model_short}.json")
            if not os.path.exists(cache_path):
                print(f"  Missing: {cache_path}")
                continue
            with open(cache_path) as f:
                cache_pts = json.load(f)
            pts.extend(cache_pts)
    return pts


def blind_prediction(pt, C_d_values):
    C_d = C_d_values.get(pt['dataset'], 0.0)
    return ALPHA_DS * pt['kappa_nearest'] + BETA_DS * pt['logKm1'] + C_d


def estimate_Cd_from_probes(probe_pts):
    """Estimate C_d per dataset from probe points."""
    C_d = {}
    for ds in DATASETS:
        ds_probes = [p for p in probe_pts if p['dataset'] == ds]
        if ds_probes:
            residuals = [p['logit_q'] - ALPHA_DS * p['kappa_nearest']
                         - BETA_DS * p['logKm1'] for p in ds_probes]
            C_d[ds] = float(np.mean(residuals))
    # Fill missing with zero
    for ds in DATASETS:
        if ds not in C_d:
            C_d[ds] = 0.0
    return C_d


def main():
    print("=" * 70)
    print("1-SHOT CALIBRATION TEST (Random-Intercept Model)")
    print(f"Fixed: alpha={ALPHA_DS:.4f}, beta={BETA_DS:.4f}")
    print("=" * 70)

    all_pts = load_all_ood_points()
    print(f"Total points: {len(all_pts)}")
    print(f"Models: {sorted(set(p['model'] for p in all_pts))}")

    # C_d from partial archs (baseline = no architecture-specific calibration)
    C_d_blind = {"banking77": 0.0167, "amazon_massive": 0.2608}

    print("\n--- BASELINE (no architecture-specific calibration) ---")
    obs_all = np.array([p['logit_q'] for p in all_pts])
    pred_blind = np.array([blind_prediction(p, C_d_blind) for p in all_pts])
    r_blind, p_blind = pearsonr(obs_all, pred_blind)
    mae_blind = float(np.mean(np.abs(obs_all - pred_blind)))
    bias_blind = float(np.mean(obs_all - pred_blind))
    print(f"  r={r_blind:.4f} (p={p_blind:.4f}), MAE={mae_blind:.4f}, bias={bias_blind:.4f}")
    print(f"  n={len(all_pts)} points across {len(MODELS)} architectures x 2 datasets x 4 layers")

    # Per-architecture calibration results
    print("\n--- PER-ARCHITECTURE N-SHOT CALIBRATION (LOO-CV) ---")
    print(f"\n{'n_probe':<10} {'r':<8} {'MAE':<8} {'bias':<8} description")

    results_by_n = {}

    for n_probe in range(1, 9):
        # For each architecture, try all combinations of n_probe points,
        # estimate C_d, predict held-out points
        all_obs, all_pred = [], []

        for model_short in MODELS:
            model_pts = [p for p in all_pts if p['model'] == model_short]
            if len(model_pts) < n_probe + 1:
                continue

            # Leave-one-out-cross-validation over probe sets
            # For each possible probe set of size n_probe, predict remaining
            combos = list(combinations(range(len(model_pts)), n_probe))
            # Sample if too many combos
            if len(combos) > 50:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(combos), size=50, replace=False)
                combos = [combos[i] for i in idx]

            for probe_idx in combos:
                probe_set = [model_pts[i] for i in probe_idx]
                test_set = [model_pts[i] for i in range(len(model_pts))
                            if i not in probe_idx]

                C_d = estimate_Cd_from_probes(probe_set)
                for pt in test_set:
                    pred = blind_prediction(pt, C_d)
                    all_obs.append(pt['logit_q'])
                    all_pred.append(pred)

        if len(all_obs) < 3:
            continue
        obs_arr = np.array(all_obs)
        pred_arr = np.array(all_pred)
        r_val, _ = pearsonr(obs_arr, pred_arr)
        mae_val = float(np.mean(np.abs(obs_arr - pred_arr)))
        bias_val = float(np.mean(obs_arr - pred_arr))
        n_pts = len(all_obs)
        results_by_n[n_probe] = {"r": float(r_val), "mae": float(mae_val),
                                  "bias": float(bias_val), "n": n_pts}
        print(f"  {n_probe:<10} {r_val:<8.4f} {mae_val:<8.4f} {bias_val:<8.4f} "
              f"(n_pred_pts={n_pts})")

    # Full calibration (all points per architecture)
    print("\n--- FULL CALIBRATION (all 8 points per arch, leave-1-out) ---")
    all_obs_full, all_pred_full = [], []
    for model_short in MODELS:
        model_pts = [p for p in all_pts if p['model'] == model_short]
        for i in range(len(model_pts)):
            probe_set = [model_pts[j] for j in range(len(model_pts)) if j != i]
            test_pt = model_pts[i]
            C_d = estimate_Cd_from_probes(probe_set)
            pred = blind_prediction(test_pt, C_d)
            all_obs_full.append(test_pt['logit_q'])
            all_pred_full.append(pred)

    obs_arr_f = np.array(all_obs_full)
    pred_arr_f = np.array(all_pred_full)
    r_full, _ = pearsonr(obs_arr_f, pred_arr_f)
    mae_full = float(np.mean(np.abs(obs_arr_f - pred_arr_f)))
    bias_full = float(np.mean(obs_arr_f - pred_arr_f))
    print(f"  r={r_full:.4f}, MAE={mae_full:.4f}, bias={bias_full:.4f}")

    # Summary table
    print("\n--- SUMMARY TABLE (MAE reduction from calibration) ---")
    print(f"  Baseline (0-shot):  MAE={mae_blind:.4f}")
    for n, res in sorted(results_by_n.items()):
        reduction_pct = 100 * (mae_blind - res['mae']) / mae_blind
        print(f"  {n}-shot calibration: MAE={res['mae']:.4f} "
              f"(reduction={reduction_pct:.1f}%)")
    print(f"  Full LOO:           MAE={mae_full:.4f}")

    output = {
        "experiment": "oneshot_calibration_ood",
        "design": ("LOO-CV calibration study. Fixed slope from 12-arch LOAO. "
                   "n_probe points per arch used to estimate C_d. "
                   "3 new architectures x 2 datasets x 4 layers = 24 pts."),
        "fixed_params": {"alpha": ALPHA_DS, "beta": BETA_DS},
        "baseline_C_d": C_d_blind,
        "models": MODELS,
        "baseline": {"r": float(r_blind), "mae": float(mae_blind),
                     "bias": float(bias_blind), "n": len(all_pts)},
        "n_shot_results": {str(k): v for k, v in results_by_n.items()},
        "full_loo": {"r": float(r_full), "mae": float(mae_full),
                     "bias": float(bias_full)},
    }
    out_path = os.path.join(RESULTS_DIR, "cti_oneshot_calibration_ood.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
