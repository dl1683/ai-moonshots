"""
NC-Loss Quantitative Prediction from CTI Law.

Uses the training-dynamics alpha from CE arm to PREDICT the NC arm delta_q.
This is a post-hoc quantitative prediction (pre-registered: delta_q >= 0.02).

Theory: logit(q) = alpha * kappa_nearest + C
  - alpha estimated from CE training trajectory (3 checkpoints)
  - NC arm increases kappa_nearest (ETF push)
  - Predicted delta_q = sigmoid(alpha * kappa_NC + C) - baseline_q

This script runs AFTER the NC arm completes and checks:
  1. Is the predicted delta_q consistent with observed delta_q?
  2. Does alpha from training match theory: alpha_theory = sqrt(d_eff) * sqrt(4/pi)?
"""

import json
import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import pearsonr


def logit(p):
    return np.log(p / (1 - p))


def compute_alpha_from_trajectory(seed_results):
    """Fit alpha from CE training trajectory for one seed."""
    kappas = [ckpt['kappa'] for ckpt in seed_results['checkpoints']
              if 0 < ckpt.get('q', 0) < 1 and ckpt.get('kappa', 0) > 0]
    qs = [ckpt['q'] for ckpt in seed_results['checkpoints']
          if 0 < ckpt.get('q', 0) < 1 and ckpt.get('kappa', 0) > 0]

    if len(kappas) < 2:
        return None, None, None

    kappas = np.array(kappas)
    logit_qs = np.array([logit(q) for q in qs])

    X = np.column_stack([kappas, np.ones(len(kappas))])
    coeffs, _, _, _ = np.linalg.lstsq(X, logit_qs, rcond=None)
    alpha, C = float(coeffs[0]), float(coeffs[1])

    pred = X @ coeffs
    ss_res = np.sum((logit_qs - pred)**2)
    ss_tot = np.sum((logit_qs - logit_qs.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

    return alpha, C, r2


def main():
    print("NC-Loss Quantitative Prediction from CTI Law")
    print("=" * 60)

    # Load quick pilot results
    try:
        with open('results/cti_nc_loss_quick.json') as f:
            quick = json.load(f)
    except Exception as e:
        print(f"Error loading quick results: {e}")
        return

    status = quick.get('status', 'unknown')
    print(f"Experiment status: {status}")
    print()

    ce_results = quick['results'].get('ce', [])
    nc_results = quick['results'].get('nc', [])

    # -------------------------------------------------------
    # Step 1: Fit alpha from CE training trajectory
    # -------------------------------------------------------
    print("=== STEP 1: Fit alpha from CE training trajectory ===")
    alphas = []
    Cs = []
    for ce_run in ce_results:
        if 'checkpoints' not in ce_run:
            continue
        alpha, C, r2 = compute_alpha_from_trajectory(ce_run)
        if alpha is not None:
            alphas.append(alpha)
            Cs.append(C)
            seed = ce_run.get('seed', '?')
            print(f"  CE seed={seed}: alpha={alpha:.4f}  C={C:.4f}  r2_fit={r2:.4f}")

    if not alphas:
        print("  No valid CE data for alpha estimation")
        return

    alpha_mean = float(np.mean(alphas))
    C_mean = float(np.mean(Cs))
    print(f"\n  Mean alpha = {alpha_mean:.4f}")
    print(f"  Mean C     = {C_mean:.4f}")
    print()

    # -------------------------------------------------------
    # Step 2: Theoretical alpha from Theorem 13
    # -------------------------------------------------------
    print("=== STEP 2: Theoretical alpha prediction ===")
    # d_eff implied by empirical alpha: d_eff = (alpha / sqrt(4/pi))^2
    d_eff_implied = (alpha_mean / np.sqrt(4/np.pi))**2
    alpha_theory_d1 = np.sqrt(1) * np.sqrt(4/np.pi)    # d_eff = 1 (perfect NC)
    alpha_theory_d2 = np.sqrt(2) * np.sqrt(4/np.pi)    # d_eff = 2 (one bit)
    print(f"  Empirical alpha = {alpha_mean:.4f}")
    print(f"  d_eff implied = {d_eff_implied:.4f}")
    print(f"  Theory: alpha(d_eff=1) = {alpha_theory_d1:.4f} (perfect NC)")
    print(f"  Theory: alpha(d_eff=2) = {alpha_theory_d2:.4f}")
    print(f"  d_eff in [{min(1, d_eff_implied):.2f}, {max(2, d_eff_implied):.2f}]")
    print()

    # -------------------------------------------------------
    # Step 3: CE baseline at final epoch
    # -------------------------------------------------------
    print("=== STEP 3: CE arm baseline ===")
    ce_finals = []
    for ce_run in ce_results:
        if 'final_q' in ce_run:
            ce_finals.append({'q': ce_run['final_q'], 'kappa': ce_run['final_kappa'],
                               'seed': ce_run.get('seed', '?')})
            print(f"  CE seed={ce_run.get('seed','?')}: q={ce_run['final_q']:.4f}  kappa={ce_run['final_kappa']:.4f}")

    if not ce_finals:
        print("  No final CE results yet")
        return

    ce_q_mean = float(np.mean([r['q'] for r in ce_finals]))
    ce_kappa_mean = float(np.mean([r['kappa'] for r in ce_finals]))
    ce_logit_mean = float(np.mean([logit(r['q']) for r in ce_finals]))
    print(f"\n  CE mean q = {ce_q_mean:.4f} (logit = {ce_logit_mean:.4f})")
    print(f"  CE mean kappa = {ce_kappa_mean:.4f}")
    print()

    # -------------------------------------------------------
    # Step 4: Predict NC arm delta_q as function of delta_kappa
    # -------------------------------------------------------
    print("=== STEP 4: Predicted delta_q vs delta_kappa ===")
    print(f"  Using alpha={alpha_mean:.4f}, baseline logit(q)={ce_logit_mean:.4f}, baseline q={ce_q_mean:.4f}")
    print()
    print(f"  delta_kappa | kappa_NC | logit_NC | q_NC  | delta_q | PASS(>=0.02)?")
    print(f"  --------------|---------|---------|-------|---------|----------")
    for dk in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        kappa_nc = ce_kappa_mean + dk
        logit_nc = ce_logit_mean + alpha_mean * dk
        q_nc = float(sigmoid(logit_nc))
        delta_q = q_nc - ce_q_mean
        passed = delta_q >= 0.02
        print(f"  delta_k={dk:.2f}     | k={kappa_nc:.3f} | l={logit_nc:.3f} | {q_nc:.4f} | {delta_q:+.4f} | {'PASS' if passed else 'FAIL'}")

    # Required delta_kappa for passing
    req_dk = 0.02 / alpha_mean  # minimum delta_kappa to get delta_q=0.02
    print(f"\n  Min delta_kappa for P3 (delta_q>=0.02): {req_dk:.4f}")
    print(f"  (Only {req_dk:.3f} kappa increase needed from NC-loss ETF push)")
    print()

    # -------------------------------------------------------
    # Step 5: Compare with NC arm results (if available)
    # -------------------------------------------------------
    if nc_results:
        print("=== STEP 5: NC arm comparison ===")
        nc_finals = []
        for nc_run in nc_results:
            if 'final_q' in nc_run:
                nc_finals.append({'q': nc_run['final_q'], 'kappa': nc_run['final_kappa'],
                                   'seed': nc_run.get('seed', '?')})
                print(f"  NC seed={nc_run.get('seed','?')}: q={nc_run['final_q']:.4f}  kappa={nc_run['final_kappa']:.4f}")

        if nc_finals:
            nc_q_mean = float(np.mean([r['q'] for r in nc_finals]))
            nc_kappa_mean = float(np.mean([r['kappa'] for r in nc_finals]))
            delta_q_obs = nc_q_mean - ce_q_mean
            delta_kappa_obs = nc_kappa_mean - ce_kappa_mean

            print(f"\n  NC mean q = {nc_q_mean:.4f}")
            print(f"  NC mean kappa = {nc_kappa_mean:.4f}")
            print(f"  delta_q (observed) = {delta_q_obs:+.4f}")
            print(f"  delta_kappa (observed) = {delta_kappa_obs:+.4f}")

            # Compare with theory
            delta_q_predicted = float(sigmoid(ce_logit_mean + alpha_mean * delta_kappa_obs)) - ce_q_mean
            print(f"\n  delta_q PREDICTED by law = {delta_q_predicted:+.4f}")
            print(f"  delta_q OBSERVED          = {delta_q_obs:+.4f}")
            if delta_q_obs != 0:
                ratio = delta_q_predicted / delta_q_obs
                print(f"  Prediction/observation ratio = {ratio:.3f}")

            print(f"\n  P1 (sign test: delta_q > 0): {'PASS' if delta_q_obs > 0 else 'FAIL'}")
            print(f"  P2 (delta_kappa > 0):         {'PASS' if delta_kappa_obs > 0 else 'FAIL'}")
            print(f"  P3 (delta_q >= 0.02):         {'PASS' if delta_q_obs >= 0.02 else 'FAIL'}")

            # Save results
            results = {
                'alpha_from_trajectory': alpha_mean,
                'alpha_cv': float(np.std(alphas)/np.mean(alphas)) if len(alphas) > 1 else 0,
                'd_eff_implied': d_eff_implied,
                'ce_q_mean': ce_q_mean,
                'ce_kappa_mean': ce_kappa_mean,
                'nc_q_mean': nc_q_mean,
                'nc_kappa_mean': nc_kappa_mean,
                'delta_q_observed': delta_q_obs,
                'delta_kappa_observed': delta_kappa_obs,
                'delta_q_predicted': delta_q_predicted,
                'prediction_error': abs(delta_q_predicted - delta_q_obs),
                'P1_pass': bool(delta_q_obs > 0),
                'P2_pass': bool(delta_kappa_obs > 0),
                'P3_pass': bool(delta_q_obs >= 0.02),
            }
            with open('results/cti_nc_loss_prediction.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved to results/cti_nc_loss_prediction.json")
    else:
        print("=== STEP 5: NC arm not yet complete ===")
        print("  Run this script again after NC arm finishes to see comparison")

    # Save partial results (even without NC arm)
    partial = {
        'alpha_from_trajectory': alpha_mean,
        'alpha_cv': float(np.std(alphas)/np.mean(alphas)) if len(alphas) > 1 else 0,
        'd_eff_implied': d_eff_implied,
        'ce_q_mean': ce_q_mean,
        'ce_kappa_mean': ce_kappa_mean,
        'min_delta_kappa_for_P3': req_dk,
        'prediction_table': [
            {'delta_kappa': dk,
             'delta_q_predicted': float(sigmoid(ce_logit_mean + alpha_mean * dk)) - ce_q_mean}
            for dk in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        ],
        'nc_complete': len(nc_results) > 0,
    }
    with open('results/cti_nc_loss_prediction.json', 'w') as f:
        json.dump(partial, f, indent=2)
    print(f"\nSaved partial results to results/cti_nc_loss_prediction.json")


if __name__ == '__main__':
    main()
