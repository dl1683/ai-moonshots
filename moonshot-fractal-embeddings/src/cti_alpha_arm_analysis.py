"""
Correct Causal Test: Per-Arm Alpha Slope Analysis

Based on Theorem 16 (Feb 22 2026), the correct causal test is NOT:
  Delta logit(q) = A_renorm * Delta(kappa_eff_gram)    [WRONG: d_eff_gram != d_eff_cls]

But rather:
  alpha_arm = d(logit_q)/d(kappa_nearest) = A_renorm * sqrt(d_eff_cls_arm)

The causal hypothesis: NC-loss increases d_eff_cls (classification-relevant dimensionality).
If d_eff_cls(NC+) > d_eff_cls(CE) > d_eff_cls(anti_NC), then:
  alpha(NC+) > alpha(CE) > alpha(anti_NC)

This should hold because:
- NC-loss (L_within + L_ETF + L_margin) pushes toward Neural Collapse, which concentrates
  within-class variance in the nearest-class direction -> increases d_eff_cls toward NC-limit
- Anti-NC loss (-L_within) maximizes within-class spread -> dilutes signal -> decreases d_eff_cls

TEST (pre-registered after NC quick pilot, Feb 22):
  alpha(NC+) > alpha(CE): YES if d_eff_cls(NC+) > d_eff_cls(CE)
  alpha(anti_NC) < alpha(CE): YES if d_eff_cls(anti_NC) < d_eff_cls(CE)

From NC quick pilot: alpha_CE = 1.21, alpha_NC = 1.39 (16% increase). This is the signal.

ANALYSIS PERFORMED:
1. Fit logit_q = alpha * kappa_nearest + C for each arm (across all seeds and checkpoints)
2. Test: alpha(NC+) > alpha(CE) and alpha(anti_NC) < alpha(CE)?
3. Compute d_eff_cls = (alpha/A_renorm(K))^2 for each arm
4. Compare to NC quick pilot results
5. Additionally: test the CROSS-ARM prediction at same epoch

USES: cti_control_law_validation.json or cti_nc_loss_training.json (full RCT)
"""

import json
import numpy as np
import os
import sys

A_RENORM_K20 = 1.0535  # Theorem 15 pre-registered constant
K = 20

def fit_alpha_from_arm(arm_results):
    """
    Fit logit_q = alpha * kappa_nearest + C from all checkpoints in an arm.
    Returns: alpha, C, r2, n_points, per_seed results
    """
    all_kappa = []
    all_logit_q = []
    per_seed = []

    for res in arm_results:
        seed_kappa = []
        seed_logit = []
        for ck in res.get('checkpoints', []):
            k = ck.get('kappa') or ck.get('kappa_nearest')
            q = ck.get('q')
            if k is None or q is None or q <= 0 or q >= 1:
                continue
            lq = float(np.log(q / (1 - q)))
            seed_kappa.append(float(k))
            seed_logit.append(lq)
            all_kappa.append(float(k))
            all_logit_q.append(lq)

        if len(seed_kappa) >= 2:
            coeffs = np.polyfit(seed_kappa, seed_logit, 1)
            seed_alpha = float(coeffs[0])
            seed_C = float(coeffs[1])
            seed_r2 = float(np.corrcoef(seed_kappa, seed_logit)[0, 1] ** 2) if len(set(seed_kappa)) > 1 else None
            per_seed.append({
                'seed': res.get('seed'),
                'alpha': seed_alpha,
                'C': seed_C,
                'r2': seed_r2,
                'n_points': len(seed_kappa),
            })

    if len(all_kappa) < 3:
        return None

    coeffs = np.polyfit(all_kappa, all_logit_q, 1)
    alpha = float(coeffs[0])
    C = float(coeffs[1])
    kappa_arr = np.array(all_kappa)
    logit_arr = np.array(all_logit_q)
    r2 = float(np.corrcoef(kappa_arr, logit_arr)[0, 1] ** 2) if kappa_arr.std() > 1e-6 else 0.0

    return {
        'alpha': alpha,
        'C': C,
        'r2': r2,
        'd_eff_cls': float((alpha / A_RENORM_K20) ** 2),
        'n_points': len(all_kappa),
        'per_seed': per_seed,
    }


def cross_arm_prediction(all_results, A_renorm=A_RENORM_K20):
    """
    At each epoch and seed: compare NC+ and anti_nc to CE.
    Test: delta_logit_q predicted by alpha_arm * delta_kappa_nearest?
    Also test: delta_logit_q predicted by A_renorm * sqrt(d_eff_cls_arm) * delta_kappa_nearest?
    """
    if 'ce' not in all_results:
        return None

    # First compute per-arm alpha
    alpha_fits = {}
    for arm, arm_results in all_results.items():
        fit = fit_alpha_from_arm(arm_results)
        if fit:
            alpha_fits[arm] = fit

    cross_pairs = []
    for ck_idx in range(20):  # max checkpoints
        for seed_idx in range(len(all_results.get('ce', []))):
            ce_res = all_results['ce'][seed_idx] if seed_idx < len(all_results['ce']) else None
            if ce_res is None:
                continue
            ce_ckpts = ce_res.get('checkpoints', [])
            if ck_idx >= len(ce_ckpts):
                break
            ce_ck = ce_ckpts[ck_idx]

            for arm in [k for k in all_results if k != 'ce']:
                if seed_idx >= len(all_results[arm]):
                    continue
                arm_res = all_results[arm][seed_idx]
                arm_ckpts = arm_res.get('checkpoints', [])
                if ck_idx >= len(arm_ckpts):
                    continue
                arm_ck = arm_ckpts[ck_idx]

                k_ce = ce_ck.get('kappa') or ce_ck.get('kappa_nearest')
                k_arm = arm_ck.get('kappa') or arm_ck.get('kappa_nearest')
                q_ce = ce_ck.get('q')
                q_arm = arm_ck.get('q')
                if any(x is None for x in [k_ce, k_arm, q_ce, q_arm]):
                    continue
                if not (0 < q_ce < 1 and 0 < q_arm < 1):
                    continue

                lq_ce = float(np.log(q_ce / (1 - q_ce)))
                lq_arm = float(np.log(q_arm / (1 - q_arm)))
                delta_logit = lq_arm - lq_ce
                delta_kappa = k_arm - k_ce

                # Prediction using CE alpha
                alpha_ce = alpha_fits.get('ce', {}).get('alpha', A_renorm)
                pred_ce_alpha = alpha_ce * delta_kappa

                # Prediction using arm's own alpha
                alpha_arm = alpha_fits.get(arm, {}).get('alpha', A_renorm)
                # At same epoch, if kappa differs, the logit difference is:
                # logit_arm - logit_ce = alpha_arm * k_arm - alpha_ce * k_ce + (C_arm - C_ce)
                # If C_arm ≈ C_ce (same starting point):
                # delta_logit ≈ alpha_arm * k_arm - alpha_ce * k_ce
                C_ce = alpha_fits.get('ce', {}).get('C', 0)
                C_arm = alpha_fits.get(arm, {}).get('C', 0)
                pred_alpha_model = alpha_arm * k_arm - alpha_ce * k_ce + (C_arm - C_ce)

                cross_pairs.append({
                    'arm': arm,
                    'seed': ce_res.get('seed'),
                    'epoch': ce_ck.get('epoch'),
                    'delta_logit_q': float(delta_logit),
                    'delta_kappa': float(delta_kappa),
                    'q_ce': float(q_ce),
                    'q_arm': float(q_arm),
                    'k_ce': float(k_ce),
                    'k_arm': float(k_arm),
                    'pred_ce_alpha': float(pred_ce_alpha),
                    'pred_alpha_model': float(pred_alpha_model) if pred_alpha_model is not None else None,
                    'resid_ce_alpha': float(delta_logit - pred_ce_alpha),
                    'resid_alpha_model': float(delta_logit - pred_alpha_model) if pred_alpha_model is not None else None,
                })

    return cross_pairs, alpha_fits


def main():
    # Try both result files
    result_files = [
        'results/cti_control_law_validation.json',
        'results/cti_nc_loss_training.json',
        'results/cti_nc_loss_quick.json',
    ]

    for path in result_files:
        if os.path.exists(path):
            print(f"Loading: {path}")
            with open(path) as f:
                data = json.load(f)
            all_results = data.get('results', {})
            if not all_results:
                continue
            print(f"Status: {data.get('status', 'unknown')}")
            print(f"Arms: {list(all_results.keys())}")
            print()

            print("=" * 70)
            print(f"PER-ARM ALPHA ANALYSIS (CORRECT CAUSAL TEST)")
            print(f"A_renorm(K=20) = {A_RENORM_K20} (Theorem 15)")
            print("=" * 70)

            alpha_results = {}
            for arm in sorted(all_results.keys()):
                fit = fit_alpha_from_arm(all_results[arm])
                alpha_results[arm] = fit
                if fit:
                    print(f"  {arm:10s}: alpha={fit['alpha']:.4f} C={fit['C']:.4f} "
                          f"R2={fit['r2']:.4f} d_eff_cls={fit['d_eff_cls']:.4f} "
                          f"n={fit['n_points']}")
                    for s in fit['per_seed']:
                        print(f"    seed={s['seed']}: alpha={s['alpha']:.4f} R2={s.get('r2', 0):.4f}")
                else:
                    print(f"  {arm:10s}: insufficient data")

            print()
            # Key tests
            alpha_ce = alpha_results.get('ce', {})
            alpha_nc = alpha_results.get('nc', {})
            alpha_anti = alpha_results.get('anti_nc', {})

            if alpha_ce and alpha_nc:
                deff_ce = alpha_ce.get('d_eff_cls', 0)
                deff_nc = alpha_nc.get('d_eff_cls', 0)
                print(f"TEST: d_eff_cls(NC+) > d_eff_cls(CE)?")
                print(f"  d_eff_cls(CE)  = {deff_ce:.4f}")
                print(f"  d_eff_cls(NC+) = {deff_nc:.4f}")
                print(f"  PASS: {deff_nc > deff_ce}")
                print()

            if alpha_ce and alpha_anti:
                deff_ce = alpha_ce.get('d_eff_cls', 0)
                deff_anti = alpha_anti.get('d_eff_cls', 0)
                print(f"TEST: d_eff_cls(anti_NC) < d_eff_cls(CE)?")
                print(f"  d_eff_cls(CE)      = {deff_ce:.4f}")
                print(f"  d_eff_cls(anti_NC) = {deff_anti:.4f}")
                print(f"  PASS: {deff_anti < deff_ce}")
                print()

            # Prediction test
            print("=" * 70)
            print("NC PILOT COMPARISON")
            print("=" * 70)
            print(f"NC quick pilot: alpha_CE={1.21:.4f} d_eff_cls={1.37:.4f}")
            print(f"NC quick pilot: alpha_NC={1.39:.4f} d_eff_cls={1.76:.4f}")
            if alpha_results.get('ce'):
                a_ce = alpha_results['ce']['alpha']
                de_ce = alpha_results['ce']['d_eff_cls']
                print(f"This run:       alpha_CE={a_ce:.4f} d_eff_cls={de_ce:.4f}")
            if alpha_results.get('nc'):
                a_nc = alpha_results['nc']['alpha']
                de_nc = alpha_results['nc']['d_eff_cls']
                print(f"This run:       alpha_NC={a_nc:.4f} d_eff_cls={de_nc:.4f}")

            # Cross-arm
            print()
            print("=" * 70)
            print("CROSS-ARM PREDICTION (same epoch, different objective)")
            print("=" * 70)
            cross_result = cross_arm_prediction(all_results)
            if cross_result:
                cross_pairs, fits = cross_result
                for arm in ['nc', 'anti_nc']:
                    arm_pairs = [p for p in cross_pairs if p['arm'] == arm]
                    if not arm_pairs:
                        continue
                    delta_qs = [p['delta_logit_q'] for p in arm_pairs]
                    delta_ks = [p['delta_kappa'] for p in arm_pairs]
                    # Sign consistency
                    if arm == 'nc':
                        sign_q = sum(1 for p in arm_pairs if p['delta_logit_q'] > 0)
                        expected_sign = 'positive delta_q'
                    else:
                        sign_q = sum(1 for p in arm_pairs if p['delta_logit_q'] < 0)
                        expected_sign = 'negative delta_q'
                    print(f"  {arm}: n={len(arm_pairs)} mean_delta_logit={np.mean(delta_qs):+.4f} "
                          f"sign_consistent={sign_q}/{len(arm_pairs)} (expected: {expected_sign})")
                    print(f"         mean_delta_kappa={np.mean(delta_ks):+.4f}")

            # Save output
            output = {
                'source': path,
                'per_arm_alpha': {arm: {k: v for k, v in (fit or {}).items() if k != 'per_seed'}
                                  for arm, fit in alpha_results.items()},
                'cross_arm_pairs': cross_pairs if cross_result else [],
                'tests': {
                    'alpha_nc_gt_ce': (alpha_results.get('nc', {}) or {}).get('alpha', 0) >
                                      (alpha_results.get('ce', {}) or {}).get('alpha', 0),
                    'alpha_anti_lt_ce': (alpha_results.get('anti_nc', {}) or {}).get('alpha', float('inf')) <
                                        (alpha_results.get('ce', {}) or {}).get('alpha', float('inf')),
                    'd_eff_cls_ce': (alpha_results.get('ce', {}) or {}).get('d_eff_cls'),
                    'd_eff_cls_nc': (alpha_results.get('nc', {}) or {}).get('d_eff_cls'),
                }
            }
            out_path = f"results/cti_alpha_arm_analysis_{os.path.basename(path).replace('.json', '')}.json"
            with open(out_path, 'w') as f:
                json.dump(output, f, indent=2,
                          default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
            print(f"\nSaved: {out_path}")
            print()


if __name__ == '__main__':
    main()
