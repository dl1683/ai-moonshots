"""
Comprehensive Control Law Analysis

Two tests (both required):

TEST 1 (Pre-registered, across-time):
  Delta logit(q) = A_renorm * Delta(kappa_eff) across checkpoints within each arm.
  Expected to FAIL because d_eff_gram decreases while q increases (training dynamics).

TEST 2 (Cross-arm, same epoch - CORRECT causal test):
  At each epoch t: [logit_q(NC+,t) - logit_q(CE,t)] = A_renorm * [kappa_eff(NC+,t) - kappa_eff(CE,t)]
  This is the true between-intervention test. NC-loss changes kappa_eff; does q follow?

KEY INSIGHT (Feb 22 2026):
  The Gram matrix d_eff (tr(W)^2/tr(W^2)) is the GLOBAL participation ratio
  (~200 at epoch 25 for d=512 ResNet-18). It decreases during neural collapse.
  The "inferred d_eff" from alpha = A_renorm*sqrt(d_eff) is the effective
  classification-relevant dimensionality (~1.37), much smaller.
  The two are different quantities:
    - d_eff_gram: total within-class variance dimensionality (global)
    - d_eff_classification: effective dimensionality for kNN decision boundary

  TRAINING DYNAMICS: kappa_eff = sqrt(d_eff_gram) * kappa_nearest
    - d_eff_gram decreases during training (NC collapse)
    - kappa_nearest increases during training (class separation)
    - Net effect on kappa_eff depends on WHICH effect dominates

THIRD TEST (Training dynamics):
  Track d_eff_gram * kappa_nearest^2 vs epoch. Is this monotone with logit_q?
  Also: is kappa_nearest alone more predictive of logit_q across checkpoints?
"""

import json
import numpy as np
import sys
import os

RESULT_PATH = "results/cti_control_law_validation.json"
OUTPUT_PATH = "results/cti_control_law_analysis.json"
A_RENORM_K20 = 1.0535
K = 20


def logit(p, eps=0.001):
    p = max(min(p, 1 - eps), eps)
    return np.log(p / (1 - p))


def analyze_control_law_across_time(all_results, A_renorm=A_RENORM_K20):
    """
    Test 1 (pre-registered): Delta logit(q) = A_renorm * Delta(kappa_eff)
    across training checkpoints within each arm.
    """
    all_deltas = []
    arm_deltas = {}

    for arm, arm_results in all_results.items():
        arm_deltas[arm] = []
        for res in arm_results:
            ckpts = res.get('checkpoints', [])
            for i in range(len(ckpts)):
                for j in range(i + 1, len(ckpts)):
                    c1, c2 = ckpts[i], ckpts[j]
                    if ('logit_q' not in c1 or 'kappa_eff' not in c1 or
                            'logit_q' not in c2 or 'kappa_eff' not in c2):
                        continue
                    delta_logit = c2['logit_q'] - c1['logit_q']
                    delta_kappa_eff = c2['kappa_eff'] - c1['kappa_eff']
                    pair = {
                        'arm': arm, 'seed': res['seed'],
                        'epoch1': c1['epoch'], 'epoch2': c2['epoch'],
                        'delta_logit_q': float(delta_logit),
                        'delta_kappa_eff': float(delta_kappa_eff),
                        'predicted_A_renorm': float(A_renorm * delta_kappa_eff),
                        'residual': float(delta_logit - A_renorm * delta_kappa_eff),
                    }
                    all_deltas.append(pair)
                    arm_deltas[arm].append(pair)

    if not all_deltas:
        return {'status': 'no_data', 'n_pairs': 0}

    actuals = np.array([p['delta_logit_q'] for p in all_deltas])
    predicted = np.array([p['predicted_A_renorm'] for p in all_deltas])
    residuals = actuals - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Linear slope of delta_logit_q vs delta_kappa_eff (should equal A_renorm)
    if len(actuals) >= 2:
        X = np.column_stack([actuals, np.ones(len(actuals))])
        # slope of delta_logit_q vs delta_kappa_eff
        X2 = np.column_stack([predicted / A_renorm, np.ones(len(predicted))])  # delta_kappa_eff
        actual_slope = np.polyfit(predicted / A_renorm, actuals, 1)
        empirical_slope = float(actual_slope[0])
    else:
        empirical_slope = None

    per_arm = {}
    for arm, pairs in arm_deltas.items():
        if pairs:
            arm_resids = np.array([p['residual'] for p in pairs])
            arm_deltas_logit = np.array([p['delta_logit_q'] for p in pairs])
            arm_deltas_ke = np.array([p['delta_kappa_eff'] for p in pairs])
            per_arm[arm] = {
                'mean_residual': float(np.mean(arm_resids)),
                'std_residual': float(np.std(arm_resids)),
                'mean_delta_logit': float(np.mean(arm_deltas_logit)),
                'mean_delta_kappa_eff': float(np.mean(arm_deltas_ke)),
            }

    arm_means = [d['mean_residual'] for d in per_arm.values() if d]
    max_arm_diff = float(max(arm_means) - min(arm_means)) if arm_means else None
    invariant = bool(max_arm_diff < 0.1) if max_arm_diff is not None else None

    return {
        'r2': r2,
        'empirical_slope': empirical_slope,
        'A_renorm_preregistered': A_renorm,
        'n_pairs': len(all_deltas),
        'PASS_r2': bool(r2 > 0.9),
        'PASS_invariance': bool(invariant) if invariant is not None else False,
        'max_arm_mean_residual_diff': max_arm_diff,
        'per_arm': per_arm,
        'all_deltas': all_deltas,
    }


def analyze_control_law_cross_arm(all_results, A_renorm=A_RENORM_K20):
    """
    Test 2 (correct causal test): at each epoch, compare arms vs CE baseline.
    Tests whether between-intervention kappa_eff differences predict logit_q differences.
    """
    if 'ce' not in all_results:
        return {'status': 'no_ce_arm'}

    cross_arm_pairs = []
    for checkpoint_idx in range(10):  # loop over checkpoint slots
        for seed_idx in range(len(all_results['ce'])):
            ce_res = all_results['ce'][seed_idx]
            ce_ckpts = ce_res.get('checkpoints', [])
            if checkpoint_idx >= len(ce_ckpts):
                break
            ce_ck = ce_ckpts[checkpoint_idx]

            for arm in ['nc', 'anti_nc']:
                if arm not in all_results:
                    continue
                if seed_idx >= len(all_results[arm]):
                    continue
                arm_res = all_results[arm][seed_idx]
                arm_ckpts = arm_res.get('checkpoints', [])
                if checkpoint_idx >= len(arm_ckpts):
                    continue
                arm_ck = arm_ckpts[checkpoint_idx]

                if ('logit_q' not in ce_ck or 'logit_q' not in arm_ck or
                        'kappa_eff' not in ce_ck or 'kappa_eff' not in arm_ck):
                    continue

                delta_logit = arm_ck['logit_q'] - ce_ck['logit_q']
                delta_kappa_eff = arm_ck['kappa_eff'] - ce_ck['kappa_eff']
                delta_kappa = arm_ck.get('kappa', 0) - ce_ck.get('kappa', 0)
                delta_d_eff = arm_ck.get('d_eff', 0) - ce_ck.get('d_eff', 0)
                delta_q = arm_ck.get('q', 0) - ce_ck.get('q', 0)

                cross_arm_pairs.append({
                    'arm': arm,
                    'seed': ce_res['seed'],
                    'epoch': ce_ck['epoch'],
                    'delta_logit_q': float(delta_logit),
                    'delta_kappa_eff': float(delta_kappa_eff),
                    'delta_kappa': float(delta_kappa),
                    'delta_d_eff': float(delta_d_eff),
                    'delta_q': float(delta_q),
                    'predicted_A_renorm': float(A_renorm * delta_kappa_eff),
                    'residual_kappa_eff': float(delta_logit - A_renorm * delta_kappa_eff),
                    'residual_kappa_only': float(delta_logit - A_renorm * delta_kappa),
                    # CE and arm values
                    'ce_q': ce_ck.get('q'), 'arm_q': arm_ck.get('q'),
                    'ce_kappa': ce_ck.get('kappa'), 'arm_kappa': arm_ck.get('kappa'),
                    'ce_d_eff': ce_ck.get('d_eff'), 'arm_d_eff': arm_ck.get('d_eff'),
                    'ce_kappa_eff': ce_ck.get('kappa_eff'), 'arm_kappa_eff': arm_ck.get('kappa_eff'),
                })

    if not cross_arm_pairs:
        return {'status': 'no_data'}

    actuals = np.array([p['delta_logit_q'] for p in cross_arm_pairs])
    predicted = np.array([p['predicted_A_renorm'] for p in cross_arm_pairs])
    residuals = actuals - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2_kappa_eff = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Also test kappa_nearest without d_eff factor
    resids_kappa = np.array([p['residual_kappa_only'] for p in cross_arm_pairs])
    ss_res_k = np.sum(resids_kappa ** 2)
    r2_kappa_only = float(1 - ss_res_k / ss_tot) if ss_tot > 0 else 0.0

    # Empirical slope: delta_logit_q vs delta_kappa_eff
    if len(actuals) >= 2:
        dke = np.array([p['delta_kappa_eff'] for p in cross_arm_pairs])
        dk = np.array([p['delta_kappa'] for p in cross_arm_pairs])
        if dke.std() > 1e-6:
            emp_slope_ke = float(np.polyfit(dke, actuals, 1)[0])
        else:
            emp_slope_ke = None
        if dk.std() > 1e-6:
            emp_slope_k = float(np.polyfit(dk, actuals, 1)[0])
        else:
            emp_slope_k = None
    else:
        emp_slope_ke = emp_slope_k = None

    # Per arm results
    per_arm = {}
    for arm in ['nc', 'anti_nc']:
        arm_pairs = [p for p in cross_arm_pairs if p['arm'] == arm]
        if arm_pairs:
            sign_correct = sum(
                1 for p in arm_pairs
                if (arm == 'nc' and p['delta_q'] > 0 and p['delta_kappa_eff'] > 0) or
                   (arm == 'nc' and p['delta_q'] > 0) or
                   (arm == 'anti_nc' and p['delta_q'] < 0)
            )
            per_arm[arm] = {
                'n_pairs': len(arm_pairs),
                'mean_delta_q': float(np.mean([p['delta_q'] for p in arm_pairs])),
                'mean_delta_kappa_eff': float(np.mean([p['delta_kappa_eff'] for p in arm_pairs])),
                'mean_delta_kappa': float(np.mean([p['delta_kappa'] for p in arm_pairs])),
                'mean_delta_d_eff': float(np.mean([p['delta_d_eff'] for p in arm_pairs])),
                'mean_residual': float(np.mean([p['residual_kappa_eff'] for p in arm_pairs])),
                'sign_consistent': int(sign_correct),
            }

    return {
        'r2_kappa_eff': r2_kappa_eff,
        'r2_kappa_only': r2_kappa_only,
        'empirical_slope_kappa_eff': emp_slope_ke,
        'empirical_slope_kappa_only': emp_slope_k,
        'A_renorm_preregistered': A_renorm,
        'n_pairs': len(cross_arm_pairs),
        'PASS_cross_arm': bool(r2_kappa_eff > 0.5),
        'per_arm': per_arm,
        'all_pairs': cross_arm_pairs,
    }


def analyze_training_dynamics(all_results):
    """
    Test 3: training dynamics.
    For each arm and seed: track d_eff, kappa_nearest, kappa_eff, logit_q vs epoch.
    Key question: does d_eff decrease as kappa_nearest increases?
    Does kappa_eff increase or decrease overall?
    """
    trajectory_summary = {}
    for arm, arm_results in all_results.items():
        for res in arm_results:
            key = f"{arm}_s{res['seed']}"
            ckpts = res.get('checkpoints', [])
            if not ckpts:
                continue
            traj = {
                'epochs': [c['epoch'] for c in ckpts],
                'q': [c.get('q', None) for c in ckpts],
                'kappa': [c.get('kappa', None) for c in ckpts],
                'd_eff': [c.get('d_eff', None) for c in ckpts],
                'kappa_eff': [c.get('kappa_eff', None) for c in ckpts],
                'logit_q': [c.get('logit_q', None) for c in ckpts],
            }
            # Correlation: does kappa_eff predict logit_q along this arm?
            ke = np.array([v for v in traj['kappa_eff'] if v is not None])
            lq = np.array([v for v in traj['logit_q'] if v is not None])
            knn = np.array([v for v in traj['kappa'] if v is not None])
            de = np.array([v for v in traj['d_eff'] if v is not None])
            if len(ke) >= 2:
                r_ke_logit = float(np.corrcoef(ke, lq)[0, 1]) if ke.std() > 0 else None
                r_knn_logit = float(np.corrcoef(knn, lq)[0, 1]) if knn.std() > 0 else None
                r_de = float(np.corrcoef(de, lq)[0, 1]) if de.std() > 0 else None
                direction_kappa_eff = 'increasing' if ke[-1] > ke[0] else 'decreasing'
                traj['r_kappa_eff_logit'] = r_ke_logit
                traj['r_kappa_only_logit'] = r_knn_logit
                traj['r_d_eff_logit'] = r_de
                traj['direction_kappa_eff'] = direction_kappa_eff
                traj['delta_kappa_eff'] = float(ke[-1] - ke[0])
                traj['delta_kappa'] = float(knn[-1] - knn[0])
                traj['delta_d_eff'] = float(de[-1] - de[0])
                traj['delta_logit_q'] = float(lq[-1] - lq[0])
            trajectory_summary[key] = traj
    return trajectory_summary


def analyze_snapshot_law(all_results, A_renorm=A_RENORM_K20):
    """
    Snapshot law: at each checkpoint, does logit(q) = A_renorm * kappa_eff + C hold?
    Fit logit(q) = slope * kappa_eff + intercept across all checkpoints from all arms/seeds.
    """
    all_snapshots = []
    for arm, arm_results in all_results.items():
        for res in arm_results:
            for ck in res.get('checkpoints', []):
                if 'logit_q' in ck and 'kappa_eff' in ck:
                    all_snapshots.append({
                        'arm': arm,
                        'seed': res['seed'],
                        'epoch': ck['epoch'],
                        'logit_q': ck['logit_q'],
                        'kappa_eff': ck['kappa_eff'],
                        'kappa': ck.get('kappa', None),
                        'd_eff': ck.get('d_eff', None),
                    })

    if len(all_snapshots) < 3:
        return {'status': 'insufficient_data'}

    ke = np.array([s['kappa_eff'] for s in all_snapshots])
    lq = np.array([s['logit_q'] for s in all_snapshots])
    knn = np.array([s['kappa'] for s in all_snapshots if s['kappa'] is not None])
    lq_knn = np.array([s['logit_q'] for s in all_snapshots if s['kappa'] is not None])

    # Fit with kappa_eff
    coeffs = np.polyfit(ke, lq, 1)
    r2_ke = float(np.corrcoef(ke, lq)[0, 1] ** 2) if ke.std() > 0 else 0.0
    emp_slope_ke = float(coeffs[0])
    emp_intercept_ke = float(coeffs[1])

    # Fit with kappa_nearest only
    if len(knn) >= 3:
        coeffs_k = np.polyfit(knn, lq_knn, 1)
        r2_k = float(np.corrcoef(knn, lq_knn)[0, 1] ** 2) if knn.std() > 0 else 0.0
        emp_slope_k = float(coeffs_k[0])
    else:
        r2_k = emp_slope_k = None

    # Zero-param prediction: logit_q = A_renorm * kappa_eff + C
    # Fit only C (intercept), keep slope fixed at A_renorm
    # C_opt = mean(logit_q - A_renorm * kappa_eff)
    C_opt = float(np.mean(lq - A_renorm * ke))
    residuals_zp = lq - (A_renorm * ke + C_opt)
    ss_res_zp = np.sum(residuals_zp ** 2)
    ss_tot = np.sum((lq - lq.mean()) ** 2)
    r2_zp = float(1 - ss_res_zp / ss_tot) if ss_tot > 0 else 0.0

    return {
        'n_snapshots': len(all_snapshots),
        'empirical_slope_kappa_eff': emp_slope_ke,
        'empirical_intercept': emp_intercept_ke,
        'r2_free_slope_kappa_eff': r2_ke,
        'empirical_slope_kappa_only': emp_slope_k,
        'r2_free_slope_kappa_only': r2_k,
        'A_renorm_preregistered': A_renorm,
        'C_optimal_intercept': C_opt,
        'r2_fixed_slope_kappa_eff': r2_zp,
        'PASS_r2_kappa_eff': bool(r2_ke > 0.9),
        'PASS_r2_zp': bool(r2_zp > 0.9),
    }


def main():
    if not os.path.exists(RESULT_PATH):
        print(f"ERROR: {RESULT_PATH} not found. Run cti_control_law_validation.py first.")
        sys.exit(1)

    with open(RESULT_PATH) as f:
        data = json.load(f)

    status = data.get('status', 'unknown')
    all_results = data.get('results', {})
    print(f"Status: {status}")

    # Count available checkpoints
    n_ckpts = {}
    for arm, arm_res in all_results.items():
        total = sum(len(r.get('checkpoints', [])) for r in arm_res)
        n_ckpts[arm] = total
    print(f"Checkpoints available: {n_ckpts}")
    print()

    # TEST 1: Across-time control law (pre-registered)
    print("=" * 70)
    print("TEST 1: Across-time Delta logit(q) = A_renorm * Delta(kappa_eff)")
    print("(pre-registered, expected to reveal dynamics inconsistency)")
    print("=" * 70)
    test1 = analyze_control_law_across_time(all_results)
    if test1.get('n_pairs', 0) > 0:
        print(f"  R2 = {test1['r2']:.4f} (PASS if > 0.9)")
        print(f"  Empirical slope = {test1.get('empirical_slope')}")
        print(f"  A_renorm pre-registered = {A_RENORM_K20}")
        print(f"  PASS (R2 > 0.9): {test1['PASS_r2']}")
        print(f"  PASS (invariance): {test1['PASS_invariance']}")
        print()
        print("  Per-arm residuals:")
        for arm, d in test1.get('per_arm', {}).items():
            print(f"    {arm}: mean_residual={d['mean_residual']:+.4f} "
                  f"mean_delta_logit={d['mean_delta_logit']:+.4f} "
                  f"mean_delta_ke={d['mean_delta_kappa_eff']:+.4f}")
    else:
        print("  Insufficient data for Test 1")

    # TEST 2: Cross-arm control law (correct causal test)
    print()
    print("=" * 70)
    print("TEST 2: Cross-arm Delta logit(q) = A_renorm * Delta(kappa_eff)")
    print("(at same epoch, correct causal test)")
    print("=" * 70)
    test2 = analyze_control_law_cross_arm(all_results)
    if test2.get('n_pairs', 0) > 0:
        print(f"  R2 (kappa_eff): {test2['r2_kappa_eff']:.4f}")
        print(f"  R2 (kappa only): {test2.get('r2_kappa_only'):.4f}")
        print(f"  Empirical slope (kappa_eff): {test2.get('empirical_slope_kappa_eff')}")
        print(f"  Empirical slope (kappa only): {test2.get('empirical_slope_kappa_only')}")
        print(f"  PASS (R2_kappa_eff > 0.5): {test2['PASS_cross_arm']}")
        print()
        for arm, d in test2.get('per_arm', {}).items():
            print(f"  {arm}: mean_delta_q={d['mean_delta_q']:+.4f} "
                  f"mean_delta_kappa_eff={d['mean_delta_kappa_eff']:+.4f} "
                  f"mean_delta_kappa={d['mean_delta_kappa']:+.4f} "
                  f"mean_delta_d_eff={d['mean_delta_d_eff']:+.2f}")
    else:
        print("  Insufficient data for Test 2 (need 2+ arms at same epoch)")

    # TEST 3: Training dynamics
    print()
    print("=" * 70)
    print("TEST 3: Training dynamics (is kappa_eff monotone with logit_q?)")
    print("=" * 70)
    test3 = analyze_training_dynamics(all_results)
    for key, traj in sorted(test3.items()):
        d_eff_vals = [v for v in traj['d_eff'] if v is not None]
        ke_vals = [v for v in traj['kappa_eff'] if v is not None]
        knn_vals = [v for v in traj['kappa'] if v is not None]
        lq_vals = [v for v in traj['logit_q'] if v is not None]
        if not ke_vals:
            continue
        print(f"  {key}: d_eff={[f'{v:.0f}' for v in d_eff_vals]} "
              f"kappa={[f'{v:.3f}' for v in knn_vals]} "
              f"kappa_eff={[f'{v:.2f}' for v in ke_vals]} "
              f"r(ke,lq)={traj.get('r_kappa_eff_logit', 'N/A'):.3f} "
              f"r(k,lq)={traj.get('r_kappa_only_logit', 'N/A'):.3f} "
              f"dir_ke={traj.get('direction_kappa_eff', 'N/A')}")

    # Snapshot law
    print()
    print("=" * 70)
    print("SNAPSHOT LAW: logit(q) = A_renorm * kappa_eff + C (all snapshots)")
    print("=" * 70)
    snap = analyze_snapshot_law(all_results)
    if snap.get('n_snapshots', 0) > 0:
        print(f"  n_snapshots: {snap['n_snapshots']}")
        print(f"  Empirical slope (kappa_eff): {snap['empirical_slope_kappa_eff']:.4f}")
        print(f"  R2 (free slope, kappa_eff): {snap['r2_free_slope_kappa_eff']:.4f}")
        print(f"  R2 (free slope, kappa only): {snap.get('r2_free_slope_kappa_only')}")
        print(f"  R2 (fixed slope=1.0535, kappa_eff): {snap['r2_fixed_slope_kappa_eff']:.4f}")
        print(f"  Optimal intercept C: {snap['C_optimal_intercept']:.4f}")
        print(f"  PASS (R2_kappa_eff > 0.9): {snap['PASS_r2_kappa_eff']}")
        print(f"  PASS (zero-param R2 > 0.9): {snap['PASS_r2_zp']}")

    # Save full analysis
    output = {
        'status': status,
        'test1_across_time': {k: v for k, v in test1.items() if k != 'all_deltas'},
        'test2_cross_arm': {k: v for k, v in test2.items() if k != 'all_pairs'},
        'test3_dynamics': {k: {kk: vv for kk, vv in v.items() if kk not in ('q', 'kappa', 'd_eff', 'kappa_eff', 'logit_q', 'epochs')}
                           for k, v in test3.items()},
        'snapshot_law': snap,
        'summary': {
            'test1_pass_r2': test1.get('PASS_r2', False),
            'test1_pass_invariance': test1.get('PASS_invariance', False),
            'test2_pass': test2.get('PASS_cross_arm', False),
            'snapshot_pass': snap.get('PASS_r2_kappa_eff', False),
            'empirical_slope_snapshot': snap.get('empirical_slope_kappa_eff'),
            'empirical_slope_cross_arm': test2.get('empirical_slope_kappa_eff'),
        }
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
