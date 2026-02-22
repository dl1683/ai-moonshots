"""
Quick analysis of control law validation results as they come in.
Run this at any time to see current state.
"""
import json
import numpy as np
import sys

RESULT_PATH = "results/cti_control_law_validation.json"
A_RENORM = 1.0535


def logit(q, eps=1e-6):
    q = max(min(q, 1 - eps), eps)
    return np.log(q / (1 - q))


def arm_stats(seeds):
    if not seeds:
        return None
    final_q = [s['final_q'] for s in seeds]
    final_k = [s['final_kappa'] for s in seeds]
    final_d = [s.get('final_d_eff', None) for s in seeds]
    final_ke = [s.get('final_kappa_eff', None) for s in seeds]

    stats = {
        'n': len(seeds),
        'q_mean': np.mean(final_q),
        'q_std': np.std(final_q),
        'kappa_mean': np.mean(final_k),
        'kappa_std': np.std(final_k),
    }
    if all(d is not None for d in final_d):
        stats['d_eff_mean'] = np.mean(final_d)
        stats['d_eff_std'] = np.std(final_d)
    if all(ke is not None for ke in final_ke):
        stats['kappa_eff_mean'] = np.mean(final_ke)
        stats['kappa_eff_std'] = np.std(final_ke)
    return stats


def analyze_control_law():
    with open(RESULT_PATH) as f:
        d = json.load(f)

    print("=" * 70)
    print("CONTROL LAW VALIDATION: CURRENT STATE")
    print("Status:", d['status'])
    print("=" * 70)
    print()

    arm_data = {}
    for arm in ['ce', 'nc', 'anti_nc']:
        seeds = d['results'].get(arm, [])
        st = arm_stats(seeds)
        arm_data[arm] = {'seeds': seeds, 'stats': st}

        print(f"ARM: {arm.upper()} ({len(seeds)} seeds complete)")
        if st:
            print(f"  q: {st['q_mean']:.4f} +/- {st['q_std']:.4f}")
            print(f"  kappa: {st['kappa_mean']:.4f} +/- {st['kappa_std']:.4f}")
            if 'd_eff_mean' in st:
                print(f"  d_eff_gram: {st['d_eff_mean']:.1f} +/- {st['d_eff_std']:.1f}")
            if 'kappa_eff_mean' in st:
                print(f"  kappa_eff_gram: {st['kappa_eff_mean']:.3f} +/- {st['kappa_eff_std']:.3f}")
        print()

    # Cross-arm analysis if we have CE + NC
    ce_st = arm_data['ce']['stats']
    nc_st = arm_data['nc']['stats']
    anti_nc_st = arm_data['anti_nc']['stats']

    if ce_st:
        print("CE BASELINE:")
        print(f"  q = {ce_st['q_mean']:.4f} (3 seeds)")
        print(f"  kappa = {ce_st['kappa_mean']:.4f}")
        if 'd_eff_mean' in ce_st:
            print(f"  d_eff_gram = {ce_st['d_eff_mean']:.1f}")
        print()

    if ce_st and nc_st:
        delta_q = nc_st['q_mean'] - ce_st['q_mean']
        delta_kappa = nc_st['kappa_mean'] - ce_st['kappa_mean']
        print("NC+ vs CE (cross-arm at epoch 60):")
        print(f"  delta_q = {delta_q:+.4f}  {'PASS (q UP)' if delta_q > 0 else 'FAIL (q DOWN)'}")
        print(f"  delta_kappa = {delta_kappa:+.4f}  {'kappa UP' if delta_kappa > 0 else 'kappa DOWN'}")
        if 'd_eff_mean' in nc_st and 'd_eff_mean' in ce_st:
            delta_d = nc_st['d_eff_mean'] - ce_st['d_eff_mean']
            delta_ke = nc_st.get('kappa_eff_mean', 0) - ce_st.get('kappa_eff_mean', 0)
            print(f"  delta_d_eff_gram = {delta_d:+.1f}  {'d_eff UP' if delta_d > 0 else 'd_eff DOWN'}")
            print(f"  delta_kappa_eff_gram = {delta_ke:+.3f}  {'kappa_eff UP' if delta_ke > 0 else 'kappa_eff DOWN'}")
        logit_ce = logit(ce_st['q_mean'])
        logit_nc = logit(nc_st['q_mean'])
        actual_delta_logit = logit_nc - logit_ce
        if 'kappa_eff_mean' in nc_st and 'kappa_eff_mean' in ce_st:
            pred_delta = A_RENORM * (nc_st['kappa_eff_mean'] - ce_st['kappa_eff_mean'])
            print(f"  Actual delta_logit_q = {actual_delta_logit:+.4f}")
            print(f"  Predicted (A_renorm * delta_kappa_eff) = {pred_delta:+.4f}")
            print(f"  Ratio (actual/pred) = {actual_delta_logit/pred_delta:.3f}" if pred_delta != 0 else "  (pred=0)")
        print()

    if ce_st and anti_nc_st:
        delta_q_anti = anti_nc_st['q_mean'] - ce_st['q_mean']
        delta_kappa_anti = anti_nc_st['kappa_mean'] - ce_st['kappa_mean']
        print("anti_NC vs CE (cross-arm at epoch 60):")
        print(f"  delta_q = {delta_q_anti:+.4f}  {'PASS (q DOWN as expected)' if delta_q_anti < 0 else 'FAIL (q UP or same)'}")
        print(f"  delta_kappa = {delta_kappa_anti:+.4f}  {'kappa DOWN as expected' if delta_kappa_anti < 0 else 'kappa UP'}")
        print()

    # Within-arm slope analysis (CE arm)
    if arm_data['ce']['seeds']:
        print("WITHIN-CE-ARM SLOPE ANALYSIS:")
        all_ckpts = []
        for s in arm_data['ce']['seeds']:
            for ck in s.get('checkpoints', []):
                all_ckpts.append(ck)
        if len(all_ckpts) >= 2:
            logit_q = np.array([c['logit_q'] for c in all_ckpts])
            kappa = np.array([c['kappa'] for c in all_ckpts])
            kappa_eff = np.array([c['kappa_eff'] for c in all_ckpts])

            A_mat = np.column_stack([kappa, np.ones(len(kappa))])
            c, _, _, _ = np.linalg.lstsq(A_mat, logit_q, rcond=None)
            alpha = c[0]
            d_eff_cls = (alpha / A_RENORM) ** 2
            pred = c[0] * kappa + c[1]
            r2 = 1 - np.sum((logit_q - pred) ** 2) / np.sum((logit_q - logit_q.mean()) ** 2)
            print(f"  Slope (logit_q vs kappa): alpha={alpha:.4f}, R2={r2:.4f}")
            print(f"  d_eff_cls = (alpha/A_renorm)^2 = {d_eff_cls:.4f}")

            A2 = np.column_stack([kappa_eff, np.ones(len(kappa_eff))])
            c2, _, _, _ = np.linalg.lstsq(A2, logit_q, rcond=None)
            pred2 = c2[0] * kappa_eff + c2[1]
            r2_2 = 1 - np.sum((logit_q - pred2) ** 2) / np.sum((logit_q - logit_q.mean()) ** 2)
            print(f"  Slope (logit_q vs kappa_eff_gram): {c2[0]:.4f} (expected {A_RENORM}), R2={r2_2:.4f}")
            print(f"  Slope ratio (actual/expected): {c2[0]/A_RENORM:.3f}")
        print()

    print("KEY PREDICTIONS (pre-registered):")
    print(f"  Test 1 (Delta logit_q = A_renorm * Delta kappa_eff): EXPECTED FAIL")
    print(f"    Reason: d_eff_gram is wrong measure (157x inflation vs d_eff_cls)")
    print(f"  Test 2 (Cross-arm direction): NC+ q > CE q, anti_NC q < CE q")
    print(f"    Verdict: {'CE DONE, NC RUNNING...' if nc_st is None else 'SEE ABOVE'}")
    print(f"  Test 3 (d_eff_sig validation): requires cti_deff_signal_validation.py")


if __name__ == '__main__':
    analyze_control_law()
