"""
Post-hoc Analysis: NC-Loss Quick Pilot Results Against Pre-Registered Criteria

Runs after cti_nc_loss_quick.py completes.

Tests:
  P1: delta_q = mean(q_NC) - mean(q_CE) > 0  [sign test]
  P2: delta_kappa > 0  [NC-loss increases kappa]
  P3: delta_q >= 0.02  [effect size threshold]
  P4: alpha_NC / alpha_CE  [does NC-loss change slope?]
  P5: d_eff_NC_inferred vs d_eff_CE_inferred [does NC change d_eff?]
  P6: delta_logit / delta_kappa ~= alpha  [causal rate]

SECONDARY ANALYSES:
  - Fit alpha per arm from within-seed trajectory
  - Infer d_eff from alpha using Theorem 15: d_eff = (alpha/A_renorm(K))^2
  - Compare d_eff_NC_inferred vs d_eff_CE_inferred
  - Compute ||kappa||_eff = sqrt(d_eff_inferred) * kappa_nearest
  - Test: ||kappa||_eff_NC > ||kappa||_eff_CE  [effective separation]
"""

import json
import numpy as np
import sys
from scipy import stats

# Pre-registered constants (Theorem 15)
K = 20
A_RENORM_K20 = 1.0535  # A_renorm for K=20 from Theorem 15
ALPHA_CE_PREREGISTERED = 1.365  # from CE training dynamics (Session 11)


def logit(q):
    return np.log(q / (1 - q))


def fit_alpha(checkpoints):
    """Fit slope alpha from within-seed trajectory (epochs 25, 40, 60)."""
    rows = [(ck['kappa'], logit(ck['q']))
            for ck in checkpoints
            if 0 < ck.get('q', 0) < 1 and ck.get('kappa', 0) > 0 and ck.get('epoch', 0) > 0]
    if len(rows) < 2:
        return None, None, None
    kappas = np.array([r[0] for r in rows])
    logits = np.array([r[1] for r in rows])
    X = np.column_stack([kappas, np.ones(len(kappas))])
    coeffs, _, _, _ = np.linalg.lstsq(X, logits, rcond=None)
    alpha, C = float(coeffs[0]), float(coeffs[1])
    pred = coeffs[0] * kappas + coeffs[1]
    ss_res = np.sum((logits - pred) ** 2)
    ss_tot = np.sum((logits - logits.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return alpha, C, float(r2)


def infer_d_eff(alpha, K=K, A_renorm=A_RENORM_K20):
    """Infer d_eff from alpha using Theorem 15: d_eff = (alpha/A_renorm(K))^2"""
    if alpha is None or alpha <= 0:
        return None
    return (alpha / A_renorm) ** 2


def kappa_eff(kappa, d_eff):
    """Effective separation: ||kappa||_eff = sqrt(d_eff) * kappa_nearest"""
    if d_eff is None or d_eff <= 0:
        return None
    return np.sqrt(d_eff) * kappa


def main():
    result_path = 'results/cti_nc_loss_quick.json'

    try:
        with open(result_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {result_path} not found. Run cti_nc_loss_quick.py first.")
        sys.exit(1)

    status = data.get('status', 'unknown')
    print(f"NC-Loss Quick Pilot Analysis")
    print(f"Status: {status}")
    print(f"=" * 70)

    if status == 'running':
        print("Experiment still running. Showing partial results.")
        print()

    results = data.get('results', {})
    ce_results = results.get('ce', [])
    nc_results = results.get('nc', [])

    if not ce_results:
        print("No CE results yet.")
        return

    print(f"CE seeds: {len(ce_results)}, NC seeds: {len(nc_results)}")
    print()

    # ================================================================
    # Per-seed analysis
    # ================================================================
    print("=== PER-SEED RESULTS ===")
    print()

    ce_analyses = []
    nc_analyses = []

    for label, arm_results, analyses in [('CE', ce_results, ce_analyses),
                                          ('NC', nc_results, nc_analyses)]:
        print(f"--- {label} arm ---")
        for res in arm_results:
            seed = res.get('seed', '?')
            checkpoints = res.get('checkpoints', [])
            final_q = res.get('final_q', None)
            final_kappa = res.get('final_kappa', None)
            alpha, C, r2 = fit_alpha(checkpoints)
            d_eff = infer_d_eff(alpha)
            ke = kappa_eff(final_kappa, d_eff) if final_kappa else None

            analysis = {
                'seed': seed,
                'final_q': final_q,
                'final_kappa': final_kappa,
                'alpha': alpha,
                'C': C,
                'r2_alpha': r2,
                'd_eff_inferred': d_eff,
                'kappa_eff': ke,
            }
            analyses.append(analysis)

            print(f"  seed={seed}: q={final_q:.4f} kappa={final_kappa:.4f} "
                  f"alpha={alpha:.4f} d_eff={d_eff:.3f} ||kappa||_eff={ke:.4f}"
                  if all(x is not None for x in [final_q, final_kappa, alpha, d_eff, ke])
                  else f"  seed={seed}: q={final_q} kappa={final_kappa}")
        print()

    if not nc_results:
        print("NC arm not yet complete.")
        return

    # ================================================================
    # Pre-registered tests
    # ================================================================
    print("=== PRE-REGISTERED TESTS ===")
    print()

    ce_qs = [r['final_q'] for r in ce_results if r['final_q'] is not None]
    nc_qs = [r['final_q'] for r in nc_results if r['final_q'] is not None]
    ce_ks = [r['final_kappa'] for r in ce_results if r['final_kappa'] is not None]
    nc_ks = [r['final_kappa'] for r in nc_results if r['final_kappa'] is not None]

    if not nc_qs:
        print("No NC results available yet.")
        return

    mean_ce_q = np.mean(ce_qs) if ce_qs else None
    mean_nc_q = np.mean(nc_qs) if nc_qs else None
    mean_ce_k = np.mean(ce_ks) if ce_ks else None
    mean_nc_k = np.mean(nc_ks) if nc_ks else None

    delta_q = mean_nc_q - mean_ce_q if (mean_nc_q and mean_ce_q) else None
    delta_kappa = mean_nc_k - mean_ce_k if (mean_nc_k and mean_ce_k) else None

    print(f"CE:  mean_q={mean_ce_q:.4f}  mean_kappa={mean_ce_k:.4f}")
    print(f"NC:  mean_q={mean_nc_q:.4f}  mean_kappa={mean_nc_k:.4f}")
    print(f"Delta_q={delta_q:+.4f}  delta_kappa={delta_kappa:+.4f}")
    print()

    p1 = delta_q > 0 if delta_q is not None else None
    p2 = delta_kappa > 0 if delta_kappa is not None else None
    p3 = delta_q >= 0.02 if delta_q is not None else None

    print(f"P1 (delta_q > 0):       {'PASS' if p1 else 'FAIL'}")
    print(f"P2 (delta_kappa > 0):   {'PASS' if p2 else 'FAIL'}")
    print(f"P3 (delta_q >= 0.02):   {'PASS' if p3 else 'FAIL'} [delta_q={delta_q:+.4f}]")
    print()

    # ================================================================
    # Secondary analyses
    # ================================================================
    print("=== SECONDARY ANALYSES (Theorem 15) ===")
    print()

    ce_alphas = [a['alpha'] for a in ce_analyses if a['alpha'] is not None]
    nc_alphas = [a['alpha'] for a in nc_analyses if a['alpha'] is not None]
    ce_d_effs = [a['d_eff_inferred'] for a in ce_analyses if a['d_eff_inferred'] is not None]
    nc_d_effs = [a['d_eff_inferred'] for a in nc_analyses if a['d_eff_inferred'] is not None]
    ce_ke = [a['kappa_eff'] for a in ce_analyses if a['kappa_eff'] is not None]
    nc_ke = [a['kappa_eff'] for a in nc_analyses if a['kappa_eff'] is not None]

    if ce_alphas:
        print(f"  CE  alpha: mean={np.mean(ce_alphas):.4f} std={np.std(ce_alphas):.4f}")
        print(f"  CE  d_eff: mean={np.mean(ce_d_effs):.4f} std={np.std(ce_d_effs):.4f}")
        print(f"  CE  ||kappa||_eff: mean={np.mean(ce_ke):.4f}")
        print(f"  Pre-registered: alpha_CE={ALPHA_CE_PREREGISTERED:.4f}")
        alpha_match = abs(np.mean(ce_alphas) - ALPHA_CE_PREREGISTERED) / ALPHA_CE_PREREGISTERED
        print(f"  Alpha replication error: {alpha_match:.3f} ({alpha_match:.1%})")
        print()

    if nc_alphas:
        print(f"  NC  alpha: mean={np.mean(nc_alphas):.4f} std={np.std(nc_alphas):.4f}")
        print(f"  NC  d_eff: mean={np.mean(nc_d_effs):.4f} std={np.std(nc_d_effs):.4f}")
        print(f"  NC  ||kappa||_eff: mean={np.mean(nc_ke):.4f}")
        print()

    if ce_alphas and nc_alphas:
        delta_alpha = np.mean(nc_alphas) - np.mean(ce_alphas)
        delta_d_eff = np.mean(nc_d_effs) - np.mean(ce_d_effs)
        delta_ke = np.mean(nc_ke) - np.mean(ce_ke)

        print(f"  delta_alpha = {delta_alpha:+.4f} [NC vs CE]")
        print(f"  delta_d_eff = {delta_d_eff:+.4f} [NC vs CE, negative = more NC-like]")
        print(f"  delta_||kappa||_eff = {delta_ke:+.4f}")
        print()

        if delta_alpha < 0:
            print(f"  INTERPRETATION: NC-loss DECREASED alpha -> d_eff DECREASED from "
                  f"{np.mean(ce_d_effs):.3f} to {np.mean(nc_d_effs):.3f}")
            print(f"  => More neural collapse (as theorized)")
        elif delta_alpha > 0:
            print(f"  INTERPRETATION: NC-loss INCREASED alpha -> d_eff INCREASED")
            print(f"  => Representations LESS NC-like (unexpected)")

        print()

        if delta_ke > 0:
            print(f"  ||kappa||_eff INCREASED by {delta_ke:+.4f}: representations MORE separable")
        else:
            print(f"  ||kappa||_eff DECREASED by {delta_ke:+.4f}: representations LESS separable")

        print()

    # ================================================================
    # Causal rate test (P6)
    # ================================================================
    if delta_q is not None and delta_kappa is not None and abs(delta_kappa) > 0.01:
        rate = delta_q / delta_kappa
        print(f"=== CAUSAL RATE TEST (P6) ===")
        print(f"  Observed rate delta_q/delta_kappa = {rate:.4f}")
        print(f"  Pre-registered alpha = {ALPHA_CE_PREREGISTERED:.4f}")
        # Expected: dq/dkappa ≈ alpha * q*(1-q) (chain rule on sigmoid)
        q_bar = (mean_ce_q + mean_nc_q) / 2
        expected_rate = ALPHA_CE_PREREGISTERED * q_bar * (1 - q_bar)
        print(f"  Expected (alpha*q*(1-q)): {expected_rate:.4f}")
        ratio = rate / expected_rate if expected_rate > 0 else None
        if ratio:
            print(f"  Ratio: {ratio:.3f} [PASS if 0.5 < ratio < 2.0: "
                  f"{'PASS' if 0.5 < ratio < 2.0 else 'FAIL'}]")
        print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Pre-registered primary criterion (P3: delta_q >= 0.02): "
          f"{'PASS' if p3 else 'FAIL'}")
    print(f"  CE mean_q: {mean_ce_q:.4f} | NC mean_q: {mean_nc_q:.4f} | delta: {delta_q:+.4f}")
    print()

    # Save analysis
    output = {
        'status': status,
        'ce': {
            'mean_q': float(mean_ce_q) if mean_ce_q else None,
            'mean_kappa': float(mean_ce_k) if mean_ce_k else None,
            'mean_alpha': float(np.mean(ce_alphas)) if ce_alphas else None,
            'mean_d_eff_inferred': float(np.mean(ce_d_effs)) if ce_d_effs else None,
            'mean_kappa_eff': float(np.mean(ce_ke)) if ce_ke else None,
        },
        'nc': {
            'mean_q': float(mean_nc_q) if mean_nc_q else None,
            'mean_kappa': float(mean_nc_k) if mean_nc_k else None,
            'mean_alpha': float(np.mean(nc_alphas)) if nc_alphas else None,
            'mean_d_eff_inferred': float(np.mean(nc_d_effs)) if nc_d_effs else None,
            'mean_kappa_eff': float(np.mean(nc_ke)) if nc_ke else None,
        },
        'tests': {
            'P1_sign': bool(p1),
            'P2_kappa': bool(p2),
            'P3_effect': bool(p3),
        },
        'secondary': {
            'delta_alpha': float(np.mean(nc_alphas) - np.mean(ce_alphas)) if (nc_alphas and ce_alphas) else None,
            'delta_d_eff_inferred': float(np.mean(nc_d_effs) - np.mean(ce_d_effs)) if (nc_d_effs and ce_d_effs) else None,
            'delta_kappa_eff': float(np.mean(nc_ke) - np.mean(ce_ke)) if (nc_ke and ce_ke) else None,
        },
        'preregistered': {
            'alpha_CE': ALPHA_CE_PREREGISTERED,
            'A_renorm_K20': A_RENORM_K20,
            'K': K,
        }
    }

    out_path = 'results/cti_nc_loss_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
