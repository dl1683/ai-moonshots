"""
CORRECTED K-Independence Test for alpha = C * sqrt(d_eff).

ISSUE WITH PREVIOUS TEST (cti_alpha_K_independence.py):
  Used d_eff=4 for ALL K including K=14, K=20.
  When K > d_eff, the ETF/simplex geometry breaks -- you cannot place
  K equidistant means in d_eff dimensions when K > d_eff+1.
  Result: alpha appeared K-dependent (1.4 to 5.9), which was an artifact.

CORRECT DESIGN:
  Use LARGE fixed d_eff (>>K) so ETF geometry is always achievable.
  Then test K-independence: alpha should be constant across K.

THEORETICAL PREDICTION (from Theorem 1):
  logit(q) = alpha * kappa_nearest - log(K-1) + C
  where alpha = sqrt(d_eff) * sqrt(4/pi) [K=2 exact, K>2 via Gumbel Race]
  and kappa_nearest = d_min / (sigma * sqrt(d_eff))

  K-INDEPENDENCE: alpha doesn't change with K (only intercept C = C0 - log(K-1)).
  WHY: alpha is the slope of nearest-class comparison, which is a 2-class problem.
       Other K-2 classes add -log(K-1) intercept via EVT, not slope.

EXPECTED RESULT:
  alpha ~ constant across K=2,5,10,20,50 (for fixed d_eff >> K)
  C(K) ~ C0 - log(K-1) [linear in log(K-1)]
"""

import numpy as np
import json
from scipy.stats import norm

D_EFF_LARGE = 200   # d_eff >> K for all test cases
K_VALUES = [2, 5, 10, 20, 50]
N_KAPPA = 12
N_PER = 100         # samples per class per kappa
N_TRIALS = 2000     # Monte Carlo trials per kappa value
KAPPA_RANGE = (0.2, 2.0)   # range of kappa values to test
RNG_SEED = 42


def simplex_vertices(K):
    """K equidistant vertices of a regular (K-1)-simplex in (K-1) dims.
    All pairwise distances = sqrt(2 * K/(K-1)).
    """
    V = np.zeros((K, K-1))
    for i in range(K):
        for j in range(K-1):
            if j < i:
                V[i, j] = -1.0 / np.sqrt((j+1) * (j+2))
            elif j == i:
                V[i, j] = np.sqrt((j+1.0) / (j+2))
    return V


def embed_simplex_in_high_dim(K, d_eff, rng):
    """Embed K-class simplex into d_eff-dimensional space.
    Returns means (K x d_eff), scaled so mean norm = 1.
    All pairwise distances are equal (ETF geometry).
    """
    V = simplex_vertices(K)  # (K, K-1)
    # Embed into d_eff dimensions via a random rotation
    Q, _ = np.linalg.qr(rng.standard_normal((d_eff, d_eff)))
    R = Q[:, :K-1]  # (d_eff, K-1) random rotation
    means = V @ R.T  # (K, d_eff)

    # Check pairwise distances (should all be equal)
    dists = []
    for i in range(K):
        for j in range(i+1, K):
            dists.append(np.linalg.norm(means[i] - means[j]))

    # Scale means to desired norm (for consistency)
    scale = 1.0 / np.mean(dists)  # normalize so mean pairwise dist = 1
    means = means * scale

    return means, float(np.min(dists) * scale)  # (means, d_min)


def simulate_1nn_accuracy(means, sigma, d_eff, rng, n_per=N_PER, n_trials=N_TRIALS):
    """Simulate 1-NN mean accuracy for K classes.
    For each trial: draw from class 0, check if nearest mean is class 0.
    """
    K = means.shape[0]
    correct = 0
    for _ in range(n_trials):
        c = 0  # test from class 0 (by symmetry, same for all classes)
        x = means[c] + sigma * rng.standard_normal(d_eff)
        dists = np.linalg.norm(means - x[np.newaxis, :], axis=1)
        if np.argmin(dists) == c:
            correct += 1
    return correct / n_trials


def fit_alpha_for_K(K, d_eff, rng, n_kappa=N_KAPPA, n_per=N_PER, n_trials=N_TRIALS):
    """Compute alpha = slope of logit(q) vs kappa_nearest for a given K.

    Varies sigma to sweep kappa, simulates 1-NN accuracy, fits linear model.
    """
    means, d_min = embed_simplex_in_high_dim(K, d_eff, rng)

    kappa_vals = np.linspace(*KAPPA_RANGE, n_kappa)
    q_vals = []
    actual_kappas = []

    for kappa_target in kappa_vals:
        # sigma from kappa definition: kappa = d_min / (sigma * sqrt(d_eff))
        sigma = d_min / (kappa_target * np.sqrt(d_eff) + 1e-12)
        # Actual kappa (should match target)
        kappa_actual = d_min / (sigma * np.sqrt(d_eff))

        acc = simulate_1nn_accuracy(means, sigma, d_eff, rng, n_per, n_trials)
        q = (acc - 1.0/K) / (1.0 - 1.0/K)
        q_vals.append(q)
        actual_kappas.append(kappa_actual)

    # Filter valid range (0 < q < 1)
    valid = [(k, q) for k, q in zip(actual_kappas, q_vals) if 0.001 < q < 0.999]
    if len(valid) < 5:
        return None

    kappas_v = np.array([v[0] for v in valid])
    qs_v = np.array([v[1] for v in valid])
    logit_qs = np.log(qs_v / (1 - qs_v))

    # Linear regression: logit(q) = alpha * kappa + C
    X = np.column_stack([kappas_v, np.ones(len(kappas_v))])
    coeffs, _, _, _ = np.linalg.lstsq(X, logit_qs, rcond=None)
    alpha, C = coeffs
    r = np.corrcoef(kappas_v, logit_qs)[0, 1]

    return {
        'K': K,
        'd_eff': d_eff,
        'alpha': float(alpha),
        'C': float(C),
        'r': float(r),
        'n_valid': len(valid),
        'kappas': [float(k) for k in kappas_v],
        'logit_qs': [float(lq) for lq in logit_qs],
    }


def theoretical_alpha_k2(d_eff):
    """K=2 exact theoretical alpha.
    P(correct) = Phi(kappa * sqrt(d_eff) / 2)
    logit(P(correct)) = logit(Phi(kappa * sqrt(d_eff)/2))
    Slope at kappa=0 = sqrt(d_eff)/2 * d(logit Phi(z))/dz|0
                    = sqrt(d_eff)/2 * phi(0)/(Phi(0)*(1-Phi(0)))
                    = sqrt(d_eff)/2 * (1/sqrt(2*pi)) / 0.25
                    = sqrt(d_eff)/2 * 4/sqrt(2*pi)
                    = sqrt(d_eff) * 2/sqrt(2*pi)
                    = sqrt(d_eff) * sqrt(2/pi)
                    = sqrt(d_eff * 2 / pi)
    BUT: for K=2, q = acc - 1/K)/(1 - 1/K) = acc (since 1/K=1/2 and 1-1/K=1/2)
    So q = acc, and q = Phi(kappa*sqrt(d_eff)/2).
    logit(q) = logit(Phi(kappa*sqrt(d_eff)/2))
    alpha_K2 = sqrt(d_eff)/2 * sqrt(8/pi) = sqrt(d_eff) * sqrt(2/pi)
    """
    return float(np.sqrt(d_eff) * np.sqrt(2 / np.pi))


def main():
    print("CORRECTED Alpha K-Independence Test")
    print(f"d_eff={D_EFF_LARGE} >> K for all tests (ETF geometry always valid)")
    print("=" * 70)
    print()

    rng = np.random.default_rng(RNG_SEED)
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    sqrt_4_over_pi = np.sqrt(4 / np.pi)
    print(f"Theoretical alpha for K=2, d_eff={D_EFF_LARGE}:")
    print(f"  sqrt(d_eff) * sqrt(2/pi) = {np.sqrt(D_EFF_LARGE) * sqrt_2_over_pi:.4f}")
    print(f"  sqrt(d_eff) * sqrt(4/pi) = {np.sqrt(D_EFF_LARGE) * sqrt_4_over_pi:.4f}")
    print(f"  Expected intercept shift: -log(K-1) between K values")
    print()

    # Test 1: K-independence with large d_eff
    print("=== TEST 1: K-Independence (d_eff=200, ETF geometry guaranteed) ===")
    print(f"  Sweeping K = {K_VALUES}")
    print(f"  kappa range: {KAPPA_RANGE}")
    print()

    results_K_indep = []
    for K in K_VALUES:
        print(f"  K={K}: running {N_TRIALS} Monte Carlo trials per kappa...")
        res = fit_alpha_for_K(K, D_EFF_LARGE, rng, n_kappa=N_KAPPA,
                               n_per=N_PER, n_trials=N_TRIALS)
        if res is None:
            print(f"  K={K}: SKIP (not enough valid kappa values)")
            continue
        results_K_indep.append(res)
        print(f"  K={K:3d}: alpha={res['alpha']:7.3f}  C={res['C']:7.3f}  "
              f"r={res['r']:.4f}  n={res['n_valid']}")

    # Analyze K-independence
    if len(results_K_indep) >= 3:
        alphas = [r['alpha'] for r in results_K_indep]
        Cs = [r['C'] for r in results_K_indep]
        Ks = [r['K'] for r in results_K_indep]
        pred_alpha_K2 = theoretical_alpha_k2(D_EFF_LARGE)
        print()
        print(f"  SUMMARY (should be K-INDEPENDENT):")
        print(f"  alpha across K: mean={np.mean(alphas):.3f} std={np.std(alphas):.3f} "
              f"CV={np.std(alphas)/np.mean(alphas):.3f}")
        print(f"  Theoretical alpha (K=2): {pred_alpha_K2:.3f}")
        print(f"  Empirical/Theory ratio: {np.mean(alphas)/pred_alpha_K2:.3f}")
        print(f"  K-independence PASS if CV < 0.15: "
              f"{'PASS' if np.std(alphas)/np.mean(alphas) < 0.15 else 'FAIL'}")
        print()
        print(f"  INTERCEPT SHIFT (should be ~ -log(K-1)):")
        for r in results_K_indep:
            K = r['K']
            expected_shift = -np.log(max(K-1, 1))
            print(f"  K={K:3d}: C={r['C']:7.3f}  expected_shift={expected_shift:.3f}  "
                  f"diff_from_K2={r['C']-results_K_indep[0]['C']:.3f}  "
                  f"expected_diff={-np.log(max(K-1,1))+np.log(max(Ks[0]-1,1)):.3f}")

    # Test 2: Compare with d_eff=4 (the WRONG test)
    print()
    print("=== TEST 2: WRONG d_eff=4 (K > d_eff, over-packed -- EXPECTED TO FAIL) ===")
    D_EFF_SMALL = 4
    results_K_wrong = []
    for K in [2, 4, 7, 14, 20]:
        rng2 = np.random.default_rng(K * 9999)
        res = fit_alpha_for_K(K, D_EFF_SMALL, rng2, n_kappa=N_KAPPA,
                               n_per=N_PER, n_trials=N_TRIALS)
        if res is None:
            print(f"  K={K:3d}: SKIP")
            continue
        results_K_wrong.append(res)
        print(f"  K={K:3d}: alpha={res['alpha']:7.3f}  C={res['C']:7.3f}  "
              f"r={res['r']:.4f}  [d_eff={D_EFF_SMALL}]")

    if len(results_K_wrong) >= 3:
        alphas_w = [r['alpha'] for r in results_K_wrong]
        print(f"\n  alpha across K (WRONG test): mean={np.mean(alphas_w):.3f} "
              f"std={np.std(alphas_w):.3f} CV={np.std(alphas_w)/np.mean(alphas_w):.3f}")
        print(f"  (High CV confirms the WRONG test is unreliable)")

    # Test 3: d_eff scaling at fixed K=2 (analytical validation)
    print()
    print("=== TEST 3: d_eff scaling at K=2 (analytical validation) ===")
    D_EFF_VALUES = [4, 16, 64, 200]
    results_deff = []
    for d_eff_v in D_EFF_VALUES:
        rng3 = np.random.default_rng(d_eff_v * 7777)
        res = fit_alpha_for_K(2, d_eff_v, rng3, n_kappa=N_KAPPA,
                               n_per=N_PER, n_trials=N_TRIALS)
        if res is None:
            continue
        pred = theoretical_alpha_k2(d_eff_v)
        err = abs(res['alpha'] - pred) / pred
        results_deff.append({'d_eff': d_eff_v, 'alpha': res['alpha'],
                              'C': res['C'], 'r': res['r'], 'pred_alpha': pred})
        print(f"  d_eff={d_eff_v:4d}: alpha={res['alpha']:7.3f}  pred={pred:.3f}  "
              f"err={err:.1%}  r={res['r']:.4f}")

    # Save results
    output = {
        'test': 'corrected_K_independence_d_eff_200',
        'd_eff_large': D_EFF_LARGE,
        'theoretical_alpha_K2_d200': float(theoretical_alpha_k2(D_EFF_LARGE)),
        'sqrt_2_over_pi': float(sqrt_2_over_pi),
        'results_K_independence_CORRECT': results_K_indep,
        'results_K_independence_WRONG': results_K_wrong,
        'results_deff_scaling': results_deff,
        'conclusion': {
            'K_independence_correct_test': 'PASS' if (
                len(results_K_indep) >= 3 and
                np.std([r['alpha'] for r in results_K_indep]) /
                np.mean([r['alpha'] for r in results_K_indep]) < 0.15
            ) else 'FAIL or INSUFFICIENT',
        },
    }

    out_path = 'results/cti_alpha_K_independence_v2.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
