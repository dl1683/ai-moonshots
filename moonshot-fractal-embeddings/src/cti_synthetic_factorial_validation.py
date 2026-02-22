#!/usr/bin/env python -u
"""
Synthetic Gaussian Factorial Validation (Feb 2026)
===================================================
Validates the 3-arm orthogonal causal factorial in PURE GAUSSIAN setting.

Motivation: ViT CIFAR-10 arm_C shows r=0.637 (should be ~0).
Hypothesis: K=10 too small in continuous visual manifold.
This test: K=20 pure Gaussian -> should give clean arm_C r~0.

Arms:
  A: Move j1 (nearest competitor) -> q_ci should change (CAUSAL)
  B: Move j2 (2nd nearest competitor only) -> q_ci should change if 2-layer
  C: Move jK (farthest class) -> q_ci should NOT change (NEGATIVE CONTROL)

Pre-registered:
  arm_A_r > 0.9 (strong causal)
  arm_C_r < 0.2 (clean negative control)
  arm_B identifies 1-layer vs 2-layer

Also tests: K=5 (small K confound, like CIFAR-10) vs K=20 (clean)
"""

import numpy as np
import json
from scipy.stats import pearsonr

np.random.seed(42)

K_VALUES = [5, 10, 20, 30]    # different K to show confound
D = 128                         # embedding dim
N_PER_CLASS = 200
N_TRIALS = 10                   # configs per K
DELTAS_AB = np.linspace(-3.0, 3.0, 13)  # for arms A and B (vary nearest competitor)
DELTAS_C_FRAC = 0.3  # arm_C: use at most 30% of the j1-jK gap (never lets jK approach j1)
A_RENORM = 1.0535
OUT_JSON = "results/cti_synthetic_factorial_validation.json"


def compute_kappa_nearest(X, y):
    classes = np.unique(y)
    K = len(classes)
    d = X.shape[1]
    means = {c: X[y == c].mean(0) for c in classes}
    within_var = np.mean([np.mean(np.sum((X[y == c] - means[c])**2, axis=1)) for c in classes])
    sigma_W = np.sqrt(within_var / d)
    min_dist = np.inf
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i >= j: continue
            dist = np.linalg.norm(means[ci] - means[cj])
            if dist < min_dist:
                min_dist = dist
    return float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))


def compute_q(X, y, K):
    """1-NN accuracy q on held-out 20%."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    return float((acc - 1.0/K) / (1.0 - 1.0/K))


def make_gaussian_dataset(K, D, sigma_W, centroid_scale=1.0, seed=0):
    """Make K Gaussian classes with random centroids."""
    rng = np.random.RandomState(seed)
    centroids = rng.randn(K, D) * centroid_scale
    X_list, y_list = [], []
    for c in range(K):
        Xc = rng.randn(N_PER_CLASS, D) * sigma_W + centroids[c]
        X_list.append(Xc)
        y_list.extend([c] * N_PER_CLASS)
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, centroids


def get_class_order(X, y, focus_class):
    """Get classes ordered by distance from focus_class mean."""
    K = len(np.unique(y))
    means = {c: X[y == c].mean(0) for c in range(K)}
    mu_i = means[focus_class]
    others = [c for c in range(K) if c != focus_class]
    dists = [(c, np.linalg.norm(mu_i - means[c])) for c in others]
    dists.sort(key=lambda x: x[1])
    return dists  # [(class, dist), ...] sorted by distance


def move_centroid(X, y, centroids, focus_class, target_class, delta):
    """
    Move target_class centroid by delta in direction of focus_class.
    delta > 0: move closer; delta < 0: move farther
    """
    mu_i = centroids[focus_class]
    mu_j = centroids[target_class].copy()
    direction = (mu_i - mu_j) / (np.linalg.norm(mu_i - mu_j) + 1e-10)

    centroids_new = centroids.copy()
    centroids_new[target_class] = mu_j + delta * direction

    X_new = X.copy()
    mask = y == target_class
    X_new[mask] = X[mask] + delta * direction  # shift all class j samples

    return X_new, centroids_new


def run_arm(X, y, centroids, focus_class, target_class, K, deltas=None):
    """Run one arm: vary target_class distance and measure q_focus."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    if deltas is None:
        deltas = DELTAS_AB
    kappas = []
    q_cis = []
    for delta in deltas:
        X_new, c_new = move_centroid(X, y, centroids, focus_class, target_class, delta)
        k = compute_kappa_nearest(X_new, y)

        # q for focus_class in context of ALL classes (held-out split)
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X_new, y, test_size=0.2,
                                                        random_state=42, stratify=y)
        except Exception:
            X_tr, X_te, y_tr, y_te = train_test_split(X_new, y, test_size=0.2, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_tr, y_tr)
        mask_te = y_te == focus_class
        if mask_te.sum() == 0:
            continue
        acc_ci = knn.score(X_te[mask_te], y_te[mask_te])
        q_ci_val = float((acc_ci - 1.0/K) / (1.0 - 1.0/K))
        kappas.append(k)
        q_cis.append(q_ci_val)

    if len(kappas) < 3:
        return float('nan'), kappas, q_cis
    # If kappas are all the same (arm_C bounded: jK never approaches j1, kappa_nearest unchanged)
    # then r is undefined from kappa perspective, but q_ci variation = f(delta) is informative
    # Return r between delta VALUES and q_ci (tests if jK movement itself changes q)
    if np.std(kappas) < 1e-6:
        # kappa doesn't change -> check if q changes with delta
        delta_arr = np.array(deltas[:len(q_cis)])
        if len(delta_arr) >= 3 and np.std(q_cis) > 1e-6:
            r_delta, _ = pearsonr(delta_arr, q_cis)
            # Negative control passes if |r_delta| < 0.2 (q_ci unaffected by jK movement)
            return float(r_delta), kappas, q_cis
        return 0.0, kappas, q_cis  # No kappa change, no q change -> perfect negative control
    r, _ = pearsonr(kappas, q_cis)
    return float(r), kappas, q_cis


def run_one_K_trial(K, trial_seed):
    """Run all 3 arms for one K and trial."""
    sigma_W = 1.0 / np.sqrt(D)
    # Scale centroids so kappa_nearest ≈ 1.0
    centroid_scale = 1.0  # start with this
    X, y, centroids = make_gaussian_dataset(K, D, sigma_W, centroid_scale, seed=trial_seed)

    # Adjust scale for kappa ~ 1.0
    for _ in range(20):
        k = compute_kappa_nearest(X, y)
        if abs(k - 1.0) < 0.1: break
        centroid_scale *= 1.0 / max(k, 0.01)
        X, y, centroids = make_gaussian_dataset(K, D, sigma_W, centroid_scale, seed=trial_seed)

    kappa_base = compute_kappa_nearest(X, y)

    # Pick focus class = 0
    focus = 0
    ordered = get_class_order(X, y, focus)
    j1 = ordered[0][0]   # nearest
    j2 = ordered[1][0]   # 2nd nearest
    jK = ordered[-1][0]  # farthest

    # Arm A: move j1 (nearest competitor)
    r_A, kappas_A, qs_A = run_arm(X, y, centroids, focus, j1, K, DELTAS_AB)

    # Arm B: move j2 (2nd nearest)
    r_B, kappas_B, qs_B = run_arm(X, y, centroids, focus, j2, K, DELTAS_AB)

    # Arm C: move jK (farthest) — use BOUNDED delta so jK never approaches j1
    # Gap = distance between j1 and jK centroids
    mu_j1 = X[y == j1].mean(0)
    mu_jK = X[y == jK].mean(0)
    mu_focus = X[y == focus].mean(0)
    gap_j1 = np.linalg.norm(mu_focus - mu_j1)
    gap_jK = np.linalg.norm(mu_focus - mu_jK)
    # Max delta for C: keep jK at least at j1's distance
    max_delta_C = (gap_jK - gap_j1) * DELTAS_C_FRAC
    deltas_C = np.linspace(-max_delta_C, max_delta_C, 13)
    r_C, kappas_C, qs_C = run_arm(X, y, centroids, focus, jK, K, deltas_C)

    return {
        'K': K, 'trial': trial_seed,
        'kappa_base': float(kappa_base),
        'j1': j1, 'j2': j2, 'jK': jK,
        'arm_A_r': float(r_A),
        'arm_B_r': float(r_B),
        'arm_C_r': float(r_C),
        'arm_A_pass': r_A > 0.9,
        'arm_C_pass': abs(r_C) < 0.2,
    }


def main():
    print("=" * 60)
    print("SYNTHETIC GAUSSIAN FACTORIAL VALIDATION")
    print("=" * 60)
    print(f"K_VALUES={K_VALUES}, D={D}, N_PER_CLASS={N_PER_CLASS}, N_TRIALS={N_TRIALS}")
    print()

    all_results = []
    for K in K_VALUES:
        print(f"\n--- K={K} ---")
        K_results = []
        for trial in range(N_TRIALS):
            rec = run_one_K_trial(K, trial * 100 + K)
            K_results.append(rec)
            all_results.append(rec)
            print(f"  Trial {trial}: kappa_base={rec['kappa_base']:.3f}, "
                  f"arm_A_r={rec['arm_A_r']:.4f}, arm_B_r={rec['arm_B_r']:.4f}, "
                  f"arm_C_r={rec['arm_C_r']:.4f}  "
                  f"[A:{'PASS' if rec['arm_A_pass'] else 'fail'}, "
                  f"C:{'PASS' if rec['arm_C_pass'] else 'fail'}]", flush=True)

        mean_A = float(np.nanmean([r['arm_A_r'] for r in K_results]))
        mean_B = float(np.nanmean([r['arm_B_r'] for r in K_results]))
        mean_C = float(np.nanmean([abs(r['arm_C_r']) for r in K_results]))
        pass_A = sum(r['arm_A_pass'] for r in K_results)
        pass_C = sum(r['arm_C_pass'] for r in K_results)
        print(f"\n  K={K} SUMMARY: mean_A={mean_A:.3f}, mean_B={mean_B:.3f}, mean_|C|={mean_C:.3f}")
        print(f"  Arm A pass: {pass_A}/{N_TRIALS}, Arm C pass: {pass_C}/{N_TRIALS}")

    print("\n\n" + "=" * 60)
    print("FULL SUMMARY BY K")
    print("=" * 60)
    print(f"{'K':>5} {'mean_A_r':>10} {'mean_|C|_r':>12} {'A_pass':>8} {'C_pass':>8}")
    by_K = {}
    for K in K_VALUES:
        recs = [r for r in all_results if r['K'] == K]
        mean_A = float(np.nanmean([r['arm_A_r'] for r in recs]))
        mean_C = float(np.nanmean([abs(r['arm_C_r']) for r in recs]))
        pass_A = sum(r['arm_A_pass'] for r in recs)
        pass_C = sum(r['arm_C_pass'] for r in recs)
        print(f"  {K:>5} {mean_A:>10.4f} {mean_C:>12.4f} {pass_A:>5}/{N_TRIALS} {pass_C:>5}/{N_TRIALS}")
        by_K[K] = {'mean_A_r': mean_A, 'mean_C_r': mean_C,
                   'pass_A_rate': pass_A/N_TRIALS, 'pass_C_rate': pass_C/N_TRIALS}

    print()
    print("KEY QUESTION: Does arm_C r increase as K decreases?")
    Ks = K_VALUES
    Cs = [by_K[k]['mean_C_r'] for k in Ks]
    if len(Ks) > 2:
        from scipy.stats import pearsonr as pr
        r_KC, _ = pr(Ks, Cs)
        print(f"r(K, arm_C_r) = {r_KC:.4f}  (negative = smaller K -> worse negative control)")

    out = {
        'experiment': 'synthetic_gaussian_factorial_validation',
        'K_VALUES': K_VALUES, 'D': D, 'N_PER_CLASS': N_PER_CLASS, 'N_TRIALS': N_TRIALS,
        'by_K': by_K,
        'records': all_results,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
