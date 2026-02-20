#!/usr/bin/env python -u
"""
TWO-KNOB IDENTIFIABILITY: Disentangle kappa_pair vs kappa_spec.

Codex verdict: C (order parameter) = 9.5/10 Nobel potential.
This experiment determines whether q tracks:
  (A) kappa_pair = average pairwise ||mu_i - mu_j||^2 / d
  (B) kappa_spec = tr(S_B)/tr(S_W) [our measured kappa]

These are NOT the same when mean structure has low rank.

DESIGN:
  Vary rank r of the mean subspace (how many dims the class means span).
  Keep total pairwise distance CONSTANT (||mu_i - mu_j||^2 = delta_pair^2 = const).
  But kappa_spec depends on r: kappa_spec ~ delta_pair^2 * r / (d * sigma_W^2).

  If q ~ sigmoid(kappa_spec/sqrt(K)):
    q should INCREASE as r grows (more kappa_spec for same pairwise separation)
  If q ~ f(kappa_pair):
    q should be CONSTANT across r (same pairwise distances)

  Also: for r << d, kappa_spec << kappa_pair -> the 'spectral' view predicts harder classification
         but the pairwise view predicts same difficulty.

  PREDICTION (order parameter hypothesis): q tracks kappa_spec, not kappa_pair.
  PREDICTION (probit hypothesis): q tracks kappa_pair (pairwise distances = what kNN sees).

KEY INSIGHT:
  kNN quality depends on the ACTUAL DISTANCES between points, not the spectral ratio.
  So q should track kappa_pair (pairwise distances) NOT kappa_spec.
  But our empirical law uses kappa_spec = tr(S_B)/tr(S_W)...
  UNLESS: in typical neural networks, kappa_spec ∝ kappa_pair (which IS the case for
  isotropic Gaussian assumptions). The experiment tests this assumption.
"""

import json
import sys
import numpy as np
from scipy import stats
from scipy.special import ndtri
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

np.random.seed(42)


def theory_predict_q(kappa, K, n_per_class, d_eff):
    """Zero-parameter probit theory with bias correction."""
    n_total = K * n_per_class
    kappa_corr = max((kappa * (n_total - K) - (K - 1)) / n_total, 0.0)
    if d_eff < 2 or kappa_corr < 1e-10:
        return 0.0, 0.0  # (raw, corrected)

    def probit_q(kap, deff):
        n = max(int(n_per_class), 2)
        m = max(int((K - 1) * n_per_class), 2)
        delta_sq = 2.0 * kap * deff
        mu_s = 2.0 * deff
        sigma_s = np.sqrt(6.0 * deff)
        mu_o = 2.0 * deff + delta_sq
        sigma_o = np.sqrt(6.0 * deff + 4.0 * delta_sq)
        p_n = 1.0 / (n + 1)
        z_n = ndtri(max(p_n, 1e-15))
        phi_z_n = max(stats.norm.pdf(z_n), 1e-20)
        mu_s_min = mu_s + sigma_s * z_n
        tau_s = sigma_s / (n * phi_z_n)
        p_m = 1.0 / (m + 1)
        z_m = ndtri(max(p_m, 1e-15))
        phi_z_m = max(stats.norm.pdf(z_m), 1e-20)
        mu_o_min = mu_o + sigma_o * z_m
        tau_o = sigma_o / (m * phi_z_m)
        mu_M = mu_o_min - mu_s_min
        sigma_M = np.sqrt(tau_o**2 + tau_s**2)
        if sigma_M < 1e-20:
            return 1.0 if mu_M > 0 else 0.0
        return float(np.clip(stats.norm.cdf(mu_M / sigma_M), 0.0, 1.0))

    q_raw = probit_q(kappa, d_eff)
    q_corr = probit_q(kappa_corr, d_eff)
    return q_raw, q_corr


def generate_rank_r_mixture(K, n_per, d, rank_r, delta_pair, sigma_w=1.0, seed=42):
    """Generate K-class Gaussians with class means spanning rank_r dimensions.

    Total pairwise squared distance is held constant:
      delta_pair^2 = E[||mu_i - mu_j||^2]
    But the means span only rank_r << d dimensions.

    When rank_r < d: most dimensions are pure within-class noise.
      kappa_spec = tr(S_B)/tr(S_W) ~ K * delta_pair^2 * rank_r / (2 * d * n * sigma_w^2)
      kappa_pair = delta_pair^2
    So kappa_spec decreases with rank_r while kappa_pair stays const.
    """
    rng = np.random.RandomState(seed)

    # Generate class means in a rank_r subspace
    # Each mean: mu_k = sum_{l=1}^{rank_r} a_{kl} * e_l
    # where e_l is the l-th standard basis vector
    # Scale so that E[||mu_i - mu_j||^2] = delta_pair^2
    # ||mu_i - mu_j||^2 = sum_l (a_{il} - a_{jl})^2
    # With a_{kl} iid N(0, sigma_a^2): E[||mu_i - mu_j||^2] = 2 * rank_r * sigma_a^2
    sigma_a = np.sqrt(delta_pair**2 / (2.0 * rank_r))

    means = np.zeros((K, d))
    a = rng.randn(K, rank_r) * sigma_a
    means[:, :rank_r] = a  # means live in first rank_r dimensions

    # Generate data
    X = np.zeros((K * n_per, d))
    y = np.repeat(np.arange(K), n_per)
    for k in range(K):
        X[k * n_per:(k + 1) * n_per] = means[k] + rng.randn(n_per, d) * sigma_w

    n_total = len(y)

    # Compute kappa_spec = tr(S_B)/tr(S_W)
    grand_mean = X.mean(0)
    tr_sb, tr_sw = 0.0, 0.0
    for k in range(K):
        Xk = X[y == k]
        mu_k = Xk.mean(0)
        tr_sb += n_per * np.sum((mu_k - grand_mean)**2)
        tr_sw += np.sum((Xk - mu_k)**2)
    kappa_spec = tr_sb / max(tr_sw, 1e-10)
    kappa_spec_corr = max((kappa_spec * (n_total - K) - (K - 1)) / n_total, 0.0)

    # Compute kappa_pair = mean pairwise ||mu_i - mu_j||^2 / d (normalized)
    pairwise = []
    for i in range(K):
        for j in range(i + 1, K):
            pairwise.append(np.sum((means[i] - means[j])**2))
    kappa_pair = float(np.mean(pairwise)) / d  # normalize by d for comparability

    # LOO-kNN accuracy (using 2nd NN = LOO for train=all)
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, y)
    _, idxs = knn.kneighbors(X)
    correct = sum(y[idxs[i, 1]] == y[i] for i in range(n_total))
    knn_acc = correct / n_total
    q_obs = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)

    # Compute d_eff (ID approximation: rank of signal)
    d_eff = float(rank_r)  # the signal dimensionality

    return {
        "rank_r": rank_r,
        "K": K, "n_per": n_per, "d": d,
        "delta_pair": delta_pair,
        "delta_pair_sq": delta_pair**2,
        "kappa_spec": float(kappa_spec),
        "kappa_spec_corr": float(kappa_spec_corr),
        "kappa_pair": float(kappa_pair),
        "q_obs": float(q_obs),
        "knn_acc": float(knn_acc),
        "d_eff_signal": float(rank_r),
    }


print("=" * 70)
print("TWO-KNOB IDENTIFIABILITY: kappa_spec vs kappa_pair")
print("=" * 70)

K = 20
n_per = 100
d = 500
delta_pair = 10.0  # sqrt(delta_pair_sq) = separation magnitude

ranks = [1, 2, 5, 10, 20, 50, 100, 200, 500]

print(f"\nSettings: K={K}, n_per={n_per}, d={d}, delta_pair={delta_pair}")
print(f"Theory: q should track kappa_spec (spectral law) vs kappa_pair (probit)\n")
print(f"{'r':>4} {'kappa_s':>9} {'kappa_sc':>10} {'kappa_p':>9} {'q_obs':>7} {'sig_s':>7} {'sig_p':>7} {'err_s':>7} {'err_p':>7}")

results = []
for rank_r in ranks:
    if rank_r > d:
        continue
    row = generate_rank_r_mixture(K, n_per, d, rank_r, delta_pair, seed=42)

    kappa_s = row["kappa_spec"]
    kappa_sc = row["kappa_spec_corr"]
    kappa_p = row["kappa_pair"]
    q_obs = row["q_obs"]

    # Sigmoid predictions
    sig_spec = float(stats.logistic.cdf(kappa_s / np.sqrt(K)))
    sig_pair = float(stats.logistic.cdf(kappa_p / np.sqrt(K)))
    err_spec = abs(q_obs - sig_spec)
    err_pair = abs(q_obs - sig_pair)

    better = "spec" if err_spec < err_pair else "pair"
    print(f"{rank_r:>4} {kappa_s:>9.4f} {kappa_sc:>10.4f} {kappa_p:>9.4f} {q_obs:>7.3f} {sig_spec:>7.3f} {sig_pair:>7.3f} {err_spec:>7.3f} {err_pair:>7.3f}  ({better})")
    row["sig_spec"] = sig_spec
    row["sig_pair"] = sig_pair
    row["err_spec"] = err_spec
    row["err_pair"] = err_pair
    results.append(row)
    sys.stdout.flush()

# Summary
err_spec_all = np.mean([r["err_spec"] for r in results])
err_pair_all = np.mean([r["err_pair"] for r in results])

# Correlation between kappa_s and q_obs, and kappa_p and q_obs
kappa_s_arr = np.array([r["kappa_spec"] for r in results])
kappa_p_arr = np.array([r["kappa_pair"] for r in results])
q_obs_arr = np.array([r["q_obs"] for r in results])
rho_spec, _ = stats.spearmanr(kappa_s_arr, q_obs_arr)
rho_pair, _ = stats.spearmanr(kappa_p_arr, q_obs_arr)

print(f"\nMAE spectral law: {err_spec_all:.4f}  |  MAE pair law: {err_pair_all:.4f}")
print(f"Spearman rho (kappa_spec vs q): {rho_spec:.4f}")
print(f"Spearman rho (kappa_pair vs q): {rho_pair:.4f}")

print("\n" + "=" * 70)
print("EXPERIMENT 2: Vary K (fixed rank_r and delta_pair)")
print("=" * 70)

K_vals = [5, 10, 20, 50, 100, 200]
rank_r_fixed = 10
results2 = []
print(f"\nrank_r={rank_r_fixed}, n_per={n_per}, d={d}, delta_pair={delta_pair}")
print(f"{'K':>5} {'kappa_s':>9} {'kappa_p':>9} {'q_obs':>7} {'sig_s':>7} {'sig_p':>7}")
for K_val in K_vals:
    row = generate_rank_r_mixture(K_val, n_per, d, rank_r_fixed, delta_pair, seed=42)
    kappa_s = row["kappa_spec_corr"]
    kappa_p = row["kappa_pair"]
    q_obs = row["q_obs"]
    sig_spec = float(stats.logistic.cdf(kappa_s / np.sqrt(K_val)))
    sig_pair = float(stats.logistic.cdf(kappa_p / np.sqrt(K_val)))
    print(f"{K_val:>5} {kappa_s:>9.4f} {kappa_p:>9.4f} {q_obs:>7.3f} {sig_spec:>7.3f} {sig_pair:>7.3f}")
    row["sig_spec"] = sig_spec
    row["sig_pair"] = sig_pair
    results2.append(row)
    sys.stdout.flush()

print("\n" + "=" * 70)
print("KEY DIAGNOSTIC: Correlation within rank sweep")
print("=" * 70)

# The key question: as rank_r increases (kappa_spec increases, kappa_pair ~const),
# does q_obs increase (tracks kappa_spec) or stay flat (tracks kappa_pair)?
print("\nRank sweep: does q_obs track kappa_spec or kappa_pair?")
print("  (kappa_pair should be ~constant across ranks, kappa_spec grows)")
for r in results:
    print(f"  rank={r['rank_r']:>3}: kappa_spec={r['kappa_spec']:.4f} "
          f"kappa_pair={r['kappa_pair']:.4f} q_obs={r['q_obs']:.3f}")

kappa_spec_arr = np.array([r["kappa_spec"] for r in results])
kappa_pair_arr = np.array([r["kappa_pair"] for r in results])
q_arr = np.array([r["q_obs"] for r in results])

rho_s, p_s = stats.spearmanr(kappa_spec_arr, q_arr)
rho_p, p_p = stats.spearmanr(kappa_pair_arr, q_arr)
print(f"\n  rho(kappa_spec, q_obs) = {rho_s:.4f} (p={p_s:.4f})")
print(f"  rho(kappa_pair, q_obs) = {rho_p:.4f} (p={p_p:.4f})")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if abs(rho_s) > abs(rho_p) + 0.1:
    verdict = "SPECTRAL: q tracks kappa_spec (order parameter confirmed)"
elif abs(rho_p) > abs(rho_s) + 0.1:
    verdict = "PAIRWISE: q tracks kappa_pair (probit mechanism confirmed)"
else:
    verdict = "MIXED: both track q, need finer analysis"

print(f"\n  VERDICT: {verdict}")
print(f"  Implication: {'kappa IS the order parameter' if 'SPECTRAL' in verdict else 'probit/pairwise mechanism dominates'}")

# Save
out = {
    "experiment1_rank_sweep": results,
    "experiment2_K_sweep": results2,
    "diagnosis": {
        "rho_kappa_spec_vs_q": float(rho_s),
        "rho_kappa_pair_vs_q": float(rho_p),
        "p_kappa_spec": float(p_s),
        "p_kappa_pair": float(p_p),
        "verdict": verdict,
    },
}
out_path = RESULTS_DIR / "cti_two_knob_identifiability.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nResults saved to {out_path.name}")
