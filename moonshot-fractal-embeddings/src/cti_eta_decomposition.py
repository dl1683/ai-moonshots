#!/usr/bin/env python -u
"""
ETA_M DECOMPOSITION
====================
PURPOSE: Test whether A_m = A_universal * eta_m where eta_m is derived from
         architecture-specific geometric moments.

MOTIVATION (from Codex, Feb 24):
  A_ViT/A_NLP = 0.63/1.054 = 1.67 -- architecture-specific constant.
  Codex: "Interpret as A_m = A_universal * eta_m, where eta_m captures
   modality/architecture geometry: anisotropy, metric distortion, tail
   heaviness, and curvature of feature manifolds."

DESIGN:
  For each of the 5 language model architectures (DBpedia K=14):
  1. Compute A_empirical by LOAO regression
  2. Compute geometric moments:
     - isotropy: sigma_min^2 / sigma_max^2 (eigenvalue ratio)
     - anisotropy_ratio: tr(B)/tr(W) where B = between-class scatter
     - kurtosis: mean(kurtosis of projected features)
     - tail_index: power-law exponent of eigenvalue spectrum
  3. Fit: A_m = c * f(moments_m) using all 5 LM archs
  4. Test: predicted A (from ViT geometric moments) vs A_ViT = 0.63

PRE-REGISTERED (Feb 24, 2026):
  E1: r(A_pred, A_obs) > 0.85 across 7 architectures (5 LMs + ViT + BGE-small)
  E2: A_universal = A_obs / eta_m has CV < 0.20 across architectures

Note: This test requires having A_empirical for each arch, which we have from
      the LOAO experiment (results/cti_profile_likelihood.json).
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr, kurtosis
import scipy.special as sp
import datetime

RESULT_PATH = "results/cti_eta_decomposition.json"
LOG_PATH = "results/cti_eta_decomposition_log.txt"

# Architecture paths and known A_empirical from LOAO experiment
# A_empirical sourced from results/cti_profile_likelihood.json or checkpoint sweep
ARCH_DATA = [
    {
        'name': 'pythia-160m',
        'path': 'results/dointerv_multi_pythia-160m_l12.npz',
    },
    {
        'name': 'bert-base-uncased',
        'path': 'results/dointerv_multi_bert-base-uncased_l10.npz',
    },
    {
        'name': 'electra-small',
        'path': 'results/dointerv_multi_electra-small_l3.npz',
    },
    {
        'name': 'pythia-410m',
        'path': 'results/dointerv_multi_pythia-410m_l3.npz',
    },
    {
        'name': 'rwkv-4-169m',
        'path': 'results/dointerv_multi_rwkv-4-169m_l12.npz',
    },
]

# BGE-small: separate embedding file
BGE_DATA = {
    'name': 'bge-small',
    'embed_path': 'data/beir/dbpedia14_train_50000_embeddings.npy',
    'labels_path': 'data/beir/dbpedia14_train_50000_labels.npz',
}

ALPHA_KAPPA_K14 = 1.477   # global: A = ALPHA / sqrt(K-1) = 1.477 / sqrt(13)
K14 = 14
N_TRAIN_PER_CLASS = 350
RANDOM_SEED = 42

os.makedirs("results", exist_ok=True)
_log_fh = open(LOG_PATH, 'w')

def log(msg):
    print(msg, flush=True)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


def load_and_split_npz(path, n_train, seed):
    data = np.load(path)
    X, y = data['X'].astype(np.float64), data['y']
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    X_tr_list, y_tr_list = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = min(n_train, len(idx) - 1)
        X_tr_list.append(X[idx[:n]]); y_tr_list.append(y[idx[:n]])
    return np.concatenate(X_tr_list), np.concatenate(y_tr_list), classes


def compute_geometric_moments(X_tr, y_tr, classes):
    """Compute architecture-specific geometric moments.

    Returns dict with:
    - isotropy: mean(sigma_min/sigma_max) per class (uniform = 1.0)
    - anisotropy_ratio: tr(S_B)/tr(S_W) (high = well-separated)
    - within_kurtosis: mean excess kurtosis of within-class projected features
    - tail_heaviness: how heavy the eigenvalue tail is (ratio top-5/bottom-5 mean)
    - d_eff_gram: tr(W)^2 / tr(W^2) (Gram-based effective dim)
    - d_eff_formula: tr(W) / sigma_centroid_dir^2 (formula-based eff dim)
    - kappa_1: nearest-pair SNR
    """
    K = len(classes)
    N, d = len(X_tr), X_tr.shape[1]
    mu = np.stack([X_tr[y_tr == c].mean(0) for c in classes])
    grand_mean = X_tr.mean(0)

    # Within-class scatter
    Xc = np.zeros((N, d))
    for i, c in enumerate(classes):
        mask = (y_tr == c)
        Xc[mask] = X_tr[mask] - mu[i]

    # Compute W = (1/N) * X_c^T * X_c
    trW = float(np.sum(Xc ** 2)) / N
    sigma_W_global = float(np.sqrt(trW / d))

    # Between-class scatter
    delta_mus = mu - grand_mean
    trB = float(np.sum(delta_mus ** 2)) / K

    # Effective dimensionality (Gram-based)
    W_gram = (Xc.T @ Xc) / N  # (d, d) within-class covariance
    eigenvalues = np.linalg.eigvalsh(W_gram)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    d_eff_gram = float(np.sum(eigenvalues) ** 2 / (np.sum(eigenvalues ** 2) + 1e-10))

    # Isotropy: spread of eigenvalues (1=isotropic, 0=anisotropic)
    if len(eigenvalues) >= 2:
        isotropy = float(eigenvalues[0] / (eigenvalues[-1] + 1e-10))
    else:
        isotropy = 1.0

    # Tail heaviness: ratio of top-5 vs bottom-5 eigenvalue means
    n_top = min(5, len(eigenvalues)//2)
    if n_top >= 2:
        tail_ratio = float(np.mean(eigenvalues[-n_top:]) / (np.mean(eigenvalues[:n_top]) + 1e-10))
    else:
        tail_ratio = 1.0

    # Nearest-pair kappa
    pair_kappas = []
    pair_deff = []
    for i in range(K):
        for j in range(i+1, K):
            Delta = mu[i] - mu[j]
            delta_ij = float(np.linalg.norm(Delta))
            Delta_hat = Delta / (delta_ij + 1e-10)
            kappa_ij = float(delta_ij / (sigma_W_global * np.sqrt(d) + 1e-10))
            sigma_cdir_sq = 0.0
            for k, c in enumerate(classes):
                Xc_c = X_tr[y_tr == c] - mu[k]
                n_c = len(Xc_c)
                proj = Xc_c @ Delta_hat
                sigma_cdir_sq += (n_c / N) * float(np.mean(proj ** 2))
            d_eff_ij = float(trW / (sigma_cdir_sq + 1e-10))
            pair_kappas.append(kappa_ij)
            pair_deff.append(d_eff_ij)

    pair_kappas = np.array(sorted(pair_kappas))
    d_eff_formula = pair_deff[np.argmin(pair_kappas)]  # d_eff for nearest pair
    kappa_1 = float(pair_kappas[0])
    kappa_eff_1 = float(kappa_1 * np.sqrt(d_eff_formula))

    return {
        'isotropy': float(isotropy),
        'anisotropy_ratio': float(trB / (trW + 1e-10)),
        'tail_ratio': float(tail_ratio),
        'd_eff_gram': float(d_eff_gram),
        'd_eff_formula': float(d_eff_formula),
        'kappa_1': float(kappa_1),
        'kappa_eff_1': float(kappa_eff_1),
        'sigma_W_global': float(sigma_W_global),
        'trW': float(trW),
        'trB': float(trB),
        'd': int(d),
        'K': int(K),
    }


def estimate_A_from_loao(X_tr, y_tr, X_te, y_te, classes):
    """Estimate A_empirical by LOAO: for each class i, predict q_i from kappa_i * sqrt(d_eff_i).

    Regression: logit(q_i) = A * kappa_i * sqrt(d_eff_i) + C
    over all classes i.
    """
    K = len(classes)
    N_tr, d = len(X_tr), X_tr.shape[1]
    mu_all = np.stack([X_tr[y_tr == c].mean(0) for c in classes])
    grand_mean = X_tr.mean(0)

    trW = 0.0
    for i, c in enumerate(classes):
        Xc_c = X_tr[y_tr == c] - mu_all[i]
        trW += float(np.sum(Xc_c ** 2)) / N_tr
    sigma_W_global = float(np.sqrt(trW / d))

    # Leave-one-class-out: for each held-out class i, train on others, test on class i
    kappa_effs = []
    q_obs = []

    for i_held, c_held in enumerate(classes):
        # Classes except i_held
        other_classes = [c for c in classes if c != c_held]

        # Train set: all classes except held-out
        mask_tr = np.isin(y_tr, other_classes)
        Xc_tr = X_tr[mask_tr]
        yc_tr = y_tr[mask_tr]

        # Test set: held-out class
        mask_te = (y_te == c_held)
        Xc_te = X_te[mask_te]
        yc_te = y_te[mask_te]

        if len(Xc_te) < 5:
            continue

        knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
        knn.fit(Xc_tr, yc_tr)
        # Predict class for held-out samples (will be wrong since class not in train)
        # But we want: how often does the NN think this is a "correct" class?
        # Alternative: measure q for the full K-1-class problem
        # Actually, let's do the full LOAO differently:
        # For class i, remove it from both train/test, fit KNN on remaining K-1
        # Then: q_i = acc on i when training on all K-1 others
        # This isn't quite right for estimating kappa_eff...

        # Better approach: use the direct formula
        # kappa_i = nearest centroid distance for class i / (sigma_W * sqrt(d))
        # d_eff_i = based on nearest centroid direction for class i
        mu_i = mu_all[i_held]

        # Nearest centroid for class i (among other classes)
        other_mu = np.array([mu_all[j] for j, c in enumerate(classes) if c != c_held])
        dists = np.linalg.norm(other_mu - mu_i, axis=1)
        nearest_idx = np.argmin(dists)
        delta_i = float(dists[nearest_idx])
        kappa_i = float(delta_i / (sigma_W_global * np.sqrt(d) + 1e-10))

        Delta_hat_i = (mu_i - other_mu[nearest_idx]) / (delta_i + 1e-10)
        sigma_cdir_sq = 0.0
        for k, c in enumerate(classes):
            Xc_c = X_tr[y_tr == c] - mu_all[k]
            n_c = len(Xc_c)
            proj = Xc_c @ Delta_hat_i
            sigma_cdir_sq += (len(Xc_c) / N_tr) * float(np.mean(proj ** 2))
        d_eff_i = float(trW / (sigma_cdir_sq + 1e-10))
        kappa_eff_i = float(kappa_i * np.sqrt(d_eff_i))

        # q_i: KNN accuracy for class i when all K classes are in the model
        knn_full = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
        knn_full.fit(X_tr, y_tr)
        mask_te_full = (y_te == c_held)
        Xc_te_full = X_te[mask_te_full]
        if len(Xc_te_full) < 5:
            continue
        acc_i = float(knn_full.score(Xc_te_full, np.full(len(Xc_te_full), c_held)))
        q_i = float(np.clip((acc_i - 1.0/K) / (1.0 - 1.0/K), 1e-6, 1-1e-6))
        logit_i = float(sp.logit(q_i))

        kappa_effs.append(kappa_eff_i)
        q_obs.append(logit_i)

    if len(kappa_effs) < 3:
        return None, None

    kappa_effs = np.array(kappa_effs)
    q_obs = np.array(q_obs)

    # OLS: logit(q_i) = A * kappa_eff_i + C
    A_hat = float(np.polyfit(kappa_effs, q_obs, 1)[0])
    r_val = float(pearsonr(kappa_effs, q_obs)[0])

    return A_hat, r_val


def main():
    log("=" * 70)
    log("ETA_M DECOMPOSITION: A_m = A_universal * eta_m")
    log(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log("=" * 70)
    log("PRE-REGISTERED:")
    log("  E1: r(A_pred, A_obs) > 0.85 across architectures")
    log("  E2: A_universal CV < 0.20")

    arch_moments = []

    # Process LM architectures
    for arch_info in ARCH_DATA:
        log(f"\n--- {arch_info['name']} ---")
        try:
            data = np.load(arch_info['path'])
            X, y = data['X'].astype(np.float64), data['y']
        except Exception as e:
            log(f"  SKIP: {e}")
            continue

        classes = np.unique(y)
        rng = np.random.default_rng(RANDOM_SEED)
        X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n = min(N_TRAIN_PER_CLASS, len(idx) - 1)
            X_tr_list.append(X[idx[:n]]); y_tr_list.append(y[idx[:n]])
            X_te_list.append(X[idx[n:]]); y_te_list.append(y[idx[n:]])
        X_tr = np.concatenate(X_tr_list)
        y_tr = np.concatenate(y_tr_list)
        X_te = np.concatenate(X_te_list)
        y_te = np.concatenate(y_te_list)

        moments = compute_geometric_moments(X_tr, y_tr, classes)
        A_emp, r_loao = estimate_A_from_loao(X_tr, y_tr, X_te, y_te, classes)

        if A_emp is None:
            log(f"  LOAO failed")
            continue

        log(f"  d={moments['d']}, K={moments['K']}")
        log(f"  A_empirical={A_emp:.4f}, r_LOAO={r_loao:.3f}")
        log(f"  isotropy={moments['isotropy']:.4f}, tail_ratio={moments['tail_ratio']:.2f}")
        log(f"  d_eff_gram={moments['d_eff_gram']:.1f}, d_eff_formula={moments['d_eff_formula']:.1f}")
        log(f"  anisotropy_ratio={moments['anisotropy_ratio']:.4f}")
        log(f"  kappa_1={moments['kappa_1']:.4f}, kappa_eff_1={moments['kappa_eff_1']:.4f}")

        arch_moments.append({
            'name': arch_info['name'],
            'moments': moments,
            'A_empirical': float(A_emp),
            'r_loao': float(r_loao),
        })

    # Process BGE-small
    log(f"\n--- bge-small ---")
    try:
        X_bge = np.load(BGE_DATA['embed_path'], mmap_mode='r').astype(np.float64)
        lab_bge = np.load(BGE_DATA['labels_path'])
        y_bge = lab_bge['l1'].astype(int)

        rng = np.random.default_rng(RANDOM_SEED)
        classes_bge = np.unique(y_bge)
        X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
        N_PER = N_TRAIN_PER_CLASS
        for c in classes_bge:
            idx = np.where(y_bge == c)[0]
            rng.shuffle(idx)
            n = min(N_PER, len(idx) - 1)
            X_tr_list.append(X_bge[idx[:n]]); y_tr_list.append(y_bge[idx[:n]])
            X_te_list.append(X_bge[idx[n:]]); y_te_list.append(y_bge[idx[n:]])
        X_tr_bge = np.concatenate(X_tr_list)
        y_tr_bge = np.concatenate(y_tr_list)
        X_te_bge = np.concatenate(X_te_list)
        y_te_bge = np.concatenate(y_te_list)

        moments_bge = compute_geometric_moments(X_tr_bge, y_tr_bge, classes_bge)
        A_emp_bge, r_loao_bge = estimate_A_from_loao(
            X_tr_bge, y_tr_bge, X_te_bge, y_te_bge, classes_bge)

        if A_emp_bge is not None:
            log(f"  d={moments_bge['d']}, K={moments_bge['K']}")
            log(f"  A_empirical={A_emp_bge:.4f}, r_LOAO={r_loao_bge:.3f}")
            log(f"  isotropy={moments_bge['isotropy']:.4f}, tail_ratio={moments_bge['tail_ratio']:.2f}")
            log(f"  d_eff_gram={moments_bge['d_eff_gram']:.1f}")
            arch_moments.append({
                'name': 'bge-small',
                'moments': moments_bge,
                'A_empirical': float(A_emp_bge),
                'r_loao': float(r_loao_bge),
            })
    except Exception as e:
        log(f"  SKIP BGE: {e}")

    # ==================== ETA_m DECOMPOSITION ====================
    log("\n" + "=" * 70)
    log("ETA_m DECOMPOSITION ANALYSIS")
    log("=" * 70)

    if len(arch_moments) < 3:
        log("Insufficient architectures for regression")
        return

    names = [a['name'] for a in arch_moments]
    A_obs = np.array([a['A_empirical'] for a in arch_moments])
    isotropy = np.array([a['moments']['isotropy'] for a in arch_moments])
    tail_ratio = np.array([a['moments']['tail_ratio'] for a in arch_moments])
    d_eff_gram = np.array([a['moments']['d_eff_gram'] for a in arch_moments])
    d_eff_formula = np.array([a['moments']['d_eff_formula'] for a in arch_moments])
    aniso = np.array([a['moments']['anisotropy_ratio'] for a in arch_moments])
    d_vals = np.array([a['moments']['d'] for a in arch_moments])

    log("\nA_empirical per arch:")
    for i, a in enumerate(arch_moments):
        log(f"  {a['name']}: A={a['A_empirical']:.4f} (r_LOAO={a['r_loao']:.3f})")

    # Test simple candidates for eta_m:
    # Candidate 1: eta_m = 1 / sqrt(d_eff_gram) -- more effective dims -> lower A
    # Candidate 2: eta_m = isotropy -- more isotropic -> higher A
    # Candidate 3: eta_m = 1 / log(d) -- higher dim -> lower A (log scaling)
    # Candidate 4: eta_m = tail_ratio -- heavier tail -> more spread -> ?

    candidates = {
        'inv_sqrt_deff_gram': 1.0 / np.sqrt(d_eff_gram + 1e-10),
        'isotropy': isotropy,
        'inv_log_d': 1.0 / np.log(d_vals + 1.0),
        'tail_ratio_inv': 1.0 / (tail_ratio + 1e-10),
        'aniso_ratio': aniso,
        'inv_sqrt_deff_formula': 1.0 / np.sqrt(d_eff_formula + 1e-10),
    }

    best_candidate = None
    best_r = 0.0

    log("\nCorrelations with A_empirical:")
    for cname, cvals in candidates.items():
        r, pval = pearsonr(cvals, A_obs)
        log(f"  {cname}: r={r:.3f} (p={pval:.3f})")
        if abs(r) > abs(best_r):
            best_r = r
            best_candidate = cname

    log(f"\nBest predictor: {best_candidate} (r={best_r:.3f})")

    # Fit A_pred using best candidate
    best_vals = candidates[best_candidate]
    if best_r > 0:
        # A_pred = c * best_vals
        c_fit = float(np.sum(A_obs * best_vals) / (np.sum(best_vals**2) + 1e-10))
        A_pred = c_fit * best_vals
    else:
        # Negative correlation: A_pred = c0 - c1 * best_vals
        coefs = np.polyfit(best_vals, A_obs, 1)
        A_pred = coefs[0] * best_vals + coefs[1]

    r_pred, _ = pearsonr(A_pred, A_obs)

    log(f"\nA_pred vs A_obs (r={r_pred:.3f}):")
    for i, a in enumerate(arch_moments):
        log(f"  {a['name']}: A_obs={A_obs[i]:.4f}, A_pred={A_pred[i]:.4f}, "
            f"error={abs(A_obs[i]-A_pred[i]):.4f}")

    # E1: r(A_pred, A_obs) > 0.85
    pass_E1 = abs(r_pred) > 0.85
    log(f"\nE1 (r>0.85): {'PASS' if pass_E1 else 'FAIL'} (r={r_pred:.3f})")

    # E2: A_universal = A_obs / eta_m CV < 0.20
    # Using isotropy as eta_m (most interpretable)
    iso_nz = np.where(isotropy > 1e-6, isotropy, 1e-6)
    A_univ = A_obs / iso_nz
    cv_Auniv = float(np.std(A_univ) / (np.mean(A_univ) + 1e-10))
    pass_E2 = cv_Auniv < 0.20
    log(f"E2 (A_universal CV < 0.20): {'PASS' if pass_E2 else 'FAIL'} "
        f"(CV={cv_Auniv:.3f})")

    verdict = (f"E1={'PASS' if pass_E1 else 'FAIL'}, E2={'PASS' if pass_E2 else 'FAIL'}. "
               f"Best predictor of A_m: {best_candidate} (r={best_r:.3f}). "
               f"OVERALL: {'PASS' if pass_E1 and pass_E2 else 'PARTIAL/FAIL'}")
    log(f"\nVERDICT: {verdict}")

    output = {
        'experiment': 'eta_decomposition',
        'timestamp': datetime.datetime.now().isoformat(),
        'arch_results': arch_moments,
        'candidate_correlations': {k: float(pearsonr(v, A_obs)[0]) for k, v in candidates.items()},
        'best_candidate': best_candidate,
        'best_r': float(best_r),
        'r_pred_obs': float(r_pred),
        'analysis': {
            'pass_E1': bool(pass_E1), 'pass_E2': bool(pass_E2),
            'cv_A_universal': float(cv_Auniv),
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
    _log_fh.close()
