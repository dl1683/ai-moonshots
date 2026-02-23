#!/usr/bin/env python -u
"""
CTI LAW: Stringer 2018b Mouse V1 Biological Validation
=======================================================
Tests CTI universal law on mouse V1 calcium imaging data:
    logit(q_norm) = A * kappa_nearest * sqrt(d_eff) + C

Dataset: Stringer/Pachitariu et al. 2018b
  - 10,079 V1 neurons (calcium imaging, mouse)
  - 2,800 natural images, ~2 presentations each
  - K=11 semantic object categories (birds, cats, flowers, ...)

This is a much stronger test than Cadieu2014 (K=7) due to:
  - K=11 classes (better law test)
  - Higher-power per-image analysis (n=5880 presentations)
  - Simultaneous recording from >10K neurons

Note: Mouse V1 is NOT an object-selective area like macaque IT.
Expected: weaker law (lower r) than IT, but direction should hold.
Pre-registered under bddec1d (NEUROSCIENCE_DATASETS.md).
"""

import json, os, time
import numpy as np
from scipy import stats
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR = "data/stringer2018b"
SESSION_FILE = f"{DATA_DIR}/natimg2800_M170604_MP031_2017-06-28.mat"
OUT_FILE = "results/cti_stringer2018b_validation.json"

SQRT_4_OVER_PI = np.sqrt(4.0 / np.pi)
PASS_R_THRESHOLD = 0.50
PASS_MAE_THRESH  = 0.10

print("=" * 60)
print("CTI Biological Validation: Stringer 2018b Mouse V1")
print(f"sqrt(4/pi) = {SQRT_4_OVER_PI:.4f}")
print("=" * 60)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\nLoading neural data...", flush=True)
t0 = time.time()
mat = sio.loadmat(SESSION_FILE)
stim = mat['stim'][0][0]
resp  = stim['resp'].astype(np.float32)     # (5880, 10079)
istim = stim['istim'].flatten().astype(int) # (5880,) stimulus indices 1..2800+

print(f"  resp: {resp.shape}  [{time.time()-t0:.1f}s]", flush=True)
print(f"  istim range: [{istim.min()}, {istim.max()}]")

# Load category assignments
cat_mat = sio.loadmat(f"{DATA_DIR}/stimuli_class_assignment_confident.mat")
class_assignment = cat_mat['class_assignment'].flatten().astype(int)  # (2800,) 0-indexed
class_names = [cat_mat['class_names'][0, i][0] for i in range(cat_mat['class_names'].shape[1])]
print(f"  K categories: {len(class_names)}: {class_names}")

# Map presentations to categories
# istim is 1-indexed into 2800 images; values > 2800 are spontaneous/other
# class_assignment[istim-1] gives category for each presentation
valid_istim_mask = (istim >= 1) & (istim <= len(class_assignment))
pres_cats = np.zeros(len(istim), dtype=int)
pres_cats[valid_istim_mask] = class_assignment[istim[valid_istim_mask] - 1]
print(f"  Out-of-range istim (spontaneous): {(~valid_istim_mask).sum()}")

# Filter out unknown (class 0) presentations
known_mask = pres_cats > 0
resp_known = resp[known_mask]       # (n_known, 10079)
cats_known = pres_cats[known_mask]  # (n_known,)
print(f"  Known presentations (excl unknown): {resp_known.shape[0]}")
print(f"  Class distribution: {dict(zip(*np.unique(cats_known, return_counts=True)))}")

K = len(np.unique(cats_known))
d = resp_known.shape[1]
print(f"  K={K}, d={d}")


# ================================================================
# 2. EFFICIENT CTI COMPUTATION FOR HIGH-DIM NEURAL DATA
# ================================================================
def compute_cti_inputs_hd(responses, labels):
    """
    Efficient CTI computation for high-dimensional (d=10K) neural data.
    Uses diagonal covariance approximation for sigma_W_global.
    Only computes sigma_cdir (directional variance) by projecting onto centroid direction.
    """
    classes = np.unique(labels)
    K = len(classes)
    d = responses.shape[1]
    n = responses.shape[0]

    print(f"  Computing centroids for {K} classes...", flush=True)
    centroids = np.array([responses[labels == c].mean(axis=0) for c in classes], dtype=np.float32)

    print(f"  Computing within-class variance (diagonal approx)...", flush=True)
    # tr(Sigma_W) = sum over neurons of pooled within-class variance
    # = sum over neurons of (1/K * sum_k var(neuron | class k))
    # This is O(n * d) and does NOT require forming full covariance matrix
    per_neuron_var = np.zeros(d, dtype=np.float64)
    for c in classes:
        mask = labels == c
        X_c = responses[mask].astype(np.float64)
        per_neuron_var += np.var(X_c, axis=0)
    per_neuron_var /= K
    tr_W = float(per_neuron_var.sum())
    sigma_W_global = np.sqrt(tr_W / d)
    print(f"  tr(Sigma_W)={tr_W:.2f}, sigma_W_global={sigma_W_global:.4f}", flush=True)

    # Pairwise centroid distances
    centroid_dists = np.full((K, K), np.inf)
    for i in range(K):
        for j in range(K):
            if i != j:
                centroid_dists[i, j] = float(np.linalg.norm(centroids[i] - centroids[j]))
    nearest_j = np.argmin(centroid_dists, axis=1)

    kappas, d_effs = [], []
    for i, c in enumerate(classes):
        j = nearest_j[i]
        delta = centroids[i].astype(np.float64) - centroids[j].astype(np.float64)
        delta_min = float(np.linalg.norm(delta))
        delta_norm = delta / (delta_min + 1e-12)

        kappa_i = delta_min / (sigma_W_global * np.sqrt(d))

        # sigma_cdir: project per-neuron variance onto centroid direction
        # sigma_cdir^2 = delta_norm @ diag(Sigma_W) @ delta_norm = (delta_norm^2) * per_neuron_var
        sigma_cdir_sq = float(np.dot(delta_norm ** 2, per_neuron_var))
        sigma_cdir = np.sqrt(max(sigma_cdir_sq, 1e-10))
        d_eff_i = tr_W / (sigma_cdir ** 2)

        kappas.append(kappa_i)
        d_effs.append(d_eff_i)

    return {
        "classes": list(classes),
        "K": K,
        "d": d,
        "n": n,
        "kappa_nearest": np.array(kappas),
        "d_eff": np.array(d_effs),
        "sigma_W_global": float(sigma_W_global),
        "tr_W": float(tr_W),
        "per_neuron_var": per_neuron_var,  # keep for diagnostics
    }


def compute_q_1nn(responses, labels):
    """1-NN accuracy per class using leave-one-out."""
    classes = np.unique(labels)
    K = len(classes)
    n = responses.shape[0]

    print(f"  Running LOO 1-NN (n={n}, d={responses.shape[1]})...", flush=True)
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1,
                                algorithm='brute')
    q_1nns = []
    q_norms = []
    logit_q_norms = []

    # Predict all at once using full leave-one-out
    pred = np.empty(n, dtype=int)
    for test_idx in range(n):
        if test_idx % 500 == 0:
            print(f"    {test_idx}/{n} [{time.time()-t0:.0f}s]", flush=True)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        knn.fit(responses[train_mask].astype(np.float32),
                labels[train_mask].astype(np.int32))
        pred[test_idx] = knn.predict(responses[[test_idx]].astype(np.float32))[0]

    print(f"    Done in {time.time()-t0:.1f}s", flush=True)

    for c in classes:
        mask_c = labels == c
        q_i = float(np.mean(pred[mask_c] == c))
        # Clip to [1/K, 1-eps] before logit
        q_i_clipped = np.clip(q_i, 1.0/K + 1e-4, 1.0 - 1e-4)
        q_norm_i = (q_i_clipped - 1.0/K) / (1.0 - 1.0/K)
        q_norm_i = float(np.clip(q_norm_i, 1e-4, 1 - 1e-4))
        q_1nns.append(q_i)
        q_norms.append(q_norm_i)
        logit_q_norms.append(float(np.log(q_norm_i / (1.0 - q_norm_i))))

    return np.array(q_1nns), np.array(q_norms), np.array(logit_q_norms)


def fit_law(kappa, d_eff, logit_q):
    K = len(kappa)
    x_pred = kappa * np.sqrt(d_eff)
    y = logit_q

    if np.std(x_pred) < 1e-10 or np.std(y) < 1e-10:
        return {}

    r, pval = stats.pearsonr(x_pred, y)
    r_kappa, _ = stats.pearsonr(kappa, y)
    X = np.column_stack([x_pred, np.ones(K)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A, C = float(coef[0]), float(coef[1])
    A_renorm = A / np.sqrt(np.mean(d_eff))

    # LOCO MAE
    q_1nn_vals = np.array([1.0/(1.0 + np.exp(-li)) * (1 - 1.0/K) + 1.0/K for li in logit_q])
    mae_vals = []
    for i in range(K):
        mask = np.ones(K, dtype=bool); mask[i] = False
        X_tr = np.column_stack([x_pred[mask], np.ones(mask.sum())])
        c_lo, _, _, _ = np.linalg.lstsq(X_tr, y[mask], rcond=None)
        lp = x_pred[i] * c_lo[0] + c_lo[1]
        qp = 1.0/(1+np.exp(-lp)) * (1 - 1.0/K) + 1.0/K
        mae_vals.append(abs(qp - q_1nn_vals[i]))

    return {
        "r": float(r), "pval": float(pval),
        "r_kappa_only": float(r_kappa),
        "A": A, "C": C,
        "A_renorm": float(A_renorm),
        "mean_d_eff": float(np.mean(d_eff)),
        "mae_loco": float(np.mean(mae_vals)),
    }


def per_image_analysis(responses, labels):
    """Per-image local margin analysis (n=5880 data points)."""
    classes = np.unique(labels)
    d = responses.shape[1]
    n = responses.shape[0]

    centroids = np.array([responses[labels == c].mean(axis=0) for c in classes], dtype=np.float32)
    per_neuron_var = np.zeros(d, dtype=np.float64)
    for c in classes:
        per_neuron_var += np.var(responses[labels == c].astype(np.float64), axis=0)
    per_neuron_var /= len(classes)
    sigma_W_global = float(np.sqrt(per_neuron_var.sum() / d))

    # For each image: distance to own centroid and nearest other centroid
    label_idx = np.array([np.where(classes == c)[0][0] for c in labels])

    print("  Computing per-image margins (vectorized)...", flush=True)
    # Distance to all centroids: (n, K)
    # Do in batches to avoid memory issues with n=5880, d=10079
    batch = 500
    dists = np.zeros((n, len(classes)), dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        batch_resp = responses[start:end].astype(np.float32)  # (batch, d)
        # (batch, K): distance from each response to each centroid
        d_batch = np.sqrt(np.sum(
            (batch_resp[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=-1))  # (batch, K)
        dists[start:end] = d_batch

    dist_own = dists[np.arange(n), label_idx]
    dists_other = dists.copy()
    dists_other[np.arange(n), label_idx] = np.inf
    dist_nearest_other = dists_other.min(axis=1)
    margin = (dist_nearest_other - dist_own) / (sigma_W_global * np.sqrt(d) + 1e-10)

    # 1-NN LOO accuracy per image (approximate: use centroid classifier instead of full LOO for speed)
    # Centroid classifier: assign to nearest centroid
    centroid_pred = np.argmin(dists, axis=1)  # predicted class index
    correct = (centroid_pred == label_idx).astype(float)

    r_margin, pval_margin = stats.pearsonr(margin, correct)
    mean_acc = float(correct.mean())

    return {
        "n": n,
        "r_margin_correct": float(r_margin),
        "pval_margin_correct": float(pval_margin),
        "mean_accuracy_centroid": mean_acc,
        "margin_correct_mean": float(margin[correct == 1].mean()),
        "margin_wrong_mean": float(margin[correct == 0].mean()),
        "note": "Centroid classifier (not LOO 1-NN) for tractability with d=10079"
    }


# ================================================================
# 3. RUN ANALYSIS
# ================================================================
print("\n--- Computing CTI geometry ---", flush=True)
geo = compute_cti_inputs_hd(resp_known, cats_known)
kappa = geo["kappa_nearest"]
d_eff = geo["d_eff"]
classes = geo["classes"]
class_name_list = [class_names[c] for c in classes]

print(f"\nKappa_nearest: {kappa}")
print(f"d_eff:         {d_eff}")
print(f"Classes:       {class_name_list}")

print("\n--- PCA reduction (used for BOTH kappa and 1-NN) ---", flush=True)
from sklearn.decomposition import PCA
# IMPORTANT: must compute kappa and 1-NN in SAME space
N_PCA = 100
pca = PCA(n_components=N_PCA, random_state=42)
resp_pca = pca.fit_transform(resp_known.astype(np.float64))
var_explained = float(pca.explained_variance_ratio_.sum())
print(f"  Top {N_PCA} PCs explain {var_explained:.3f} of variance", flush=True)

# Recompute CTI geometry in PCA space (so kappa and q are in the same space)
print("\n--- Re-computing CTI geometry in PCA space ---", flush=True)
geo_pca = compute_cti_inputs_hd(resp_pca.astype(np.float32), cats_known)
kappa = geo_pca["kappa_nearest"]
d_eff = geo_pca["d_eff"]

print(f"\nKappa_nearest (PCA): {kappa}")
print(f"d_eff (PCA):         {d_eff}")

print("\n--- Computing 1-NN accuracy ---", flush=True)
q_1nn, q_norm, logit_q = compute_q_1nn(resp_pca, cats_known)

print(f"\nq_1nn:        {q_1nn}")
print(f"logit_q_norm: {logit_q}")

print("\n--- Fitting CTI law ---", flush=True)
fit = fit_law(kappa, d_eff, logit_q)
print(f"\n  r(kappa*sqrt(d_eff), logit_q) = {fit['r']:.4f}  (threshold: {PASS_R_THRESHOLD})")
print(f"  r(kappa_nearest, logit_q)     = {fit['r_kappa_only']:.4f}")
print(f"  A = {fit['A']:.4f}, C = {fit['C']:.4f}")
print(f"  A_renorm = {fit['A_renorm']:.4f} (NLP: 1.129, sqrt(4/pi): {SQRT_4_OVER_PI:.4f})")
print(f"  mean_d_eff = {fit['mean_d_eff']:.2f}")
print(f"  LOCO MAE = {fit['mae_loco']:.4f}  (threshold: {PASS_MAE_THRESH})")

h1 = fit['r'] > PASS_R_THRESHOLD
h3 = fit['mae_loco'] < PASS_MAE_THRESH
h2 = 0.70 <= fit['A_renorm'] <= 1.80
h4 = h1 and h2
print(f"\n  H1 (r>{PASS_R_THRESHOLD}): {'PASS' if h1 else 'FAIL'}")
print(f"  H2 (A_renorm in [0.7,1.8]): {'PASS' if h2 else 'FAIL'}")
print(f"  H3 (MAE<{PASS_MAE_THRESH}): {'PASS' if h3 else 'FAIL'}")
print(f"  H4 (substrate-independent): {'PASS' if h4 else 'FAIL'}")

print("\n--- Per-image analysis (centroid classifier, PCA space) ---", flush=True)
pimg = per_image_analysis(resp_pca.astype(np.float32), cats_known)
print(f"  r(margin, correct) = {pimg['r_margin_correct']:.4f} (p={pimg['pval_margin_correct']:.4g})")
print(f"  Centroid accuracy = {pimg['mean_accuracy_centroid']:.3f}")
print(f"  margin_correct={pimg['margin_correct_mean']:.4f} vs margin_wrong={pimg['margin_wrong_mean']:.4f}")

# ================================================================
# 4. SAVE
# ================================================================
output = {
    "experiment": "CTI_stringer2018b_mouse_V1",
    "preregistration_commit": "bddec1d",
    "session": "natimg2800_M170604_MP031_2017-06-28",
    "K": int(K), "d": int(d), "n_presentations": int(resp_known.shape[0]),
    "classes": class_name_list,
    "kappa_nearest": kappa.tolist(),
    "d_eff": d_eff.tolist(),
    "q_1nn": q_1nn.tolist(),
    "q_norm": q_norm.tolist(),
    "logit_q_norm": logit_q.tolist(),
    "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
    "fit": fit,
    "hypotheses": {
        "H1_r_pass": bool(h1), "H2_renorm_pass": bool(h2),
        "H3_mae_pass": bool(h3), "H4_substrate_independent": bool(h4),
    },
    "per_image_exploratory": pimg,
    "cadieu2014_comparison": {
        "IT_r": 0.1856, "IT_A_renorm": 0.0692, "IT_H3": True,
        "IT_per_image_r": 0.4136,
        "note": "Cadieu2014 macaque IT: different substrate, K=7"
    },
    "artificial_reference": {
        "NLP_A_renorm": 1.129, "NLP_r_LOAO": 0.977,
        "universal_constant_sqrt4pi": float(SQRT_4_OVER_PI),
    }
}
os.makedirs("results", exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUT_FILE}")
