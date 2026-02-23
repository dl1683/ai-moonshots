#!/usr/bin/env python -u
"""
CTI LAW: BIOLOGICAL NEURAL VALIDATION — Cadieu2014 Macaque IT + V4
===================================================================
Tests whether the CTI universal law:
    logit(q_norm) = A * kappa_nearest * sqrt(d_eff) + C

holds for biological macaque neural recordings, and whether the
renormalized constant A_renorm = A / sqrt(d_eff) ≈ sqrt(4/pi) ≈ 1.128
matches the artificial neural network value.

Dataset: Cadieu et al. 2014, PLOS Comp Bio
  - 168 IT multi-unit sites, macaque, K=7 object categories
  - 64 V4 multi-unit sites, macaque (comparison)
  - 280 images per category (1960 total)
  - features: mean normalized firing rates

Pre-registered (commit: bddec1d):
  H1: r(kappa_nearest, logit_q_1nn) > 0.50
  H2: A_renorm_bio = A_fit/sqrt(d_eff) in [0.70, 1.80] around 1.128
  H3: MAE < 0.10 on LOCO prediction
  H4: H1 AND H2 → substrate-independent law

Note: This uses Cadieu2014 instead of MajajHong2015 (access issue) -
same macaque IT cortex recordings, same type, K=7 instead of K=8.
"""

import json, os, sys, time
import numpy as np
from scipy import stats
from scipy.special import logit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import scipy.io as sio

PREREQ_COMMIT = "bddec1d"  # pre-registration commit
OUT_FILE = "results/cti_neuroscience_cadieu2014.json"

PASS_R_THRESHOLD = 0.50
PASS_RENORM_LOW  = 0.70
PASS_RENORM_HIGH = 1.80
PASS_MAE_THRESH  = 0.10
SQRT_4_OVER_PI   = np.sqrt(4.0 / np.pi)  # = 1.1284

print("=" * 60)
print("CTI Biological Validation: Cadieu2014 Macaque IT + V4")
print(f"Pre-registered commit: {PREREQ_COMMIT}")
print(f"sqrt(4/pi) = {SQRT_4_OVER_PI:.4f} (universal constant prediction)")
print("=" * 60)


# ================================================================
# 1. LOAD DATA
# ================================================================
DATA_DIR = "data/cadieu2014"

def load_region(region, cell_type="multiunits"):
    fname = f"{DATA_DIR}/NeuralData_{region}_{cell_type}.mat"
    mat = sio.loadmat(fname)
    features = mat["features"].astype(np.float64)  # (n_images, n_sites)
    meta = mat["meta"]
    # Parse: 'images/hash.png Category ...'
    categories = np.array([m.split()[1] if len(m.split()) >= 2 else "UNKNOWN" for m in meta])
    return features, categories

print("\nLoading IT multiunits...", flush=True)
it_features, it_cats = load_region("IT")
print(f"  IT features: {it_features.shape}, K={len(np.unique(it_cats))}")

print("Loading V4 multiunits...", flush=True)
v4_features, v4_cats = load_region("V4")
print(f"  V4 features: {v4_features.shape}, K={len(np.unique(v4_cats))}")


# ================================================================
# 2. CTI LAW COMPUTATION
# ================================================================
def compute_cti_inputs(responses, labels):
    """
    responses: (N_images, N_sites) — mean firing rates
    labels: (N_images,) — category labels
    Returns per-class kappa_nearest, d_eff, q_1nn
    """
    classes = np.unique(labels)
    K = len(classes)
    d = responses.shape[1]

    # Class centroids
    centroids = np.array([responses[labels == c].mean(axis=0) for c in classes])

    # Pooled within-class covariance (average across classes)
    W_list = []
    for c in classes:
        X_c = responses[labels == c]
        W_list.append(np.cov(X_c.T))  # (d, d)
    Sigma_W = np.mean(W_list, axis=0)
    tr_W = np.trace(Sigma_W)
    sigma_W_global = np.sqrt(tr_W / d)  # sqrt(mean per-dimension within-class variance)

    # Nearest-class pairwise distances
    centroid_dists = np.full((K, K), np.inf)
    for i in range(K):
        for j in range(K):
            if i != j:
                centroid_dists[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    nearest_j = np.argmin(centroid_dists, axis=1)

    kappas, d_effs, q_1nns, q_norms, logit_q_norms = [], [], [], [], []
    for i, c in enumerate(classes):
        j = nearest_j[i]
        delta = centroids[i] - centroids[j]
        delta_norm = delta / np.linalg.norm(delta)
        delta_min = np.linalg.norm(delta)

        # kappa_nearest: gap to nearest class in units of global within-class std
        kappa_i = delta_min / (sigma_W_global * np.sqrt(d))

        # d_eff: anisotropy ratio
        sigma_cdir = float(np.sqrt(delta_norm @ Sigma_W @ delta_norm))
        if sigma_cdir < 1e-10:
            sigma_cdir = 1e-10
        d_eff_i = tr_W / (sigma_cdir ** 2)

        kappas.append(kappa_i)
        d_effs.append(d_eff_i)

    # 1-NN accuracy per class using leave-one-out
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    n = responses.shape[0]
    pred = np.empty(n, dtype=object)
    for test_idx in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        knn.fit(responses[train_mask], labels[train_mask])
        pred[test_idx] = knn.predict(responses[[test_idx]])[0]

    for i, c in enumerate(classes):
        mask_c = labels == c
        q_i = np.mean(pred[mask_c] == c)
        # Clip q to [1/K, 1.0] — sub-chance accuracy is numerical noise / small-K effect
        q_i_clipped = np.clip(q_i, 1.0/K + 1e-4, 1.0 - 1e-4)
        q_1nns.append(float(q_i))  # Store raw q
        # Normalize: q_norm = (q - 1/K) / (1 - 1/K)
        q_norm_i = (q_i_clipped - 1.0/K) / (1.0 - 1.0/K)
        q_norm_i = np.clip(q_norm_i, 1e-4, 1 - 1e-4)
        q_norms.append(float(q_norm_i))
        logit_q_norms.append(float(np.log(q_norm_i / (1.0 - q_norm_i))))

    return {
        "classes": list(classes),
        "K": K,
        "d": d,
        "kappa_nearest": np.array(kappas),
        "d_eff": np.array(d_effs),
        "q_1nn": np.array(q_1nns),
        "q_norm": np.array(q_norms),
        "logit_q_norm": np.array(logit_q_norms),
        "sigma_W_global": float(sigma_W_global),
        "tr_W": float(tr_W),
    }


# ================================================================
# 3. FITTING AND TEST
# ================================================================
def fit_cti_law(res):
    """Fit logit(q_norm) = A * kappa_nearest * sqrt(d_eff) + C"""
    K = res["K"]
    x_kappa = res["kappa_nearest"]
    x_deff = res["d_eff"]
    y = res["logit_q_norm"]

    # Form predictor
    x_pred = x_kappa * np.sqrt(x_deff)

    # OLS fit
    if np.std(x_pred) < 1e-10 or np.std(y) < 1e-10:
        return {"r": np.nan, "A": np.nan, "C": np.nan, "A_renorm": np.nan}

    r, pval = stats.pearsonr(x_pred, y)
    # Fit A, C
    X = np.column_stack([x_pred, np.ones(len(y))])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A, C = coef[0], coef[1]

    # Renormalized constant: A_renorm = A / sqrt(mean_d_eff)
    # Since d_eff is roughly class-invariant within a region
    A_renorm = A / np.sqrt(np.mean(x_deff))

    # Fit using kappa_nearest only (simplest form)
    r_kappa, pval_kappa = stats.pearsonr(x_kappa, y)
    X_k = np.column_stack([x_kappa, np.ones(len(y))])
    coef_k, _, _, _ = np.linalg.lstsq(X_k, y, rcond=None)
    A_k, C_k = coef_k[0], coef_k[1]

    # LOCO MAE: leave one class out
    q_pred_loco = np.empty_like(y)
    for i in range(K):
        train_mask = np.ones(K, dtype=bool)
        train_mask[i] = False
        if np.sum(train_mask) < 2:
            q_pred_loco[i] = np.nan
            continue
        X_tr = np.column_stack([x_pred[train_mask], np.ones(train_mask.sum())])
        y_tr = y[train_mask]
        c_loco, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        logit_pred = x_pred[i] * c_loco[0] + c_loco[1]
        # Convert back to q
        q_pred_norm = 1.0 / (1.0 + np.exp(-logit_pred))
        q_pred = q_pred_norm * (1.0 - 1.0/K) + 1.0/K
        q_pred_loco[i] = q_pred

    mae_loco = np.nanmean(np.abs(q_pred_loco - res["q_1nn"]))

    return {
        "r": float(r),
        "pval": float(pval),
        "A": float(A),
        "C": float(C),
        "A_renorm": float(A_renorm),
        "r_kappa_only": float(r_kappa),
        "A_kappa_only": float(A_k),
        "C_kappa_only": float(C_k),
        "mae_loco": float(mae_loco),
        "mean_d_eff": float(np.mean(x_deff)),
        "std_d_eff": float(np.std(x_deff)),
    }


def run_region(name, features, labels):
    print(f"\n{'-'*50}")
    print(f"Region: {name}", flush=True)
    t0 = time.time()
    res = compute_cti_inputs(features, labels)
    elapsed = time.time() - t0
    print(f"  Computed in {elapsed:.1f}s")
    print(f"  K={res['K']}, d={res['d']}")
    print(f"  kappa_nearest: {res['kappa_nearest']}")
    print(f"  d_eff:         {res['d_eff']}")
    print(f"  q_1nn:         {res['q_1nn']}")
    print(f"  logit_q_norm:  {res['logit_q_norm']}")

    fit = fit_cti_law(res)
    print(f"\n  CTI Law Fit:")
    print(f"  r(kappa*sqrt(d_eff), logit_q) = {fit['r']:.4f}  (threshold: {PASS_R_THRESHOLD})")
    print(f"  r(kappa_nearest, logit_q)     = {fit['r_kappa_only']:.4f}")
    print(f"  A = {fit['A']:.4f}, C = {fit['C']:.4f}")
    print(f"  A_renorm = A/sqrt(mean_d_eff) = {fit['A_renorm']:.4f}")
    print(f"  sqrt(4/pi) prediction         = {SQRT_4_OVER_PI:.4f}")
    print(f"  mean_d_eff = {fit['mean_d_eff']:.2f}")
    print(f"  LOCO MAE  = {fit['mae_loco']:.4f}  (threshold: {PASS_MAE_THRESH})")

    # Hypothesis tests
    h1_pass = fit['r'] > PASS_R_THRESHOLD
    h2_pass = PASS_RENORM_LOW <= fit['A_renorm'] <= PASS_RENORM_HIGH
    h3_pass = fit['mae_loco'] < PASS_MAE_THRESH
    h4_pass = h1_pass and h2_pass

    print(f"\n  H1 (r>{PASS_R_THRESHOLD}): {'PASS' if h1_pass else 'FAIL'}")
    print(f"  H2 (A_renorm in [{PASS_RENORM_LOW},{PASS_RENORM_HIGH}]): {'PASS' if h2_pass else 'FAIL'} (A_renorm={fit['A_renorm']:.4f})")
    print(f"  H3 (MAE<{PASS_MAE_THRESH}): {'PASS' if h3_pass else 'FAIL'} (MAE={fit['mae_loco']:.4f})")
    print(f"  H4 (substrate-independent, H1+H2): {'PASS' if h4_pass else 'FAIL'}")

    return {
        "region": name,
        "K": int(res['K']),
        "d": int(res['d']),
        "n_images": int(len(labels)),
        "categories": res['classes'],
        "kappa_nearest": res['kappa_nearest'].tolist(),
        "d_eff": res['d_eff'].tolist(),
        "q_1nn": res['q_1nn'].tolist(),
        "q_norm": res['q_norm'].tolist(),
        "logit_q_norm": res['logit_q_norm'].tolist(),
        "fit": fit,
        "hypotheses": {
            "H1_r_pass": bool(h1_pass),
            "H2_renorm_pass": bool(h2_pass),
            "H3_mae_pass": bool(h3_pass),
            "H4_substrate_independent": bool(h4_pass),
        }
    }


# ================================================================
# 3b. EXPLORATORY: PER-IMAGE LOCAL KAPPA (higher statistical power)
# ================================================================
def per_image_local_kappa(responses, labels):
    """
    For each image, compute 'local kappa': distance to nearest class centroid
    normalized by within-class noise in the direction to that centroid.
    Test: do images with higher local kappa get classified correctly more often?
    n = 1960 data points (vs K=7 class-level)
    """
    classes = np.unique(labels)
    K = len(classes)
    d = responses.shape[1]
    n = responses.shape[0]

    centroids = np.array([responses[labels == c].mean(axis=0) for c in classes])
    W_list = [np.cov(responses[labels == c].T) for c in classes]
    Sigma_W = np.mean(W_list, axis=0)
    tr_W = np.trace(Sigma_W)
    sigma_W_global = np.sqrt(tr_W / d)

    # Per-image: compute distance to true class centroid vs nearest other centroid
    # local_kappa[i] = (dist to own centroid margin) / (sigma_W * sqrt(d))
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    correct = np.zeros(n, dtype=float)
    for test_idx in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        knn.fit(responses[train_mask], labels[train_mask])
        pred_label = knn.predict(responses[[test_idx]])[0]
        correct[test_idx] = float(pred_label == labels[test_idx])

    # Local kappa: distance to own centroid (margin proxy)
    label_idx = np.array([np.where(classes == c)[0][0] for c in labels])
    own_centroid = centroids[label_idx]  # (n, d)
    dist_own = np.linalg.norm(responses - own_centroid, axis=1)  # (n,)

    # Also: distance to nearest OTHER centroid minus distance to own centroid (margin)
    dists_to_all = np.stack([np.linalg.norm(responses - centroids[j], axis=1) for j in range(K)], axis=1)
    # Mask own class distance with inf
    for i, li in enumerate(label_idx):
        dists_to_all[i, li] = np.inf
    dist_nearest_other = dists_to_all.min(axis=1)  # (n,)
    margin = (dist_nearest_other - dist_own) / (sigma_W_global * np.sqrt(d))

    r_margin, pval_margin = stats.pearsonr(margin, correct)
    r_dist_own, _ = stats.pearsonr(-dist_own, correct)  # closer to own centroid = more likely correct

    return {
        "n": n,
        "r_margin_correct": float(r_margin),
        "pval_margin_correct": float(pval_margin),
        "r_dist_own_correct": float(r_dist_own),
        "mean_accuracy": float(correct.mean()),
        "margin_correct_mean": float(margin[correct == 1].mean()),
        "margin_wrong_mean": float(margin[correct == 0].mean()),
    }


# ================================================================
# 4. RUN BOTH REGIONS
# ================================================================
results = {}
results["IT"] = run_region("IT_multiunits", it_features, it_cats)
results["V4"] = run_region("V4_multiunits", v4_features, v4_cats)

# Exploratory: per-image local margin analysis (n=1960 data points)
print("\n[EXPLORATORY] Per-image local kappa margin analysis (n=1960)...")
it_per_image = per_image_local_kappa(it_features, it_cats)
v4_per_image = per_image_local_kappa(v4_features, v4_cats)
print(f"  IT: r(margin, correct) = {it_per_image['r_margin_correct']:.4f} (p={it_per_image['pval_margin_correct']:.4f})")
print(f"  IT: margin_correct={it_per_image['margin_correct_mean']:.4f} vs margin_wrong={it_per_image['margin_wrong_mean']:.4f}")
print(f"  V4: r(margin, correct) = {v4_per_image['r_margin_correct']:.4f} (p={v4_per_image['pval_margin_correct']:.4f})")
results["IT"]["per_image_exploratory"] = it_per_image
results["V4"]["per_image_exploratory"] = v4_per_image

# Compare A_renorm between regions and vs artificial
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nsqrt(4/pi) universal constant = {SQRT_4_OVER_PI:.4f}")
print(f"NLP artificial:  A_renorm ~= 1.129 (1.477/sqrt(1.71))")
print(f"ViT artificial:  A_renorm ~= unknown (need d_eff measurement)")
print()
for region, res in results.items():
    fit = res["fit"]
    print(f"{region}:")
    print(f"  A_renorm_bio = {fit['A_renorm']:.4f}")
    print(f"  r = {fit['r']:.4f}")
    print(f"  H1={'PASS' if res['hypotheses']['H1_r_pass'] else 'FAIL'}, "
          f"H2={'PASS' if res['hypotheses']['H2_renorm_pass'] else 'FAIL'}, "
          f"H3={'PASS' if res['hypotheses']['H3_mae_pass'] else 'FAIL'}, "
          f"H4={'PASS' if res['hypotheses']['H4_substrate_independent'] else 'FAIL'}")

# Substrate independence verdict
it_h4 = results["IT"]["hypotheses"]["H4_substrate_independent"]
print(f"\n{'[PASS] SUBSTRATE-INDEPENDENT LAW CONFIRMED' if it_h4 else 'Substrate independence NOT confirmed (A_bio != A_art)'}")
print(f"(IT cortex: H1+H2 = {'PASS' if it_h4 else 'FAIL'})")

# ================================================================
# 5. SAVE RESULTS
# ================================================================
output = {
    "experiment": "CTI_biological_validation_Cadieu2014",
    "preregistration_commit": PREREQ_COMMIT,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "dataset": "Cadieu2014 PLoSCB, macaque IT + V4 multiunits",
    "note": "Cadieu2014 used instead of MajajHong2015 (AWS credentials required); same macaque IT cortex, K=7 instead of K=8",
    "universal_constant_prediction": SQRT_4_OVER_PI,
    "nlp_artificial_A_renorm": 1.129,
    "results": results,
    "substrate_independence": {
        "verdict": "PASS" if it_h4 else "FAIL",
        "IT_H4": bool(it_h4),
        "IT_A_renorm": float(results["IT"]["fit"]["A_renorm"]),
        "distance_from_prediction": float(abs(results["IT"]["fit"]["A_renorm"] - SQRT_4_OVER_PI)),
    }
}

os.makedirs("results", exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUT_FILE}")
