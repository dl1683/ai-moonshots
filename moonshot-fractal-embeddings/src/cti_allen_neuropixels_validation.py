"""
CTI Universal Law Validation: Allen Brain Observatory Neuropixels
=================================================================
Tests whether logit(q_norm) = A * kappa_nearest * sqrt(d_eff) + C holds
for mouse visual cortex neural population responses to K=118 natural scenes.

Dataset: DANDI 000021 - Allen Institute Visual Coding Neuropixels
         Brain Observatory 1.1 Stimulus Set
         K=118 distinct natural images, ~50 trials each

Pre-registration commit: bddec1d (covers all biological neural datasets)
Session tested: sub-707296975_ses-721123822 (smallest session, 1736 MB)
"""

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import json
import time
import warnings
warnings.filterwarnings('ignore')


def build_response_matrix(all_spike_times, all_idx, good_idx, valid_starts, n_pres,
                          response_start=0.05, response_end=0.25):
    """
    Build response matrix R (n_pres x n_good_units): spike counts in response window.
    all_spike_times: all concatenated spike times (sorted per unit)
    all_idx: cumulative end-indices per unit
    good_idx: indices of 'good' quality units
    valid_starts: stimulus onset times for each valid presentation
    """
    n_units = len(good_idx)
    R = np.zeros((n_pres, n_units), dtype=np.float32)

    stim_starts = valid_starts + response_start
    stim_stops = valid_starts + response_end

    # Build prev_ends array for fast indexing
    prev_ends = np.concatenate([[0], all_idx[:-1]])

    for j, unit_j in enumerate(good_idx):
        if j % 100 == 0:
            print(f"  Building responses: unit {j}/{n_units}...", flush=True)
        start_offset = int(prev_ends[unit_j])
        end_offset = int(all_idx[unit_j])
        spikes = np.sort(all_spike_times[start_offset:end_offset])
        R[:, j] = (np.searchsorted(spikes, stim_stops) -
                   np.searchsorted(spikes, stim_starts))
    return R


def compute_cti_inputs_bio(responses, labels, n_pca=100):
    """
    Compute kappa_nearest and d_eff per class for biological neural data.
    Uses PCA reduction for tractability with large d.
    CRITICAL: kappa and 1-NN computed in SAME PCA space.

    Returns: kappa_nearest, d_eff, q_1nn, q_norm, logit_q_norm per class
    """
    K = len(np.unique(labels))
    n_pca_actual = min(n_pca, responses.shape[1] - 1, responses.shape[0] - K)

    # PCA reduce
    pca = PCA(n_components=n_pca_actual)
    R_pca = pca.fit_transform(responses.astype(np.float64))
    d = n_pca_actual

    class_ids = np.unique(labels)
    # Compute class centroids
    centroids = np.array([R_pca[labels == c].mean(axis=0) for c in class_ids])

    # Global within-class statistics (diagonal covariance approx)
    all_within = []
    for c in class_ids:
        X_c = R_pca[labels == c]
        all_within.append(X_c - X_c.mean(axis=0))
    all_within = np.vstack(all_within)

    # tr(Sigma_W) / d = mean within-class variance across PCs
    sigma_W_sq_per_dim = np.var(all_within, axis=0, ddof=1)
    tr_Sigma_W = sigma_W_sq_per_dim.sum()
    sigma_W_global = np.sqrt(tr_Sigma_W / d)  # sqrt(mean_variance)

    # Per-class statistics
    kappa_nearest_list = []
    d_eff_list = []
    q_1nn_list = []
    q_norm_list = []
    logit_q_norm_list = []

    for i, c in enumerate(class_ids):
        X_i = R_pca[labels == c]
        mu_i = centroids[i]
        n_i = len(X_i)

        # Nearest other class centroid
        dists = np.array([np.linalg.norm(mu_i - centroids[j])
                          for j in range(K) if j != i])
        delta_min = dists.min()

        # kappa_nearest = delta_min / (sigma_W_global * sqrt(d))
        kappa_i = delta_min / (sigma_W_global * np.sqrt(d))

        # d_eff_formula = tr(Sigma_W) / sigma_centroid_dir^2
        # sigma_centroid_dir^2 = variance of within-class responses projected onto
        # direction toward nearest centroid
        nearest_j = np.argmin([np.linalg.norm(mu_i - centroids[j])
                                for j in range(K) if j != i])
        j_idx = [j for j in range(K) if j != i][nearest_j]
        direction = centroids[j_idx] - mu_i
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        X_i_centered = X_i - mu_i
        proj = X_i_centered @ direction
        sigma_centroid_dir_sq = np.var(proj, ddof=1)
        if sigma_centroid_dir_sq > 0:
            d_eff_i = tr_Sigma_W / (d * sigma_centroid_dir_sq)
        else:
            d_eff_i = 1.0

        # LOO 1-NN accuracy for class i
        correct = 0
        for n in range(n_i):
            query = X_i[n]
            # Leave one out: compute distances to all other samples
            other_X = np.delete(X_i, n, axis=0)
            other_labels_i = np.full(n_i - 1, c)

            # All other class samples
            other_X_all = []
            other_labels_all = []
            for jj, cc in enumerate(class_ids):
                if cc == c:
                    continue
                X_jj = R_pca[labels == cc]
                other_X_all.append(X_jj)
                other_labels_all.extend([cc] * len(X_jj))

            if len(other_X) > 0:
                all_others = np.vstack([other_X] + other_X_all)
                all_other_labels = list(other_labels_i) + other_labels_all
            else:
                all_others = np.vstack(other_X_all)
                all_other_labels = other_labels_all

            dists_to_query = np.linalg.norm(all_others - query, axis=1)
            nn_label = all_other_labels[np.argmin(dists_to_query)]
            if nn_label == c:
                correct += 1

        q_i = correct / n_i if n_i > 0 else 0.0

        # Normalize and logit
        q_norm_i = max((q_i - 1.0 / K) / (1.0 - 1.0 / K), 1e-6)
        q_norm_i = min(q_norm_i, 1.0 - 1e-6)
        logit_i = np.log(q_norm_i / (1.0 - q_norm_i))

        kappa_nearest_list.append(float(kappa_i))
        d_eff_list.append(float(d_eff_i))
        q_1nn_list.append(float(q_i))
        q_norm_list.append(float(q_norm_i))
        logit_q_norm_list.append(float(logit_i))

    return (np.array(kappa_nearest_list), np.array(d_eff_list),
            np.array(q_1nn_list), np.array(q_norm_list), np.array(logit_q_norm_list))


def loco_fit(kappa, d_eff, logit_q):
    """LOO coefficient-of-determination fit: A * kappa * sqrt(d_eff) + C"""
    x = kappa * np.sqrt(d_eff)
    from sklearn.linear_model import LinearRegression
    mae_vals = []
    for i in range(len(kappa)):
        mask = np.ones(len(kappa), dtype=bool)
        mask[i] = False
        X_tr = x[mask].reshape(-1, 1)
        y_tr = logit_q[mask]
        reg = LinearRegression().fit(X_tr, y_tr)
        pred = reg.predict(x[i:i+1].reshape(-1, 1))
        mae_vals.append(abs(pred[0] - logit_q[i]))
    return np.mean(mae_vals)


def per_image_margin(R_pca, labels, class_ids, centroids):
    """
    Per-image margin: distance to own centroid vs nearest other centroid.
    Returns (margin, correct) for each presentation.
    """
    sigma_W = np.std(R_pca - np.array([centroids[list(class_ids).index(l)]
                                        for l in labels]), axis=None)
    d = R_pca.shape[1]

    margins = []
    corrects = []

    # Process in batches for efficiency
    batch_size = 500
    for start in range(0, len(R_pca), batch_size):
        end = min(start + batch_size, len(R_pca))
        batch = R_pca[start:end]  # (batch, d)
        batch_labels = labels[start:end]

        # Distance to all centroids: (batch, K)
        d_to_centroids = np.sqrt(np.sum(
            (batch[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=-1))

        for i in range(len(batch)):
            true_class_idx = list(class_ids).index(batch_labels[i])
            d_own = d_to_centroids[i, true_class_idx]

            # Nearest other centroid
            d_others = np.delete(d_to_centroids[i], true_class_idx)
            d_nearest_other = d_others.min()

            # Predicted class (centroid classifier)
            pred_idx = np.argmin(d_to_centroids[i])
            correct = (pred_idx == true_class_idx)

            # Margin: positive = own is closer (correct geometry)
            margin = (d_nearest_other - d_own) / (sigma_W * np.sqrt(d) + 1e-10)
            margins.append(margin)
            corrects.append(int(correct))

    return np.array(margins), np.array(corrects)


def run_validation():
    from dandi.dandiapi import DandiAPIClient
    import remfile
    import h5py

    print("=" * 60)
    print("Allen Neuropixels CTI Validation")
    print("Pre-registration: bddec1d")
    print("=" * 60)

    # 1. Connect and find session
    print("\n1. Connecting to DANDI 000021...")
    client = DandiAPIClient()
    dandiset = client.get_dandiset('000021')
    assets = list(dandiset.get_assets())
    session_assets = [a for a in assets
                      if '_probe-' not in a.path and a.path.endswith('.nwb')]
    smallest = min(session_assets, key=lambda x: x.size)
    print(f"   Session: {smallest.path} ({smallest.size//1e6:.0f} MB)")

    session_id = smallest.path.split('/')[0]

    # 2. Load data lazily
    print("\n2. Loading data (lazy remote read)...")
    url = smallest.get_content_url(follow_redirects=True, strip_query=True)
    rf = remfile.File(url)
    f = h5py.File(rf, 'r')

    # 3. Read stimulus table
    ns = f['intervals']['natural_scenes_presentations']
    frames = ns['frame'][:]
    start_times = ns['start_time'][:]
    stop_times = ns['stop_time'][:]

    # Filter to valid (non-blank) presentations
    valid_mask = frames >= 0
    valid_frames = frames[valid_mask].astype(int)
    valid_starts = start_times[valid_mask]
    valid_stops = stop_times[valid_mask]

    K = len(np.unique(valid_frames))
    n_pres = valid_mask.sum()
    trial_dur = np.median(valid_stops - valid_starts)
    print(f"   K={K} images, {n_pres} valid presentations, {trial_dur*1000:.0f}ms/trial")

    # Check trial counts per image
    trials_per_image = np.array([np.sum(valid_frames == k) for k in range(K)])
    print(f"   Trials/image: mean={trials_per_image.mean():.1f}, "
          f"min={trials_per_image.min()}, max={trials_per_image.max()}")

    # 4. Read units
    units = f['units']
    quality = units['quality'][:]
    good_idx = np.where(quality == b'good')[0]
    print(f"\n3. Units: {len(good_idx)} good quality")

    # 5. Read all spike times
    print("\n4. Reading all spike times...")
    t0 = time.time()
    all_idx = units['spike_times_index'][:]
    all_spike_times = units['spike_times'][:]
    print(f"   {len(all_spike_times)/1e6:.1f}M spikes in {time.time()-t0:.1f}s")

    # 6. Build response matrix
    print(f"\n5. Building response matrix ({n_pres} x {len(good_idx)})...")
    t0 = time.time()
    R = build_response_matrix(
        all_spike_times, all_idx, good_idx, valid_starts, n_pres,
        response_start=0.05, response_end=0.25
    )
    print(f"   Done in {time.time()-t0:.1f}s. "
          f"Mean firing rate: {R.mean()/(0.25-0.05)*1000:.1f} Hz")

    # 7. Filter units with sufficient firing
    mean_rate = R.mean(axis=0) / (0.25 - 0.05)  # spikes/second
    active_mask = mean_rate > 0.5  # at least 0.5 Hz mean
    R_active = R[:, active_mask]
    print(f"\n6. Active units (>0.5Hz): {active_mask.sum()}")
    d_raw = R_active.shape[1]

    # 8. Compute CTI inputs using PCA
    print(f"\n7. Computing CTI inputs (PCA-100, K={K})...")
    n_pca = min(100, d_raw - 1)

    # PCA on the response matrix
    pca = PCA(n_components=n_pca)
    R_pca = pca.fit_transform(R_active.astype(np.float64))
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"   PCA {n_pca}d: {var_explained*100:.1f}% variance explained")

    # Class labels = frame indices
    labels_arr = valid_frames  # shape (n_pres,)
    class_ids = np.arange(K)

    # Centroids
    centroids = np.array([R_pca[labels_arr == c].mean(axis=0) for c in class_ids])

    # 9. Compute kappa_nearest per image
    print("\n8. Computing kappa_nearest and d_eff per image...")
    d = n_pca

    # Global within-class spread
    all_within = []
    for c in class_ids:
        X_c = R_pca[labels_arr == c]
        all_within.append(X_c - X_c.mean(axis=0))
    all_within = np.vstack(all_within)
    sigma_W_sq_per_dim = np.var(all_within, axis=0, ddof=1)
    tr_Sigma_W = sigma_W_sq_per_dim.sum()
    sigma_W_global = np.sqrt(tr_Sigma_W / d)
    print(f"   sigma_W_global = {sigma_W_global:.4f}")

    # Pairwise centroid distances (K x K)
    from scipy.spatial.distance import cdist
    centroid_dists = cdist(centroids, centroids)
    np.fill_diagonal(centroid_dists, np.inf)

    kappa_nearest_arr = []
    d_eff_arr = []
    nearest_class_arr = []

    for i in range(K):
        # Nearest centroid
        nearest_j = np.argmin(centroid_dists[i])
        delta_min = centroid_dists[i, nearest_j]

        # kappa_nearest
        kappa_i = delta_min / (sigma_W_global * np.sqrt(d))

        # d_eff: direction toward nearest centroid
        direction = centroids[nearest_j] - centroids[i]
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        X_i = R_pca[labels_arr == i]
        proj = (X_i - centroids[i]) @ direction
        sigma_centroid_dir_sq = np.var(proj, ddof=1)
        d_eff_i = tr_Sigma_W / max(sigma_centroid_dir_sq, 1e-10)

        kappa_nearest_arr.append(float(kappa_i))
        d_eff_arr.append(float(d_eff_i))
        nearest_class_arr.append(int(nearest_j))

    kappa_nearest = np.array(kappa_nearest_arr)
    d_eff_arr = np.array(d_eff_arr)

    # 10. Compute accuracy: centroid classifier + LOO 1-NN
    print("\n9. Computing accuracy per image...")

    # Centroid classifier (fast, K=118)
    pred_centroid = np.argmin(cdist(R_pca, centroids), axis=1)
    per_class_centroid = np.array([
        np.mean(pred_centroid[labels_arr == c] == c) for c in class_ids
    ])
    print(f"   Centroid classifier mean acc: {per_class_centroid.mean():.3f} "
          f"(chance={1/K:.3f})")

    # LOO 1-NN: full pairwise distance matrix (5900^2 in 100D, ~2-5s)
    print("   Computing LOO 1-NN (full dist matrix)...")
    t_nn = time.time()
    dist_mat = cdist(R_pca, R_pca, metric='euclidean').astype(np.float32)
    np.fill_diagonal(dist_mat, np.inf)
    nn_preds = labels_arr[np.argmin(dist_mat, axis=1)]
    per_class_1nn = np.array([
        np.mean(nn_preds[labels_arr == c] == c) for c in class_ids
    ])
    del dist_mat  # free memory
    print(f"   LOO 1-NN mean acc: {per_class_1nn.mean():.3f} "
          f"(done in {time.time()-t_nn:.1f}s)")

    # Use LOO 1-NN as q_1nn (matches CTI theory)
    q_1nn = per_class_1nn

    # 11. q_norm and logit
    q_norm = np.clip((q_1nn - 1.0 / K) / (1.0 - 1.0 / K), 1e-4, 1.0 - 1e-4)
    logit_q_norm = np.log(q_norm / (1.0 - q_norm))

    # 12. CTI law fit
    # Pre-registered H1: r(kappa_nearest, logit_q_1nn) > 0.50 (kappa alone, per prereg bddec1d)
    r_kappa_only, p_kappa = stats.pearsonr(kappa_nearest, logit_q_norm)
    # Full law: kappa * sqrt(d_eff)
    x_fit_full = kappa_nearest * np.sqrt(d_eff_arr)
    r_full, p_full = stats.pearsonr(x_fit_full, logit_q_norm)

    # OLS fit: kappa_nearest alone (matches pre-registration H1 metric)
    X_kappa = np.column_stack([kappa_nearest, np.ones(K)])
    A_kappa, C_kappa = np.linalg.lstsq(X_kappa, logit_q_norm, rcond=None)[0]

    # OLS fit: full law kappa*sqrt(d_eff)
    X_full = np.column_stack([x_fit_full, np.ones(K)])
    A_fit, C_fit = np.linalg.lstsq(X_full, logit_q_norm, rcond=None)[0]

    mean_d_eff = d_eff_arr.mean()
    A_renorm = A_fit / np.sqrt(mean_d_eff)

    # LOCO MAE using kappa_nearest alone (per pre-registration H3)
    mae_vals = []
    for i in range(K):
        mask = np.ones(K, dtype=bool)
        mask[i] = False
        A_i, C_i = np.linalg.lstsq(X_kappa[mask], logit_q_norm[mask], rcond=None)[0]
        pred_logit = A_i * kappa_nearest[i] + C_i
        pred_q_norm = 1.0 / (1.0 + np.exp(-pred_logit))
        pred_q = pred_q_norm * (1.0 - 1.0/K) + 1.0/K
        mae_vals.append(abs(pred_q - q_1nn[i]))
    mae_loco = float(np.mean(mae_vals))

    print(f"\n10. CTI Law Fit:")
    print(f"   r(kappa_nearest, logit_q) = {r_kappa_only:.4f} (p={p_kappa:.2e})  [PRE-REG H1]")
    print(f"   r(kappa*sqrt(d_eff), logit_q) = {r_full:.4f} (p={p_full:.2e})  [full law]")
    print(f"   A(kappa only) = {A_kappa:.4f}, C = {C_kappa:.4f}")
    print(f"   A(full law) = {A_fit:.4f}, C = {C_fit:.4f}")
    print(f"   A_renorm = A_full/sqrt(mean_d_eff) = {A_renorm:.4f}")
    print(f"   mean_d_eff = {mean_d_eff:.2f}")
    print(f"   LOCO MAE (kappa-only fit, q units) = {mae_loco:.4f}")

    # 13. Hypothesis tests (per pre-registration bddec1d)
    # H1: r(kappa_nearest, logit_q_1nn) > 0.50
    H1 = bool(r_kappa_only > 0.50)
    # H2: A_renorm in [0.70, 1.80]
    H2 = bool(0.70 <= A_renorm <= 1.80)
    # H3: LOCO MAE < 0.10
    H3 = bool(mae_loco < 0.10)
    # H4: H1 AND H2
    H4 = bool(H1 and H2)

    print(f"\n11. Hypothesis Tests (per bddec1d pre-registration):")
    print(f"   H1 r(kappa_nearest,logit_q)>0.50: {H1} (r={r_kappa_only:.3f})")
    print(f"   H2 A_renorm in [0.70,1.80]: {H2} (A_renorm={A_renorm:.4f})")
    print(f"   H3 LOCO MAE < 0.10: {H3} (MAE={mae_loco:.4f})")
    print(f"   H4 substrate independence (H1 AND H2): {H4}")

    # 14. Per-image exploratory analysis
    print("\n12. Per-image margin analysis (exploratory)...")
    margins, corrects = per_image_margin(R_pca, labels_arr, class_ids, centroids)
    r_margin, p_margin = stats.pearsonr(margins, corrects)
    mean_acc = corrects.mean()
    margin_correct_mean = margins[corrects == 1].mean()
    margin_wrong_mean = margins[corrects == 0].mean()
    print(f"   n={len(margins)} images, mean_acc={mean_acc:.3f}")
    print(f"   r(margin, correct) = {r_margin:.4f} (p={p_margin:.2e})")
    print(f"   Margin correct: {margin_correct_mean:.4f}, wrong: {margin_wrong_mean:.4f}")

    # 15. Save results
    results = {
        "experiment": "CTI_allen_neuropixels_visual_coding",
        "preregistration_commit": "bddec1d",
        "dataset": "DANDI:000021",
        "session": session_id,
        "K": int(K),
        "d_raw": int(d_raw),
        "d_pca": int(d),
        "n_presentations": int(n_pres),
        "n_units_good": int(len(good_idx)),
        "n_units_active": int(active_mask.sum()),
        "pca_var_explained": float(var_explained),
        "trials_per_image_mean": float(trials_per_image.mean()),
        "kappa_nearest": kappa_nearest.tolist(),
        "d_eff": d_eff_arr.tolist(),
        "q_centroid": per_class_centroid.tolist(),
        "q_1nn": per_class_1nn.tolist(),
        "q_norm": q_norm.tolist(),
        "logit_q_norm": logit_q_norm.tolist(),
        "fit": {
            "r_kappa_nearest": float(r_kappa_only),
            "pval_kappa": float(p_kappa),
            "r_kappa_sqrt_deff": float(r_full),
            "pval_full": float(p_full),
            "A_kappa_only": float(A_kappa),
            "C_kappa_only": float(C_kappa),
            "A_full": float(A_fit),
            "C_full": float(C_fit),
            "A_renorm": float(A_renorm),
            "mean_d_eff": float(mean_d_eff),
            "mae_loco_q": float(mae_loco),
            "note": "H1 per-registered as r(kappa_nearest,logit_q)>0.50; full law A*kappa*sqrt(d_eff)+C also reported"
        },
        "hypotheses": {
            "H1_r_kappa_nearest_pass": H1,
            "H2_renorm_pass": H2,
            "H3_mae_pass": H3,
            "H4_substrate_independent": H4
        },
        "per_image_exploratory": {
            "n": int(len(margins)),
            "r_margin_correct": float(r_margin),
            "pval_margin_correct": float(p_margin),
            "mean_accuracy_centroid": float(mean_acc),
            "margin_correct_mean": float(margin_correct_mean),
            "margin_wrong_mean": float(margin_wrong_mean),
            "note": "Centroid classifier (not LOO 1-NN) for tractability with K=118"
        },
        "prior_bio_validation": {
            "cadieu2014_IT_r": 0.1856,
            "cadieu2014_IT_per_image_r": 0.4136,
            "stringer2018b_V1_r": -0.4834,
            "stringer2018b_V1_per_image_r": 0.6435
        }
    }

    out_path = "results/cti_allen_neuropixels_validation.json"
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == '__main__':
    run_validation()
