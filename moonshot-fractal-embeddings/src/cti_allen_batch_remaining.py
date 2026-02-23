"""Run CTI validation on all remaining Allen Neuropixels sessions (DANDI:000021)."""
import numpy as np
import h5py
import remfile
import time
import json
import sys
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


def logit_safe(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def process_session(url, session_key):
    t0 = time.time()
    try:
        rf = remfile.File(url)
        f = h5py.File(rf, "r")
    except Exception as e:
        return None, f"open_fail: {e}"

    try:
        stim = f["intervals"]["natural_scenes_presentations"]
        if "frame_index" in stim:
            frame_idx = stim["frame_index"][:]
        else:
            frame_idx = stim["frame"][:]
        start_times = stim["start_time"][:]
        valid_mask = frame_idx != -1
        valid_starts = start_times[valid_mask]
        valid_frames = frame_idx[valid_mask]
        n_pres = len(valid_starts)
        if n_pres < 1000:
            return None, f"too_few_pres: {n_pres}"
        K = len(np.unique(valid_frames))
        if K != 118:
            return None, f"K={K} not 118"
    except KeyError as e:
        return None, f"no_natural_scenes: {e}"

    units = f["units"]
    quality = units["quality"][:].astype(str) if "quality" in units else None
    if quality is not None:
        good_mask = np.array([q.strip() == "good" for q in quality])
    else:
        good_mask = np.ones(len(units["id"][:]), dtype=bool)
    good_idx = np.where(good_mask)[0]
    n_units_good = len(good_idx)
    if n_units_good < 50:
        return None, f"too_few_units: {n_units_good}"

    all_spike_times = units["spike_times"][:]
    all_idx = units["spike_times_index"][:]
    f.close()

    response_start, response_end = 0.05, 0.25
    prev_ends = np.concatenate([[0], all_idx[:-1]])
    R = np.zeros((n_pres, n_units_good), dtype=np.float32)
    stim_starts_arr = valid_starts + response_start
    stim_stops_arr = valid_starts + response_end

    for j, unit_j in enumerate(good_idx):
        spikes = np.sort(
            all_spike_times[int(prev_ends[unit_j]) : int(all_idx[unit_j])]
        )
        R[:, j] = np.searchsorted(spikes, stim_stops_arr) - np.searchsorted(
            spikes, stim_starts_arr
        )

    n_pca = min(100, n_units_good - 1)
    pca = PCA(n_components=n_pca)
    R_pca = pca.fit_transform(R)

    # LOO 1-NN via full distance matrix
    dist_mat = np.sum(
        (R_pca[:, None, :] - R_pca[None, :, :]) ** 2, axis=-1
    )
    np.fill_diagonal(dist_mat, np.inf)
    nn_idx = np.argmin(dist_mat, axis=1)
    nn_labels = valid_frames[nn_idx]
    correct = (nn_labels == valid_frames).astype(float)

    classes = np.unique(valid_frames)
    q_arr = np.array([correct[valid_frames == c].mean() for c in classes])

    # kappa_nearest per class
    centroids = np.array([R_pca[valid_frames == c].mean(0) for c in classes])
    Sigma_W = np.zeros((n_pca, n_pca))
    for c in classes:
        Xi = R_pca[valid_frames == c] - centroids[classes == c][0]
        Sigma_W += Xi.T @ Xi
    Sigma_W /= n_pres
    sigma_W_global = np.sqrt(np.trace(Sigma_W) / n_pca)

    kappa_arr = np.zeros(len(classes))
    for ci, c in enumerate(classes):
        diffs = centroids - centroids[ci]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists[ci] = np.inf
        delta_min = dists.min()
        kappa_arr[ci] = delta_min / (sigma_W_global * np.sqrt(n_pca))

    active_mask = q_arr > 1.5 / 118
    q_a = q_arr[active_mask]
    kappa_a = kappa_arr[active_mask]
    n_active = int(active_mask.sum())
    if n_active < 10:
        return None, f"too_few_active: {n_active}"

    logit_q = logit_safe(q_a)
    r_kappa, _ = pearsonr(kappa_a, logit_q)

    # Per-image margin
    margins = np.zeros(n_pres)
    for pi in range(n_pres):
        ci = np.where(classes == valid_frames[pi])[0][0]
        resp = R_pca[pi]
        dists_to_centroids = np.sqrt(((centroids - resp) ** 2).sum(axis=1))
        dists_to_centroids[ci] = np.inf
        nearest_wrong_dist = dists_to_centroids.min()
        own_dist = np.sqrt(((centroids[ci] - resp) ** 2).sum())
        margins[pi] = nearest_wrong_dist - own_dist
    r_margin, _ = pearsonr(margins, correct)

    t1 = time.time()
    return {
        "session": session_key,
        "K": int(K),
        "n_pres": int(n_pres),
        "n_active": n_active,
        "n_units": n_units_good,
        "r_kappa": float(r_kappa),
        "r_margin": float(r_margin),
        "mean_q": float(q_a.mean()),
        "H1": bool(r_kappa > 0.50),
        "elapsed_s": round(t1 - t0, 1),
    }, None


def main():
    from dandi.dandiapi import DandiAPIClient
    client = DandiAPIClient()
    dandiset = client.get_dandiset("000021")
    assets = list(dandiset.get_assets())
    main_nwb = [
        a for a in assets
        if a.path.endswith(".nwb")
        and "_probe-" not in a.path
        and "_ecephys" not in a.path
    ]

    done_sessions = {
        "sub-707296975_ses-721123822",
        "sub-719828686_ses-754312389",
        "sub-740268983_ses-759883607",
        "sub-757329617_ses-773418906",
        "sub-739783158_ses-760345702",
        "sub-744915196_ses-762602078",
        "sub-722882751_ses-743475441",
    }

    existing = [
        {"session": "sub-707296975_ses-721123822", "r_kappa": 0.851, "r_margin": 0.747, "mean_q": 0.342, "n_active": 963, "H1": True},
        {"session": "sub-719828686_ses-754312389", "r_kappa": 0.755, "r_margin": 0.714, "mean_q": 0.194, "n_active": 995, "H1": True},
        {"session": "sub-740268983_ses-759883607", "r_kappa": 0.801, "r_margin": 0.749, "mean_q": 0.385, "n_active": 1011, "H1": True},
        {"session": "sub-757329617_ses-773418906", "r_kappa": 0.511, "r_margin": 0.725, "mean_q": 0.116, "n_active": 1061, "H1": True},
        {"session": "sub-739783158_ses-760345702", "r_kappa": 0.759, "r_margin": 0.635, "mean_q": 0.817, "n_active": 1016, "H1": True},
        {"session": "sub-744915196_ses-762602078", "r_kappa": 0.885, "r_margin": 0.482, "mean_q": 0.640, "n_active": 1003, "H1": True},
        {"session": "sub-722882751_ses-743475441", "r_kappa": 0.570, "r_margin": 0.772, "mean_q": 0.189, "n_active": 1139, "H1": True},
    ]

    todo = [
        (i, a) for i, a in enumerate(main_nwb)
        if a.path.split("/")[-1].replace(".nwb", "") not in done_sessions
    ]
    print(f"Remaining sessions: {len(todo)}")

    batch_results = []
    for batch_i, (orig_idx, asset) in enumerate(todo):
        key = asset.path.split("/")[-1].replace(".nwb", "")
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
        print(f"\n[{batch_i+1}/{len(todo)}] {key} ({asset.size/1e9:.1f}GB)...", flush=True)
        result, err = process_session(url, key)
        if err:
            print(f"  ERROR: {err}")
            batch_results.append({"session": key, "error": err})
        else:
            batch_results.append(result)
            h1 = "PASS" if result["H1"] else "FAIL"
            print(
                f"  r_kappa={result['r_kappa']:.3f} ({h1}), "
                f"r_margin={result['r_margin']:.3f}, "
                f"n_active={result['n_active']}, time={result['elapsed_s']}s"
            )
        # Save progress after each session
        with open("results/cti_allen_extended_batch.json", "w") as fp:
            json.dump(batch_results, fp, indent=2)
        sys.stdout.flush()

    # Compile full 32-session summary
    all_valid = existing + [r for r in batch_results if "error" not in r]
    r_kappa_vals = [r["r_kappa"] for r in all_valid]
    r_margin_vals = [r["r_margin"] for r in all_valid]
    h1_count = sum(1 for r in all_valid if r["H1"])
    n_total = len(all_valid)

    summary = {
        "experiment": "CTI_allen_neuropixels_all_sessions",
        "preregistration_commit": "bddec1d",
        "dataset": "DANDI:000021",
        "K": 118,
        "n_sessions_attempted": len(existing) + len(batch_results),
        "n_sessions_valid": n_total,
        "sessions": all_valid,
        "summary": {
            "mean_r_kappa": float(np.mean(r_kappa_vals)),
            "std_r_kappa": float(np.std(r_kappa_vals)),
            "cv_r_kappa": float(np.std(r_kappa_vals) / np.mean(r_kappa_vals)),
            "mean_r_margin": float(np.mean(r_margin_vals)),
            "H1_pass_rate": f"{h1_count}/{n_total}",
            "H1_threshold": 0.50,
        },
    }

    with open("results/cti_allen_all_sessions_complete.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n=== FINAL SUMMARY ===")
    print(f"Valid sessions: {n_total}/32")
    print(f"H1 PASS: {h1_count}/{n_total}")
    print(f"Mean r_kappa: {np.mean(r_kappa_vals):.3f} +/- {np.std(r_kappa_vals):.3f}")
    print(f"Mean r_margin: {np.mean(r_margin_vals):.3f}")
    print(f"CV r_kappa: {np.std(r_kappa_vals)/np.mean(r_kappa_vals)*100:.1f}%")


if __name__ == "__main__":
    main()
