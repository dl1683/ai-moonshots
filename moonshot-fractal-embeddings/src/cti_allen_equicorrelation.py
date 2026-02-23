#!/usr/bin/env python -u
"""
ALLEN NEUROPIXELS EQUICORRELATION TEST
=======================================

Extension of Session 69 equicorrelation result to biological neural data.

QUESTION: Does the biological visual cortex (Allen Neuropixels, K=118 visual
categories) also show rho ~ 0.5 (regular simplex / Neural Collapse geometry)?

If YES: the near-simplex centroid arrangement is universal across both
  artificial (LM) and biological (visual cortex) representational systems.
  This would be a profound discovery linking CTI universality to a deep
  geometric principle of representation learning.

If NO: the near-simplex geometry is specific to trained LM embeddings,
  likely arising from the training objective (cross-entropy).

PREDICTION (from Session 69):
  rho = avg Sigma_W-whitened cosine similarity of centroid differences
  For regular simplex: rho = 0.5 exactly, independent of K
  For LMs: rho ~ 0.45 (CV=3.9% across 5 archs, CV=7.8% across K=4,14,77)

PROTOCOL:
  Load 5 representative Allen sessions (already run in cti_allen_all_sessions_complete.json)
  For each: extract response matrix, run PCA, compute class centroids + Sigma_W
  Measure rho and d_eff_comp = 1/(1-rho)
  Compare to LM result (rho~0.45) and simplex prediction (rho=0.5)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA, TruncatedSVD

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# 5 representative sessions from the successful 30/32 batch
# Session keys selected from cti_allen_all_sessions_complete.json
TARGET_SESSIONS = [
    "sub-699733573_ses-715093703",    # r_kappa=0.863 (high)
    "sub-718643564_ses-737581020",    # r_kappa=0.851 (high)
    "sub-726170927_ses-746083955",    # r_kappa=0.845 (high)
    "sub-734865729_ses-756029989",    # r_kappa=0.852 (high)
    "sub-738651046_ses-760693773",    # r_kappa=0.817
]

N_PCA = 100   # PCA components for neural data (standard from Allen batch)


def load_session_response(url, session_key):
    """Load neural response matrix for a session."""
    import h5py
    import remfile

    t0 = time.time()
    print(f"  Loading {session_key}...", flush=True)

    rf = remfile.File(url)
    f = h5py.File(rf, "r")

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
    K = len(np.unique(valid_frames))
    print(f"  n_pres={n_pres}, K={K}", flush=True)

    units = f["units"]
    quality = units["quality"][:].astype(str) if "quality" in units else None
    if quality is not None:
        good_mask = np.array([q.strip() == "good" for q in quality])
    else:
        good_mask = np.ones(len(units["id"][:]), dtype=bool)
    good_idx = np.where(good_mask)[0]
    n_units = len(good_idx)

    all_spike_times = units["spike_times"][:]
    all_idx = units["spike_times_index"][:]
    f.close()

    response_start, response_end = 0.05, 0.25
    prev_ends = np.concatenate([[0], all_idx[:-1]])
    R = np.zeros((n_pres, n_units), dtype=np.float32)
    stim_starts_arr = valid_starts + response_start
    stim_stops_arr = valid_starts + response_end

    for j, unit_j in enumerate(good_idx):
        spikes = np.sort(all_spike_times[int(prev_ends[unit_j]):int(all_idx[unit_j])])
        R[:, j] = (np.searchsorted(spikes, stim_stops_arr) -
                   np.searchsorted(spikes, stim_starts_arr))

    elapsed = time.time() - t0
    print(f"  Loaded R={R.shape}, n_active={n_units}, elapsed={elapsed:.1f}s", flush=True)
    return R, valid_frames, K, n_units


def compute_equicorrelation_neural(R, labels, n_pca=100):
    """
    Compute competition equicorrelation rho for neural response data.

    For neural data: each 'dimension' is a neuron, 'class' is visual category.
    Same formula as for LM embeddings.
    """
    classes = np.unique(labels)
    K = len(classes)
    n, d_raw = R.shape

    # Apply PCA to reduce dimensionality (same as Allen batch script)
    n_comp = min(n_pca, d_raw - 1, n - 1)
    pca = PCA(n_components=n_comp)
    X = pca.fit_transform(R).astype(np.float64)  # shape (n_pres, n_pca)

    # Compute class centroids in PCA space
    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = X[mask].mean(0)

    if len(centroids) < K:
        return None, None, None

    centroid_arr = np.array([centroids[c] for c in sorted(centroids)])  # K x n_pca

    # Within-class covariance in PCA space
    Xc_list = []
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            Xc_list.append(X[mask] - centroids[c])
    Xc = np.concatenate(Xc_list)  # shape (N, n_pca)
    N_total = len(Xc)

    # For neural data with n_pca components, compute full Sigma_W (n_pca x n_pca)
    # since n_pca is small (100)
    Sigma_W = (Xc.T @ Xc) / N_total  # (n_pca, n_pca)

    # Cholesky decomposition of Sigma_W for whitening
    # Add small regularization for stability
    Sigma_W_reg = Sigma_W + 1e-6 * np.trace(Sigma_W) / n_comp * np.eye(n_comp)
    try:
        L = np.linalg.cholesky(Sigma_W_reg)  # Sigma_W = L @ L.T
        # Sigma_W^{1/2} whitened delta: L^T @ delta (or L @ delta depending on convention)
        # We want: whitened_delta = Sigma_W^{1/2} delta = L @ L.T delta...
        # Actually: delta^T Sigma_W delta = (L^T delta)^T (L^T delta) = ||L^T delta||^2
        # So whitened_delta_j = L.T @ delta_j
        use_chol = True
    except np.linalg.LinAlgError:
        # Fallback: SVD-based whitening
        use_chol = False
        U, s, Vt = np.linalg.svd(Sigma_W_reg)
        sqrt_S = np.diag(np.sqrt(np.maximum(s, 1e-12)))

    classes_sorted = sorted(centroids.keys())
    rho_per_class = []

    for ci, c in enumerate(classes_sorted):
        other = [i for i in range(K) if i != ci]
        deltas = centroid_arr[other] - centroids[c]  # (K-1, n_pca)

        if use_chol:
            # whitened_delta = L.T @ delta.T for each delta
            wh = (L.T @ deltas.T).T  # (K-1, n_pca): each row is L^T @ delta_j
        else:
            wh = (sqrt_S @ U.T @ deltas.T).T

        norms = np.linalg.norm(wh, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        wh_n = wh / norms
        cos_mat = wh_n @ wh_n.T  # (K-1, K-1)

        off = ~np.eye(K-1, dtype=bool)
        rho_per_class.append(float(cos_mat[off].mean()))

    rho = float(np.mean(rho_per_class))
    rho_std = float(np.std(rho_per_class))
    d_eff = 1.0 / (1.0 - rho) if rho < 1.0 else float("inf")

    return rho, d_eff, rho_std


def main():
    from dandi.dandiapi import DandiAPIClient

    print("ALLEN NEUROPIXELS EQUICORRELATION TEST", flush=True)
    print(f"Prediction: rho ~ 0.45-0.50 (near-simplex, same as LM decoders)", flush=True)
    print(f"LM result: mean rho=0.452, CV=3.9% (Session 69)", flush=True)
    print(f"Target {len(TARGET_SESSIONS)} Allen sessions (K=118 visual categories)", flush=True)

    # Get DANDI asset URLs
    print("\nFetching DANDI asset list...", flush=True)
    client = DandiAPIClient()
    dandiset = client.get_dandiset("000021")
    assets = list(dandiset.get_assets())
    main_nwb = [
        a for a in assets
        if a.path.endswith(".nwb")
        and "_probe-" not in a.path
        and "_ecephys" not in a.path
    ]
    # Build map from session key -> URL
    session_url_map = {}
    for a in main_nwb:
        key = a.path.split("/")[-1].replace(".nwb", "")
        session_url_map[key] = a.download_url
    print(f"  Found {len(session_url_map)} main sessions", flush=True)

    results = {}

    for session_key in TARGET_SESSIONS:
        url = session_url_map.get(session_key)
        if url is None:
            print(f"  WARNING: session {session_key} not found in DANDI", flush=True)
            continue
        print(f"\n--- {session_key} ---", flush=True)
        t0 = time.time()
        try:
            R, labels, K, n_units = load_session_response(url, session_key)
        except Exception as e:
            print(f"  ERROR loading: {e}", flush=True)
            continue

        try:
            rho, d_eff, rho_std = compute_equicorrelation_neural(R, labels, N_PCA)
        except Exception as e:
            print(f"  ERROR computing rho: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0

        if rho is None:
            print(f"  rho computation failed", flush=True)
            continue

        print(f"  rho = {rho:.4f} +/- {rho_std:.4f}", flush=True)
        print(f"  d_eff_comp = {d_eff:.4f}", flush=True)
        print(f"  elapsed = {elapsed:.1f}s", flush=True)

        results[session_key] = {
            "rho": float(rho),
            "rho_std_per_class": float(rho_std),
            "d_eff_comp": float(d_eff),
            "K": int(K),
            "n_units": int(n_units),
            "elapsed_s": float(elapsed),
        }

    if not results:
        print("ERROR: No sessions processed", flush=True)
        return

    # Summary
    print("\n" + "="*60, flush=True)
    print("ALLEN EQUICORRELATION SUMMARY", flush=True)
    print("="*60, flush=True)

    rhos = [results[s]["rho"] for s in results]
    deffs = [results[s]["d_eff_comp"] for s in results]

    for s in results:
        r = results[s]
        print(f"  {s[:30]}: rho={r['rho']:.4f}, d_eff={r['d_eff_comp']:.4f}", flush=True)

    mean_rho = float(np.mean(rhos))
    cv_rho = float(np.std(rhos) / (np.mean(rhos) + 1e-12))
    mean_deff = float(np.mean(deffs))
    cv_deff = float(np.std(deffs) / (np.mean(deffs) + 1e-12))

    print(f"\n  mean rho = {mean_rho:.4f} (LM: 0.452, simplex: 0.500)", flush=True)
    print(f"  CV(rho) = {cv_rho:.4f}", flush=True)
    print(f"  mean d_eff_comp = {mean_deff:.4f} (LM: 1.829, simplex: 2.000)", flush=True)
    print(f"  CV(d_eff) = {cv_deff:.4f}", flush=True)

    # Assessment
    if abs(mean_rho - 0.5) < 0.1:
        assessment = "NEAR-SIMPLEX: biological visual cortex shows near-NC geometry similar to LMs"
    elif abs(mean_rho - 0.452) < 0.05:
        assessment = "LM-LIKE: biological rho matches LM decoders closely"
    elif mean_rho < 0.2:
        assessment = "NON-SIMPLEX: biological centroids spread out more uniformly"
    else:
        assessment = f"INTERMEDIATE: rho={mean_rho:.3f}, between random (0) and simplex (0.5)"

    print(f"\n  Assessment: {assessment}", flush=True)

    out = {
        "experiment": "cti_allen_equicorrelation",
        "session_69_lm_result": {"mean_rho": 0.452, "cv_rho": 0.039, "mean_d_eff_comp": 1.829},
        "simplex_prediction": {"rho": 0.5, "d_eff_comp": 2.0},
        "sessions": results,
        "summary": {
            "mean_rho": mean_rho,
            "std_rho": float(np.std(rhos)),
            "cv_rho": cv_rho,
            "mean_d_eff_comp": mean_deff,
            "cv_d_eff_comp": cv_deff,
            "assessment": assessment,
        }
    }

    out_path = RESULTS_DIR / "cti_allen_equicorrelation.json"
    with open(out_path, "w", encoding="ascii") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
