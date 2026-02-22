"""
CTI Universal Law: Analysis on BIOLOGICAL NEURONS (DANDI:000039).
Two-photon calcium imaging, mouse V1, drifting gratings (8 orientations).
Tests whether logit(q_norm) = alpha * kappa_nearest + C holds for biological neural codes.

Data already downloaded to data/neuro_session.nwb.

FIX v2: Use np.searchsorted for robust timestamp alignment (not strict masking).
         Use leave-one-out CV for accuracy (not 1-NN on training set).
"""
import numpy as np
from scipy.stats import pearsonr, linregress
from scipy.special import logit
from sklearn.neighbors import KNeighborsClassifier
import pynwb
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

print("=== CTI Universal Law: Biological Neural Analysis ===")
print("Dataset: DANDI:000039, Session sub-673647168_ses-698273664")
print("Modality: Two-photon calcium imaging, mouse visual cortex (V1)")
print()

# --- Load NWB ---
io = pynwb.NWBHDF5IO('data/neuro_session.nwb', 'r')
nwb = io.read()

# Get fluorescence (DfOverF)
bop = nwb.processing['brain_observatory_pipeline']
fluor = bop.data_interfaces['Fluorescence']
dff = fluor.roi_response_series['DfOverF']
F = dff.data[:]        # [T, N] = [65616, 15]
timestamps = dff.timestamps[:]
print(f"Neural data: {F.shape} (time x neurons)")
print(f"DFF timestamps: [{timestamps[0]:.2f}, {timestamps[-1]:.2f}] s")
dt = np.median(np.diff(timestamps))
fs = 1.0 / dt
print(f"Sampling: dt={dt*1000:.2f}ms, fs={fs:.1f} Hz")

# Get stimulus table
epochs_df = nwb.intervals['epochs'].to_dataframe()
print(f"\nStimulus table: {epochs_df.shape}")
print(f"Epoch time range: [{epochs_df['start_time'].min():.2f}, {epochs_df['stop_time'].max():.2f}] s")
print(f"Directions: {sorted(epochs_df['direction'].dropna().unique())}")
print(f"TFs: {sorted(epochs_df['temporal_frequency'].dropna().unique())}")
print(f"Contrasts: {sorted(epochs_df['contrast'].dropna().unique())}")

io.close()

# Filter to high-contrast trials
epochs_df = epochs_df.dropna(subset=['direction', 'temporal_frequency', 'contrast'])
epochs_hc = epochs_df[epochs_df['contrast'] >= 0.4].copy()
print(f"\nHigh-contrast trials (contrast>=0.4): {len(epochs_hc)}")
print(f"Direction counts:\n{epochs_hc['direction'].value_counts().sort_index().to_dict()}")

# --- DIAGNOSTIC: check a few trial windows ---
print("\n--- DIAGNOSTIC: First 5 trial windows vs DFF timestamps ---")
for i, (_, trial) in enumerate(epochs_hc.head(5).iterrows()):
    t_start = trial['start_time']
    t_stop  = trial['stop_time']
    # How many DFF points fall in this window?
    mask = (timestamps >= t_start) & (timestamps <= t_stop)
    # Index-based
    idx_s = np.searchsorted(timestamps, t_start)
    idx_e = np.searchsorted(timestamps, t_stop)
    print(f"  Trial {i}: [{t_start:.3f}, {t_stop:.3f}]s "
          f"mask_count={mask.sum()} idx_range=[{idx_s},{idx_e}] "
          f"ts_at_idx={timestamps[idx_s]:.3f}")

# --- Compute neural responses using searchsorted (robust) ---
N_NEURONS = F.shape[1]
RESP_DELAY = 0.1   # 100ms delay
RESP_WINDOW = 0.5  # 500ms window
N_FRAMES = max(1, int(round(RESP_WINDOW * fs)))  # expected frames in window

print(f"\n--- Extracting neural responses (searchsorted method) ---")
print(f"Response window: +{RESP_DELAY*1000:.0f}ms, duration {RESP_WINDOW*1000:.0f}ms ({N_FRAMES} frames expected)")

responses_by_dir = {}
n_valid = 0
n_skip = 0
for _, trial in epochs_hc.iterrows():
    t_resp_start = trial['start_time'] + RESP_DELAY
    # Use searchsorted to find nearest frame
    idx_start = np.searchsorted(timestamps, t_resp_start)
    idx_end   = idx_start + N_FRAMES
    if idx_start >= len(timestamps) or idx_end > len(timestamps):
        n_skip += 1
        continue
    # Sanity check: make sure we're actually near the trial time
    time_error = abs(timestamps[idx_start] - t_resp_start)
    if time_error > 2.0:  # more than 2 seconds off = wrong
        n_skip += 1
        continue
    mean_resp = F[idx_start:idx_end].mean(axis=0)  # [N_neurons]
    direction = int(trial['direction'])
    if direction not in responses_by_dir:
        responses_by_dir[direction] = []
    responses_by_dir[direction].append(mean_resp)
    n_valid += 1

print(f"Valid trials extracted: {n_valid}, skipped: {n_skip}")

directions = sorted(responses_by_dir.keys())
K = len(directions)
print(f"Directions with responses: {K} ({directions})")
for d_dir in directions:
    print(f"  Direction {d_dir:>3}: {len(responses_by_dir[d_dir])} trials")

if K < 3:
    print("ERROR: Too few directions, cannot proceed. Check data.")
    import sys; sys.exit(1)

# Build data matrix
X_all = []
y_all = []
dir_to_idx = {d: i for i, d in enumerate(directions)}

for direction, resps in responses_by_dir.items():
    for r in resps:
        X_all.append(r)
        y_all.append(dir_to_idx[direction])

X = np.array(X_all)   # [N_trials, N_neurons]
y = np.array(y_all)
print(f"\nTotal trials: {len(X)}, Neurons: {N_NEURONS}")
print(f"X range: [{X.min():.4f}, {X.max():.4f}], mean={X.mean():.4f}")

# --- CTI Law Analysis ---
d = N_NEURONS
print(f"\n--- CTI Law Analysis (K={K} directions, d={d} neurons) ---")

# Compute centroids and sigma_W
centroids = np.array([X[y == dir_to_idx[di]].mean(axis=0) for di in directions])
per_class_vars = [np.var(X[y == dir_to_idx[di]], axis=0).mean() for di in directions]
sigma_W = np.sqrt(np.mean(per_class_vars))
print(f"sigma_W = {sigma_W:.6f}")
print(f"Centroids shape: {centroids.shape}")
print(f"Centroid range: [{centroids.min():.4f}, {centroids.max():.4f}]")

# kappa_nearest for each direction
kappas = []
nearest_dirs = []
for i, di in enumerate(directions):
    dists = np.linalg.norm(centroids - centroids[i], axis=1)
    dists[i] = np.inf
    min_dist = dists.min()
    min_j = dists.argmin()
    kappas.append(min_dist / (sigma_W * np.sqrt(d)))
    nearest_dirs.append(directions[min_j])
kappas = np.array(kappas)
print(f"kappa range: [{kappas.min():.4f}, {kappas.max():.4f}], mean={kappas.mean():.4f}")

# Leave-one-out 1-NN accuracy per direction (proper CV, avoids trivial self-prediction)
# For each sample, train on all OTHER samples, predict this one
print("\nComputing leave-one-out 1-NN accuracy...")
preds_loo = np.zeros(len(X), dtype=int)
for i in range(len(X)):
    mask_train = np.ones(len(X), dtype=bool)
    mask_train[i] = False
    X_train, y_train = X[mask_train], y[mask_train]
    X_test = X[i:i+1]
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(X_train, y_train)
    preds_loo[i] = knn.predict(X_test)[0]

per_dir_acc = np.array([(preds_loo[y == dir_to_idx[di]] == dir_to_idx[di]).mean()
                         for di in directions])
print(f"Per-direction LOO accuracy: {dict(zip(directions, per_dir_acc.round(3)))}")
print(f"Mean LOO accuracy: {per_dir_acc.mean():.4f}")

# Normalize: q_norm = (q - 1/K) / (1 - 1/K)
q_norm = (per_dir_acc - 1/K) / (1 - 1/K)
q_norm = np.clip(q_norm, 0.01, 0.99)
logit_q = logit(q_norm)

print(f"\nq_norm: {q_norm.round(3)}")
print(f"logit_q: {logit_q.round(3)}")
print(f"kappas:  {kappas.round(3)}")

# Fit: logit(q_norm) = alpha * kappa + C
if len(np.unique(logit_q)) < 2:
    print("WARNING: All logit_q values identical — cannot fit. All accuracies the same.")
    slope = float('nan'); intercept = float('nan')
    r_pearson = float('nan'); p_pearson = float('nan')
else:
    slope, intercept, r, p, se = linregress(kappas, logit_q)
    r_pearson, p_pearson = pearsonr(kappas, logit_q)

print(f"\n=== CTI Law Fit: Biological Neurons (V1, drifting gratings) ===")
print(f"N = {K} directions, n_trials = {len(X)}")
print(f"alpha = {slope:.4f} (NLP reference: 1.477)")
print(f"C     = {intercept:.4f}")
print(f"Pearson r = {r_pearson:.4f} (p={p_pearson:.4f})")
print(f"R^2   = {r_pearson**2:.4f}")
print()
print(f"Comparison:")
print(f"  NLP 12-arch: r^2=0.955, alpha=1.477")
print(f"  ViT CIFAR-10: r^2=0.964, alpha=0.63")
print(f"  ResNet50 CIFAR-100 K=100: r=0.749")
print(f"  V1 BIOLOGICAL (K={K}): r={r_pearson:.4f}, alpha={slope:.4f}")
print()

if slope > 0 and not np.isnan(slope):
    d_eff_implied = float((slope/1.128)**2)
    print(f"Theory: alpha = sqrt(4/pi) * sqrt(d_eff) => d_eff = (alpha/sqrt(4/pi))^2")
    print(f"For biological V1 neurons (d={d} neurons): d_eff_implied = {d_eff_implied:.4f}")
else:
    d_eff_implied = None

# Per-direction details
print("\nPer-direction breakdown:")
print(f"{'Direction':>10} {'kappa':>8} {'nearest':>8} {'acc_loo':>8} {'q_norm':>8} {'logit_q':>8}")
for i, di in enumerate(directions):
    print(f"{di:>10} {kappas[i]:>8.4f} {nearest_dirs[i]:>8} "
          f"{per_dir_acc[i]:>8.3f} {q_norm[i]:>8.4f} {logit_q[i]:>8.4f}")

# Save results
out = {
    'experiment': 'cti_neuroscience_v1',
    'description': 'CTI Law: Biological V1 neurons (two-photon calcium imaging, drifting gratings)',
    'dataset': 'DANDI:000039',
    'session': 'sub-673647168_ses-698273664',
    'modality': 'Two-photon calcium imaging, mouse visual cortex V1',
    'organism': 'Mus musculus',
    'stimulus': 'Drifting gratings, 8 directions, high-contrast (>=0.4)',
    'method': 'LOO-1NN (leave-one-out cross-validation)',
    'K': K,
    'directions': directions,
    'N_neurons': N_NEURONS,
    'N_trials': len(X),
    'trials_per_direction': {str(d_dir): len(responses_by_dir[d_dir]) for d_dir in directions},
    'sigma_W': float(sigma_W),
    'kappa_values': kappas.tolist(),
    'per_direction_accuracy': per_dir_acc.tolist(),
    'alpha': float(slope) if not np.isnan(slope) else None,
    'intercept': float(intercept) if not np.isnan(intercept) else None,
    'pearson_r': float(r_pearson) if not np.isnan(r_pearson) else None,
    'pearson_r2': float(r_pearson**2) if not np.isnan(r_pearson) else None,
    'p_value': float(p_pearson) if not np.isnan(p_pearson) else None,
    'comparison': {
        'NLP_12arch_r2': 0.955,
        'NLP_12arch_alpha': 1.477,
        'ViT_CIFAR10_r2': 0.964,
        'CNN_CIFAR100_r': 0.749,
    },
    'd_eff_implied': d_eff_implied,
}
with open('results/cti_neuroscience_v1.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/cti_neuroscience_v1.json")
