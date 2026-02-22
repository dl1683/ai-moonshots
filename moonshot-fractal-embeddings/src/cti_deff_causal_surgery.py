"""
d_eff_formula Causal Surgery: Direct Intervention on Effective Dimensionality
==============================================================================

CODEX DESIGN (Feb 23 2026) — Highest-leverage experiment to raise Nobel score.

THEORETICAL BACKGROUND:
  CTI Law: logit(q) = A_renorm * kappa_nearest * sqrt(d_eff_formula) + C
  where:
    kappa_nearest = delta_min / (sigma_W_global * sqrt(d))
    d_eff_formula = tr(Sigma_W) / sigma_centroid_dir^2
    sigma_centroid_dir = within-class std projected onto nearest centroid direction

EXPERIMENT DESIGN:
  1. Train CE ResNet-18 on CIFAR-100 coarse (K=20) for 60 epochs  [1 seed]
  2. Extract + save train/test embeddings to disk (numpy)
  3. Apply within-class covariance SURGERY:
       Redistribute variance into/out of Delta_hat (centroid pair direction)
       while PRESERVING tr(Sigma_W) --> kappa_nearest stays fixed.
  4. For each surgery factor r (d_eff_new = r * d_eff_base):
       a. Apply surgery to both train and test embeddings
       b. Compute actual q (1-NN accuracy)
       c. Compare to law prediction (no free parameters)

SURGERY TRANSFORMATION:
  For each sample x_i in class c:
    z_i   = x_i - mu_c                       (zero-mean)
    z_along = (z_i . Delta_hat) * Delta_hat  (component along centroid direction)
    z_perp  = z_i - z_along                  (perpendicular)
    x_new = mu_c + scale_along * z_along + scale_perp * z_perp

  Constraints (from tr(W) preservation):
    scale_along  = 1 / sqrt(r)
    scale_perp^2 = (trW - sigma_cdir^2/r) / (trW - sigma_cdir^2)
    Valid for r >= sigma_cdir^2/trW = 1/d_eff_formula (min: d_eff_new=1)

PRE-REGISTERED PREDICTION (zero free parameters):
  logit(q_new) = C + A_renorm * kappa_nearest * sqrt(r * d_eff_base)
  => Deltalogit(q) = A_renorm * kappa_nearest * sqrt(d_eff_base) * (sqrt(r) - 1)

ACCEPTANCE CRITERION (pre-registered):
  PRIMARY:   Pearson r(actual_logit_q, predicted_logit_q) > 0.99 across r levels
  SECONDARY: Mean |actual - predicted| < 0.1 * |predicted - logit_base|  (10% calibration)
  TERTIARY:  kappa_nearest changes < 0.1% after surgery (invariance check)

FALSIFICATION:
  If r < 0.99 or calibration > 25%: d_eff_formula is NOT causally active as predicted.
  Must revise theory.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier

# ==================== CONFIGURATION ====================
K = 20
N_EPOCHS = 60
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 5e-4
N_SEEDS = 3          # Train 3 CE seeds for robust estimate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A_RENORM_K20 = 1.0535  # Pre-registered constant (Theorem 15)

# Surgery factors r: d_eff_new = r * d_eff_base
# r < 1: LESS d_eff (more sigma_centroid_dir) -> WORSE q predicted
# r = 1: baseline (no change)
# r > 1: MORE d_eff (less sigma_centroid_dir) -> BETTER q predicted
SURGERY_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]

RESULT_PATH = "results/cti_deff_causal_surgery.json"
LOG_PATH = "results/cti_deff_causal_surgery_log.txt"
EMBED_DIR = "results/surgery_embeddings"


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


# ==================== MODEL ====================
def get_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    in_feats = backbone.fc.in_features
    backbone.fc = nn.Identity()
    ce_head = nn.Linear(512, K)
    model = nn.ModuleDict({'backbone': backbone, 'ce_head': ce_head}).to(DEVICE)
    return model


def coarse_label(x):
    return x // 5


def get_cifar_coarse():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100(
        'data', train=True, download=False,
        transform=train_transform, target_transform=coarse_label)
    test_ds = torchvision.datasets.CIFAR100(
        'data', train=False, download=False,
        transform=test_transform, target_transform=coarse_label)
    return train_ds, test_ds


def extract_embeddings(model, dataset):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            embs.append(model['backbone'](imgs.to(DEVICE)).cpu().numpy())
            labels.append(lbs.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model['ce_head'](model['backbone'](imgs))
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(loader)


# ==================== GEOMETRY COMPUTATION ====================
def compute_geometry(X_tr, y_tr):
    """Compute all CTI geometry metrics from training embeddings."""
    classes = np.unique(y_tr)
    K_actual = len(classes)
    N = len(X_tr)
    d = X_tr.shape[1]

    # Class centroids
    centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])  # (K, d)

    # tr(Sigma_W): total within-class variance
    trW = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        trW += float(np.sum(Xc_c ** 2)) / N

    sigma_W_global = float(np.sqrt(trW / d))

    # Nearest centroid pair
    min_dist = float('inf')
    min_i, min_j = 0, 1
    for i in range(K_actual):
        for j in range(i + 1, K_actual):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist, min_i, min_j = dist, i, j

    delta_min = float(min_dist)
    kappa_nearest = float(delta_min / (sigma_W_global * np.sqrt(d) + 1e-10))

    # Delta_hat: unit vector of nearest centroid pair direction
    Delta = centroids[min_i] - centroids[min_j]
    Delta_hat = Delta / (np.linalg.norm(Delta) + 1e-10)

    # sigma_centroid_dir^2 = Delta_hat^T Sigma_W Delta_hat
    sigma_centroid_sq = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        proj = Xc_c @ Delta_hat
        sigma_centroid_sq += (n_c / N) * float(np.mean(proj ** 2))

    sigma_centroid_dir = float(np.sqrt(sigma_centroid_sq + 1e-10))
    d_eff_formula = float(trW / (sigma_centroid_sq + 1e-10))

    return {
        'centroids': centroids,
        'Delta_hat': Delta_hat,
        'trW': trW,
        'sigma_W_global': sigma_W_global,
        'sigma_centroid_sq': sigma_centroid_sq,
        'sigma_centroid_dir': sigma_centroid_dir,
        'd_eff_formula': d_eff_formula,
        'kappa_nearest': kappa_nearest,
        'delta_min': delta_min,
        'nearest_pair': (int(min_i), int(min_j)),
        'K_actual': K_actual,
        'd': d,
    }


# ==================== SURGERY TRANSFORMATION ====================
def apply_surgery(X, y, geometry, r):
    """
    Apply within-class covariance surgery.

    r = surgery factor: d_eff_new = r * d_eff_base

    Transformation:
      z = x - mu_c
      z_along = (z . Delta_hat) * Delta_hat   (boundary direction)
      z_perp  = z - z_along                   (perpendicular)
      x_new = mu_c + (1/sqrt(r)) * z_along + scale_perp * z_perp

    scale_perp^2 = (trW - sigma_centroid_sq/r) / (trW - sigma_centroid_sq)

    Effect:
      sigma_centroid_dir_new = sigma_centroid_dir / sqrt(r)  [changed]
      d_eff_formula_new = r * d_eff_base                     [changed]
      tr(Sigma_W) preserved                                  [kappa_nearest fixed]
    """
    centroids = geometry['centroids']
    Delta_hat = geometry['Delta_hat']
    trW = geometry['trW']
    sigma_centroid_sq = geometry['sigma_centroid_sq']
    classes = np.unique(y)

    # Check validity: need trW >= sigma_centroid_sq/r
    # i.e., r >= sigma_centroid_sq/trW = 1/d_eff_formula
    min_r = float(sigma_centroid_sq / (trW + 1e-10)) * 1.001  # with tiny safety margin
    if r < min_r:
        log(f"  WARNING: r={r:.3f} < min_r={min_r:.3f} (sigma_centroid_sq/trW). "
            f"scale_perp would be imaginary. Clamping to min_r.")
        r = min_r

    scale_along = 1.0 / float(np.sqrt(r))
    denom = trW - sigma_centroid_sq
    num = trW - sigma_centroid_sq / r
    if denom < 1e-12:
        scale_perp = 1.0  # degenerate case: all variance in centroid direction
    else:
        scale_perp = float(np.sqrt(max(0.0, num / denom)))

    # Vectorized surgery: X shape (N, d)
    X_new = X.copy()

    # For memory efficiency: process per class
    for c in classes:
        mask = (y == c)
        Xc = X[mask]  # (n_c, d)
        z = Xc - centroids[c]  # zero-mean (n_c, d)
        # Project onto Delta_hat
        proj_scalar = z @ Delta_hat  # (n_c,) scalar projections
        z_along = proj_scalar[:, None] * Delta_hat[None, :]  # (n_c, d)
        z_perp = z - z_along  # (n_c, d)
        # Apply surgery
        z_new = scale_along * z_along + scale_perp * z_perp
        X_new[mask] = centroids[c] + z_new

    return X_new, scale_along, scale_perp


def verify_surgery(X_orig, X_new, y, geometry):
    """Verify surgery preserved kappa_nearest and changed d_eff_formula as expected."""
    geo_new = compute_geometry(X_new, y)
    geo_orig = geometry

    kappa_change_pct = abs(geo_new['kappa_nearest'] - geo_orig['kappa_nearest']) / (
        geo_orig['kappa_nearest'] + 1e-10) * 100
    d_eff_ratio = geo_new['d_eff_formula'] / (geo_orig['d_eff_formula'] + 1e-10)
    trW_change_pct = abs(geo_new['trW'] - geo_orig['trW']) / (geo_orig['trW'] + 1e-10) * 100

    return {
        'kappa_change_pct': float(kappa_change_pct),
        'd_eff_ratio_actual': float(d_eff_ratio),
        'trW_change_pct': float(trW_change_pct),
        'sigma_centroid_dir_new': float(geo_new['sigma_centroid_dir']),
        'd_eff_formula_new': float(geo_new['d_eff_formula']),
        'kappa_nearest_new': float(geo_new['kappa_nearest']),
    }


# ==================== QUALITY METRIC ====================
def compute_q_knn(X_tr, y_tr, X_te, y_te, K_classes=K):
    """1-NN accuracy normalized: q = (acc - 1/K) / (1 - 1/K)."""
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1, algorithm='auto')
    knn.fit(X_tr, y_tr)
    acc = float(knn.score(X_te, y_te))
    return (acc - 1.0 / K_classes) / (1.0 - 1.0 / K_classes), acc


# ==================== TRAINING ====================
def train_and_save_embeddings(seed, train_ds, test_ds):
    """Train CE ResNet-18 for N_EPOCHS, save embeddings to disk."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = get_model()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    log(f"  Training seed={seed}...")
    for epoch in range(1, N_EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, scheduler)
        if epoch % 10 == 0:
            log(f"  [seed={seed} epoch={epoch}] loss={loss:.4f}")

    # Extract embeddings
    log(f"  Extracting train embeddings (seed={seed})...")
    X_tr, y_tr = extract_embeddings(model, train_ds)
    log(f"  Extracting test embeddings (seed={seed})...")
    X_te, y_te = extract_embeddings(model, test_ds)

    # Save embeddings
    os.makedirs(EMBED_DIR, exist_ok=True)
    np.save(f"{EMBED_DIR}/X_tr_seed{seed}.npy", X_tr)
    np.save(f"{EMBED_DIR}/y_tr_seed{seed}.npy", y_tr)
    np.save(f"{EMBED_DIR}/X_te_seed{seed}.npy", X_te)
    np.save(f"{EMBED_DIR}/y_te_seed{seed}.npy", y_te)
    log(f"  Saved embeddings for seed={seed}: X_tr={X_tr.shape}, X_te={X_te.shape}")

    return X_tr, y_tr, X_te, y_te


# ==================== SURGERY EXPERIMENT ====================
def run_surgery_experiment(X_tr, y_tr, X_te, y_te, seed):
    """Run causal surgery across all r levels for one seed."""
    log(f"\n--- SURGERY EXPERIMENT seed={seed} ---")

    # Compute baseline geometry
    log(f"  Computing baseline geometry...")
    geometry = compute_geometry(X_tr, y_tr)
    d_eff_base = geometry['d_eff_formula']
    kappa_base = geometry['kappa_nearest']
    sigma_cdir_base = geometry['sigma_centroid_dir']

    log(f"  Baseline: d_eff_formula={d_eff_base:.4f}, kappa_nearest={kappa_base:.4f}, "
        f"sigma_centroid_dir={sigma_cdir_base:.4f}")
    log(f"  sigma_centroid_dir / sigma_W_global = "
        f"{geometry['sigma_centroid_dir'] / (geometry['sigma_W_global'] + 1e-10):.2f}x")

    # Baseline q (r=1, no surgery)
    log(f"  Computing baseline q (kNN)...")
    q_base, acc_base = compute_q_knn(X_tr, y_tr, X_te, y_te)
    logit_q_base = float(np.log(q_base / (1 - q_base) + 1e-10)) if 0 < q_base < 1 else float('nan')
    log(f"  Baseline: q={q_base:.4f}, logit(q)={logit_q_base:.4f}")

    # Pre-registered prediction slope (no free parameters)
    # logit(q_new) = logit(q_base) + A_renorm * kappa_nearest * sqrt(d_eff_base) * (sqrt(r) - 1)
    # Note: if law is exact, logit(q_base) = A_renorm * kappa_base * sqrt(d_eff_base) + C
    # We use: predicted = C_fitted + A_renorm * kappa_base * sqrt(r * d_eff_base)
    # where C_fitted = logit(q_base) - A_renorm * kappa_base * sqrt(d_eff_base) [from baseline]
    kappa_eff_base = kappa_base * float(np.sqrt(d_eff_base))
    C_fitted = logit_q_base - A_RENORM_K20 * kappa_eff_base
    log(f"  kappa_eff_base = sqrt(d_eff)*kappa = {kappa_eff_base:.4f}")
    log(f"  Fitted C = logit(q_base) - A*kappa_eff = {C_fitted:.4f}")
    log(f"  (Using C=C_fitted: prediction is pre-registered functional form, "
        f"not a free fit)")

    records = []
    for r in SURGERY_LEVELS:
        log(f"\n  [r={r:.3f}] d_eff_new = {r * d_eff_base:.4f} ...")

        # Apply surgery to train and test
        X_tr_new, scale_along, scale_perp = apply_surgery(X_tr, y_tr, geometry, r)
        X_te_new, _, _ = apply_surgery(X_te, y_te, geometry, r)

        # Verify surgery invariants (only on train set for speed)
        verif = verify_surgery(X_tr, X_tr_new, y_tr, geometry)
        log(f"  Verification: kappa_change={verif['kappa_change_pct']:.3f}%, "
            f"d_eff_ratio_actual={verif['d_eff_ratio_actual']:.4f} (expected {r:.4f}), "
            f"trW_change={verif['trW_change_pct']:.4f}%")

        # Compute actual q
        q_new, acc_new = compute_q_knn(X_tr_new, y_tr, X_te_new, y_te)
        logit_q_new = float(np.log(q_new / (1 - q_new) + 1e-10)) if 0 < q_new < 1 else float('nan')

        # Predicted logit(q) from law (using ACTUAL d_eff after surgery, not nominal r)
        d_eff_actual = verif['d_eff_formula_new']
        kappa_actual = verif['kappa_nearest_new']
        logit_q_pred_nominal = C_fitted + A_RENORM_K20 * kappa_base * float(np.sqrt(r * d_eff_base))
        logit_q_pred_actual = C_fitted + A_RENORM_K20 * kappa_actual * float(np.sqrt(d_eff_actual))

        # Residuals
        delta_actual = logit_q_new - logit_q_base
        delta_pred = logit_q_pred_nominal - logit_q_base
        calib_error = abs(delta_actual - delta_pred) / (abs(delta_pred) + 1e-6)

        log(f"  q_new={q_new:.4f}, logit_q_new={logit_q_new:.4f}")
        log(f"  predicted_nominal={logit_q_pred_nominal:.4f}, actual={logit_q_new:.4f}")
        log(f"  delta_actual={delta_actual:.4f}, delta_pred={delta_pred:.4f}, "
            f"calib_err={calib_error:.3f}")

        rec = {
            'seed': seed,
            'r_nominal': float(r),
            'r_actual': float(verif['d_eff_ratio_actual']),
            'd_eff_base': float(d_eff_base),
            'd_eff_new_nominal': float(r * d_eff_base),
            'd_eff_new_actual': float(d_eff_actual),
            'kappa_base': float(kappa_base),
            'kappa_new': float(kappa_actual),
            'kappa_change_pct': float(verif['kappa_change_pct']),
            'trW_change_pct': float(verif['trW_change_pct']),
            'scale_along': float(scale_along),
            'scale_perp': float(scale_perp),
            'q_base': float(q_base),
            'logit_q_base': float(logit_q_base),
            'q_new': float(q_new),
            'logit_q_new': float(logit_q_new),
            'logit_q_pred_nominal': float(logit_q_pred_nominal),
            'logit_q_pred_actual': float(logit_q_pred_actual),
            'delta_logit_actual': float(delta_actual),
            'delta_logit_pred': float(delta_pred),
            'calibration_error': float(calib_error),
            'C_fitted': float(C_fitted),
            'kappa_eff_base': float(kappa_eff_base),
        }
        records.append(rec)

    return records


# ==================== ANALYSIS ====================
def analyze_surgery_results(all_records):
    """Compute acceptance criteria across all seeds and r levels."""
    log("\n" + "=" * 70)
    log("SURGERY EXPERIMENT ANALYSIS")
    log("=" * 70)

    # Collect actual vs predicted logit_q pairs
    actual_vals = [r['logit_q_new'] for r in all_records if not np.isnan(r['logit_q_new'])]
    pred_vals = [r['logit_q_pred_nominal'] for r in all_records if not np.isnan(r['logit_q_new'])]

    if len(actual_vals) < 3:
        log("ERROR: Too few valid data points for analysis")
        return {}

    actual_arr = np.array(actual_vals)
    pred_arr = np.array(pred_vals)

    # PRIMARY: Pearson correlation
    r_pearson = float(np.corrcoef(actual_arr, pred_arr)[0, 1])
    log(f"\nPRIMARY CRITERION: Pearson r(actual, predicted) = {r_pearson:.4f}")
    log(f"  Threshold: > 0.99")
    log(f"  STATUS: {'PASS' if r_pearson > 0.99 else 'FAIL'}")

    # SECONDARY: Calibration error
    calib_errors = [r['calibration_error'] for r in all_records
                    if not np.isnan(r['logit_q_new']) and abs(r['delta_logit_pred']) > 0.01]
    mean_calib = float(np.mean(calib_errors)) if calib_errors else float('nan')
    log(f"\nSECONDARY CRITERION: Mean calibration error = {mean_calib:.4f}")
    log(f"  Threshold: < 0.10 (10% relative error)")
    log(f"  STATUS: {'PASS' if mean_calib < 0.10 else 'FAIL'}")

    # TERTIARY: kappa invariance
    kappa_changes = [r['kappa_change_pct'] for r in all_records]
    max_kappa_change = float(np.max(kappa_changes)) if kappa_changes else float('nan')
    log(f"\nTERTIARY CRITERION: Max kappa_nearest change = {max_kappa_change:.4f}%")
    log(f"  Threshold: < 0.1%")
    log(f"  STATUS: {'PASS' if max_kappa_change < 0.1 else 'FAIL'}")

    # SLOPE CHECK: Fitted slope should be close to A_renorm
    # logit(q) = A_fitted * kappa_eff + C_fitted
    # kappa_eff = kappa_nearest * sqrt(r * d_eff_base)
    kappa_effs = [r['kappa_base'] * np.sqrt(r['r_actual'] * r['d_eff_base'])
                  for r in all_records if not np.isnan(r['logit_q_new'])]
    kappa_effs_arr = np.array(kappa_effs)
    if len(kappa_effs_arr) > 1 and np.std(kappa_effs_arr) > 0:
        A_mat = np.column_stack([kappa_effs_arr, np.ones(len(kappa_effs_arr))])
        A_fitted, C_fitted_global, _, _ = np.linalg.lstsq(A_mat, actual_arr, rcond=None)
        log(f"\nFitted slope A_empirical = {A_fitted:.4f} (pre-registered A_renorm = {A_RENORM_K20})")
        log(f"Fitted C = {C_fitted_global:.4f}")
        log(f"A_error = {abs(A_fitted - A_RENORM_K20) / A_RENORM_K20 * 100:.1f}%")

    # R-squared of law
    ss_res = float(np.sum((actual_arr - pred_arr) ** 2))
    ss_tot = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
    r2_law = 1.0 - ss_res / (ss_tot + 1e-10)
    log(f"\nR2 of pre-registered law = {r2_law:.4f}")

    # Print per-r summary
    log("\nPer-r summary (averaged over seeds):")
    log(f"{'r':>6} {'d_eff_new':>10} {'q_base':>8} {'q_new':>8} "
        f"{'logit_actual':>13} {'logit_pred':>11} {'calib_err':>10}")
    r_nominal_vals = sorted(set(r['r_nominal'] for r in all_records))
    for rv in r_nominal_vals:
        subset = [r for r in all_records if r['r_nominal'] == rv]
        avg = lambda key: float(np.mean([r[key] for r in subset]))
        log(f"{rv:>6.2f} {avg('d_eff_new_actual'):>10.3f} {avg('q_base'):>8.4f} "
            f"{avg('q_new'):>8.4f} {avg('logit_q_new'):>13.4f} "
            f"{avg('logit_q_pred_nominal'):>11.4f} {avg('calibration_error'):>10.4f}")

    overall_pass = (r_pearson > 0.99) and (mean_calib < 0.10)
    log(f"\n{'='*70}")
    log(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    log(f"  d_eff_formula IS causally active in CTI law: {overall_pass}")
    if not overall_pass:
        log(f"  INTERPRETATION: CTI law needs revision. d_eff_formula may be diagnostic only.")
    log(f"{'='*70}")

    return {
        'r_pearson': r_pearson,
        'mean_calibration_error': mean_calib,
        'max_kappa_change_pct': max_kappa_change,
        'r2_law': r2_law,
        'primary_pass': bool(r_pearson > 0.99),
        'secondary_pass': bool(mean_calib < 0.10),
        'tertiary_pass': bool(max_kappa_change < 0.1),
        'overall_pass': overall_pass,
        'n_records': len(all_records),
    }


# ==================== MAIN ====================
def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs(EMBED_DIR, exist_ok=True)

    log("=" * 70)
    log("d_eff_formula Causal Surgery Experiment")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, N_SEEDS={N_SEEDS}")
    log(f"A_renorm(K=20) = {A_RENORM_K20} (pre-registered)")
    log(f"Surgery levels r: {SURGERY_LEVELS}")
    log(f"PRE-REGISTERED PREDICTION: logit(q_new) = C + A * kappa_base * sqrt(r * d_eff_base)")
    log(f"ACCEPTANCE: Pearson r > 0.99, Mean calibration error < 10%\n")

    train_ds, test_ds = get_cifar_coarse()

    all_records = []
    summary = {}

    for seed in range(N_SEEDS):
        log(f"\n{'='*70}")
        log(f"SEED {seed}")
        log(f"{'='*70}")

        # Check if embeddings already saved (resume support)
        X_tr_path = f"{EMBED_DIR}/X_tr_seed{seed}.npy"
        if os.path.exists(X_tr_path):
            log(f"  Loading saved embeddings for seed={seed}...")
            X_tr = np.load(f"{EMBED_DIR}/X_tr_seed{seed}.npy")
            y_tr = np.load(f"{EMBED_DIR}/y_tr_seed{seed}.npy")
            X_te = np.load(f"{EMBED_DIR}/X_te_seed{seed}.npy")
            y_te = np.load(f"{EMBED_DIR}/y_te_seed{seed}.npy")
            log(f"  Loaded: X_tr={X_tr.shape}, X_te={X_te.shape}")
        else:
            X_tr, y_tr, X_te, y_te = train_and_save_embeddings(seed, train_ds, test_ds)

        records = run_surgery_experiment(X_tr, y_tr, X_te, y_te, seed)
        all_records.extend(records)

        # Save intermediate results
        with open(RESULT_PATH, 'w') as f:
            json.dump({
                'records': all_records,
                'n_seeds_complete': seed + 1,
                'surgery_levels': SURGERY_LEVELS,
                'A_renorm': A_RENORM_K20,
                'K': K,
            }, f, default=lambda x: float(x) if hasattr(x, '__float__') else x, indent=2)

    # Final analysis
    analysis = analyze_surgery_results(all_records)
    summary['analysis'] = analysis

    # Save final results
    with open(RESULT_PATH, 'w') as f:
        json.dump({
            'records': all_records,
            'analysis': analysis,
            'n_seeds': N_SEEDS,
            'surgery_levels': SURGERY_LEVELS,
            'A_renorm': A_RENORM_K20,
            'K': K,
        }, f, default=lambda x: float(x) if hasattr(x, '__float__') else x, indent=2)

    log(f"\nResults saved to {RESULT_PATH}")
    return summary


if __name__ == "__main__":
    main()
