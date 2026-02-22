"""
Linear-Regime Causal Surgery: Test CTI Law in the Proper Regime
================================================================

PROBLEM WITH ORIGINAL SURGERY:
  At epoch 60, kappa_eff = kappa * sqrt(d_eff) = 1.45 * sqrt(37.7) = 8.93.
  This is deep in the saturation regime. The linear law
  logit(q) = A * kappa * sqrt(d_eff) + C only holds for kappa_eff ~ 0.5-2.0.

SOLUTION:
  Find the training checkpoint where kappa_eff is in the linear regime [0.5, 2.0].
  Apply surgery at THAT checkpoint. This is the proper test.

DESIGN:
  1. Train ResNet-18 on CIFAR-100 coarse (K=20) for 60 epochs.
  2. Every 5 epochs, extract embeddings and compute:
     - kappa_nearest, d_eff_formula, kappa_eff = kappa * sqrt(d_eff)
  3. After training, find the checkpoint(s) with kappa_eff in [0.5, 2.0].
  4. Apply causal surgery at the selected checkpoint.
  5. Pre-registered prediction: logit(q_new) = C + A * kappa_base * sqrt(r * d_eff_base)
     Acceptance: Pearson r > 0.99, Mean calibration error < 10%

PRE-REGISTERED CONSTANTS:
  A_renorm(K=20) = 1.0535
  Linear regime: kappa_eff in [0.5, 2.0]
  Target checkpoint: argmin |kappa_eff - 1.0| subject to kappa_eff in [0.5, 2.0]
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier

# ==================== CONFIGURATION ====================
K = 20
N_EPOCHS = 60
CHECKPOINT_EVERY = 5   # Save embeddings every N epochs
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 5e-4
N_SEEDS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A_RENORM_K20 = 1.0535  # Pre-registered

# Linear regime target
KAPPA_EFF_TARGET = 1.0     # Ideal kappa_eff for surgery
KAPPA_EFF_MIN = 0.5
KAPPA_EFF_MAX = 2.0

# Surgery factors (same as original)
SURGERY_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]

RESULT_PATH = "results/cti_linear_regime_surgery.json"
LOG_PATH = "results/cti_linear_regime_surgery_log.txt"
EMBED_DIR = "results/linear_regime_surgery_embeddings"

# Clear old log on startup
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


# ==================== MODEL ====================
def get_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
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
        dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)
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


# ==================== GEOMETRY ====================
def compute_geometry(X_tr, y_tr):
    """Compute CTI geometry: kappa_nearest, d_eff_formula, kappa_eff."""
    classes = np.unique(y_tr)
    K_actual = len(classes)
    N = len(X_tr)
    d = X_tr.shape[1]

    centroids = np.stack([X_tr[y_tr == c].mean(0) for c in classes])

    trW = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c] - centroids[c]
        trW += float(np.sum(Xc ** 2)) / N

    sigma_W_global = float(np.sqrt(trW / d))

    min_dist = float('inf')
    min_i, min_j = 0, 1
    for i in range(K_actual):
        for j in range(i + 1, K_actual):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist, min_i, min_j = dist, i, j

    delta_min = float(min_dist)
    kappa_nearest = float(delta_min / (sigma_W_global * np.sqrt(d) + 1e-10))

    Delta = centroids[min_i] - centroids[min_j]
    Delta_hat = Delta / (np.linalg.norm(Delta) + 1e-10)

    sigma_centroid_sq = 0.0
    for c in classes:
        Xc = X_tr[y_tr == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        proj = Xc_c @ Delta_hat
        sigma_centroid_sq += (n_c / N) * float(np.mean(proj ** 2))

    d_eff_formula = float(trW / (sigma_centroid_sq + 1e-10))
    kappa_eff = kappa_nearest * float(np.sqrt(d_eff_formula))

    return {
        'centroids': centroids,
        'Delta_hat': Delta_hat,
        'trW': trW,
        'sigma_W_global': sigma_W_global,
        'sigma_centroid_sq': sigma_centroid_sq,
        'd_eff_formula': d_eff_formula,
        'kappa_nearest': kappa_nearest,
        'kappa_eff': kappa_eff,
        'delta_min': delta_min,
        'nearest_pair': (int(min_i), int(min_j)),
    }


def compute_q(X_tr, y_tr, X_te, y_te):
    """1-NN normalized accuracy."""
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1)
    knn.fit(X_tr, y_tr)
    acc = float(knn.score(X_te, y_te))
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    return float(q), float(acc)


# ==================== SURGERY ====================
def apply_surgery(X, y, geometry, r):
    """Redistribute variance: d_eff_new = r * d_eff_base, kappa unchanged."""
    centroids = geometry['centroids']
    Delta_hat = geometry['Delta_hat']
    trW = geometry['trW']
    sigma_centroid_sq = geometry['sigma_centroid_sq']
    classes = np.unique(y)

    min_r = float(sigma_centroid_sq / (trW + 1e-10)) * 1.001
    if r < min_r:
        log(f"    WARNING: r={r:.3f} < min_r={min_r:.4f}. Clamping.")
        r = min_r

    scale_along = 1.0 / float(np.sqrt(r))
    denom = trW - sigma_centroid_sq
    num = trW - sigma_centroid_sq / r
    scale_perp = float(np.sqrt(max(0.0, num / denom))) if denom > 1e-12 else 1.0

    X_new = X.copy()
    for c in classes:
        mask = (y == c)
        z = X[mask] - centroids[c]
        proj = z @ Delta_hat
        z_along = proj[:, None] * Delta_hat[None, :]
        z_perp = z - z_along
        X_new[mask] = centroids[c] + scale_along * z_along + scale_perp * z_perp

    return X_new


def verify_surgery(X_orig, X_new, y, geometry, r_nominal):
    """Verify: kappa unchanged, d_eff changed by factor r."""
    geo_new = compute_geometry(X_new, y)
    kappa_change_pct = abs(geo_new['kappa_nearest'] - geometry['kappa_nearest']) / (
        geometry['kappa_nearest'] + 1e-10) * 100
    d_eff_ratio = geo_new['d_eff_formula'] / (geometry['d_eff_formula'] + 1e-10)
    trW_change_pct = abs(geo_new['trW'] - geometry['trW']) / (geometry['trW'] + 1e-10) * 100
    return {
        'kappa_change_pct': float(kappa_change_pct),
        'd_eff_ratio_actual': float(d_eff_ratio),
        'trW_change_pct': float(trW_change_pct),
        'd_eff_new': float(geo_new['d_eff_formula']),
        'kappa_new': float(geo_new['kappa_nearest']),
    }


# ==================== MAIN TRAINING + ANALYSIS ====================
def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs(EMBED_DIR, exist_ok=True)

    log("=" * 70)
    log("Linear-Regime Causal Surgery Experiment")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, N_SEEDS={N_SEEDS}")
    log(f"A_renorm(K=20) = {A_RENORM_K20} (pre-registered)")
    log(f"Linear regime target: kappa_eff in [{KAPPA_EFF_MIN}, {KAPPA_EFF_MAX}]")
    log(f"Checkpoints every {CHECKPOINT_EVERY} epochs")
    log(f"Surgery levels r: {SURGERY_LEVELS}")

    train_ds, test_ds = get_cifar_coarse()

    all_results = []

    for seed in range(N_SEEDS):
        log(f"\n{'='*70}")
        log(f"SEED {seed}")
        log(f"{'='*70}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        model = get_model()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

        # Training trajectory: record kappa_eff at each checkpoint
        trajectory = []
        best_epoch = None
        best_checkpoint = None  # (X_tr, y_tr, X_te, y_te, geometry, q_base)

        log(f"  Training seed={seed} with trajectory monitoring...")
        for epoch in range(1, N_EPOCHS + 1):
            loss = train_epoch(model, loader, optimizer, scheduler)
            if epoch % 10 == 0:
                log(f"  [seed={seed} epoch={epoch}] loss={loss:.4f}")

            # Save checkpoint
            if epoch % CHECKPOINT_EVERY == 0:
                X_tr, y_tr = extract_embeddings(model, train_ds)
                X_te, y_te = extract_embeddings(model, test_ds)
                geo = compute_geometry(X_tr, y_tr)
                q_base, acc_base = compute_q(X_tr, y_tr, X_te, y_te)
                logit_q_base = float(np.log(q_base / (1 - q_base) + 1e-10))

                traj_entry = {
                    'epoch': epoch,
                    'loss': float(loss),
                    'kappa_nearest': float(geo['kappa_nearest']),
                    'd_eff_formula': float(geo['d_eff_formula']),
                    'kappa_eff': float(geo['kappa_eff']),
                    'q': float(q_base),
                    'logit_q': float(logit_q_base),
                }
                trajectory.append(traj_entry)
                log(f"    [checkpoint ep={epoch}] kappa_eff={geo['kappa_eff']:.4f}, "
                    f"d_eff={geo['d_eff_formula']:.3f}, kappa={geo['kappa_nearest']:.4f}, "
                    f"q={q_base:.4f}")

                # Check if this is in linear regime and closer to target than best so far
                if KAPPA_EFF_MIN <= geo['kappa_eff'] <= KAPPA_EFF_MAX:
                    if best_epoch is None or abs(geo['kappa_eff'] - KAPPA_EFF_TARGET) < abs(
                            best_checkpoint['kappa_eff'] - KAPPA_EFF_TARGET):
                        best_epoch = epoch
                        best_checkpoint = {
                            'X_tr': X_tr.copy(), 'y_tr': y_tr.copy(),
                            'X_te': X_te.copy(), 'y_te': y_te.copy(),
                            'geometry': geo,
                            'q_base': q_base,
                            'logit_q_base': logit_q_base,
                            'kappa_eff': float(geo['kappa_eff']),
                            'epoch': epoch,
                        }

        # Report trajectory
        log(f"\n  Training trajectory (kappa_eff over epochs):")
        for t in trajectory:
            marker = " <-- LINEAR REGIME" if KAPPA_EFF_MIN <= t['kappa_eff'] <= KAPPA_EFF_MAX else ""
            log(f"    epoch={t['epoch']:3d}: kappa_eff={t['kappa_eff']:.4f}, "
                f"d_eff={t['d_eff_formula']:.3f}, q={t['q']:.4f}{marker}")

        if best_epoch is None:
            log(f"\n  WARNING: No checkpoint in linear regime [{KAPPA_EFF_MIN}, {KAPPA_EFF_MAX}]!")
            log(f"  Kappa_eff range: [{min(t['kappa_eff'] for t in trajectory):.4f}, "
                f"{max(t['kappa_eff'] for t in trajectory):.4f}]")
            # Fall back to epoch with kappa_eff closest to target
            best_traj = min(trajectory, key=lambda t: abs(t['kappa_eff'] - KAPPA_EFF_TARGET))
            log(f"  Using closest checkpoint: epoch={best_traj['epoch']}, "
                f"kappa_eff={best_traj['kappa_eff']:.4f}")
            # Re-extract for that epoch (model is at final epoch, need to re-train)
            # Skip surgery for this seed if no linear regime found
            log(f"  Skipping surgery for seed={seed} (no linear regime checkpoint retained).")
            all_results.append({
                'seed': seed, 'status': 'no_linear_regime',
                'trajectory': trajectory,
                'kappa_eff_range': [min(t['kappa_eff'] for t in trajectory),
                                    max(t['kappa_eff'] for t in trajectory)],
            })
            continue

        log(f"\n  Selected checkpoint: epoch={best_epoch}, "
            f"kappa_eff={best_checkpoint['kappa_eff']:.4f}")

        # Save selected checkpoint embeddings
        np.save(f"{EMBED_DIR}/X_tr_seed{seed}_ep{best_epoch}.npy", best_checkpoint['X_tr'])
        np.save(f"{EMBED_DIR}/y_tr_seed{seed}_ep{best_epoch}.npy", best_checkpoint['y_tr'])

        # ============ APPLY SURGERY ============
        log(f"\n  --- SURGERY at epoch={best_epoch}, "
            f"kappa_eff={best_checkpoint['kappa_eff']:.4f} ---")

        geo = best_checkpoint['geometry']
        X_tr = best_checkpoint['X_tr']
        y_tr = best_checkpoint['y_tr']
        X_te = best_checkpoint['X_te']
        y_te = best_checkpoint['y_te']
        q_base = best_checkpoint['q_base']
        logit_q_base = best_checkpoint['logit_q_base']
        d_eff_base = float(geo['d_eff_formula'])
        kappa_base = float(geo['kappa_nearest'])
        kappa_eff_base = float(geo['kappa_eff'])

        log(f"  Baseline: d_eff={d_eff_base:.4f}, kappa={kappa_base:.4f}, "
            f"kappa_eff={kappa_eff_base:.4f}, q={q_base:.4f}")

        C_fitted = logit_q_base - A_RENORM_K20 * kappa_eff_base
        log(f"  C_fitted = {C_fitted:.4f} (pre-registered formula: logit=A*kappa_eff+C)")

        surgery_records = []
        for r in SURGERY_LEVELS:
            X_tr_new = apply_surgery(X_tr, y_tr, geo, r)
            X_te_new = apply_surgery(X_te, y_te, geo, r)
            verif = verify_surgery(X_tr, X_tr_new, y_tr, geo, r)
            q_new, acc_new = compute_q(X_tr_new, y_tr, X_te_new, y_te)
            logit_q_new = float(np.log(q_new / (1 - q_new) + 1e-10))

            logit_q_pred = C_fitted + A_RENORM_K20 * kappa_base * float(np.sqrt(r * d_eff_base))
            delta_actual = logit_q_new - logit_q_base
            delta_pred = logit_q_pred - logit_q_base
            calib_err = abs(delta_actual - delta_pred) / (abs(delta_pred) + 1e-6)

            log(f"  [r={r:.3f}] d_eff_new={verif['d_eff_new']:.3f}, "
                f"kappa_chg={verif['kappa_change_pct']:.4f}%, "
                f"q={q_new:.4f}, logit_act={logit_q_new:.4f}, "
                f"logit_pred={logit_q_pred:.4f}, calib={calib_err:.4f}")

            surgery_records.append({
                'seed': seed, 'epoch_used': best_epoch,
                'r_nominal': float(r), 'r_actual': float(verif['d_eff_ratio_actual']),
                'd_eff_base': d_eff_base, 'd_eff_new': float(verif['d_eff_new']),
                'kappa_base': kappa_base, 'kappa_new': float(verif['kappa_new']),
                'kappa_eff_base': kappa_eff_base,
                'kappa_change_pct': float(verif['kappa_change_pct']),
                'trW_change_pct': float(verif['trW_change_pct']),
                'q_base': float(q_base), 'logit_q_base': float(logit_q_base),
                'q_new': float(q_new), 'logit_q_new': float(logit_q_new),
                'logit_q_pred': float(logit_q_pred),
                'delta_logit_actual': float(delta_actual),
                'delta_logit_pred': float(delta_pred),
                'calibration_error': float(calib_err),
                'C_fitted': float(C_fitted),
            })

        # Quick analysis for this seed
        actual = np.array([r['logit_q_new'] for r in surgery_records])
        pred = np.array([r['logit_q_pred'] for r in surgery_records])
        r_pearson = float(np.corrcoef(actual, pred)[0, 1])
        calib_errs = [r['calibration_error'] for r in surgery_records
                      if abs(r['delta_logit_pred']) > 0.01]
        mean_calib = float(np.mean(calib_errs)) if calib_errs else float('nan')
        max_kappa_chg = max(r['kappa_change_pct'] for r in surgery_records)

        pass_status = (r_pearson > 0.99) and (mean_calib < 0.10) and (max_kappa_chg < 0.1)
        log(f"\n  SEED {seed} RESULT: Pearson r={r_pearson:.4f}, "
            f"calib={mean_calib:.4f}, kappa_chg={max_kappa_chg:.6f}%")
        log(f"  STATUS: {'PASS' if pass_status else 'FAIL'} "
            f"(PRIMARY: {'PASS' if r_pearson > 0.99 else 'FAIL'}, "
            f"SECONDARY: {'PASS' if mean_calib < 0.10 else 'FAIL'})")

        all_results.append({
            'seed': seed, 'status': 'complete',
            'epoch_used': best_epoch,
            'kappa_eff_at_surgery': kappa_eff_base,
            'surgery_records': surgery_records,
            'trajectory': trajectory,
            'analysis': {
                'r_pearson': r_pearson,
                'mean_calib': mean_calib,
                'max_kappa_chg': max_kappa_chg,
                'pass': pass_status,
            }
        })

        # Save intermediate
        with open(RESULT_PATH, 'w') as f:
            json.dump({'results': all_results, 'n_seeds_complete': seed + 1}, f, indent=2)

    # ============ FINAL ANALYSIS ============
    log(f"\n{'='*70}")
    log("FINAL ANALYSIS")
    log(f"{'='*70}")

    complete = [r for r in all_results if r.get('status') == 'complete']
    if not complete:
        log("ERROR: No seeds completed surgery (no linear regime found).")
        log("DIAGNOSIS: Adjust KAPPA_EFF_MIN/MAX or use fewer training epochs.")
        return

    # Pool all surgery records
    all_recs = []
    for r in complete:
        all_recs.extend(r['surgery_records'])
        log(f"  Seed {r['seed']}: epoch={r['epoch_used']}, "
            f"kappa_eff={r['kappa_eff_at_surgery']:.4f}")

    actual_all = np.array([r['logit_q_new'] for r in all_recs])
    pred_all = np.array([r['logit_q_pred'] for r in all_recs])
    r_pearson_all = float(np.corrcoef(actual_all, pred_all)[0, 1])
    calib_all = [r['calibration_error'] for r in all_recs if abs(r['delta_logit_pred']) > 0.01]
    mean_calib_all = float(np.mean(calib_all)) if calib_all else float('nan')
    max_kappa_all = float(max(r['kappa_change_pct'] for r in all_recs))

    # Fit empirical A
    kappa_effs = np.array([r['kappa_base'] * np.sqrt(r['r_actual'] * r['d_eff_base'])
                           for r in all_recs])
    if np.std(kappa_effs) > 0:
        A_mat = np.column_stack([kappa_effs, np.ones(len(kappa_effs))])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, actual_all, rcond=None)
        A_empirical = float(coeffs[0])
        A_error_pct = abs(A_empirical - A_RENORM_K20) / A_RENORM_K20 * 100
    else:
        A_empirical = float('nan')
        A_error_pct = float('nan')

    log(f"\nPRIMARY  (Pearson r > 0.99):    r={r_pearson_all:.4f}  "
        f"{'PASS' if r_pearson_all > 0.99 else 'FAIL'}")
    log(f"SECONDARY (calib < 10%):         calib={mean_calib_all:.4f}  "
        f"{'PASS' if mean_calib_all < 0.10 else 'FAIL'}")
    log(f"TERTIARY  (kappa_chg < 0.1%):    max={max_kappa_all:.6f}%  "
        f"{'PASS' if max_kappa_all < 0.1 else 'FAIL'}")
    log(f"A_empirical = {A_empirical:.4f} vs A_renorm = {A_RENORM_K20} "
        f"(error = {A_error_pct:.1f}%)")

    overall_pass = (r_pearson_all > 0.99) and (mean_calib_all < 0.10) and (max_kappa_all < 0.1)
    log(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")

    if overall_pass:
        log("CONCLUSION: d_eff_formula IS causally active in the CTI linear regime.")
        log(f"  The law logit(q) = A*kappa*sqrt(d_eff)+C holds with A={A_empirical:.4f} "
            f"(pre-reg A={A_RENORM_K20}, error={A_error_pct:.1f}%).")
    elif r_pearson_all > 0.99:
        log("CONCLUSION: Direction confirmed (Pearson r > 0.99), but scale calibration off.")
        log(f"  A_empirical={A_empirical:.4f} vs A_renorm={A_RENORM_K20} "
            f"({A_error_pct:.1f}% error).")
        log("  Theory may need A recalibration for CIFAR regime.")
    else:
        log("CONCLUSION: Surgery shows wrong functional form. Theory needs revision.")

    # Per-r summary
    log(f"\nPer-r summary (pooled over {len(complete)} seeds):")
    log(f"{'r':>6} {'d_eff_base':>10} {'d_eff_new':>9} {'kappa_eff':>10} "
        f"{'q_base':>8} {'q_new':>7} {'logit_act':>10} {'logit_pred':>11} {'calib':>7}")
    for rv in sorted(set(r['r_nominal'] for r in all_recs)):
        sub = [r for r in all_recs if r['r_nominal'] == rv]
        avg = lambda k: float(np.mean([r[k] for r in sub]))
        log(f"{rv:>6.2f} {avg('d_eff_base'):>10.3f} {avg('d_eff_new'):>9.3f} "
            f"{avg('kappa_eff_base'):>10.4f} "
            f"{avg('q_base'):>8.4f} {avg('q_new'):>7.4f} "
            f"{avg('logit_q_new'):>10.4f} {avg('logit_q_pred'):>11.4f} "
            f"{avg('calibration_error'):>7.4f}")

    # Save final results
    final = {
        'results': all_results,
        'final_analysis': {
            'r_pearson': r_pearson_all,
            'mean_calib': mean_calib_all,
            'max_kappa_chg': max_kappa_all,
            'A_empirical': A_empirical,
            'A_error_pct': A_error_pct,
            'overall_pass': overall_pass,
            'n_seeds': len(complete),
            'n_records': len(all_recs),
        }
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(final, f, indent=2)
    log(f"\nResults saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
