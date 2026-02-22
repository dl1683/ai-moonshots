#!/usr/bin/env python -u
"""
MULTI-DIRECTION SURGERY (Feb 22 2026)
======================================
Theory: logit(q) = A_renorm * kappa_nearest * sqrt(d_eff) + C

Single-direction surgery (existing): compresses variance along ONE pair direction
  -> tests 1 of K-1 competitive edges
  -> A_emp = A_renorm / (K-1) [observed: 0.064 vs 0.055 predicted, ratio 1.17]

Multi-direction surgery (THIS SCRIPT): compresses variance along ALL K-1 centroid directions
  -> tests ALL K-1 competitive edges simultaneously
  -> PREDICTION: delta_logit_multi = (K-1) * delta_logit_single
  -> PREDICTION: A_emp_multi = A_renorm (full formula restored!)

PRE-REGISTERED PREDICTIONS:
  At kappa_eff=1.03, r=10:
  - delta_logit_single = 0.136 (from seeds 0,1 of linear_regime_surgery)
  - delta_logit_multi = (K-1) * 0.136 = 19 * 0.136 = 2.584
  - Theory (A_renorm): delta_logit = A_renorm * kappa_eff * (sqrt(10)-1) = 1.054 * 1.03 * 2.162 = 2.347
  - Ratio: delta_multi / theory = 2.584 / 2.347 = 1.10 (within 10% of theory)

If confirmed: CTI formula IS causally valid at the K-direction aggregate level.
The 16x "failure" was measuring 1 edge out of K-1 = 19 edges.

Surgeries:
  1. Single: compress along Delta_hat (nearest centroid pair) [replicates linear_regime_surgery]
  2. Multi:  compress along ALL K-1 principal centroid directions
             x_new = mu_c + P_sub * (1/sqrt(r) * P_sub^T (x-mu_c)) + P_orth * scale_orth * P_orth^T (x-mu_c)
             where P_sub spans the K-1 centroid subspace (from SVD of centroid matrix)
             scale_orth chosen to preserve tr(Sigma_W)

Both surgeries preserve kappa_nearest EXACTLY (verified).
Multi surgery preserves kappa_nearest for ALL pairs (not just nearest).
"""

import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ================================================================
# CONFIG
# ================================================================
K = 20
N_EPOCHS = 60
SEEDS = [0, 1, 2]
BATCH_SIZE = 256
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
N_EVAL_SUBSAMPLE = 5000

CHECKPOINT_EPOCHS = list(range(1, 16)) + list(range(20, 65, 5))
KAPPA_EFF_TARGET = 1.0
KAPPA_EFF_MIN = 0.5
KAPPA_EFF_MAX = 2.0
SURGERY_LEVELS = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

A_RENORM_K20 = 1.0535

RESULT_PATH = "results/cti_multidirection_surgery.json"
LOG_PATH = "results/cti_multidirection_surgery_log.txt"

# Pre-registered predictions
SINGLE_DELTA_R10 = 0.136  # from linear_regime_surgery seeds 0+1 average
MULTI_DELTA_R10_PRED = (K - 1) * SINGLE_DELTA_R10  # = 2.584
THEORY_DELTA_R10 = A_RENORM_K20 * 1.03 * (np.sqrt(10) - 1)  # = 2.347


# ================================================================
# DATA
# ================================================================
def _coarse_label(x):
    return x // 5

def get_cifar_coarse():
    norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    train_ds = torchvision.datasets.CIFAR100(
        root="data", train=True, download=True,
        transform=train_tf, target_transform=_coarse_label
    )
    test_ds = torchvision.datasets.CIFAR100(
        root="data", train=False, download=True,
        transform=test_tf, target_transform=_coarse_label
    )
    return train_ds, test_ds


def get_loaders(train_ds, test_ds):
    tr_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
    )
    te_ld = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=False
    )
    return tr_ld, te_ld


# ================================================================
# MODEL
# ================================================================
def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, K)
    return model.to(DEVICE)


# ================================================================
# EMBED
# ================================================================
def extract_embeddings(model, loader, subsample=None):
    model.eval()
    embs, lbls = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            feat = model.avgpool(model.layer4(model.layer3(model.layer2(
                model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))))))
            feat = torch.flatten(feat, 1)
            embs.append(feat.cpu().numpy())
            lbls.append(y.numpy())
    X = np.concatenate(embs)
    y = np.concatenate(lbls)
    if subsample and len(X) > subsample:
        idx = np.random.choice(len(X), subsample, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def compute_q(X_tr, y_tr, X_te, y_te, K_classes=K):
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X_tr, y_tr)
    acc = float(knn.score(X_te, y_te))
    return (acc - 1.0 / K_classes) / (1.0 - 1.0 / K_classes)


def compute_kappa_and_deff(X, y, K_classes=K):
    classes = np.unique(y)
    centroids = np.array([X[y == c].mean(axis=0) for c in classes])
    n = len(X)
    X_centered = np.vstack([X[y == c] - centroids[i] for i, c in enumerate(classes)])
    Sigma_W = X_centered.T @ X_centered / n
    d = X.shape[1]
    tr_W = np.trace(Sigma_W)
    sigma_W_global = np.sqrt(tr_W / d)

    # Nearest centroid pair
    delta_min = np.inf
    nearest_pair = (0, 1)
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < delta_min:
                delta_min = dist
                nearest_pair = (i, j)

    Delta_hat = (centroids[nearest_pair[0]] - centroids[nearest_pair[1]])
    Delta_hat = Delta_hat / np.linalg.norm(Delta_hat)
    sigma_cdir = np.sqrt(Delta_hat @ Sigma_W @ Delta_hat)

    kappa_nearest = delta_min / (sigma_W_global * np.sqrt(d))
    d_eff = tr_W / (sigma_cdir ** 2) if sigma_cdir > 1e-12 else 1.0
    kappa_eff = kappa_nearest * np.sqrt(d_eff)
    return kappa_nearest, d_eff, kappa_eff, sigma_W_global, tr_W, Sigma_W, centroids, nearest_pair, Delta_hat


# ================================================================
# SINGLE-DIRECTION SURGERY (replicates linear_regime_surgery)
# ================================================================
def apply_single_surgery(X, y, r, centroids, nearest_pair, Delta_hat, tr_W, Sigma_W):
    classes = np.unique(y)
    sigma_cdir_sq = float(Delta_hat @ Sigma_W @ Delta_hat)
    remaining = tr_W - sigma_cdir_sq
    scale_along = 1.0 / np.sqrt(r)
    if remaining > 1e-12:
        scale_perp = np.sqrt((tr_W - sigma_cdir_sq / r) / remaining)
    else:
        scale_perp = 1.0

    X_new = X.copy()
    for i, c in enumerate(classes):
        mask = (y == c)
        x_c = X[mask] - centroids[i]
        comp_along = (x_c @ Delta_hat)[:, None] * Delta_hat[None, :]
        comp_perp = x_c - comp_along
        X_new[mask] = centroids[i] + scale_along * comp_along + scale_perp * comp_perp
    return X_new


# ================================================================
# MULTI-DIRECTION SURGERY (all K-1 centroid directions)
# ================================================================
def compute_centroid_subspace(centroids, K_classes=K):
    """
    SVD of mean-centered centroid matrix -> K-1 principal centroid directions.
    Returns U_sub: (d, K-1) orthonormal basis spanning centroid subspace.
    """
    centroid_mean = centroids.mean(axis=0)
    M = centroids - centroid_mean  # (K, d)
    # SVD: M = U S V^T, but we want directions in d-space
    # M.T @ M = V S^2 V^T, top K-1 eigenvectors are centroid directions
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    # Vt: (K, d), top K-1 rows are the K-1 principal centroid directions
    U_sub = Vt[:K_classes - 1].T  # shape (d, K-1)
    return U_sub


def apply_multi_surgery(X, y, r, centroids, tr_W, Sigma_W, K_classes=K):
    """
    Compress variance in all K-1 centroid directions by 1/sqrt(r).
    Expand variance in the perpendicular (d-(K-1)) directions to preserve tr(Sigma_W).
    """
    classes = np.unique(y)
    d = X.shape[1]

    # Get centroid subspace (K-1 directions): U_sub (d, K-1)
    U_sub = compute_centroid_subspace(centroids, K_classes)  # (d, K-1)

    # Efficient tr(Sigma_W in subspace): tr(U_sub^T Sigma_W U_sub) without forming P_sub
    # Shape: U_sub^T (K-1, d) @ Sigma_W (d, d) @ U_sub (d, K-1) -> (K-1, K-1)
    tr_sub = float(np.trace(U_sub.T @ Sigma_W @ U_sub))
    tr_orth = tr_W - tr_sub

    # Scale to preserve tr(Sigma_W):
    # (1/r) * tr_sub + scale_orth^2 * tr_orth = tr_W
    if tr_orth > 1e-12:
        scale_orth = np.sqrt((tr_W - tr_sub / r) / tr_orth)
    else:
        scale_orth = 1.0
    scale_sub = 1.0 / np.sqrt(r)

    X_new = X.copy()
    for i, c in enumerate(classes):
        mask = (y == c)
        x_c = X[mask] - centroids[i]  # (n_c, d)
        # Efficient projection: coords (n_c, K-1) then back to d-space
        coords_sub = x_c @ U_sub            # (n_c, K-1)
        comp_sub = coords_sub @ U_sub.T      # (n_c, d) - projection onto centroid subspace
        comp_orth = x_c - comp_sub           # (n_c, d) - orthogonal complement
        X_new[mask] = centroids[i] + scale_sub * comp_sub + scale_orth * comp_orth

    return X_new


# ================================================================
# LOG
# ================================================================
_logfile = None
def log(msg):
    print(msg, flush=True)
    if _logfile:
        _logfile.write(msg + "\n")
        _logfile.flush()


# ================================================================
# MAIN
# ================================================================
def main():
    global _logfile
    _logfile = open(LOG_PATH, "w", encoding="ascii", errors="replace")
    log(f"Multi-Direction Surgery Experiment")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, SEEDS={SEEDS}")
    log(f"PRE-REGISTERED: delta_multi(r=10) = {MULTI_DELTA_R10_PRED:.3f} = {K-1}*{SINGLE_DELTA_R10}")
    log(f"  Theory (A_renorm): {THEORY_DELTA_R10:.3f}")
    log(f"  Pass criterion: ratio delta_multi/delta_single in [0.7*(K-1), 1.3*(K-1)] = [{0.7*(K-1):.1f}, {1.3*(K-1):.1f}]")
    log(f"{'='*60}")

    all_results = []
    train_ds, test_ds = get_cifar_coarse()
    _, test_ds_full = get_cifar_coarse()
    train_ld, test_ld = get_loaders(train_ds, test_ds)

    seed_summary = {}

    for seed in SEEDS:
        log(f"\n{'='*60}")
        log(f"SEED {seed}")
        log(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = get_model()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 58], gamma=0.1)
        ce_loss_fn = nn.CrossEntropyLoss()

        # Track best linear-regime checkpoint
        best_epoch = None
        best_dist = np.inf
        saved_X = None
        saved_y = None
        saved_X_te = None
        saved_y_te = None

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            for x, y_batch in train_ld:
                x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                feat = model.avgpool(model.layer4(model.layer3(model.layer2(
                    model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))))))
                feat = torch.flatten(feat, 1)
                logits = model.fc(feat)
                loss = ce_loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if epoch in CHECKPOINT_EPOCHS:
                np.random.seed(seed * 1000 + epoch)
                X_tr, y_tr = extract_embeddings(model, train_ld, N_EVAL_SUBSAMPLE)
                X_te, y_te = extract_embeddings(model, test_ld)
                kappa, d_eff, kappa_eff, sigma_W, tr_W, Sigma_W, centroids, np_pair, Delta_hat = \
                    compute_kappa_and_deff(X_tr, y_tr)
                q = compute_q(X_tr, y_tr, X_te, y_te)
                log(f"  [ep={epoch:2d}] kappa_eff={kappa_eff:.4f}, d_eff={d_eff:.3f}, kappa={kappa:.4f}, q={q:.4f}")

                if KAPPA_EFF_MIN <= kappa_eff <= KAPPA_EFF_MAX:
                    dist = abs(kappa_eff - KAPPA_EFF_TARGET)
                    if dist < best_dist:
                        best_dist = dist
                        best_epoch = epoch
                        saved_X = X_tr.copy()
                        saved_y = y_tr.copy()
                        saved_X_te = X_te.copy()
                        saved_y_te = y_te.copy()
                        saved_kappa = kappa
                        saved_deff = d_eff
                        saved_kappa_eff = kappa_eff
                        saved_q = q
                        saved_trW = tr_W
                        saved_SigmaW = Sigma_W.copy()
                        saved_centroids = centroids.copy()
                        saved_pair = np_pair
                        saved_Delta_hat = Delta_hat.copy()

            if epoch % 10 == 0:
                log(f"  [seed={seed} epoch={epoch}] training in progress")

        if best_epoch is None:
            log(f"  WARNING: No linear regime checkpoint found for seed={seed}")
            continue

        logit_base = np.log(saved_q / (1 - saved_q + 1e-10) + 1e-10)
        log(f"\n  Selected checkpoint: epoch={best_epoch}, kappa_eff={saved_kappa_eff:.4f}")
        log(f"  Baseline: d_eff={saved_deff:.3f}, kappa={saved_kappa:.4f}, kappa_eff={saved_kappa_eff:.4f}, q={saved_q:.4f} [TEST SET]")

        C_fitted = logit_base - A_RENORM_K20 * saved_kappa_eff

        single_deltas = []
        multi_deltas = []

        for r in SURGERY_LEVELS:
            # -- SINGLE DIRECTION --
            X_single = apply_single_surgery(saved_X, saved_y, r, saved_centroids,
                                            saved_pair, saved_Delta_hat, saved_trW, saved_SigmaW)
            X_single_te = apply_single_surgery(saved_X_te, saved_y_te, r, saved_centroids,
                                               saved_pair, saved_Delta_hat, saved_trW, saved_SigmaW)
            kappa_single, _, _, _, _, _, _, _, _ = compute_kappa_and_deff(X_single, saved_y)
            kappa_chg_single = abs(kappa_single - saved_kappa) / (abs(saved_kappa) + 1e-12) * 100
            q_single = compute_q(X_single, saved_y, X_single_te, saved_y_te)
            logit_single = np.log(q_single / (1 - q_single)) if 0 < q_single < 1 else np.nan
            logit_pred = C_fitted + A_RENORM_K20 * saved_kappa * np.sqrt(r * saved_deff)
            calib_single = abs(logit_single - logit_pred) / (abs(logit_pred - logit_base) + 1e-8)
            delta_single = logit_single - logit_base
            single_deltas.append(delta_single)

            # -- MULTI DIRECTION --
            X_multi = apply_multi_surgery(saved_X, saved_y, r, saved_centroids,
                                          saved_trW, saved_SigmaW, K_classes=K)
            X_multi_te = apply_multi_surgery(saved_X_te, saved_y_te, r, saved_centroids,
                                             saved_trW, saved_SigmaW, K_classes=K)
            kappa_multi, _, _, _, _, _, _, _, _ = compute_kappa_and_deff(X_multi, saved_y)
            kappa_chg_multi = abs(kappa_multi - saved_kappa) / (abs(saved_kappa) + 1e-12) * 100
            q_multi = compute_q(X_multi, saved_y, X_multi_te, saved_y_te)
            logit_multi = np.log(q_multi / (1 - q_multi)) if 0 < q_multi < 1 else np.nan
            delta_multi = logit_multi - logit_base
            multi_deltas.append(delta_multi)

            ratio = delta_multi / delta_single if abs(delta_single) > 1e-6 else np.nan
            log(f"  [r={r:.2f}] single: q={q_single:.4f}, logit={logit_single:.4f} | "
                f"multi: q={q_multi:.4f}, logit={logit_multi:.4f} | ratio={ratio:.2f}")

            result = {
                "seed": seed, "r": r,
                "best_epoch": best_epoch,
                "kappa_base": float(saved_kappa), "d_eff_base": float(saved_deff),
                "kappa_eff_base": float(saved_kappa_eff), "q_base": float(saved_q),
                "logit_base": float(logit_base),
                "q_single": float(q_single), "logit_single": float(logit_single),
                "kappa_chg_single": float(kappa_chg_single),
                "delta_single": float(delta_single), "logit_pred": float(logit_pred),
                "calib_single": float(calib_single),
                "q_multi": float(q_multi), "logit_multi": float(logit_multi),
                "kappa_chg_multi": float(kappa_chg_multi),
                "delta_multi": float(delta_multi),
                "ratio_multi_single": float(ratio) if not np.isnan(ratio) else None,
            }
            all_results.append(result)

        # Summary for this seed
        # At r=10 (last level):
        r10_single = single_deltas[-1]
        r10_multi = multi_deltas[-1]
        r10_ratio = r10_multi / r10_single if abs(r10_single) > 1e-6 else np.nan
        r10_theory = A_RENORM_K20 * saved_kappa_eff * (np.sqrt(10) - 1)
        log(f"\n  SEED {seed} SUMMARY at r=10:")
        log(f"    delta_single = {r10_single:.4f}")
        log(f"    delta_multi  = {r10_multi:.4f}")
        log(f"    ratio = {r10_ratio:.2f}  (predicted K-1={K-1})")
        log(f"    theory (A_renorm) delta = {r10_theory:.4f}")
        log(f"    delta_multi / theory = {r10_multi/r10_theory:.3f}  (ideal 1.0)")
        seed_summary[seed] = {"single": r10_single, "multi": r10_multi, "ratio": r10_ratio,
                              "theory": r10_theory}

    # Final pooled analysis
    log(f"\n{'='*60}")
    log(f"FINAL POOLED ANALYSIS")
    log(f"{'='*60}")

    seeds_done = list(seed_summary.keys())
    if seeds_done:
        ratios = [seed_summary[s]["ratio"] for s in seeds_done if seed_summary[s]["ratio"] is not None]
        singles = [seed_summary[s]["single"] for s in seeds_done]
        multis = [seed_summary[s]["multi"] for s in seeds_done]
        theories = [seed_summary[s]["theory"] for s in seeds_done]
        mean_ratio = np.mean(ratios)
        mean_single = np.mean(singles)
        mean_multi = np.mean(multis)
        mean_theory = np.mean(theories)
        log(f"\nMean delta_single(r=10) = {mean_single:.4f}")
        log(f"Mean delta_multi(r=10)  = {mean_multi:.4f}")
        log(f"Mean ratio multi/single = {mean_ratio:.2f}  (expected K-1={K-1})")
        log(f"Mean theory delta       = {mean_theory:.4f}")
        log(f"Mean delta_multi / theory = {mean_multi/mean_theory:.3f}  (ideal 1.0)")
        log(f"\nPASS CRITERIA:")
        log(f"  PRIMARY: ratio in [0.7*(K-1), 1.3*(K-1)] = [{0.7*(K-1):.1f}, {1.3*(K-1):.1f}]")
        log(f"    -> {mean_ratio:.2f} {'PASS' if 0.7*(K-1) <= mean_ratio <= 1.3*(K-1) else 'FAIL'}")
        log(f"  SECONDARY: delta_multi / theory in [0.7, 1.3]")
        log(f"    -> {mean_multi/mean_theory:.3f} {'PASS' if 0.7 <= mean_multi/mean_theory <= 1.3 else 'FAIL'}")

    with open(RESULT_PATH, "w", encoding="ascii", errors="replace") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {RESULT_PATH}")
    _logfile.close()


if __name__ == "__main__":
    main()
