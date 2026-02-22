#!/usr/bin/env python -u
"""
TOP-M COMPETITOR SWEEP (Feb 22 2026)
======================================
Design: Rank K-1 competitor classes by margin kappa_j = dist(mu_y, mu_j) / (sigma_W * sqrt(d))
Apply surgery to the TOP m competitor directions (m=1,2,3,...,K-1).

THREE COMPETING MODELS:
  A. Nearest-only:     delta(m) = delta(1) for all m (only nearest matters)
  B. Equal-additive:   delta(m) = m * delta(1)  (all K-1 pairs equally weighted)
  C. Sparse-effective: delta(m) = sum_{j=1}^m w_j * delta_j_pred
     where w_j ~ softmax(-kappa_j^2 / 2) (Gumbel Race theory)
     and delta_j_pred = A_renorm / (K-1) * kappa_eff_j * (sqrt(r) - 1)

KEY PREDICTION: If Model C matches data -> w_j is DERIVED from geometry (zero free params).
This would mean K_eff is PREDICTABLE from kappa distribution, not just observed.

K_eff = (sum_j w_j)^2 / sum_j(w_j^2) where w_j = softmax(-kappa_j^2 / 2)

PRE-REGISTERED:
  At r=10, for seed with kappa_eff~1.0:
  - Model A: delta_m = delta_1 (constant ~0.073) for all m
  - Model B: delta_m = m * 0.073 (linear)
  - Model C: delta_m = weighted sum using softmax weights
  - From Session 19: delta_m at m=K-1 was 1.048, delta_1 was 0.082 -> K_eff ~= 13
  - Model C PREDICTED K_eff_pred = 1/sum(w_j^2) must match ~13

PASS CRITERION:
  Model C delta(m) values match observed within 30% (MSE / total variance < 0.3)
  AND K_eff_predicted (from softmax) matches K_eff_observed within 25%
"""

import json
import sys
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

# Surgery: apply to top m directions at this r
SURGERY_R = 10.0
M_VALUES = list(range(1, K))  # m = 1, 2, ..., K-1 = 19

A_RENORM_K20 = 1.0535

RESULT_PATH = "results/cti_top_m_sweep.json"
LOG_PATH = "results/cti_top_m_sweep_log.txt"


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


# ================================================================
# GEOMETRY
# ================================================================
def compute_full_geometry(X, y, K_classes=K):
    """
    Compute geometry for all K-1 competitor directions, not just nearest.
    Returns:
      - centroids: (K, d)
      - Sigma_W: (d, d) within-class covariance
      - tr_W: trace of Sigma_W
      - sigma_W_global: sqrt(tr_W/d)
      - competitor_dirs: list of K-1 direction vectors (sorted by kappa_j ascending)
      - kappa_vals: sorted kappa values (ascending, nearest first)
      - kappa_eff: kappa_nearest * sqrt(d_eff)
      - d_eff: based on nearest pair
    """
    classes = np.unique(y)
    assert len(classes) == K_classes
    centroids = np.array([X[y == c].mean(axis=0) for c in classes])
    n = len(X)
    d = X.shape[1]

    X_centered = np.vstack([X[y == c] - centroids[i] for i, c in enumerate(classes)])
    Sigma_W = X_centered.T @ X_centered / n
    tr_W = float(np.trace(Sigma_W))
    sigma_W_global = float(np.sqrt(tr_W / d))

    # For each class pair (y vs j), compute kappa_j
    # We treat class 0 as "target" across all pairs for generality:
    # Actually, surgery is defined for a specific target class.
    # Here, we compute PAIR-WISE: for each j != target class, find the direction
    # mu_0 - mu_j and compute kappa_j = ||mu_0 - mu_j|| / (sigma_W * sqrt(d))
    # We take class 0 (or whichever has the most samples) as target.
    # For the global formula, we want the average over all target classes.
    # SIMPLIFICATION: Use class 0 as fixed target. The formula is symmetric.

    # Actually, the surgery framework tests with respect to the NEAREST pair.
    # Here we want: for a FIXED target class (class 0), rank all K-1 competitor classes
    # by their margin kappa_{0j} = ||mu_0 - mu_j|| / (sigma_W * sqrt(d))

    target_class_idx = 0  # use class 0 as target
    target_centroid = centroids[target_class_idx]

    competitor_info = []
    for j, c in enumerate(classes):
        if j == target_class_idx:
            continue
        diff = target_centroid - centroids[j]
        dist = float(np.linalg.norm(diff))
        dir_hat = diff / (dist + 1e-12)
        kappa_j = dist / (sigma_W_global * np.sqrt(d))
        sigma_cdir_j = float(np.sqrt(dir_hat @ Sigma_W @ dir_hat))
        d_eff_j = tr_W / (sigma_cdir_j**2) if sigma_cdir_j > 1e-12 else 1.0
        competitor_info.append({
            "class_idx": j,
            "class_label": int(c),
            "dir_hat": dir_hat,
            "kappa_j": kappa_j,
            "dist": dist,
            "sigma_cdir_j": sigma_cdir_j,
            "d_eff_j": d_eff_j,
        })

    # Sort by kappa_j ASCENDING (nearest = smallest kappa = hardest)
    competitor_info.sort(key=lambda x: x["kappa_j"])

    # Standard d_eff and kappa_nearest (using nearest competitor)
    nearest = competitor_info[0]
    kappa_nearest = nearest["kappa_j"]
    d_eff = tr_W / (nearest["sigma_cdir_j"]**2) if nearest["sigma_cdir_j"] > 1e-12 else 1.0
    kappa_eff = kappa_nearest * float(np.sqrt(d_eff))

    return {
        "centroids": centroids,
        "Sigma_W": Sigma_W,
        "tr_W": tr_W,
        "sigma_W_global": sigma_W_global,
        "kappa_nearest": kappa_nearest,
        "d_eff": d_eff,
        "kappa_eff": kappa_eff,
        "target_class_idx": target_class_idx,
        "competitor_info": competitor_info,  # sorted by kappa ascending
    }


# ================================================================
# TOP-M SURGERY
# ================================================================
def apply_top_m_surgery(X, y, r, geo, m, target_class_idx=0):
    """
    Apply surgery to the TOP m competitor directions (sorted by kappa ascending).
    Preserves tr(Sigma_W). Returns None if infeasible.
    """
    competitor_info = geo["competitor_info"]
    Sigma_W = geo["Sigma_W"]
    tr_W = geo["tr_W"]
    centroids = geo["centroids"]
    classes = np.unique(y)

    # Get top-m direction vectors (orthonormalize via QR)
    dirs = np.array([competitor_info[j]["dir_hat"] for j in range(m)])  # (m, d)

    # Orthonormalize via QR
    Q, R = np.linalg.qr(dirs.T, mode="reduced")  # Q: (d, m)
    U_sub = Q  # (d, m), orthonormal basis for top-m centroid directions

    # Compute tr(Sigma_W in subspace)
    tr_sub = float(np.trace(U_sub.T @ Sigma_W @ U_sub))
    tr_orth = tr_W - tr_sub

    # Check feasibility
    numerator = tr_W - tr_sub / r
    if tr_orth < 1e-12 or numerator < 0:
        return None  # infeasible

    scale_orth = float(np.sqrt(numerator / tr_orth))
    scale_sub = 1.0 / float(np.sqrt(r))

    X_new = X.copy()
    for i, c in enumerate(classes):
        mask = (y == c)
        x_c = X[mask] - centroids[i]  # (n_c, d)
        coords_sub = x_c @ U_sub       # (n_c, m)
        comp_sub = coords_sub @ U_sub.T  # (n_c, d)
        comp_orth = x_c - comp_sub     # (n_c, d)
        X_new[mask] = centroids[i] + scale_sub * comp_sub + scale_orth * comp_orth

    return X_new


# ================================================================
# SOFTMAX WEIGHT PREDICTION
# ================================================================
def compute_softmax_weights(competitor_info):
    """
    Predict w_j = softmax(-kappa_j^2 / 2) for all K-1 competitors.
    Returns normalized weights and K_eff = (sum w_j)^2 / sum(w_j^2)
    """
    kappas = np.array([c["kappa_j"] for c in competitor_info])
    log_w = -kappas**2 / 2.0
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    K_eff = float((w.sum()**2) / (w**2).sum())
    return w, K_eff


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
    log("Top-m Competitor Sweep Experiment")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, SEEDS={SEEDS}, r={SURGERY_R}")
    log(f"M_VALUES={M_VALUES}")
    log(f"{'='*60}")
    log("THREE COMPETING MODELS:")
    log("  A. Nearest-only:     delta(m) = delta(1) for all m")
    log("  B. Equal-additive:   delta(m) = m * delta(1)")
    log("  C. Sparse-effective: delta(m) uses softmax weights w_j ~ exp(-kappa_j^2/2)")
    log(f"{'='*60}")

    all_results = []
    train_ds, test_ds = get_cifar_coarse()
    train_ld, test_ld = get_loaders(train_ds, test_ds)

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

        best_epoch = None
        best_dist = np.inf
        saved_X = saved_y = saved_X_te = saved_y_te = None
        saved_geo = None
        saved_kappa_eff = saved_q = None

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
                geo = compute_full_geometry(X_tr, y_tr)
                kappa_eff = geo["kappa_eff"]
                q = compute_q(X_tr, y_tr, X_te, y_te)
                log(f"  [ep={epoch:2d}] kappa_eff={kappa_eff:.4f}, d_eff={geo['d_eff']:.3f}, "
                    f"kappa={geo['kappa_nearest']:.4f}, q={q:.4f}")

                if KAPPA_EFF_MIN <= kappa_eff <= KAPPA_EFF_MAX:
                    dist = abs(kappa_eff - KAPPA_EFF_TARGET)
                    if dist < best_dist:
                        best_dist = dist
                        best_epoch = epoch
                        saved_X = X_tr.copy()
                        saved_y = y_tr.copy()
                        saved_X_te = X_te.copy()
                        saved_y_te = y_te.copy()
                        saved_geo = geo
                        saved_kappa_eff = kappa_eff
                        saved_q = q

            if epoch % 10 == 0:
                log(f"  [seed={seed} epoch={epoch}] training in progress")

        if best_epoch is None:
            log(f"  WARNING: No linear regime checkpoint found for seed={seed}")
            continue

        geo = saved_geo
        logit_base = float(np.log(saved_q / (1 - saved_q + 1e-10) + 1e-10))
        kappas = [c["kappa_j"] for c in geo["competitor_info"]]
        w_softmax, K_eff_pred = compute_softmax_weights(geo["competitor_info"])

        log(f"\n  Selected checkpoint: epoch={best_epoch}, kappa_eff={saved_kappa_eff:.4f}")
        log(f"  Baseline: d_eff={geo['d_eff']:.3f}, kappa={geo['kappa_nearest']:.4f}, q={saved_q:.4f}")
        log(f"\n  Competitor kappa values (sorted ascending, nearest first):")
        for j, ci in enumerate(geo["competitor_info"][:5]):
            log(f"    rank={j+1}: kappa={ci['kappa_j']:.4f}, w_softmax={w_softmax[j]:.4f}")
        log(f"    ...")
        log(f"  K_eff_predicted (from softmax) = {K_eff_pred:.2f}")
        log(f"\n  Starting top-m sweep at r={SURGERY_R}...")

        delta_m_actual = []
        delta_m_pred_A = []  # nearest-only
        delta_m_pred_B = []  # equal-additive
        delta_m_pred_C = []  # sparse-effective

        delta_1 = None  # will be filled after m=1

        for m in M_VALUES:
            # Apply surgery to top m directions
            X_new_tr = apply_top_m_surgery(saved_X, saved_y, SURGERY_R, geo, m)
            if X_new_tr is None:
                log(f"  [m={m:2d}] INFEASIBLE")
                delta_m_actual.append(np.nan)
                delta_m_pred_A.append(np.nan)
                delta_m_pred_B.append(np.nan)
                delta_m_pred_C.append(np.nan)
                continue

            X_new_te = apply_top_m_surgery(saved_X_te, saved_y_te, SURGERY_R, geo, m)
            if X_new_te is None:
                log(f"  [m={m:2d}] test INFEASIBLE")
                delta_m_actual.append(np.nan)
                delta_m_pred_A.append(np.nan)
                delta_m_pred_B.append(np.nan)
                delta_m_pred_C.append(np.nan)
                continue

            q_new = compute_q(X_new_tr, saved_y, X_new_te, saved_y_te)
            logit_new = float(np.log(q_new / (1 - q_new + 1e-10) + 1e-10))
            delta = logit_new - logit_base
            delta_m_actual.append(delta)

            if m == 1:
                delta_1 = delta

            # Model A: nearest-only (constant = delta_1)
            pred_A = delta_1 if delta_1 is not None else np.nan
            delta_m_pred_A.append(pred_A)

            # Model B: equal-additive
            pred_B = m * delta_1 if delta_1 is not None else np.nan
            delta_m_pred_B.append(pred_B)

            # Model C: sparse-effective (cumulative softmax weight * total)
            # C_delta(m) = sum_{j=1}^m w_j * (full_formula / K-1) / w_j_norm
            # Simplified: scale each pair's contribution by softmax weight
            # Total effect at m=K-1: = delta_1 * K_eff_pred (observed)
            # Effect at m: = delta_1 * sum_{j=1}^m (w_j / w_1)  (relative to nearest)
            w_cum_sum = float(w_softmax[:m].sum())
            pred_C = delta_1 * (w_cum_sum / w_softmax[0]) if (delta_1 is not None and w_softmax[0] > 1e-12) else np.nan
            delta_m_pred_C.append(pred_C)

            log(f"  [m={m:2d}] delta={delta:.4f} | A={pred_A:.4f} | B={pred_B:.4f} | C={pred_C:.4f}")

            result = {
                "seed": seed, "best_epoch": best_epoch, "r": SURGERY_R, "m": m,
                "kappa_eff_base": float(saved_kappa_eff), "q_base": float(saved_q),
                "logit_base": float(logit_base),
                "q_new": float(q_new), "logit_new": float(logit_new), "delta_actual": float(delta),
                "delta_pred_A": float(pred_A) if not np.isnan(pred_A) else None,
                "delta_pred_B": float(pred_B) if not np.isnan(pred_B) else None,
                "delta_pred_C": float(pred_C) if not np.isnan(pred_C) else None,
                "kappa_j": float(kappas[m-1]) if m-1 < len(kappas) else None,
                "w_softmax_j": float(w_softmax[m-1]) if m-1 < len(w_softmax) else None,
            }
            all_results.append(result)

        # Seed summary
        valid = [(d, pA, pB, pC) for d, pA, pB, pC in
                 zip(delta_m_actual, delta_m_pred_A, delta_m_pred_B, delta_m_pred_C)
                 if not any(np.isnan(x) for x in [d, pA, pB, pC])]
        if valid:
            deltas, predsA, predsB, predsC = zip(*valid)
            deltas = np.array(deltas)
            predsA, predsB, predsC = np.array(predsA), np.array(predsB), np.array(predsC)
            SS = np.sum((deltas - deltas.mean())**2)
            R2_A = 1 - np.sum((deltas - predsA)**2) / SS if SS > 0 else 0
            R2_B = 1 - np.sum((deltas - predsB)**2) / SS if SS > 0 else 0
            R2_C = 1 - np.sum((deltas - predsC)**2) / SS if SS > 0 else 0
            # K_eff observed
            delta_total = max(deltas)
            K_eff_obs = delta_total / delta_1 if delta_1 and abs(delta_1) > 1e-6 else np.nan

            log(f"\n  SEED {seed} SUMMARY:")
            log(f"    delta_1 (m=1): {delta_1:.4f}")
            log(f"    delta_K-1 (m={K-1}): {max(deltas):.4f}")
            log(f"    K_eff_observed  = {K_eff_obs:.2f}")
            log(f"    K_eff_predicted = {K_eff_pred:.2f}  (from softmax weights)")
            log(f"    Model A R2 = {R2_A:.4f}  (nearest-only, should be ~0)")
            log(f"    Model B R2 = {R2_B:.4f}  (equal-additive)")
            log(f"    Model C R2 = {R2_C:.4f}  (sparse-effective, should be ~1)")
            match_pct = abs(K_eff_obs - K_eff_pred) / (abs(K_eff_pred) + 1e-6) * 100 if not np.isnan(K_eff_obs) else np.nan
            log(f"    K_eff match: |obs-pred|/pred = {match_pct:.1f}%  (pass if < 25%)")

    log(f"\n{'='*60}")
    log("FINAL SUMMARY")
    log(f"{'='*60}")
    log("Best model = Model with highest R2 across seeds")
    log("Nobel-track result if: K_eff_predicted matches K_eff_observed within 25%")

    with open(RESULT_PATH, "w", encoding="ascii", errors="replace") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {RESULT_PATH}")
    _logfile.close()


if __name__ == "__main__":
    main()
