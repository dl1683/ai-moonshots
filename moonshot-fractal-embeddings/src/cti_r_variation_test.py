#!/usr/bin/env python -u
"""
R-VARIATION COLLAPSE TEST (Feb 22 2026)
========================================
Tests whether c(r) = K_eff/d_eff follows the theoretical prediction:
  c(r) = 1 - 1/sqrt(r)    [Codex derivation from trace-preserving constraint]

PRE-REGISTERED PREDICTIONS (before any data):
  r=2:  c_pred = 1 - 1/sqrt(2) = 0.293
  r=5:  c_pred = 1 - 1/sqrt(5) = 0.553
  r=10: c_pred = 1 - 1/sqrt(10) = 0.684  [existing data: observed 0.66]

PASS CRITERION: |c_obs(r) - c_pred(r)| / c_pred(r) < 20% for each r

DESIGN:
1. Train 2 seeds of ResNet-18 on CIFAR-100 coarse (K=20)
2. At kappa_eff ~ 1, save embeddings
3. For each r in R_VALUES: run top-m sweep on SAME saved embeddings (no retraining)
4. Compute K_eff_obs(r) = delta_max / delta_1 and d_eff for each (seed, r)
5. Compute c(r) = K_eff_obs / d_eff
6. Compare to c_pred(r) = 1 - 1/sqrt(r)

FIX from Codex review:
- Use GLOBALLY nearest centroid pair as target (not fixed class 0)
- This avoids arbitrary class-0 bias

ADDITIONAL TEST - Curve Collapse:
- Plot delta(m)/delta_max vs m/d_eff for each (seed, r)
- If universal: all curves should collapse to same shape
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
SEEDS = [0, 1]
BATCH_SIZE = 256
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
N_EVAL_SUBSAMPLE = 5000

CHECKPOINT_EPOCHS = list(range(1, 16)) + list(range(20, 65, 5))
KAPPA_EFF_TARGET = 1.0
KAPPA_EFF_MIN = 0.5
KAPPA_EFF_MAX = 2.0

# R values to test: r=10 is already done (K_eff~0.66*d_eff)
# New: r=2 and r=5 to test the c(r) formula
R_VALUES = [2.0, 5.0, 10.0]
M_VALUES = list(range(1, K))  # m = 1, 2, ..., K-1 = 19

# Pre-registered predictions c_pred(r) = 1 - 1/sqrt(r)
C_PRED = {r: 1.0 - 1.0 / float(np.sqrt(r)) for r in R_VALUES}

RESULT_PATH = "results/cti_r_variation_test.json"
LOG_PATH = "results/cti_r_variation_test_log.txt"


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
# GEOMETRY (fixed: uses globally nearest centroid pair as target)
# ================================================================
def compute_full_geometry(X, y, K_classes=K):
    """
    Compute geometry for all K-1 competitor directions relative to the
    GLOBALLY NEAREST centroid pair (not fixed class 0).

    FIX from Codex review: class 0 is arbitrary; use the pair (i*, j*) that
    minimizes kappa_ij over all class pairs. This is the class pair the CTI
    law is actually about (kappa_nearest refers to this pair).
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

    # Find globally nearest centroid pair (i*, j*) = argmin_{i!=j} ||mu_i - mu_j|| / (sigma_W*sqrt(d))
    min_kappa = np.inf
    target_class_idx = 0
    nearest_competitor_idx = 1
    for i in range(K_classes):
        for j in range(K_classes):
            if i == j:
                continue
            diff = centroids[i] - centroids[j]
            dist = float(np.linalg.norm(diff))
            kappa_ij = dist / (sigma_W_global * np.sqrt(d))
            if kappa_ij < min_kappa:
                min_kappa = kappa_ij
                target_class_idx = i
                nearest_competitor_idx = j

    # Build competitor list from target_class_idx perspective
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
        "nearest_competitor_idx": nearest_competitor_idx,
        "competitor_info": competitor_info,
    }


# ================================================================
# TOP-M SURGERY
# ================================================================
def apply_top_m_surgery(X, y, r, geo, m):
    """Apply surgery to the TOP m competitor directions (sorted by kappa ascending)."""
    competitor_info = geo["competitor_info"]
    Sigma_W = geo["Sigma_W"]
    tr_W = geo["tr_W"]
    centroids = geo["centroids"]
    classes = np.unique(y)

    dirs = np.array([competitor_info[j]["dir_hat"] for j in range(m)])
    Q, R = np.linalg.qr(dirs.T, mode="reduced")
    U_sub = Q

    tr_sub = float(np.trace(U_sub.T @ Sigma_W @ U_sub))
    tr_orth = tr_W - tr_sub

    numerator = tr_W - tr_sub / r
    if tr_orth < 1e-12 or numerator < 0:
        return None  # infeasible

    scale_orth = float(np.sqrt(numerator / tr_orth))
    scale_sub = 1.0 / float(np.sqrt(r))

    X_new = X.copy()
    for i, c in enumerate(classes):
        mask = (y == c)
        x_c = X[mask] - centroids[i]
        coords_sub = x_c @ U_sub
        comp_sub = coords_sub @ U_sub.T
        comp_orth = x_c - comp_sub
        X_new[mask] = centroids[i] + scale_sub * comp_sub + scale_orth * comp_orth

    return X_new


# ================================================================
# THEORETICAL c(r) PREDICTION
# ================================================================
def c_pred(r):
    """Codex theoretical prediction: c(r) = 1 - 1/sqrt(r)"""
    return 1.0 - 1.0 / float(np.sqrt(r))


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

    log("R-Variation Collapse Test")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, SEEDS={SEEDS}")
    log(f"R_VALUES={R_VALUES}")
    log(f"M_VALUES=1..{K-1}")
    log(f"{'='*60}")
    log("PRE-REGISTERED PREDICTIONS (c_pred(r) = 1 - 1/sqrt(r)):")
    for r in R_VALUES:
        log(f"  r={r:.1f}: c_pred = {c_pred(r):.4f}")
    log(f"PASS CRITERION: |c_obs - c_pred| / c_pred < 0.20 for each r")
    log(f"{'='*60}")

    all_results = []
    per_seed_summary = {}

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

        # === STAGE 1: TRAIN ===
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
                    f"kappa={geo['kappa_nearest']:.4f}, target_cls={geo['target_class_idx']}, q={q:.4f}")

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
        d_eff = geo["d_eff"]
        log(f"\n  Selected checkpoint: epoch={best_epoch}, kappa_eff={saved_kappa_eff:.4f}")
        log(f"  Baseline: d_eff={d_eff:.3f}, kappa={geo['kappa_nearest']:.4f}, "
            f"target_cls={geo['target_class_idx']}, q={saved_q:.4f}")

        per_seed_summary[seed] = {"d_eff": d_eff, "r_results": {}}

        # === STAGE 2: R SWEEP (pure numpy, no retraining) ===
        log(f"\n  Starting r sweep on saved embeddings...")

        for r in R_VALUES:
            log(f"\n  --- r={r:.1f} ---")
            logit_base = float(np.log(max(saved_q, 1e-6) / max(1 - saved_q, 1e-6) + 1e-10))

            delta_m_actual = []
            delta_1 = None

            for m in M_VALUES:
                X_new_tr = apply_top_m_surgery(saved_X, saved_y, r, geo, m)
                if X_new_tr is None:
                    log(f"  [r={r:.1f} m={m:2d}] INFEASIBLE")
                    delta_m_actual.append(np.nan)
                    continue

                X_new_te = apply_top_m_surgery(saved_X_te, saved_y_te, r, geo, m)
                if X_new_te is None:
                    delta_m_actual.append(np.nan)
                    continue

                q_new = compute_q(X_new_tr, saved_y, X_new_te, saved_y_te)
                logit_new = float(np.log(max(q_new, 1e-6) / max(1 - q_new, 1e-6) + 1e-10))
                delta = logit_new - logit_base
                delta_m_actual.append(delta)

                if m == 1:
                    delta_1 = delta

                log(f"  [r={r:.1f} m={m:2d}] delta={delta:.4f}")

                result = {
                    "seed": seed, "r": r, "m": m,
                    "d_eff": float(d_eff),
                    "kappa_eff_base": float(saved_kappa_eff),
                    "q_base": float(saved_q),
                    "q_new": float(q_new),
                    "delta_actual": float(delta),
                    "best_epoch": best_epoch,
                }
                all_results.append(result)

            # Compute K_eff_obs and c_obs for this r
            valid_deltas = [d for d in delta_m_actual if not np.isnan(d)]
            if valid_deltas and delta_1 is not None and abs(delta_1) > 1e-6:
                delta_max = max(valid_deltas)
                K_eff_obs = delta_max / delta_1
                c_obs = K_eff_obs / d_eff
                c_predicted = c_pred(r)
                rel_error = abs(c_obs - c_predicted) / (c_predicted + 1e-10)
                passes = rel_error < 0.20

                log(f"\n  r={r:.1f} SUMMARY (seed={seed}):")
                log(f"    delta_1 = {delta_1:.4f}")
                log(f"    delta_max = {delta_max:.4f}")
                log(f"    K_eff_obs = {K_eff_obs:.3f}")
                log(f"    d_eff = {d_eff:.3f}")
                log(f"    c_obs = {c_obs:.4f}")
                log(f"    c_pred = {c_predicted:.4f}")
                log(f"    rel_error = {rel_error:.3f}  {'PASS' if passes else 'FAIL'} (threshold 0.20)")

                per_seed_summary[seed]["r_results"][r] = {
                    "K_eff_obs": float(K_eff_obs),
                    "c_obs": float(c_obs),
                    "c_pred": float(c_predicted),
                    "rel_error": float(rel_error),
                    "passes": passes,
                    "delta_m_curve": [float(d) if not np.isnan(d) else None for d in delta_m_actual],
                    "delta_1": float(delta_1),
                    "delta_max": float(delta_max),
                }
            else:
                log(f"  r={r:.1f}: Could not compute K_eff_obs (all INFEASIBLE or delta_1=0)")
                per_seed_summary[seed]["r_results"][r] = {"error": "infeasible or delta_1=0"}

    # === FINAL ANALYSIS ===
    log(f"\n{'='*60}")
    log("FINAL ANALYSIS: c(r) = K_eff/d_eff vs predicted 1 - 1/sqrt(r)")
    log(f"{'='*60}")

    log(f"\n{'r':>6} {'c_pred':>8} | {'seed 0 c_obs':>12} {'err%':>6} {'pass':>5} | "
        f"{'seed 1 c_obs':>12} {'err%':>6} {'pass':>5}")
    log(f"{'-'*75}")

    all_pass = True
    r_results_pooled = {}

    for r in R_VALUES:
        c_predicted = c_pred(r)
        s0 = per_seed_summary.get(0, {}).get("r_results", {}).get(r, {})
        s1 = per_seed_summary.get(1, {}).get("r_results", {}).get(r, {})

        c0 = s0.get("c_obs", float("nan"))
        e0 = s0.get("rel_error", float("nan"))
        p0 = s0.get("passes", False)
        c1 = s1.get("c_obs", float("nan"))
        e1 = s1.get("rel_error", float("nan"))
        p1 = s1.get("passes", False)

        both_pass = p0 and p1
        if not both_pass:
            all_pass = False

        log(f"{r:>6.1f} {c_predicted:>8.4f} | {c0:>12.4f} {e0*100:>5.1f}% {'PASS' if p0 else 'FAIL':>5} | "
            f"{c1:>12.4f} {e1*100:>5.1f}% {'PASS' if p1 else 'FAIL':>5}")

        r_results_pooled[r] = {
            "c_pred": float(c_predicted),
            "c_obs_s0": float(c0),
            "c_obs_s1": float(c1),
            "rel_error_s0": float(e0),
            "rel_error_s1": float(e1),
            "pass_s0": p0,
            "pass_s1": p1,
        }

    log(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'} -- c(r) = 1 - 1/sqrt(r) theory")

    # Curve collapse check: does delta(m)/delta_max vs m/d_eff collapse?
    log(f"\n{'='*60}")
    log("CURVE COLLAPSE CHECK: delta(m)/delta_max vs m/d_eff")
    log(f"{'='*60}")
    log("If curves collapse: all (r, seed) combinations show same shape")
    log("Expected shape: linear for m < K_eff, then saturation")

    # Save results
    final = {
        "experiment": "r_variation_collapse_test",
        "K": K,
        "seeds": SEEDS,
        "R_VALUES": R_VALUES,
        "c_pred_formula": "1 - 1/sqrt(r)",
        "per_seed_summary": {
            str(seed): {
                "d_eff": per_seed_summary.get(seed, {}).get("d_eff", None),
                "r_results": {
                    str(r): v for r, v in per_seed_summary.get(seed, {}).get("r_results", {}).items()
                }
            }
            for seed in SEEDS
        },
        "r_results_pooled": {str(k): v for k, v in r_results_pooled.items()},
        "overall_pass": all_pass,
        "results": all_results,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(final, f, indent=2)
    log(f"\nResults saved to {RESULT_PATH}")
    _logfile.close()


if __name__ == "__main__":
    main()
