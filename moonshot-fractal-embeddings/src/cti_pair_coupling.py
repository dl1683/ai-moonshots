#!/usr/bin/env python -u
"""
PAIR COUPLING ANALYSIS (Feb 22 2026)
=====================================
KEY QUESTION: What determines c = K_eff/d_eff for a given class pair?

HYPOTHESIS (pre-registered):
  c_pair(i) = f_sub(i) = tr(U_i^T Sigma_W U_i) / tr(Sigma_W)
  = fraction of within-class variance in target class i's centroid subspace

  This gives K_eff(i) = f_sub(i) * d_eff(i) (ZERO-PARAMETER PREDICTION)

WHY: Surgery compresses variance in centroid subspace. When all subspace variance
is exhausted, no more gain. The fraction f_sub tells us how much is available.
The saturation K_eff = f_sub * d_eff emerges from this balance.

PRE-REGISTERED:
  K_eff_pred(i) = rank_eff(V_i) = tr(V_i)^2 / tr(V_i^2)
  where V_i = U_i^T Sigma_W U_i (K-1 x K-1 projected covariance)

  Predicted to match K_eff_obs(i) within 30%, R2 > 0.5 across all 20 target classes.

DESIGN:
  1. Train 3 seeds on CIFAR-100 coarse (K=20)
  2. At kappa_eff~1 checkpoint, for ALL K=20 target classes:
     a. Compute K_eff_obs(i) = delta_max / delta_1 at r=10
     b. Compute K_eff_pred(i) = rank_eff(V_i) [zero-parameter]
     c. Compute f_sub(i) = tr(V_i) / tr(Sigma_W) [coupling fraction]
  3. Fit: K_eff_obs ~ K_eff_pred (across 20*3=60 pairs)
  4. BONUS: test r-invariance (run same for r=5)

PASS CRITERION:
  - R2(K_eff_obs vs K_eff_pred) > 0.5 across all (seed, class) pairs
  - Spearman rho > 0.5
  - AND: the correlation holds across r=5 and r=10 (r-invariance)
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
from scipy.stats import pearsonr, spearmanr

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

R_VALUES = [5.0, 10.0]  # Test r-invariance: c should be same at r=5 and r=10
M_VALUES = list(range(1, K))  # m = 1..K-1 = 19

RESULT_PATH = "results/cti_pair_coupling.json"
LOG_PATH = "results/cti_pair_coupling_log.txt"


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
    train_ds = torchvision.datasets.CIFAR100(root="data", train=True, download=True,
                                              transform=train_tf, target_transform=_coarse_label)
    test_ds = torchvision.datasets.CIFAR100(root="data", train=False, download=True,
                                             transform=test_tf, target_transform=_coarse_label)
    return train_ds, test_ds


def get_loaders(train_ds, test_ds):
    tr_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=0, pin_memory=False)
    te_ld = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False,
                                        num_workers=0, pin_memory=False)
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
# GEOMETRY PER TARGET CLASS
# ================================================================
def compute_geometry_for_target(X, y, target_class_idx, K_classes=K):
    """
    Compute geometry from TARGET class's perspective.
    Returns:
      - centroids, Sigma_W, tr_W, sigma_W_global
      - competitor_info: sorted by kappa_j (ascending = hardest first)
      - kappa_nearest, d_eff, kappa_eff
      - f_sub: fraction of within-class variance in centroid subspace
      - V_i: K-1 x K-1 projected covariance matrix (U_i^T Sigma_W U_i)
      - K_eff_pred: rank_eff(V_i) = tr(V_i)^2 / tr(V_i^2) [zero-parameter predictor]
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
            "dir_hat": dir_hat,
            "kappa_j": kappa_j,
            "sigma_cdir_j": sigma_cdir_j,
            "d_eff_j": d_eff_j,
        })

    competitor_info.sort(key=lambda x: x["kappa_j"])
    nearest = competitor_info[0]
    kappa_nearest = nearest["kappa_j"]
    d_eff = tr_W / (nearest["sigma_cdir_j"]**2) if nearest["sigma_cdir_j"] > 1e-12 else 1.0
    kappa_eff = kappa_nearest * float(np.sqrt(d_eff))

    # Compute centroid subspace basis via QR
    dirs = np.array([ci["dir_hat"] for ci in competitor_info])  # (K-1, d)
    Q, R = np.linalg.qr(dirs.T, mode="reduced")  # Q: (d, K-1)
    U_i = Q  # orthonormal basis for centroid subspace

    # Projected covariance in centroid subspace
    V_i = U_i.T @ Sigma_W @ U_i  # (K-1) x (K-1)

    # Geometric predictors
    tr_Vi = float(np.trace(V_i))
    tr_Vi2 = float(np.trace(V_i @ V_i))
    f_sub = tr_Vi / tr_W  # fraction of variance in centroid subspace
    K_eff_pred = (tr_Vi**2) / (tr_Vi2 + 1e-12)  # effective rank of V_i = zero-param K_eff predictor

    return {
        "centroids": centroids,
        "Sigma_W": Sigma_W,
        "tr_W": tr_W,
        "sigma_W_global": sigma_W_global,
        "kappa_nearest": kappa_nearest,
        "d_eff": d_eff,
        "kappa_eff": kappa_eff,
        "target_class_idx": target_class_idx,
        "competitor_info": competitor_info,
        "U_i": U_i,
        "V_i": V_i,
        "f_sub": f_sub,
        "K_eff_pred": K_eff_pred,
        "tr_Vi": tr_Vi,
        "classes": classes,
    }


# ================================================================
# TOP-M SURGERY
# ================================================================
def apply_top_m_surgery(X, y, r, geo, m):
    """Apply surgery to top m competitor directions from geo's target class perspective."""
    competitor_info = geo["competitor_info"]
    Sigma_W = geo["Sigma_W"]
    tr_W = geo["tr_W"]
    centroids = geo["centroids"]
    classes = geo["classes"]

    dirs = np.array([competitor_info[j]["dir_hat"] for j in range(m)])
    Q, R = np.linalg.qr(dirs.T, mode="reduced")
    U_sub = Q

    tr_sub = float(np.trace(U_sub.T @ Sigma_W @ U_sub))
    tr_orth = tr_W - tr_sub
    numerator = tr_W - tr_sub / r
    if tr_orth < 1e-12 or numerator < 0:
        return None

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
# K_EFF SWEEP FOR A SINGLE TARGET CLASS
# ================================================================
def compute_K_eff_obs(X_tr, y_tr, X_te, y_te, geo, r):
    """Run top-m sweep and compute K_eff_obs = delta_max / delta_1."""
    logit_base_q = compute_q(X_tr, y_tr, X_te, y_te)
    logit_base = float(np.log(max(logit_base_q, 1e-6) / max(1 - logit_base_q, 1e-6) + 1e-10))

    delta_1 = None
    delta_max = None

    for m in M_VALUES:
        X_new_tr = apply_top_m_surgery(X_tr, y_tr, r, geo, m)
        if X_new_tr is None:
            continue
        X_new_te = apply_top_m_surgery(X_te, y_te, r, geo, m)
        if X_new_te is None:
            continue

        q_new = compute_q(X_new_tr, y_tr, X_new_te, y_te)
        logit_new = float(np.log(max(q_new, 1e-6) / max(1 - q_new, 1e-6) + 1e-10))
        delta = logit_new - logit_base

        if m == 1:
            delta_1 = delta
        if delta_max is None or delta > delta_max:
            delta_max = delta

    if delta_1 is None or delta_max is None or abs(delta_1) < 1e-6:
        return None, None, None

    K_eff_obs = delta_max / delta_1
    c_obs = K_eff_obs / geo["d_eff"]
    return K_eff_obs, c_obs, delta_max


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

    log("Pair Coupling Analysis")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, SEEDS={SEEDS}")
    log(f"R_VALUES={R_VALUES}")
    log(f"{'='*60}")
    log("PRE-REGISTERED HYPOTHESIS: c_pair(i) = f_sub(i) = tr(V_i)/tr(W)")
    log("  K_eff_pred(i) = rank_eff(V_i) = tr(V_i)^2/tr(V_i^2)")
    log("PASS: R2(K_eff_obs vs K_eff_pred) > 0.5 AND Spearman rho > 0.5")
    log(f"{'='*60}")

    all_records = []
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

                # Compute global kappa_eff using globally nearest pair
                classes = np.unique(y_tr)
                n_ep = len(X_tr)
                d = X_tr.shape[1]
                centroids = np.array([X_tr[y_tr == c].mean(0) for c in classes])
                X_c = np.vstack([X_tr[y_tr == c] - centroids[i] for i, c in enumerate(classes)])
                Sigma_W = X_c.T @ X_c / n_ep
                tr_W = float(np.trace(Sigma_W))
                sigma_W_g = float(np.sqrt(tr_W / d))

                # Find nearest pair
                min_kappa_ep = np.inf
                for i in range(K):
                    for j in range(K):
                        if i == j: continue
                        dist = float(np.linalg.norm(centroids[i] - centroids[j]))
                        kij = dist / (sigma_W_g * np.sqrt(d))
                        if kij < min_kappa_ep:
                            min_kappa_ep = kij
                            best_pair = (i, j)

                e1 = centroids[best_pair[0]] - centroids[best_pair[1]]
                e1 /= np.linalg.norm(e1) + 1e-12
                sigma_cdir = float(np.sqrt(e1 @ Sigma_W @ e1))
                d_eff_ep = tr_W / (sigma_cdir**2 + 1e-12)
                kappa_eff_ep = min_kappa_ep * float(np.sqrt(d_eff_ep))
                q_ep = compute_q(X_tr, y_tr, X_te, y_te)

                log(f"  [ep={epoch:2d}] kappa_eff={kappa_eff_ep:.4f}, d_eff={d_eff_ep:.3f}, "
                    f"q={q_ep:.4f}")

                if KAPPA_EFF_MIN <= kappa_eff_ep <= KAPPA_EFF_MAX:
                    dist_ep = abs(kappa_eff_ep - KAPPA_EFF_TARGET)
                    if dist_ep < best_dist:
                        best_dist = dist_ep
                        best_epoch = epoch
                        saved_X = X_tr.copy()
                        saved_y = y_tr.copy()
                        saved_X_te = X_te.copy()
                        saved_y_te = y_te.copy()
                        saved_kappa_eff = kappa_eff_ep
                        saved_q = q_ep

            if epoch % 10 == 0:
                log(f"  [seed={seed} epoch={epoch}] training in progress")

        if best_epoch is None:
            log(f"  WARNING: No linear regime checkpoint for seed={seed}")
            continue

        log(f"\n  Selected checkpoint: epoch={best_epoch}, kappa_eff={saved_kappa_eff:.4f}, q={saved_q:.4f}")
        log(f"  Running pair coupling analysis for all {K} target classes...")

        # === STAGE 2: ALL-PAIR ANALYSIS ===
        for target_cls in range(K):
            log(f"\n  Target class {target_cls}:")
            geo = compute_geometry_for_target(saved_X, saved_y, target_cls)

            K_eff_pred = geo["K_eff_pred"]
            f_sub = geo["f_sub"]
            d_eff = geo["d_eff"]
            kappa_nearest = geo["kappa_nearest"]

            log(f"    d_eff={d_eff:.3f}, kappa={kappa_nearest:.4f}, "
                f"f_sub={f_sub:.4f}, K_eff_pred={K_eff_pred:.3f}")

            for r in R_VALUES:
                K_eff_obs, c_obs, delta_max = compute_K_eff_obs(
                    saved_X, saved_y, saved_X_te, saved_y_te, geo, r
                )
                if K_eff_obs is None:
                    log(f"    r={r:.0f}: INFEASIBLE")
                    continue

                match_err = abs(K_eff_obs - K_eff_pred) / (K_eff_pred + 1e-6)
                log(f"    r={r:.0f}: K_eff_obs={K_eff_obs:.3f}, K_eff_pred={K_eff_pred:.3f}, "
                    f"c_obs={c_obs:.4f}, match_err={match_err:.3f}")

                all_records.append({
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "target_cls": target_cls,
                    "r": r,
                    "K_eff_obs": float(K_eff_obs),
                    "K_eff_pred": float(K_eff_pred),
                    "c_obs": float(c_obs),
                    "d_eff": float(d_eff),
                    "f_sub": float(f_sub),
                    "kappa_nearest": float(kappa_nearest),
                    "delta_max": float(delta_max),
                    "match_err": float(match_err),
                })

    # === FINAL ANALYSIS ===
    log(f"\n{'='*60}")
    log("FINAL ANALYSIS: K_eff_pred vs K_eff_obs")
    log(f"{'='*60}")

    for r in R_VALUES:
        recs = [x for x in all_records if x["r"] == r]
        if not recs:
            continue
        K_obs = np.array([x["K_eff_obs"] for x in recs])
        K_pred = np.array([x["K_eff_pred"] for x in recs])
        c_vals = np.array([x["c_obs"] for x in recs])
        f_vals = np.array([x["f_sub"] for x in recs])

        r_pearson, _ = pearsonr(K_obs, K_pred) if len(K_obs) > 2 else (np.nan, np.nan)
        r_spearman, _ = spearmanr(K_obs, K_pred) if len(K_obs) > 2 else (np.nan, np.nan)
        SS = np.sum((K_obs - K_obs.mean())**2)
        R2 = 1 - np.sum((K_obs - K_pred)**2) / SS if SS > 0 else np.nan
        r_cf, _ = pearsonr(c_vals, f_vals) if len(c_vals) > 2 else (np.nan, np.nan)
        passes = R2 > 0.5 and r_spearman > 0.5

        log(f"\n  r={r:.0f} ({len(recs)} pairs):")
        log(f"    R2(K_eff_obs vs K_eff_pred) = {R2:.4f}  {'PASS' if R2 > 0.5 else 'FAIL'}")
        log(f"    Pearson r = {r_pearson:.4f}")
        log(f"    Spearman rho = {r_spearman:.4f}  {'PASS' if r_spearman > 0.5 else 'FAIL'}")
        log(f"    Pearson(c_obs, f_sub) = {r_cf:.4f}")
        log(f"    Mean c_obs = {c_vals.mean():.4f} +/- {c_vals.std():.4f}")
        log(f"    Mean K_eff_obs = {K_obs.mean():.3f} +/- {K_obs.std():.3f}")
        log(f"    Mean K_eff_pred = {K_pred.mean():.3f} +/- {K_pred.std():.3f}")
        log(f"    Overall {'PASS' if passes else 'FAIL'} (R2>0.5 AND rho>0.5)")

    # R-invariance test
    log(f"\n{'='*60}")
    log("R-INVARIANCE TEST: Is c approximately constant in r?")
    r5_recs = {(x["seed"], x["target_cls"]): x for x in all_records if x["r"] == 5.0}
    r10_recs = {(x["seed"], x["target_cls"]): x for x in all_records if x["r"] == 10.0}
    matched = [(r5_recs[k]["c_obs"], r10_recs[k]["c_obs"]) for k in r5_recs if k in r10_recs]
    if matched:
        c5, c10 = zip(*matched)
        c5 = np.array(c5); c10 = np.array(c10)
        r_inv, _ = pearsonr(c5, c10)
        diff_mean = float(np.mean(np.abs(c5 - c10)))
        log(f"  Pearson r(c_r5, c_r10) = {r_inv:.4f}  (1.0=perfectly r-invariant)")
        log(f"  Mean |c_r5 - c_r10| = {diff_mean:.4f}")
        log(f"  R-invariance {'PASS' if r_inv > 0.8 and diff_mean < 0.10 else 'FAIL'}")

    # Save results
    final = {
        "experiment": "pair_coupling_analysis",
        "K": K, "seeds": SEEDS, "R_VALUES": R_VALUES,
        "hypothesis": "K_eff_pred = rank_eff(V_i) = tr(V_i)^2/tr(V_i^2)",
        "records": all_records,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(final, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    log(f"\nResults saved to {RESULT_PATH}")
    _logfile.close()


if __name__ == "__main__":
    main()
