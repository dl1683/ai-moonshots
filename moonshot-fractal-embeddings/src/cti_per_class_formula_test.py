"""
Per-Class Formula Test: Unified CTI Law Validation
===================================================
Tests which formula best predicts PER-CLASS logit(q_i) across 20 classes.

Models compared:
- M0: logit(q_i) = A * kappa_i * sqrt(d_eff_i)           [original cross-model law]
- M2: logit(q_i) = A * kappa_i * sqrt(d_eff_i/K_eff_obs_i) [unified formula]
- M3: logit(q_i) = A * kappa_i / sqrt(K_eff_obs_i)         [pure 2-layer, no d_eff]
- M4: logit(q_i) = A * kappa_i                              [simplest baseline]

Where:
- q_i = per-class recall (fraction of test samples of class i that 1-NN assigns to class i)
- logit(q_i) = log(q_i / (1 - q_i))  [un-normalized, raw recall]
- kappa_i = min_j ||mu_i - mu_j|| / (sigma_W * sqrt(d))  (per-class SNR)
- d_eff_i = tr(Sigma_W) / sigma^2_{nearest_centroid_dir_i}  (per-class anisotropy)
- K_eff_obs_i = measured effective competitors from top-m surgery at r=5

PRE-REGISTERED:
- M2 R2 > M0 R2 (unified formula beats original within-class)
- Sign test: regression coefficient of d_eff_i for M0 is POSITIVE
             regression coefficient of K_eff_obs_i for M2/M3 denominator is POSITIVE (inverse)

Design:
- 3 seeds, 30 epochs each, ResNet-18 on CIFAR-20 coarse labels
- Checkpoint at kappa ~ 1.0-3.0 (similar to pair coupling)
- 20 classes x 3 seeds = 60 per-class data points
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
from scipy.special import logit as logit_fn
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import lstsq

K = 20
D = 512
N_EPOCHS = 30     # shorter than pair coupling (checkpoint found faster)
N_SEEDS = 3
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_SURGERY = 5.0   # Surgery ratio for K_eff_obs
M_VALUES = [1, 3, 6, 10, 15, 19]  # m values for surgery sweep (fewer than full for speed)
LOG_FILE = "results/cti_per_class_formula_log.txt"
RESULTS_FILE = "results/cti_per_class_formula.json"

COARSE_LABELS = [
    [72, 4, 95, 30, 55], [73, 32, 67, 91, 1], [92, 70, 82, 54, 62],
    [16, 61, 9, 10, 28], [51, 0, 53, 57, 83], [40, 39, 22, 87, 86],
    [20, 25, 94, 84, 5], [14, 24, 6, 7, 18], [43, 97, 42, 3, 88],
    [37, 17, 76, 12, 68], [49, 33, 71, 23, 60], [15, 21, 19, 31, 38],
    [75, 63, 66, 64, 34], [77, 26, 45, 99, 79], [11, 2, 35, 46, 98],
    [29, 93, 27, 78, 44], [65, 50, 74, 36, 80], [56, 52, 47, 59, 96],
    [8, 58, 90, 13, 48], [81, 69, 41, 89, 85],
]
FINE_TO_COARSE = {}
for c, group in enumerate(COARSE_LABELS):
    for f in group:
        FINE_TO_COARSE[f] = c


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="ascii") as f:
        f.write(msg + "\n")


class CoarseDataset(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset):
        self.dataset = cifar_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, fine = self.dataset[idx]
        return img, FINE_TO_COARSE[fine]


def make_loaders(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    mean = [0.5071, 0.4867, 0.4408]
    std  = [0.2675, 0.2565, 0.2761]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    raw_tr = torchvision.datasets.CIFAR100(root="data", train=True,  download=True, transform=train_tf)
    raw_te = torchvision.datasets.CIFAR100(root="data", train=False, download=True, transform=test_tf)
    tr_loader = torch.utils.data.DataLoader(CoarseDataset(raw_tr), batch_size=256, shuffle=True,
                                             num_workers=0, pin_memory=False, generator=g)
    te_loader = torch.utils.data.DataLoader(CoarseDataset(raw_te), batch_size=256, shuffle=False,
                                             num_workers=0, pin_memory=False)
    return tr_loader, te_loader


class ResNet18Embed(nn.Module):
    def __init__(self, num_classes=K):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        net.conv1  = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(512, num_classes)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.fc = net.fc
    def forward(self, x):
        z = self.backbone(x).flatten(1)
        return z, self.fc(z)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        _, logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        n += len(imgs)
    return total_loss / n


def extract(model, loader):
    model.eval()
    embs, lbls = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            embs.append(model(imgs.to(DEVICE))[0].cpu().numpy())
            lbls.append(labels.numpy())
    return np.concatenate(embs), np.concatenate(lbls)


def compute_per_class_q(X_tr, y_tr, X_te, y_te):
    """Compute per-class recall (q_i = fraction of class i test correctly classified by 1-NN)."""
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X_tr, y_tr)
    preds = knn.predict(X_te)
    per_class_q = []
    for c in range(K):
        mask = y_te == c
        if mask.sum() == 0:
            per_class_q.append(np.nan)
        else:
            recall_c = float((preds[mask] == c).mean())
            # Normalize: (recall - 1/K) / (1 - 1/K)
            q_c = (recall_c - 1.0/K) / (1.0 - 1.0/K)
            per_class_q.append(q_c)
    return np.array(per_class_q)


def compute_geometry(X_tr, y_tr):
    """Compute per-class geometry: kappa, d_eff, V_i, K_eff_in, f_sub."""
    d = X_tr.shape[1]
    centroids = np.array([X_tr[y_tr == k].mean(0) for k in range(K)])
    X_c = np.zeros_like(X_tr)
    for k in range(K):
        X_c[y_tr == k] = X_tr[y_tr == k] - centroids[k]
    Sigma_W = (X_c.T @ X_c) / len(X_tr)
    tr_W = float(np.trace(Sigma_W))
    sigma_W = float(np.sqrt(tr_W / d))

    # Pre-compute all centroid directions
    dists_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                dists_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    geo = []
    for i in range(K):
        kappa_ij = dists_matrix[i] / (sigma_W * np.sqrt(d))  # shape (K,)
        kappa_ij[i] = np.inf
        kappa_nearest = float(np.min(kappa_ij[np.isfinite(kappa_ij)]))
        j_nearest = int(np.argmin(kappa_ij))

        # d_eff_i: anisotropy in direction toward nearest competitor
        dir_hat = centroids[j_nearest] - centroids[i]
        norm = np.linalg.norm(dir_hat)
        dir_hat = dir_hat / (norm + 1e-10)
        sigma_cdir_sq = float(dir_hat @ Sigma_W @ dir_hat)
        d_eff_i = tr_W / sigma_cdir_sq if sigma_cdir_sq > 1e-10 else float(d)

        # Centroid subspace: K-1 orthogonal directions from class i to all other classes
        # Use QR to get orthonormal basis
        dirs_matrix = np.zeros((d, K-1))
        k_idx = 0
        for j in range(K):
            if j != i:
                v = centroids[j] - centroids[i]
                v_norm = np.linalg.norm(v)
                dirs_matrix[:, k_idx] = v / (v_norm + 1e-10)
                k_idx += 1
        # QR decomposition for orthonormal basis of centroid subspace
        Q, _ = np.linalg.qr(dirs_matrix, mode="reduced")  # d x (K-1)

        # V_i = Q^T Sigma_W Q  (K-1 x K-1 projected covariance in centroid subspace)
        V_i = Q.T @ Sigma_W @ Q
        tr_Vi = float(np.trace(V_i))
        tr_Vi2 = float(np.sum(V_i ** 2))  # tr(V_i^2) = sum of squared entries for PSD matrix
        K_eff_in_i = (tr_Vi ** 2) / (tr_Vi2 + 1e-10) if tr_Vi2 > 0 else 1.0
        f_sub_i = tr_Vi / (tr_W + 1e-10)

        geo.append({
            "class": i,
            "kappa_nearest": kappa_nearest,
            "j_nearest": j_nearest,
            "d_eff": d_eff_i,
            "K_eff_in": K_eff_in_i,  # theoretical (from V_i), should be ~constant (Neural Collapse)
            "f_sub": f_sub_i,
            "tr_Vi": tr_Vi,
        })

    return centroids, Sigma_W, tr_W, sigma_W, geo


def apply_surgery(X_tr, y_tr, X_te, y_te, centroids, Sigma_W, target_cls, r, m):
    """Apply top-m surgery: scale top-m centroid directions by 1/r (reduce separation variance)."""
    d = X_tr.shape[1]
    tr_W = float(np.trace(Sigma_W))

    # Sort competitors by DISTANCE (nearest first)
    dists = [(j, float(np.linalg.norm(centroids[target_cls] - centroids[j])))
             for j in range(K) if j != target_cls]
    dists.sort(key=lambda x: x[1])

    # Top-m centroid directions
    dirs = [(centroids[target_cls] - centroids[j]) / (np.linalg.norm(centroids[target_cls] - centroids[j]) + 1e-10)
            for j, _ in dists[:m]]
    M_mat = np.array(dirs).T  # d x m
    if m == 1:
        Q = M_mat / (np.linalg.norm(M_mat) + 1e-10)
    else:
        Q, _ = np.linalg.qr(M_mat, mode="reduced")  # d x m

    tr_sub  = float(np.trace(Q.T @ Sigma_W @ Q))
    tr_orth = tr_W - tr_sub
    scale_sub  = 1.0 / np.sqrt(r)
    tr_sub_new = tr_sub / r
    scale_orth = float(np.sqrt((tr_W - tr_sub_new) / (tr_orth + 1e-10))) if tr_orth > 1e-10 else 1.0

    def transform(X):
        X_c = X - centroids[target_cls]
        proj_sub  = (X_c @ Q) @ Q.T
        proj_orth = X_c - proj_sub
        return centroids[target_cls] + scale_sub * proj_sub + scale_orth * proj_orth

    X_tr_new = X_tr.copy()
    X_te_new = X_te.copy()
    X_tr_new[y_tr == target_cls] = transform(X_tr[y_tr == target_cls])
    X_te_new[y_te == target_cls] = transform(X_te[y_te == target_cls])
    return X_tr_new, X_te_new


def measure_K_eff_obs(X_tr, y_tr, X_te, y_te, centroids, Sigma_W, target_cls):
    """Measure K_eff_obs for target class via top-m surgery sweep."""
    knn_base = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn_base.fit(X_tr, y_tr)
    preds_base = knn_base.predict(X_te)
    mask_te = y_te == target_cls
    q_base_cls = float((preds_base[mask_te] == target_cls).mean())
    logit_base = float(logit_fn(np.clip(q_base_cls, 1e-6, 1 - 1e-6)))

    deltas = []
    for m in M_VALUES:
        X_tr_s, X_te_s = apply_surgery(X_tr, y_tr, X_te, y_te, centroids, Sigma_W,
                                        target_cls, R_SURGERY, m)
        knn_s = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
        knn_s.fit(X_tr_s, y_tr)
        preds_s = knn_s.predict(X_te_s)
        q_new_cls = float((preds_s[mask_te] == target_cls).mean())
        logit_new = float(logit_fn(np.clip(q_new_cls, 1e-6, 1 - 1e-6)))
        deltas.append(logit_new - logit_base)

    if not deltas or deltas[0] <= 0:
        return None, float(q_base_cls)

    delta_1   = deltas[0]
    delta_max = max(deltas)
    K_eff_obs = delta_max / (delta_1 + 1e-10)
    return float(K_eff_obs), float(q_base_cls)


def analyze_models(records):
    """Test M0 vs M2 vs M3 vs M4 head-to-head."""
    valid = [r for r in records
             if r["logit_q"] is not None and np.isfinite(r["logit_q"])]
    if len(valid) < 5:
        return {}

    logits  = np.array([r["logit_q"] for r in valid])
    kappa   = np.array([r["kappa_nearest"] for r in valid])
    d_eff   = np.array([r["d_eff"] for r in valid])
    K_eff_in = np.array([r["K_eff_in"] for r in valid])

    results = {}
    for name, pred in [
        ("M0_kappa_sqrt_deff",         kappa * np.sqrt(d_eff)),
        ("M4_kappa_only",              kappa),
        ("M5_kappa_sqrt_deff_over_Kin", kappa * np.sqrt(d_eff / K_eff_in)),
        ("deff_only",                  d_eff),
        ("kappa_sq",                   kappa ** 2),
    ]:
        X_fit = np.column_stack([pred, np.ones(len(valid))])
        coeffs, _, _, _ = lstsq(X_fit, logits, rcond=None)
        A_fit, C_fit = coeffs
        pred_fit = A_fit * pred + C_fit
        ss_res = np.sum((logits - pred_fit)**2)
        ss_tot = np.sum((logits - logits.mean())**2)
        R2 = 1 - ss_res / ss_tot
        r_pearson, p_pearson = stats.pearsonr(pred, logits)
        results[name] = {"R2": float(R2), "r_pearson": float(r_pearson),
                         "A_fit": float(A_fit), "C_fit": float(C_fit), "n": len(valid)}

    # Joint additive model: logit ~ a*kappa + b*d_eff + c
    X_joint = np.column_stack([kappa, d_eff, np.ones(len(valid))])
    c_joint, _, _, _ = lstsq(X_joint, logits, rcond=None)
    pred_joint = X_joint @ c_joint
    ss_res_j = np.sum((logits - pred_joint)**2)
    ss_tot_j = np.sum((logits - logits.mean())**2)
    R2_joint = 1 - ss_res_j / ss_tot_j
    results["joint_kappa_deff"] = {"R2": float(R2_joint), "a_kappa": float(c_joint[0]),
                                    "a_deff": float(c_joint[1]), "c": float(c_joint[2]), "n": len(valid)}

    # Partial correlation: d_eff controlling for kappa
    X_k = np.column_stack([kappa, np.ones(len(valid))])
    res_logit = logits - (X_k @ lstsq(X_k, logits, rcond=None)[0])
    res_deff  = d_eff  - (X_k @ lstsq(X_k, d_eff,  rcond=None)[0])
    pc, pcp = stats.pearsonr(res_deff, res_logit)
    results["partial_r_deff_given_kappa"] = {"r": float(pc), "p": float(pcp),
                                              "sign": "POSITIVE" if pc > 0 else "NEGATIVE"}

    # Spearman correlations
    for name, arr in [("kappa", kappa), ("d_eff", d_eff),
                      ("kappa_sqrt_deff", kappa * np.sqrt(d_eff))]:
        rho, p = stats.spearmanr(arr, logits)
        results[f"spearman_{name}"] = {"rho": float(rho), "p": float(p)}

    return results


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log("=" * 70)
    log("Per-Class CTI Formula Test")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, SEEDS={SEEDS}")
    log(f"M_VALUES={M_VALUES}, R_SURGERY={R_SURGERY}")
    log("=" * 70)
    log("PRE-REGISTERED:")
    log("  M2 R2 > M0 R2  (kappa*sqrt(d_eff/K_eff_obs) beats kappa*sqrt(d_eff))")
    log("  rho(kappa/sqrt(K_eff_obs), logit_q) > 0.5")
    log("=" * 70)

    all_records = []

    for seed in SEEDS:
        log(f"\n{'='*60}")
        log(f"SEED {seed}")
        log(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model     = ResNet18Embed().to(DEVICE)
        tr_loader, te_loader = make_loaders(seed)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        best_X_tr = best_X_te = best_y_tr = best_y_te = None
        best_kappa = None

        for epoch in range(1, N_EPOCHS + 1):
            loss = train_epoch(model, tr_loader, optimizer, criterion)
            scheduler.step()

            if epoch % 5 == 0 or epoch <= 15:
                X_tr_ep, y_tr_ep = extract(model, tr_loader)
                X_te_ep, y_te_ep = extract(model, te_loader)
                d = X_tr_ep.shape[1]
                centroids_ep = np.array([X_tr_ep[y_tr_ep == k].mean(0) for k in range(K)])
                X_c_ep = np.zeros_like(X_tr_ep)
                for k in range(K):
                    X_c_ep[y_tr_ep == k] = X_tr_ep[y_tr_ep == k] - centroids_ep[k]
                Sigma_ep = (X_c_ep.T @ X_c_ep) / len(X_tr_ep)
                tr_W_ep = float(np.trace(Sigma_ep))
                sigma_W_ep = float(np.sqrt(tr_W_ep / d))

                min_kappa = np.inf
                for i in range(K):
                    for j in range(i+1, K):
                        dist = float(np.linalg.norm(centroids_ep[i] - centroids_ep[j]))
                        kij = dist / (sigma_W_ep * np.sqrt(d))
                        if kij < min_kappa:
                            min_kappa = kij

                q_ep = (KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(X_tr_ep, y_tr_ep)
                        .score(X_te_ep, y_te_ep) - 1.0/K) / (1.0 - 1.0/K)

                log(f"  [ep={epoch:3d}] loss={loss:.4f}, kappa={min_kappa:.4f}, q={q_ep:.4f}")

                # Target: kappa in 0.3-1.0 range (BEFORE ETF convergence)
                # At kappa > 1.5, Neural Collapse makes all per-class kappas uniform
                # -> K_eff_kappa = K-1 = constant, no variation to test
                if 0.3 <= min_kappa <= 1.0 and (best_kappa is None or abs(min_kappa - 0.7) < abs(best_kappa - 0.7)):
                    best_kappa = min_kappa
                    best_X_tr  = X_tr_ep.copy()
                    best_y_tr  = y_tr_ep.copy()
                    best_X_te  = X_te_ep.copy()
                    best_y_te  = y_te_ep.copy()

        if best_X_tr is None:
            log(f"  No kappa in [0.3,1.0] range — using last epoch")
            best_X_tr, best_y_tr = extract(model, tr_loader)
            best_X_te, best_y_te = extract(model, te_loader)

        log(f"\n  [GEOMETRY] Computing per-class geometry at best checkpoint...")
        centroids, Sigma_W, tr_W, sigma_W, geo = compute_geometry(best_X_tr, best_y_tr)

        log(f"  sigma_W={sigma_W:.4f}, tr_W={tr_W:.2f}")
        log(f"  {'cls':>4} {'kappa':>8} {'d_eff':>8} {'K_eff_in':>10} {'f_sub':>7}")
        for g in geo:
            log(f"  {g['class']:>4d} {g['kappa_nearest']:>8.4f} {g['d_eff']:>8.2f} "
                f"{g['K_eff_in']:>10.3f} {g['f_sub']:>7.4f}")

        # Per-class recall
        log(f"\n  [Q_i] Computing per-class recall...")
        q_per_class = compute_per_class_q(best_X_tr, best_y_tr, best_X_te, best_y_te)
        for c in range(K):
            if np.isfinite(q_per_class[c]):
                log(f"  class {c}: q_i={q_per_class[c]:.4f}, logit={logit_fn(np.clip(q_per_class[c],1e-6,1-1e-6)):.4f}")

        # Per-class records (WITHOUT K_EFF_obs surgery — too slow)
        # K_EFF_in is CONSTANT (Neural Collapse), so skip surgery
        log(f"\n  [RECORDS] Building per-class records (no K_EFF_obs surgery - constant K_eff_in)...")
        for target_cls in range(K):
            g = geo[target_cls]
            q_i = float(q_per_class[target_cls]) if np.isfinite(q_per_class[target_cls]) else None
            logit_q_val = float(logit_fn(np.clip(q_i, 1e-6, 1-1e-6))) if q_i is not None else None

            rec = {
                "seed": seed,
                "class": target_cls,
                "q_i": q_i,
                "logit_q": logit_q_val,
                "kappa_nearest": float(g["kappa_nearest"]),
                "d_eff": float(g["d_eff"]),
                "K_eff_obs": None,  # NOT measured (surgery too slow)
                "K_eff_in": float(g["K_eff_in"]),
                "f_sub": float(g["f_sub"]),
            }
            all_records.append(rec)
            q_str = f"{q_i:.4f}" if q_i is not None else "N/A"
            logit_str = f"{logit_q_val:.4f}" if logit_q_val is not None else "N/A"
            log(f"  cls {target_cls}: kappa={g['kappa_nearest']:.4f}, d_eff={g['d_eff']:.2f}, "
                f"K_eff_in={g['K_eff_in']:.2f}, q_i={q_str}, logit={logit_str}")

        # Per-seed analysis
        seed_recs = [r for r in all_records if r["seed"] == seed]
        seed_analysis = analyze_models(seed_recs)
        log(f"\n  === SEED {seed} MODEL COMPARISON ===")
        for model_name in ["M0_kappa_sqrt_deff", "M4_kappa_only", "deff_only", "M5_kappa_sqrt_deff_over_Kin"]:
            if model_name in seed_analysis:
                a = seed_analysis[model_name]
                log(f"    {model_name:45s}: R2={a['R2']:+.4f}, r={a['r_pearson']:+.4f}, A={a['A_fit']:.3f}")
        if "joint_kappa_deff" in seed_analysis:
            j = seed_analysis["joint_kappa_deff"]
            log(f"    {'joint_additive (kappa+d_eff)':45s}: R2={j['R2']:+.4f}, a_kappa={j['a_kappa']:.3f}, a_deff={j['a_deff']:.4f}")
        if "partial_r_deff_given_kappa" in seed_analysis:
            p = seed_analysis["partial_r_deff_given_kappa"]
            log(f"    partial r(d_eff|kappa) = {p['r']:+.4f} p={p['p']:.4f} ({p['sign']})")

    # === FINAL ANALYSIS ===
    log(f"\n{'='*60}")
    log("FINAL ANALYSIS (all seeds)")
    log(f"{'='*60}")

    final_analysis = analyze_models(all_records)

    log("\nMODEL COMPARISON:")
    for model_name in ["M0_kappa_sqrt_deff", "M4_kappa_only", "deff_only", "M5_kappa_sqrt_deff_over_Kin", "kappa_sq"]:
        if model_name in final_analysis:
            a = final_analysis[model_name]
            log(f"  {model_name:45s}: R2={a['R2']:+.4f}, r={a['r_pearson']:+.4f}, A={a['A_fit']:.3f}, n={a['n']}")
    if "joint_kappa_deff" in final_analysis:
        j = final_analysis["joint_kappa_deff"]
        log(f"  {'joint_additive (kappa+d_eff)':45s}: R2={j['R2']:+.4f}, a_kappa={j['a_kappa']:.3f}, a_deff={j['a_deff']:.4f}")
    if "partial_r_deff_given_kappa" in final_analysis:
        p = final_analysis["partial_r_deff_given_kappa"]
        log(f"  partial r(d_eff|kappa) = {p['r']:+.4f} p={p['p']:.4f} ({p['sign']})")

    log("\nSPEARMAN CORRELATIONS with logit(q_i):")
    for key in ["kappa", "d_eff", "kappa_sqrt_deff"]:
        skey = f"spearman_{key}"
        if skey in final_analysis:
            a = final_analysis[skey]
            log(f"  {key:35s}: rho={a['rho']:+.4f}, p={a['p']:.4e}")

    # Main pre-registered check: kappa alone vs M0
    log("\nKEY FINDINGS:")
    m0_r2 = final_analysis.get("M0_kappa_sqrt_deff", {}).get("R2", np.nan)
    m4_r2 = final_analysis.get("M4_kappa_only", {}).get("R2", np.nan)
    joint_r2 = final_analysis.get("joint_kappa_deff", {}).get("R2", np.nan)
    log(f"  kappa alone R2 = {m4_r2:+.4f}")
    log(f"  M0 (kappa*sqrt(d_eff)) R2 = {m0_r2:+.4f}  ({'BETTER' if m4_r2 > m0_r2 else 'WORSE'} than kappa alone)")
    log(f"  Joint (kappa+d_eff) R2 = {joint_r2:+.4f}")

    # d_eff sign check
    log(f"\nD_EFF SIGN CHECK:")
    rho_deff = final_analysis.get("spearman_d_eff", {}).get("rho", np.nan)
    rho_kappa = final_analysis.get("spearman_kappa", {}).get("rho", np.nan)
    log(f"  Spearman rho(d_eff, logit_q)    = {rho_deff:+.4f}  (near zero within-class)")
    log(f"  Spearman rho(kappa, logit_q)    = {rho_kappa:+.4f}  (expected: positive, ~0.9)")

    out = {
        "experiment": "per_class_formula_test",
        "K": K, "N_EPOCHS": N_EPOCHS, "seeds": SEEDS, "R_SURGERY": R_SURGERY,
        "records": all_records,
        "analysis": final_analysis,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
