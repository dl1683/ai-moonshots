"""
K_eff Eigenspectrum Surgery Experiment
Tests Codex 2-layer model: logit(q_i) = C + A * kappa_i * g(K_eff_i)
  where K_eff_i = rank_eff(V_i) = tr(V_i)^2 / tr(V_i^2), g(K) ~ sqrt(K)

Surgery: manipulate eigenspectrum of V_i = U_i^T Sigma_W U_i
  - FLATTEN: redistribute eigenvalues to uniform (increases K_eff)
  - SPIKE: concentrate all variance in first eigenvector (decreases K_eff to ~1)
  - ROTATE: rotate eigenspectrum (no change in K_eff, falsification control)

Constraints:
  - tr(Sigma_W) preserved (total variance unchanged)
  - kappa_nearest preserved (centroid distances unchanged)
  - U_i orientation preserved (centroid subspace directions unchanged)

PRE-REGISTERED HYPOTHESIS: logit(q) = A * kappa_i * sqrt(K_eff_i) + C
  where A is architecture-universal constant

Pass criterion: R2(delta_logit, A * kappa_i * (sqrt(K_eff_new) - sqrt(K_eff_base))) > 0.5
               Pearson r(delta_logit_obs, delta_logit_pred) > 0.8
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import sys
import os
from scipy.special import logit, expit

# ---- CONFIG ----
K = 20          # CIFAR-100 coarse classes
D = 512         # embedding dim
N_EPOCHS = 60
N_SEEDS = 3
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = "results/cti_K_eff_surgery_log.txt"
RESULTS_FILE = "results/cti_K_eff_surgery.json"

# K_eff surgery levels: fraction of target K_eff_max achievable (K-1=19)
FLATTEN_LEVELS = [0.25, 0.5, 0.75, 1.0]   # fraction of max uniform (K-1)
SPIKE_LEVELS = [0.01, 0.05, 0.1, 0.25]    # fraction of max spike (1)

# Pre-registered: A = alpha / sqrt(2) where alpha=1.536 (LOAO result)
ALPHA_LOAO = 1.536
A_PREREGISTERED = ALPHA_LOAO / np.sqrt(2.0)  # ~1.086

def log(msg, file=LOG_FILE):
    print(msg, flush=True)
    with open(file, "a", encoding="ascii") as f:
        f.write(msg + "\n")

# ---- CIFAR-100 coarse setup (same as other scripts) ----
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
    std = [0.2675, 0.2565, 0.2761]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    raw_tr = torchvision.datasets.CIFAR100(root="data", train=True, download=True, transform=train_tf)
    raw_te = torchvision.datasets.CIFAR100(root="data", train=False, download=True, transform=test_tf)
    tr_loader = torch.utils.data.DataLoader(CoarseDataset(raw_tr), batch_size=256, shuffle=True, num_workers=0, pin_memory=False, generator=g)
    te_loader = torch.utils.data.DataLoader(CoarseDataset(raw_te), batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    return tr_loader, te_loader

# ---- ResNet-18 ----
class ResNet18Embed(nn.Module):
    def __init__(self, num_classes=K):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(512, num_classes)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.fc = net.fc
    def forward(self, x):
        z = self.backbone(x).flatten(1)
        return z, self.fc(z)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
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

def extract_embeddings(model, loader):
    model.eval()
    all_emb = []
    all_lbl = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            emb, _ = model(imgs)
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(labels.numpy())
    return np.concatenate(all_emb), np.concatenate(all_lbl)

def compute_q_knn(X_tr, y_tr, X_te, y_te):
    """Normalized 1-NN accuracy."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    q = (acc - 1.0/K) / (1.0 - 1.0/K)
    return float(q)

def compute_geometry(X, y, K_classes=K):
    """Compute centroids, Sigma_W, kappa_nearest, d_eff_formula, K_eff_formula for ALL classes."""
    d = X.shape[1]
    centroids = np.array([X[y == k].mean(0) for k in range(K_classes)])

    # Within-class covariance (pooled)
    X_c = np.zeros_like(X)
    for k in range(K_classes):
        mask = y == k
        X_c[mask] = X[mask] - centroids[k]
    Sigma_W = (X_c.T @ X_c) / len(X)  # d x d
    tr_W = float(np.trace(Sigma_W))
    sigma_W_global = float(np.sqrt(tr_W / d))

    # Find globally nearest centroid pair
    min_kappa = np.inf
    global_target = 0
    for i in range(K_classes):
        for j in range(K_classes):
            if i == j: continue
            diff = centroids[i] - centroids[j]
            dist = float(np.linalg.norm(diff))
            kappa_ij = dist / (sigma_W_global * np.sqrt(d))
            if kappa_ij < min_kappa:
                min_kappa = kappa_ij
                global_target = i

    # Compute per-class V_i and K_eff_pred
    class_geometries = []
    for target in range(K_classes):
        # Centroid directions from target to all competitors
        dirs = []
        for j in range(K_classes):
            if j == target: continue
            diff = centroids[target] - centroids[j]
            norm = float(np.linalg.norm(diff))
            if norm < 1e-10:
                dirs.append(np.zeros(d))
            else:
                dirs.append(diff / norm)
        dirs = np.array(dirs)  # (K-1, d)

        # QR decomposition for orthonormal basis of centroid subspace
        d_qr = max(d, K)
        Q, R = np.linalg.qr(dirs.T, mode="reduced")  # Q: (d, K-1)
        U_i = Q  # (d, K-1), columns are orthonormal

        # Projected covariance in centroid subspace
        V_i = U_i.T @ Sigma_W @ U_i  # (K-1, K-1)

        tr_Vi = float(np.trace(V_i))

        # Eigendecomposition of V_i
        eigvals = np.linalg.eigvalsh(V_i)
        eigvals = np.maximum(eigvals, 0)  # clip numerical negatives
        eigvals_sorted = np.sort(eigvals)[::-1]  # descending

        tr_Vi2 = float(np.sum(eigvals**2))
        K_eff_pred = (tr_Vi**2) / (tr_Vi2 + 1e-12)
        f_sub = tr_Vi / tr_W

        # kappa to nearest competitor
        dists = [np.linalg.norm(centroids[target] - centroids[j]) for j in range(K_classes) if j != target]
        min_dist = min(dists)
        kappa_i = min_dist / (sigma_W_global * np.sqrt(d))

        class_geometries.append({
            "target": target,
            "K_eff_pred": K_eff_pred,
            "f_sub": f_sub,
            "tr_Vi": tr_Vi,
            "eigvals": eigvals_sorted.tolist(),
            "V_i": V_i,
            "U_i": U_i,
            "kappa_i": kappa_i,
            "d_eff_formula": tr_W / (sigma_W_global * np.sqrt(d))**2 if sigma_W_global > 0 else 0,
        })

    return {
        "centroids": centroids,
        "Sigma_W": Sigma_W,
        "tr_W": tr_W,
        "sigma_W_global": sigma_W_global,
        "global_target": global_target,
        "kappa_nearest": min_kappa,
        "class_geometries": class_geometries,
    }

def apply_K_eff_surgery(X, y, geo, target_class, surgery_type="flatten", level=0.5, K_classes=K):
    """
    Manipulate eigenspectrum of V_i for target_class.
    - flatten: move eigenvalues toward uniform
    - spike: concentrate all variance in first eigenvector
    - rotate: rotate eigenspectrum (control, K_eff unchanged)

    Operates WITHIN the centroid subspace U_i.
    Preserves: tr(Sigma_W), kappa_nearest (centroid distances).
    """
    cls_geo = geo["class_geometries"][target_class]
    V_i = cls_geo["V_i"]
    U_i = cls_geo["U_i"]  # (d, K-1)
    eigvals = np.array(cls_geo["eigvals"])
    tr_Vi = cls_geo["tr_Vi"]
    K_minus1 = K_classes - 1

    # Current eigendecomposition of V_i
    eigvals_sym, eigvecs = np.linalg.eigh(V_i)  # ascending
    eigvals_sym = np.maximum(eigvals_sym, 0)
    eigvals_desc = eigvals_sym[::-1]  # descending
    eigvecs_desc = eigvecs[:, ::-1]  # corresponding

    K_eff_base = (tr_Vi**2) / (np.sum(eigvals_sym**2) + 1e-12)

    if surgery_type == "flatten":
        # Redistribute eigenvalues toward uniform
        # target: (1-level)*current + level*uniform
        uniform_eigs = np.ones(K_minus1) * tr_Vi / K_minus1
        new_eigvals = (1 - level) * eigvals_desc + level * uniform_eigs
    elif surgery_type == "spike":
        # Concentrate variance in first eigenvector
        # target: (1-level)*current + level*spike
        spike_eigs = np.zeros(K_minus1)
        spike_eigs[0] = tr_Vi  # all in first
        new_eigvals = (1 - level) * eigvals_desc + level * spike_eigs
    elif surgery_type == "rotate":
        # Rotate eigenvectors by random orthogonal matrix (K_eff unchanged)
        np.random.seed(42 + target_class)
        rot = np.linalg.qr(np.random.randn(K_minus1, K_minus1))[0]
        # Apply rotation to eigenvectors, keep eigenvalues
        new_eigvecs = eigvecs_desc @ rot  # (K-1, K-1)
        new_eigvals = eigvals_desc  # unchanged
        # Reconstruct V_i_new with rotated eigenvectors
        V_i_new = new_eigvecs @ np.diag(new_eigvals) @ new_eigvecs.T
        tr_Vi_new = float(np.trace(V_i_new))
        K_eff_new = (tr_Vi_new**2) / (np.sum(new_eigvals**2) + 1e-12)
        # Apply surgery: reconstruct Sigma_W_new
        # Sigma_W_new = Sigma_W_base + U_i @ (V_i_new - V_i) @ U_i.T
        delta_V = V_i_new - V_i
        Sigma_W_new = geo["Sigma_W"] + U_i @ delta_V @ U_i.T
        # Apply transform to X
        X_new = apply_covariance_transform(X, y, geo, Sigma_W_new, K_classes)
        return X_new, K_eff_new, K_eff_base

    # Sort descending
    sort_idx = np.argsort(new_eigvals)[::-1]
    new_eigvals_sorted = new_eigvals[sort_idx]
    new_eigvecs_sorted = eigvecs_desc[:, sort_idx]

    # Reconstruct V_i_new
    V_i_new = new_eigvecs_sorted @ np.diag(new_eigvals_sorted) @ new_eigvecs_sorted.T

    tr_Vi_new = float(np.trace(V_i_new))
    K_eff_new = (tr_Vi_new**2) / (np.sum(new_eigvals_sorted**2) + 1e-12)

    # Reconstruct Sigma_W_new: update the centroid subspace component
    delta_V = V_i_new - V_i
    Sigma_W_new = geo["Sigma_W"] + U_i @ delta_V @ U_i.T

    # Apply covariance transform to X
    X_new = apply_covariance_transform(X, y, geo, Sigma_W_new, K_classes)

    return X_new, K_eff_new, K_eff_base

def apply_covariance_transform(X, y, geo, Sigma_W_new, K_classes=K):
    """
    Transform X so that within-class covariance becomes Sigma_W_new.
    Method: Cholesky-based covariance shaping.
    X_transformed = centroids + (X - centroids) @ L_new.T @ L_old^{-T}
    where L_old @ L_old.T = Sigma_W_base, L_new @ L_new.T = Sigma_W_new
    """
    Sigma_W_base = geo["Sigma_W"]
    centroids = geo["centroids"]
    d = X.shape[1]

    # Add small diagonal for numerical stability
    eps = 1e-6 * np.eye(d)

    # Cholesky of base covariance
    try:
        L_old = np.linalg.cholesky(Sigma_W_base + eps)
    except np.linalg.LinAlgError:
        # Use eigendecomposition fallback
        eigvals, eigvecs = np.linalg.eigh(Sigma_W_base + eps)
        eigvals = np.maximum(eigvals, 1e-12)
        L_old = eigvecs @ np.diag(np.sqrt(eigvals))

    try:
        L_new = np.linalg.cholesky(Sigma_W_new + eps)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma_W_new + eps)
        eigvals = np.maximum(eigvals, 1e-12)
        L_new = eigvecs @ np.diag(np.sqrt(eigvals))

    # Transform matrix: T = L_new @ L_old^{-1}
    L_old_inv = np.linalg.solve(L_old.T, np.eye(d)).T  # L_old^{-1}
    T = L_new @ L_old_inv  # (d, d)

    # Apply to within-class residuals
    X_new = np.zeros_like(X)
    for k in range(K_classes):
        mask = y == k
        X_c = X[mask] - centroids[k]
        X_new[mask] = centroids[k] + X_c @ T.T

    return X_new

def run_K_eff_surgery_experiment(X_tr, y_tr, X_te, y_te, geo, seed):
    """Run K_eff surgery for all target classes."""
    log(f"\n--- K_eff SURGERY seed={seed} ---")

    q_base = compute_q_knn(X_tr, y_tr, X_te, y_te)
    logit_q_base = float(logit(np.clip(q_base, 1e-6, 1-1e-6)))

    log(f"  Baseline: q={q_base:.4f}, logit(q)={logit_q_base:.4f}")
    log(f"  kappa_nearest={geo['kappa_nearest']:.4f}, tr_W={geo['tr_W']:.2f}")

    results = []

    # Test on global target class (most informative)
    target = geo["global_target"]
    cls_geo = geo["class_geometries"][target]
    K_eff_base = cls_geo["K_eff_pred"]
    kappa_i = cls_geo["kappa_i"]

    log(f"\n  Target class: {target}, K_eff_base={K_eff_base:.2f}, kappa_i={kappa_i:.4f}")

    for surgery_type in ["flatten", "spike", "rotate"]:
        levels = FLATTEN_LEVELS if surgery_type == "flatten" else SPIKE_LEVELS if surgery_type == "spike" else [0.5]
        for level in levels:
            try:
                X_tr_new, K_eff_new, _ = apply_K_eff_surgery(
                    X_tr, y_tr, geo, target, surgery_type=surgery_type, level=level)
                X_te_new, K_eff_new_te, _ = apply_K_eff_surgery(
                    X_te, y_te, geo, target, surgery_type=surgery_type, level=level)

                q_new = compute_q_knn(X_tr_new, y_tr, X_te_new, y_te)
                logit_q_new = float(logit(np.clip(q_new, 1e-6, 1-1e-6)))
                delta_logit_obs = logit_q_new - logit_q_base

                # Predicted change from 2-layer model: logit(q) = A * kappa_i * sqrt(K_eff_i) + C
                # delta_logit_pred = A * kappa_i * (sqrt(K_eff_new) - sqrt(K_eff_base))
                delta_logit_pred = A_PREREGISTERED * kappa_i * (
                    np.sqrt(max(K_eff_new, 0.01)) - np.sqrt(max(K_eff_base, 0.01)))

                log(f"  [{surgery_type} lv={level:.2f}] K_eff: {K_eff_base:.2f}->{K_eff_new:.2f}, "
                    f"q: {q_base:.4f}->{q_new:.4f}, "
                    f"delta_logit: obs={delta_logit_obs:.4f} pred={delta_logit_pred:.4f}")

                results.append({
                    "surgery_type": surgery_type,
                    "level": float(level),
                    "target_class": int(target),
                    "K_eff_base": float(K_eff_base),
                    "K_eff_new": float(K_eff_new),
                    "q_base": float(q_base),
                    "q_new": float(q_new),
                    "logit_q_base": float(logit_q_base),
                    "logit_q_new": float(logit_q_new),
                    "delta_logit_obs": float(delta_logit_obs),
                    "delta_logit_pred": float(delta_logit_pred),
                    "kappa_i": float(kappa_i),
                })
            except Exception as e:
                log(f"  [{surgery_type} lv={level:.2f}] ERROR: {e}")

    return results

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    log("=" * 70)
    log("K_eff Eigenspectrum Surgery Experiment")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, SEEDS={SEEDS}")
    log(f"PRE-REGISTERED: logit(q) = A * kappa_i * sqrt(K_eff_i) + C")
    log(f"  A_preregistered = {A_PREREGISTERED:.4f} (alpha={ALPHA_LOAO}/sqrt(2))")
    log(f"  K_eff_i = rank_eff(V_i) = tr(V_i)^2 / tr(V_i^2)")
    log(f"PASS: Pearson r(delta_logit_obs, delta_logit_pred) > 0.8")
    log("=" * 70)

    all_results = []

    for seed in SEEDS:
        log(f"\n{'='*70}")
        log(f"SEED {seed}")
        log(f"{'='*70}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = ResNet18Embed(num_classes=K).to(DEVICE)
        tr_loader, te_loader = make_loaders(seed)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        for epoch in range(1, N_EPOCHS + 1):
            loss = train_one_epoch(model, tr_loader, optimizer, criterion)
            scheduler.step()
            if epoch % 10 == 0:
                log(f"  [ep={epoch:3d}] loss={loss:.4f}")

        log(f"  Training complete. Extracting embeddings...")
        X_tr, y_tr = extract_embeddings(model, tr_loader)
        X_te, y_te = extract_embeddings(model, te_loader)
        log(f"  X_tr={X_tr.shape}, X_te={X_te.shape}")

        log(f"  Computing full geometry...")
        geo = compute_geometry(X_tr, y_tr)
        log(f"  kappa_nearest={geo['kappa_nearest']:.4f}, tr_W={geo['tr_W']:.2f}")

        seed_results = run_K_eff_surgery_experiment(X_tr, y_tr, X_te, y_te, geo, seed)
        all_results.extend(seed_results)

        # Save after each seed
        with open(RESULTS_FILE, "w") as f:
            json.dump({"status": "running", "seeds_complete": seed + 1,
                       "records": all_results, "A_preregistered": A_PREREGISTERED}, f, indent=2)
        log(f"\n  [Seed {seed} complete, {len(seed_results)} results saved]")

    # Final analysis
    log("\n" + "="*70)
    log("FINAL ANALYSIS")
    log("="*70)

    from scipy import stats

    # Flatten and spike (exclude rotate as control)
    active_results = [r for r in all_results if r["surgery_type"] in ("flatten", "spike")]
    rotate_results = [r for r in all_results if r["surgery_type"] == "rotate"]

    if len(active_results) >= 4:
        obs = [r["delta_logit_obs"] for r in active_results]
        pred = [r["delta_logit_pred"] for r in active_results]
        pearson_r, p_pearson = stats.pearsonr(obs, pred)

        obs_arr = np.array(obs)
        pred_arr = np.array(pred)
        ss_res = np.sum((obs_arr - pred_arr)**2)
        ss_tot = np.sum((obs_arr - obs_arr.mean())**2)
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

        log(f"\nPRIMARY: Flatten+Spike results (n={len(active_results)})")
        log(f"  Pearson r(obs, pred): {pearson_r:.4f}  {'PASS' if pearson_r > 0.8 else 'FAIL'} (threshold 0.8)")
        log(f"  R2(obs, pred):        {r2:.4f}")
        log(f"  Mean obs:   {obs_arr.mean():.4f}")
        log(f"  Mean pred:  {pred_arr.mean():.4f}")

        # Control: rotation should give zero delta
        if rotate_results:
            rot_obs = [r["delta_logit_obs"] for r in rotate_results]
            log(f"\nCONTROL (rotate, expect ~0): mean={np.mean(rot_obs):.4f}, std={np.std(rot_obs):.4f}")

    # Calibration: slope of obs vs pred
    if len(active_results) >= 4:
        slope, intercept, r_val, p_val, se = stats.linregress(pred, obs)
        log(f"\n  Fitted slope (obs/pred): {slope:.4f} (ideal=1.0)")
        log(f"  Fitted intercept:        {intercept:.4f} (ideal=0.0)")
        log(f"  Pre-registered A: {A_PREREGISTERED:.4f}")
        log(f"  Implied A from slope: {A_PREREGISTERED * slope:.4f}")

    final_data = {
        "status": "complete",
        "seeds_complete": len(SEEDS),
        "records": all_results,
        "A_preregistered": float(A_PREREGISTERED),
        "n_active": len(active_results),
        "n_rotate": len(rotate_results),
    }
    if len(active_results) >= 4:
        final_data["pearson_r"] = float(pearson_r)
        final_data["r2"] = float(r2)
        final_data["slope_fitted"] = float(slope)
        final_data["PASS"] = bool(pearson_r > 0.8)

    with open(RESULTS_FILE, "w") as f:
        json.dump(final_data, f, indent=2)

    log(f"\nSaved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
