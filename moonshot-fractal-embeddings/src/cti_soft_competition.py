"""
Soft Competition Law Test
Tests the hypothesis that per-class K_eff_obs is governed by the full
competitor set {kappa_ij}, not just kappa_nearest.

Key tests:
1. K_eff_kappa: effective rank of {kappa_ij} distribution
   K_eff_kappa(i) = (sum_j kappa_ij)^2 / sum_j kappa_ij^2
2. Phi({kappa_ij}) = soft-competition metric (log-sum-exp):
   Phi(i, tau) = -tau * log(sum_j exp(-kappa_ij / tau))
3. Does replacing kappa_nearest with Phi in CTI law improve fit?
   logit(q) = A * Phi({kappa_ij}) * sqrt(d_eff) + C

Also tests: whether K_eff_obs ~ 1/kappa_nearest (observed inverse relationship)

Design: Train ResNet-18 on CIFAR-100 coarse (K=20), extract embeddings at
kappa_eff ~ 1.5-2.5 checkpoint, compute full K x K kappa matrix, run
per-class K_eff_obs via top-m surgery.

PRE-REGISTERED: Spearman rho(K_eff_obs, K_eff_kappa) > 0.5 (within-seed)
               AND Phi with optimized tau beats kappa_nearest in logit(q) fit R2
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

K = 20
D = 512
N_EPOCHS = 60
N_SEEDS = 3
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_VALUE = 5.0   # Surgery ratio for K_eff_obs measurement
M_VALUES = list(range(1, K))  # m = 1 to K-1
TAU_VALUES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]  # soft-competition temperatures
LOG_FILE = "results/cti_soft_competition_log.txt"
RESULTS_FILE = "results/cti_soft_competition.json"

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
    std = [0.2675, 0.2565, 0.2761]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    raw_tr = torchvision.datasets.CIFAR100(root="data", train=True, download=True, transform=train_tf)
    raw_te = torchvision.datasets.CIFAR100(root="data", train=False, download=True, transform=test_tf)
    tr_loader = torch.utils.data.DataLoader(CoarseDataset(raw_tr), batch_size=256, shuffle=True, num_workers=0, pin_memory=False, generator=g)
    te_loader = torch.utils.data.DataLoader(CoarseDataset(raw_te), batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    return tr_loader, te_loader

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

def compute_q(X_tr, y_tr, X_te, y_te):
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    return (acc - 1.0/K) / (1.0 - 1.0/K)

def compute_full_geometry(X, y):
    """Compute centroids, Sigma_W, and FULL K x K kappa matrix."""
    d = X.shape[1]
    centroids = np.array([X[y == k].mean(0) for k in range(K)])
    X_c = np.zeros_like(X)
    for k in range(K):
        mask = y == k
        X_c[mask] = X[mask] - centroids[k]
    Sigma_W = (X_c.T @ X_c) / len(X)
    tr_W = float(np.trace(Sigma_W))
    sigma_W = float(np.sqrt(tr_W / d))

    # Full kappa matrix (K x K)
    kappa_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                kappa_matrix[i, j] = np.inf
            else:
                dist = float(np.linalg.norm(centroids[i] - centroids[j]))
                kappa_matrix[i, j] = dist / (sigma_W * np.sqrt(d))

    # Per-class geometry
    class_geo = []
    for i in range(K):
        kappa_ij = kappa_matrix[i, :]
        finite_kappas = [k for k in kappa_ij if np.isfinite(k)]

        # Nearest competitor kappa
        kappa_nearest = min(finite_kappas)

        # Effective rank of kappa distribution (treats kappas like "eigenvalues")
        # K_eff_kappa = (sum_j kappa_ij)^2 / sum_j kappa_ij^2
        sum_k = sum(finite_kappas)
        sum_k2 = sum(kk**2 for kk in finite_kappas)
        K_eff_kappa = (sum_k**2) / (sum_k2 + 1e-12) if sum_k2 > 0 else 1.0

        # Soft competition Phi(tau) = -tau * log(sum_j exp(-kappa_ij/tau))
        phi_tau = {}
        for tau in TAU_VALUES:
            if tau == np.inf:
                # tau -> inf: all competitors contribute equally
                phi_tau[float(tau)] = float(np.mean(finite_kappas))
            else:
                kappas_arr = np.array(finite_kappas)
                # For numerical stability: Phi = min + tau * log(K-1) - tau * log(sum exp(-(kappa_ij - min)/tau))
                kmin = min(finite_kappas)
                log_sum = np.log(np.sum(np.exp(-(kappas_arr - kmin) / tau)))
                phi = kmin + tau * (np.log(K-1) - log_sum)
                phi_tau[float(tau)] = float(phi)
                # Note: when tau->0, phi -> kappa_nearest
                # when tau->inf, phi -> kappa_mean

        # d_eff formula
        dir_hat = (centroids[i] - centroids[np.argmin(kappa_ij)])
        norm = np.linalg.norm(dir_hat)
        if norm > 1e-10:
            dir_hat = dir_hat / norm
        sigma_cdir_sq = float(dir_hat @ Sigma_W @ dir_hat)
        d_eff = tr_W / sigma_cdir_sq if sigma_cdir_sq > 0 else d

        class_geo.append({
            "class": i,
            "kappa_nearest": float(kappa_nearest),
            "kappa_ij": [float(k) for k in kappa_ij],
            "K_eff_kappa": float(K_eff_kappa),
            "phi_tau": phi_tau,
            "d_eff": float(d_eff),
        })

    return {"centroids": centroids, "Sigma_W": Sigma_W, "tr_W": tr_W, "sigma_W": sigma_W,
            "class_geo": class_geo}

def apply_surgery_top_m(X_tr, y_tr, X_te, y_te, centroids, Sigma_W, target_cls, r, m):
    """Apply top-m surgery to target class."""
    d = X_tr.shape[1]
    tr_W = float(np.trace(Sigma_W))
    sigma_W = float(np.sqrt(tr_W / d))

    # Sort competitors by distance to target
    dists = [(j, float(np.linalg.norm(centroids[target_cls] - centroids[j])))
             for j in range(K) if j != target_cls]
    dists.sort(key=lambda x: x[1])  # nearest first

    # Top-m centroid directions
    dirs = [centroids[target_cls] - centroids[j] for j, _ in dists[:m]]
    dirs = [d / (np.linalg.norm(d) + 1e-10) for d in dirs]

    # Build orthonormal basis for top-m subspace
    if m == 1:
        Q = np.array(dirs[0]).reshape(-1, 1)
    else:
        M = np.array(dirs).T  # d x m
        Q, _ = np.linalg.qr(M, mode="reduced")  # d x m

    # Sigma_W decomposed into subspace and complementary
    tr_sub = float(np.trace(Q.T @ Sigma_W @ Q))
    scale_sub = 1.0 / np.sqrt(r + 1e-12)
    tr_sub_new = tr_sub * (1.0 / r)
    tr_orth = tr_W - tr_sub
    tr_orth_new = tr_W - tr_sub_new
    if tr_orth > 1e-10:
        scale_orth = float(np.sqrt(tr_orth_new / tr_orth))
    else:
        scale_orth = 1.0

    def transform(X):
        X_c = X - centroids[target_cls]
        proj_sub = (X_c @ Q) @ Q.T    # component in subspace
        proj_orth = X_c - proj_sub    # orthogonal component
        X_c_new = scale_sub * proj_sub + scale_orth * proj_orth
        return centroids[target_cls] + X_c_new

    X_tr_new = np.copy(X_tr)
    X_te_new = np.copy(X_te)
    mask_tr = y_tr == target_cls
    mask_te = y_te == target_cls
    X_tr_new[mask_tr] = transform(X_tr[mask_tr])
    X_te_new[mask_te] = transform(X_te[mask_te])
    return X_tr_new, X_te_new

def compute_K_eff_obs(X_tr, y_tr, X_te, y_te, centroids, Sigma_W, target_cls, r=R_VALUE):
    """Measure K_eff_obs for one target class via top-m surgery sweep."""
    q_base = compute_q(X_tr, y_tr, X_te, y_te)
    logit_base = float(logit_fn(np.clip(q_base, 1e-6, 1-1e-6)))

    deltas = []
    for m in M_VALUES:
        X_tr_new, X_te_new = apply_surgery_top_m(X_tr, y_tr, X_te, y_te, centroids, Sigma_W, target_cls, r, m)
        q_new = compute_q(X_tr_new, y_tr, X_te_new, y_te)
        logit_new = float(logit_fn(np.clip(q_new, 1e-6, 1-1e-6)))
        deltas.append(logit_new - logit_base)

    if not deltas or deltas[0] <= 0:
        return None, None

    delta_1 = deltas[0]
    delta_max = max(deltas)
    K_eff_obs = delta_max / (delta_1 + 1e-10)
    return float(K_eff_obs), float(delta_max)

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    log("=" * 70)
    log("Soft Competition Law Test")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    log(f"K={K}, N_EPOCHS={N_EPOCHS}, SEEDS={SEEDS}")
    log(f"TAU_VALUES={TAU_VALUES}")
    log(f"PRE-REGISTERED: Spearman rho(K_eff_obs, K_eff_kappa) > 0.5")
    log("=" * 70)

    all_records = []

    for seed in SEEDS:
        log(f"\n{'='*60}")
        log(f"SEED {seed}")
        log(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = ResNet18Embed().to(DEVICE)
        tr_loader, te_loader = make_loaders(seed)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        best_X_tr = best_X_te = best_y_tr = best_y_te = None
        best_kappa_eff = None

        for epoch in range(1, N_EPOCHS + 1):
            loss = train_epoch(model, tr_loader, optimizer, criterion)
            scheduler.step()

            if epoch % 5 == 0 or epoch <= 15:
                X_tr_ep, y_tr_ep = extract(model, tr_loader)
                X_te_ep, y_te_ep = extract(model, te_loader)
                q_ep = compute_q(X_tr_ep, y_tr_ep, X_te_ep, y_te_ep)
                d = X_tr_ep.shape[1]
                centroids = np.array([X_tr_ep[y_tr_ep == k].mean(0) for k in range(K)])
                X_c = np.zeros_like(X_tr_ep)
                for kk in range(K):
                    mask = y_tr_ep == kk
                    X_c[mask] = X_tr_ep[mask] - centroids[kk]
                Sigma_W_ep = (X_c.T @ X_c) / len(X_tr_ep)
                tr_W_ep = float(np.trace(Sigma_W_ep))
                sigma_W_ep = float(np.sqrt(tr_W_ep / d))

                # Global kappa_nearest
                min_kappa = np.inf
                for i in range(K):
                    for j in range(i+1, K):
                        dist = float(np.linalg.norm(centroids[i] - centroids[j]))
                        kappa_ij = dist / (sigma_W_ep * np.sqrt(d))
                        if kappa_ij < min_kappa:
                            min_kappa = kappa_ij

                d_eff_ep = 1.0  # simplified for checkpoint selection
                kappa_eff_ep = min_kappa * np.sqrt(d_eff_ep)

                log(f"  [ep={epoch:3d}] loss={loss:.4f}, kappa={min_kappa:.4f}, q={q_ep:.4f}")

                # Select checkpoint at kappa ~ 1.5-2.5 (interesting regime)
                if 1.0 <= min_kappa <= 3.0 and (best_kappa_eff is None or abs(min_kappa - 2.0) < abs(best_kappa_eff - 2.0)):
                    best_kappa_eff = min_kappa
                    best_X_tr = X_tr_ep.copy()
                    best_y_tr = y_tr_ep.copy()
                    best_X_te = X_te_ep.copy()
                    best_y_te = y_te_ep.copy()

        if best_X_tr is None:
            log(f"  No good checkpoint found for seed {seed}, using last")
            best_X_tr, best_y_tr = extract(model, tr_loader)
            best_X_te, best_y_te = extract(model, te_loader)

        log(f"\n  Computing full geometry at selected checkpoint...")
        geo = compute_full_geometry(best_X_tr, best_y_tr)
        log(f"  sigma_W={geo['sigma_W']:.4f}, tr_W={geo['tr_W']:.2f}")

        # Print full kappa matrix (row = target, sorted)
        log(f"\n  Per-class geometry:")
        log(f"  {'cls':>4} {'kappa_near':>11} {'K_eff_kappa':>12} {'d_eff':>8} {'Phi_tau0.5':>11}")
        for cg in geo["class_geo"]:
            log(f"  {cg['class']:>4d} {cg['kappa_nearest']:>11.4f} {cg['K_eff_kappa']:>12.4f} "
                f"{cg['d_eff']:>8.2f} {cg['phi_tau'].get(0.5, 0.0):>11.4f}")

        # Measure K_eff_obs for ALL classes
        log(f"\n  Running K_eff_obs for all {K} classes (r={R_VALUE})...")
        q_base_global = compute_q(best_X_tr, best_y_tr, best_X_te, best_y_te)
        log(f"  Global q_base = {q_base_global:.4f}")

        for target_cls in range(K):
            cg = geo["class_geo"][target_cls]

            K_eff_obs, delta_max = compute_K_eff_obs(
                best_X_tr, best_y_tr, best_X_te, best_y_te,
                geo["centroids"], geo["Sigma_W"], target_cls, R_VALUE
            )

            if K_eff_obs is None:
                log(f"  class {target_cls}: INFEASIBLE")
                continue

            c_obs = K_eff_obs / cg["d_eff"] if cg["d_eff"] > 0 else np.nan
            log(f"  class {target_cls}: K_eff_obs={K_eff_obs:.3f}, K_eff_kappa={cg['K_eff_kappa']:.3f}, "
                f"kappa_near={cg['kappa_nearest']:.4f}, c_obs={c_obs:.4f}")

            rec = {
                "seed": seed,
                "class": target_cls,
                "K_eff_obs": float(K_eff_obs),
                "K_eff_kappa": float(cg["K_eff_kappa"]),
                "kappa_nearest": float(cg["kappa_nearest"]),
                "d_eff": float(cg["d_eff"]),
                "c_obs": float(c_obs) if np.isfinite(c_obs) else None,
                "delta_max": float(delta_max),
            }
            for tau_key, phi_val in cg["phi_tau"].items():
                rec[f"phi_{tau_key}"] = float(phi_val)

            all_records.append(rec)

    # === FINAL ANALYSIS ===
    log(f"\n{'='*60}")
    log("FINAL ANALYSIS")
    log(f"{'='*60}")

    if len(all_records) >= 5:
        K_eff_obs_arr = np.array([r["K_eff_obs"] for r in all_records])
        K_eff_kappa_arr = np.array([r["K_eff_kappa"] for r in all_records])
        kappa_near_arr = np.array([r["kappa_nearest"] for r in all_records])

        # Spearman rho for K_eff_kappa vs K_eff_obs (pre-registered)
        spearman_keff, p_spearman_keff = stats.spearmanr(K_eff_obs_arr, K_eff_kappa_arr)
        # Anti-correlation with kappa_nearest (observed)
        spearman_kappa, p_spearman_kappa = stats.spearmanr(K_eff_obs_arr, kappa_near_arr)

        log(f"\nPRE-REGISTERED: Spearman rho(K_eff_obs, K_eff_kappa):")
        log(f"  rho = {spearman_keff:.4f}  {'PASS' if spearman_keff > 0.5 else 'FAIL'}  (threshold 0.5)")
        log(f"\nOBSERVED: Spearman rho(K_eff_obs, kappa_nearest):")
        log(f"  rho = {spearman_kappa:.4f}  (expect ~-0.5)")

        # Phi at different taus
        log(f"\nPhi comparison (Spearman with K_eff_obs):")
        tau_results = {}
        for tau in TAU_VALUES:
            tau_key = f"phi_{float(tau)}"
            phi_vals = [r.get(tau_key) for r in all_records]
            if any(v is None for v in phi_vals):
                continue
            phi_arr = np.array(phi_vals)
            if np.any(np.isnan(phi_arr)):
                continue
            rho_phi, _ = stats.spearmanr(K_eff_obs_arr, phi_arr)
            log(f"  tau={tau:.2f}: Spearman rho(K_eff_obs, Phi) = {rho_phi:.4f}")
            tau_results[float(tau)] = float(rho_phi)

        best_tau = max(tau_results, key=lambda t: abs(tau_results[t])) if tau_results else None
        if best_tau is not None:
            log(f"\n  Best tau = {best_tau:.2f} (rho = {tau_results[best_tau]:.4f})")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"records": all_records, "status": "complete"}, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    log(f"\nSaved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
