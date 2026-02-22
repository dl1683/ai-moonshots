"""
2x2 Causal Factorial: Decoupling kappa_nearest and d_eff_sig

CODEX RECOMMENDATION (Feb 22 2026): The strongest single experiment for Nobel
trajectory. Test iso-kappa_eff_sig invariance by independently manipulating:
  Factor A (kappa_nearest): L_margin loss -- pushes minimum centroid distance up
  Factor B (d_eff_sig):     L_ETF loss   -- restructures signal subspace geometry

PRE-REGISTERED:
  logit(q) = A_renorm(K) * kappa_eff_sig + C
  where kappa_eff_sig = sqrt(d_eff_sig) * kappa_nearest
  and A_renorm(K=20) = 1.0535 (Theorem 15, zero free parameters)

  If this law holds across ALL 4 arms with the SAME pre-registered slope,
  this proves the product form kappa_eff_sig = sqrt(d_eff_sig) * kappa is
  the CORRECT causal parameterization (not kappa or d_eff_sig alone).

ISO-KAPPA_EFF_SIG INVARIANCE TEST:
  If two conditions from different arms have similar kappa_eff_sig
  but different (kappa, d_eff_sig) breakdowns, they should have the
  same logit(q). This is the strongest causal test: d_eff_sig and
  kappa_nearest are SUBSTITUTABLE in the law via their product.

ARMS:
  ce:         Pure CE (baseline). kappa=baseline, d_eff_sig=baseline.
  nc_margin:  CE + L_margin ONLY. kappa increases, d_eff_sig approx unchanged.
  nc_etf:     CE + L_ETF ONLY. d_eff_sig changes (signal subspace restructured).
  nc_full:    CE + L_within + L_ETF + L_margin. Both kappa and d_eff_sig change.

d_eff_sig is measured using the CORRECTED formula (Theorem 16):
  - Signal subspace: exact K-1 principal directions of class centroid matrix
  - W_sig = sum_c (n_c/N) Sigma_c_sig  [pooled covariance in signal subspace]
  - d_eff_sig = tr(W_sig)^2 / tr(W_sig^2)  [includes ALL cross-class terms]
  - This avoids the K-fold inflation from omitting cross terms
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

# Hyperparameters
K = 20
N_EPOCHS = 60
WARMUP_EPOCHS = 25
RAMP_END_EPOCH = 50
LAMBDA_MAX = 0.15
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 5e-4
N_SEEDS = 2  # 2 seeds per arm for efficiency (4 arms x 2 seeds = 8 runs)
PROJ_DIM = 256
MARGIN = 1.0
EMA_MOMENTUM = 0.95
CHECKPOINT_EPOCHS = [25, 60]  # 2 checkpoints for efficiency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "results/cti_2x2_factorial.json"
A_RENORM_K20 = 1.0535  # Theorem 15 pre-registered constant


def get_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    ce_head = nn.Linear(512, K)
    proj_head = nn.Sequential(nn.Linear(512, PROJ_DIM), nn.BatchNorm1d(PROJ_DIM))
    model = nn.ModuleDict({'backbone': backbone, 'ce_head': ce_head, 'proj_head': proj_head})
    return model.to(DEVICE)


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


def extract_all_embeddings(model, dataset):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0)
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            embs.append(model['backbone'](imgs.to(DEVICE)).cpu().numpy())
            labels.append(lbs.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def compute_d_eff_gram(X, y):
    """Global participation ratio (includes ALL dimensions, including noise)."""
    classes = np.unique(y)
    N = len(X)
    trW = 0.0
    trW2 = 0.0
    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        mu_c = Xc.mean(0)
        Xc_centered = Xc - mu_c
        trSigma_k = float(np.sum(Xc_centered ** 2)) / n_c
        trW += n_c * trSigma_k / N
        G = (Xc_centered @ Xc_centered.T) / n_c
        trSigma_k2 = float(np.sum(G ** 2))
        trW2 += (n_c / N) ** 2 * trSigma_k2
    return float(trW ** 2 / (trW2 + 1e-12)), float(trW)


def compute_d_eff_formula(X, y):
    """
    CORRECT d_eff for the CTI law (discovered Feb 23 2026).
    d_eff_formula = tr(Sigma_W) / sigma_centroid_dir^2
    where sigma_centroid_dir = sqrt(Delta_min^T Sigma_W Delta_min / ||Delta_min||^2)
    This measures ANISOTROPY: how concentrated within-class variance is in centroid direction.
    Expected: ~1.46 for CIFAR CE arm (vs d_eff_sig ~ 15).
    """
    classes = np.unique(y)
    N = len(X)
    d = X.shape[1]
    centroids = np.stack([X[y == c].mean(0) for c in classes])

    # tr(Sigma_W)
    trW = 0.0
    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        Xc_c = Xc - centroids[c]
        trW += (n_c / N) * float(np.sum(Xc_c ** 2)) / n_c

    # Nearest centroid pair direction
    min_dist = float('inf')
    min_i, min_j = 0, 1
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < min_dist:
                min_dist, min_i, min_j = dist, i, j
    Delta = centroids[min_i] - centroids[min_j]
    Delta_hat = Delta / (np.linalg.norm(Delta) + 1e-10)

    # sigma_centroid_dir^2 = sum_c (n_c/N) * E[(Delta_hat^T x_c_centered)^2]
    sigma_centroid_sq = 0.0
    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        proj = (Xc - centroids[c]) @ Delta_hat
        sigma_centroid_sq += (n_c / N) * float(np.mean(proj ** 2))

    sigma_W_global = float(np.sqrt(trW / d))
    sigma_centroid_dir = float(np.sqrt(sigma_centroid_sq + 1e-10))
    d_eff_formula = float(trW / (sigma_centroid_sq + 1e-10))
    ratio = float(sigma_centroid_dir / (sigma_W_global + 1e-10))
    return d_eff_formula, sigma_centroid_dir, ratio


def compute_d_eff_sig(X, y):
    """
    CORRECTED signal-subspace participation ratio (Theorem 16).

    Uses exact pooled covariance W_sig = sum_c p_c Sigma_c_sig (n_sig x n_sig)
    and computes tr(W_sig^2) = ||W_sig||_F^2, which includes ALL cross-class terms.
    Signal subspace: exactly K-1 principal directions of class centroid matrix.
    """
    classes = np.unique(y)
    K_actual = len(classes)
    N = len(X)
    d = X.shape[1]

    # Signal subspace: K-1 principal directions of centered centroid matrix
    grand_mean = X.mean(0)
    centroids = np.stack([X[y == c].mean(0) for c in classes])
    centroids_centered = centroids - grand_mean

    U, S, Vt = np.linalg.svd(centroids_centered, full_matrices=False)
    n_sig = max(1, min(K_actual - 1, d, len(S)))
    P_B = Vt[:n_sig, :]  # (n_sig, d)

    # Accumulate per-class covariance in signal subspace (n_sig x n_sig)
    W_sig = np.zeros((n_sig, n_sig), dtype=np.float64)
    trW_sig = 0.0

    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        Xc_centered = (Xc - Xc.mean(0)).astype(np.float64)
        Xc_proj = Xc_centered @ P_B.T  # (n_c, n_sig)
        Sigma_c_sig = (Xc_proj.T @ Xc_proj) / n_c  # (n_sig, n_sig)
        W_sig += (n_c / N) * Sigma_c_sig
        trW_sig += (n_c / N) * float(np.trace(Sigma_c_sig))

    trW2_sig = float(np.sum(W_sig ** 2))  # tr(W_sig^2) with all cross-class terms
    d_eff_sig = float(trW_sig ** 2 / (trW2_sig + 1e-12))
    return d_eff_sig, float(trW_sig), int(n_sig)


def compute_kappa_nearest(X, y):
    classes = np.unique(y)
    d = X.shape[1]
    means, within_vars = {}, []
    for c in classes:
        Xc = X[y == c]
        means[c] = Xc.mean(0)
        within_vars.append(np.mean(np.sum((Xc - means[c]) ** 2, axis=1)))
    sigma_W = np.sqrt(np.mean(within_vars) / d)
    min_dist = min(
        np.linalg.norm(means[classes[i]] - means[classes[j]])
        for i in range(len(classes)) for j in range(i + 1, len(classes))
    )
    return float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))


def compute_q(X, y, random_state=42):
    sss = StratifiedShuffleSplit(1, test_size=0.3, random_state=random_state)
    tr, te = next(sss.split(X, y))
    knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
    knn.fit(X[tr], y[tr])
    acc = float(knn.score(X[te], y[te]))
    return (acc - 1.0 / K) / (1.0 - 1.0 / K)


def compute_aux_loss(arm, z, y, class_means):
    """
    Compute auxiliary loss for each arm:
      ce:        no auxiliary loss
      nc_margin: L_margin ONLY (pushes min centroid distance up)
      nc_etf:    L_ETF ONLY (restructures centroid geometry toward simplex)
      nc_full:   L_within + L_ETF + L_margin (full NC-loss)
    """
    if arm == 'ce' or arm is None:
        return None

    M = class_means  # (K, PROJ_DIM) normalized class means
    G = M @ M.t()
    dists = (torch.cdist(M.unsqueeze(0), M.unsqueeze(0)).squeeze(0)
             + torch.eye(K, device=z.device) * 1e6)

    if arm == 'nc_margin':
        # Factor A: change kappa_nearest only
        # L_margin increases minimum inter-centroid distance
        L_margin = F.softplus(MARGIN - dists.min())
        return L_margin

    elif arm == 'nc_etf':
        # Factor B: change d_eff_sig (signal subspace structure)
        # L_ETF pushes centroids toward equidistant simplex (ETF)
        G_etf = (torch.eye(K, device=z.device) * (1 + 1.0 / (K - 1))
                 - (1.0 / (K - 1)) * torch.ones(K, K, device=z.device))
        L_ETF = ((G - G_etf) ** 2).sum() / (K ** 2)
        return L_ETF

    elif arm == 'nc_full':
        # Full NC-loss: L_within + L_ETF + L_margin
        mu_yi = class_means[y]
        L_within = ((z - mu_yi) ** 2).mean()
        G_etf = (torch.eye(K, device=z.device) * (1 + 1.0 / (K - 1))
                 - (1.0 / (K - 1)) * torch.ones(K, K, device=z.device))
        L_ETF = ((G - G_etf) ** 2).sum() / (K ** 2)
        L_margin = F.softplus(MARGIN - dists.min())
        return L_within + 0.5 * L_ETF + 0.5 * L_margin

    return None


def update_ema_means(class_means, z, y):
    with torch.no_grad():
        for c in range(K):
            mask = (y == c)
            if mask.sum() > 0:
                class_means[c] = (EMA_MOMENTUM * class_means[c]
                                  + (1 - EMA_MOMENTUM) * z[mask].mean(0))
                class_means[c] = F.normalize(class_means[c].unsqueeze(0), dim=1).squeeze(0)
    return class_means


def get_lambda(epoch):
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    elif epoch <= RAMP_END_EPOCH:
        return LAMBDA_MAX * (epoch - WARMUP_EPOCHS) / (RAMP_END_EPOCH - WARMUP_EPOCHS)
    else:
        return LAMBDA_MAX


def train_one_arm(seed, arm, train_ds, test_ds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = get_model()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=0.9,
        weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    ce_loss_fn = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    class_means = F.normalize(torch.randn(K, PROJ_DIM, device=DEVICE), dim=1)
    checkpoints = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        lam = get_lambda(epoch) if arm != 'ce' else 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            h = model['backbone'](imgs)
            logits = model['ce_head'](h)
            loss_ce = ce_loss_fn(logits, labels)

            if lam > 0:
                z_raw = model['proj_head'](h)
                z = F.normalize(z_raw, dim=1)
                class_means = update_ema_means(class_means, z.detach(), labels)
                loss_aux = compute_aux_loss(arm, z, labels, class_means)
                if loss_aux is not None:
                    loss = loss_ce + lam * loss_aux
                else:
                    loss = loss_ce
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch in CHECKPOINT_EPOCHS:
            X_test, y_test = extract_all_embeddings(model, test_ds)
            q_val = compute_q(X_test, y_test)
            kappa_val = compute_kappa_nearest(X_test, y_test)
            logit_q = float(np.log(max(q_val, 0.001) / max(1 - q_val, 0.001)))

            print(f"  [seed={seed} arm={arm} epoch={epoch}] q={q_val:.4f} "
                  f"kappa={kappa_val:.4f} lam={lam:.3f} extracting train embs...",
                  end=' ', flush=True)

            X_train, y_train = extract_all_embeddings(model, train_ds)
            d_eff_gram, trW = compute_d_eff_gram(X_train, y_train)
            d_eff_sig, trW_sig, n_sig = compute_d_eff_sig(X_train, y_train)
            d_eff_formula, sigma_cd, sigma_ratio = compute_d_eff_formula(X_train, y_train)

            kappa_eff_gram = float(np.sqrt(d_eff_gram) * kappa_val)
            kappa_eff_sig = float(np.sqrt(d_eff_sig) * kappa_val)
            kappa_eff_formula = float(np.sqrt(d_eff_formula) * kappa_val)

            print(f"d_eff_gram={d_eff_gram:.1f} d_eff_sig={d_eff_sig:.3f} "
                  f"d_eff_formula={d_eff_formula:.3f} sigma_ratio={sigma_ratio:.1f}x "
                  f"kappa_eff_formula={kappa_eff_formula:.4f}", flush=True)

            checkpoints.append({
                'epoch': epoch,
                'q': float(q_val),
                'kappa': float(kappa_val),
                'logit_q': float(logit_q),
                'd_eff_gram': float(d_eff_gram),
                'd_eff_sig': float(d_eff_sig),
                'd_eff_formula': float(d_eff_formula),
                'sigma_centroid_dir': float(sigma_cd),
                'sigma_centroid_vs_global_ratio': float(sigma_ratio),
                'n_sig': int(n_sig),
                'kappa_eff_gram': float(kappa_eff_gram),
                'kappa_eff_formula': float(kappa_eff_formula),
                'kappa_eff_sig': float(kappa_eff_sig),
                'predicted_logit_sig': float(A_RENORM_K20 * kappa_eff_sig),
                'predicted_logit_formula': float(A_RENORM_K20 * kappa_eff_formula),
                'predicted_logit_gram': float(A_RENORM_K20 * kappa_eff_gram),
                'lambda': float(lam),
            })

    return {
        'seed': seed, 'arm': arm,
        'checkpoints': checkpoints,
        'final_q': checkpoints[-1]['q'] if checkpoints else None,
        'final_kappa': checkpoints[-1]['kappa'] if checkpoints else None,
        'final_d_eff_sig': checkpoints[-1]['d_eff_sig'] if checkpoints else None,
        'final_d_eff_formula': checkpoints[-1]['d_eff_formula'] if checkpoints else None,
    }


def analyze_results(all_results):
    """
    PRE-REGISTERED ANALYSIS:
    1. Does logit_q ~ A_renorm * kappa_eff_sig (FIXED slope) hold across ALL arms?
    2. Is kappa_eff_sig a BETTER predictor than kappa alone or d_eff_sig alone?
    3. Are Factor A (L_margin) and Factor B (L_ETF) truly decoupled?
    4. ISO-KAPPA_EFF_SIG TEST: matched pairs from different arms
    """
    all_snaps = []
    for arm, arm_res in all_results.items():
        for res in arm_res:
            for ck in res.get('checkpoints', []):
                all_snaps.append({'arm': arm, 'seed': res['seed'], **ck})

    if not all_snaps:
        return {'status': 'no_data'}

    logits = np.array([s['logit_q'] for s in all_snaps])
    ke_sig = np.array([s['kappa_eff_sig'] for s in all_snaps])
    ke_gram = np.array([s['kappa_eff_gram'] for s in all_snaps])
    ke_formula = np.array([s.get('kappa_eff_formula', s['kappa_eff_sig']) for s in all_snaps])
    kappa = np.array([s['kappa'] for s in all_snaps])
    d_sig = np.array([s['d_eff_sig'] for s in all_snaps])
    d_gram = np.array([s['d_eff_gram'] for s in all_snaps])
    d_formula = np.array([s.get('d_eff_formula', 1.0) for s in all_snaps])

    ss_tot = float(np.sum((logits - logits.mean()) ** 2))
    analysis = {'n_snaps': len(all_snaps)}

    def r2_fixed_slope(x, y_obs, A):
        """R2 with fixed slope A, free intercept."""
        C = float(np.mean(y_obs - A * x))
        resids = y_obs - (A * x + C)
        ss_res = float(np.sum(resids ** 2))
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0, C

    def r2_free(x, y_obs):
        if x.std() < 1e-6:
            return 0.0, None
        return float(np.corrcoef(x, y_obs)[0, 1] ** 2), float(np.polyfit(x, y_obs, 1)[0])

    # PRIMARY TEST: kappa_eff_sig with pre-registered slope
    r2_ke_sig_fixed, C_sig = r2_fixed_slope(ke_sig, logits, A_RENORM_K20)
    r2_ke_sig_free, slope_ke_sig = r2_free(ke_sig, logits)

    # NEW: d_eff_formula test (discovered Feb 23 2026)
    r2_ke_formula_fixed, C_formula = r2_fixed_slope(ke_formula, logits, A_RENORM_K20)
    r2_ke_formula_free, slope_ke_formula = r2_free(ke_formula, logits)

    # COMPARISON tests
    r2_ke_gram_fixed, C_gram = r2_fixed_slope(ke_gram, logits, A_RENORM_K20)
    r2_ke_gram_free, slope_ke_gram = r2_free(ke_gram, logits)
    r2_kappa_free, slope_kappa = r2_free(kappa, logits)
    r2_dsig_free, slope_dsig = r2_free(d_sig, logits)

    analysis['PRIMARY'] = {
        'r2_kappa_eff_sig_fixed_slope': r2_ke_sig_fixed,
        'r2_kappa_eff_sig_free_slope': r2_ke_sig_free,
        'empirical_slope': slope_ke_sig,
        'A_renorm_preregistered': A_RENORM_K20,
        'C_optimal': C_sig,
        'PASS': bool(r2_ke_sig_fixed > 0.8 and r2_ke_sig_free > 0.8),
        'slope_match': bool(slope_ke_sig is not None and abs(slope_ke_sig - A_RENORM_K20) / A_RENORM_K20 < 0.2),
    }

    # NEW: d_eff_formula comparison (key discovery from Feb 23 2026)
    analysis['D_EFF_FORMULA'] = {
        'r2_kappa_eff_formula_fixed_slope': r2_ke_formula_fixed,
        'r2_kappa_eff_formula_free_slope': r2_ke_formula_free,
        'empirical_slope': slope_ke_formula,
        'C_optimal': C_formula,
        'mean_d_eff_formula': float(d_formula.mean()),
        'PASS': bool(r2_ke_formula_fixed > 0.8),
        'formula_beats_sig': bool(r2_ke_formula_fixed > r2_ke_sig_fixed),
    }

    analysis['COMPARISONS'] = {
        'r2_kappa_eff_formula_fixed': r2_ke_formula_fixed,
        'r2_kappa_eff_sig_fixed': r2_ke_sig_fixed,
        'r2_kappa_eff_gram_fixed': r2_ke_gram_fixed,
        'r2_kappa_eff_gram_free': r2_ke_gram_free,
        'r2_kappa_only_free': r2_kappa_free,
        'r2_deff_sig_only_free': r2_dsig_free,
        'ranking': sorted([
            ('kappa_eff_formula', r2_ke_formula_fixed),
            ('kappa_eff_sig', r2_ke_sig_free),
            ('kappa_eff_gram', r2_ke_gram_free),
            ('kappa_only', r2_kappa_free),
            ('d_eff_sig', r2_dsig_free),
        ], key=lambda x: -(x[1] or 0)),
    }

    # FACTOR SEPARATION: per-arm d_eff changes vs CE
    analysis['FACTOR_SEPARATION'] = {}
    ce_snaps = [s for s in all_snaps if s['arm'] == 'ce']
    for arm in ['nc_margin', 'nc_etf', 'nc_full']:
        arm_snaps = [s for s in all_snaps if s['arm'] == arm]
        if ce_snaps and arm_snaps:
            delta_kappa = float(np.mean([s['kappa'] for s in arm_snaps])
                                - np.mean([s['kappa'] for s in ce_snaps]))
            delta_d_sig = float(np.mean([s['d_eff_sig'] for s in arm_snaps])
                                - np.mean([s['d_eff_sig'] for s in ce_snaps]))
            delta_d_formula = float(
                np.mean([s.get('d_eff_formula', 0) for s in arm_snaps])
                - np.mean([s.get('d_eff_formula', 0) for s in ce_snaps]))
            delta_ke_sig = float(np.mean([s['kappa_eff_sig'] for s in arm_snaps])
                                 - np.mean([s['kappa_eff_sig'] for s in ce_snaps]))
            delta_ke_formula = float(
                np.mean([s.get('kappa_eff_formula', 0) for s in arm_snaps])
                - np.mean([s.get('kappa_eff_formula', 0) for s in ce_snaps]))
            analysis['FACTOR_SEPARATION'][arm] = {
                'delta_kappa': delta_kappa,
                'delta_d_eff_sig': delta_d_sig,
                'delta_d_eff_formula': delta_d_formula,
                'delta_kappa_eff_sig': delta_ke_sig,
                'delta_kappa_eff_formula': delta_ke_formula,
                'kappa_dominated': bool(abs(delta_kappa) > 2 * abs(delta_d_sig)),
                'dsig_dominated': bool(abs(delta_d_sig) > 2 * abs(delta_kappa)),
                'dformula_dominated': bool(abs(delta_d_formula) > 0.1),
            }

    # ISO-KAPPA_EFF_SIG TEST: find matched pairs across different arms
    # Two snapshots are "matched" if kappa_eff_sig is within 5% of each other
    # but from different arms and different (kappa, d_eff_sig) breakdown
    iso_pairs = []
    for i, s1 in enumerate(all_snaps):
        for j, s2 in enumerate(all_snaps):
            if j <= i or s1['arm'] == s2['arm']:
                continue
            ke1, ke2 = s1['kappa_eff_sig'], s2['kappa_eff_sig']
            if ke1 < 0.01 or ke2 < 0.01:
                continue
            ke_match = abs(ke1 - ke2) / max(ke1, ke2) < 0.05  # within 5%
            kappa_diff = abs(s1['kappa'] - s2['kappa']) / max(s1['kappa'], s2['kappa'], 0.001)
            dsig_diff = abs(s1['d_eff_sig'] - s2['d_eff_sig']) / max(s1['d_eff_sig'], s2['d_eff_sig'], 0.001)
            if ke_match and (kappa_diff > 0.05 or dsig_diff > 0.05):
                logit_diff = abs(s1['logit_q'] - s2['logit_q'])
                iso_pairs.append({
                    'arm1': s1['arm'], 'arm2': s2['arm'],
                    'seed1': s1['seed'], 'seed2': s2['seed'],
                    'epoch1': s1['epoch'], 'epoch2': s2['epoch'],
                    'kappa1': s1['kappa'], 'kappa2': s2['kappa'],
                    'd_eff_sig1': s1['d_eff_sig'], 'd_eff_sig2': s2['d_eff_sig'],
                    'kappa_eff_sig1': ke1, 'kappa_eff_sig2': ke2,
                    'logit_q1': s1['logit_q'], 'logit_q2': s2['logit_q'],
                    'logit_diff': float(logit_diff),
                    'kappa_relative_diff': float(kappa_diff),
                    'dsig_relative_diff': float(dsig_diff),
                    'PASS': bool(logit_diff < 0.3),  # matched kappa_eff_sig -> matched logit_q
                })

    if iso_pairs:
        iso_pass_rate = sum(1 for p in iso_pairs if p['PASS']) / len(iso_pairs)
        analysis['ISO_KAPPA_EFF_SIG'] = {
            'n_pairs': len(iso_pairs),
            'n_pass': sum(1 for p in iso_pairs if p['PASS']),
            'pass_rate': float(iso_pass_rate),
            'mean_logit_diff_matched': float(np.mean([p['logit_diff'] for p in iso_pairs])),
            'PASS': bool(iso_pass_rate > 0.7),
            'pairs': iso_pairs[:10],  # top 10 pairs
        }
    else:
        analysis['ISO_KAPPA_EFF_SIG'] = {
            'n_pairs': 0, 'PASS': False,
            'note': 'No matched kappa_eff_sig pairs found across arms'}

    # Per-arm summary
    analysis['per_arm'] = {}
    for arm in ['ce', 'nc_margin', 'nc_etf', 'nc_full']:
        arm_snaps = [s for s in all_snaps if s['arm'] == arm]
        if arm_snaps:
            analysis['per_arm'][arm] = {
                'n': len(arm_snaps),
                'mean_q': float(np.mean([s['q'] for s in arm_snaps])),
                'mean_kappa': float(np.mean([s['kappa'] for s in arm_snaps])),
                'mean_d_eff_sig': float(np.mean([s['d_eff_sig'] for s in arm_snaps])),
                'mean_d_eff_gram': float(np.mean([s['d_eff_gram'] for s in arm_snaps])),
                'mean_kappa_eff_sig': float(np.mean([s['kappa_eff_sig'] for s in arm_snaps])),
            }

    return analysis


def main():
    print("2x2 Causal Factorial: kappa_nearest x d_eff_sig Decoupling")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Arms: CE, NC_margin (Factor A), NC_ETF (Factor B), NC_full")
    print(f"N_SEEDS={N_SEEDS}, N_EPOCHS={N_EPOCHS}, K={K}")
    print()
    print("PRE-REGISTERED:")
    print(f"  logit(q) = A_renorm * kappa_eff_sig + C  [A_renorm = {A_RENORM_K20}]")
    print("  kappa_eff_sig = sqrt(d_eff_sig) * kappa_nearest")
    print("  PASS: R2 > 0.8, slope within 20% of A_renorm, across ALL arms")
    print()

    train_ds, test_ds = get_cifar_coarse()
    all_results = {'ce': [], 'nc_margin': [], 'nc_etf': [], 'nc_full': []}

    for arm in ['ce', 'nc_margin', 'nc_etf', 'nc_full']:
        print(f"\n=== ARM: {arm} ===")
        for seed in range(N_SEEDS):
            print(f"\n--- seed={seed} ---")
            res = train_one_arm(seed, arm, train_ds, test_ds)
            print(f"  DONE: q={res['final_q']:.4f} d_eff_sig={res['final_d_eff_sig']:.3f}")
            all_results[arm].append(res)
            with open(RESULT_PATH, 'w') as f:
                json.dump({'status': 'running', 'results': all_results}, f,
                          default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    print("\n" + "=" * 70)
    print("ANALYSIS")
    analysis = analyze_results(all_results)

    primary = analysis.get('PRIMARY', {})
    print(f"\nPRIMARY TEST (kappa_eff_sig, fixed slope A_renorm={A_RENORM_K20}):")
    print(f"  R2 (fixed slope): {primary.get('r2_kappa_eff_sig_fixed_slope', 'N/A'):.3f}")
    print(f"  R2 (free slope):  {primary.get('r2_kappa_eff_sig_free_slope', 'N/A'):.3f}")
    print(f"  Empirical slope:  {primary.get('empirical_slope', 'N/A')}")
    print(f"  PASS: {primary.get('PASS', False)}")

    comps = analysis.get('COMPARISONS', {})
    print(f"\nCOMPARISON (free slopes):")
    for name, r2 in comps.get('ranking', []):
        print(f"  {name}: R2={r2:.3f}")

    print(f"\nFACTOR SEPARATION:")
    for arm, sep in analysis.get('FACTOR_SEPARATION', {}).items():
        print(f"  {arm}: delta_kappa={sep['delta_kappa']:+.4f} "
              f"delta_d_eff_sig={sep['delta_d_eff_sig']:+.4f} "
              f"kappa_dom={sep['kappa_dominated']} dsig_dom={sep['dsig_dominated']}")

    iso = analysis.get('ISO_KAPPA_EFF_SIG', {})
    print(f"\nISO-KAPPA_EFF_SIG TEST:")
    print(f"  Matched pairs: {iso.get('n_pairs', 0)}, PASS rate: {iso.get('pass_rate', 0):.2f}")
    print(f"  PASS: {iso.get('PASS', False)}")

    print(f"\nPER-ARM SUMMARY:")
    for arm, arm_stats in analysis.get('per_arm', {}).items():
        print(f"  {arm}: q={arm_stats['mean_q']:.4f} kappa={arm_stats['mean_kappa']:.4f} "
              f"d_eff_sig={arm_stats['mean_d_eff_sig']:.3f} "
              f"kappa_eff_sig={arm_stats['mean_kappa_eff_sig']:.4f}")

    final_results = {
        'status': 'complete',
        'results': all_results,
        'analysis': analysis,
        'A_renorm_preregistered': A_RENORM_K20,
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(final_results, f,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nResults saved to {RESULT_PATH}")


if __name__ == '__main__':
    main()
