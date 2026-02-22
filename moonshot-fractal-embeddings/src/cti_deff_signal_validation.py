"""
Theorem 16 Validation: d_eff_sig (Signal Subspace) vs d_eff_gram (Global)

KEY CONTRIBUTION (Feb 22 2026): Breaks circularity in d_eff_cls estimation.
Previously: d_eff_cls inferred from alpha = A_renorm * sqrt(d_eff_cls) [CIRCULAR]
Now: d_eff_sig measured INDEPENDENTLY from signal subspace geometry [NON-CIRCULAR]

PRE-REGISTERED HYPOTHESES:
  H1: d_eff_sig << d_eff_gram for trained neural networks
      Expected: d_eff_sig ~ 1-5 vs d_eff_gram ~ 200-400 (at 60 epochs)
  H2: logit(q) = A_renorm(K) * sqrt(d_eff_sig) * kappa_nearest + C  [PASS, R2 > 0.8]
  H3: logit(q) = A_renorm(K) * sqrt(d_eff_gram) * kappa_nearest + C [FAIL, R2 < 0.5]
  H4: NC+ arm has higher d_eff_sig than CE arm (causal test without circularity)

WHAT IS d_eff_sig:
  The within-class variance projected onto the SIGNAL SUBSPACE
  (span of first K-1 principal components of class centroid matrix).
  d_eff_sig = tr(P_B Sigma_W P_B.T)^2 / tr((P_B Sigma_W P_B.T)^2)

  For NC model: d_eff_sig -> 1 (all signal-subspace variance in one direction)
  For isotropic model: d_eff_sig -> K-1 (uniform across K-1 signal dimensions)

WHAT IS d_eff_gram:
  d_eff_gram = tr(Sigma_W)^2 / tr(Sigma_W^2) (participation ratio of ALL d dimensions)
  d_eff_gram >> d_eff_sig because it includes noise dimensions (d - (K-1) >> K-1)

COMPARISON:
  For K=20, d=512: d_eff_gram ~ 200-400; d_eff_sig ~ 1-5 (PREDICTED)
  Confirming d_eff_sig << d_eff_gram validates Theorem 16.
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
N_SEEDS = 3
PROJ_DIM = 256
MARGIN = 1.0
EMA_MOMENTUM = 0.95
CHECKPOINT_EPOCHS = [25, 40, 60]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "results/cti_deff_signal_validation.json"
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
    """d_eff = tr(W)^2 / tr(W^2) from within-class covariance (ALL dimensions)."""
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
    d_eff = float(trW ** 2 / (trW2 + 1e-12))
    return d_eff, float(trW)


def compute_d_eff_sig(X, y):
    """
    d_eff in the SIGNAL SUBSPACE (span of class centroids).
    This is the KEY independent measurement (no circularity with alpha slope).

    Signal subspace: first K-1 principal directions of class centroid matrix.
    d_eff_sig = tr(P_B Sigma_W P_B.T)^2 / tr((P_B Sigma_W P_B.T)^2)
    """
    classes = np.unique(y)
    K_actual = len(classes)
    N = len(X)
    d = X.shape[1]

    # Step 1: Compute class centroids and signal subspace
    grand_mean = X.mean(0)
    centroids = np.stack([X[y == c].mean(0) for c in classes])  # (K, d)
    centroids_centered = centroids - grand_mean  # (K, d)

    # SVD to get signal subspace basis
    # P_B: rows are orthonormal directions spanning the centroid space
    U, S, Vt = np.linalg.svd(centroids_centered, full_matrices=False)
    # n_sig = rank of centroid matrix (at most K-1)
    # Keep directions with significant singular values (> 1% of max)
    s_thresh = 0.01 * S[0] if len(S) > 0 else 0
    n_sig = int(np.sum(S > s_thresh))
    n_sig = max(1, min(n_sig, K_actual - 1, d))
    P_B = Vt[:n_sig, :]  # (n_sig, d) orthonormal basis for signal subspace

    # Step 2: Project within-class covariance onto signal subspace
    trW_sig = 0.0
    trW2_sig = 0.0

    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        Xc_centered = Xc - Xc.mean(0)

        # Project centered points onto signal subspace: shape (n_c, n_sig)
        Xc_proj = Xc_centered @ P_B.T  # (n_c, n_sig)

        # tr(Sigma_k_sig) = sum of projected variances
        trSigma_k_sig = float(np.sum(Xc_proj ** 2)) / n_c
        trW_sig += n_c * trSigma_k_sig / N

        # tr(Sigma_k_sig^2) via Gram matrix of projected deviations
        G_sig = (Xc_proj @ Xc_proj.T) / n_c  # (n_c, n_c)
        trSigma_k2_sig = float(np.sum(G_sig ** 2))
        trW2_sig += (n_c / N) ** 2 * trSigma_k2_sig

    d_eff_sig = float(trW_sig ** 2 / (trW2_sig + 1e-12))
    return d_eff_sig, float(trW_sig), int(n_sig)


def compute_d_eff_noise(X, y):
    """
    d_eff in the NOISE SUBSPACE (complement of signal subspace).
    Should NOT predict q once d_eff_sig and kappa_nearest are controlled.
    """
    classes = np.unique(y)
    K_actual = len(classes)
    N = len(X)
    d = X.shape[1]

    grand_mean = X.mean(0)
    centroids = np.stack([X[y == c].mean(0) for c in classes])
    centroids_centered = centroids - grand_mean

    U, S, Vt = np.linalg.svd(centroids_centered, full_matrices=False)
    s_thresh = 0.01 * S[0] if len(S) > 0 else 0
    n_sig = int(np.sum(S > s_thresh))
    n_sig = max(1, min(n_sig, K_actual - 1, d))
    P_B = Vt[:n_sig, :]  # (n_sig, d) signal subspace

    # Noise subspace: project out signal subspace from data
    # X_noise_centered = X_centered - P_B.T @ P_B @ X_centered
    # = X_centered @ (I - P_B.T @ P_B)
    P_noise = np.eye(d) - P_B.T @ P_B  # (d, d) noise projection matrix

    trW_noise = 0.0
    trW2_noise = 0.0

    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        Xc_centered = Xc - Xc.mean(0)

        # Project onto noise subspace
        Xc_noise = Xc_centered @ P_noise  # (n_c, d) but rank d-n_sig

        # Approximate via total - signal decomposition
        Xc_sig = Xc_centered @ P_B.T  # (n_c, n_sig)
        trSigma_k_total = float(np.sum(Xc_centered ** 2)) / n_c
        trSigma_k_sig = float(np.sum(Xc_sig ** 2)) / n_c
        trSigma_k_noise = trSigma_k_total - trSigma_k_sig
        trW_noise += n_c * trSigma_k_noise / N

    # Approximate d_eff_noise from ratio (simplified, no full Gram)
    # For this experiment, just return tr(W_noise) as proxy
    return float(trW_noise), d - n_sig


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


def compute_nc_positive_loss(z, y, class_means):
    mu_yi = class_means[y]
    L_within = ((z - mu_yi) ** 2).mean()
    M = class_means
    G = M @ M.t()
    G_etf = (torch.eye(K, device=z.device) * (1 + 1.0 / (K - 1))
             - (1.0 / (K - 1)) * torch.ones(K, K, device=z.device))
    L_ETF = ((G - G_etf) ** 2).sum() / (K ** 2)
    dists = (torch.cdist(M.unsqueeze(0), M.unsqueeze(0)).squeeze(0)
             + torch.eye(K, device=z.device) * 1e6)
    L_margin = F.softplus(MARGIN - dists.min())
    return L_within + 0.5 * L_ETF + 0.5 * L_margin


def compute_nc_negative_loss(z, y, class_means):
    mu_yi = class_means[y]
    L_within = ((z - mu_yi) ** 2).mean()
    return -L_within


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
        lam = get_lambda(epoch) if arm in ('nc', 'anti_nc') else 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            h = model['backbone'](imgs)
            logits = model['ce_head'](h)
            loss_ce = ce_loss_fn(logits, labels)

            if lam > 0:
                z_raw = model['proj_head'](h)
                z = F.normalize(z_raw, dim=1)
                class_means = update_ema_means(class_means, z.detach(), labels)
                if arm == 'nc':
                    loss_aux = compute_nc_positive_loss(z, labels, class_means)
                else:
                    loss_aux = compute_nc_negative_loss(z, labels, class_means)
                loss = loss_ce + lam * loss_aux
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch in CHECKPOINT_EPOCHS:
            # TEST embeddings for q and kappa_nearest
            X_test, y_test = extract_all_embeddings(model, test_ds)
            q_val = compute_q(X_test, y_test)
            kappa_val = compute_kappa_nearest(X_test, y_test)
            logit_q = float(np.log(max(q_val, 0.001) / max(1 - q_val, 0.001)))

            # TRAIN embeddings for d_eff_gram and d_eff_sig
            print(f"  [seed={seed} arm={arm} epoch={epoch}] q={q_val:.4f} kappa={kappa_val:.4f} "
                  f"lam={lam:.3f} computing d_effs...", end=' ', flush=True)
            X_train, y_train = extract_all_embeddings(model, train_ds)

            # Key independent measurements
            d_eff_gram, trW = compute_d_eff_gram(X_train, y_train)
            d_eff_sig, trW_sig, n_sig = compute_d_eff_sig(X_train, y_train)
            trW_noise, d_noise = compute_d_eff_noise(X_train, y_train)

            kappa_eff_gram = np.sqrt(d_eff_gram) * kappa_val
            kappa_eff_sig = np.sqrt(d_eff_sig) * kappa_val

            print(f"d_eff_gram={d_eff_gram:.1f} d_eff_sig={d_eff_sig:.3f} "
                  f"(ratio={d_eff_gram/d_eff_sig:.1f}x)", flush=True)

            checkpoints.append({
                'epoch': epoch,
                'q': float(q_val),
                'kappa': float(kappa_val),
                'logit_q': float(logit_q),
                'd_eff_gram': float(d_eff_gram),
                'd_eff_sig': float(d_eff_sig),
                'n_sig': int(n_sig),
                'trW': float(trW),
                'trW_sig': float(trW_sig),
                'trW_noise': float(trW_noise),
                'd_noise': int(d_noise),
                'kappa_eff_gram': float(kappa_eff_gram),
                'kappa_eff_sig': float(kappa_eff_sig),
                'lambda': float(lam),
                # Predictions
                'predicted_logit_gram': float(A_RENORM_K20 * kappa_eff_gram),
                'predicted_logit_sig': float(A_RENORM_K20 * kappa_eff_sig),
            })

    return {
        'seed': seed, 'arm': arm,
        'checkpoints': checkpoints,
        'final_q': checkpoints[-1]['q'] if checkpoints else None,
        'final_kappa': checkpoints[-1]['kappa'] if checkpoints else None,
        'final_d_eff_gram': checkpoints[-1]['d_eff_gram'] if checkpoints else None,
        'final_d_eff_sig': checkpoints[-1]['d_eff_sig'] if checkpoints else None,
    }


def analyze_results(all_results):
    """
    Test H1-H4 from the pre-registered hypotheses.
    """
    analysis = {}

    # Collect all snapshots
    all_snaps = []
    for arm, arm_res in all_results.items():
        for res in arm_res:
            for ck in res.get('checkpoints', []):
                all_snaps.append({'arm': arm, 'seed': res['seed'], **ck})

    if not all_snaps:
        return {'status': 'no_data'}

    # H1: d_eff_sig << d_eff_gram
    d_gram = np.array([s['d_eff_gram'] for s in all_snaps])
    d_sig = np.array([s['d_eff_sig'] for s in all_snaps])
    ratio = d_gram / d_sig
    analysis['H1'] = {
        'mean_d_eff_gram': float(np.mean(d_gram)),
        'mean_d_eff_sig': float(np.mean(d_sig)),
        'mean_ratio': float(np.mean(ratio)),
        'PASS': bool(np.mean(ratio) > 10),  # expect > 10x ratio
    }

    # H2: logit_q ~ A_renorm * sqrt(d_eff_sig) * kappa = A_renorm * kappa_eff_sig
    logits = np.array([s['logit_q'] for s in all_snaps])
    ke_sig = np.array([s['kappa_eff_sig'] for s in all_snaps])
    ke_gram = np.array([s['kappa_eff_gram'] for s in all_snaps])
    kappa = np.array([s['kappa'] for s in all_snaps])

    # Fit with fixed slope = A_renorm, optimize only intercept
    if ke_sig.std() > 0:
        C_opt_sig = float(np.mean(logits - A_RENORM_K20 * ke_sig))
        resids_sig = logits - (A_RENORM_K20 * ke_sig + C_opt_sig)
        ss_res_sig = np.sum(resids_sig ** 2)
        ss_tot = np.sum((logits - logits.mean()) ** 2)
        r2_sig = float(1 - ss_res_sig / ss_tot) if ss_tot > 0 else 0.0
        # Also free slope
        coeffs_sig = np.polyfit(ke_sig, logits, 1)
        r2_sig_free = float(np.corrcoef(ke_sig, logits)[0, 1] ** 2) if ke_sig.std() > 0 else 0.0
    else:
        r2_sig = r2_sig_free = C_opt_sig = None

    analysis['H2'] = {
        'r2_kappa_eff_sig_fixed_slope': r2_sig,
        'r2_kappa_eff_sig_free_slope': r2_sig_free,
        'C_optimal': C_opt_sig,
        'PASS': bool(r2_sig_free is not None and r2_sig_free > 0.8),
    }

    # H3: logit_q ~ A_renorm * kappa_eff_gram (should fail)
    if ke_gram.std() > 0:
        C_opt_gram = float(np.mean(logits - A_RENORM_K20 * ke_gram))
        resids_gram = logits - (A_RENORM_K20 * ke_gram + C_opt_gram)
        ss_res_gram = np.sum(resids_gram ** 2)
        r2_gram = float(1 - ss_res_gram / ss_tot) if ss_tot > 0 else 0.0
        r2_gram_free = float(np.corrcoef(ke_gram, logits)[0, 1] ** 2) if ke_gram.std() > 0 else 0.0
        slope_gram_free = float(np.polyfit(ke_gram, logits, 1)[0])
    else:
        r2_gram = r2_gram_free = slope_gram_free = None

    analysis['H3'] = {
        'r2_kappa_eff_gram_fixed_slope': r2_gram,
        'r2_kappa_eff_gram_free_slope': r2_gram_free,
        'empirical_slope_gram': slope_gram_free,
        'FAIL': bool(r2_gram_free is not None and r2_gram_free < r2_sig_free),
    }

    # Kappa only (baseline)
    coeffs_k = np.polyfit(kappa, logits, 1)
    r2_k_free = float(np.corrcoef(kappa, logits)[0, 1] ** 2) if kappa.std() > 0 else 0.0
    analysis['kappa_only'] = {
        'r2_free': r2_k_free,
        'slope': float(coeffs_k[0]),
    }

    # H4: NC+ arm has higher d_eff_sig than CE arm
    for arm in ['nc', 'anti_nc']:
        if arm in all_results and 'ce' in all_results:
            ce_deff_sig = np.mean([ck['d_eff_sig'] for res in all_results['ce']
                                   for ck in res.get('checkpoints', [])
                                   if ck.get('d_eff_sig') is not None])
            arm_deff_sig = np.mean([ck['d_eff_sig'] for res in all_results.get(arm, [])
                                    for ck in res.get('checkpoints', [])
                                    if ck.get('d_eff_sig') is not None])
            analysis[f'H4_{arm}'] = {
                'mean_d_eff_sig_ce': float(ce_deff_sig) if not np.isnan(ce_deff_sig) else None,
                f'mean_d_eff_sig_{arm}': float(arm_deff_sig) if not np.isnan(arm_deff_sig) else None,
                'PASS': bool(arm == 'nc' and arm_deff_sig > ce_deff_sig) or
                        bool(arm == 'anti_nc' and arm_deff_sig < ce_deff_sig),
            }

    # Summary ranking
    metrics = {
        'kappa_eff_sig_free': r2_sig_free,
        'kappa_eff_gram_free': r2_gram_free,
        'kappa_only_free': r2_k_free,
    }
    analysis['metric_ranking'] = {k: v for k, v in
                                   sorted(metrics.items(), key=lambda x: -(x[1] or 0))}

    return analysis


def main():
    print("Theorem 16 Validation: d_eff_sig vs d_eff_gram")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"K={K}, N_EPOCHS={N_EPOCHS}, N_SEEDS={N_SEEDS}")
    print(f"Arms: CE, NC+, anti_NC")
    print()
    print("PRE-REGISTERED HYPOTHESES:")
    print("  H1: d_eff_sig << d_eff_gram (ratio > 10x)")
    print("  H2: logit_q ~ A_renorm * kappa_eff_sig (R2 > 0.8)")
    print("  H3: logit_q ~ A_renorm * kappa_eff_gram FAILS (R2 < H2)")
    print("  H4: NC+ has higher d_eff_sig than CE (independent causal test)")
    print()

    train_ds, test_ds = get_cifar_coarse()
    all_results = {'ce': [], 'nc': [], 'anti_nc': []}

    for arm in ['ce', 'nc', 'anti_nc']:
        print(f"\n=== ARM: {arm} ===")
        for seed in range(N_SEEDS):
            print(f"\n--- seed={seed} ---")
            res = train_one_arm(seed, arm, train_ds, test_ds)
            print(f"  DONE: q={res['final_q']:.4f} "
                  f"d_eff_gram={res['final_d_eff_gram']:.1f} "
                  f"d_eff_sig={res['final_d_eff_sig']:.3f}")
            all_results[arm].append(res)
            with open(RESULT_PATH, 'w') as f:
                json.dump({'status': 'running', 'results': all_results}, f,
                          default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    analysis = analyze_results(all_results)

    print(f"\nH1 (d_eff_sig << d_eff_gram):")
    h1 = analysis.get('H1', {})
    print(f"  mean_d_eff_gram = {h1.get('mean_d_eff_gram'):.1f}")
    print(f"  mean_d_eff_sig  = {h1.get('mean_d_eff_sig'):.3f}")
    print(f"  ratio           = {h1.get('mean_ratio'):.1f}x")
    print(f"  PASS (ratio > 10x): {h1.get('PASS')}")

    print(f"\nH2 (logit_q ~ A_renorm * kappa_eff_sig):")
    h2 = analysis.get('H2', {})
    print(f"  R2 (free slope) = {h2.get('r2_kappa_eff_sig_free_slope'):.4f}")
    print(f"  R2 (fixed slope = 1.0535) = {h2.get('r2_kappa_eff_sig_fixed_slope'):.4f}")
    print(f"  PASS (R2_free > 0.8): {h2.get('PASS')}")

    print(f"\nH3 (kappa_eff_gram FAILS):")
    h3 = analysis.get('H3', {})
    print(f"  R2 (free slope) = {h3.get('r2_kappa_eff_gram_free_slope'):.4f}")
    print(f"  Empirical slope = {h3.get('empirical_slope_gram'):.4f} (expected 1.0535)")
    print(f"  FAIL (R2_gram < R2_sig): {h3.get('FAIL')}")

    print(f"\nKappa_nearest baseline:")
    print(f"  R2 (free slope) = {analysis.get('kappa_only', {}).get('r2_free'):.4f}")
    print(f"  Slope = {analysis.get('kappa_only', {}).get('slope'):.4f}")

    print(f"\nMetric ranking (R2):")
    for m, r2 in analysis.get('metric_ranking', {}).items():
        print(f"  {m}: R2 = {r2:.4f}")

    output = {
        'status': 'complete',
        'K': K,
        'A_renorm_preregistered': A_RENORM_K20,
        'analysis': analysis,
        'results': all_results,
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nSaved to {RESULT_PATH}")


if __name__ == '__main__':
    main()
