"""
d_eff Extraction: Verify Renormalized Universality Empirically.

GOAL: For ResNet-18 on CIFAR-100 coarse after CE training:
  1. Extract train embeddings at final epoch
  2. Compute d_eff from within-class covariance: d_eff = tr(W)^2 / tr(W^2)
  3. Compare with d_eff_implied from alpha (training trajectory)
  4. Verify: alpha / sqrt(d_eff_measured) = sqrt(4/pi) = 1.1284

THEOREM 14 PREDICTION: alpha/sqrt(d_eff) = sqrt(4/pi) universally.

If this passes, Renormalized Universality is empirically confirmed on a REAL model.
This bridges the gap between "analytical proof" and "empirical validation."
"""

import numpy as np
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit


SQRT_4_OVER_PI = np.sqrt(4.0 / np.pi)
K = 20
N_EPOCHS = 60
WARMUP_EPOCHS = 25
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 5e-4
CHECKPOINT_EPOCHS = [0, 25, 40, 60]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def coarse_label(x):
    return x // 5


def get_cifar_coarse():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100('data', train=True, download=False,
        transform=train_transform, target_transform=coarse_label)
    test_ds = torchvision.datasets.CIFAR100('data', train=False, download=False,
        transform=test_transform, target_transform=coarse_label)
    return train_ds, test_ds


def get_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    ce_head = nn.Linear(512, K)
    model = nn.ModuleDict({'backbone': backbone, 'ce_head': ce_head})
    return model.to(DEVICE)


def extract_embeddings(model, dataset, device=DEVICE):
    """Extract full embedding matrix from training set."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            embs.append(model['backbone'](imgs.to(device)).cpu().numpy())
            labels.append(lbs.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def compute_kappa_nearest(X, y, K=K):
    """Compute kappa_nearest from embedding matrix."""
    classes = np.unique(y)
    d = X.shape[1]
    means = {c: X[y == c].mean(0) for c in classes}
    within_vars = [np.mean(np.sum((X[y == c] - means[c])**2, axis=1)) for c in classes]
    sigma_W = np.sqrt(np.mean(within_vars) / d)
    min_dist = min(np.linalg.norm(means[classes[i]] - means[classes[j]])
                   for i in range(len(classes)) for j in range(i+1, len(classes)))
    return float(min_dist / (sigma_W * np.sqrt(d) + 1e-10))


def compute_d_eff(X, y):
    """
    Compute effective dimensionality d_eff from within-class covariance.

    d_eff = tr(W)^2 / tr(W^2)

    where W = pooled within-class covariance matrix (d x d).
    This is the participation ratio = inverse of concentration of eigenvalues.

    For d >> n_per: use tr(W^2) computed via Frobenius norm of Gram matrix.
    NUMERICALLY STABLE: Use W = X_c.T @ X_c / n where X_c is mean-centered.

    Note: d_eff is the "effective number of dimensions with non-zero variance."
    Perfect NC: W = sigma^2 * I_d (uniform), but within-class all collapses to 0.
    Near-NC: W is low-rank, d_eff ~ 1 to K-1.
    """
    classes = np.unique(y)
    d = X.shape[1]

    # Compute within-class covariance: W = sum_k n_k * Sigma_k / N
    # For numerical stability with d=512, n_k=2500: use Gram approach
    # d_eff = (sum eigenvalues)^2 / (sum eigenvalues^2) = tr(W)^2 / tr(W^2)

    # Compute tr(W) and tr(W^2) efficiently without forming W explicitly
    # W = (1/N) * sum_k sum_{i in k} (x_i - mu_k)(x_i - mu_k)^T
    # tr(W) = (1/N) * sum_k sum_{i in k} ||x_i - mu_k||^2   [scalar]
    # tr(W^2) = ||W||_F^2 = harder... use Gram approach

    N = len(X)
    trW = 0.0
    trW2 = 0.0

    for c in classes:
        Xc = X[y == c]
        n_c = len(Xc)
        mu_c = Xc.mean(0)
        Xc_centered = Xc - mu_c  # (n_c, d)

        # tr(Sigma_k) = tr(Xc_centered.T @ Xc_centered / n_c) = ||Xc_centered||_F^2 / n_c
        trSigma_k = float(np.sum(Xc_centered**2)) / n_c
        trW += n_c * trSigma_k / N  # weighted by class size

        # tr(Sigma_k^2): use Gram matrix G = Xc_centered @ Xc_centered.T / n_c (n_c x n_c)
        # tr(Sigma_k^2) = tr((X.T X / n)^2) = ||X.T X / n||_F^2
        # = sum_{ij} (X_i . X_j / n)^2 / n^2
        # = tr(G^2) / n_c^2 where G = Xc_centered @ Xc_centered.T
        G = (Xc_centered @ Xc_centered.T) / n_c  # (n_c, n_c)
        trSigma_k2 = float(np.sum(G**2))  # = tr(G^2) = ||G||_F^2
        trW2 += (n_c / N)**2 * trSigma_k2  # weighted by (class fraction)^2

    d_eff = float(trW**2 / (trW2 + 1e-12))
    return d_eff, trW, trW2


def compute_q_from_embs(X, y, K=K):
    """Compute q from embedding matrix."""
    sss = StratifiedShuffleSplit(1, test_size=0.3, random_state=42)
    tr, te = next(sss.split(X, y))
    knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
    knn.fit(X[tr], y[tr])
    acc = float(knn.score(X[te], y[te]))
    return (acc - 1.0/K) / (1.0 - 1.0/K)


def main():
    print("d_eff Extraction: Verify Renormalized Universality")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Target: alpha/sqrt(d_eff) = sqrt(4/pi) = {SQRT_4_OVER_PI:.4f}")
    print()

    train_ds, test_ds = get_cifar_coarse()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    results = []
    rng = np.random.default_rng(42)

    # Run for 3 seeds
    for seed in [0, 1, 2]:
        print(f"--- seed={seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = get_model()
        optimizer = optim.SGD(list(model['backbone'].parameters()) +
                              list(model['ce_head'].parameters()),
                              lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
        criterion = nn.CrossEntropyLoss()

        checkpoints = []

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            for imgs, lbs in train_loader:
                imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
                optimizer.zero_grad()
                feats = model['backbone'](imgs)
                logits = model['ce_head'](feats)
                loss = criterion(logits, lbs)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if epoch in CHECKPOINT_EPOCHS:
                # Evaluate on test set
                test_loader = torch.utils.data.DataLoader(
                    test_ds, batch_size=512, shuffle=False, num_workers=0)
                model.eval()
                embs_test, labels_test = [], []
                with torch.no_grad():
                    for imgs, lbs in test_loader:
                        embs_test.append(model['backbone'](imgs.to(DEVICE)).cpu().numpy())
                        labels_test.append(lbs.numpy())
                X_test = np.concatenate(embs_test)
                y_test = np.concatenate(labels_test)

                kappa = compute_kappa_nearest(X_test, y_test)
                q = compute_q_from_embs(X_test, y_test)
                checkpoints.append({'epoch': epoch, 'kappa': kappa, 'q': float(q)})
                print(f"  epoch={epoch}: kappa={kappa:.4f} q={q:.4f}")

        # Extract TRAINING embeddings at final epoch for d_eff computation
        print(f"  Extracting train embeddings for d_eff...", end=' ', flush=True)
        X_train, y_train = extract_embeddings(model, train_ds)
        d_eff, trW, trW2 = compute_d_eff(X_train, y_train)
        print(f"d_eff={d_eff:.4f}")

        # Fit alpha from training trajectory
        valid = [(ck['kappa'], ck['q']) for ck in checkpoints if 0 < ck['q'] < 1]
        if len(valid) >= 3:
            kappas = np.array([v[0] for v in valid])
            logit_qs = np.log([v[1]/(1-v[1]) for v in valid])
            X_fit = np.column_stack([kappas, np.ones(len(kappas))])
            coeffs, _, _, _ = np.linalg.lstsq(X_fit, logit_qs, rcond=None)
            alpha = float(coeffs[0])
        else:
            alpha = None

        A_renorm = alpha / np.sqrt(d_eff) if alpha and d_eff > 0 else None
        d_eff_implied = (alpha / SQRT_4_OVER_PI)**2 if alpha else None

        print(f"  alpha={alpha:.4f}  d_eff_measured={d_eff:.4f}  "
              f"d_eff_implied={d_eff_implied:.4f}  A_renorm={A_renorm:.4f}  "
              f"target={SQRT_4_OVER_PI:.4f}")
        print(f"  Renorm error: {abs(A_renorm - SQRT_4_OVER_PI)/SQRT_4_OVER_PI:.4f}")
        print()

        results.append({
            'seed': seed,
            'checkpoints': checkpoints,
            'alpha': alpha,
            'd_eff_measured': float(d_eff),
            'd_eff_implied': d_eff_implied,
            'trW': float(trW),
            'trW2': float(trW2),
            'A_renorm': A_renorm,
            'target_constant': SQRT_4_OVER_PI,
            'renorm_error': float(abs(A_renorm - SQRT_4_OVER_PI)/SQRT_4_OVER_PI) if A_renorm else None,
        })

    # Summary
    print("=== SUMMARY: RENORMALIZED UNIVERSALITY TEST ===")
    alphas = [r['alpha'] for r in results if r['alpha']]
    d_effs = [r['d_eff_measured'] for r in results if r['d_eff_measured']]
    A_renorms = [r['A_renorm'] for r in results if r['A_renorm']]
    d_effs_implied = [r['d_eff_implied'] for r in results if r['d_eff_implied']]

    print(f"  alpha: mean={np.mean(alphas):.4f} std={np.std(alphas):.4f}")
    print(f"  d_eff_measured: mean={np.mean(d_effs):.4f} std={np.std(d_effs):.4f}")
    print(f"  d_eff_implied: mean={np.mean(d_effs_implied):.4f}")
    print(f"  d_eff ratio (implied/measured): {np.mean(d_effs_implied)/np.mean(d_effs):.4f}")
    print(f"  A_renorm: mean={np.mean(A_renorms):.4f} std={np.std(A_renorms):.4f}")
    print(f"  Target: sqrt(4/pi) = {SQRT_4_OVER_PI:.4f}")
    print(f"  Match error: {abs(np.mean(A_renorms) - SQRT_4_OVER_PI)/SQRT_4_OVER_PI:.4f}")
    print()
    print("  THEOREM 14 PREDICTION: A_renorm = sqrt(4/pi) = 1.1284")
    print(f"  RESULT: A_renorm = {np.mean(A_renorms):.4f}")
    d_eff_vs_implied = np.mean(d_effs_implied) / np.mean(d_effs)
    if abs(d_eff_vs_implied - 1.0) < 0.3:
        print(f"  d_eff measured vs implied: {d_eff_vs_implied:.3f} [CONSISTENT]")
    else:
        print(f"  d_eff measured vs implied: {d_eff_vs_implied:.3f} [DISCREPANCY]")

    output = {
        'theorem': 'Theorem 14: A_renorm = alpha/sqrt(d_eff) = sqrt(4/pi)',
        'target_constant': SQRT_4_OVER_PI,
        'results': results,
        'summary': {
            'mean_alpha': float(np.mean(alphas)),
            'mean_d_eff_measured': float(np.mean(d_effs)),
            'mean_d_eff_implied': float(np.mean(d_effs_implied)),
            'd_eff_ratio': float(np.mean(d_effs_implied)/np.mean(d_effs)),
            'mean_A_renorm': float(np.mean(A_renorms)),
            'std_A_renorm': float(np.std(A_renorms)),
            'renorm_error': float(abs(np.mean(A_renorms) - SQRT_4_OVER_PI)/SQRT_4_OVER_PI),
            'PASS': bool(abs(np.mean(A_renorms) - SQRT_4_OVER_PI)/SQRT_4_OVER_PI < 0.10),
        }
    }

    out_path = 'results/cti_deff_extraction.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
