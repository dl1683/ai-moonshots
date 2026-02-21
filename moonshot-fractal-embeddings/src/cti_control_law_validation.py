"""
Control Law Validation: Delta logit(q) = A_renorm(K) * Delta(kappa_eff) + epsilon

CLOSING THE KEY GAP (Codex, Feb 22 2026):
  To go from 3/10 to 7/10 Nobel, prove:
    Delta logit(q) = A_renorm(K) * Delta(kappa_eff) + epsilon
  with epsilon tightly bounded and INVARIANT across:
    - CE arm (baseline)
    - NC+ arm (increase kappa_eff via NC-loss)
    - NC- arm (decrease kappa_eff via -L_within)
    - At least one orthogonal intervention

  This requires DIRECT d_eff MEASUREMENT (not inference from alpha).

METHOD:
  1. Train CE, NC+, NC- for 60 epochs (3 seeds each)
  2. At checkpoints 25, 40, 60: extract FULL TRAIN EMBEDDINGS, compute d_eff
  3. Compute kappa_eff = sqrt(d_eff) * kappa_nearest at each checkpoint
  4. Test: Delta logit(q) ≈ A_renorm(K) * Delta(kappa_eff) across arms

PRE-REGISTERED HYPOTHESIS:
  For any two checkpoints (t1, t2) in any arm:
    [logit(q(t2)) - logit(q(t1))] ≈ A_renorm(K) * [kappa_eff(t2) - kappa_eff(t1)]

  Equivalently: slope of logit(q) vs kappa_eff = A_renorm(K) = 1.0535 (K=20)

  If this holds with R2 > 0.9 across ALL arms: control law confirmed.
  If epsilon is invariant across CE, NC+, NC-: causal sufficiency established.
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
RESULT_PATH = "results/cti_control_law_validation.json"

# Pre-registered constants (Theorem 15)
A_RENORM_K20 = 1.0535


def get_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    ce_head = nn.Linear(512, K)
    proj_head = nn.Sequential(nn.Linear(512, PROJ_DIM), nn.BatchNorm1d(PROJ_DIM))
    model = nn.ModuleDict({'backbone': backbone, 'ce_head': ce_head, 'proj_head': proj_head})
    return model.to(DEVICE)


def coarse_label(x): return x // 5


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
    """Extract full embedding matrix from dataset (for d_eff computation)."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0)
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            embs.append(model['backbone'](imgs.to(DEVICE)).cpu().numpy())
            labels.append(lbs.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def compute_d_eff(X, y):
    """
    d_eff = tr(W)^2 / tr(W^2) from within-class covariance.
    Uses Gram matrix for numerical stability.
    """
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

        # Gram matrix: G = Xc_centered @ Xc_centered.T / n_c
        G = (Xc_centered @ Xc_centered.T) / n_c  # (n_c, n_c)
        trSigma_k2 = float(np.sum(G ** 2))  # = tr(G^2)
        trW2 += (n_c / N) ** 2 * trSigma_k2

    d_eff = float(trW ** 2 / (trW2 + 1e-12))
    return d_eff, float(trW), float(trW2)


def compute_kappa_nearest(X, y):
    """Compute kappa_nearest from embedding matrix."""
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
    """Compute normalized 1-NN accuracy q from test embeddings."""
    sss = StratifiedShuffleSplit(1, test_size=0.3, random_state=random_state)
    tr, te = next(sss.split(X, y))
    knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
    knn.fit(X[tr], y[tr])
    acc = float(knn.score(X[te], y[te]))
    return (acc - 1.0 / K) / (1.0 - 1.0 / K)


# NC loss components
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
    """NC-negative: -L_within only. Decreases kappa_nearest."""
    mu_yi = class_means[y]
    L_within = ((z - mu_yi) ** 2).mean()
    return -L_within  # NEGATIVE: maximize within-class spread


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
                else:  # anti_nc
                    loss_aux = compute_nc_negative_loss(z, labels, class_means)
                loss = loss_ce + lam * loss_aux
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch in CHECKPOINT_EPOCHS:
            # Extract TEST embeddings for q and kappa
            X_test, y_test = extract_all_embeddings(model, test_ds)
            q_val = compute_q(X_test, y_test)
            kappa_val = compute_kappa_nearest(X_test, y_test)

            # Extract TRAIN embeddings for d_eff (expensive but necessary)
            print(f"  [seed={seed} arm={arm} epoch={epoch}] "
                  f"q={q_val:.4f} kappa={kappa_val:.4f} lam={lam:.3f} "
                  f"extracting train embs...", end=' ', flush=True)
            X_train, y_train = extract_all_embeddings(model, train_ds)
            d_eff, trW, trW2 = compute_d_eff(X_train, y_train)
            kappa_eff = np.sqrt(d_eff) * kappa_val
            logit_q = float(np.log(max(q_val, 0.001) / max(1 - q_val, 0.001)))

            print(f"d_eff={d_eff:.3f} kappa_eff={kappa_eff:.4f}", flush=True)

            checkpoints.append({
                'epoch': epoch,
                'q': float(q_val),
                'kappa': float(kappa_val),
                'd_eff': float(d_eff),
                'kappa_eff': float(kappa_eff),
                'logit_q': float(logit_q),
                'trW': float(trW),
                'trW2': float(trW2),
                'lambda': float(lam),
            })

    return {
        'seed': seed, 'arm': arm,
        'final_q': checkpoints[-1]['q'] if checkpoints else None,
        'final_kappa': checkpoints[-1]['kappa'] if checkpoints else None,
        'final_d_eff': checkpoints[-1]['d_eff'] if checkpoints else None,
        'final_kappa_eff': checkpoints[-1]['kappa_eff'] if checkpoints else None,
        'checkpoints': checkpoints,
    }


def analyze_control_law(all_results, A_renorm=A_RENORM_K20):
    """
    Test: Delta logit(q) = A_renorm * Delta(kappa_eff) + epsilon
    """
    # Collect all (delta_logit_q, delta_kappa_eff) pairs across arms
    all_deltas = []
    arm_deltas = {}

    for arm, arm_results in all_results.items():
        arm_deltas[arm] = []
        for res in arm_results:
            ckpts = res.get('checkpoints', [])
            for i in range(len(ckpts)):
                for j in range(i + 1, len(ckpts)):
                    c1, c2 = ckpts[i], ckpts[j]
                    delta_logit = c2['logit_q'] - c1['logit_q']
                    delta_kappa_eff = c2['kappa_eff'] - c1['kappa_eff']
                    pair = {
                        'arm': arm, 'seed': res['seed'],
                        'epoch1': c1['epoch'], 'epoch2': c2['epoch'],
                        'delta_logit_q': delta_logit,
                        'delta_kappa_eff': delta_kappa_eff,
                        'predicted': A_renorm * delta_kappa_eff,
                        'residual': delta_logit - A_renorm * delta_kappa_eff,
                    }
                    all_deltas.append(pair)
                    arm_deltas[arm].append(pair)

    if not all_deltas:
        return None

    # Overall R2 for control law
    actuals = np.array([p['delta_logit_q'] for p in all_deltas])
    predicted = np.array([p['predicted'] for p in all_deltas])
    residuals = actuals - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Residual statistics
    resid_mean = float(np.mean(residuals))
    resid_std = float(np.std(residuals))

    # Per-arm residuals
    per_arm_resid = {}
    for arm, pairs in arm_deltas.items():
        if pairs:
            arm_resids = np.array([p['residual'] for p in pairs])
            per_arm_resid[arm] = {
                'mean': float(np.mean(arm_resids)),
                'std': float(np.std(arm_resids)),
            }

    # Invariance test: are residuals similar across arms?
    arm_means = {arm: d['mean'] for arm, d in per_arm_resid.items() if d}
    if len(arm_means) >= 2:
        mean_values = list(arm_means.values())
        max_diff = max(mean_values) - min(mean_values)
        invariant = bool(max_diff < 0.1)  # threshold: 0.1 logit units
    else:
        invariant = None
        max_diff = None

    return {
        'r2_control_law': float(r2),
        'A_renorm_preregistered': A_renorm,
        'residual_mean': resid_mean,
        'residual_std': resid_std,
        'per_arm_residuals': per_arm_resid,
        'invariant_across_arms': invariant,
        'max_arm_mean_diff': float(max_diff) if max_diff is not None else None,
        'n_pairs': len(all_deltas),
        'PASS_r2': bool(r2 > 0.9),
        'PASS_invariance': bool(invariant) if invariant is not None else False,
        'all_pairs': all_deltas,
    }


def main():
    print("Control Law Validation: Delta logit(q) = A_renorm(K) * Delta(kappa_eff)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"A_renorm(K=20) = {A_RENORM_K20} (pre-registered from Theorem 15)")
    print(f"Arms: CE, NC+ (maximize kappa_eff), NC- (minimize kappa_eff)")
    print()
    print("NOTE: Extracts FULL TRAIN EMBEDDINGS at each checkpoint (expensive)")
    print("       Adds ~10-20 min per seed vs NC quick pilot")
    print()

    train_ds, test_ds = get_cifar_coarse()
    all_results = {'ce': [], 'nc': [], 'anti_nc': []}

    for arm in ['ce', 'nc', 'anti_nc']:
        print(f"\n=== ARM: {arm} ===")
        for seed in range(N_SEEDS):
            print(f"\n--- seed={seed} ---")
            res = train_one_arm(seed, arm, train_ds, test_ds)
            final_q = res.get('final_q', 0)
            final_k = res.get('final_kappa', 0)
            final_d_eff = res.get('final_d_eff', 0)
            final_ke = res.get('final_kappa_eff', 0)
            print(f"  DONE: q={final_q:.4f} kappa={final_k:.4f} "
                  f"d_eff={final_d_eff:.3f} kappa_eff={final_ke:.4f}")
            all_results[arm].append(res)
            with open(RESULT_PATH, 'w') as f:
                json.dump({'status': 'running', 'results': all_results}, f,
                          default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    # ================================================================
    # Control Law Analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("CONTROL LAW ANALYSIS")
    print("=" * 70)

    analysis = analyze_control_law(all_results)

    if analysis:
        print(f"  Delta logit(q) = A_renorm * Delta(kappa_eff) + epsilon")
        print(f"  A_renorm = {A_RENORM_K20:.4f} (pre-registered)")
        print(f"  R2 = {analysis['r2_control_law']:.4f}")
        print(f"  Residual: mean={analysis['residual_mean']:+.4f} std={analysis['residual_std']:.4f}")
        print(f"  PASS (R2 > 0.9): {analysis['PASS_r2']}")
        print()
        print("  Per-arm residuals:")
        for arm, resid in analysis['per_arm_residuals'].items():
            print(f"    {arm}: mean={resid['mean']:+.4f} std={resid['std']:.4f}")
        print(f"  Max arm mean diff: {analysis.get('max_arm_mean_diff'):.4f}")
        print(f"  PASS (invariant, diff < 0.1): {analysis['PASS_invariance']}")
        print()

    # Summary
    def mean_final(arm_key, metric):
        vals = [r[f'final_{metric}'] for r in all_results[arm_key]
                if r.get(f'final_{metric}') is not None]
        return float(np.mean(vals)) if vals else None

    print("SUMMARY:")
    for arm in ['ce', 'nc', 'anti_nc']:
        mq = mean_final(arm, 'q')
        mk = mean_final(arm, 'kappa')
        md = mean_final(arm, 'd_eff')
        mke = mean_final(arm, 'kappa_eff')
        print(f"  {arm:8s}: q={mq:.4f} kappa={mk:.4f} d_eff={md:.3f} kappa_eff={mke:.4f}")

    output = {
        'theorem': 'Control Law: Delta logit(q) = A_renorm(K) * Delta(kappa_eff)',
        'A_renorm': A_RENORM_K20,
        'K': K,
        'control_law_analysis': analysis,
        'results': all_results,
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nSaved to {RESULT_PATH}")


if __name__ == '__main__':
    main()
