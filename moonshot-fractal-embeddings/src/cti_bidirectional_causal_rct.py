"""
Bidirectional Causal RCT: kappa_nearest -> q mechanism.

THREE ARMS:
1. CE baseline:   L = L_CE
2. NC-positive:   L = L_CE + lambda * (L_within + 0.5*L_ETF + 0.5*L_margin)
   => increases kappa_nearest (more compact, better separated) => q should increase
3. NC-negative:   L = L_CE - lambda * L_within
   => decreases kappa_nearest (more dispersed within-class) => q should decrease

PRE-REGISTERED PREDICTIONS:
- P_pos: kappa_NC+ > kappa_CE AND q_NC+ > q_CE    [NC-positive works]
- P_neg: kappa_NC- < kappa_CE AND q_NC- < q_CE    [NC-negative works]
- P_bidir: both P_pos AND P_neg hold simultaneously [bidirectional causal test]
- P_alpha: (delta_q+/delta_kappa+) ~= (delta_q-/delta_kappa-) ~= alpha ~= 1.365
   [same causal RATE in both directions = causal law confirmed]

WHY THIS IS A STRONG CAUSAL TEST:
- NC-loss could succeed due to regularization, better optimization, etc. (confounders)
- Anti-NC should FAIL if those confounders drive results
- But if kappa is THE causal variable: NC+ raises kappa+q, NC- lowers kappa+q
- The rate alpha should MATCH in both directions (same underlying law)
- This falsifies the "kappa is just a proxy" hypothesis
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

# Hyperparameters -- matched to quick pilot for comparability
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
CHECKPOINT_EPOCHS = [0, 25, 40, 60]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "results/cti_bidirectional_causal_rct.json"

# Pre-registered alpha from CE training trajectory (Session 11)
ALPHA_PREREGISTERED = 1.365


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
        transforms.ToTensor(), transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100(
        'data', train=True, download=False,
        transform=train_transform, target_transform=coarse_label)
    test_ds = torchvision.datasets.CIFAR100(
        'data', train=False, download=False,
        transform=test_transform, target_transform=coarse_label)
    return train_ds, test_ds


def compute_nc_positive_loss(z, y, class_means, K):
    """NC-positive: within-compact + ETF + margin. Maximizes kappa_nearest."""
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
    return L_within + 0.5 * L_ETF + 0.5 * L_margin, L_within.item()


def compute_nc_negative_loss(z, y, class_means):
    """NC-negative: -L_within only. Disperses within-class, decreases kappa_nearest."""
    mu_yi = class_means[y]
    L_within = ((z - mu_yi) ** 2).mean()
    return -L_within, L_within.item()  # NEGATIVE: maximize within-class spread


def update_ema_means(class_means, z, y, K):
    with torch.no_grad():
        for c in range(K):
            mask = (y == c)
            if mask.sum() > 0:
                class_means[c] = (EMA_MOMENTUM * class_means[c]
                                  + (1 - EMA_MOMENTUM) * z[mask].mean(0))
                class_means[c] = F.normalize(class_means[c].unsqueeze(0), dim=1).squeeze(0)
    return class_means


def compute_kappa_nearest(X, y, K=K):
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


def compute_q_and_kappa(model, test_ds, K=K):
    model.eval()
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=0)
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            embs.append(model['backbone'](imgs.to(DEVICE)).cpu().numpy())
            labels.append(lbs.numpy())
    X, y = np.concatenate(embs), np.concatenate(labels)
    sss = StratifiedShuffleSplit(1, test_size=0.3, random_state=42)
    tr, te = next(sss.split(X, y))
    knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
    knn.fit(X[tr], y[tr])
    acc = float(knn.score(X[te], y[te]))
    q = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    kappa = compute_kappa_nearest(X, y, K=K)
    return float(q), float(kappa)


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
        n_b = 0
        e_ce = 0.0
        e_aux = 0.0
        e_within = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            h = model['backbone'](imgs)
            logits = model['ce_head'](h)
            loss_ce = ce_loss_fn(logits, labels)

            if lam > 0:
                z_raw = model['proj_head'](h)
                z = F.normalize(z_raw, dim=1)
                class_means = update_ema_means(class_means, z.detach(), labels, K)

                if arm == 'nc':
                    loss_aux, within_val = compute_nc_positive_loss(
                        z, labels, class_means, K)
                    loss = loss_ce + lam * loss_aux
                elif arm == 'anti_nc':
                    loss_aux, within_val = compute_nc_negative_loss(
                        z, labels, class_means)
                    loss = loss_ce + lam * loss_aux  # note: loss_aux is already negative
                else:
                    loss = loss_ce
                    loss_aux = torch.tensor(0.0)
                    within_val = 0.0

                e_aux += loss_aux.item() if hasattr(loss_aux, 'item') else float(loss_aux)
                e_within += within_val
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            e_ce += loss_ce.item()
            n_b += 1

        scheduler.step()

        if epoch in CHECKPOINT_EPOCHS:
            q_val, kappa_val = compute_q_and_kappa(model, test_ds)
            checkpoints.append({
                'epoch': epoch, 'q': q_val, 'kappa': kappa_val,
                'loss_ce': e_ce / n_b, 'loss_aux': e_aux / n_b,
                'within': e_within / n_b, 'lambda': lam,
            })
            print(f"  [seed={seed} arm={arm} epoch={epoch}] "
                  f"q={q_val:.4f} kappa={kappa_val:.4f} lam={lam:.3f}")
            sys.stdout.flush()

    final = checkpoints[-1] if checkpoints else {}
    return {
        'seed': seed, 'arm': arm,
        'final_q': final.get('q'),
        'final_kappa': final.get('kappa'),
        'checkpoints': checkpoints,
    }


def fit_alpha_from_trajectory(checkpoints):
    """Fit slope alpha from within-seed epoch trajectory."""
    rows = [(ck['kappa'], np.log(ck['q'] / (1 - ck['q'])))
            for ck in checkpoints if 0 < ck.get('q', 0) < 1 and ck.get('kappa', 0) > 0]
    if len(rows) < 3:
        return None
    kappas = np.array([r[0] for r in rows])
    logits = np.array([r[1] for r in rows])
    X = np.column_stack([kappas, np.ones(len(kappas))])
    coeffs, _, _, _ = np.linalg.lstsq(X, logits, rcond=None)
    return float(coeffs[0])


def main():
    print("Bidirectional Causal RCT: kappa_nearest -> q")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Arms: CE, NC+ (maximize kappa), NC- (minimize kappa)")
    print(f"Pre-registered alpha: {ALPHA_PREREGISTERED}")
    print()

    train_ds, test_ds = get_cifar_coarse()
    all_results = {'ce': [], 'nc': [], 'anti_nc': []}

    for arm in ['ce', 'nc', 'anti_nc']:
        print(f"\n=== ARM: {arm} ===")
        for seed in range(N_SEEDS):
            print(f"\n--- seed={seed} ---")
            res = train_one_arm(seed, arm, train_ds, test_ds)
            print(f"  DONE: final_q={res['final_q']:.4f} final_kappa={res['final_kappa']:.4f}")
            all_results[arm].append(res)
            # Save incrementally
            with open(RESULT_PATH, 'w') as f:
                json.dump({'status': 'running', 'results': all_results}, f,
                          default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    # ===== Analysis =====
    def mean_final(arm_key, metric):
        return float(np.mean([r['final_' + metric] for r in all_results[arm_key]]))

    ce_q = mean_final('ce', 'q')
    nc_q = mean_final('nc', 'q')
    anti_q = mean_final('anti_nc', 'q')
    ce_k = mean_final('ce', 'kappa')
    nc_k = mean_final('nc', 'kappa')
    anti_k = mean_final('anti_nc', 'kappa')

    delta_q_pos = nc_q - ce_q
    delta_q_neg = anti_q - ce_q
    delta_k_pos = nc_k - ce_k
    delta_k_neg = anti_k - ce_k

    # P_alpha: rate consistency test
    rate_pos = delta_q_pos / (delta_k_pos + 1e-8) if abs(delta_k_pos) > 0.01 else None
    rate_neg = delta_q_neg / (delta_k_neg + 1e-8) if abs(delta_k_neg) > 0.01 else None

    # Per-seed alpha from trajectory
    alphas = {}
    for arm_key in ['ce', 'nc', 'anti_nc']:
        arm_alphas = []
        for r in all_results[arm_key]:
            a = fit_alpha_from_trajectory(r['checkpoints'])
            if a is not None:
                arm_alphas.append(a)
        alphas[arm_key] = arm_alphas

    # Pre-registered tests
    p_pos = bool(delta_q_pos > 0 and delta_k_pos > 0)
    p_neg = bool(delta_q_neg < 0 and delta_k_neg < 0)
    p_bidir = bool(p_pos and p_neg)
    p_alpha = None
    if rate_pos is not None and rate_neg is not None:
        # Rate should match (sign-adjusted)
        # delta_q+ / delta_k+ should ~ alpha ~ delta_q- / delta_k-
        rate_ratio = abs(rate_pos / (rate_neg + 1e-8)) if abs(rate_neg) > 1e-6 else None
        p_alpha = bool(rate_ratio is not None and 0.5 < rate_ratio < 2.0)

    summary = {
        'status': 'complete',
        'ce': {'mean_q': ce_q, 'mean_kappa': ce_k},
        'nc': {'mean_q': nc_q, 'mean_kappa': nc_k,
               'delta_q': delta_q_pos, 'delta_kappa': delta_k_pos,
               'rate': rate_pos},
        'anti_nc': {'mean_q': anti_q, 'mean_kappa': anti_k,
                    'delta_q': delta_q_neg, 'delta_kappa': delta_k_neg,
                    'rate': rate_neg},
        'alphas': {k: {'mean': float(np.mean(v)) if v else None,
                       'std': float(np.std(v)) if v else None}
                   for k, v in alphas.items()},
        'tests': {
            'P_pos': p_pos,
            'P_neg': p_neg,
            'P_bidir': p_bidir,
            'P_alpha': p_alpha,
            'rate_ratio': float(rate_ratio) if rate_ratio is not None else None,
        },
        'preregistered_alpha': ALPHA_PREREGISTERED,
        'results': all_results,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    # Print summary
    print("\n" + "=" * 60)
    print("BIDIRECTIONAL CAUSAL RCT RESULTS")
    print("=" * 60)
    print(f"CE:       q={ce_q:.4f}  kappa={ce_k:.4f}")
    print(f"NC+:      q={nc_q:.4f}  kappa={nc_k:.4f}  "
          f"delta_q={delta_q_pos:+.4f}  delta_k={delta_k_pos:+.4f}  rate={rate_pos:.3f if rate_pos else 'N/A'}")
    print(f"NC-:      q={anti_q:.4f}  kappa={anti_k:.4f}  "
          f"delta_q={delta_q_neg:+.4f}  delta_k={delta_k_neg:+.4f}  rate={rate_neg:.3f if rate_neg else 'N/A'}")
    print()
    print(f"P_pos  (NC+ raises both kappa and q): {p_pos}")
    print(f"P_neg  (NC- lowers both kappa and q): {p_neg}")
    print(f"P_bidir (both directions work):        {p_bidir}")
    print(f"P_alpha (rate consistent, ratio~1.0):  {p_alpha}")
    print()
    for arm_key, arm_alphas in alphas.items():
        if arm_alphas:
            print(f"Alpha ({arm_key}): mean={np.mean(arm_alphas):.3f} std={np.std(arm_alphas):.3f}")
    print()
    print(f"Saved to {RESULT_PATH}")


if __name__ == '__main__':
    main()
