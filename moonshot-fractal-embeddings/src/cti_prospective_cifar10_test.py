"""
PROSPECTIVE CIFAR-10 TEST: Universal Control Law on Different K

PRE-REGISTERED HYPOTHESIS (before running):
  For CE-trained ResNet-18 on CIFAR-10 (K=10):
    slope of logit(q) vs kappa_eff = A_renorm(K=10) = 1.0503 +/- 0.15

This tests K-UNIVERSALITY of the control law.
- A_renorm(K) from Theorem 15 (ZERO FREE PARAMETERS)
- K=10 uses CIFAR-10 (10 classes, different from CIFAR-100 coarse K=20)
- If slope matches, the K-dependent form is confirmed experimentally

THEOREM 15 PREDICTION:
  A_renorm(K) = alpha/sqrt(d_eff) depends ONLY on K.
  For K=10: A_renorm = 1.0503 (from K_specific_constants in cti_theorem15_K_corrected.json)

COMPARISON:
  K=20 (CIFAR-100 coarse, ResNet-18):  A_renorm = 1.0535 [control law validation]
  K=10 (CIFAR-10, ResNet-18):          A_renorm = 1.0503 [THIS TEST]
  Ratio: 1.0503/1.0535 = 0.997 (near-identical, K has very weak effect in [10,20])
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

# Pre-registered from Theorem 15 K_specific_constants
K = 10
A_RENORM_K10 = 1.0503  # from cti_theorem15_K_corrected.json
A_TOLERANCE = 0.15

N_EPOCHS = 60
CHECKPOINT_EPOCHS = [25, 40, 60]
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 5e-4
N_SEEDS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "results/cti_prospective_cifar10_test.json"


def get_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    ce_head = nn.Linear(512, K)
    model = nn.ModuleDict({'backbone': backbone, 'ce_head': ce_head})
    return model.to(DEVICE)


def get_cifar10():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        'data', train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(
        'data', train=False, download=True, transform=test_transform)
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


def compute_d_eff(X, y):
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
    return d_eff


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


def compute_q(X_test, y_test, random_state=42):
    sss = StratifiedShuffleSplit(1, test_size=0.3, random_state=random_state)
    tr, te = next(sss.split(X_test, y_test))
    knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
    knn.fit(X_test[tr], y_test[tr])
    acc = float(knn.score(X_test[te], y_test[te]))
    return (acc - 1.0 / K) / (1.0 - 1.0 / K)


def fit_control_law(checkpoints):
    rows = [(ck['kappa_eff'], ck['logit_q'])
            for ck in checkpoints
            if 0 < ck.get('q', 0) < 1 and ck.get('kappa_eff', 0) > 0]
    if len(rows) < 2:
        return None, None, None
    kappa_effs = np.array([r[0] for r in rows])
    logit_qs = np.array([r[1] for r in rows])
    X_fit = np.column_stack([kappa_effs, np.ones(len(kappa_effs))])
    coeffs, _, _, _ = np.linalg.lstsq(X_fit, logit_qs, rcond=None)
    slope = float(coeffs[0])
    pred = coeffs[0] * kappa_effs + coeffs[1]
    ss_res = np.sum((logit_qs - pred) ** 2)
    ss_tot = np.sum((logit_qs - logit_qs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, float(coeffs[1]), float(r2)


def train_seed(seed, train_ds, test_ds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = get_model()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

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
            X_test, y_test = extract_all_embeddings(model, test_ds)
            q_val = compute_q(X_test, y_test)
            kappa_val = compute_kappa_nearest(X_test, y_test)

            print(f"  [seed={seed} epoch={epoch}] q={q_val:.4f} kappa={kappa_val:.4f} "
                  f"computing d_eff...", end=' ', flush=True)
            X_train, y_train = extract_all_embeddings(model, train_ds)
            d_eff = compute_d_eff(X_train, y_train)
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
            })

    slope, intercept, r2 = fit_control_law(checkpoints)
    print(f"  DONE: q={checkpoints[-1]['q']:.4f} slope={slope:.4f} "
          f"(target={A_RENORM_K10}) R2={r2:.4f}", flush=True)

    return {
        'seed': seed,
        'checkpoints': checkpoints,
        'final_q': checkpoints[-1]['q'] if checkpoints else None,
        'slope_control_law': slope,
        'intercept': intercept,
        'r2_control_law': r2,
        'slope_pass': bool(abs(slope - A_RENORM_K10) < A_TOLERANCE) if slope else False,
    }


def main():
    print("Prospective CIFAR-10 Test: K=10, A_renorm = 1.0503")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Dataset: CIFAR-10 (K={K})")
    print(f"Pre-registered A_renorm(K={K}) = {A_RENORM_K10} (from Theorem 15)")
    print(f"Pre-registered tolerance = +-{A_TOLERANCE}")
    print()

    train_ds, test_ds = get_cifar10()
    results = []

    for seed in range(N_SEEDS):
        print(f"\n--- seed={seed} ---")
        res = train_seed(seed, train_ds, test_ds)
        results.append(res)
        with open(RESULT_PATH, 'w') as f:
            json.dump({'status': 'running', 'results': results}, f,
                      default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    slopes = [r['slope_control_law'] for r in results if r['slope_control_law']]
    r2s = [r['r2_control_law'] for r in results if r['r2_control_law']]
    passes = [r['slope_pass'] for r in results]

    print("\n" + "=" * 70)
    print(f"SUMMARY: CIFAR-10 PROSPECTIVE TEST (K={K})")
    print("=" * 70)
    print(f"  Pre-registered A_renorm(K={K}) = {A_RENORM_K10}")
    for res in results:
        print(f"  seed={res['seed']}: slope={res['slope_control_law']:.4f} "
              f"R2={res['r2_control_law']:.4f} PASS={res['slope_pass']}")
    if slopes:
        mean_slope = np.mean(slopes)
        print(f"\n  Mean slope = {mean_slope:.4f} (target = {A_RENORM_K10})")
        print(f"  Relative error = {abs(mean_slope - A_RENORM_K10) / A_RENORM_K10:.3f}")
        print(f"  Seeds passing: {sum(passes)}/{len(passes)}")
        p_overall = bool(sum(passes) >= 2 and abs(mean_slope - A_RENORM_K10) < A_TOLERANCE)
        print(f"  OVERALL PASS: {p_overall}")

    output = {
        'status': 'complete',
        'K': K,
        'A_renorm_preregistered': A_RENORM_K10,
        'tolerance': A_TOLERANCE,
        'results': results,
        'summary': {
            'mean_slope': float(np.mean(slopes)) if slopes else None,
            'std_slope': float(np.std(slopes)) if slopes else None,
            'mean_r2': float(np.mean(r2s)) if r2s else None,
            'seeds_pass': int(sum(passes)),
            'OVERALL_PASS': bool(sum(passes) >= 2 and slopes and
                                  abs(np.mean(slopes) - A_RENORM_K10) < A_TOLERANCE),
        }
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nSaved to {RESULT_PATH}")


if __name__ == '__main__':
    main()
