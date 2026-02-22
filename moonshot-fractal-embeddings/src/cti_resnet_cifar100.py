"""
CTI Law test: ResNet50 on CIFAR-100.
CNN architecture (no attention, no recurrence), K=100 classes.
Tests whether the per-class law logit(q_norm) = alpha * kappa + C holds.
Compares alpha_cnn to alpha_vit (0.63) and alpha_nlp (1.477).
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import logit as scipy_logit
from scipy.stats import pearsonr, linregress
import json
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---- Extract ResNet50 features from multiple layers ----
LAYERS_TO_TEST = ['layer1', 'layer2', 'layer3', 'layer4']
N_PER_CLASS = 100  # 100 classes * 100 = 10,000 samples
K = 100

print("Loading ResNet50 (ImageNet pretrained)...")
model = models.resnet50(weights='IMAGENET1K_V1').to(device)
model.eval()

# Register hooks for all layers
layer_features = {}
hooks = []
for layer_name in LAYERS_TO_TEST:
    def make_hook(name):
        def hook(module, input, output):
            layer_features[name] = output.detach()
        return hook
    layer = getattr(model, layer_name)
    hooks.append(layer.register_forward_hook(make_hook(layer_name)))

# ---- Load CIFAR-100 ----
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_set = CIFAR100(root='./data', train=True,  download=False, transform=transform)
test_set  = CIFAR100(root='./data', train=False, download=False, transform=transform)

# Group by class (train for KNN, test for evaluation)
from collections import defaultdict
train_by_class = defaultdict(list)
test_by_class  = defaultdict(list)
N_TRAIN = 500   # 500 train per class
N_TEST  = 100   # 100 test per class

for img, label in train_set:
    if len(train_by_class[label]) < N_TRAIN:
        train_by_class[label].append(img)
for img, label in test_set:
    if len(test_by_class[label]) < N_TEST:
        test_by_class[label].append(img)

print(f"Train samples per class: {N_TRAIN}, Test: {N_TEST}")
N_PER_CLASS = N_TRAIN

def extract_split(by_class_dict, split_name):
    """Extract embeddings for all classes in a split."""
    embs_by_layer = {ln: [] for ln in LAYERS_TO_TEST}
    labels = []
    t0 = time.time()
    with torch.no_grad():
        for cls_idx in range(K):
            imgs = torch.stack(by_class_dict[cls_idx]).to(device)
            _ = model(imgs)
            for ln in LAYERS_TO_TEST:
                feat = layer_features[ln]
                if feat.dim() == 4:
                    feat = feat.mean(dim=[2, 3])
                embs_by_layer[ln].append(feat.cpu().float().numpy())
            labels.extend([cls_idx] * len(imgs))
            if (cls_idx + 1) % 25 == 0:
                print(f"  [{split_name}] Class {cls_idx+1}/100, {time.time()-t0:.1f}s")
    labels = np.array(labels)
    for ln in LAYERS_TO_TEST:
        embs_by_layer[ln] = np.vstack(embs_by_layer[ln])
    return embs_by_layer, labels

print("Extracting TRAIN embeddings (500/class * 100 classes = 50K)...")
train_embs, train_labels = extract_split(train_by_class, 'train')
print("Extracting TEST embeddings (100/class * 100 classes = 10K)...")
test_embs, test_labels = extract_split(test_by_class, 'test')

for ln in LAYERS_TO_TEST:
    print(f"  {ln}: train={train_embs[ln].shape}, test={test_embs[ln].shape}")

# Remove hooks
for h in hooks:
    h.remove()

# ---- CTI Law per-layer ----
results = []

for layer_name in LAYERS_TO_TEST:
    print(f"\nProcessing {layer_name}...")
    X_tr = train_embs[layer_name]  # [50K, d]
    X_te = test_embs[layer_name]   # [10K, d]
    y_tr = train_labels
    y_te = test_labels
    d = X_tr.shape[1]

    # Compute centroids and sigma_W from TRAIN set
    centroids = np.zeros((K, d))
    per_class_vars = []
    for c in range(K):
        mask = (y_tr == c)
        X_c = X_tr[mask]
        centroids[c] = X_c.mean(axis=0)
        per_class_vars.append(np.var(X_c, axis=0).mean())

    sigma_W_sq = np.mean(per_class_vars)
    sigma_W = np.sqrt(sigma_W_sq)
    print(f"  d={d}, sigma_W={sigma_W:.4f}")

    # kappa_nearest per class (from train centroids)
    kappas = []
    for c in range(K):
        dists = np.linalg.norm(centroids - centroids[c], axis=1)
        dists[c] = np.inf
        kappas.append(dists.min() / (sigma_W * np.sqrt(d)))
    kappas = np.array(kappas)
    print(f"  kappa range: [{kappas.min():.4f}, {kappas.max():.4f}], mean={kappas.mean():.4f}")

    # 1-NN accuracy per class: TRAIN -> TEST
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine', n_jobs=-1)
    knn.fit(X_tr, y_tr)
    preds = knn.predict(X_te)
    per_class_acc = np.array([(preds[y_te == c] == c).mean() for c in range(K)])
    print(f"  Per-class acc: mean={per_class_acc.mean():.4f}, range=[{per_class_acc.min():.4f}, {per_class_acc.max():.4f}]")

    # Normalize: q_norm = (q - 1/K) / (1 - 1/K)
    q_norm = (per_class_acc - 1/K) / (1 - 1/K)
    # Clip to avoid logit explosion
    q_norm = np.clip(q_norm, 0.01, 0.99)
    logit_q = scipy_logit(q_norm)

    # Fit: logit(q_norm) = alpha * kappa + C
    slope, intercept, r, p, se = linregress(kappas, logit_q)
    print(f"  Law fit: alpha={slope:.4f}, C={intercept:.4f}, r={r:.4f}, r^2={r**2:.4f}")

    # Pearson r
    r_val, p_val = pearsonr(kappas, logit_q)

    results.append({
        'layer': layer_name,
        'd': d,
        'K': K,
        'sigma_W': float(sigma_W),
        'kappa_mean': float(kappas.mean()),
        'kappa_min': float(kappas.min()),
        'kappa_max': float(kappas.max()),
        'per_class_acc_mean': float(per_class_acc.mean()),
        'alpha': float(slope),
        'intercept': float(intercept),
        'pearson_r': float(r_val),
        'pearson_r2': float(r_val**2),
        'p_value': float(p_val),
    })

# ---- Summary ----
print("\n\n=== CTI LAW: ResNet50 CIFAR-100 (CNN, K=100) ===")
print(f"{'Layer':<12} {'d':>6} {'alpha':>8} {'r':>8} {'r^2':>8} {'acc':>8}")
print("-" * 60)
for r in results:
    print(f"{r['layer']:<12} {r['d']:>6} {r['alpha']:>8.4f} {r['pearson_r']:>8.4f} {r['pearson_r2']:>8.4f} {r['per_class_acc_mean']:>8.4f}")

# Compare to known values
best = max(results, key=lambda x: x['pearson_r'])
print(f"\nBest layer: {best['layer']}, alpha={best['alpha']:.4f}, r={best['pearson_r']:.4f}")
print(f"Compare to: alpha_NLP=1.477, alpha_ViT=0.63")
print(f"Theory prediction: alpha = sqrt(4/pi) * sqrt(d_eff)")
print(f"For alpha_CNN={best['alpha']:.4f}: d_eff_implied = (alpha/sqrt(4/pi))^2 = {(best['alpha']/1.1284)**2:.4f}")

# Save
out = {
    'experiment': 'cti_resnet50_cifar100',
    'description': 'CTI Law per-class test: ResNet50 (CNN, no attention), CIFAR-100, K=100',
    'architecture': 'ResNet50',
    'family': 'CNN',
    'dataset': 'CIFAR-100',
    'K': K,
    'N_per_class': N_PER_CLASS,
    'reference_alpha_NLP': 1.477,
    'reference_alpha_ViT': 0.63,
    'layers': results
}
with open('results/cti_resnet50_cifar100.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/cti_resnet50_cifar100.json")
