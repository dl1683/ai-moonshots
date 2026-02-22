"""
CTI Law: ViT-Base on CIFAR-100 (K=100).
Purpose: isolate whether r=0.75 seen with ResNet50 is due to K=100 or CNN architecture.
If ViT-Base CIFAR-100 r >> 0.75, then it's architecture-specific (attention vs conv).
If ViT-Base CIFAR-100 r ~ 0.75, then it's K=100 regime.

Uses train/test split: train (500/class) for centroid+KNN, test (100/class) for q estimation.
"""
import torch
import numpy as np
from transformers import ViTModel, ViTImageProcessor
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import logit
from scipy.stats import pearsonr, linregress
from PIL import Image
import json
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

MODEL_NAME = "google/vit-base-patch16-224"
K = 100
N_TRAIN = 500
N_TEST  = 100
LAYERS_TO_TEST = [4, 8, 12]  # Layer indices out of 12

print(f"Loading {MODEL_NAME}...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# --- Load CIFAR-100 ---
print("Loading CIFAR-100 (train + test)...")
train_ds = CIFAR100(root='./data', train=True,  download=False)
test_ds  = CIFAR100(root='./data', train=False, download=False)

from collections import defaultdict
train_by_class = defaultdict(list)
test_by_class  = defaultdict(list)

for img, label in train_ds:
    if len(train_by_class[label]) < N_TRAIN:
        train_by_class[label].append(img)  # PIL image
for img, label in test_ds:
    if len(test_by_class[label]) < N_TEST:
        test_by_class[label].append(img)

print(f"Train: {sum(len(v) for v in train_by_class.values())} samples")
print(f"Test:  {sum(len(v) for v in test_by_class.values())} samples")

# --- Extract embeddings ---
def extract_vit_embeddings(by_class_dict, split_name, batch_size=50):
    """Extract CLS token embeddings from all specified layers."""
    all_embs = {l: [] for l in LAYERS_TO_TEST}
    all_labels = []
    t0 = time.time()

    for cls_idx in range(K):
        imgs = by_class_dict[cls_idx]  # list of PIL images
        # Process in batches
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size]
            # Convert PIL images to processor format
            inputs = processor(images=batch, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            # Extract CLS token from each requested layer
            for layer_idx in LAYERS_TO_TEST:
                cls_emb = outputs.hidden_states[layer_idx][:, 0, :].cpu().float().numpy()
                all_embs[layer_idx].append(cls_emb)
            all_labels.extend([cls_idx] * len(batch))
        if (cls_idx + 1) % 25 == 0:
            print(f"  [{split_name}] Class {cls_idx+1}/100, {time.time()-t0:.1f}s")

    all_labels = np.array(all_labels)
    for l in LAYERS_TO_TEST:
        all_embs[l] = np.vstack(all_embs[l])
    return all_embs, all_labels

print(f"\nExtracting TRAIN embeddings (ViT, {N_TRAIN}/class)...")
train_embs, train_labels = extract_vit_embeddings(train_by_class, 'train')
print(f"\nExtracting TEST embeddings (ViT, {N_TEST}/class)...")
test_embs, test_labels = extract_vit_embeddings(test_by_class, 'test')

for l in LAYERS_TO_TEST:
    print(f"  Layer {l:2d}: train={train_embs[l].shape}, test={test_embs[l].shape}")

# --- CTI Law per layer ---
print("\n=== CTI Law: ViT-Base CIFAR-100 ===")
results = []

for layer_idx in LAYERS_TO_TEST:
    X_tr = train_embs[layer_idx]
    X_te = test_embs[layer_idx]
    y_tr = train_labels
    y_te = test_labels
    d = X_tr.shape[1]

    # Centroids and sigma_W from train
    centroids = np.zeros((K, d))
    per_class_vars = []
    for c in range(K):
        mask = (y_tr == c)
        X_c = X_tr[mask]
        centroids[c] = X_c.mean(axis=0)
        per_class_vars.append(np.var(X_c, axis=0).mean())

    sigma_W = np.sqrt(np.mean(per_class_vars))

    # kappa_nearest per class
    kappas = []
    for c in range(K):
        dists = np.linalg.norm(centroids - centroids[c], axis=1)
        dists[c] = np.inf
        kappas.append(dists.min() / (sigma_W * np.sqrt(d)))
    kappas = np.array(kappas)

    # 1-NN accuracy: train -> test
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine', n_jobs=-1)
    knn.fit(X_tr, y_tr)
    preds = knn.predict(X_te)
    per_class_acc = np.array([(preds[y_te == c] == c).mean() for c in range(K)])

    # Normalize and fit
    q_norm = (per_class_acc - 1/K) / (1 - 1/K)
    q_norm = np.clip(q_norm, 0.01, 0.99)
    logit_q = logit(q_norm)

    slope, intercept, r, p, se = linregress(kappas, logit_q)
    r_pearson, p_pearson = pearsonr(kappas, logit_q)

    print(f"  Layer {layer_idx:2d}: d={d}, sigma_W={sigma_W:.4f}, "
          f"kappa=[{kappas.min():.3f},{kappas.max():.3f}], "
          f"acc_mean={per_class_acc.mean():.3f}, "
          f"alpha={slope:.4f}, r={r_pearson:.4f}, r^2={r_pearson**2:.4f}")

    results.append({
        'layer': layer_idx,
        'd': d,
        'K': K,
        'sigma_W': float(sigma_W),
        'kappa_mean': float(kappas.mean()),
        'kappa_min': float(kappas.min()),
        'kappa_max': float(kappas.max()),
        'per_class_acc_mean': float(per_class_acc.mean()),
        'alpha': float(slope),
        'intercept': float(intercept),
        'pearson_r': float(r_pearson),
        'pearson_r2': float(r_pearson**2),
    })

best = max(results, key=lambda x: x['pearson_r'])
print(f"\n=== SUMMARY ===")
print(f"ViT-Base CIFAR-100 (K=100): best r = {best['pearson_r']:.4f} at layer {best['layer']}")
print(f"ResNet50 CIFAR-100 (K=100): best r = 0.7492 at layer3")
print(f"ViT-Base CIFAR-10  (K=10):  best r ~0.976 (from r^2=0.96)")
print(f"Noise-floor sim    (K=100): E[r] = 0.875")
print(f"alpha_ViT_CIFAR100 = {best['alpha']:.4f}, alpha_NLP = 1.477, alpha_ViT_CIFAR10 = 0.63")

# Save
out = {
    'experiment': 'cti_vit_cifar100',
    'description': 'CTI Law: ViT-Base on CIFAR-100 (K=100). Tests architecture vs K=100 effect.',
    'model': MODEL_NAME,
    'dataset': 'CIFAR-100',
    'K': K,
    'N_train': N_TRAIN,
    'N_test': N_TEST,
    'comparison': {
        'ResNet50_CIFAR100_r': 0.7492,
        'ViT_CIFAR10_r2': 0.96,
        'sim_noise_floor_K100': 0.875,
        'NLP_alpha': 1.477,
        'ViT_CIFAR10_alpha': 0.63,
    },
    'layers': results
}
with open('results/cti_vit_cifar100.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved to results/cti_vit_cifar100.json")
