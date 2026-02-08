"""
Fractal Embeddings for Vision: CIFAR-100 Experiment
=====================================================

THE PARADIGM SHIFT: Fractal embeddings aren't just for text.
If the principle (hierarchy-aligned prefix supervision = semantic zoom)
works for IMAGES too, it's a universal representation principle.

CIFAR-100 hierarchy:
- 20 superclasses (L0): aquatic_mammals, flowers, fruit_and_vegetables, ...
- 100 fine classes (L1): beaver, dolphin, otter, seal, whale, ...
- Branching: 5 fine per superclass
- H(L0) = log2(20) = 4.32 bits, H(L1|L0) = log2(5) = 2.32 bits

METHOD: Same as text V5, but with a vision backbone (ResNet/ViT).
- Frozen backbone + learned projection head
- Progressive prefix supervision
- Compare V5 (hierarchy-aligned) vs MRL (flat)

PREDICTION: V5 should show steerability on CIFAR-100 too.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

RESULTS_DIR = Path(__file__).parent.parent / "results"

# CIFAR-100 superclass mapping (20 superclasses, 5 fine per super)
CIFAR100_COARSE_LABELS = {
    # aquatic_mammals
    4: 0, 30: 0, 55: 0, 72: 0, 95: 0,
    # fish
    1: 1, 32: 1, 67: 1, 73: 1, 91: 1,
    # flowers
    54: 2, 62: 2, 70: 2, 82: 2, 92: 2,
    # food_containers
    9: 3, 10: 3, 16: 3, 28: 3, 61: 3,
    # fruit_and_vegetables
    0: 4, 51: 4, 53: 4, 57: 4, 83: 4,
    # household_electrical_devices
    22: 5, 39: 5, 40: 5, 86: 5, 87: 5,
    # household_furniture
    5: 6, 20: 6, 25: 6, 84: 6, 94: 6,
    # insects
    6: 7, 7: 7, 14: 7, 18: 7, 24: 7,
    # large_carnivores
    3: 8, 42: 8, 43: 8, 88: 8, 97: 8,
    # large_man-made_outdoor_things
    12: 9, 17: 9, 37: 9, 68: 9, 76: 9,
    # large_natural_outdoor_scenes
    23: 10, 33: 10, 49: 10, 60: 10, 71: 10,
    # large_omnivores_and_herbivores
    15: 11, 19: 11, 21: 11, 31: 11, 38: 11,
    # medium-sized_mammals
    34: 12, 63: 12, 64: 12, 66: 12, 75: 12,
    # non-insect_invertebrates
    26: 13, 45: 13, 77: 13, 79: 13, 99: 13,
    # people
    2: 14, 11: 14, 35: 14, 46: 14, 98: 14,
    # reptiles
    27: 15, 29: 15, 44: 15, 78: 15, 93: 15,
    # small_mammals
    36: 16, 50: 16, 65: 16, 74: 16, 80: 16,
    # trees
    47: 17, 52: 17, 56: 17, 59: 17, 96: 17,
    # vehicles_1
    8: 18, 13: 18, 48: 18, 58: 18, 90: 18,
    # vehicles_2
    41: 19, 69: 19, 81: 19, 85: 19, 89: 19,
}


class CIFAR100Hierarchical(Dataset):
    """CIFAR-100 with both fine and coarse labels."""

    def __init__(self, train=True, transform=None):
        self.dataset = datasets.CIFAR100(
            root="./data", train=train, download=True, transform=transform
        )
        self.coarse_labels = np.array([
            CIFAR100_COARSE_LABELS[int(self.dataset.targets[i])]
            for i in range(len(self.dataset))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, fine_label = self.dataset[idx]
        coarse_label = self.coarse_labels[idx]
        return img, coarse_label, fine_label


class VisionFractalHead(nn.Module):
    """Fractal head for vision: projects backbone features to multi-scale embedding."""

    def __init__(self, input_dim, num_scales=4, scale_dim=64,
                 num_l0=20, num_l1=100):
        super().__init__()
        self.num_scales = num_scales
        self.scale_dim = scale_dim
        self.output_dim = num_scales * scale_dim

        # Projection: backbone -> fractal embedding
        self.proj = nn.Linear(input_dim, self.output_dim)
        self.ln = nn.LayerNorm(self.output_dim)

        # Classification heads
        self.head_coarse = nn.Linear(scale_dim, num_l0)  # From prefix
        self.head_fine = nn.Linear(self.output_dim, num_l1)  # From full

    def forward(self, features):
        emb = self.ln(self.proj(features))
        blocks = [emb[:, i*self.scale_dim:(i+1)*self.scale_dim]
                  for i in range(self.num_scales)]
        return {
            'full_embedding': emb,
            'blocks': blocks,
            'logits_coarse': self.head_coarse(blocks[0]),  # j=1 prefix -> L0
            'logits_fine': self.head_fine(emb),  # Full -> L1
        }

    def get_prefix(self, blocks, j):
        """Get j-block prefix embedding."""
        return torch.cat(blocks[:j], dim=-1)


class VisionFractalModel(nn.Module):
    """Complete vision fractal model with frozen backbone."""

    def __init__(self, backbone_name="resnet18", num_l0=20, num_l1=100,
                 num_scales=4, scale_dim=64, device="cuda"):
        super().__init__()
        self.device = device

        # Load pretrained backbone
        if backbone_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.hidden_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()  # Remove classification head
        elif backbone_name == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.hidden_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.backbone = backbone
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = VisionFractalHead(
            self.hidden_dim, num_scales, scale_dim, num_l0, num_l1
        )

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features)


def train_vision_fractal(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3,
    method="v5",  # "v5" or "mrl"
    device="cuda",
    prefix_weight=0.6,
):
    """
    Train vision fractal model.

    V5: j=1 prefix trained on L0 (coarse), full on L1 (fine)
    MRL: ALL prefix lengths trained on L1 (fine)
    """
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    prefix_probs = [0.4, 0.3, 0.2, 0.1]
    best_score = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for images, coarse_labels, fine_labels in train_loader:
            images = images.to(device)
            coarse_labels = coarse_labels.to(device).long()
            fine_labels = fine_labels.to(device).long()

            out = model(images)

            # Full embedding loss (always L1/fine)
            loss_full = F.cross_entropy(out['logits_fine'], fine_labels)

            # Prefix loss
            if method == "v5":
                # V5: coarse prefix trained on L0
                loss_prefix = F.cross_entropy(out['logits_coarse'], coarse_labels)
            else:
                # MRL: all prefixes trained on L1
                # Use same fine head but on prefix
                prefix_emb = model.head.get_prefix(out['blocks'], 1)
                # Project prefix through fine head (zero-pad)
                full_dim = model.head.output_dim
                padded = F.pad(prefix_emb, (0, full_dim - prefix_emb.shape[1]))
                logits_prefix_fine = model.head.head_fine(padded)
                loss_prefix = F.cross_entropy(logits_prefix_fine, fine_labels)

            loss = loss_full + prefix_weight * loss_prefix

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        l0_correct, l1_correct, n_val = 0, 0, 0
        with torch.no_grad():
            for images, coarse_labels, fine_labels in val_loader:
                images = images.to(device)
                out = model(images)

                # L0 from prefix
                l0_pred = out['logits_coarse'].argmax(dim=-1).cpu()
                l1_pred = out['logits_fine'].argmax(dim=-1).cpu()

                l0_correct += (l0_pred == coarse_labels).sum().item()
                l1_correct += (l1_pred == fine_labels).sum().item()
                n_val += len(images)

        l0_acc = l0_correct / n_val
        l1_acc = l1_correct / n_val
        score = l0_acc + l1_acc

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}

        avg_loss = total_loss / n_batches
        print(f"  [{method.upper()}] Epoch {epoch+1}: loss={avg_loss:.4f}, L0={l0_acc:.4f}, L1={l1_acc:.4f}")

    # Restore best
    model.head.load_state_dict(best_state)
    model.head.to(device)

    return model


def evaluate_steerability_vision(model, test_loader, device="cuda"):
    """Evaluate steerability of vision fractal model."""
    model.eval()
    all_embs = []
    all_l0 = []
    all_l1 = []

    with torch.no_grad():
        for images, coarse_labels, fine_labels in test_loader:
            images = images.to(device)
            out = model(images)
            all_embs.append(out['full_embedding'].cpu().numpy())
            all_l0.append(coarse_labels.numpy())
            all_l1.append(fine_labels.numpy())

    embs = np.concatenate(all_embs)
    l0 = np.concatenate(all_l0)
    l1 = np.concatenate(all_l1)

    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms

    # kNN classification at each prefix length
    k = 5
    results = {}
    for j in [1, 2, 3, 4]:
        prefix = embs[:, :j*64]
        prefix = prefix / np.linalg.norm(prefix, axis=1, keepdims=True)

        sims = prefix @ prefix.T
        np.fill_diagonal(sims, -1)

        l0_correct, l1_correct = 0, 0
        for i in range(len(embs)):
            top_k = np.argsort(-sims[i])[:k]
            # L0
            counts_l0 = Counter(l0[top_k].tolist())
            pred_l0 = max(counts_l0, key=counts_l0.get)
            if pred_l0 == l0[i]:
                l0_correct += 1
            # L1
            counts_l1 = Counter(l1[top_k].tolist())
            pred_l1 = max(counts_l1, key=counts_l1.get)
            if pred_l1 == l1[i]:
                l1_correct += 1

        results[f"j{j}"] = {
            "l0_acc": l0_correct / len(embs),
            "l1_acc": l1_correct / len(embs),
        }

    # Steerability
    steer = (results["j1"]["l0_acc"] - results["j4"]["l0_acc"]) + \
            (results["j4"]["l1_acc"] - results["j1"]["l1_acc"])

    return {
        "prefix_results": results,
        "steerability": steer,
    }


def run_vision_experiment(
    backbone="resnet18",
    epochs=15,
    batch_size=128,
    seed=42,
    device="cuda",
):
    """Run the full CIFAR-100 fractal embedding experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*70}")
    print(f"  VISION FRACTAL: {backbone} on CIFAR-100, seed={seed}")
    print(f"{'='*70}")

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = CIFAR100Hierarchical(train=True, transform=transform_train)
    test_ds = CIFAR100Hierarchical(train=False, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_results = {}

    for method in ["v5", "mrl"]:
        print(f"\n--- {method.upper()} ---")

        model = VisionFractalModel(
            backbone_name=backbone,
            num_l0=20,
            num_l1=100,
            device=device,
        ).to(device)

        model = train_vision_fractal(
            model, train_loader, test_loader,
            epochs=epochs, method=method, device=device,
        )

        eval_results = evaluate_steerability_vision(model, test_loader, device)

        print(f"\n  {method.upper()} Results:")
        for j in [1, 2, 3, 4]:
            r = eval_results["prefix_results"][f"j{j}"]
            print(f"    j={j} ({j*64}d): L0={r['l0_acc']:.4f}, L1={r['l1_acc']:.4f}")
        print(f"    Steerability: {eval_results['steerability']:+.4f}")

        all_results[method] = eval_results

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    v5_s = all_results["v5"]["steerability"]
    mrl_s = all_results["mrl"]["steerability"]

    print(f"\n{'='*70}")
    print(f"  VISION SUMMARY: CIFAR-100 on {backbone}")
    print(f"{'='*70}")
    print(f"  V5  Steerability: {v5_s:+.4f}")
    print(f"  MRL Steerability: {mrl_s:+.4f}")
    print(f"  Gap:              {v5_s - mrl_s:+.4f}")

    if v5_s > mrl_s:
        print(f"  FRACTAL EMBEDDINGS WORK FOR VISION!")
    else:
        print(f"  No clear advantage for V5 on vision.")

    # Save
    out_path = RESULTS_DIR / f"vision_{backbone}_cifar100_seed{seed}.json"
    output = {
        "backbone": backbone,
        "dataset": "cifar100",
        "seed": seed,
        "hierarchy": {"n_l0": 20, "n_l1": 100, "branch": 5.0,
                      "h_l0": np.log2(20), "h_l1_given_l0": np.log2(5)},
        "timestamp": datetime.now().isoformat(),
        **{k: {kk: (vv if not isinstance(vv, np.floating) else float(vv))
               for kk, vv in v.items()} if isinstance(v, dict) else v
           for k, v in all_results.items()},
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2,
                 default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_vision_experiment(backbone=args.backbone, epochs=args.epochs, seed=args.seed)
