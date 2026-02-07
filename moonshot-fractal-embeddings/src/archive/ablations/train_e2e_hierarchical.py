"""
End-to-End Hierarchical Fine-Tuning
=====================================

Based on analysis: Head-only approaches can't create new representational power.
The breakthrough requires backbone fine-tuning with hierarchical structure learning.

Key innovations:
1. Partial backbone unfreezing (last N layers)
2. Hierarchical consistency loss: P(L0) ≈ marginalized P(L1)
3. Differential learning rates (backbone << head)

Comparison:
- Phase 1: Train classifier with FROZEN backbone (baseline)
- Phase 2: Train classifier with UNFROZEN backbone (end-to-end)

Target: 20-30% improvement over frozen baseline (72.9% L0, 66.75% L1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from multi_model_pipeline import MODELS
from hierarchical_datasets import load_hierarchical_dataset


class HierarchicalClassifier(nn.Module):
    """Hierarchical classifier with consistency enforcement."""

    def __init__(
        self,
        hidden_dim: int,
        num_l0: int,
        num_l1: int,
        l1_to_l0_mapping: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer('l1_to_l0', l1_to_l0_mapping)
        self.num_l0 = num_l0
        self.num_l1 = num_l1

        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.l0_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_l0)
        )

        self.l1_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_l1)
        )

    def forward(self, x):
        shared = self.shared_proj(x)
        l0_logits = self.l0_head(shared)
        l1_logits = self.l1_head(shared)
        return l0_logits, l1_logits

    def get_l0_marginal(self, l1_logits):
        """Compute marginalized L0 probabilities from L1 logits."""
        l1_probs = F.softmax(l1_logits, dim=-1)
        batch_size = l1_logits.shape[0]
        l0_marginal = torch.zeros(batch_size, self.num_l0, device=l1_logits.device)
        for l1_idx in range(self.num_l1):
            l0_parent = self.l1_to_l0[l1_idx].item()
            l0_marginal[:, l0_parent] += l1_probs[:, l1_idx]
        return l0_marginal


def encode_batch(backbone, tokenizer, texts, device):
    """Encode texts through frozen backbone."""
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=512, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = backbone(**inputs)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state[:, -1, :]
        else:
            hidden = outputs[0][:, -1, :]
    return hidden.float()


def encode_batch_trainable(backbone, tokenizer, texts, device):
    """Encode texts through trainable backbone (no no_grad)."""
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=512, return_tensors="pt"
    ).to(device)

    outputs = backbone(**inputs)
    if hasattr(outputs, 'last_hidden_state'):
        hidden = outputs.last_hidden_state[:, -1, :]
    else:
        hidden = outputs[0][:, -1, :]
    return hidden.float()


def train_epoch_frozen(classifier, backbone, tokenizer, train_data, optimizer, device, batch_size, consistency_weight=0.0):
    """Train with frozen backbone."""
    classifier.train()
    backbone.eval()

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training (frozen)", leave=False):
        texts = [item.text for item in batch]
        l0_labels = torch.tensor([item.level0_label for item in batch], device=device)
        l1_labels = torch.tensor([item.level1_label for item in batch], device=device)

        optimizer.zero_grad()
        hidden = encode_batch(backbone, tokenizer, texts, device)
        l0_logits, l1_logits = classifier(hidden)

        l0_loss = ce_loss_fn(l0_logits, l0_labels)
        l1_loss = ce_loss_fn(l1_logits, l1_labels)

        if consistency_weight > 0:
            l0_marginal = classifier.get_l0_marginal(l1_logits)
            l0_probs = F.softmax(l0_logits, dim=-1)
            consistency_loss = F.kl_div(
                torch.log(l0_marginal + 1e-8),
                l0_probs.detach(),
                reduction='batchmean'
            )
            loss = l0_loss + l1_loss + consistency_weight * consistency_loss
        else:
            loss = l0_loss + l1_loss

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def train_epoch_e2e(classifier, backbone, tokenizer, train_data, optimizer, device, batch_size, consistency_weight=0.5):
    """Train end-to-end with unfrozen backbone."""
    classifier.train()
    # Keep backbone in eval mode to preserve batch norm/dropout behavior
    # but still allow gradients to flow through unfrozen layers
    backbone.eval()

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training (e2e)", leave=False):
        texts = [item.text for item in batch]
        l0_labels = torch.tensor([item.level0_label for item in batch], device=device)
        l1_labels = torch.tensor([item.level1_label for item in batch], device=device)

        optimizer.zero_grad()
        hidden = encode_batch_trainable(backbone, tokenizer, texts, device)
        l0_logits, l1_logits = classifier(hidden)

        l0_loss = ce_loss_fn(l0_logits, l0_labels)
        l1_loss = ce_loss_fn(l1_logits, l1_labels)

        # Hierarchical consistency
        l0_marginal = classifier.get_l0_marginal(l1_logits)
        l0_probs = F.softmax(l0_logits, dim=-1)
        consistency_loss = F.kl_div(
            torch.log(l0_marginal + 1e-8),
            l0_probs.detach(),
            reduction='batchmean'
        )

        loss = l0_loss + l1_loss + consistency_weight * consistency_loss

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(classifier.parameters()) + list(backbone.parameters()),
            1.0
        )
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(classifier, backbone, tokenizer, val_data, device, batch_size):
    """Evaluate classifier."""
    classifier.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    all_l0_preds, all_l1_preds = [], []
    all_l0_labels, all_l1_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]
        l1_labels = [item.level1_label for item in batch]

        hidden = encode_batch(backbone, tokenizer, texts, device)
        l0_logits, l1_logits = classifier(hidden)

        all_l0_preds.extend(l0_logits.argmax(dim=1).cpu().tolist())
        all_l1_preds.extend(l1_logits.argmax(dim=1).cpu().tolist())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
    l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])
    hier_acc = np.mean([(p0 == l0) and (p1 == l1) for p0, l0, p1, l1 in zip(all_l0_preds, all_l0_labels, all_l1_preds, all_l1_labels)])

    return {'l0_acc': l0_acc, 'l1_acc': l1_acc, 'hier_acc': hier_acc}


def unfreeze_backbone_layers(backbone, num_layers: int):
    """Unfreeze the last N layers of backbone and convert to FP32 for stable training."""
    # First, freeze everything
    for param in backbone.parameters():
        param.requires_grad = False

    if num_layers == 0:
        return 0

    # Find transformer layers
    if hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
        layers = backbone.model.layers
        norm = backbone.model.norm if hasattr(backbone.model, 'norm') else None
    elif hasattr(backbone, 'layers'):
        layers = backbone.layers
        norm = backbone.norm if hasattr(backbone, 'norm') else None
    elif hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layer'):
        layers = backbone.encoder.layer
        norm = None
    else:
        print("  Warning: Could not find transformer layers")
        return 0

    total_layers = len(layers)
    layers_to_unfreeze = min(num_layers, total_layers)

    # Unfreeze last N layers AND convert to FP32 to avoid NaN during training
    for i, layer in enumerate(layers):
        if i >= total_layers - layers_to_unfreeze:
            layer.float()  # Convert to FP32
            for param in layer.parameters():
                param.requires_grad = True

    # Unfreeze final norm
    if norm is not None:
        norm.float()  # Convert to FP32
        for param in norm.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    return trainable


def build_l1_to_l0_mapping(train_data, num_l0: int, num_l1: int):
    """Build mapping from L1 classes to their L0 parents."""
    l1_to_l0 = torch.zeros(num_l1, dtype=torch.long)
    for item in train_data:
        l1_to_l0[item.level1_label] = item.level0_label
    return l1_to_l0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--frozen_epochs", type=int, default=10)
    parser.add_argument("--e2e_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=2e-6)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--unfreeze_layers", type=int, default=4)
    parser.add_argument("--consistency_weight", type=float, default=0.3)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--val_samples", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading dataset...")
    train_data = load_hierarchical_dataset(args.dataset, split="train", max_samples=args.train_samples)
    val_data = load_hierarchical_dataset(args.dataset, split="test", max_samples=args.val_samples)

    num_l0 = len(set(item.level0_label for item in train_data))
    num_l1 = len(set(item.level1_label for item in train_data))
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"  L0: {num_l0} classes, L1: {num_l1} classes")

    l1_to_l0 = build_l1_to_l0_mapping(train_data, num_l0, num_l1)
    print(f"  L1->L0 mapping: {l1_to_l0.tolist()}")

    # Load backbone
    print(f"\nLoading backbone: {args.model}...")
    model_config = MODELS[args.model]

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_path, trust_remote_code=model_config.trust_remote_code)
    if model_config.pooling == "last":
        tokenizer.padding_side = "left"

    backbone = AutoModel.from_pretrained(
        model_config.hf_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16,
    ).to(device)

    # Freeze backbone initially
    for p in backbone.parameters():
        p.requires_grad = False

    hidden_dim = model_config.hidden_dim
    print(f"  Hidden dim: {hidden_dim}")

    # Create classifier
    classifier = HierarchicalClassifier(hidden_dim, num_l0, num_l1, l1_to_l0.to(device)).to(device)
    print(f"  Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")

    # ==========================================
    # PHASE 1: Train with FROZEN backbone
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 1: FROZEN BACKBONE (Baseline)")
    print("="*60)

    optimizer_frozen = AdamW(classifier.parameters(), lr=args.head_lr, weight_decay=0.01)

    best_frozen = {'l0_acc': 0, 'l1_acc': 0, 'hier_acc': 0}

    for epoch in range(1, args.frozen_epochs + 1):
        train_loss = train_epoch_frozen(classifier, backbone, tokenizer, train_data, optimizer_frozen, device, args.batch_size)
        results = evaluate(classifier, backbone, tokenizer, val_data, device, args.batch_size)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, L0={results['l0_acc']:.4f}, L1={results['l1_acc']:.4f}")

        if results['l0_acc'] + results['l1_acc'] > best_frozen['l0_acc'] + best_frozen['l1_acc']:
            best_frozen = results.copy()
            best_frozen['epoch'] = epoch

    print(f"\nFrozen baseline: L0={best_frozen['l0_acc']:.4f}, L1={best_frozen['l1_acc']:.4f}")

    # Save frozen classifier weights for E2E initialization
    frozen_classifier_state = {k: v.clone() for k, v in classifier.state_dict().items()}

    # ==========================================
    # PHASE 2: Train END-TO-END
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 2: END-TO-END FINE-TUNING")
    print("="*60)

    # Initialize from frozen-trained classifier (don't start from scratch!)
    classifier.load_state_dict(frozen_classifier_state)
    print("  Initialized classifier from frozen-trained weights")

    # Unfreeze backbone layers
    backbone_trainable = unfreeze_backbone_layers(backbone, args.unfreeze_layers)
    print(f"  Backbone trainable params: {backbone_trainable:,}")
    print(f"  Backbone LR: {args.backbone_lr}, Head LR: {args.head_lr}")

    # Differential learning rates
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    classifier_params = list(classifier.parameters())

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
    param_groups.append({'params': classifier_params, 'lr': args.head_lr})

    optimizer_e2e = AdamW(param_groups, weight_decay=0.01)

    best_e2e = {'l0_acc': 0, 'l1_acc': 0, 'hier_acc': 0}

    for epoch in range(1, args.e2e_epochs + 1):
        train_loss = train_epoch_e2e(classifier, backbone, tokenizer, train_data, optimizer_e2e, device, args.batch_size, args.consistency_weight)
        results = evaluate(classifier, backbone, tokenizer, val_data, device, args.batch_size)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, L0={results['l0_acc']:.4f}, L1={results['l1_acc']:.4f}")

        if results['l0_acc'] + results['l1_acc'] > best_e2e['l0_acc'] + best_e2e['l1_acc']:
            best_e2e = results.copy()
            best_e2e['epoch'] = epoch

    # ==========================================
    # RESULTS
    # ==========================================
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Frozen Baseline: L0={best_frozen['l0_acc']:.4f}, L1={best_frozen['l1_acc']:.4f}, Hier={best_frozen['hier_acc']:.4f}")
    print(f"E2E Fine-tuned:  L0={best_e2e['l0_acc']:.4f}, L1={best_e2e['l1_acc']:.4f}, Hier={best_e2e['hier_acc']:.4f}")
    print(f"\nDelta: L0={best_e2e['l0_acc']-best_frozen['l0_acc']:+.4f}, L1={best_e2e['l1_acc']-best_frozen['l1_acc']:+.4f}")

    if best_frozen['l0_acc'] > 0:
        l0_pct = (best_e2e['l0_acc'] - best_frozen['l0_acc']) / best_frozen['l0_acc'] * 100
        l1_pct = (best_e2e['l1_acc'] - best_frozen['l1_acc']) / best_frozen['l1_acc'] * 100
        print(f"Improvement: L0={l0_pct:+.1f}%, L1={l1_pct:+.1f}%")
    else:
        l0_pct, l1_pct = 0, 0

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"e2e_hierarchical_{args.model}.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "config": {
                "backbone_lr": args.backbone_lr,
                "head_lr": args.head_lr,
                "unfreeze_layers": args.unfreeze_layers,
                "consistency_weight": args.consistency_weight,
                "frozen_epochs": args.frozen_epochs,
                "e2e_epochs": args.e2e_epochs,
            },
            "frozen_baseline": best_frozen,
            "e2e_finetuned": best_e2e,
            "improvement_pct": {"l0": l0_pct, "l1": l1_pct}
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
