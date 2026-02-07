"""
Hierarchical Contrastive Learning V2
=====================================

Directly optimize the embedding space for hierarchical structure.

Key insight: The backbone embeddings don't naturally have hierarchical structure.
We need to LEARN a projection that creates that structure.

Approach:
1. Project embeddings into a learned space
2. Use hierarchical contrastive loss:
   - L0 siblings should be closer than L0 non-siblings
   - L1 siblings should be closest of all
3. Joint training with classification

This creates an embedding space where hierarchy emerges naturally.
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
import random

from multi_model_pipeline import MODELS
from hierarchical_datasets import load_hierarchical_dataset


class HierarchicalProjector(nn.Module):
    """
    Projects embeddings into a space with hierarchical structure.

    Architecture:
    - Shared encoder
    - L0 projection (coarse structure)
    - L1 projection (fine structure conditioned on L0)
    """

    def __init__(self, hidden_dim: int, proj_dim: int, num_l0: int, num_l1: int):
        super().__init__()

        self.proj_dim = proj_dim
        self.num_l0 = num_l0
        self.num_l1 = num_l1

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # L0 projection (coarse)
        self.l0_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # L1 projection (fine, conditioned on L0 embedding)
        self.l1_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2 + proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Classification heads
        self.l0_head = nn.Linear(proj_dim, num_l0)
        self.l1_head = nn.Linear(proj_dim, num_l1)

    def forward(self, x):
        """
        Args:
            x: (B, hidden_dim) backbone embeddings
        Returns:
            l0_logits, l1_logits, l0_emb, l1_emb
        """
        # Shared encoding
        shared = self.encoder(x)

        # L0 projection
        l0_emb = self.l0_proj(shared)
        l0_emb_norm = F.normalize(l0_emb, dim=-1)

        # L1 projection (conditioned on L0)
        l1_input = torch.cat([shared, l0_emb], dim=-1)
        l1_emb = self.l1_proj(l1_input)
        l1_emb_norm = F.normalize(l1_emb, dim=-1)

        # Classification
        l0_logits = self.l0_head(l0_emb_norm)
        l1_logits = self.l1_head(l1_emb_norm)

        return l0_logits, l1_logits, l0_emb_norm, l1_emb_norm


def hierarchical_contrastive_loss(l0_emb, l1_emb, l0_labels, l1_labels, temperature=0.1):
    """
    Hierarchical contrastive loss.

    Pull together:
    - L0 siblings (same L0 category)
    - L1 siblings (same L1 category) - even stronger pull

    Push apart:
    - Different L0 categories
    """
    batch_size = l0_emb.shape[0]
    device = l0_emb.device

    # Compute similarity matrices
    l0_sim = l0_emb @ l0_emb.T / temperature
    l1_sim = l1_emb @ l1_emb.T / temperature

    # Create masks
    l0_labels_exp = l0_labels.unsqueeze(0).expand(batch_size, -1)
    l0_same_mask = (l0_labels_exp == l0_labels_exp.T).float()

    l1_labels_exp = l1_labels.unsqueeze(0).expand(batch_size, -1)
    l1_same_mask = (l1_labels_exp == l1_labels_exp.T).float()

    # Remove diagonal (self-similarity)
    mask_diag = 1.0 - torch.eye(batch_size, device=device)
    l0_same_mask = l0_same_mask * mask_diag
    l1_same_mask = l1_same_mask * mask_diag

    # L0 contrastive loss (InfoNCE style)
    # For each sample, positives are L0 siblings, negatives are different L0
    l0_loss = 0.0
    for i in range(batch_size):
        pos_mask = l0_same_mask[i]
        if pos_mask.sum() > 0:
            # Log-softmax over all other samples, then average over positives
            log_probs = F.log_softmax(l0_sim[i] * mask_diag[i], dim=0)
            l0_loss += -(pos_mask * log_probs).sum() / pos_mask.sum()
    l0_loss = l0_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)

    # L1 contrastive loss
    l1_loss = 0.0
    for i in range(batch_size):
        pos_mask = l1_same_mask[i]
        if pos_mask.sum() > 0:
            log_probs = F.log_softmax(l1_sim[i] * mask_diag[i], dim=0)
            l1_loss += -(pos_mask * log_probs).sum() / pos_mask.sum()
    l1_loss = l1_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)

    return l0_loss + l1_loss


def encode_batch(backbone, tokenizer, texts, device):
    """Encode texts through backbone."""
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


def sample_balanced_batch(train_data, batch_size):
    """Sample a batch with better class balance for contrastive learning."""
    # Group by L1 label
    by_l1 = {}
    for idx, item in enumerate(train_data):
        l1 = item.level1_label
        if l1 not in by_l1:
            by_l1[l1] = []
        by_l1[l1].append(idx)

    # Sample evenly from each L1 class
    samples_per_class = max(2, batch_size // len(by_l1))
    selected_indices = []

    for l1, indices in by_l1.items():
        n = min(samples_per_class, len(indices))
        selected_indices.extend(random.sample(indices, n))

    # Trim or pad to exact batch size
    if len(selected_indices) > batch_size:
        selected_indices = random.sample(selected_indices, batch_size)
    elif len(selected_indices) < batch_size:
        # Add random samples to fill
        remaining = batch_size - len(selected_indices)
        all_indices = list(range(len(train_data)))
        selected_indices.extend(random.sample(all_indices, remaining))

    return [train_data[i] for i in selected_indices]


def train_epoch(projector, backbone, tokenizer, train_data, optimizer, device, batch_size, contrastive_weight=1.0):
    """Train for one epoch."""
    projector.train()
    backbone.eval()

    num_batches = len(train_data) // batch_size
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0

    for _ in tqdm(range(num_batches), desc="Training", leave=False):
        # Use balanced sampling for better contrastive learning
        batch = sample_balanced_batch(train_data, batch_size)

        texts = [item.text for item in batch]
        l0_labels = torch.tensor([item.level0_label for item in batch], device=device)
        l1_labels = torch.tensor([item.level1_label for item in batch], device=device)

        optimizer.zero_grad()

        hidden = encode_batch(backbone, tokenizer, texts, device)
        l0_logits, l1_logits, l0_emb, l1_emb = projector(hidden)

        # Classification loss
        l0_cls_loss = ce_loss_fn(l0_logits, l0_labels)
        l1_cls_loss = ce_loss_fn(l1_logits, l1_labels)
        cls_loss = l0_cls_loss + l1_cls_loss

        # Hierarchical contrastive loss
        contrast_loss = hierarchical_contrastive_loss(l0_emb, l1_emb, l0_labels, l1_labels)

        loss = cls_loss + contrastive_weight * contrast_loss

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(projector, backbone, tokenizer, val_data, device, batch_size):
    """Evaluate on validation set."""
    projector.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    all_l0_preds, all_l1_preds = [], []
    all_l0_labels, all_l1_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]
        l1_labels = [item.level1_label for item in batch]

        hidden = encode_batch(backbone, tokenizer, texts, device)
        l0_logits, l1_logits, _, _ = projector(hidden)

        all_l0_preds.extend(l0_logits.argmax(dim=1).cpu().tolist())
        all_l1_preds.extend(l1_logits.argmax(dim=1).cpu().tolist())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
    l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])
    hier_acc = np.mean([(p0 == l0) and (p1 == l1) for p0, l0, p1, l1 in zip(all_l0_preds, all_l0_labels, all_l1_preds, all_l1_labels)])

    return {'l0_acc': l0_acc, 'l1_acc': l1_acc, 'hier_acc': hier_acc}


@torch.no_grad()
def analyze_embedding_structure(projector, backbone, tokenizer, val_data, device, batch_size):
    """Analyze the hierarchical structure of learned embeddings."""
    projector.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_l0_embs, all_l1_embs = [], []
    all_l0_labels, all_l1_labels = [], []

    for batch in dataloader:
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]
        l1_labels = [item.level1_label for item in batch]

        hidden = encode_batch(backbone, tokenizer, texts, device)
        _, _, l0_emb, l1_emb = projector(hidden)

        all_l0_embs.append(l0_emb.cpu())
        all_l1_embs.append(l1_emb.cpu())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    l0_embs = torch.cat(all_l0_embs, dim=0).numpy()
    l1_embs = torch.cat(all_l1_embs, dim=0).numpy()
    l0_labels = np.array(all_l0_labels)
    l1_labels = np.array(all_l1_labels)

    # Compute intra-class vs inter-class similarity
    def compute_separation(embs, labels):
        intra_sims = []
        inter_sims = []
        n = min(500, len(embs))  # Sample for speed
        indices = np.random.choice(len(embs), n, replace=False)

        for i in indices:
            for j in indices:
                if i >= j:
                    continue
                sim = (embs[i] * embs[j]).sum()
                if labels[i] == labels[j]:
                    intra_sims.append(sim)
                else:
                    inter_sims.append(sim)

        if intra_sims and inter_sims:
            return np.mean(intra_sims) - np.mean(inter_sims)
        return 0.0

    l0_sep = compute_separation(l0_embs, l0_labels)
    l1_sep = compute_separation(l1_embs, l1_labels)

    return {'l0_separation': l0_sep, 'l1_separation': l1_sep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=48)  # Larger for contrastive
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
    parser.add_argument("--proj_dim", type=int, default=256)
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

    for p in backbone.parameters():
        p.requires_grad = False

    hidden_dim = model_config.hidden_dim

    # Create projector
    projector = HierarchicalProjector(hidden_dim, args.proj_dim, num_l0, num_l1).to(device)
    print(f"  Projector params: {sum(p.numel() for p in projector.parameters()):,}")

    # Optimizer
    optimizer = AdamW(projector.parameters(), lr=args.lr, weight_decay=0.01)

    # Training
    print("\n" + "="*60)
    print("HIERARCHICAL CONTRASTIVE TRAINING")
    print("="*60)

    best_results = {'l0_acc': 0, 'l1_acc': 0, 'hier_acc': 0}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(projector, backbone, tokenizer, train_data, optimizer, device, args.batch_size, args.contrastive_weight)
        results = evaluate(projector, backbone, tokenizer, val_data, device, args.batch_size)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, L0={results['l0_acc']:.4f}, L1={results['l1_acc']:.4f}")

        if results['l0_acc'] + results['l1_acc'] > best_results['l0_acc'] + best_results['l1_acc']:
            best_results = results.copy()
            best_results['epoch'] = epoch

    # Analyze final embedding structure
    print("\nAnalyzing embedding structure...")
    structure = analyze_embedding_structure(projector, backbone, tokenizer, val_data, device, args.batch_size)
    print(f"  L0 separation: {structure['l0_separation']:.4f}")
    print(f"  L1 separation: {structure['l1_separation']:.4f}")

    # Summary
    frozen_baseline = {'l0_acc': 0.729, 'l1_acc': 0.6675}
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Frozen baseline:      L0={frozen_baseline['l0_acc']:.4f}, L1={frozen_baseline['l1_acc']:.4f}")
    print(f"Hierarchical Contrast: L0={best_results['l0_acc']:.4f}, L1={best_results['l1_acc']:.4f}")
    print(f"\nDelta: L0={best_results['l0_acc']-frozen_baseline['l0_acc']:+.4f}, L1={best_results['l1_acc']-frozen_baseline['l1_acc']:+.4f}")

    improvement_l0 = (best_results['l0_acc'] - frozen_baseline['l0_acc']) / frozen_baseline['l0_acc'] * 100
    improvement_l1 = (best_results['l1_acc'] - frozen_baseline['l1_acc']) / frozen_baseline['l1_acc'] * 100
    print(f"Improvement: L0={improvement_l0:+.1f}%, L1={improvement_l1:+.1f}%")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"hierarchical_contrastive_v2_{args.model}.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "config": {
                "batch_size": args.batch_size,
                "lr": args.lr,
                "contrastive_weight": args.contrastive_weight,
                "proj_dim": args.proj_dim,
            },
            "frozen_baseline": frozen_baseline,
            "trained": best_results,
            "embedding_structure": structure,
            "improvement_pct": {"l0": improvement_l0, "l1": improvement_l1}
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
