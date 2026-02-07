"""
Label-Text Semantic Matching for Hierarchical Classification
============================================================

Instead of classification heads, leverage the pre-trained embedding space:
1. Encode label descriptions as embeddings
2. Classify by nearest neighbor in embedding space
3. Train lightweight adapters to improve label-text alignment

This preserves the general-purpose embedding capabilities while
adapting to the hierarchical structure.

Approach:
- L0 labels: "Knowledge", "Lifestyle", "Entertainment", "Professional"
- L1 labels: Full label names like "Science & Mathematics"
- Learn lightweight projections that maximize label-text similarity
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
from hierarchical_datasets import load_hierarchical_dataset, YahooAnswersHierarchical


class LabelEncoder(nn.Module):
    """Encodes label descriptions into embeddings."""

    def __init__(self, backbone, tokenizer, device):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def encode_labels(self, label_texts):
        """Encode a list of label text descriptions."""
        inputs = self.tokenizer(
            label_texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(self.device)

        outputs = self.backbone(**inputs)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state[:, -1, :]
        else:
            hidden = outputs[0][:, -1, :]

        return F.normalize(hidden.float(), dim=-1)


class HierarchicalLabelMatcher(nn.Module):
    """
    Hierarchical label matching with learned projections.

    Instead of classification heads, we learn projections that map
    text embeddings closer to their corresponding label embeddings.
    """

    def __init__(
        self,
        hidden_dim: int,
        l0_embeddings: torch.Tensor,  # (num_l0, hidden_dim)
        l1_embeddings: torch.Tensor,  # (num_l1, hidden_dim)
        l1_to_l0: torch.Tensor,        # (num_l1,) mapping
    ):
        super().__init__()

        self.register_buffer('l0_embeddings', l0_embeddings)
        self.register_buffer('l1_embeddings', l1_embeddings)
        self.register_buffer('l1_to_l0', l1_to_l0)

        self.num_l0 = l0_embeddings.shape[0]
        self.num_l1 = l1_embeddings.shape[0]

        # Lightweight text adapter (preserves structure)
        self.text_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, text_emb, return_sims=False):
        """
        Args:
            text_emb: (B, hidden_dim) text embeddings
        Returns:
            l0_logits: (B, num_l0)
            l1_logits: (B, num_l1)
        """
        # Adapt text embeddings
        adapted = text_emb + self.text_adapter(text_emb)  # Residual
        adapted = F.normalize(adapted, dim=-1)

        # Compute similarities to label embeddings
        l0_sims = adapted @ self.l0_embeddings.T / self.temperature.abs()
        l1_sims = adapted @ self.l1_embeddings.T / self.temperature.abs()

        if return_sims:
            return l0_sims, l1_sims, adapted
        return l0_sims, l1_sims

    def get_l0_marginal(self, l1_sims):
        """Marginalize L1 probabilities to L0."""
        l1_probs = F.softmax(l1_sims, dim=-1)
        batch_size = l1_sims.shape[0]
        l0_marginal = torch.zeros(batch_size, self.num_l0, device=l1_sims.device)
        for l1_idx in range(self.num_l1):
            l0_parent = self.l1_to_l0[l1_idx].item()
            l0_marginal[:, l0_parent] += l1_probs[:, l1_idx]
        return l0_marginal


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

    return F.normalize(hidden.float(), dim=-1)


def train_epoch(matcher, backbone, tokenizer, train_data, optimizer, device, batch_size, consistency_weight=0.3):
    """Train for one epoch."""
    matcher.train()
    backbone.eval()

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        texts = [item.text for item in batch]
        l0_labels = torch.tensor([item.level0_label for item in batch], device=device)
        l1_labels = torch.tensor([item.level1_label for item in batch], device=device)

        optimizer.zero_grad()

        text_emb = encode_batch(backbone, tokenizer, texts, device)
        l0_logits, l1_logits = matcher(text_emb)

        # Classification losses
        l0_loss = ce_loss_fn(l0_logits, l0_labels)
        l1_loss = ce_loss_fn(l1_logits, l1_labels)

        # Hierarchical consistency
        l0_marginal = matcher.get_l0_marginal(l1_logits)
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
        torch.nn.utils.clip_grad_norm_(matcher.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(matcher, backbone, tokenizer, val_data, device, batch_size):
    """Evaluate on validation set."""
    matcher.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    all_l0_preds, all_l1_preds = [], []
    all_l0_labels, all_l1_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]
        l1_labels = [item.level1_label for item in batch]

        text_emb = encode_batch(backbone, tokenizer, texts, device)
        l0_logits, l1_logits = matcher(text_emb)

        all_l0_preds.extend(l0_logits.argmax(dim=1).cpu().tolist())
        all_l1_preds.extend(l1_logits.argmax(dim=1).cpu().tolist())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
    l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])
    hier_acc = np.mean([(p0 == l0) and (p1 == l1) for p0, l0, p1, l1 in zip(all_l0_preds, all_l0_labels, all_l1_preds, all_l1_labels)])

    return {'l0_acc': l0_acc, 'l1_acc': l1_acc, 'hier_acc': hier_acc}


@torch.no_grad()
def evaluate_zero_shot(backbone, tokenizer, l0_embs, l1_embs, val_data, device, batch_size):
    """Zero-shot evaluation (no training)."""
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    all_l0_preds, all_l1_preds = [], []
    all_l0_labels, all_l1_labels = [], []

    for batch in tqdm(dataloader, desc="Zero-shot eval", leave=False):
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]
        l1_labels = [item.level1_label for item in batch]

        text_emb = encode_batch(backbone, tokenizer, texts, device)

        # Simple cosine similarity
        l0_sims = text_emb @ l0_embs.T
        l1_sims = text_emb @ l1_embs.T

        all_l0_preds.extend(l0_sims.argmax(dim=1).cpu().tolist())
        all_l1_preds.extend(l1_sims.argmax(dim=1).cpu().tolist())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
    l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])

    return {'l0_acc': l0_acc, 'l1_acc': l1_acc}


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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
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

    # Get label names
    l0_names = train_data.level0_names
    l1_names = train_data.level1_names
    num_l0 = len(l0_names)
    num_l1 = len(l1_names)

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"  L0: {num_l0} classes: {l0_names}")
    print(f"  L1: {num_l1} classes: {l1_names}")

    l1_to_l0 = build_l1_to_l0_mapping(train_data, num_l0, num_l1)

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

    # Freeze backbone completely
    for p in backbone.parameters():
        p.requires_grad = False

    hidden_dim = model_config.hidden_dim

    # Create label descriptions for better semantic matching
    l0_descriptions = [
        f"This question is about {name.lower()}."
        for name in l0_names
    ]
    l1_descriptions = [
        f"This question is about {name.lower()}."
        for name in l1_names
    ]

    print("\nEncoding label descriptions...")
    label_encoder = LabelEncoder(backbone, tokenizer, device)
    l0_embs = label_encoder.encode_labels(l0_descriptions)
    l1_embs = label_encoder.encode_labels(l1_descriptions)
    print(f"  L0 embeddings: {l0_embs.shape}")
    print(f"  L1 embeddings: {l1_embs.shape}")

    # Zero-shot baseline
    print("\n" + "="*60)
    print("ZERO-SHOT BASELINE (no training)")
    print("="*60)
    zero_shot = evaluate_zero_shot(backbone, tokenizer, l0_embs, l1_embs, val_data, device, args.batch_size)
    print(f"  L0: {zero_shot['l0_acc']:.4f}, L1: {zero_shot['l1_acc']:.4f}")

    # Create matcher
    matcher = HierarchicalLabelMatcher(
        hidden_dim, l0_embs, l1_embs, l1_to_l0.to(device)
    ).to(device)

    print(f"\nMatcher params: {sum(p.numel() for p in matcher.parameters()):,}")

    # Optimizer
    optimizer = AdamW(matcher.parameters(), lr=args.lr, weight_decay=0.01)

    # Training
    print("\n" + "="*60)
    print("TRAINING LABEL MATCHER")
    print("="*60)

    best_results = {'l0_acc': 0, 'l1_acc': 0, 'hier_acc': 0}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(matcher, backbone, tokenizer, train_data, optimizer, device, args.batch_size, args.consistency_weight)
        results = evaluate(matcher, backbone, tokenizer, val_data, device, args.batch_size)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, L0={results['l0_acc']:.4f}, L1={results['l1_acc']:.4f}, temp={matcher.temperature.item():.3f}")

        if results['l0_acc'] + results['l1_acc'] > best_results['l0_acc'] + best_results['l1_acc']:
            best_results = results.copy()
            best_results['epoch'] = epoch

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Zero-shot:     L0={zero_shot['l0_acc']:.4f}, L1={zero_shot['l1_acc']:.4f}")
    print(f"Label Matcher: L0={best_results['l0_acc']:.4f}, L1={best_results['l1_acc']:.4f}")
    print(f"\nDelta vs zero-shot: L0={best_results['l0_acc']-zero_shot['l0_acc']:+.4f}, L1={best_results['l1_acc']-zero_shot['l1_acc']:+.4f}")

    # Compare to frozen head baseline (from previous experiments: ~73%/67%)
    frozen_baseline = {'l0_acc': 0.729, 'l1_acc': 0.6675}
    print(f"\nCompare to frozen head baseline: L0={frozen_baseline['l0_acc']:.4f}, L1={frozen_baseline['l1_acc']:.4f}")
    print(f"Delta vs frozen head: L0={best_results['l0_acc']-frozen_baseline['l0_acc']:+.4f}, L1={best_results['l1_acc']-frozen_baseline['l1_acc']:+.4f}")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"label_matching_{args.model}.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "zero_shot": zero_shot,
            "trained": best_results,
            "frozen_head_baseline": frozen_baseline,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
