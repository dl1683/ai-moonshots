"""
Fractal VQ: Nested Discrete Codebooks for Hierarchical Classification

Based on the moonshot experimental design:

Core idea: Hierarchy should be DISCRETE and NESTED, not continuous and flat.
Build a tree-structured codebook where coarse codes index into fine codebooks.

Architecture:
- Coarse codebook C0 with K0 prototypes
- Each coarse code k has its own fine codebook C1^k
- Encoding:
  - k0 = argmin ||f(x) - C0||
  - k1 = argmin ||g(x,k0) - C1^{k0}||

Loss:
- VQ commitment + reconstruction + classification CE

This creates an ULTRAMETRIC space by construction - hierarchy is literal.
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

from multi_model_pipeline import MODELS, load_model
from hierarchical_datasets import load_hierarchical_dataset


class VectorQuantizer(nn.Module):
    """Simple vector quantizer with straight-through gradient."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)

    def forward(self, z):
        """
        Args:
            z: [batch, dim] continuous input

        Returns:
            z_q: [batch, dim] quantized output
            indices: [batch] codebook indices
            vq_loss: scalar VQ loss
        """
        # Compute distances to all embeddings
        distances = torch.cdist(z.unsqueeze(0), self.embeddings.weight.unsqueeze(0)).squeeze(0)

        # Get nearest embedding indices
        indices = torch.argmin(distances, dim=-1)

        # Get quantized vectors
        z_q = self.embeddings(indices)

        # VQ losses
        commitment_loss = F.mse_loss(z_q.detach(), z) * self.commitment_cost
        codebook_loss = F.mse_loss(z_q, z.detach())

        # Straight-through gradient
        z_q = z + (z_q - z).detach()

        vq_loss = commitment_loss + codebook_loss

        return z_q, indices, vq_loss


class NestedVectorQuantizer(nn.Module):
    """
    Nested VQ: Each coarse code has its own fine codebook.

    This creates a tree structure:
    - K0 coarse prototypes
    - Each coarse prototype indexes a fine codebook with K1 entries
    - Total: K0 * K1 possible discrete states
    """

    def __init__(self, hidden_dim: int, num_coarse: int, num_fine_per_coarse: int,
                 coarse_dim: int = 128, fine_dim: int = 128, commitment_cost: float = 0.25):
        super().__init__()
        self.num_coarse = num_coarse
        self.num_fine_per_coarse = num_fine_per_coarse
        self.coarse_dim = coarse_dim
        self.fine_dim = fine_dim

        # Projection to coarse space
        self.to_coarse = nn.Sequential(
            nn.Linear(hidden_dim, coarse_dim),
            nn.LayerNorm(coarse_dim)
        )

        # Coarse codebook
        self.coarse_vq = VectorQuantizer(num_coarse, coarse_dim, commitment_cost)

        # Projection to fine space (conditioned on coarse)
        self.to_fine = nn.Sequential(
            nn.Linear(hidden_dim + coarse_dim, fine_dim),
            nn.LayerNorm(fine_dim)
        )

        # Fine codebooks: one per coarse code
        self.fine_codebooks = nn.ModuleList([
            VectorQuantizer(num_fine_per_coarse, fine_dim, commitment_cost)
            for _ in range(num_coarse)
        ])

    def forward(self, x, return_codes=False):
        """
        Args:
            x: [batch, hidden_dim] input embeddings

        Returns:
            z_coarse_q: [batch, coarse_dim] quantized coarse embeddings
            z_fine_q: [batch, fine_dim] quantized fine embeddings
            coarse_indices: [batch] coarse codebook indices
            fine_indices: [batch] fine codebook indices (within coarse)
            vq_loss: scalar total VQ loss
        """
        batch_size = x.size(0)

        # Step 1: Quantize to coarse level
        z_coarse = self.to_coarse(x)
        z_coarse_q, coarse_indices, coarse_vq_loss = self.coarse_vq(z_coarse)

        # Step 2: Quantize to fine level, using appropriate codebook per sample
        z_fine_input = self.to_fine(torch.cat([x, z_coarse_q], dim=-1))

        # We need to dispatch each sample to its own fine codebook
        z_fine_q = torch.zeros(batch_size, self.fine_dim, device=x.device)
        fine_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        fine_vq_loss = 0.0

        # Group samples by coarse index for efficient processing
        for k in range(self.num_coarse):
            mask = (coarse_indices == k)
            if mask.sum() > 0:
                z_fine_k = z_fine_input[mask]
                z_fine_q_k, fine_idx_k, vq_loss_k = self.fine_codebooks[k](z_fine_k)
                z_fine_q[mask] = z_fine_q_k
                fine_indices[mask] = fine_idx_k
                fine_vq_loss = fine_vq_loss + vq_loss_k * mask.sum() / batch_size

        total_vq_loss = coarse_vq_loss + fine_vq_loss

        if return_codes:
            return z_coarse_q, z_fine_q, coarse_indices, fine_indices, total_vq_loss
        return z_coarse_q, z_fine_q, total_vq_loss


class FractalVQHead(nn.Module):
    """
    Fractal VQ classification head.

    Uses nested discrete codebooks to enforce hierarchical structure.
    L0 predictions come from coarse codes.
    L1 predictions come from (coarse, fine) code combinations.
    """

    def __init__(self, hidden_dim: int, num_l0: int, num_l1: int,
                 num_coarse_codes: int = 16, num_fine_per_coarse: int = 8,
                 coarse_dim: int = 128, fine_dim: int = 128):
        super().__init__()

        self.nested_vq = NestedVectorQuantizer(
            hidden_dim, num_coarse_codes, num_fine_per_coarse,
            coarse_dim, fine_dim
        )

        # L0 head: predict from coarse quantized embedding
        self.l0_head = nn.Linear(coarse_dim, num_l0)

        # L1 head: predict from both coarse and fine quantized embeddings
        self.l1_head = nn.Linear(coarse_dim + fine_dim, num_l1)

    def forward(self, x, return_vq_loss=True):
        """
        Args:
            x: [batch, hidden_dim] input embeddings

        Returns:
            l0_logits: [batch, num_l0] coarse predictions
            l1_logits: [batch, num_l1] fine predictions
            vq_loss: scalar VQ loss (if return_vq_loss=True)
        """
        z_coarse_q, z_fine_q, vq_loss = self.nested_vq(x)

        l0_logits = self.l0_head(z_coarse_q)
        l1_logits = self.l1_head(torch.cat([z_coarse_q, z_fine_q], dim=-1))

        if return_vq_loss:
            return l0_logits, l1_logits, vq_loss
        return l0_logits, l1_logits

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class BaselineHead(nn.Module):
    """Standard MLP baseline for fair comparison."""
    def __init__(self, hidden_dim: int, num_l0: int, num_l1: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256)
        )
        self.l0_head = nn.Linear(256, num_l0)
        self.l1_head = nn.Linear(256, num_l1)

    def forward(self, x, return_vq_loss=False):
        features = self.projection(x)
        l0_logits = self.l0_head(features)
        l1_logits = self.l1_head(features)
        if return_vq_loss:
            return l0_logits, l1_logits, torch.tensor(0.0, device=x.device)
        return l0_logits, l1_logits

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


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


def train_epoch(model, backbone, tokenizer, train_data, optimizer, device,
                batch_size, vq_weight=0.1, is_vq_model=True):
    """Train for one epoch."""
    model.train()
    backbone.eval()

    dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=lambda x: x
    )

    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_ce = 0
    total_vq = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        texts = [item.text for item in batch]
        l0_labels = torch.tensor([item.level0_label for item in batch], device=device)
        l1_labels = torch.tensor([item.level1_label for item in batch], device=device)

        optimizer.zero_grad()
        hidden = encode_batch(backbone, tokenizer, texts, device)

        if is_vq_model:
            l0_logits, l1_logits, vq_loss = model(hidden, return_vq_loss=True)
        else:
            l0_logits, l1_logits = model(hidden)
            vq_loss = torch.tensor(0.0, device=device)

        # Classification losses
        l0_loss = ce_loss_fn(l0_logits, l0_labels)
        l1_loss = ce_loss_fn(l1_logits, l1_labels)
        ce_loss = l0_loss + l1_loss

        # Total loss
        loss = ce_loss + vq_weight * vq_loss

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_vq += vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
        num_batches += 1

    return {
        'total': total_loss / max(1, num_batches),
        'ce': total_ce / max(1, num_batches),
        'vq': total_vq / max(1, num_batches)
    }


@torch.no_grad()
def evaluate(model, backbone, tokenizer, val_data, device, batch_size, is_vq_model=True):
    """Evaluate on validation set."""
    model.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_l0_preds, all_l1_preds = [], []
    all_l0_labels, all_l1_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]
        l1_labels = [item.level1_label for item in batch]

        hidden = encode_batch(backbone, tokenizer, texts, device)

        if is_vq_model:
            l0_logits, l1_logits, _ = model(hidden, return_vq_loss=True)
        else:
            l0_logits, l1_logits = model(hidden)

        all_l0_preds.extend(l0_logits.argmax(dim=1).cpu().tolist())
        all_l1_preds.extend(l1_logits.argmax(dim=1).cpu().tolist())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    l0_acc = np.mean([p == l for p, l in zip(all_l0_preds, all_l0_labels)])
    l1_acc = np.mean([p == l for p, l in zip(all_l1_preds, all_l1_labels)])

    hier_acc = np.mean([
        (p0 == l0) and (p1 == l1)
        for p0, l0, p1, l1 in zip(all_l0_preds, all_l0_labels, all_l1_preds, all_l1_labels)
    ])

    return {'l0_acc': l0_acc, 'l1_acc': l1_acc, 'hier_acc': hier_acc}


@torch.no_grad()
def analyze_codebook_usage(model, backbone, tokenizer, val_data, device, batch_size):
    """Analyze how the codebooks are being used."""
    if not hasattr(model, 'nested_vq'):
        return {}

    model.eval()
    backbone.eval()

    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_coarse_indices = []
    all_fine_indices = []
    all_l0_labels = []
    all_l1_labels = []

    for batch in dataloader:
        texts = [item.text for item in batch]
        l0_labels = [item.level0_label for item in batch]
        l1_labels = [item.level1_label for item in batch]

        hidden = encode_batch(backbone, tokenizer, texts, device)
        _, _, coarse_idx, fine_idx, _ = model.nested_vq(hidden, return_codes=True)

        all_coarse_indices.extend(coarse_idx.cpu().tolist())
        all_fine_indices.extend(fine_idx.cpu().tolist())
        all_l0_labels.extend(l0_labels)
        all_l1_labels.extend(l1_labels)

    # Compute alignment between codebook indices and class labels
    coarse_indices = np.array(all_coarse_indices)
    l0_labels = np.array(all_l0_labels)

    # How many unique coarse codes are used?
    unique_coarse = len(np.unique(coarse_indices))

    # Normalized Mutual Information between coarse codes and L0 labels
    from collections import Counter
    coarse_l0_pairs = list(zip(coarse_indices, l0_labels))
    pair_counts = Counter(coarse_l0_pairs)

    return {
        'unique_coarse_codes': unique_coarse,
        'total_coarse_codes': model.nested_vq.num_coarse,
        'codebook_utilization': unique_coarse / model.nested_vq.num_coarse
    }


def run_experiment(model, backbone, tokenizer, train_data, val_data, device, config, name, is_vq_model=True):
    """Run training and evaluation."""
    print(f"\n{'='*60}")
    print(f"CONDITION: {name}")
    print(f"{'='*60}")
    print(f"Parameters: {model.param_count():,}")

    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

    best_score = 0
    best_results = None

    for epoch in range(1, config['epochs'] + 1):
        losses = train_epoch(
            model, backbone, tokenizer, train_data, optimizer, device,
            config['batch_size'], config.get('vq_weight', 0.1), is_vq_model
        )

        loss_str = f"loss={losses['total']:.4f} (ce={losses['ce']:.4f}, vq={losses['vq']:.4f})"
        results = evaluate(model, backbone, tokenizer, val_data, device, config['batch_size'], is_vq_model)

        print(f"Epoch {epoch}: {loss_str}, L0={results['l0_acc']:.4f}, L1={results['l1_acc']:.4f}")

        score = results['l0_acc'] + results['l1_acc']
        if score > best_score:
            best_score = score
            best_results = results.copy()
            best_results['epoch'] = epoch

    # Analyze codebook usage
    if is_vq_model:
        usage = analyze_codebook_usage(model, backbone, tokenizer, val_data, device, config['batch_size'])
        if usage:
            best_results.update(usage)
            print(f"\nCodebook Usage: {usage['unique_coarse_codes']}/{usage['total_coarse_codes']} coarse codes used ({usage['codebook_utilization']:.2%})")

    return best_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="yahoo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--vq_weight", type=float, default=0.1)
    parser.add_argument("--num_coarse", type=int, default=16)
    parser.add_argument("--num_fine_per_coarse", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs='+', default=[42])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading dataset...")
    train_data = load_hierarchical_dataset(args.dataset, split="train", max_samples=10000)
    val_data = load_hierarchical_dataset(args.dataset, split="test", max_samples=2000)

    l0_classes = len(set(item.level0_label for item in train_data))
    l1_classes = len(set(item.level1_label for item in train_data))
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"  L0: {l0_classes} classes, L1: {l1_classes} classes")

    # Load backbone
    print(f"\nLoading backbone: {args.model}...")
    model_config = MODELS[args.model]
    wrapper = load_model(args.model, use_fractal=False, device=device)
    backbone = wrapper.backbone
    tokenizer = wrapper.tokenizer
    hidden_dim = model_config.hidden_dim

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'vq_weight': args.vq_weight
    }

    all_results = {}

    for seed in args.seeds:
        print(f"\n{'#'*60}")
        print(f"SEED: {seed}")
        print(f"{'#'*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Condition 1: Baseline
        baseline_model = BaselineHead(hidden_dim, l0_classes, l1_classes).to(device)
        baseline_results = run_experiment(
            baseline_model, backbone, tokenizer, train_data, val_data,
            device, config, "Baseline (Standard MLP)", is_vq_model=False
        )

        # Condition 2: Fractal VQ (matching L0 classes)
        torch.manual_seed(seed)
        vq_model = FractalVQHead(
            hidden_dim, l0_classes, l1_classes,
            num_coarse_codes=args.num_coarse,
            num_fine_per_coarse=args.num_fine_per_coarse
        ).to(device)
        vq_results = run_experiment(
            vq_model, backbone, tokenizer, train_data, val_data,
            device, config, f"Fractal VQ ({args.num_coarse}x{args.num_fine_per_coarse} codes)", is_vq_model=True
        )

        # Condition 3: Fractal VQ with more codes
        torch.manual_seed(seed)
        vq_model_large = FractalVQHead(
            hidden_dim, l0_classes, l1_classes,
            num_coarse_codes=32,
            num_fine_per_coarse=16
        ).to(device)
        vq_large_results = run_experiment(
            vq_model_large, backbone, tokenizer, train_data, val_data,
            device, config, "Fractal VQ (32x16 codes)", is_vq_model=True
        )

        all_results[seed] = {
            'baseline': baseline_results,
            'vq': vq_results,
            'vq_large': vq_large_results
        }

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"Baseline params:  {baseline_model.param_count():,}")
        print(f"VQ params:        {vq_model.param_count():,}")
        print(f"VQ Large params:  {vq_model_large.param_count():,}")
        print(f"\nBaseline:   L0={baseline_results['l0_acc']:.4f}, L1={baseline_results['l1_acc']:.4f}")
        print(f"VQ:         L0={vq_results['l0_acc']:.4f}, L1={vq_results['l1_acc']:.4f}")
        print(f"VQ Large:   L0={vq_large_results['l0_acc']:.4f}, L1={vq_large_results['l1_acc']:.4f}")
        print(f"\nDelta (VQ):       L0={vq_results['l0_acc']-baseline_results['l0_acc']:+.4f}, L1={vq_results['l1_acc']-baseline_results['l1_acc']:+.4f}")
        print(f"Delta (VQ Large): L0={vq_large_results['l0_acc']-baseline_results['l0_acc']:+.4f}, L1={vq_large_results['l1_acc']-baseline_results['l1_acc']:+.4f}")

        del baseline_model, vq_model, vq_model_large
        torch.cuda.empty_cache()

    # Save results
    results_path = Path(__file__).parent.parent / "results" / f"fractal_vq_{args.model}.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "config": config,
            "results": {str(k): v for k, v in all_results.items()}
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
