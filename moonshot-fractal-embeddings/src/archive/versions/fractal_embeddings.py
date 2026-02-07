"""
Fractal Embeddings: Self-Similar Multi-Scale Representations
=============================================================

Core insight: Natural language has hierarchical structure (words → phrases →
sentences → paragraphs → documents). Current embeddings flatten this into
a single vector. Fractal embeddings preserve structure at multiple scales
using self-similar encoding.

Key innovations:
1. Recursive shared-weight encoder (same block at all scales)
2. Multi-scale pooling (extract embeddings at each iteration)
3. Nested Matryoshka structure (information at dim k exists at dim k/2)
4. Scale-aware contrastive loss (train at each granularity)

Based on:
- Matryoshka Representation Learning (nested dimensions)
- FractalMind (shared-weight recursive computation)
- Hyperbolic embeddings (natural for hierarchies)
- Renormalization group theory (scale invariance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FractalEmbeddingConfig:
    """Configuration for Fractal Embedding model."""
    vocab_size: int = 32000
    hidden_dim: int = 256  # Base hidden dimension
    num_heads: int = 8
    num_scales: int = 4  # Number of fractal scales
    scale_dim: int = 64  # Dimension per scale
    max_seq_len: int = 512
    dropout: float = 0.1
    pooling: str = "mean"  # mean, max, cls, attention

    @property
    def total_embed_dim(self) -> int:
        """Total embedding dimension = sum of all scales."""
        return self.scale_dim * self.num_scales


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # Expand mask from (B, T) to (B, 1, 1, T) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FractalBlock(nn.Module):
    """
    The core self-similar block applied at all scales.

    This is the "fractal" part - the same transformation repeated
    at every level of the hierarchy, like a fractal pattern.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

        # Scale-aware modulation (different scales can behave slightly differently)
        self.scale_gamma = nn.Parameter(torch.ones(1))
        self.scale_beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        x_input: torch.Tensor,
        scale_idx: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Current hidden state (B, T, D)
            x_input: Original input for residual injection
            scale_idx: Which scale we're at (for modulation)
            mask: Optional attention mask
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), mask)

        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))

        # Input injection (maintains connection to original)
        # Decays with scale (deeper = less input injection)
        injection_weight = 0.1 / (1 + scale_idx * 0.5)
        x = x + injection_weight * x_input

        # Scale modulation (learned per-scale adjustment)
        x = self.scale_gamma * x + self.scale_beta

        return x


class ScalePooling(nn.Module):
    """
    Pools hidden states to produce scale-specific embeddings.

    Different pooling strategies capture different aspects:
    - mean: Average information across all tokens
    - max: Most salient features
    - attention: Learned weighted combination
    """

    def __init__(self, dim: int, output_dim: int, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling

        # Project to scale-specific dimension
        self.proj = nn.Linear(dim, output_dim)

        if pooling == "attention":
            self.attn_weights = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Hidden states (B, T, D)
            mask: Padding mask (B, T)
        Returns:
            Pooled embedding (B, output_dim)
        """
        if mask is not None:
            # Expand mask for broadcasting
            mask_expanded = mask.unsqueeze(-1).float()
            x = x * mask_expanded

        if self.pooling == "mean":
            if mask is not None:
                pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        elif self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled = x.max(dim=1)[0]

        elif self.pooling == "attention":
            weights = self.attn_weights(x).squeeze(-1)  # (B, T)
            if mask is not None:
                weights = weights.masked_fill(~mask, float('-inf'))
            weights = F.softmax(weights, dim=-1)
            pooled = (x * weights.unsqueeze(-1)).sum(dim=1)

        elif self.pooling == "cls":
            pooled = x[:, 0]  # First token

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.proj(pooled)


class FractalEmbeddingModel(nn.Module):
    """
    Fractal Embedding Model: Self-similar multi-scale representations.

    Architecture:
    1. Input -> Embedding -> Hidden state
    2. Apply FractalBlock iteratively (shared weights)
    3. At each iteration, extract scale-specific embedding
    4. Final embedding = concatenation of all scales

    The key insight: Each scale captures information at a different
    level of abstraction, and the SAME encoder processes all scales
    (self-similarity).

    Scale 0: Coarse/global (after 1 iteration)
    Scale 1: Medium (after 2 iterations)
    Scale 2: Fine (after 3 iterations)
    ...

    This is like a fractal: zoom in and you see the same pattern.
    """

    def __init__(self, config: FractalEmbeddingConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02
        )

        # Single fractal block (shared across all scales)
        self.fractal_block = FractalBlock(
            config.hidden_dim,
            config.num_heads,
            config.dropout
        )

        # Scale-specific pooling heads
        self.scale_poolers = nn.ModuleList([
            ScalePooling(config.hidden_dim, config.scale_dim, config.pooling)
            for _ in range(config.num_scales)
        ])

        # Optional: learnable scale weights for combining
        self.scale_weights = nn.Parameter(torch.ones(config.num_scales))

        # Final normalization
        self.final_norm = nn.LayerNorm(config.total_embed_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_scales: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Token indices (B, T)
            attention_mask: Padding mask (B, T), 1 = real token, 0 = padding
            return_all_scales: If True, return individual scale embeddings

        Returns:
            Dictionary with:
            - 'embedding': Full fractal embedding (B, total_embed_dim)
            - 'scale_embeddings': List of per-scale embeddings (if return_all_scales)
            - 'nested_embeddings': Matryoshka-style nested prefixes
        """
        B, T = input_ids.shape

        # Embed tokens
        x = self.token_embed(input_ids) + self.pos_embed[:, :T, :]
        x_input = x.clone()

        # Collect embeddings at each scale
        scale_embeddings = []

        for scale_idx in range(self.config.num_scales):
            # Apply fractal block
            x = self.fractal_block(x, x_input, scale_idx, attention_mask)

            # Pool to get scale-specific embedding
            scale_embed = self.scale_poolers[scale_idx](x, attention_mask)
            scale_embeddings.append(scale_embed)

        # Concatenate all scales for full embedding
        full_embedding = torch.cat(scale_embeddings, dim=-1)
        full_embedding = self.final_norm(full_embedding)

        # Create nested (Matryoshka-style) embeddings
        # Each prefix contains all coarser scales
        nested_embeddings = []
        for i in range(1, self.config.num_scales + 1):
            nested = torch.cat(scale_embeddings[:i], dim=-1)
            nested_embeddings.append(nested)

        result = {
            'embedding': full_embedding,
            'nested_embeddings': nested_embeddings,
        }

        if return_all_scales:
            result['scale_embeddings'] = scale_embeddings

        return result

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_scales: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get embedding at specified granularity.

        Args:
            input_ids: Token indices
            attention_mask: Padding mask
            num_scales: How many scales to use (1 = coarsest, all = finest)
                       None = use all scales
        """
        result = self.forward(input_ids, attention_mask, return_all_scales=True)

        if num_scales is None:
            return result['embedding']
        else:
            # Return nested embedding up to specified scale
            return result['nested_embeddings'][num_scales - 1]

    def similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        scale: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute similarity between embeddings.

        Can compute at specific scale or aggregate across scales.
        """
        if scale is not None:
            # Extract specific scale
            start = scale * self.config.scale_dim
            end = start + self.config.scale_dim
            emb1 = emb1[..., start:end]
            emb2 = emb2[..., start:end]

        # Cosine similarity
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        return (emb1 * emb2).sum(dim=-1)


class FractalEmbeddingTrainer:
    """
    Trainer with multi-scale contrastive loss.

    Key innovation: We train at EACH scale, not just the final embedding.
    This ensures all scales capture meaningful information.
    """

    def __init__(
        self,
        model: FractalEmbeddingModel,
        lr: float = 1e-4,
        temperature: float = 0.05,
        scale_loss_weights: Optional[List[float]] = None,
    ):
        self.model = model
        self.temperature = temperature

        # Loss weights per scale (default: equal)
        if scale_loss_weights is None:
            self.scale_loss_weights = [1.0] * model.config.num_scales
        else:
            self.scale_loss_weights = scale_loss_weights

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        Embeddings from same class should be similar,
        embeddings from different classes should be dissimilar.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        sim_matrix = embeddings @ embeddings.T / self.temperature

        # Create label mask (1 if same class, 0 otherwise)
        labels = labels.unsqueeze(0)
        mask = (labels == labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(mask.size(0), device=mask.device)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)

        # For each anchor, sum over positives / sum over all
        pos_sum = (exp_sim * mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1) - torch.diag(exp_sim)

        # Avoid division by zero
        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)

        # Only compute for anchors that have positives
        valid_mask = mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return loss

    def multi_scale_loss(
        self,
        scale_embeddings: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss at each scale.
        """
        total_loss = 0.0
        metrics = {}

        for i, (scale_emb, weight) in enumerate(zip(scale_embeddings, self.scale_loss_weights)):
            scale_loss = self.contrastive_loss(scale_emb, labels)
            total_loss = total_loss + weight * scale_loss
            metrics[f'loss_scale_{i}'] = scale_loss.item()

        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        result = self.model(input_ids, attention_mask, return_all_scales=True)

        # Multi-scale contrastive loss
        loss, metrics = self.multi_scale_loss(result['scale_embeddings'], labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return metrics


# =============================================================================
# SIMPLE TESTS
# =============================================================================

def test_fractal_embeddings():
    """Test the fractal embedding architecture."""
    print("=" * 60)
    print("Testing Fractal Embedding Architecture")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    config = FractalEmbeddingConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=4,
        num_scales=4,
        scale_dim=32,
        max_seq_len=64,
    )

    model = FractalEmbeddingModel(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Total embedding dim: {config.total_embed_dim}")

    # Test input
    batch_size = 8
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)

    # Forward pass
    print("\nForward pass...")
    result = model(input_ids, attention_mask, return_all_scales=True)

    print(f"  Full embedding shape: {result['embedding'].shape}")
    print(f"  Number of scales: {len(result['scale_embeddings'])}")
    for i, scale_emb in enumerate(result['scale_embeddings']):
        print(f"    Scale {i}: {scale_emb.shape}")

    print(f"\n  Nested embeddings (Matryoshka-style):")
    for i, nested in enumerate(result['nested_embeddings']):
        print(f"    Prefix {i+1} scales: {nested.shape}")

    # Test similarity
    print("\nSimilarity computation...")
    emb1 = result['embedding'][0:1]
    emb2 = result['embedding'][1:2]

    full_sim = model.similarity(emb1, emb2)
    print(f"  Full similarity: {full_sim.item():.4f}")

    for scale in range(config.num_scales):
        scale_sim = model.similarity(emb1, emb2, scale=scale)
        print(f"  Scale {scale} similarity: {scale_sim.item():.4f}")

    # Test variable-scale embedding
    print("\nVariable-scale embeddings...")
    for num_scales in range(1, config.num_scales + 1):
        emb = model.get_embedding(input_ids, attention_mask, num_scales=num_scales)
        print(f"  {num_scales} scale(s): {emb.shape}")

    # Test trainer
    print("\nTraining test...")
    trainer = FractalEmbeddingTrainer(model, lr=1e-3)
    labels = torch.randint(0, 4, (batch_size,)).to(device)  # 4 classes

    for step in range(3):
        metrics = trainer.train_step(input_ids, attention_mask, labels)
        print(f"  Step {step+1}: total_loss={metrics['total_loss']:.4f}, " +
              ", ".join(f"s{i}={metrics[f'loss_scale_{i}']:.3f}"
                       for i in range(config.num_scales)))

    print("\nFractal Embedding test passed!")
    return model, config


if __name__ == '__main__':
    test_fractal_embeddings()
