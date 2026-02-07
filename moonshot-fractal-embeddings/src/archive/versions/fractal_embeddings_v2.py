"""
Fractal Embeddings V2: Hierarchy-Aware Multi-Scale Representations
===================================================================

Improvements over V1:
1. Hierarchy-aware contrastive loss (train different scales for different granularities)
2. Hyperbolic components (natural for tree-like structures)
3. Scale-specific training objectives
4. Iterative refinement with convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FractalEmbeddingConfigV2:
    """Configuration for Fractal Embedding V2."""
    vocab_size: int = 32000
    hidden_dim: int = 256
    num_heads: int = 8
    num_scales: int = 4
    scale_dim: int = 64
    max_seq_len: int = 512
    dropout: float = 0.1
    use_hyperbolic: bool = True  # Use hyperbolic geometry for hierarchy
    curvature: float = 1.0  # Hyperbolic curvature

    @property
    def total_embed_dim(self) -> int:
        return self.scale_dim * self.num_scales


# =============================================================================
# HYPERBOLIC OPERATIONS
# =============================================================================

class HyperbolicOps:
    """Operations in the Poincare ball model of hyperbolic space."""

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Mobius addition in the Poincare ball."""
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
        return num / (denom + 1e-8)

    @staticmethod
    def exp_map(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Exponential map from tangent space at origin to Poincare ball."""
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sqrt_c = math.sqrt(c)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)

    @staticmethod
    def log_map(y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Logarithmic map from Poincare ball to tangent space at origin."""
        y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-8, max=1-1e-5)
        sqrt_c = math.sqrt(c)
        return torch.arctanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Hyperbolic distance in the Poincare ball."""
        diff = HyperbolicOps.mobius_add(-x, y, c)
        diff_norm = diff.norm(dim=-1).clamp(min=1e-8, max=1-1e-5)
        return 2 / math.sqrt(c) * torch.arctanh(math.sqrt(c) * diff_norm)

    @staticmethod
    def project_to_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """Project points to inside the Poincare ball."""
        max_norm = (1 - eps) / math.sqrt(c)
        norm = x.norm(dim=-1, keepdim=True)
        return x * (max_norm / norm.clamp(min=max_norm))


# =============================================================================
# FRACTAL BLOCK V2
# =============================================================================

class MultiHeadAttentionV2(nn.Module):
    """Attention with optional hyperbolic distance-based attention."""

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
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FractalBlockV2(nn.Module):
    """
    Enhanced fractal block with:
    1. Scale-dependent transformation
    2. Convergence checking
    3. Optional hyperbolic projection
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_hyperbolic: bool = False,
        curvature: float = 1.0,
    ):
        super().__init__()

        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionV2(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Scale-specific modulation
        self.scale_proj = nn.Linear(dim, dim)

        # Convergence gate
        self.convergence_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_input: torch.Tensor,
        scale_idx: int,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_new: Updated hidden state
            convergence: Per-sample convergence probability
        """
        # Self-attention
        x = x + self.attn(self.norm1(x), mask)

        # Feed-forward
        x = x + self.ffn(self.norm2(x))

        # Input injection (decays with scale)
        injection_weight = 0.1 / (1 + scale_idx * 0.3)
        x = x + injection_weight * x_input

        # Scale-specific transformation
        scale_factor = 1.0 / math.sqrt(1 + scale_idx)
        x = x + scale_factor * self.scale_proj(x)

        # Convergence check
        convergence = self.convergence_gate(x.mean(dim=1)).squeeze(-1)

        # Optional hyperbolic projection
        if self.use_hyperbolic:
            x = HyperbolicOps.project_to_ball(x, self.curvature)

        return x, convergence


# =============================================================================
# SCALE POOLING WITH HYPERBOLIC OPTION
# =============================================================================

class ScalePoolingV2(nn.Module):
    """Pooling with optional hyperbolic centroid."""

    def __init__(
        self,
        dim: int,
        output_dim: int,
        use_hyperbolic: bool = False,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature

        self.proj = nn.Linear(dim, output_dim)
        self.attn_weights = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention pooling
        weights = self.attn_weights(x).squeeze(-1)
        if mask is not None:
            weights = weights.masked_fill(~mask.bool(), float('-inf'))
        weights = F.softmax(weights, dim=-1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)

        # Project
        out = self.proj(pooled)

        # Optional hyperbolic projection
        if self.use_hyperbolic:
            out = HyperbolicOps.project_to_ball(out, self.curvature)

        return out


# =============================================================================
# FRACTAL EMBEDDING MODEL V2
# =============================================================================

class FractalEmbeddingModelV2(nn.Module):
    """
    Fractal Embedding V2 with hierarchy-aware multi-scale representations.

    Key improvements:
    1. Different scales explicitly target different hierarchy levels
    2. Optional hyperbolic geometry for tree-like structures
    3. Convergence-aware iteration
    4. Better scale separation
    """

    def __init__(self, config: FractalEmbeddingConfigV2):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02
        )

        # Single fractal block (shared)
        self.fractal_block = FractalBlockV2(
            config.hidden_dim,
            config.num_heads,
            config.dropout,
            config.use_hyperbolic,
            config.curvature,
        )

        # Scale-specific pooling heads
        self.scale_poolers = nn.ModuleList([
            ScalePoolingV2(
                config.hidden_dim,
                config.scale_dim,
                config.use_hyperbolic,
                config.curvature,
            )
            for _ in range(config.num_scales)
        ])

        # Scale orthogonality projection (encourages different scales to capture different info)
        self.scale_ortho = nn.Parameter(torch.eye(config.num_scales) * 0.1)

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
        return_convergence: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape

        # Embed
        x = self.token_embed(input_ids) + self.pos_embed[:, :T, :]
        x_input = x.clone()

        # Collect embeddings at each scale
        scale_embeddings = []
        convergence_probs = []

        for scale_idx in range(self.config.num_scales):
            x, convergence = self.fractal_block(x, x_input, scale_idx, attention_mask)
            scale_embed = self.scale_poolers[scale_idx](x, attention_mask)
            scale_embeddings.append(scale_embed)
            convergence_probs.append(convergence)

        # Concatenate all scales
        full_embedding = torch.cat(scale_embeddings, dim=-1)
        full_embedding = self.final_norm(full_embedding)

        # Nested embeddings (Matryoshka-style)
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

        if return_convergence:
            result['convergence'] = torch.stack(convergence_probs, dim=1)

        return result

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_scales: Optional[int] = None,
    ) -> torch.Tensor:
        result = self.forward(input_ids, attention_mask, return_all_scales=True)
        if num_scales is None:
            return result['embedding']
        return result['nested_embeddings'][num_scales - 1]

    def similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        scale: Optional[int] = None,
    ) -> torch.Tensor:
        if scale is not None:
            start = scale * self.config.scale_dim
            end = start + self.config.scale_dim
            emb1 = emb1[..., start:end]
            emb2 = emb2[..., start:end]

        if self.config.use_hyperbolic:
            # Hyperbolic distance (negated for similarity)
            return -HyperbolicOps.distance(emb1, emb2, self.config.curvature)
        else:
            # Cosine similarity
            emb1 = F.normalize(emb1, dim=-1)
            emb2 = F.normalize(emb2, dim=-1)
            return (emb1 * emb2).sum(dim=-1)


# =============================================================================
# HIERARCHY-AWARE TRAINER
# =============================================================================

class HierarchyAwareTrainer:
    """
    Trains fractal embeddings with hierarchy-aware loss.

    Key insight: Different scales should be optimized for different
    hierarchy levels.

    Scale 0 (coarse): Super-category discrimination
    Scale 1: Category discrimination
    Scale 2+: Instance discrimination
    """

    def __init__(
        self,
        model: FractalEmbeddingModelV2,
        lr: float = 1e-4,
        temperature: float = 0.05,
        ortho_weight: float = 0.1,
    ):
        self.model = model
        self.temperature = temperature
        self.ortho_weight = ortho_weight

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def hierarchy_contrastive_loss(
        self,
        scale_embeddings: List[torch.Tensor],
        super_cat_labels: torch.Tensor,
        cat_labels: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hierarchy-aware contrastive loss.

        Scale 0: Group by super-category
        Scale 1: Group by category
        Scale 2+: Group by instance
        """
        total_loss = 0.0
        metrics = {}

        num_scales = len(scale_embeddings)

        for scale_idx, scale_emb in enumerate(scale_embeddings):
            # Choose label based on scale
            if scale_idx == 0:
                labels = super_cat_labels
                scale_name = "super"
            elif scale_idx == 1:
                labels = cat_labels
                scale_name = "cat"
            else:
                labels = instance_labels
                scale_name = "inst"

            # Compute contrastive loss
            if self.model.config.use_hyperbolic:
                # Hyperbolic contrastive loss
                loss = self._hyperbolic_contrastive(scale_emb, labels)
            else:
                loss = self._euclidean_contrastive(scale_emb, labels)

            total_loss = total_loss + loss
            metrics[f'loss_scale_{scale_idx}_{scale_name}'] = loss.item()

        # Orthogonality regularization (encourage different scales to be different)
        ortho_loss = self._orthogonality_loss(scale_embeddings)
        total_loss = total_loss + self.ortho_weight * ortho_loss
        metrics['ortho_loss'] = ortho_loss.item()

        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics

    def _euclidean_contrastive(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        sim_matrix = embeddings @ embeddings.T / self.temperature

        # Create positive mask
        labels = labels.unsqueeze(0)
        mask = (labels == labels.T).float()
        mask = mask - torch.eye(mask.size(0), device=mask.device)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1) - torch.diag(exp_sim)

        valid_mask = mask.sum(dim=1) > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
        return loss[valid_mask].mean()

    def _hyperbolic_contrastive(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Contrastive loss using hyperbolic distance."""
        B = embeddings.size(0)
        c = self.model.config.curvature

        # Compute pairwise distances
        dist_matrix = torch.zeros(B, B, device=embeddings.device)
        for i in range(B):
            for j in range(B):
                if i != j:
                    dist_matrix[i, j] = HyperbolicOps.distance(
                        embeddings[i:i+1], embeddings[j:j+1], c
                    )

        # Convert to similarity (negative distance)
        sim_matrix = -dist_matrix / self.temperature

        # Same contrastive loss
        labels = labels.unsqueeze(0)
        mask = (labels == labels.T).float()
        mask = mask - torch.eye(mask.size(0), device=mask.device)

        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1) - torch.diag(exp_sim)

        valid_mask = mask.sum(dim=1) > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
        return loss[valid_mask].mean()

    def _orthogonality_loss(
        self,
        scale_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """Encourage different scales to capture different information."""
        # Stack scale means
        scale_means = torch.stack([e.mean(dim=0) for e in scale_embeddings])
        scale_means = F.normalize(scale_means, dim=-1)

        # Gram matrix
        gram = scale_means @ scale_means.T

        # Penalize off-diagonal elements (want orthogonality)
        eye = torch.eye(gram.size(0), device=gram.device)
        ortho_loss = ((gram - eye) ** 2).mean()

        return ortho_loss

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        super_cat_labels: torch.Tensor,
        cat_labels: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        result = self.model(input_ids, attention_mask, return_all_scales=True)

        loss, metrics = self.hierarchy_contrastive_loss(
            result['scale_embeddings'],
            super_cat_labels,
            cat_labels,
            instance_labels,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return metrics


# =============================================================================
# TEST
# =============================================================================

def test_fractal_embeddings_v2():
    print("=" * 60)
    print("Testing Fractal Embeddings V2")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    config = FractalEmbeddingConfigV2(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=4,
        num_scales=4,
        scale_dim=32,
        max_seq_len=64,
        use_hyperbolic=False,  # Start without hyperbolic for simplicity
    )

    model = FractalEmbeddingModelV2(config).to(device)

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
    result = model(input_ids, attention_mask, return_all_scales=True, return_convergence=True)

    print(f"  Full embedding: {result['embedding'].shape}")
    print(f"  Convergence: {result['convergence'].shape}")
    print(f"  Convergence values: {result['convergence'][0].tolist()}")

    for i, emb in enumerate(result['scale_embeddings']):
        print(f"  Scale {i}: {emb.shape}")

    # Test trainer
    print("\nTrainer test...")
    trainer = HierarchyAwareTrainer(model, lr=1e-3)

    super_cat = torch.randint(0, 4, (batch_size,)).to(device)
    cat = torch.randint(0, 16, (batch_size,)).to(device)
    instance = torch.arange(batch_size).to(device)

    for step in range(3):
        metrics = trainer.train_step(input_ids, attention_mask, super_cat, cat, instance)
        loss_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if 'loss' in k)
        print(f"  Step {step+1}: {loss_str}")

    print("\nFractal Embeddings V2 test passed!")
    return model


if __name__ == '__main__':
    test_fractal_embeddings_v2()
