"""
Hyperbolic Fractal Head (HFH) - V6 Architecture

Combines:
1. Hyperbolic geometry with learnable curvature per scale
2. Fractal (self-similar) block structure
3. Poincare for fine-grained, Lorentz for coarse scales
4. Hierarchical distance-based classification
5. Correlation dimension regularizer for fractal preservation
6. Wavelet-style multi-scale filtering for fast convergence

Based on:
- HiM (Hyperbolic Hierarchical Mamba): arxiv.org/abs/2505.18973
- WaveletGPT: arxiv.org/abs/2409.12924
- Correlation Dimension: arxiv.org/abs/2510.21258
- Our V5 fractal embeddings breakthrough
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


# ============================================================================
# HYPERBOLIC GEOMETRY OPERATIONS
# ============================================================================

class HyperbolicOps:
    """Hyperbolic geometry operations for Poincare and Lorentz models."""

    @staticmethod
    def poincare_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Mobius addition in Poincare ball with curvature c."""
        c = c.abs().clamp(min=1e-6)
        sqrt_c = c.sqrt()

        x_norm_sq = (x * x).sum(dim=-1, keepdim=True).clamp(max=1 - 1e-5)
        y_norm_sq = (y * y).sum(dim=-1, keepdim=True).clamp(max=1 - 1e-5)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_norm_sq * y_norm_sq

        return num / denom.clamp(min=1e-6)

    @staticmethod
    def poincare_dist(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance in Poincare ball."""
        c = c.abs().clamp(min=1e-6)
        sqrt_c = c.sqrt()

        diff = x - y
        diff_norm_sq = (diff * diff).sum(dim=-1).clamp(min=1e-10)

        x_norm_sq = (x * x).sum(dim=-1).clamp(max=1 - 1e-5)
        y_norm_sq = (y * y).sum(dim=-1).clamp(max=1 - 1e-5)

        num = 2 * diff_norm_sq
        denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)

        arg = 1 + num / denom.clamp(min=1e-6)
        return (2 / sqrt_c) * torch.acosh(arg.clamp(min=1.0 + 1e-6))

    @staticmethod
    def exp_map_poincare(v: torch.Tensor, c: torch.Tensor, base: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Exponential map from tangent space to Poincare ball."""
        c = c.abs().clamp(min=1e-6)
        sqrt_c = c.sqrt()

        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        result = torch.tanh(sqrt_c * v_norm / 2) * v / (sqrt_c * v_norm)

        if base is not None:
            result = HyperbolicOps.poincare_add(base, result, c)

        # Project to ball
        result_norm = result.norm(dim=-1, keepdim=True)
        max_norm = (1 - 1e-5) / sqrt_c
        result = result * (max_norm / result_norm.clamp(min=1e-6)).clamp(max=1.0)

        return result

    @staticmethod
    def log_map_poincare(x: torch.Tensor, c: torch.Tensor, base: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logarithmic map from Poincare ball to tangent space."""
        c = c.abs().clamp(min=1e-6)
        sqrt_c = c.sqrt()

        if base is not None:
            # Transport to origin first
            neg_base = -base
            x = HyperbolicOps.poincare_add(neg_base, x, c)

        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-10, max=(1 - 1e-5) / sqrt_c)
        return (2 / sqrt_c) * torch.atanh(sqrt_c * x_norm) * x / x_norm

    @staticmethod
    def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Minkowski inner product for Lorentz model."""
        # x, y have shape (..., d+1) where first component is time
        return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)

    @staticmethod
    def lorentz_dist(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance in Lorentz model."""
        c = c.abs().clamp(min=1e-6)
        inner = -HyperbolicOps.lorentz_inner(x, y).squeeze(-1)
        return torch.acosh(inner.clamp(min=1.0 + 1e-6)) / c.sqrt()

    @staticmethod
    def exp_map_lorentz(v: torch.Tensor, c: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Exponential map in Lorentz model."""
        c = c.abs().clamp(min=1e-6)
        sqrt_c = c.sqrt()

        v_inner = HyperbolicOps.lorentz_inner(v, v).clamp(min=1e-10)
        v_norm = v_inner.sqrt()

        return torch.cosh(sqrt_c * v_norm) * base + torch.sinh(sqrt_c * v_norm) * v / v_norm

    @staticmethod
    def project_to_lorentz(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Project to Lorentz hyperboloid."""
        c = c.abs().clamp(min=1e-6)
        spatial = x[..., 1:]
        spatial_norm_sq = (spatial * spatial).sum(dim=-1, keepdim=True)
        time = torch.sqrt(1 / c + spatial_norm_sq)
        return torch.cat([time, spatial], dim=-1)


# ============================================================================
# WAVELET-STYLE MULTI-SCALE FILTERING
# ============================================================================

class WaveletFilter(nn.Module):
    """Causal multi-scale wavelet-style filtering for scale specialization."""

    def __init__(self, dim: int, num_scales: int = 4):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales

        # Learnable scale-specific filters (inspired by WaveletGPT)
        self.scale_filters = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim) * 0.02)
            for _ in range(num_scales)
        ])

        # Scale mixing weights
        self.scale_gates = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Apply multi-scale filtering, return per-scale outputs."""
        outputs = []
        gates = F.softmax(self.scale_gates, dim=0)

        for i, filt in enumerate(self.scale_filters):
            # Each scale gets progressively smoother/coarser
            scale_factor = 2 ** i

            # Apply filter
            filtered = F.linear(x, filt)

            # Scale-specific normalization
            filtered = F.layer_norm(filtered, [self.dim])

            outputs.append(filtered * gates[i])

        return outputs


# ============================================================================
# FRACTAL BLOCK (Self-Similar Transformation)
# ============================================================================

class FractalBlock(nn.Module):
    """Self-similar transformation block applied at each scale."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.dim = dim

        # Self-attention for capturing within-scale relationships
        self.self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim) or (batch, seq, dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dim
            squeeze = True
        else:
            squeeze = False

        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        x = self.norm2(x + self.ffn(x))

        if squeeze:
            x = x.squeeze(1)

        return x


# ============================================================================
# HYPERBOLIC SCALE HEAD
# ============================================================================

class HyperbolicScaleHead(nn.Module):
    """Single scale head with hyperbolic projection."""

    def __init__(
        self,
        input_dim: int,
        scale_dim: int,
        num_classes: int,
        manifold: str = "poincare",  # "poincare" or "lorentz"
        init_curvature: float = 1.0,
    ):
        super().__init__()
        self.scale_dim = scale_dim
        self.manifold = manifold

        # Learnable curvature (negative for hyperbolic)
        self.log_curvature = nn.Parameter(torch.tensor(math.log(init_curvature)))

        # Projection to scale dimension (in tangent space)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, scale_dim * 2),
            nn.GELU(),
            nn.Linear(scale_dim * 2, scale_dim),
        )

        # Fractal block for self-similar processing
        self.fractal = FractalBlock(scale_dim)

        # Hyperbolic prototypes for each class
        if manifold == "lorentz":
            # Lorentz: d+1 dimensions (time + space)
            self.prototypes = nn.Parameter(torch.randn(num_classes, scale_dim + 1) * 0.1)
        else:
            # Poincare: d dimensions
            self.prototypes = nn.Parameter(torch.randn(num_classes, scale_dim) * 0.1)

        # Euclidean classifier as stabilizer
        self.euclidean_head = nn.Linear(scale_dim, num_classes)

    @property
    def curvature(self) -> torch.Tensor:
        return self.log_curvature.exp().clamp(min=1e-4, max=10.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch, input_dim)

        Returns:
            hyp_logits: Hyperbolic distance-based logits
            euc_logits: Euclidean classifier logits (stabilizer)
            embedding: Hyperbolic embedding for this scale
        """
        # Project to tangent space
        tangent = self.projector(x)

        # Apply fractal transformation
        tangent = self.fractal(tangent)

        c = self.curvature

        if self.manifold == "poincare":
            # Map to Poincare ball
            embedding = HyperbolicOps.exp_map_poincare(tangent, c)

            # Normalize prototypes to ball
            proto_norm = self.prototypes.norm(dim=-1, keepdim=True)
            max_norm = (1 - 1e-5) / c.sqrt()
            prototypes = self.prototypes * (max_norm / proto_norm.clamp(min=1e-6)).clamp(max=1.0)

            # Distance-based logits
            dists = torch.stack([
                HyperbolicOps.poincare_dist(embedding, prototypes[i:i+1].expand_as(embedding), c)
                for i in range(prototypes.shape[0])
            ], dim=-1)

        else:  # lorentz
            # Pad tangent to d+1 for Lorentz
            tangent_padded = F.pad(tangent, (1, 0), value=0)

            # Create base point on hyperboloid
            base = torch.zeros_like(tangent_padded)
            base[..., 0] = 1 / c.sqrt()

            # Map to hyperboloid
            embedding = HyperbolicOps.project_to_lorentz(tangent_padded, c)

            # Project prototypes to hyperboloid
            prototypes = HyperbolicOps.project_to_lorentz(self.prototypes, c)

            # Distance-based logits
            dists = torch.stack([
                HyperbolicOps.lorentz_dist(embedding, prototypes[i:i+1].expand_as(embedding), c)
                for i in range(prototypes.shape[0])
            ], dim=-1)

        # Convert distances to logits (smaller distance = higher logit)
        hyp_logits = -dists

        # Euclidean stabilizer (use tangent space representation)
        euc_logits = self.euclidean_head(tangent)

        return hyp_logits, euc_logits, embedding


# ============================================================================
# CORRELATION DIMENSION ESTIMATOR
# ============================================================================

class CorrelationDimensionEstimator(nn.Module):
    """
    Estimates correlation dimension of embedding trajectories.
    Used as regularizer to preserve natural language fractal structure.

    Based on: arxiv.org/abs/2510.21258
    Target: Natural language ~6.5, code ~5, degenerate <2
    """

    def __init__(self, target_dim: float = 6.5, epsilon_range: Tuple[float, float] = (0.1, 2.0)):
        super().__init__()
        self.target_dim = target_dim
        self.eps_min, self.eps_max = epsilon_range

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Estimate correlation dimension and return regularization loss.

        Args:
            embeddings: (batch, dim) or (batch, num_scales, dim)

        Returns:
            loss: Penalty for deviating from target correlation dimension
        """
        if embeddings.dim() == 3:
            # Flatten scales
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        batch_size = embeddings.shape[0]
        if batch_size < 10:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings)

        # Remove diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        dists = dists[mask].reshape(batch_size, batch_size - 1)

        # Estimate correlation dimension at multiple scales
        eps_values = torch.logspace(
            math.log10(self.eps_min),
            math.log10(self.eps_max),
            steps=10,
            device=embeddings.device
        )

        log_eps = torch.log(eps_values)
        log_C = []

        for eps in eps_values:
            # Correlation integral: fraction of pairs within distance eps
            C = (dists < eps).float().mean()
            log_C.append(torch.log(C.clamp(min=1e-10)))

        log_C = torch.stack(log_C)

        # Linear regression to estimate dimension (slope of log-log plot)
        # dim = d(log C) / d(log eps)
        valid_mask = torch.isfinite(log_C) & (log_C > -20)
        if valid_mask.sum() < 3:
            return torch.tensor(0.0, device=embeddings.device)

        log_eps_valid = log_eps[valid_mask]
        log_C_valid = log_C[valid_mask]

        # Simple linear regression
        n = log_eps_valid.shape[0]
        sum_x = log_eps_valid.sum()
        sum_y = log_C_valid.sum()
        sum_xy = (log_eps_valid * log_C_valid).sum()
        sum_xx = (log_eps_valid * log_eps_valid).sum()

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + 1e-10)

        # Penalty for deviating from target dimension
        dim_loss = (slope - self.target_dim) ** 2

        return dim_loss


# ============================================================================
# MAIN HYPERBOLIC FRACTAL HEAD (HFH)
# ============================================================================

@dataclass
class HFHConfig:
    """Configuration for Hyperbolic Fractal Head."""
    input_dim: int = 1024
    num_scales: int = 4
    scale_dim: int = 64
    num_l0_classes: int = 10  # Coarse classes
    num_l1_classes: int = 60  # Fine classes

    # Scale-specific manifolds (coarse=lorentz for stability, fine=poincare for detail)
    manifolds: Tuple[str, ...] = ("lorentz", "lorentz", "poincare", "poincare")

    # Initial curvatures (can be different per scale)
    init_curvatures: Tuple[float, ...] = (0.5, 1.0, 1.0, 2.0)

    # Loss weights
    hyperbolic_weight: float = 1.0
    euclidean_weight: float = 0.3  # Stabilizer
    corr_dim_weight: float = 0.1  # Fractal preservation

    # Target correlation dimension (natural language ~6.5)
    target_corr_dim: float = 6.5

    # Wavelet filtering
    use_wavelet: bool = True


class HyperbolicFractalHead(nn.Module):
    """
    Hyperbolic Fractal Head (HFH) - V6 Architecture

    Combines hyperbolic geometry with fractal structure for hierarchical embeddings.
    Each scale operates in its own hyperbolic manifold with learnable curvature.
    """

    def __init__(self, config: HFHConfig):
        super().__init__()
        self.config = config

        # Wavelet-style multi-scale filtering
        if config.use_wavelet:
            self.wavelet = WaveletFilter(config.input_dim, config.num_scales)
        else:
            self.wavelet = None

        # Scale-specific hyperbolic heads
        # Coarse scales predict L0, fine scales predict L1
        self.scale_heads = nn.ModuleList()
        for i in range(config.num_scales):
            manifold = config.manifolds[i] if i < len(config.manifolds) else "poincare"
            curvature = config.init_curvatures[i] if i < len(config.init_curvatures) else 1.0

            # First half of scales for L0, second half for L1
            if i < config.num_scales // 2:
                num_classes = config.num_l0_classes
            else:
                num_classes = config.num_l1_classes

            self.scale_heads.append(HyperbolicScaleHead(
                input_dim=config.input_dim,
                scale_dim=config.scale_dim,
                num_classes=num_classes,
                manifold=manifold,
                init_curvature=curvature,
            ))

        # Cross-scale fusion for final predictions
        # Account for Lorentz having d+1 dims (we keep spatial only = scale_dim)
        n_coarse = config.num_scales // 2
        n_fine = config.num_scales - n_coarse

        self.l0_fusion = nn.Sequential(
            nn.Linear(config.scale_dim * n_coarse, config.scale_dim),
            nn.GELU(),
            nn.Linear(config.scale_dim, config.num_l0_classes),
        )

        self.l1_fusion = nn.Sequential(
            nn.Linear(config.scale_dim * n_fine, config.scale_dim),
            nn.GELU(),
            nn.Linear(config.scale_dim, config.num_l1_classes),
        )

        # Correlation dimension regularizer
        self.corr_dim = CorrelationDimensionEstimator(target_dim=config.target_corr_dim)

        # Learned fusion temperatures
        self.hyp_temp = nn.Parameter(torch.tensor(1.0))
        self.euc_temp = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict:
        """
        Forward pass through HFH.

        Args:
            x: Input features from backbone (batch, input_dim)
            return_embeddings: Whether to return scale embeddings

        Returns:
            dict with:
                - l0_logits: Coarse classification logits
                - l1_logits: Fine classification logits
                - scale_curvatures: Learned curvatures per scale
                - embeddings (optional): Per-scale hyperbolic embeddings
        """
        batch_size = x.shape[0]

        # Apply wavelet filtering if enabled
        if self.wavelet is not None:
            filtered_inputs = self.wavelet(x)
        else:
            filtered_inputs = [x] * self.config.num_scales

        # Process each scale
        all_hyp_logits = []
        all_euc_logits = []
        all_embeddings = []
        curvatures = []

        for i, (scale_head, scale_input) in enumerate(zip(self.scale_heads, filtered_inputs)):
            hyp_logits, euc_logits, embedding = scale_head(scale_input)
            all_hyp_logits.append(hyp_logits)
            all_euc_logits.append(euc_logits)
            all_embeddings.append(embedding)
            curvatures.append(scale_head.curvature.item())

        # Aggregate L0 predictions (from coarse scales)
        n_coarse = self.config.num_scales // 2
        l0_hyp = torch.stack(all_hyp_logits[:n_coarse], dim=1).mean(dim=1)
        l0_euc = torch.stack(all_euc_logits[:n_coarse], dim=1).mean(dim=1)

        # Fused L0 from embeddings (strip time dim for Lorentz)
        coarse_embeds_list = []
        for i, emb in enumerate(all_embeddings[:n_coarse]):
            manifold = self.config.manifolds[i] if i < len(self.config.manifolds) else "poincare"
            if manifold == "lorentz":
                coarse_embeds_list.append(emb[..., 1:])  # Remove time dim
            else:
                coarse_embeds_list.append(emb)
        coarse_embeds = torch.cat(coarse_embeds_list, dim=-1)
        l0_fused = self.l0_fusion(coarse_embeds)

        # Aggregate L1 predictions (from fine scales)
        l1_hyp = torch.stack(all_hyp_logits[n_coarse:], dim=1).mean(dim=1)
        l1_euc = torch.stack(all_euc_logits[n_coarse:], dim=1).mean(dim=1)

        # Fused L1 from embeddings (strip time dim for Lorentz)
        fine_embeds_list = []
        for i, emb in enumerate(all_embeddings[n_coarse:]):
            idx = n_coarse + i
            manifold = self.config.manifolds[idx] if idx < len(self.config.manifolds) else "poincare"
            if manifold == "lorentz":
                fine_embeds_list.append(emb[..., 1:])  # Remove time dim
            else:
                fine_embeds_list.append(emb)
        fine_embeds = torch.cat(fine_embeds_list, dim=-1)
        l1_fused = self.l1_fusion(fine_embeds)

        # Temperature-scaled combination
        hyp_temp = self.hyp_temp.abs().clamp(min=0.1)
        euc_temp = self.euc_temp.abs().clamp(min=0.1)

        l0_logits = (
            self.config.hyperbolic_weight * l0_hyp / hyp_temp +
            self.config.euclidean_weight * l0_euc / euc_temp +
            l0_fused
        )

        l1_logits = (
            self.config.hyperbolic_weight * l1_hyp / hyp_temp +
            self.config.euclidean_weight * l1_euc / euc_temp +
            l1_fused
        )

        result = {
            "l0_logits": l0_logits,
            "l1_logits": l1_logits,
            "scale_curvatures": curvatures,
        }

        if return_embeddings:
            result["embeddings"] = all_embeddings

        return result

    def compute_loss(
        self,
        outputs: dict,
        l0_labels: torch.Tensor,
        l1_labels: torch.Tensor,
        embeddings_for_corr_dim: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            outputs: Forward pass outputs
            l0_labels: Coarse labels (batch,)
            l1_labels: Fine labels (batch,)
            embeddings_for_corr_dim: Optional embeddings for correlation dimension loss

        Returns:
            dict with loss components
        """
        # Classification losses
        l0_loss = F.cross_entropy(outputs["l0_logits"], l0_labels)
        l1_loss = F.cross_entropy(outputs["l1_logits"], l1_labels)

        # Correlation dimension regularizer
        if embeddings_for_corr_dim is not None and self.config.corr_dim_weight > 0:
            corr_loss = self.corr_dim(embeddings_for_corr_dim)
        else:
            corr_loss = torch.tensor(0.0, device=l0_labels.device)

        # Total loss
        total_loss = (
            l0_loss +
            l1_loss +
            self.config.corr_dim_weight * corr_loss
        )

        return {
            "total": total_loss,
            "l0": l0_loss,
            "l1": l1_loss,
            "corr_dim": corr_loss,
        }


# ============================================================================
# FULL MODEL WRAPPER
# ============================================================================

class HyperbolicFractalModel(nn.Module):
    """
    Full model: Frozen backbone + Hyperbolic Fractal Head
    """

    def __init__(
        self,
        backbone: nn.Module,
        hfh_config: HFHConfig,
        pooling: str = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.hfh = HyperbolicFractalHead(hfh_config)
        self.pooling = pooling

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict:
        # Get backbone hidden states
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )

        # Pool hidden states
        hidden = outputs.last_hidden_state

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        elif self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = hidden.mean(dim=1)

        # Apply HFH
        return self.hfh(pooled, return_embeddings=True)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Hyperbolic Fractal Head (HFH)...")

    # Create config
    config = HFHConfig(
        input_dim=1024,
        num_scales=4,
        scale_dim=64,
        num_l0_classes=10,
        num_l1_classes=60,
    )

    # Create model
    model = HyperbolicFractalHead(config)

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.input_dim)

    outputs = model(x, return_embeddings=True)

    print(f"L0 logits shape: {outputs['l0_logits'].shape}")
    print(f"L1 logits shape: {outputs['l1_logits'].shape}")
    print(f"Scale curvatures: {outputs['scale_curvatures']}")
    print(f"Number of embeddings: {len(outputs['embeddings'])}")

    # Test loss computation
    l0_labels = torch.randint(0, config.num_l0_classes, (batch_size,))
    l1_labels = torch.randint(0, config.num_l1_classes, (batch_size,))

    losses = model.compute_loss(outputs, l0_labels, l1_labels)

    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Test backward pass
    losses["total"].backward()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\nHFH test passed!")
