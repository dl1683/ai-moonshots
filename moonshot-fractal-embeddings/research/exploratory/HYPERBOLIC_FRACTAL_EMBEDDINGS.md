# Hyperbolic Fractal Embeddings: Curved Geometry for Semantic Hierarchies

## The Core Insight

**Current paradigm:** Semantic space is flat (Euclidean). Distance = straight line. All directions equal.

**New paradigm:** Semantic space is curved (Hyperbolic). Hierarchies embed naturally. Abstraction = distance from center.

```
FLAT (Euclidean):                 CURVED (Hyperbolic):

    dog ●───────● cat                      thing
        \      /                             │
         \    /                        ┌─────┴─────┐
          \  /                      animal      vehicle
      animal ●  ← squashed!          /   \        /   \
                                   dog   cat    car   boat

Trees don't fit naturally.        Trees embed PERFECTLY.
```

---

## Why Hyperbolic Space?

### The Problem with Flat Space

In Euclidean space, a circle of radius r has circumference 2πr (linear growth).

**Problem:** Hierarchies are trees. Trees have exponentially many leaves. You can't fit exponential growth in linear space without distortion.

### The Hyperbolic Solution

In hyperbolic space, a circle of radius r has circumference that grows **exponentially** with r.

**Result:** Trees embed with **zero distortion**. Every hierarchy fits perfectly.

### The Poincaré Ball Model

We use the Poincaré ball - hyperbolic space mapped to a unit ball:

```
         ┌─────────────────────┐
        /                       \
       /    abstract concepts    \
      /      (near center)        \
     │            ●                │
     │           /|\               │
     │          / | \              │
     │         /  |  \             │
     │        ●   ●   ●            │
     │       /|\ /|\ /|\           │
     │      ● ● ● ● ● ● ●          │  ← specific concepts
      \    (near boundary)        /      (more "room" here)
       \                         /
        \_______________________/
```

**Key properties:**
- Center = most abstract
- Boundary = most specific
- Distance stretches near boundary (more room for specifics)
- Hierarchical relationships = radial structure

---

## Mathematical Foundation

### Poincaré Ball Model

The Poincaré ball B^n is the unit ball with hyperbolic metric:

```
B^n = {x ∈ ℝ^n : ||x|| < 1}
```

**Hyperbolic distance:**
```
d_H(x, y) = arcosh(1 + 2 * ||x - y||² / ((1 - ||x||²)(1 - ||y||²)))
```

**Key insight:** As points approach the boundary (||x|| → 1), distances stretch to infinity. This creates "room" for exponentially many specific concepts.

### Möbius Operations

In hyperbolic space, we can't use regular addition. We use **Möbius addition**:

```
x ⊕ y = ((1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y) / (1 + 2⟨x,y⟩ + ||x||²||y||²)
```

**Möbius scalar multiplication:**
```
r ⊗ x = tanh(r * arctanh(||x||)) * x/||x||
```

### Exponential and Logarithmic Maps

To move between Euclidean and hyperbolic:

**Exponential map** (Euclidean → Hyperbolic):
```
exp_0(v) = tanh(||v||) * v/||v||
```

**Logarithmic map** (Hyperbolic → Euclidean):
```
log_0(x) = arctanh(||x||) * x/||x||
```

---

## Architecture: Hyperbolic Fractal Head

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Hyperbolic Fractal Head                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Backbone (frozen) ──► h ∈ ℝ^d (Euclidean)                 │
│                              │                               │
│                              ▼                               │
│                    ┌──────────────────┐                      │
│                    │   Projection     │                      │
│                    │   + Exp Map      │                      │
│                    └────────┬─────────┘                      │
│                             │                                │
│                             ▼                                │
│                    z₀ ∈ B^d (Poincaré ball)                  │
│                             │                                │
│                             ▼                                │
│              ┌─────────────────────────────┐                 │
│              │   Hyperbolic Fractal Block  │                 │
│              │   (Möbius attention + FFN)  │──► s₀ (center)  │
│              └──────────────┬──────────────┘                 │
│                             │ ⊕ (Möbius add residual)        │
│              ┌──────────────▼──────────────┐                 │
│              │   Hyperbolic Fractal Block  │──► s₁           │
│              └──────────────┬──────────────┘                 │
│                             │ ⊕                              │
│              ┌──────────────▼──────────────┐                 │
│              │   Hyperbolic Fractal Block  │──► s₂           │
│              └──────────────┬──────────────┘                 │
│                             │ ⊕                              │
│              ┌──────────────▼──────────────┐                 │
│              │   Hyperbolic Fractal Block  │──► s₃ (edge)    │
│              └─────────────────────────────┘                 │
│                                                              │
│   Scale 0: Near center (abstract)                            │
│   Scale 3: Near boundary (specific)                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Hyperbolic Fractal Block

```python
class HyperbolicFractalBlock(nn.Module):
    """
    Fractal block operating in hyperbolic space.

    All operations use Möbius arithmetic to stay on the manifold.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, curvature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.c = curvature  # Curvature of hyperbolic space

        # Hyperbolic attention
        self.attention = HyperbolicMultiheadAttention(dim, num_heads, curvature)

        # Hyperbolic FFN (project to tangent space, transform, project back)
        self.ffn = HyperbolicFFN(dim, dim * 4, curvature)

        # Hyperbolic layer norm
        self.norm1 = HyperbolicLayerNorm(dim, curvature)
        self.norm2 = HyperbolicLayerNorm(dim, curvature)

    def forward(self, x):
        # x is on Poincaré ball

        # Hyperbolic attention with residual (Möbius addition)
        attended = self.attention(self.norm1(x))
        x = mobius_add(x, attended, self.c)

        # Hyperbolic FFN with residual
        transformed = self.ffn(self.norm2(x))
        x = mobius_add(x, transformed, self.c)

        return x
```

### Key Operations

```python
def mobius_add(x, y, c=1.0):
    """
    Möbius addition in Poincaré ball.

    This is the hyperbolic equivalent of vector addition.
    """
    x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
    y_norm_sq = (y * y).sum(dim=-1, keepdim=True)
    xy_inner = (x * y).sum(dim=-1, keepdim=True)

    numerator = (1 + 2*c*xy_inner + c*y_norm_sq) * x + (1 - c*x_norm_sq) * y
    denominator = 1 + 2*c*xy_inner + c*c*x_norm_sq*y_norm_sq

    return numerator / denominator.clamp(min=1e-15)


def hyperbolic_distance(x, y, c=1.0):
    """
    Distance in Poincaré ball.
    """
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1)

    return 2 / math.sqrt(c) * torch.arctanh(math.sqrt(c) * diff_norm)


def exp_map_zero(v, c=1.0):
    """
    Exponential map from origin (Euclidean tangent space → Poincaré ball).
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    return torch.tanh(math.sqrt(c) * v_norm) * v / (math.sqrt(c) * v_norm)


def log_map_zero(x, c=1.0):
    """
    Logarithmic map to origin (Poincaré ball → Euclidean tangent space).
    """
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    return torch.arctanh(math.sqrt(c) * x_norm) * x / (math.sqrt(c) * x_norm)
```

---

## Why Combine Hyperbolic + Fractal?

### Complementary Strengths

| Aspect | Euclidean Fractal | Hyperbolic Fractal |
|--------|-------------------|-------------------|
| Geometry | Flat (wrong for hierarchies) | Curved (natural for hierarchies) |
| Multi-scale | Explicit scales | Scales + natural depth |
| Abstraction | Learned | Geometric (distance from center) |
| Tree embedding | Approximate | Perfect |

### The Hypothesis

Our Euclidean fractal embeddings improved +10-28% because they added **some** hierarchical awareness.

Hyperbolic fractal embeddings should improve **even more** because:
1. The geometry itself encodes hierarchy
2. Scales map to radial distance (abstraction level)
3. No need to "fight" the geometry

### Predicted Improvements

| Task | Euclidean Fractal | Hyperbolic Fractal (Predicted) |
|------|-------------------|-------------------------------|
| L0 Classification | +9-17% | +15-25% |
| L1 Classification | +9-19% | +20-35% |
| Hierarchical Retrieval | +10-28% | +25-50% |
| Tree Reconstruction | Poor | Near-perfect |

---

## Training Objectives

### 1. Hyperbolic Contrastive Loss

```python
def hyperbolic_contrastive_loss(anchor, positive, negatives, c=1.0, temp=0.07):
    """
    Contrastive loss using hyperbolic distance.
    """
    # Compute hyperbolic distances
    pos_dist = hyperbolic_distance(anchor, positive, c)
    neg_dists = torch.stack([hyperbolic_distance(anchor, neg, c) for neg in negatives])

    # Convert to similarities (closer = more similar)
    pos_sim = -pos_dist / temp
    neg_sims = -neg_dists / temp

    # InfoNCE
    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims], dim=0)
    labels = torch.zeros(1, dtype=torch.long, device=anchor.device)

    return F.cross_entropy(logits.unsqueeze(0), labels)
```

### 2. Hierarchy-Aware Radial Loss

Encourage abstract concepts to be near center, specific concepts near boundary:

```python
def radial_hierarchy_loss(embeddings, abstraction_levels, c=1.0):
    """
    Abstract concepts should be near center (small norm).
    Specific concepts should be near boundary (large norm).
    """
    # Compute norms (distance from center in Poincaré ball)
    norms = embeddings.norm(dim=-1)

    # Abstraction level 0 = most abstract, should have small norm
    # Higher levels = more specific, should have larger norm
    target_norms = abstraction_levels / abstraction_levels.max()  # Normalize to [0, 1]

    # But Poincaré ball has max norm < 1, so scale appropriately
    target_norms = target_norms * 0.9  # Stay away from boundary

    return F.mse_loss(norms, target_norms)
```

### 3. Scale-Radius Alignment

Each fractal scale should correspond to a radial band:

```python
def scale_radius_loss(scale_embeddings, c=1.0):
    """
    Scale 0 should be near center, Scale 3 near boundary.
    """
    losses = []
    num_scales = len(scale_embeddings)

    for i, scale_emb in enumerate(scale_embeddings):
        target_radius = (i + 1) / (num_scales + 1) * 0.9  # Evenly spaced bands
        actual_radius = scale_emb.norm(dim=-1)
        losses.append(F.mse_loss(actual_radius, torch.full_like(actual_radius, target_radius)))

    return sum(losses) / num_scales
```

---

## Experiments

### Experiment 1: Hyperbolic vs Euclidean Fractal

**Question:** Does hyperbolic geometry improve over our Euclidean fractal?

**Setup:**
- Same fractal architecture
- Replace Euclidean ops with Möbius ops
- Same training data (AG News, Yahoo)

**Metrics:**
- L0/L1 classification accuracy
- Hierarchical retrieval P@10
- Cluster separation

### Experiment 2: Tree Embedding Quality

**Question:** Do hierarchies embed better in hyperbolic space?

**Setup:**
- Embed WordNet hierarchy
- Measure reconstruction error (Gromov hyperbolicity)

**Prediction:** Hyperbolic should have near-zero reconstruction error vs high error for Euclidean.

### Experiment 3: Abstraction as Radius

**Question:** Do abstract concepts naturally end up near center?

**Setup:**
- Embed words with known abstraction levels (WordNet depth)
- Measure correlation between abstraction and distance from center

**Prediction:** Strong negative correlation (abstract = small radius).

### Experiment 4: Learnable Curvature

**Question:** What curvature is optimal for semantic space?

**Setup:**
- Make curvature c a learnable parameter
- Train on different datasets
- See what curvature emerges

**Insight:** If optimal c varies by domain, semantic spaces have **locally varying curvature**.

### Experiment 5: Scaling Analysis

**Question:** Do gains increase with hierarchy depth?

**Setup:**
- Test on hierarchies of varying depth
- Compare Euclidean fractal vs Hyperbolic fractal

**Prediction:** Hyperbolic advantage grows with hierarchy depth.

---

## Implementation Plan

### Phase 1: Core Hyperbolic Ops (Day 1)

```python
# Implement or use geoopt library
- mobius_add(x, y, c)
- mobius_scalar_mult(r, x, c)
- hyperbolic_distance(x, y, c)
- exp_map_zero(v, c)
- log_map_zero(x, c)
- project_to_ball(x)  # Ensure ||x|| < 1
```

### Phase 2: Hyperbolic Layers (Day 1-2)

```python
- HyperbolicLinear  # Linear in tangent space, project back
- HyperbolicLayerNorm
- HyperbolicMultiheadAttention
- HyperbolicFFN
```

### Phase 3: Hyperbolic Fractal Head (Day 2-3)

```python
- HyperbolicFractalBlock
- HyperbolicFractalHead
- Integration with existing backbones
```

### Phase 4: Training (Day 3-4)

```python
- Hyperbolic contrastive loss
- Radial hierarchy loss
- Riemannian Adam optimizer (for proper gradients on manifold)
```

### Phase 5: Evaluation (Day 4-5)

```python
- Compare with Euclidean fractal
- Tree reconstruction
- Abstraction-radius correlation
```

---

## Code Skeleton

```python
"""
Hyperbolic Fractal Embeddings
=============================

Combining hyperbolic geometry with fractal multi-scale structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# Core Hyperbolic Operations
# ============================================================================

class PoincareBall:
    """Operations in the Poincaré ball model of hyperbolic space."""

    def __init__(self, c: float = 1.0, eps: float = 1e-15):
        self.c = c
        self.eps = eps

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition: x ⊕ y"""
        c = self.c
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True).clamp(min=self.eps)
        y_norm_sq = (y * y).sum(dim=-1, keepdim=True).clamp(min=self.eps)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2*c*xy + c*y_norm_sq) * x + (1 - c*x_norm_sq) * y
        denom = 1 + 2*c*xy + c*c*x_norm_sq*y_norm_sq

        return num / denom.clamp(min=self.eps)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance."""
        sqrt_c = math.sqrt(self.c)
        mob_add = self.mobius_add(-x, y)
        norm = mob_add.norm(dim=-1).clamp(max=1-self.eps)
        return 2 / sqrt_c * torch.arctanh(sqrt_c * norm)

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at origin."""
        sqrt_c = math.sqrt(self.c)
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)

    def log_map_zero(self, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at origin."""
        sqrt_c = math.sqrt(self.c)
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps, max=1-self.eps)
        return torch.arctanh(sqrt_c * x_norm) * x / (sqrt_c * x_norm)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball (ensure ||x|| < 1)."""
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = 1 - self.eps
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)


# ============================================================================
# Hyperbolic Neural Layers
# ============================================================================

class HyperbolicLinear(nn.Module):
    """Linear layer in hyperbolic space."""

    def __init__(self, in_features: int, out_features: int, c: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.ball = PoincareBall(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Log map to tangent space → linear → exp map back
        x_tangent = self.ball.log_map_zero(x)
        y_tangent = self.linear(x_tangent)
        return self.ball.project(self.ball.exp_map_zero(y_tangent))


class HyperbolicFFN(nn.Module):
    """Feed-forward network in hyperbolic space."""

    def __init__(self, dim: int, hidden_dim: int, c: float = 1.0):
        super().__init__()
        self.ball = PoincareBall(c)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Work in tangent space
        x_tangent = self.ball.log_map_zero(x)
        h = self.act(self.fc1(x_tangent))
        out_tangent = self.fc2(h)
        return self.ball.project(self.ball.exp_map_zero(out_tangent))


# ============================================================================
# Hyperbolic Fractal Block
# ============================================================================

class HyperbolicFractalBlock(nn.Module):
    """
    Single fractal block operating in hyperbolic space.

    Same block is applied at each scale (self-similarity).
    """

    def __init__(self, dim: int = 256, hidden_dim: int = 512, c: float = 1.0):
        super().__init__()
        self.ball = PoincareBall(c)

        # Transform in tangent space
        self.transform = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        # Scale for residual
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor, inject: torch.Tensor = None) -> torch.Tensor:
        # Transform in tangent space
        x_tangent = self.ball.log_map_zero(x)
        transformed = self.transform(x_tangent) * self.residual_scale

        # Add residual in hyperbolic space
        delta = self.ball.exp_map_zero(transformed)
        out = self.ball.mobius_add(x, delta)

        # Inject original input (prevent information loss)
        if inject is not None:
            out = self.ball.mobius_add(out, inject)

        return self.ball.project(out)


# ============================================================================
# Full Hyperbolic Fractal Head
# ============================================================================

class HyperbolicFractalHead(nn.Module):
    """
    Multi-scale fractal head in hyperbolic space.

    Each scale corresponds to a radial band:
    - Scale 0: Near center (abstract)
    - Scale N: Near boundary (specific)
    """

    def __init__(
        self,
        input_dim: int,
        num_scales: int = 4,
        scale_dim: int = 64,
        c: float = 1.0,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.scale_dim = scale_dim
        self.ball = PoincareBall(c)

        # Project to hyperbolic space
        self.input_proj = nn.Linear(input_dim, num_scales * scale_dim)

        # Shared fractal block (self-similarity!)
        self.fractal_block = HyperbolicFractalBlock(
            dim=num_scales * scale_dim,
            hidden_dim=num_scales * scale_dim * 2,
            c=c,
        )

        # Scale-specific projections (to different radial bands)
        self.scale_projs = nn.ModuleList([
            nn.Linear(num_scales * scale_dim, scale_dim)
            for _ in range(num_scales)
        ])

        # Learnable radial targets for each scale
        self.target_radii = nn.Parameter(
            torch.linspace(0.2, 0.8, num_scales)
        )

    def forward(self, backbone_output: torch.Tensor, return_scales: bool = False):
        # Project to hyperbolic space
        h = self.input_proj(backbone_output)
        h = self.ball.exp_map_zero(h)

        # Store input for injection
        h_input = h

        scale_embeddings = []

        for i in range(self.num_scales):
            # Apply shared fractal block
            h = self.fractal_block(h, inject=h_input if i > 0 else None)

            # Project to scale-specific embedding
            h_tangent = self.ball.log_map_zero(h)
            scale_emb = self.scale_projs[i](h_tangent)

            # Push to target radius
            scale_emb = self.ball.exp_map_zero(scale_emb)
            current_radius = scale_emb.norm(dim=-1, keepdim=True)
            target_radius = self.target_radii[i]
            scale_emb = scale_emb * (target_radius / current_radius.clamp(min=1e-6))
            scale_emb = self.ball.project(scale_emb)

            scale_embeddings.append(scale_emb)

        # Full embedding: concatenate all scales
        full_embedding = torch.cat(scale_embeddings, dim=-1)

        if return_scales:
            return {
                'embedding': full_embedding,
                'scale_embeddings': scale_embeddings,
            }

        return {'embedding': full_embedding}


# ============================================================================
# Losses
# ============================================================================

def hyperbolic_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
    c: float = 1.0,
) -> torch.Tensor:
    """Contrastive loss using hyperbolic distance."""
    ball = PoincareBall(c)

    batch_size = anchor.shape[0]

    # Compute all pairwise hyperbolic distances
    # anchor: [B, D], positive: [B, D]

    # Distance between anchor[i] and positive[j]
    dists = torch.zeros(batch_size, batch_size, device=anchor.device)
    for i in range(batch_size):
        for j in range(batch_size):
            dists[i, j] = ball.distance(anchor[i], positive[j])

    # Convert to similarities (negative distance)
    sims = -dists / temperature

    # Diagonal should be positive pairs
    labels = torch.arange(batch_size, device=anchor.device)

    return F.cross_entropy(sims, labels)


def radial_hierarchy_loss(
    scale_embeddings: list,
    target_radii: torch.Tensor,
) -> torch.Tensor:
    """Encourage scales to occupy their target radial bands."""
    loss = 0
    for i, scale_emb in enumerate(scale_embeddings):
        actual_radius = scale_emb.norm(dim=-1)
        target = target_radii[i].expand_as(actual_radius)
        loss += F.mse_loss(actual_radius, target)
    return loss / len(scale_embeddings)
```

---

## Comparison: Euclidean vs Hyperbolic Fractal

| Aspect | Euclidean Fractal | Hyperbolic Fractal |
|--------|-------------------|-------------------|
| Geometry | Flat | Curved (natural for trees) |
| Distance | ||x - y|| | arcosh-based (stretches near boundary) |
| Abstraction | Learned implicitly | Geometric (distance from center) |
| Scale meaning | Arbitrary | Radial bands (center→edge) |
| Tree embedding | Distorted | Perfect |
| Operations | Standard +, × | Möbius ⊕, ⊗ |
| Expected gain | +10-28% | +25-50% (predicted) |

---

## Risks and Mitigations

### Risk 1: Numerical Instability
**Problem:** Hyperbolic ops can be unstable near boundary
**Mitigation:**
- Clamp norms away from 1
- Use eps in all divisions
- Periodic projection to ball

### Risk 2: Optimization Difficulty
**Problem:** Riemannian gradients needed
**Mitigation:**
- Use geoopt library for proper Riemannian optimizers
- Or work in tangent space and project

### Risk 3: Curvature Mismatch
**Problem:** Wrong curvature c hurts performance
**Mitigation:**
- Make c learnable
- Try multiple fixed values
- Per-scale curvature

### Risk 4: Computational Cost
**Problem:** Hyperbolic ops more expensive
**Mitigation:**
- Efficient batched implementations
- Most cost is in backbone anyway

---

## Success Criteria

### Minimum Viable Success
- Model trains stably
- Performance matches Euclidean fractal
- Embeddings show radial structure

### Good Success
- Beats Euclidean fractal on hierarchical tasks
- Abstract concepts near center empirically
- Tree embedding quality improves

### Revolutionary Success
- Large gains (+25-50%) on hierarchical tasks
- Learned curvature reveals structure of semantic space
- Zero-shot transfer improves (geometry matches reality)
- New theoretical insights about meaning

---

## Next Session Checklist

1. [ ] Install geoopt (or implement core hyperbolic ops)
2. [ ] Build HyperbolicFractalBlock
3. [ ] Build HyperbolicFractalHead
4. [ ] Implement hyperbolic contrastive loss
5. [ ] Train on AG News
6. [ ] Compare with Euclidean fractal
7. [ ] Visualize radial structure
8. [ ] Measure tree embedding quality

---

## Key Equations Summary

**Hyperbolic Distance:**
$$d_H(x, y) = \text{arcosh}\left(1 + 2\frac{||x - y||^2}{(1 - ||x||^2)(1 - ||y||^2)}\right)$$

**Möbius Addition:**
$$x \oplus y = \frac{(1 + 2\langle x, y \rangle + ||y||^2)x + (1 - ||x||^2)y}{1 + 2\langle x, y \rangle + ||x||^2||y||^2}$$

**Exponential Map:**
$$\exp_0(v) = \tanh(||v||) \frac{v}{||v||}$$

**The Deep Question:**
If semantic space is truly hyperbolic, what is its curvature? Does it vary by domain? What does this tell us about the fundamental structure of meaning?

---

*Document created: January 31, 2026*
*Status: Ready to implement*
*Priority: HIGH - Most tractable path to validating curved geometry hypothesis*
*Dependency: Can run in parallel with Trajectory Embeddings*
