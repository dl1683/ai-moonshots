# Trajectory Embeddings: Meaning as Dynamics, Not Points

## The Revolutionary Idea

**Current paradigm:** A sentence maps to a point in vector space. Similar meanings = nearby points.

**New paradigm:** A sentence maps to a *trajectory* through a dynamical system. Similar meanings = trajectories that converge to the same attractor.

```
CURRENT:  "The cat sat on the mat" → [0.23, -0.41, 0.87, ...] (a point)

PROPOSED: "The cat sat on the mat" → trajectory τ(t) where:
          - τ(0) = initial encoding
          - τ(∞) = attractor (stable meaning)
          - The PATH matters, not just the endpoint
```

---

## Why This Could Be Earth-Shattering

### 1. Meaning IS Dynamic

When you read "bank", your understanding *evolves*:
- t=0: Ambiguous (financial? river?)
- t=1: Context narrows it ("river bank" vs "bank account")
- t=2: Specific meaning crystallizes
- t=∞: Stable interpretation

Current embeddings capture only t=∞ (if that). We lose the *process* of understanding.

### 2. Similarity Isn't Distance

Two sentences can be:
- Far apart initially but converge (different words, same meaning)
- Close initially but diverge (same words, different meanings - puns, ambiguity)

Point-based similarity can't capture this. Trajectory alignment can.

### 3. Abstraction as Attractor Depth

Abstract concepts = deeper attractors (more stable, harder to reach)
Concrete concepts = shallow attractors (quick convergence)

"Justice" requires more cognitive "iterations" than "cat". This maps to trajectory length.

### 4. Composition Becomes Flow

Composing meanings = composing vector fields
"The [adj] [noun]" = flowing through adj's field, then noun's field
Order matters! This naturally captures non-commutativity.

---

## Theoretical Foundation

### Neural ODEs for Semantics

Instead of:
```python
embedding = encoder(text)  # Point in R^d
```

We have:
```python
z0 = encoder(text)  # Initial condition
trajectory = odesolve(dynamics, z0, t=[0, T])  # Solve ODE
embedding = trajectory  # The whole path, or attractor
```

The dynamics are learned:
```
dz/dt = f_θ(z, t)
```

Where f_θ is a neural network defining the "semantic vector field".

### Key Insight: Shared Dynamics

ALL meanings evolve according to the SAME dynamics f_θ. Different texts just start at different z0.

This is like physics: all objects follow the same laws (gravity), just from different initial conditions.

**Implication:** Learning f_θ = learning the "laws of meaning"

### Attractor Structure

The vector field f_θ has attractors - stable points/cycles where dz/dt ≈ 0.

- **Point attractors:** Unambiguous meanings
- **Limit cycles:** Meanings with inherent tension/duality
- **Strange attractors:** Complex, context-dependent meanings

Similar meanings = trajectories in the same basin of attraction.

---

## Architecture Design

### Version 1: Basic Trajectory Embedding

```
┌─────────────────────────────────────────────────────────────┐
│                    Trajectory Embedding                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Text ──► [Frozen Backbone] ──► z₀ (initial condition)     │
│                                      │                       │
│                                      ▼                       │
│                          ┌──────────────────┐                │
│                          │  Semantic Field  │                │
│                          │   f_θ(z, t)      │◄──┐            │
│                          └────────┬─────────┘   │            │
│                                   │             │            │
│                                   ▼             │            │
│                          ┌──────────────────┐   │            │
│                          │   ODE Solver     │   │            │
│                          │  dz/dt = f_θ     │───┘            │
│                          └────────┬─────────┘   (iterate)    │
│                                   │                          │
│                                   ▼                          │
│                          trajectory τ(t)                     │
│                          [z₀, z₁, z₂, ..., z_T]              │
│                                   │                          │
│                                   ▼                          │
│                    ┌──────────────────────────┐              │
│                    │   Trajectory Encoder     │              │
│                    │  (pool/summarize path)   │              │
│                    └──────────────┬───────────┘              │
│                                   │                          │
│                                   ▼                          │
│                          Final Embedding                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### The Semantic Field Network

```python
class SemanticField(nn.Module):
    """
    Defines the vector field dz/dt = f(z, t)

    This is the "laws of meaning" - how semantic states evolve.
    """
    def __init__(self, dim=256, hidden_dim=512):
        super().__init__()
        # Time embedding (different dynamics at different "depths")
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # State transformation
        self.net = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t, z):
        """
        Compute dz/dt at state z and time t.
        """
        t_emb = self.time_embed(t.view(-1, 1))
        combined = torch.cat([z, t_emb.expand(z.shape[0], -1)], dim=-1)
        return self.net(combined)
```

### Trajectory Encoder

How do we go from a trajectory to a usable embedding?

**Option A: Final State (Attractor)**
```python
embedding = trajectory[-1]  # Where it converges
```

**Option B: Path Integral**
```python
embedding = trajectory.mean(dim=0)  # Average over path
```

**Option C: Multi-Scale (Best)**
```python
# Different "checkpoints" capture different granularities
embedding = torch.cat([
    trajectory[T//4],   # Early (surface meaning)
    trajectory[T//2],   # Middle (contextualized)
    trajectory[3*T//4], # Late (refined)
    trajectory[-1],     # Final (stable meaning)
], dim=-1)
```

This naturally gives us fractal-like multi-scale structure!

---

## Training Objectives

### 1. Contrastive Trajectory Alignment

Similar meanings should have trajectories that:
- Converge to the same region
- Have similar paths (not just endpoints)

```python
def trajectory_contrastive_loss(traj_anchor, traj_positive, traj_negatives):
    # Compare full trajectories, not just final points

    # Endpoint similarity
    endpoint_sim = cosine_sim(traj_anchor[-1], traj_positive[-1])

    # Path similarity (Dynamic Time Warping or simple correlation)
    path_sim = trajectory_alignment(traj_anchor, traj_positive)

    # Combined loss
    loss = -log(exp(endpoint_sim + path_sim) / sum(exp(neg_sims)))
    return loss
```

### 2. Attractor Regularization

Encourage the system to have stable attractors:

```python
def attractor_loss(trajectory):
    # Velocity should decrease over time (converging)
    velocities = torch.diff(trajectory, dim=0)
    velocity_norms = velocities.norm(dim=-1)

    # Should decrease: v[t+1] < v[t]
    convergence_loss = F.relu(velocity_norms[1:] - velocity_norms[:-1]).mean()

    return convergence_loss
```

### 3. Abstraction Depth Loss

More abstract concepts should need more iterations:

```python
def abstraction_depth_loss(trajectory, abstraction_level):
    # When did the trajectory stabilize?
    velocities = torch.diff(trajectory, dim=0).norm(dim=-1)
    convergence_time = (velocities > threshold).sum()

    # Should correlate with abstraction level
    loss = F.mse_loss(convergence_time, abstraction_level)
    return loss
```

---

## Similarity Computation

### Point-Based (Traditional)
```python
sim = cosine_similarity(emb1, emb2)
```

### Trajectory-Based (New)

**Option 1: Attractor Similarity**
```python
sim = cosine_similarity(traj1[-1], traj2[-1])
```

**Option 2: Path Correlation**
```python
# Are the paths similar shape?
sim = pearson_correlation(traj1.flatten(), traj2.flatten())
```

**Option 3: Basin Membership (Most Principled)**
```python
# Do they converge to the same attractor?
# Run both to convergence, check if they're in same basin
final1, final2 = traj1[-1], traj2[-1]
sim = 1.0 if same_basin(final1, final2) else basin_distance(final1, final2)
```

**Option 4: Trajectory Distance (OT)**
```python
# Optimal transport distance between trajectories
sim = -wasserstein_distance(traj1, traj2)
```

---

## Experiments to Run

### Experiment 1: Basic Feasibility

**Question:** Can we train a semantic field that produces meaningful trajectories?

**Setup:**
- Backbone: BGE-Large (frozen)
- Semantic field: Small MLP
- ODE solver: Euler or RK4
- T = 10 steps
- Dataset: AG News

**Success criteria:**
- Trajectories converge (velocity decreases)
- Similar texts have similar trajectories
- Classification accuracy >= baseline

### Experiment 2: Trajectory vs Point

**Question:** Does trajectory similarity outperform point similarity?

**Setup:**
- Same model, compare:
  - Point embedding (z0)
  - Final state (z_T)
  - Full trajectory encoding

**Metrics:**
- Retrieval P@10
- Classification accuracy
- Clustering quality (silhouette score)

### Experiment 3: Abstraction Depth

**Question:** Do abstract concepts need more iterations?

**Setup:**
- Dataset with abstraction annotations (or use WordNet depth as proxy)
- Measure convergence time for concrete vs abstract concepts

**Prediction:** "Justice", "Freedom", "Beauty" converge slower than "Cat", "Table", "Red"

### Experiment 4: Ambiguity as Basin Boundaries

**Question:** Are ambiguous sentences near attractor basin boundaries?

**Setup:**
- Dataset of ambiguous sentences (puns, garden path sentences)
- Measure distance to nearest basin boundary

**Prediction:** Ambiguous sentences have trajectories that almost diverge (sensitive to small perturbations)

### Experiment 5: Composition as Flow

**Question:** Does adjective+noun composition correspond to sequential flow?

**Setup:**
- Encode "big cat" vs encode "big" then flow through "cat" field
- Compare results

**Prediction:** Flow composition ≈ direct composition, but captures order effects better

---

## Implementation Plan

### Phase 1: Infrastructure (Day 1)

1. **Neural ODE wrapper**
   - Integrate `torchdiffeq` or implement simple Euler solver
   - Handle batching efficiently

2. **Semantic field network**
   - Basic MLP version
   - Time conditioning

3. **Trajectory storage**
   - Efficient storage of variable-length trajectories
   - Checkpointing (don't store every step)

### Phase 2: Basic Training (Day 2-3)

1. **Contrastive training loop**
   - Adapt our existing hierarchy-aware loss
   - Add trajectory-specific terms

2. **Attractor regularization**
   - Ensure convergence
   - Prevent collapse (all trajectories to same point)

3. **Evaluation pipeline**
   - Trajectory visualization
   - Comparison with point embeddings

### Phase 3: Advanced Features (Day 4-5)

1. **Multi-scale trajectory encoding**
   - Combine checkpoints like fractal scales

2. **Trajectory similarity functions**
   - Implement multiple options
   - Compare effectiveness

3. **Abstraction experiments**
   - Test convergence depth hypothesis

### Phase 4: Analysis (Day 6-7)

1. **Visualize the vector field**
   - What do the "semantic laws" look like?
   - Are there identifiable attractor basins?

2. **Ablations**
   - Trajectory length
   - Field architecture
   - Similarity function choice

3. **Write up findings**
   - Whether paradigm shift or interesting failure

---

## Code Skeleton

```python
"""
Trajectory Embeddings: Meaning as Dynamics
==========================================

Core implementation for the trajectory embedding paradigm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint  # Or implement custom solver

class SemanticField(nn.Module):
    """The vector field defining semantic dynamics."""

    def __init__(self, dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main dynamics network
        self.dynamics = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),  # Bound the velocity
        )

        # Scale factor (learnable "speed" of dynamics)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt at state z and time t."""
        # Handle scalar t from ODE solver
        if t.dim() == 0:
            t = t.expand(z.shape[0])

        t_emb = self.time_mlp(t.view(-1, 1))
        combined = torch.cat([z, t_emb], dim=-1)
        velocity = self.dynamics(combined)

        return self.scale * velocity


class TrajectoryEmbedding(nn.Module):
    """
    Full trajectory embedding model.

    Maps text → initial condition → trajectory → embedding
    """

    def __init__(
        self,
        backbone,  # Frozen text encoder
        dim: int = 256,
        num_steps: int = 10,
        solver: str = 'euler',
    ):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_steps = num_steps
        self.solver = solver

        # Project backbone output to trajectory space
        self.initial_proj = nn.Linear(backbone.config.hidden_dim, dim)

        # The semantic field (shared dynamics)
        self.field = SemanticField(dim=dim)

        # Trajectory encoder (convert path to embedding)
        self.traj_encoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),  # 4 checkpoints
            nn.LayerNorm(dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def get_initial_condition(self, input_ids, attention_mask):
        """Get z0 from backbone."""
        with torch.no_grad():
            backbone_out = self.backbone(input_ids, attention_mask)

        # Pool and project
        pooled = backbone_out.last_hidden_state[:, 0]  # CLS token
        z0 = self.initial_proj(pooled)
        return z0

    def integrate_trajectory(self, z0: torch.Tensor) -> torch.Tensor:
        """Solve the ODE to get trajectory."""
        t = torch.linspace(0, 1, self.num_steps, device=z0.device)

        if self.solver == 'euler':
            # Simple Euler integration
            trajectory = [z0]
            z = z0
            dt = 1.0 / (self.num_steps - 1)
            for i in range(1, self.num_steps):
                dz = self.field(t[i-1], z)
                z = z + dt * dz
                trajectory.append(z)
            trajectory = torch.stack(trajectory, dim=1)  # [batch, steps, dim]
        else:
            # Use torchdiffeq
            trajectory = odeint(self.field, z0, t)  # [steps, batch, dim]
            trajectory = trajectory.permute(1, 0, 2)  # [batch, steps, dim]

        return trajectory

    def encode_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Convert trajectory to fixed-size embedding."""
        # Multi-scale checkpoints
        T = trajectory.shape[1]
        checkpoints = torch.cat([
            trajectory[:, T//4],      # Early
            trajectory[:, T//2],      # Middle
            trajectory[:, 3*T//4],    # Late
            trajectory[:, -1],        # Final
        ], dim=-1)

        return self.traj_encoder(checkpoints)

    def forward(
        self,
        input_ids,
        attention_mask,
        return_trajectory: bool = False,
    ):
        """Full forward pass."""
        # Get initial condition
        z0 = self.get_initial_condition(input_ids, attention_mask)

        # Integrate trajectory
        trajectory = self.integrate_trajectory(z0)

        # Encode to embedding
        embedding = self.encode_trajectory(trajectory)

        if return_trajectory:
            return {
                'embedding': embedding,
                'trajectory': trajectory,
                'z0': z0,
                'z_final': trajectory[:, -1],
            }

        return {'embedding': embedding}

    def encode(self, texts: list, batch_size: int = 32) -> torch.Tensor:
        """Encode texts to embeddings."""
        # Tokenize and encode
        # (Implementation depends on backbone tokenizer)
        pass


class TrajectoryContrastiveLoss(nn.Module):
    """Contrastive loss for trajectories."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_traj: torch.Tensor,    # [batch, steps, dim]
        positive_traj: torch.Tensor,  # [batch, steps, dim]
        labels: torch.Tensor = None,
    ):
        batch_size = anchor_traj.shape[0]

        # Endpoint similarity
        anchor_final = F.normalize(anchor_traj[:, -1], dim=-1)
        positive_final = F.normalize(positive_traj[:, -1], dim=-1)

        endpoint_logits = anchor_final @ positive_final.T / self.temperature

        # Path similarity (simplified: correlation of flattened trajectories)
        anchor_flat = anchor_traj.flatten(1)
        positive_flat = positive_traj.flatten(1)
        path_sim = F.cosine_similarity(anchor_flat, positive_flat, dim=-1)

        # Combined loss
        if labels is None:
            labels = torch.arange(batch_size, device=anchor_traj.device)

        loss = F.cross_entropy(endpoint_logits, labels)

        # Add path alignment bonus
        loss = loss - 0.1 * path_sim.mean()

        return loss


def convergence_loss(trajectory: torch.Tensor) -> torch.Tensor:
    """Encourage trajectories to converge (decreasing velocity)."""
    # Compute velocities
    velocities = torch.diff(trajectory, dim=1)  # [batch, steps-1, dim]
    velocity_norms = velocities.norm(dim=-1)    # [batch, steps-1]

    # Velocity should decrease over time
    velocity_diff = velocity_norms[:, 1:] - velocity_norms[:, :-1]
    convergence_penalty = F.relu(velocity_diff).mean()

    return convergence_penalty


def diversity_loss(trajectories: torch.Tensor) -> torch.Tensor:
    """Prevent all trajectories collapsing to same attractor."""
    final_states = trajectories[:, -1]  # [batch, dim]

    # Pairwise distances should not all be zero
    dists = torch.cdist(final_states, final_states)

    # Encourage diversity (penalize small average distance)
    diversity = dists.mean()

    return -diversity  # Negative because we want to maximize
```

---

## Risks and Mitigations

### Risk 1: ODE Instability
**Problem:** Trajectories explode or collapse
**Mitigation:**
- Bound velocity with Tanh
- Add gradient clipping
- Use adaptive step solvers

### Risk 2: Computational Cost
**Problem:** ODE solving is expensive
**Mitigation:**
- Use few steps (10-20)
- Use simple Euler solver
- Checkpoint, don't store all steps

### Risk 3: No Improvement
**Problem:** Trajectories don't help
**Mitigation:**
- Start with small experiments
- Compare multiple similarity functions
- May find theoretical insights even if metrics don't improve

### Risk 4: Training Collapse
**Problem:** All trajectories converge to same point
**Mitigation:**
- Diversity loss
- Contrastive training
- Different initial conditions should stay different

---

## What Success Looks Like

### Minimum Viable Success
- Trajectories converge (system is stable)
- Classification accuracy matches baseline
- Interesting visualizations of semantic dynamics

### Good Success
- Retrieval/classification beats point embeddings
- Abstraction depth hypothesis confirmed
- Clear attractor structure visible

### Revolutionary Success
- Significant improvement on hard tasks
- Ambiguity detection works (basin boundary hypothesis)
- Composition as flow works
- The "semantic laws" are interpretable
- Paper-worthy theoretical framework

---

## Next Session Checklist

1. [ ] Set up torchdiffeq or implement Euler solver
2. [ ] Build SemanticField network
3. [ ] Build TrajectoryEmbedding model
4. [ ] Integrate with existing backbone loading
5. [ ] Implement basic training loop
6. [ ] Run Experiment 1 (feasibility)
7. [ ] Visualize trajectories
8. [ ] Compare with point embeddings
9. [ ] If promising, run remaining experiments

---

## Key Equations Summary

**Dynamics:**
$$\frac{dz}{dt} = f_\theta(z, t)$$

**Trajectory:**
$$\tau(t) = z_0 + \int_0^t f_\theta(z(s), s) ds$$

**Similarity (Attractor):**
$$\text{sim}(x, y) = \cos(\tau_x(\infty), \tau_y(\infty))$$

**Similarity (Path):**
$$\text{sim}(x, y) = \text{corr}(\tau_x, \tau_y)$$

**The Deep Question:**
If this works, what are the "semantic laws" f_θ? What do they tell us about the structure of meaning?

---

*Document created: January 31, 2026*
*Status: Ready to implement*
*Priority: HIGH - Paradigm shift potential*
