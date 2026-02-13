"""
HEAL Baseline: Hierarchy-Enhanced Adaptive Learning (ICLR 2025)
================================================================

Adapts the core HEAL idea for fair comparison with V5:
- Level-wise supervised contrastive loss at each hierarchy level
- Hierarchy distance weighting for negative pairs
- MLP projection head producing 256d output (4 x 64d blocks)
- Trains on frozen bge-small embeddings

Key HEAL design principles:
1. Level-aware contrastive: separate SupCon loss for L0 and L1
2. Hierarchy-distance weighting: negatives that are closer in the
   hierarchy tree get larger penalty (harder negatives)
3. Progressive allocation: coarse info in early dims, fine in later

We implement this as a thin MLP head on frozen embeddings to match
our experimental framework (same backbone, same eval).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

from .common import (
    BaselineHead,
    BaselineTrainer,
    ExternalRunConfig,
)


class HEALHead(BaselineHead):
    """
    HEAL-style projection head.

    Architecture:
    - Shared MLP backbone: input_dim -> hidden -> hidden
    - Level-0 projector: hidden -> scale_dim (first 64d for coarse)
    - Level-1 projector: hidden -> 3*scale_dim (remaining 192d for fine)
    - Output: concat [L0_proj, L1_proj] = 256d

    This enforces structural separation: early dimensions encode
    coarse hierarchy, later dimensions encode fine-grained detail.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        hidden_dim: int = 512,
        num_l0: int = 10,
        num_l1: int = 100,
        scale_dim: int = 64,
    ):
        super().__init__(input_dim, output_dim)
        self.scale_dim = scale_dim
        self.num_l0 = num_l0
        self.num_l1 = num_l1

        # Shared MLP trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Level-specific projectors
        # L0 projector: produces first scale_dim dimensions (coarse)
        self.l0_proj = nn.Sequential(
            nn.Linear(hidden_dim, scale_dim),
        )

        # L1 projector: produces remaining 3*scale_dim dimensions (fine)
        self.l1_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim - scale_dim),
        )

        # Classification heads (for auxiliary CE loss)
        self.l0_classifier = nn.Linear(scale_dim, num_l0)
        self.l1_classifier = nn.Linear(output_dim, num_l1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map frozen embeddings to 256d output."""
        h = self.trunk(x)
        l0_emb = self.l0_proj(h)           # [batch, 64]
        l1_emb = self.l1_proj(h)            # [batch, 192]
        full = torch.cat([l0_emb, l1_emb], dim=-1)  # [batch, 256]
        return F.normalize(full, dim=-1)

    def forward_decomposed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (full_emb, l0_emb, l1_emb) for loss computation."""
        h = self.trunk(x)
        l0_emb = self.l0_proj(h)
        l1_emb = self.l1_proj(h)
        full = torch.cat([l0_emb, l1_emb], dim=-1)
        full = F.normalize(full, dim=-1)
        l0_emb_norm = F.normalize(l0_emb, dim=-1)
        return full, l0_emb_norm, h


class HEALTrainer(BaselineTrainer):
    """HEAL-style training with level-wise supervised contrastive loss."""

    # Loss weights
    SUPCON_WEIGHT = 1.0
    CE_WEIGHT = 0.5
    HIERARCHY_WEIGHT = 0.3  # Extra weight for hierarchy-distance negatives
    TEMPERATURE = 0.1

    def build_head(self, input_dim: int, num_l0: int, num_l1: int) -> BaselineHead:
        return HEALHead(
            input_dim=input_dim,
            output_dim=self.config.output_dim,
            hidden_dim=512,
            num_l0=num_l0,
            num_l1=num_l1,
            scale_dim=self.config.scale_dim,
        )

    def _supcon_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Supervised contrastive loss (Khosla et al., 2020).

        For each anchor, positives = same label, negatives = different label.
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Similarity matrix
        sim = embeddings @ embeddings.T / temperature  # [B, B]

        # Mask: same class = 1, different = 0, diagonal = 0
        labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B]
        mask_pos = labels_eq.float()
        mask_pos.fill_diagonal_(0)

        # Number of positives per anchor
        num_pos = mask_pos.sum(dim=1)  # [B]

        # For numerical stability
        sim_max = sim.max(dim=1, keepdim=True).values.detach()
        sim = sim - sim_max

        # Log-sum-exp over all (including negatives)
        exp_sim = torch.exp(sim)
        # Exclude self
        exp_sim.fill_diagonal_(0)
        log_sum_exp = torch.log(exp_sim.sum(dim=1) + 1e-8)  # [B]

        # Mean of log(exp(sim_pos) / sum(exp(sim_all)))
        # = mean of (sim_pos - log_sum_exp)
        pos_sim_sum = (sim * mask_pos).sum(dim=1)  # [B]

        # Only count anchors with at least one positive
        valid = num_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        loss = -(pos_sim_sum[valid] / num_pos[valid].clamp(min=1) - log_sum_exp[valid])
        return loss.mean()

    def _hierarchy_weighted_supcon(
        self,
        embeddings: torch.Tensor,
        l0_labels: torch.Tensor,
        l1_labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Hierarchy-distance weighted supervised contrastive loss.

        Negatives that share the same L0 but differ in L1 get a larger
        weight (they are "harder" negatives -- close in hierarchy tree
        but different at the fine level).
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        sim = embeddings @ embeddings.T / temperature

        # Positive mask: same L1 label
        mask_pos = (l1_labels.unsqueeze(1) == l1_labels.unsqueeze(0)).float()
        mask_pos.fill_diagonal_(0)

        # Hard negative mask: same L0 but different L1
        same_l0 = (l0_labels.unsqueeze(1) == l0_labels.unsqueeze(0)).float()
        diff_l1 = 1.0 - (l1_labels.unsqueeze(1) == l1_labels.unsqueeze(0)).float()
        hard_neg = same_l0 * diff_l1
        hard_neg.fill_diagonal_(0)

        # Weight: 1.0 for easy negatives, 1.0 + HIERARCHY_WEIGHT for hard negatives
        neg_weight = torch.ones(batch_size, batch_size, device=embeddings.device)
        neg_weight = neg_weight + self.HIERARCHY_WEIGHT * hard_neg
        neg_weight.fill_diagonal_(0)

        # Stability
        sim_max = sim.max(dim=1, keepdim=True).values.detach()
        sim = sim - sim_max

        exp_sim = torch.exp(sim) * neg_weight
        exp_sim.fill_diagonal_(0)
        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)

        num_pos = mask_pos.sum(dim=1)
        pos_sim_sum = (sim * mask_pos).sum(dim=1)

        valid = num_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        loss = -(pos_sim_sum[valid] / num_pos[valid].clamp(min=1) - log_denom[valid])
        return loss.mean()

    def compute_loss(
        self,
        head: HEALHead,
        embeddings: torch.Tensor,
        l0_labels: torch.Tensor,
        l1_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        HEAL loss = L0_SupCon(l0_proj) + Hierarchy_SupCon(full) + CE_L0 + CE_L1

        Components:
        1. L0 SupCon on the coarse projection (first 64d)
        2. Hierarchy-weighted SupCon on full embedding (256d)
        3. Cross-entropy for L0 (from l0_proj)
        4. Cross-entropy for L1 (from full)
        """
        full_emb, l0_emb, trunk_h = head.forward_decomposed(embeddings)

        # 1. L0 supervised contrastive on coarse projection
        l0_supcon = self._supcon_loss(l0_emb, l0_labels, self.TEMPERATURE)

        # 2. Hierarchy-weighted SupCon on full embedding
        hier_supcon = self._hierarchy_weighted_supcon(
            full_emb, l0_labels, l1_labels, self.TEMPERATURE
        )

        # 3. CE for L0 from l0 projection
        l0_logits = head.l0_classifier(l0_emb * np.sqrt(head.scale_dim))  # undo norm
        ce_l0 = F.cross_entropy(l0_logits, l0_labels)

        # 4. CE for L1 from full embedding
        l1_logits = head.l1_classifier(full_emb * np.sqrt(head.output_dim))
        ce_l1 = F.cross_entropy(l1_logits, l1_labels)

        total = (
            self.SUPCON_WEIGHT * l0_supcon
            + self.SUPCON_WEIGHT * hier_supcon
            + self.CE_WEIGHT * (ce_l0 + ce_l1)
        )

        return total


def run_heal(dataset: str, seed: int = 42, device: str = "cuda", **kwargs) -> Dict:
    """Convenience function to run HEAL baseline."""
    config = ExternalRunConfig(
        method="heal",
        dataset=dataset,
        seed=seed,
        device=device,
        epochs=kwargs.get("epochs", 30),
        lr=kwargs.get("lr", 5e-4),
        batch_size=kwargs.get("batch_size", 128),
    )
    trainer = HEALTrainer(config)
    return trainer.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="clinc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_heal(args.dataset, args.seed)
