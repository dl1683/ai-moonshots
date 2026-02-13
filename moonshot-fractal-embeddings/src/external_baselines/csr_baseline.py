"""
CSR Baseline: Contextual Sparse Representations (ICML 2025 / CSRv2 ICLR 2026)
================================================================================

Implements sparse autoencoder + top-k sparsification approach:
- Encoder: MLP producing over-complete representation
- Sparsification: top-k activation per sample (k controls dimensionality)
- Decoder: reconstruct original embedding from sparse code
- Classification: linear probes on sparse codes at various k

Key CSR/CSRv2 design principles:
1. Sparse overcomplete codes are inherently multi-resolution:
   top-k with small k captures coarse features, large k adds fine detail
2. Reconstruction loss preserves information
3. Sparsity pressure encourages interpretable dimensions
4. CSRv2 adds progressive k-annealing during training

We use this as a baseline because sparsity naturally provides
dimensionality reduction without explicit hierarchy awareness.
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
    evaluate_prefix_knn,
    compute_steer,
)


class CSRHead(BaselineHead):
    """
    CSR-style sparse autoencoder head.

    Architecture:
    - Encoder: input_dim -> overcomplete_dim (e.g., 1024)
    - Top-k sparsification: keep only top-k activations
    - Projection: overcomplete_dim -> output_dim (256)
    - Decoder: output_dim -> input_dim (reconstruction)

    For multi-resolution evaluation, we:
    - Sort dimensions by average activation magnitude (most important first)
    - First 64d = most activated = coarsest features
    - Last 64d = least activated = finest features

    This gives a natural ordering for prefix evaluation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        overcomplete_dim: int = 1024,
        sparsity_k: int = 64,
        num_l0: int = 10,
        num_l1: int = 100,
    ):
        super().__init__(input_dim, output_dim)
        self.overcomplete_dim = overcomplete_dim
        self.sparsity_k = sparsity_k

        # Encoder: dense -> overcomplete sparse
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, overcomplete_dim),
            nn.LayerNorm(overcomplete_dim),
            nn.ReLU(),  # ReLU for natural sparsity
        )

        # Projection: overcomplete -> output_dim
        self.projector = nn.Sequential(
            nn.Linear(overcomplete_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Decoder: output_dim -> input_dim (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
        )

        # Classification heads
        self.l0_classifier = nn.Linear(output_dim, num_l0)
        self.l1_classifier = nn.Linear(output_dim, num_l1)

        # Learnable importance ordering (CSRv2 idea)
        # Initialized to uniform; training will differentiate
        self.dim_importance = nn.Parameter(torch.zeros(output_dim))

    def _topk_sparse(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k sparsification: keep only k largest activations."""
        if k >= x.shape[-1]:
            return x

        # Get top-k indices
        _, topk_idx = torch.topk(x.abs(), k, dim=-1)

        # Create sparse mask
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_idx, 1.0)

        return x * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, sparsify, project to output_dim."""
        # Encode to overcomplete
        sparse_code = self.encoder(x)

        # Top-k sparsification
        sparse_code = self._topk_sparse(sparse_code, self.sparsity_k)

        # Project to output dim
        projected = self.projector(sparse_code)

        # Apply importance-based reordering
        # Sort dims by learned importance (descending) so most important = first
        importance_order = torch.argsort(self.dim_importance, descending=True)
        projected = projected[:, importance_order]

        return F.normalize(projected, dim=-1)

    def forward_with_reconstruction(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (output_emb, reconstruction, sparse_code) for loss."""
        sparse_code = self.encoder(x)
        sparse_code = self._topk_sparse(sparse_code, self.sparsity_k)
        projected = self.projector(sparse_code)

        # Reorder by importance
        importance_order = torch.argsort(self.dim_importance, descending=True)
        projected_ordered = projected[:, importance_order]
        output = F.normalize(projected_ordered, dim=-1)

        # Reconstruct from projected
        reconstruction = self.decoder(projected)

        return output, reconstruction, sparse_code


class CSRTrainer(BaselineTrainer):
    """CSR-style training with reconstruction + sparsity + classification."""

    RECON_WEIGHT = 1.0
    SPARSITY_WEIGHT = 0.01  # L1 penalty on sparse codes
    CE_WEIGHT = 0.5
    CONTRASTIVE_WEIGHT = 0.5
    TEMPERATURE = 0.1

    def build_head(self, input_dim: int, num_l0: int, num_l1: int) -> BaselineHead:
        return CSRHead(
            input_dim=input_dim,
            output_dim=self.config.output_dim,
            overcomplete_dim=min(1024, input_dim * 3),
            sparsity_k=max(64, input_dim // 4),
            num_l0=num_l0,
            num_l1=num_l1,
        )

    def compute_loss(
        self,
        head: CSRHead,
        embeddings: torch.Tensor,
        l0_labels: torch.Tensor,
        l1_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        CSR loss = Reconstruction + Sparsity_L1 + CE_L0 + CE_L1 + Contrastive

        1. Reconstruction: MSE between original and decoded
        2. Sparsity: L1 norm of overcomplete sparse codes
        3. CE: classification at both levels from output embedding
        4. Contrastive: InfoNCE on full output for fine-grained clustering
        """
        output, reconstruction, sparse_code = head.forward_with_reconstruction(embeddings)

        # 1. Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, embeddings)

        # 2. Sparsity penalty (L1 on sparse activations)
        sparsity_loss = sparse_code.abs().mean()

        # 3. Classification
        ce_l0 = F.cross_entropy(head.l0_classifier(output), l0_labels)
        ce_l1 = F.cross_entropy(head.l1_classifier(output), l1_labels)

        # 4. InfoNCE contrastive (same L1 = positive)
        batch_size = output.shape[0]
        sim = output @ output.T / self.TEMPERATURE
        # Use L1 labels for positives
        pos_mask = (l1_labels.unsqueeze(1) == l1_labels.unsqueeze(0)).float()
        pos_mask.fill_diagonal_(0)

        num_pos = pos_mask.sum(dim=1)
        valid = num_pos > 0

        if valid.any():
            sim_max = sim.max(dim=1, keepdim=True).values.detach()
            sim_stable = sim - sim_max
            exp_sim = torch.exp(sim_stable)
            exp_sim.fill_diagonal_(0)
            log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)
            pos_sum = (sim_stable * pos_mask).sum(dim=1)
            contrastive = -(pos_sum[valid] / num_pos[valid].clamp(min=1) - log_denom[valid]).mean()
        else:
            contrastive = torch.tensor(0.0, device=embeddings.device)

        total = (
            self.RECON_WEIGHT * recon_loss
            + self.SPARSITY_WEIGHT * sparsity_loss
            + self.CE_WEIGHT * (ce_l0 + ce_l1)
            + self.CONTRASTIVE_WEIGHT * contrastive
        )

        return total


def run_csr(dataset: str, seed: int = 42, device: str = "cuda", **kwargs) -> Dict:
    """Convenience function to run CSR baseline."""
    config = ExternalRunConfig(
        method="csr",
        dataset=dataset,
        seed=seed,
        device=device,
        epochs=kwargs.get("epochs", 30),
        lr=kwargs.get("lr", 5e-4),
        batch_size=kwargs.get("batch_size", 128),
    )
    trainer = CSRTrainer(config)
    return trainer.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="clinc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_csr(args.dataset, args.seed)
