"""
V5 Cached: V5 head architecture on cached embeddings
=====================================================

Same architecture as V6Head but WITHOUT the adversary.
This provides a fair apples-to-apples comparison:
- V5 cached vs V6 cached (same framework, same embeddings)
- Shows the pure effect of the gradient-reversal adversary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from .common import (
    BaselineHead,
    BaselineTrainer,
    ExternalRunConfig,
)


class V5CachedHead(BaselineHead):
    """
    V5 head on cached embeddings.
    Matches V6Head architecture minus the adversary.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        num_l0: int = 10,
        num_l1: int = 100,
        scale_dim: int = 64,
        num_scales: int = 4,
    ):
        super().__init__(input_dim, output_dim)
        self.scale_dim = scale_dim
        self.num_scales = num_scales

        self.input_proj = nn.Linear(input_dim, input_dim)

        self.ffn_blocks = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.scale_projs = nn.ModuleList()

        for _ in range(num_scales):
            self.ffn_norms.append(nn.LayerNorm(input_dim))
            self.ffn_blocks.append(nn.Sequential(
                nn.Linear(input_dim, input_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim * 4, input_dim),
            ))
            self.scale_projs.append(nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, scale_dim),
            ))

        self.final_norm = nn.LayerNorm(output_dim)

        self.l0_classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, num_l0),
        )
        self.l1_classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, num_l1),
        )

    def _compute_blocks(self, x: torch.Tensor):
        h = self.input_proj(x)
        blocks = []
        for i in range(self.num_scales):
            normed = self.ffn_norms[i](h)
            h = h + self.ffn_blocks[i](normed)
            block = self.scale_projs[i](h)
            blocks.append(block)
        return blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = self._compute_blocks(x)
        full = torch.cat(blocks, dim=-1)
        full = self.final_norm(full)
        return F.normalize(full, dim=-1)

    def forward_with_blocks(self, x: torch.Tensor):
        blocks = self._compute_blocks(x)
        full = torch.cat(blocks, dim=-1)
        full = self.final_norm(full)
        return F.normalize(full, dim=-1), blocks


class V5CachedTrainer(BaselineTrainer):
    """V5 training on cached embeddings: L0 CE (prefix) + L1 CE (full)."""

    def build_head(self, input_dim: int, num_l0: int, num_l1: int) -> V5CachedHead:
        return V5CachedHead(
            input_dim=input_dim,
            output_dim=self.config.output_dim,
            num_l0=num_l0,
            num_l1=num_l1,
            scale_dim=self.config.scale_dim,
            num_scales=self.config.num_scales,
        )

    def compute_loss(
        self,
        head: V5CachedHead,
        embeddings: torch.Tensor,
        l0_labels: torch.Tensor,
        l1_labels: torch.Tensor,
    ) -> torch.Tensor:
        full_emb, blocks = head.forward_with_blocks(embeddings)

        # L0 CE from prefix (first block, zero-padded)
        prefix_emb = torch.cat([
            blocks[0],
            torch.zeros(
                len(blocks[0]),
                head.output_dim - head.scale_dim,
                device=blocks[0].device,
            )
        ], dim=-1)
        prefix_emb = F.normalize(prefix_emb, dim=-1)
        ce_l0 = F.cross_entropy(head.l0_classifier(prefix_emb), l0_labels)

        # L1 CE from full
        ce_l1 = F.cross_entropy(head.l1_classifier(full_emb), l1_labels)

        return ce_l0 + ce_l1


def run_v5_cached(
    dataset: str,
    seed: int = 42,
    device: str = "cuda",
    **kwargs,
) -> Dict:
    """Run V5 cached baseline."""
    config = ExternalRunConfig(
        method="v5_cached",
        dataset=dataset,
        seed=seed,
        device=device,
        epochs=kwargs.get("epochs", 20),
        lr=kwargs.get("lr", 1e-3),
        batch_size=kwargs.get("batch_size", 64),
    )
    trainer = V5CachedTrainer(config)
    return trainer.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="clinc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_v5_cached(args.dataset, args.seed)
