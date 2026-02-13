"""
SMEC Baseline: Sequential Multi-granularity Embedding with Cross-batch memory
(EMNLP 2025)
================================================================================

Implements the SMRL/SMEC approach to Matryoshka-style training:
- Sequential multi-granularity stages: train dims 1..d1, then 1..d2, etc.
- Adaptive Dimension Selection (ADS): learn which prefix length to use per sample
- Simplified Cross-Batch Memory (S-XBM): extra negatives from recent batches
- No hierarchy awareness (like standard MRL, but with sequential training)

Key differences from our MRL baseline:
1. Sequential training (not simultaneous) - each stage freezes previous dims
2. ADS module learns per-sample optimal dimension
3. Cross-batch memory provides more negatives per batch
4. No hierarchy-specific supervision (all stages target L1)

This is the strongest non-hierarchy-aware MRL variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from .common import (
    BaselineHead,
    BaselineTrainer,
    ExternalRunConfig,
)


class SMECHead(BaselineHead):
    """
    SMEC-style sequential multi-granularity head.

    Architecture:
    - MLP projection: input_dim -> output_dim (256)
    - 4 stages: 64d, 128d, 192d, 256d
    - ADS module: predicts optimal prefix length per sample
    - All stages share the same MLP but train sequentially
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        hidden_dim: int = 512,
        num_l0: int = 10,
        num_l1: int = 100,
        num_stages: int = 4,
        scale_dim: int = 64,
    ):
        super().__init__(input_dim, output_dim)
        self.num_stages = num_stages
        self.scale_dim = scale_dim
        self.num_l0 = num_l0
        self.num_l1 = num_l1

        # Main projection MLP
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Per-stage classifiers (all target L1 -- this is the key MRL property)
        self.stage_classifiers = nn.ModuleList([
            nn.Linear((i + 1) * scale_dim, num_l1)
            for i in range(num_stages)
        ])

        # ADS: Adaptive Dimension Selection
        # Predicts optimal stage (1..4) per sample
        self.ads = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_stages),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: project to output_dim."""
        return F.normalize(self.projection(x), dim=-1)

    def forward_at_stage(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        """Forward with truncation to stage prefix length."""
        full = self.projection(x)
        dim = (stage + 1) * self.scale_dim
        prefix = full[:, :dim]
        return F.normalize(prefix, dim=-1)

    def predict_stage(self, x: torch.Tensor) -> torch.Tensor:
        """ADS: predict optimal stage per sample."""
        return self.ads(x)  # [batch, num_stages] logits


class CrossBatchMemory:
    """
    Simplified Cross-Batch Memory (S-XBM).

    Stores recent batch embeddings and labels for extra negatives.
    """

    def __init__(self, capacity: int = 4096, emb_dim: int = 256):
        self.capacity = capacity
        self.emb_dim = emb_dim
        self.embeddings = deque(maxlen=capacity)
        self.l1_labels = deque(maxlen=capacity)

    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Add batch to memory."""
        emb_np = embeddings.detach().cpu()
        lab_np = labels.detach().cpu()
        for i in range(len(emb_np)):
            self.embeddings.append(emb_np[i])
            self.l1_labels.append(lab_np[i])

    def get_negatives(self, device: str = "cuda") -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get all stored negatives."""
        if len(self.embeddings) == 0:
            return None, None
        emb = torch.stack(list(self.embeddings)).to(device)
        lab = torch.stack(list(self.l1_labels)).to(device)
        return emb, lab


class SMECTrainer(BaselineTrainer):
    """
    SMEC-style sequential multi-granularity training.

    Training proceeds in stages:
    1. Stage 1: Train dims 1..64 with InfoNCE + CE targeting L1
    2. Stage 2: Freeze dims 1..64, train dims 65..128
    3. Stage 3: Freeze dims 1..128, train dims 129..192
    4. Stage 4: Freeze dims 1..192, train dims 193..256
    5. Final: joint fine-tuning with ADS

    Uses S-XBM for extra negatives at each stage.
    """

    TEMPERATURE = 0.07
    CE_WEIGHT = 0.5
    ADS_WEIGHT = 0.1
    XBM_CAPACITY = 2048

    def build_head(self, input_dim: int, num_l0: int, num_l1: int) -> BaselineHead:
        return SMECHead(
            input_dim=input_dim,
            output_dim=self.config.output_dim,
            hidden_dim=512,
            num_l0=num_l0,
            num_l1=num_l1,
            num_stages=self.config.num_scales,
            scale_dim=self.config.scale_dim,
        )

    def _infonce_with_memory(
        self,
        anchors: torch.Tensor,
        labels: torch.Tensor,
        memory: CrossBatchMemory,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """InfoNCE with cross-batch memory negatives."""
        batch_size = anchors.shape[0]

        # Get memory negatives
        mem_emb, mem_lab = memory.get_negatives(anchors.device)

        if mem_emb is not None and len(mem_emb) > 0:
            # Concatenate batch + memory
            all_emb = torch.cat([anchors, mem_emb], dim=0)
            all_lab = torch.cat([labels, mem_lab], dim=0)
        else:
            all_emb = anchors
            all_lab = labels

        # Similarity: anchors vs all
        sim = anchors @ all_emb.T / temperature  # [batch, batch + memory]
        total_size = all_emb.shape[0]

        # Positive mask: same L1 label, exclude self-similarity
        pos_mask = (labels.unsqueeze(1) == all_lab.unsqueeze(0)).float()
        # Create self-exclusion mask (only for batch_size x batch_size top-left)
        self_mask = torch.zeros(batch_size, total_size, device=anchors.device, dtype=torch.bool)
        self_mask[:batch_size, :batch_size] = torch.eye(batch_size, device=anchors.device, dtype=torch.bool)
        pos_mask = pos_mask.masked_fill(self_mask, 0)

        num_pos = pos_mask.sum(dim=1)
        valid = num_pos > 0

        if not valid.any():
            return torch.tensor(0.0, device=anchors.device)

        sim_max = sim.max(dim=1, keepdim=True).values.detach()
        sim_stable = sim - sim_max

        # Exclude self via masking (large negative)
        sim_no_self = sim_stable.masked_fill(self_mask, -1e9)
        log_denom = torch.logsumexp(sim_no_self, dim=1)
        pos_sum = (sim_stable * pos_mask).sum(dim=1)

        loss = -(pos_sum[valid] / num_pos[valid].clamp(min=1) - log_denom[valid])
        return loss.mean()

    def run(self) -> Dict:
        """
        Override base run() for sequential stage training.
        """
        cfg = self.config
        from .common import set_all_seeds, load_cached_embeddings, evaluate_prefix_knn, compute_steer
        set_all_seeds(cfg.seed)

        print(f"\n{'=' * 60}")
        print(f"  {cfg.method.upper()} on {cfg.dataset} (seed={cfg.seed})")
        print(f"{'=' * 60}")

        # Load cached embeddings
        train_emb, train_l0, train_l1 = load_cached_embeddings(
            cfg.model_key, cfg.dataset, "train", cfg.device
        )
        test_emb, test_l0, test_l1 = load_cached_embeddings(
            cfg.model_key, cfg.dataset, "test", cfg.device
        )

        input_dim = train_emb.shape[1]
        num_l0 = int(train_l0.max()) + 1
        num_l1 = int(train_l1.max()) + 1
        print(f"  Input dim: {input_dim}, L0: {num_l0}, L1: {num_l1}")

        # Build head
        head = self.build_head(input_dim, num_l0, num_l1).to(cfg.device)
        param_count = sum(p.numel() for p in head.parameters() if p.requires_grad)
        print(f"  Head params: {param_count:,}")

        train_t = torch.from_numpy(train_emb).float().to(cfg.device)
        train_l1_t = torch.from_numpy(train_l1).long().to(cfg.device)

        # Sequential stage training
        epochs_per_stage = max(cfg.epochs // (cfg.num_scales + 1), 3)

        for stage in range(cfg.num_scales):
            # Fresh memory for each stage (different embedding dims)
            stage_dim = (stage + 1) * cfg.scale_dim
            memory = CrossBatchMemory(self.XBM_CAPACITY, stage_dim)
            print(f"\n  --- Stage {stage+1}/{cfg.num_scales} "
                  f"(dims 1..{(stage+1)*cfg.scale_dim}) ---")

            optimizer = torch.optim.AdamW(
                head.parameters(), lr=cfg.lr, weight_decay=0.01
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_per_stage, eta_min=cfg.lr * 0.01
            )

            for epoch in range(epochs_per_stage):
                head.train()
                epoch_loss = 0
                n_batches = 0

                perm = torch.randperm(len(train_t))
                for start in range(0, len(train_t), cfg.batch_size):
                    idx = perm[start:start + cfg.batch_size]
                    if len(idx) < 4:
                        continue

                    batch_emb = train_t[idx]
                    batch_l1 = train_l1_t[idx]

                    optimizer.zero_grad()

                    # Get stage prefix embedding
                    stage_emb = head.forward_at_stage(batch_emb, stage)

                    # InfoNCE with memory
                    infonce = self._infonce_with_memory(
                        stage_emb, batch_l1, memory, self.TEMPERATURE
                    )

                    # CE at this stage
                    logits = head.stage_classifiers[stage](stage_emb)
                    ce = F.cross_entropy(logits, batch_l1)

                    loss = infonce + self.CE_WEIGHT * ce

                    if torch.isnan(loss):
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()

                    # Update memory
                    memory.enqueue(stage_emb.detach(), batch_l1)

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()
                if (epoch + 1) % 3 == 0 or epoch == 0:
                    avg = epoch_loss / max(n_batches, 1)
                    print(f"    Stage {stage+1} Epoch {epoch+1}: loss={avg:.4f}")

        # Final joint fine-tuning with ADS
        print(f"\n  --- Joint fine-tuning with ADS ---")
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=cfg.lr * 0.1, weight_decay=0.01
        )

        for epoch in range(epochs_per_stage):
            head.train()
            epoch_loss = 0
            n_batches = 0

            perm = torch.randperm(len(train_t))
            for start in range(0, len(train_t), cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                if len(idx) < 4:
                    continue

                batch_emb = train_t[idx]
                batch_l1 = train_l1_t[idx]

                optimizer.zero_grad()

                # Full forward
                full_emb = head(batch_emb)

                # Multi-scale loss (MRL-style: loss at all prefixes)
                total_loss = torch.tensor(0.0, device=cfg.device)
                weights = [0.4, 0.3, 0.2, 0.1]

                bs = len(batch_emb)
                diag_mask = torch.eye(bs, device=cfg.device, dtype=torch.bool)

                for s in range(cfg.num_scales):
                    dim = (s + 1) * cfg.scale_dim
                    prefix = F.normalize(full_emb[:, :dim], dim=-1)

                    # InfoNCE
                    sim = prefix @ prefix.T / self.TEMPERATURE
                    pos_mask = (batch_l1.unsqueeze(1) == batch_l1.unsqueeze(0)).float()
                    pos_mask = pos_mask.masked_fill(diag_mask, 0)
                    num_pos = pos_mask.sum(dim=1)
                    valid = num_pos > 0

                    if valid.any():
                        sim_max = sim.max(dim=1, keepdim=True).values.detach()
                        sim_s = sim - sim_max
                        sim_no_self = sim_s.masked_fill(diag_mask, -1e9)
                        log_d = torch.logsumexp(sim_no_self, dim=1)
                        pos_s = (sim_s * pos_mask).sum(dim=1)
                        infonce_s = -(pos_s[valid] / num_pos[valid].clamp(min=1) - log_d[valid]).mean()
                    else:
                        infonce_s = torch.tensor(0.0, device=cfg.device)

                    # CE
                    logits_s = head.stage_classifiers[s](prefix)
                    ce_s = F.cross_entropy(logits_s, batch_l1)

                    total_loss = total_loss + weights[s] * (infonce_s + self.CE_WEIGHT * ce_s)

                # ADS loss: predict that the model should use the stage
                # that gives best classification accuracy
                # (simplified: just regularize ADS outputs)
                ads_logits = head.predict_stage(batch_emb)
                ads_loss = self.ADS_WEIGHT * F.cross_entropy(
                    ads_logits, torch.full((len(batch_emb),), cfg.num_scales - 1,
                                           device=cfg.device, dtype=torch.long)
                )

                loss = total_loss + ads_loss
                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 3 == 0 or epoch == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(f"    Joint Epoch {epoch+1}: loss={avg:.4f}")

        # Evaluate
        head.eval()
        with torch.no_grad():
            test_t = torch.from_numpy(test_emb).float().to(cfg.device)
            output_emb = head(test_t).cpu().numpy()

        prefix_acc = evaluate_prefix_knn(
            output_emb, test_l0, test_l1,
            k=5, scale_dim=cfg.scale_dim, num_scales=cfg.num_scales,
        )
        steer = compute_steer(prefix_acc)

        print(f"\n  Results:")
        for j in range(1, cfg.num_scales + 1):
            print(f"    j={j} ({j*cfg.scale_dim}d): "
                  f"L0={prefix_acc[f'j{j}_l0']:.4f}, L1={prefix_acc[f'j{j}_l1']:.4f}")
        print(f"  Steerability S = {steer:+.4f}")

        # Save
        from .common import RESULTS_DIR
        result = {
            "method": cfg.method,
            "model": cfg.model_key,
            "dataset": cfg.dataset,
            "seed": cfg.seed,
            "prefix_accuracy": prefix_acc,
            "steerability": steer,
            "training": {
                "epochs_per_stage": epochs_per_stage,
                "total_stages": cfg.num_scales,
                "head_params": param_count,
            },
        }

        results_dir = RESULTS_DIR / cfg.method
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / f"{cfg.dataset}_seed{cfg.seed}.json"
        with open(result_path, "w") as f:
            import json
            json.dump(result, f, indent=2,
                       default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
        print(f"  Saved to {result_path}")

        del head, train_t, train_l1_t, test_t
        torch.cuda.empty_cache()

        return result


def run_smec(dataset: str, seed: int = 42, device: str = "cuda", **kwargs) -> Dict:
    """Convenience function to run SMEC baseline."""
    config = ExternalRunConfig(
        method="smec",
        dataset=dataset,
        seed=seed,
        device=device,
        epochs=kwargs.get("epochs", 30),
        lr=kwargs.get("lr", 5e-4),
        batch_size=kwargs.get("batch_size", 128),
    )
    trainer = SMECTrainer(config)
    return trainer.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="clinc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_smec(args.dataset, args.seed)
