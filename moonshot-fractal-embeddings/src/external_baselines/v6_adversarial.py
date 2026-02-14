"""
V6: V5 + Conditional Gradient-Reversal Leakage Adversary
=========================================================

Codex-designed architecture (Feb 13, 2026):
- Same V5 projection head: 384d -> 256d (4 blocks x 64d)
- Train-only adversary predicts L1 from prefix M1 (64d) conditioned on L0
- Gradient reversal layer (GRL) suppresses fine info leaking into prefix
- Grounded in Theorem 3: S = H(L1|L0) - I(M1; L1|L0) - Delta
  Maximizing S requires minimizing I(M1; L1|L0), which GRL achieves

Architecture:
  V5 head: frozen_emb (384d) -> 4x FFN blocks -> 4x scale_proj (64d each) = 256d
  Adversary: GRL(M1=block0, 64d) concat L0_embed(16d) -> MLP(80->128->K1)
  Loss: L_V6 = L_V5 + lambda_adv * CE(adversary_masked, L1)

Inference: Adversary is discarded. Output is identical to V5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from torch.autograd import Function

from .common import (
    BaselineHead,
    BaselineTrainer,
    ExternalRunConfig,
    evaluate_prefix_knn,
    compute_steer,
    load_cached_embeddings,
    set_all_seeds,
    RESULTS_DIR,
)


# ============================================================================
# Gradient Reversal Layer (Ganin et al., 2016)
# ============================================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ============================================================================
# V6 Head: V5 architecture + leakage adversary
# ============================================================================

class V6Head(BaselineHead):
    """
    V6 projection head = V5 head + conditional leakage adversary.

    V5 component (kept at inference):
    - 4 FFN blocks: each applies LayerNorm -> Linear(384->1536) -> GELU -> Linear(1536->384)
    - 4 scale projections: LayerNorm(384) -> Linear(384->64)
    - L0 classifier on prefix (zero-padded 256d)
    - L1 classifier on full 256d

    Adversary (train-only, discarded at inference):
    - Input: GRL(M1) = first 64d block after gradient reversal
    - Concatenated with learned L0 embedding (16d)
    - MLP: 80 -> 128 -> K1 (num fine classes)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        num_l0: int = 10,
        num_l1: int = 100,
        scale_dim: int = 64,
        num_scales: int = 4,
        l0_embed_dim: int = 16,
        adv_hidden: int = 128,
        adv_dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)
        self.scale_dim = scale_dim
        self.num_scales = num_scales
        self.num_l0 = num_l0
        self.num_l1 = num_l1

        # ---- V5 component: FFN blocks + scale projections ----
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

        # V5 classifiers
        self.l0_classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, num_l0),
        )
        self.l1_classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, num_l1),
        )

        # ---- V6 adversary: conditional leakage predictor ----
        self.grl = GradientReversalLayer(alpha=1.0)
        self.l0_embed = nn.Embedding(num_l0, l0_embed_dim)
        self.adversary = nn.Sequential(
            nn.Linear(scale_dim + l0_embed_dim, adv_hidden),
            nn.ReLU(),
            nn.Dropout(adv_dropout),
            nn.Linear(adv_hidden, num_l1),
        )

        # Build L0->L1 compatibility mask (filled during first forward)
        self.register_buffer(
            "l0_l1_mask",
            torch.zeros(num_l0, num_l1, dtype=torch.bool),
        )
        self._mask_built = False

    def build_hierarchy_mask(self, l0_labels: torch.Tensor, l1_labels: torch.Tensor):
        """Build L0->L1 compatibility mask from training data."""
        if self._mask_built:
            return
        for l0, l1 in zip(l0_labels.cpu().numpy(), l1_labels.cpu().numpy()):
            self.l0_l1_mask[int(l0), int(l1)] = True
        self._mask_built = True

    def _compute_blocks(self, x: torch.Tensor):
        """Compute scale blocks from frozen embeddings."""
        h = self.input_proj(x)  # [batch, input_dim]
        blocks = []

        for i in range(self.num_scales):
            normed = self.ffn_norms[i](h)
            h = h + self.ffn_blocks[i](normed)
            block = self.scale_projs[i](h)  # [batch, scale_dim]
            blocks.append(block)

        return blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: returns L2-normalized 256d embedding."""
        blocks = self._compute_blocks(x)
        full = torch.cat(blocks, dim=-1)
        full = self.final_norm(full)
        return F.normalize(full, dim=-1)

    def forward_with_adversary(
        self, x: torch.Tensor, l0_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Training forward: returns (full_emb, adversary_logits, blocks).

        The adversary takes GRL(M1) + L0_embed and predicts L1.
        Gradient from adversary is reversed before flowing into M1,
        which pushes M1 to NOT encode L1 information.
        """
        blocks = self._compute_blocks(x)
        full = torch.cat(blocks, dim=-1)
        full = self.final_norm(full)
        full_norm = F.normalize(full, dim=-1)

        # Adversary input: GRL(first block) + L0 embedding
        m1 = blocks[0]                         # [batch, 64]
        m1_reversed = self.grl(m1)             # gradient reversed
        l0_emb = self.l0_embed(l0_labels)      # [batch, 16]
        adv_input = torch.cat([m1_reversed, l0_emb], dim=-1)  # [batch, 80]
        adv_logits = self.adversary(adv_input)  # [batch, num_l1]

        # Mask logits to valid L1 classes for each L0
        if self._mask_built:
            mask = self.l0_l1_mask[l0_labels]  # [batch, num_l1]
            adv_logits = adv_logits.masked_fill(~mask, -1e9)

        return full_norm, adv_logits, blocks

    def get_prefix(self, x: torch.Tensor, j: int, scale_dim: int = 64) -> torch.Tensor:
        """Get first j*scale_dim dims (for prefix evaluation)."""
        full = self.forward(x)
        return full[:, :j * scale_dim]


# ============================================================================
# V6 Trainer
# ============================================================================

class V6Trainer(BaselineTrainer):
    """
    V6 trainer: V5 losses + adversarial leakage minimization.

    Loss = L_V5 + lambda_adv * CE(adversary, L1)
    where L_V5 = CE(L0_classifier(prefix), L0) + CE(L1_classifier(full), L1)

    The GRL ensures that gradient from the adversary loss REVERSES direction
    when flowing into M1 (first block), pushing M1 to encode LESS L1 info.
    """

    # Hyperparameters
    LAMBDA_ADV = 0.2        # Adversary loss weight
    ADV_WARMUP_EPOCHS = 1   # Epochs before adversary kicks in
    TEMPERATURE = 0.07

    def __init__(self, config: ExternalRunConfig, lambda_adv: float = 0.2):
        super().__init__(config)
        self.lambda_adv = lambda_adv

    def build_head(self, input_dim: int, num_l0: int, num_l1: int) -> V6Head:
        return V6Head(
            input_dim=input_dim,
            output_dim=self.config.output_dim,
            num_l0=num_l0,
            num_l1=num_l1,
            scale_dim=self.config.scale_dim,
            num_scales=self.config.num_scales,
        )

    def compute_loss(
        self,
        head: V6Head,
        embeddings: torch.Tensor,
        l0_labels: torch.Tensor,
        l1_labels: torch.Tensor,
    ) -> torch.Tensor:
        """V6 loss = V5 CE losses + adversarial leakage penalty."""
        # Build hierarchy mask on first call
        head.build_hierarchy_mask(l0_labels, l1_labels)

        # Forward with adversary
        full_emb, adv_logits, blocks = head.forward_with_adversary(embeddings, l0_labels)

        # V5 L0 CE: classify L0 from prefix (first block, zero-padded)
        prefix_emb = torch.cat([
            blocks[0],
            torch.zeros(len(blocks[0]), head.output_dim - head.scale_dim, device=blocks[0].device)
        ], dim=-1)
        prefix_emb = F.normalize(prefix_emb, dim=-1)
        l0_logits = head.l0_classifier(prefix_emb)
        ce_l0 = F.cross_entropy(l0_logits, l0_labels)

        # V5 L1 CE: classify L1 from full embedding
        l1_logits = head.l1_classifier(full_emb)
        ce_l1 = F.cross_entropy(l1_logits, l1_labels)

        # V5 base loss
        loss_v5 = ce_l0 + ce_l1

        # Adversarial loss (GRL already applied in forward_with_adversary)
        ce_adv = F.cross_entropy(adv_logits, l1_labels)
        loss = loss_v5 + self.lambda_adv * ce_adv

        return loss

    def run(self) -> Dict:
        """Full training + evaluation pipeline with adversary warmup."""
        cfg = self.config
        set_all_seeds(cfg.seed)

        print(f"\n{'=' * 60}")
        print(f"  V6 (Adversarial) on {cfg.dataset} (seed={cfg.seed}, "
              f"lambda={self.lambda_adv})")
        print(f"{'=' * 60}")

        # Load cached embeddings
        print("  Loading cached embeddings...")
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
        print(f"  Train: {len(train_emb)}, Test: {len(test_emb)}")

        # Build head
        head = self.build_head(input_dim, num_l0, num_l1).to(cfg.device)
        param_count = sum(p.numel() for p in head.parameters() if p.requires_grad)
        v5_params = param_count - sum(
            p.numel() for p in list(head.adversary.parameters()) +
            list(head.l0_embed.parameters()) + list(head.grl.parameters())
            if p.requires_grad
        )
        adv_params = param_count - v5_params
        print(f"  Total params: {param_count:,} (V5 head: {v5_params:,}, "
              f"adversary: {adv_params:,})")

        # Separate optimizers for head and adversary
        head_params = (
            list(head.input_proj.parameters()) +
            list(head.ffn_blocks.parameters()) +
            list(head.ffn_norms.parameters()) +
            list(head.scale_projs.parameters()) +
            list(head.final_norm.parameters()) +
            list(head.l0_classifier.parameters()) +
            list(head.l1_classifier.parameters())
        )
        adv_params_list = (
            list(head.adversary.parameters()) +
            list(head.l0_embed.parameters())
        )

        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": cfg.lr},
            {"params": adv_params_list, "lr": cfg.lr * 2},  # 2x LR for adversary
        ], weight_decay=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )

        train_t = torch.from_numpy(train_emb).float().to(cfg.device)
        train_l0_t = torch.from_numpy(train_l0).long().to(cfg.device)
        train_l1_t = torch.from_numpy(train_l1).long().to(cfg.device)

        # Build hierarchy mask from all training data
        head.build_hierarchy_mask(train_l0_t, train_l1_t)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = 10

        for epoch in range(cfg.epochs):
            head.train()

            # Adversary warmup: ramp lambda from 0 to target
            if epoch < self.ADV_WARMUP_EPOCHS:
                current_lambda = self.lambda_adv * (epoch + 1) / self.ADV_WARMUP_EPOCHS
            else:
                current_lambda = self.lambda_adv

            # Update GRL alpha based on training progress
            p = epoch / cfg.epochs
            head.grl.alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # DANN schedule

            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(len(train_t))
            for start in range(0, len(train_t), cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                if len(idx) < 4:
                    continue

                batch_emb = train_t[idx]
                batch_l0 = train_l0_t[idx]
                batch_l1 = train_l1_t[idx]

                optimizer.zero_grad()

                # Forward with adversary
                full_emb, adv_logits, blocks = head.forward_with_adversary(
                    batch_emb, batch_l0
                )

                # V5 losses
                prefix_emb = torch.cat([
                    blocks[0],
                    torch.zeros(
                        len(blocks[0]),
                        head.output_dim - head.scale_dim,
                        device=blocks[0].device,
                    )
                ], dim=-1)
                prefix_emb = F.normalize(prefix_emb, dim=-1)

                ce_l0 = F.cross_entropy(head.l0_classifier(prefix_emb), batch_l0)
                ce_l1 = F.cross_entropy(head.l1_classifier(full_emb), batch_l1)

                # Adversarial loss
                ce_adv = F.cross_entropy(adv_logits, batch_l1)

                loss = ce_l0 + ce_l1 + current_lambda * ce_adv

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{cfg.epochs}: loss={avg_loss:.4f}, "
                      f"lambda={current_lambda:.3f}, grl_alpha={head.grl.alpha:.3f}")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best
        if best_state is not None:
            head.load_state_dict(best_state)

        # Evaluate (adversary not used - standard forward only)
        head.eval()
        with torch.no_grad():
            test_t = torch.from_numpy(test_emb).float().to(cfg.device)
            output_emb = head(test_t).cpu().numpy()

        prefix_acc = evaluate_prefix_knn(
            output_emb, test_l0, test_l1,
            k=5, scale_dim=cfg.scale_dim, num_scales=cfg.num_scales,
        )
        steer = compute_steer(prefix_acc)

        print(f"  Results:")
        for j in range(1, cfg.num_scales + 1):
            print(f"    j={j} ({j*cfg.scale_dim}d): "
                  f"L0={prefix_acc[f'j{j}_l0']:.4f}, L1={prefix_acc[f'j{j}_l1']:.4f}")
        print(f"  Steerability S = {steer:+.4f}")

        # Save results
        result = {
            "method": "v6_adversarial",
            "model": cfg.model_key,
            "dataset": cfg.dataset,
            "seed": cfg.seed,
            "lambda_adv": self.lambda_adv,
            "prefix_accuracy": prefix_acc,
            "steerability": steer,
            "training": {
                "epochs_run": epoch + 1,
                "best_loss": best_loss,
                "total_params": param_count,
                "v5_head_params": v5_params,
                "adversary_params": adv_params,
            },
        }

        results_dir = RESULTS_DIR / "v6"
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / f"{cfg.dataset}_seed{cfg.seed}_lambda{self.lambda_adv}.json"
        with open(result_path, "w") as f:
            import json
            json.dump(result, f, indent=2,
                      default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
        print(f"  Saved to {result_path}")

        # Cleanup
        del head, train_t, train_l0_t, train_l1_t, test_t
        torch.cuda.empty_cache()

        return result


# ============================================================================
# Convenience functions
# ============================================================================

def run_v6(
    dataset: str,
    seed: int = 42,
    lambda_adv: float = 0.2,
    device: str = "cuda",
    **kwargs,
) -> Dict:
    """Run V6 adversarial baseline."""
    config = ExternalRunConfig(
        method="v6_adversarial",
        dataset=dataset,
        seed=seed,
        device=device,
        epochs=kwargs.get("epochs", 20),
        lr=kwargs.get("lr", 1e-3),
        batch_size=kwargs.get("batch_size", 64),
    )
    trainer = V6Trainer(config, lambda_adv=lambda_adv)
    return trainer.run()


def sweep_lambda(
    dataset: str = "clinc",
    seed: int = 42,
    lambdas: list = None,
    device: str = "cuda",
) -> list:
    """Sweep lambda_adv values on a single dataset."""
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.5]

    results = []
    for lam in lambdas:
        print(f"\n--- Lambda = {lam} ---")
        r = run_v6(dataset, seed=seed, lambda_adv=lam, device=device)
        results.append(r)
        print(f"  S = {r['steerability']:+.4f}")

    print(f"\n{'=' * 60}")
    print(f"Lambda sweep summary ({dataset}, seed={seed}):")
    print(f"{'Lambda':>8} | {'Steerability':>13} | {'j1_L0':>6} | {'j4_L1':>6}")
    print(f"{'-'*45}")
    for r in results:
        print(f"{r['lambda_adv']:>8.2f} | {r['steerability']:>+13.4f} | "
              f"{r['prefix_accuracy']['j1_l0']:>6.4f} | "
              f"{r['prefix_accuracy']['j4_l1']:>6.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="clinc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-adv", type=float, default=0.2)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep lambda values")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.sweep:
        sweep_lambda(args.dataset, args.seed, device=args.device)
    else:
        run_v6(args.dataset, args.seed, args.lambda_adv, device=args.device)
