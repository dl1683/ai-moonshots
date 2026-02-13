"""
Common framework for external baseline comparisons.
====================================================

Provides:
- ExternalRunConfig: run configuration dataclass
- Frozen embedding cache (bge-small 384d)
- k-NN evaluation at all prefix lengths (j1=64d..j4=256d)
- Steerability metric computation
- Seed management
- Per-dataset sample limits
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Ensure src/ is on path (resolve to avoid '..' in module __file__ paths)
_SRC_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ============================================================================
# Configuration
# ============================================================================

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache_embeddings"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "external_baselines"

# Per-dataset sample limits (match V5/MRL experiments exactly)
DATASET_LIMITS = {
    "yahoo":          {"max_train": 10000, "max_test": 2000},
    "goemotions":     {"max_train": 10000, "max_test": 2000},
    "newsgroups":     {"max_train": 10000, "max_test": 2000},
    "trec":           {"max_train": 10000, "max_test": 2000},
    "arxiv":          {"max_train": 10000, "max_test": 2000},
    "dbpedia_classes": {"max_train": 10000, "max_test": 2000},
    "clinc":          {"max_train": 10000, "max_test": 2000},
    "wos":            {"max_train": 10000, "max_test": 2000},
    "hupd_sec_cls":   {"max_train": 15000, "max_test": 3000},
    "hupd_sec_sub":   {"max_train": 30000, "max_test": 5000},
    "hwv_l0_l2":      {"max_train": 5635,  "max_test": 2000},
    "hwv_l0_l3":      {"max_train": 3402,  "max_test": 1500},
}

ALL_DATASETS = list(DATASET_LIMITS.keys())
ALL_SEEDS = [42, 123, 456, 789, 1024]


@dataclass
class ExternalRunConfig:
    """Configuration for an external baseline run."""
    method: str                     # e.g., "heal", "csr", "smec"
    model_key: str = "bge-small"
    dataset: str = "clinc"
    seed: int = 42
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    device: str = "cuda"
    output_dim: int = 256           # Must match V5 (4 blocks x 64d)
    num_scales: int = 4
    scale_dim: int = 64


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Frozen Embedding Cache
# ============================================================================

def _get_cache_path(model_key: str, dataset: str, split: str) -> Path:
    """Get cache file path for frozen embeddings."""
    cache_dir = CACHE_DIR / model_key / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{split}.npz"


def _encode_with_model(model_key: str, texts: List[str], device: str = "cuda",
                        batch_size: int = 64) -> np.ndarray:
    """Encode texts using a frozen backbone model."""
    from multi_model_pipeline import MODELS, ModelConfig
    from transformers import AutoModel, AutoTokenizer

    config = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_path, trust_remote_code=config.trust_remote_code
    )
    if config.pooling == "last":
        tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        config.hf_path, trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.float32
    ).to(device).eval()

    if config.prefix_query:
        texts = [config.prefix_query + t for t in texts]

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=min(config.max_seq_len, 512),
                return_tensors="pt"
            ).to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            hidden = outputs.last_hidden_state

            # Pool
            if config.pooling == "cls":
                pooled = hidden[:, 0]
            elif config.pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            elif config.pooling == "last":
                pooled = hidden[:, -1]
            else:
                pooled = hidden[:, 0]

            all_embs.append(pooled.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


def load_cached_embeddings(
    model_key: str, dataset: str, split: str, device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached frozen embeddings for a dataset split.

    Returns: (embeddings [N, hidden_dim], l0_labels [N], l1_labels [N])
    """
    cache_path = _get_cache_path(model_key, dataset, split)

    if cache_path.exists():
        data = np.load(cache_path)
        return data["embeddings"], data["l0_labels"], data["l1_labels"]

    # Cache miss: compute and save
    print(f"  Cache miss for {model_key}/{dataset}/{split}, computing...")
    from hierarchical_datasets import load_hierarchical_dataset

    limits = DATASET_LIMITS.get(dataset, {"max_train": 10000, "max_test": 2000})
    max_samples = limits["max_train"] if split == "train" else limits["max_test"]

    ds = load_hierarchical_dataset(dataset, split=split, max_samples=max_samples)
    texts = [s.text for s in ds.samples]
    l0_labels = np.array([s.level0_label for s in ds.samples])
    l1_labels = np.array([s.level1_label for s in ds.samples])

    embeddings = _encode_with_model(model_key, texts, device=device)

    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        l0_labels=l0_labels,
        l1_labels=l1_labels,
    )
    print(f"  Cached {len(embeddings)} embeddings to {cache_path}")

    return embeddings, l0_labels, l1_labels


def load_split_cached_embeddings(
    dataset: str, split: str, model_key: str = "bge-small",
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper matching Codex's API specification."""
    return load_cached_embeddings(model_key, dataset, split, device)


def get_num_classes(dataset: str, model_key: str = "bge-small",
                    device: str = "cuda") -> Tuple[int, int]:
    """Get (num_l0_classes, num_l1_classes) for a dataset."""
    _, l0, l1 = load_cached_embeddings(model_key, dataset, "train", device)
    return int(l0.max()) + 1, int(l1.max()) + 1


# ============================================================================
# Evaluation: k-NN at all prefix lengths
# ============================================================================

def evaluate_prefix_knn(
    embeddings: np.ndarray,
    l0_labels: np.ndarray,
    l1_labels: np.ndarray,
    k: int = 5,
    scale_dim: int = 64,
    num_scales: int = 4,
) -> Dict[str, float]:
    """
    Evaluate k-NN accuracy at each prefix length j in {1,2,3,4}.

    Args:
        embeddings: [N, output_dim] where output_dim = num_scales * scale_dim
        l0_labels: [N] coarse labels
        l1_labels: [N] fine labels
        k: number of neighbors
        scale_dim: dimension per scale block (64)
        num_scales: number of scale blocks (4)

    Returns:
        Dict with j1_l0, j1_l1, j2_l0, j2_l1, ..., j4_l0, j4_l1
    """
    results = {}
    n = len(embeddings)

    for j in range(1, num_scales + 1):
        dim = j * scale_dim
        prefix_emb = embeddings[:, :dim].copy()

        # L2-normalize
        norms = np.linalg.norm(prefix_emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        prefix_emb = prefix_emb / norms

        # Compute similarity matrix
        sims = prefix_emb @ prefix_emb.T

        # k-NN for L0
        l0_correct = 0
        l1_correct = 0
        for i in range(n):
            sims[i, i] = -float("inf")
            top_k = np.argpartition(-sims[i], k)[:k]

            # L0 prediction (majority vote)
            l0_neighbors = l0_labels[top_k]
            vals, counts = np.unique(l0_neighbors, return_counts=True)
            if l0_labels[i] == vals[np.argmax(counts)]:
                l0_correct += 1

            # L1 prediction (majority vote)
            l1_neighbors = l1_labels[top_k]
            vals, counts = np.unique(l1_neighbors, return_counts=True)
            if l1_labels[i] == vals[np.argmax(counts)]:
                l1_correct += 1

        results[f"j{j}_l0"] = l0_correct / n
        results[f"j{j}_l1"] = l1_correct / n

    return results


def compute_steer(prefix_accuracy: Dict[str, float]) -> float:
    """
    Compute steerability metric S from prefix accuracy dict.

    S = (L0@j1 - L0@j4) + (L1@j4 - L1@j1)

    Positive S means the model exhibits scale separation:
    coarse info concentrated in early dimensions, fine info in later.
    """
    return (
        (prefix_accuracy.get("j1_l0", 0) - prefix_accuracy.get("j4_l0", 0))
        + (prefix_accuracy.get("j4_l1", 0) - prefix_accuracy.get("j1_l1", 0))
    )


# ============================================================================
# Base Trainer Class
# ============================================================================

class BaselineHead(nn.Module):
    """Base class for external baseline projection heads.

    All baselines share the same interface:
    - Input: frozen backbone embeddings (384d for bge-small)
    - Output: multi-resolution embeddings (256d = 4 x 64d)
    - Evaluation: k-NN at each prefix length
    """

    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map frozen embeddings to output space. Returns [batch, output_dim]."""
        raise NotImplementedError

    def get_prefix(self, x: torch.Tensor, j: int, scale_dim: int = 64) -> torch.Tensor:
        """Get first j*scale_dim dimensions of the output."""
        full = self.forward(x)
        return full[:, :j * scale_dim]


class BaselineTrainer:
    """Generic training loop for baseline heads on cached embeddings."""

    def __init__(self, config: ExternalRunConfig):
        self.config = config
        self.device = config.device

    def build_head(self, input_dim: int, num_l0: int, num_l1: int) -> BaselineHead:
        """Override in subclass to build the method-specific head."""
        raise NotImplementedError

    def compute_loss(
        self,
        head: BaselineHead,
        embeddings: torch.Tensor,
        l0_labels: torch.Tensor,
        l1_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Override in subclass to compute method-specific loss."""
        raise NotImplementedError

    def run(self) -> Dict:
        """Full training + evaluation pipeline."""
        cfg = self.config
        set_all_seeds(cfg.seed)

        print(f"\n{'=' * 60}")
        print(f"  {cfg.method.upper()} on {cfg.dataset} (seed={cfg.seed})")
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
        print(f"  Head params: {param_count:,}")

        # Train
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )

        train_t = torch.from_numpy(train_emb).float().to(cfg.device)
        train_l0_t = torch.from_numpy(train_l0).long().to(cfg.device)
        train_l1_t = torch.from_numpy(train_l1).long().to(cfg.device)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = 10

        for epoch in range(cfg.epochs):
            head.train()
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle
            perm = torch.randperm(len(train_t))
            for start in range(0, len(train_t), cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                if len(idx) < 4:
                    continue

                batch_emb = train_t[idx]
                batch_l0 = train_l0_t[idx]
                batch_l1 = train_l1_t[idx]

                optimizer.zero_grad()
                loss = self.compute_loss(head, batch_emb, batch_l0, batch_l1)

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
                print(f"    Epoch {epoch+1}/{cfg.epochs}: loss={avg_loss:.4f}")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best
        if best_state is not None:
            head.load_state_dict(best_state)

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

        print(f"  Results:")
        for j in range(1, cfg.num_scales + 1):
            print(f"    j={j} ({j*cfg.scale_dim}d): "
                  f"L0={prefix_acc[f'j{j}_l0']:.4f}, L1={prefix_acc[f'j{j}_l1']:.4f}")
        print(f"  Steerability S = {steer:+.4f}")

        # Save results
        result = {
            "method": cfg.method,
            "model": cfg.model_key,
            "dataset": cfg.dataset,
            "seed": cfg.seed,
            "prefix_accuracy": prefix_acc,
            "steerability": steer,
            "training": {
                "epochs_run": epoch + 1,
                "best_loss": best_loss,
                "head_params": param_count,
            },
        }

        results_dir = RESULTS_DIR / cfg.method
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / f"{cfg.dataset}_seed{cfg.seed}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
        print(f"  Saved to {result_path}")

        # Cleanup
        del head, train_t, train_l0_t, train_l1_t, test_t
        torch.cuda.empty_cache()

        return result
