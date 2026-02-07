"""
Multi-Model Fractal Embedding Pipeline
======================================

Test fractal heads across multiple embedding backbones:
- Qwen3-Embedding
- BGE
- E5
- GTE
- Jina
- Nomic

For each backbone:
1. Load base model
2. Add fractal head
3. Train on hierarchical data
4. Evaluate with/without fractal
5. Compare multi-scale properties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    name: str
    hf_path: str
    hidden_dim: int
    max_seq_len: int
    pooling: str = "cls"  # cls, mean, last
    prefix_query: str = ""
    prefix_doc: str = ""
    trust_remote_code: bool = False


# Model configurations
MODELS = {
    "qwen3-0.6b": ModelConfig(
        name="Qwen3-Embedding-0.6B",
        hf_path="Qwen/Qwen3-Embedding-0.6B",
        hidden_dim=1024,
        max_seq_len=8192,
        pooling="last",
        trust_remote_code=False,
    ),
    "bge-large": ModelConfig(
        name="BGE-Large-v1.5",
        hf_path="BAAI/bge-large-en-v1.5",
        hidden_dim=1024,
        max_seq_len=512,
        pooling="cls",
        prefix_query="Represent this sentence for searching: ",
    ),
    "e5-large": ModelConfig(
        name="E5-Large-v2",
        hf_path="intfloat/e5-large-v2",
        hidden_dim=1024,
        max_seq_len=512,
        pooling="mean",
        prefix_query="query: ",
        prefix_doc="passage: ",
    ),
    "gte-large": ModelConfig(
        name="GTE-Large-v1.5",
        hf_path="Alibaba-NLP/gte-large-en-v1.5",
        hidden_dim=1024,
        max_seq_len=8192,
        pooling="cls",
        trust_remote_code=True,
    ),
    "nomic": ModelConfig(
        name="Nomic-Embed-v1.5",
        hf_path="nomic-ai/nomic-embed-text-v1.5",
        hidden_dim=768,
        max_seq_len=8192,
        pooling="mean",
        prefix_query="search_query: ",
        prefix_doc="search_document: ",
        trust_remote_code=True,
    ),
    # Additional models for comprehensive testing
    "bge-base": ModelConfig(
        name="BGE-Base-v1.5",
        hf_path="BAAI/bge-base-en-v1.5",
        hidden_dim=768,
        max_seq_len=512,
        pooling="cls",
        prefix_query="Represent this sentence for searching: ",
    ),
    "bge-small": ModelConfig(
        name="BGE-Small-v1.5",
        hf_path="BAAI/bge-small-en-v1.5",
        hidden_dim=384,
        max_seq_len=512,
        pooling="cls",
        prefix_query="Represent this sentence for searching: ",
    ),
    "e5-base": ModelConfig(
        name="E5-Base-v2",
        hf_path="intfloat/e5-base-v2",
        hidden_dim=768,
        max_seq_len=512,
        pooling="mean",
        prefix_query="query: ",
        prefix_doc="passage: ",
    ),
    "e5-small": ModelConfig(
        name="E5-Small-v2",
        hf_path="intfloat/e5-small-v2",
        hidden_dim=384,
        max_seq_len=512,
        pooling="mean",
        prefix_query="query: ",
        prefix_doc="passage: ",
    ),
    "minilm": ModelConfig(
        name="MiniLM-L6-v2",
        hf_path="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=384,
        max_seq_len=256,
        pooling="mean",
    ),
    "mpnet": ModelConfig(
        name="MPNet-Base-v2",
        hf_path="sentence-transformers/all-mpnet-base-v2",
        hidden_dim=768,
        max_seq_len=384,
        pooling="mean",
    ),
    # =========================================================================
    # LLM-BASED EMBEDDING MODELS (Decoder→Encoder, SOTA on MTEB)
    # =========================================================================
    "gte-qwen2-1.5b": ModelConfig(
        name="GTE-Qwen2-1.5B-Instruct",
        hf_path="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        hidden_dim=1536,
        max_seq_len=8192,
        pooling="last",
        trust_remote_code=True,
    ),
    "stella-1.5b": ModelConfig(
        name="Stella-EN-1.5B-v5",
        hf_path="dunzhang/stella_en_1.5B_v5",
        hidden_dim=1536,
        max_seq_len=8192,
        pooling="last",
        trust_remote_code=True,
    ),
    "multilingual-e5-large": ModelConfig(
        name="Multilingual-E5-Large-Instruct",
        hf_path="intfloat/multilingual-e5-large-instruct",
        hidden_dim=1024,
        max_seq_len=512,
        pooling="mean",
        prefix_query="query: ",
        prefix_doc="passage: ",
    ),
    "embedding-gemma": ModelConfig(
        name="EmbeddingGemma-300M",
        hf_path="google/embeddinggemma-300m",
        hidden_dim=768,
        max_seq_len=2048,
        pooling="mean",
        trust_remote_code=True,
    ),
    "bge-m3": ModelConfig(
        name="BGE-M3",
        hf_path="BAAI/bge-m3",
        hidden_dim=1024,
        max_seq_len=8192,
        pooling="cls",
        trust_remote_code=True,
    ),
    "nomic-v2": ModelConfig(
        name="Nomic-Embed-Text-v2-MoE",
        hf_path="nomic-ai/nomic-embed-text-v2-moe",
        hidden_dim=768,
        max_seq_len=8192,
        pooling="mean",
        trust_remote_code=True,
    ),
}


class FractalHead(nn.Module):
    """Universal fractal head that works with any backbone."""

    def __init__(
        self,
        input_dim: int,
        num_scales: int = 4,
        scale_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_scales = num_scales
        self.scale_dim = scale_dim
        self.total_dim = num_scales * scale_dim

        # Project input
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Shared fractal block
        self.norm = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
        )
        self.ffn_norm = nn.LayerNorm(input_dim)

        # Scale-specific projections
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, scale_dim),
            )
            for _ in range(num_scales)
        ])

        self.final_norm = nn.LayerNorm(self.total_dim)

    def forward(self, x: torch.Tensor, return_scales: bool = False) -> Dict:
        """
        Args:
            x: (B, D) backbone embedding
        Returns:
            Dict with 'embedding' and optionally 'scale_embeddings'
        """
        # Add sequence dim
        x = x.unsqueeze(1)  # (B, 1, D)
        x = self.input_proj(x)
        x_input = x.clone()

        scale_embeddings = []

        for scale_idx in range(self.num_scales):
            # Self-attention
            normed = self.norm(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out

            # FFN
            x = x + self.ffn(self.ffn_norm(x))

            # Input injection
            injection = 0.1 / (1 + scale_idx * 0.3)
            x = x + injection * x_input

            # Scale output
            scale_emb = self.scale_projs[scale_idx](x.squeeze(1))
            scale_embeddings.append(scale_emb)

        # Full embedding
        full_emb = torch.cat(scale_embeddings, dim=-1)
        full_emb = self.final_norm(full_emb)

        result = {'embedding': full_emb}
        if return_scales:
            result['scale_embeddings'] = scale_embeddings
            result['nested_embeddings'] = [
                torch.cat(scale_embeddings[:i+1], dim=-1)
                for i in range(self.num_scales)
            ]
        return result


class UniversalEmbeddingModel(nn.Module):
    """
    Universal wrapper that loads any embedding model
    and optionally adds a fractal head.
    """

    def __init__(
        self,
        config: ModelConfig,
        use_fractal: bool = True,
        num_scales: int = 4,
        scale_dim: int = 64,
        device: str = "cuda",
    ):
        super().__init__()

        self.config = config
        self.use_fractal = use_fractal
        self.device = device

        # Load backbone
        print(f"Loading {config.name}...")
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.hf_path,
            trust_remote_code=config.trust_remote_code,
        )

        # Set padding side for last-token pooling
        if config.pooling == "last":
            self.tokenizer.padding_side = "left"

        self.backbone = AutoModel.from_pretrained(
            config.hf_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.float16,
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Add fractal head if requested
        if use_fractal:
            self.fractal_head = FractalHead(
                input_dim=config.hidden_dim,
                num_scales=num_scales,
                scale_dim=scale_dim,
            )
            self.embed_dim = num_scales * scale_dim
        else:
            self.fractal_head = None
            self.embed_dim = config.hidden_dim

        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Fractal: {use_fractal}")
        print(f"  Output dim: {self.embed_dim}")

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states according to config."""
        if self.config.pooling == "cls":
            return hidden_states[:, 0]
        elif self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.config.pooling == "last":
            # Last non-padding token
            if self.tokenizer.padding_side == "left":
                return hidden_states[:, -1]
            else:
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                return hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    seq_lens
                ]
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

    def get_backbone_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get embedding from backbone."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        return pooled.float()  # Convert from fp16

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_scales: bool = False,
    ) -> Dict:
        """Forward pass."""
        backbone_emb = self.get_backbone_embedding(input_ids, attention_mask)

        if self.fractal_head is not None:
            result = self.fractal_head(backbone_emb, return_scales)
        else:
            result = {'embedding': backbone_emb}

        # Normalize
        result['embedding'] = F.normalize(result['embedding'], dim=-1)

        return result

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False,
    ) -> torch.Tensor:
        """Encode texts to embeddings."""
        # Add prefixes if needed
        if is_query and self.config.prefix_query:
            texts = [self.config.prefix_query + t for t in texts]
        elif not is_query and self.config.prefix_doc:
            texts = [self.config.prefix_doc + t for t in texts]

        self.eval()
        all_embs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=min(self.config.max_seq_len, 512),  # Limit for speed
                    return_tensors="pt",
                ).to(self.device)

                result = self.forward(inputs['input_ids'], inputs['attention_mask'])
                all_embs.append(result['embedding'].cpu())

        return torch.cat(all_embs, dim=0)


def load_model(
    model_key: str,
    use_fractal: bool = True,
    device: str = "cuda",
) -> UniversalEmbeddingModel:
    """Load a model by key."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    config = MODELS[model_key]
    model = UniversalEmbeddingModel(config, use_fractal=use_fractal, device=device)
    return model.to(device)


def evaluate_on_hierarchical(
    model: UniversalEmbeddingModel,
    dataset,
    device: str = "cuda",
) -> Dict:
    """Evaluate model on hierarchical dataset."""
    from scipy.stats import spearmanr

    # Get samples
    samples = dataset.samples[:min(1000, len(dataset.samples))]

    texts = [s.text for s in samples]
    l0_labels = np.array([s.level0_label for s in samples])
    l1_labels = np.array([s.level1_label for s in samples])

    # Encode
    embeddings = model.encode(texts, batch_size=32).numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute clustering quality
    def compute_separation(emb, labels):
        """Compute intra-class vs inter-class similarity."""
        intra_sims = []
        inter_sims = []

        for i in range(len(emb)):
            for j in range(i + 1, min(i + 100, len(emb))):  # Limit for speed
                sim = (emb[i] * emb[j]).sum()
                if labels[i] == labels[j]:
                    intra_sims.append(sim)
                else:
                    inter_sims.append(sim)

        if intra_sims and inter_sims:
            return np.mean(intra_sims) - np.mean(inter_sims)
        return 0.0

    l0_sep = compute_separation(embeddings, l0_labels)
    l1_sep = compute_separation(embeddings, l1_labels)

    # Classification accuracy (5-NN)
    def knn_accuracy(emb, labels, k=5):
        correct = 0
        for i in range(len(emb)):
            # Find k nearest neighbors (excluding self)
            sims = emb @ emb[i]
            sims[i] = -float('inf')
            top_k = np.argsort(-sims)[:k]
            neighbor_labels = labels[top_k]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            pred = unique[np.argmax(counts)]
            if pred == labels[i]:
                correct += 1
        return correct / len(emb)

    l0_acc = knn_accuracy(embeddings, l0_labels, k=5)
    l1_acc = knn_accuracy(embeddings, l1_labels, k=5)

    return {
        'l0_separation': l0_sep,
        'l1_separation': l1_sep,
        'l0_knn_accuracy': l0_acc,
        'l1_knn_accuracy': l1_acc,
    }


def run_multi_model_comparison(
    model_keys: List[str] = None,
    dataset_name: str = "agnews",
    device: str = "cuda",
):
    """
    Run comparison across multiple models.

    For each model, compare:
    - Base model (no fractal)
    - Model + fractal head
    """
    if model_keys is None:
        model_keys = ["bge-large", "e5-large"]  # Start with reliable ones

    # Load dataset
    from hierarchical_datasets import load_hierarchical_dataset
    print(f"\nLoading {dataset_name} dataset...")
    dataset = load_hierarchical_dataset(dataset_name, split="train", max_samples=2000)
    print(f"  Samples: {len(dataset.samples)}")
    print(f"  L0 categories: {len(dataset.level0_names)}")
    print(f"  L1 categories: {len(dataset.level1_names)}")

    results = {}

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_key}")
        print(f"{'='*60}")

        try:
            # Test without fractal
            print("\nWithout fractal head...")
            model_base = load_model(model_key, use_fractal=False, device=device)
            base_results = evaluate_on_hierarchical(model_base, dataset, device)
            print(f"  L0 separation: {base_results['l0_separation']:.4f}")
            print(f"  L1 separation: {base_results['l1_separation']:.4f}")
            print(f"  L0 5-NN acc: {base_results['l0_knn_accuracy']:.4f}")
            print(f"  L1 5-NN acc: {base_results['l1_knn_accuracy']:.4f}")

            del model_base
            torch.cuda.empty_cache()

            # Test with fractal
            print("\nWith fractal head...")
            model_fractal = load_model(model_key, use_fractal=True, device=device)
            fractal_results = evaluate_on_hierarchical(model_fractal, dataset, device)
            print(f"  L0 separation: {fractal_results['l0_separation']:.4f}")
            print(f"  L1 separation: {fractal_results['l1_separation']:.4f}")
            print(f"  L0 5-NN acc: {fractal_results['l0_knn_accuracy']:.4f}")
            print(f"  L1 5-NN acc: {fractal_results['l1_knn_accuracy']:.4f}")

            del model_fractal
            torch.cuda.empty_cache()

            results[model_key] = {
                'base': base_results,
                'fractal': fractal_results,
            }

        except Exception as e:
            print(f"Error with {model_key}: {e}")
            results[model_key] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Type':<10} {'L0 Sep':<12} {'L1 Sep':<12} {'L0 Acc':<12} {'L1 Acc':<12}")
    print("-" * 78)

    for model_key, res in results.items():
        if 'error' in res:
            print(f"{model_key:<20} ERROR: {res['error'][:40]}")
            continue

        for type_name, metrics in res.items():
            print(f"{model_key:<20} {type_name:<10} {metrics['l0_separation']:<12.4f} "
                  f"{metrics['l1_separation']:<12.4f} {metrics['l0_knn_accuracy']:<12.4f} "
                  f"{metrics['l1_knn_accuracy']:<12.4f}")

    # Save results - convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = Path(__file__).parent.parent / "results" / "multi_model_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line args
    model_keys = ["bge-large", "e5-large"]
    dataset_name = "agnews"

    if len(sys.argv) > 1:
        model_keys = sys.argv[1].split(",")
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]

    results = run_multi_model_comparison(
        model_keys=model_keys,
        dataset_name=dataset_name,
    )
