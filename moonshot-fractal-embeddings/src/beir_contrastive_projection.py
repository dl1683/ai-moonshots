"""Retrieval-aware projection for BEIR million-scale benchmark.

Strategy (per Codex review):
  Phase 1: SVD baseline - optimal 256d linear projection from 384d
           (preserves maximum variance, CPU-only)
  Phase 2: Hierarchy-aligned projection - SVD init + fine-tune with
           distance preservation + hierarchy prefix loss
  Phase 3: Apply to 4.6M corpus + evaluate retrieval

Usage: python src/beir_contrastive_projection.py
"""

import os
import sys
import json
import time
import gc
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/beir")
PREFIX_DIMS = [64, 128, 192, 256]
TOTAL_DIM = 256
INPUT_DIM = 384
SCALE_DIM = 64

# DBPedia-14 hierarchy
DBPEDIA14_CLASSES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
]
L1_TO_L0 = {
    "Company": "Organisation", "EducationalInstitution": "Organisation",
    "Artist": "Person", "Athlete": "Person", "OfficeHolder": "Person",
    "MeanOfTransportation": "Transport",
    "Building": "Place", "NaturalPlace": "Place", "Village": "Place",
    "Animal": "Nature", "Plant": "Nature",
    "Album": "Work", "Film": "Work", "WrittenWork": "Work",
}
L0_NAMES = sorted(set(L1_TO_L0.values()))
L0_TO_ID = {n: i for i, n in enumerate(L0_NAMES)}
L1_TO_ID = {n: i for i, n in enumerate(DBPEDIA14_CLASSES)}
L1_TO_L0_ID = {L1_TO_ID[l1]: L0_TO_ID[l0] for l1, l0 in L1_TO_L0.items()}


def compute_corpus_svd(corpus_path, n_components=256, sample_size=500000):
    """Compute PCA/SVD projection from 384d -> n_components using corpus stats.

    Uses incremental covariance estimation on a subsample for memory efficiency.
    """
    print(f"=== Phase 1: SVD Baseline ===")
    print(f"Loading corpus from {corpus_path}...")

    corpus = np.memmap(corpus_path, dtype=np.float16, mode='r').reshape(-1, INPUT_DIM)
    n_total = corpus.shape[0]
    print(f"  Corpus: {n_total} x {INPUT_DIM}")

    # Subsample for covariance estimation (500K is plenty for 384d)
    rng = np.random.RandomState(42)
    if n_total > sample_size:
        indices = rng.choice(n_total, sample_size, replace=False)
        indices.sort()
        print(f"  Subsampling {sample_size} vectors for covariance...")
        X = corpus[indices].astype(np.float32)
    else:
        X = np.array(corpus[:], dtype=np.float32)

    # Compute mean and covariance
    print("  Computing mean + covariance...")
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD on centered data
    print(f"  Computing SVD (top {n_components} components)...")
    from scipy.linalg import svd
    # Use covariance approach: faster for n >> d
    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top n_components
    W = eigenvectors[:, :n_components]  # (384, 256)

    # Explained variance
    total_var = eigenvalues.sum()
    explained = eigenvalues[:n_components].sum() / total_var
    print(f"  Explained variance: {explained:.4f} ({n_components}/{INPUT_DIM} dims)")

    # Variance by prefix
    for pd in PREFIX_DIMS:
        ev = eigenvalues[:pd].sum() / total_var
        print(f"    {pd}d: {ev:.4f}")

    return W.T.astype(np.float32), mean.astype(np.float32), eigenvalues


def evaluate_retrieval(doc_ids, corpus_projected, query_ids, query_projected,
                       qrels, prefix_dims=PREFIX_DIMS):
    """Evaluate retrieval at each prefix dimension using FAISS."""
    import faiss

    results = {}
    for pd in prefix_dims:
        print(f"  Evaluating retrieval at {pd}d...")

        # Get prefix embeddings
        corpus_prefix = np.ascontiguousarray(corpus_projected[:, :pd].astype(np.float32))
        query_prefix = np.ascontiguousarray(query_projected[:, :pd].astype(np.float32))

        # L2 normalize for cosine similarity
        corpus_norms = np.linalg.norm(corpus_prefix, axis=1, keepdims=True)
        corpus_norms = np.maximum(corpus_norms, 1e-8)
        corpus_prefix = corpus_prefix / corpus_norms

        query_norms = np.linalg.norm(query_prefix, axis=1, keepdims=True)
        query_norms = np.maximum(query_norms, 1e-8)
        query_prefix = query_prefix / query_norms

        # Build FAISS index
        index = faiss.IndexFlatIP(pd)
        index.add(corpus_prefix)

        # Search
        k = 100
        t0 = time.time()
        scores, indices = index.search(query_prefix, k)
        latency = (time.time() - t0) / len(query_prefix) * 1000

        # Compute metrics
        ndcg_10 = compute_ndcg(query_ids, doc_ids, indices, scores, qrels, k=10)
        recall_100 = compute_recall(query_ids, doc_ids, indices, qrels, k=100)

        results[str(pd)] = {
            "ndcg_at_10": ndcg_10,
            "recall_at_100": recall_100,
            "latency_ms": latency,
            "n_queries": len(query_ids),
        }
        print(f"    nDCG@10={ndcg_10:.4f}, R@100={recall_100:.4f}, lat={latency:.1f}ms")

        del corpus_prefix, query_prefix, index
        gc.collect()

    return results


def compute_ndcg(query_ids, doc_ids, indices, scores, qrels, k=10):
    """Compute nDCG@k."""
    ndcg_sum = 0
    n_queries = 0

    for qi, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel = qrels[qid]

        # DCG
        dcg = 0
        for rank in range(min(k, indices.shape[1])):
            did = doc_ids[indices[qi, rank]]
            if did in rel:
                dcg += rel[did] / np.log2(rank + 2)

        # Ideal DCG
        ideal_rels = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))

        if idcg > 0:
            ndcg_sum += dcg / idcg
            n_queries += 1

    return ndcg_sum / max(n_queries, 1)


def compute_recall(query_ids, doc_ids, indices, qrels, k=100):
    """Compute Recall@k."""
    recall_sum = 0
    n_queries = 0

    for qi, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel = qrels[qid]
        n_rel = len(rel)

        retrieved = set()
        for rank in range(min(k, indices.shape[1])):
            did = doc_ids[indices[qi, rank]]
            if did in rel:
                retrieved.add(did)

        if n_rel > 0:
            recall_sum += len(retrieved) / n_rel
            n_queries += 1

    return recall_sum / max(n_queries, 1)


def load_beir_data():
    """Load BEIR DBPedia-Entity corpus embeddings, queries, qrels."""
    from beir.datasets.data_loader import GenericDataLoader

    # Load corpus via BEIR
    data_path = str(DATA_DIR / "dbpedia-entity")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    doc_ids = list(corpus.keys())
    query_ids = list(queries.keys())

    # Load corpus embeddings (memmap)
    corpus_path = DATA_DIR / "corpus_embeddings_fp16.npy"
    print("Loading corpus embeddings...")
    corpus_384 = np.memmap(str(corpus_path), dtype=np.float16, mode='r',
                           shape=(len(doc_ids), INPUT_DIM))
    print(f"  Corpus: {corpus_384.shape[0]} docs, {corpus_384.shape[1]}d")

    # Load query embeddings
    query_emb_path = DATA_DIR / "query_embeddings_384.npy"
    if query_emb_path.exists():
        query_384 = np.load(str(query_emb_path)).astype(np.float32)
    else:
        raise FileNotFoundError(f"Query embeddings not found at {query_emb_path}")

    print(f"  Queries: {query_384.shape[0]}, {query_384.shape[1]}d")
    print(f"  Qrels: {len(qrels)} queries")

    return corpus_384, doc_ids, query_384, query_ids, qrels


class ContrastiveProjectionHead(nn.Module):
    """Projection head with distance preservation + hierarchy alignment."""

    def __init__(self, input_dim, output_dim, num_l0, num_l1, scale_dim=64,
                 svd_init=None):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.head_l0 = nn.Linear(scale_dim, num_l0)
        self.head_l1 = nn.Linear(output_dim, num_l1)
        self.scale_dim = scale_dim
        self.output_dim = output_dim

        if svd_init is not None:
            # Initialize with SVD projection (256 x 384)
            with torch.no_grad():
                self.projection.weight.copy_(torch.tensor(svd_init, dtype=torch.float32))

    def forward(self, x):
        return self.projection(x)

    def classify(self, z):
        l0_logits = self.head_l0(z[:, :self.scale_dim])
        l1_logits = self.head_l1(z)
        return l0_logits, l1_logits


def train_contrastive_projection(W_svd, mean, train_embs, train_l0, train_l1,
                                  val_embs, val_l0, val_l1,
                                  alpha_preserve=1.0, alpha_hierarchy=0.1,
                                  epochs=20, lr=1e-4, batch_size=512, seed=42):
    """Train hierarchy-aligned projection with distance preservation.

    Loss = alpha_preserve * L_preserve + alpha_hierarchy * L_hierarchy

    L_preserve: MSE between original and projected cosine similarities
    L_hierarchy: V5-style prefix classification (L0 at 64d, L1 at 256d)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_l0 = len(L0_NAMES)
    num_l1 = len(DBPEDIA14_CLASSES)

    head = ContrastiveProjectionHead(
        INPUT_DIM, TOTAL_DIM, num_l0, num_l1, SCALE_DIM,
        svd_init=W_svd  # (256, 384)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head = head.to(device)

    # Center embeddings
    train_centered = train_embs - mean
    val_centered = val_embs - mean

    # Create datasets
    train_x = torch.tensor(train_centered, dtype=torch.float32)
    train_y0 = torch.tensor(train_l0, dtype=torch.long)
    train_y1 = torch.tensor(train_l1, dtype=torch.long)
    val_x = torch.tensor(val_centered, dtype=torch.float32)
    val_y0 = torch.tensor(val_l0, dtype=torch.long)
    val_y1 = torch.tensor(val_l1, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(train_x, train_y0, train_y1)
    val_ds = torch.utils.data.TensorDataset(val_x, val_y0, val_y1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    PREFIX_PROBS = [0.4, 0.3, 0.2, 0.1]
    PREFIX_WEIGHT = 0.6

    best_score = -1
    best_state = None

    for epoch in range(epochs):
        head.train()
        total_loss = 0
        total_preserve = 0
        total_hierarchy = 0
        n_batches = 0

        for x, y0, y1 in train_loader:
            x, y0, y1 = x.to(device), y0.to(device), y1.to(device)
            B = x.shape[0]

            z = head(x)  # (B, 256)

            # === Distance preservation loss ===
            # Compute cosine similarities in original 384d space
            x_norm = F.normalize(x, dim=1)
            # Random pairs within batch
            perm = torch.randperm(B, device=device)
            cos_orig = (x_norm * x_norm[perm]).sum(dim=1)  # (B,)

            # Cosine similarities in projected 256d space
            z_norm = F.normalize(z, dim=1)
            cos_proj = (z_norm * z_norm[perm]).sum(dim=1)  # (B,)

            loss_preserve = F.mse_loss(cos_proj, cos_orig)

            # === Hierarchy classification loss (V5-style) ===
            # L0 at prefix (64d)
            l0_logits = head.head_l0(z[:, :SCALE_DIM])
            loss_l0 = F.cross_entropy(l0_logits, y0)

            # L1 at full (256d)
            l1_logits = head.head_l1(z)
            loss_l1 = F.cross_entropy(l1_logits, y1)

            loss_hierarchy = loss_l1 + PREFIX_WEIGHT * loss_l0

            # Total loss
            loss = alpha_preserve * loss_preserve + alpha_hierarchy * loss_hierarchy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_preserve += loss_preserve.item()
            total_hierarchy += loss_hierarchy.item()
            n_batches += 1

        scheduler.step()

        # Validate
        head.eval()
        l0_correct, l1_correct, total = 0, 0, 0
        val_preserve_loss = 0
        n_val = 0
        with torch.no_grad():
            for x, y0, y1 in val_loader:
                x, y0, y1 = x.to(device), y0.to(device), y1.to(device)
                z = head(x)

                # Classification accuracy
                l0_logits = head.head_l0(z[:, :SCALE_DIM])
                l1_logits = head.head_l1(z)
                l0_correct += (l0_logits.argmax(1) == y0).sum().item()
                l1_correct += (l1_logits.argmax(1) == y1).sum().item()
                total += y0.shape[0]

                # Preserve loss
                x_norm = F.normalize(x, dim=1)
                z_norm = F.normalize(z, dim=1)
                perm = torch.randperm(x.shape[0], device=device)
                cos_orig = (x_norm * x_norm[perm]).sum(dim=1)
                cos_proj = (z_norm * z_norm[perm]).sum(dim=1)
                val_preserve_loss += F.mse_loss(cos_proj, cos_orig).item()
                n_val += 1

        l0_acc = l0_correct / total
        l1_acc = l1_correct / total
        avg_preserve = val_preserve_loss / max(n_val, 1)

        score = l0_acc + l1_acc - 5.0 * avg_preserve  # Heavily penalize similarity distortion

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f} "
              f"(preserve={total_preserve/n_batches:.4f}, hier={total_hierarchy/n_batches:.4f}) "
              f"L0={l0_acc:.4f}, L1={l1_acc:.4f}, sim_dist={avg_preserve:.6f}")

    head.load_state_dict(best_state)
    return head


def project_corpus(corpus_384, projection_matrix, mean, batch_size=100000):
    """Apply projection to entire corpus in batches. Returns float16."""
    n_total = corpus_384.shape[0]
    n_out = projection_matrix.shape[0]

    result = np.empty((n_total, n_out), dtype=np.float16)

    for i in range(0, n_total, batch_size):
        end = min(i + batch_size, n_total)
        batch = corpus_384[i:end].astype(np.float32) - mean
        projected = batch @ projection_matrix.T  # (batch, 256)
        result[i:end] = projected.astype(np.float16)

        if i % (batch_size * 5) == 0:
            print(f"  Projected {i}/{n_total}...")

    print(f"  Done: {n_total} vectors projected to {n_out}d")
    return result


def main():
    print("=" * 70)
    print("BEIR RETRIEVAL-AWARE PROJECTION EXPERIMENT")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print()
    sys.stdout.flush()

    # Load BEIR data
    corpus_384, doc_ids, query_384, query_ids, qrels = load_beir_data()

    # ====== Phase 1: SVD Baseline ======
    corpus_path = DATA_DIR / "corpus_embeddings_fp16.npy"
    W_svd, mean, eigenvalues = compute_corpus_svd(str(corpus_path), n_components=TOTAL_DIM)
    # W_svd is (256, 384)

    print("\nProjecting corpus with SVD...")
    sys.stdout.flush()
    corpus_svd = project_corpus(corpus_384, W_svd, mean)

    print("\nProjecting queries with SVD...")
    query_svd = ((query_384 - mean) @ W_svd.T).astype(np.float16)

    print("\n--- SVD Baseline Retrieval ---")
    sys.stdout.flush()
    svd_results = evaluate_retrieval(doc_ids, corpus_svd, query_ids, query_svd, qrels)

    # ====== Phase 2: Hierarchy-Aligned Projection ======
    print("\n=== Phase 2: Hierarchy-Aligned Projection ===")
    sys.stdout.flush()

    # Load training data
    train_emb_path = DATA_DIR / "dbpedia14_train_50000_embeddings.npy"
    train_lab_path = DATA_DIR / "dbpedia14_train_50000_labels.npz"

    if not train_emb_path.exists():
        print("Training data not found. Run beir_v5_train.py first to cache embeddings.")
        sys.stdout.flush()
        return

    train_embs = np.load(str(train_emb_path))
    train_labs = np.load(str(train_lab_path))
    train_l0, train_l1 = train_labs["l0"], train_labs["l1"]

    # 90/10 train/val split
    n = len(train_embs)
    n_val = n // 10
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    print(f"Training: {len(train_idx)}, Validation: {len(val_idx)}")
    sys.stdout.flush()

    # Sweep alpha_hierarchy to find best retrieval-hierarchy tradeoff
    results_all = {"svd_baseline": svd_results}

    alphas = [0.01, 0.05, 0.1]
    for alpha_h in alphas:
        print(f"\n--- Training with alpha_hierarchy={alpha_h} ---")
        sys.stdout.flush()

        head = train_contrastive_projection(
            W_svd, mean,
            train_embs[train_idx], train_l0[train_idx], train_l1[train_idx],
            train_embs[val_idx], train_l0[val_idx], train_l1[val_idx],
            alpha_preserve=1.0, alpha_hierarchy=alpha_h,
            epochs=15, lr=5e-5, batch_size=512, seed=42
        )

        # Extract projection matrix
        W_aligned = head.projection.weight.detach().cpu().numpy()  # (256, 384)

        # Project corpus and queries
        print(f"\nProjecting corpus (alpha_h={alpha_h})...")
        sys.stdout.flush()
        corpus_aligned = project_corpus(corpus_384, W_aligned, mean)

        query_centered = query_384 - mean
        query_aligned = (query_centered @ W_aligned.T).astype(np.float16)

        print(f"\n--- Retrieval (alpha_h={alpha_h}) ---")
        sys.stdout.flush()
        aligned_results = evaluate_retrieval(
            doc_ids, corpus_aligned, query_ids, query_aligned, qrels
        )

        results_all[f"aligned_alpha_{alpha_h}"] = aligned_results

        # Cleanup
        del corpus_aligned, query_aligned, head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save all results
    output = {
        "experiment": "beir_contrastive_projection",
        "timestamp": datetime.now().isoformat(),
        "model": "bge-small",
        "corpus_size": len(doc_ids),
        "n_queries": len(query_ids),
        "method": "svd_init_distance_preserve_hierarchy_align",
        "results": results_all,
    }

    output_path = RESULTS_DIR / "beir_contrastive_projection.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'64d':>8} {'128d':>8} {'192d':>8} {'256d':>8}")
    print("-" * 70)

    for method, res in results_all.items():
        vals = [res[str(pd)]["ndcg_at_10"] for pd in PREFIX_DIMS]
        print(f"{method:<30} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f}")

    # Compare to base model
    base_results = {
        "64": 0.1818, "128": 0.3012, "192": 0.3358, "256": 0.3561
    }
    vals = [base_results[str(pd)] for pd in PREFIX_DIMS]
    print(f"{'base_model_truncation':<30} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
