"""Train V5/MRL projection heads for BEIR million-scale retrieval.

Strategy:
  1. Train on dbpedia_14 (560K labeled abstracts, 14->6 hierarchy)
  2. Apply learned projection to cached 4.6M BEIR corpus embeddings
  3. Evaluate retrieval at each prefix dimension

This avoids re-encoding the entire corpus. The projection head learns
to reorder dimensions so prefixes capture coarse semantic info.

Usage: python src/beir_v5_train.py
"""

import os
import sys
import json
import time
import gc
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/beir")

# DBPedia-14 class hierarchy
DBPEDIA14_CLASSES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
]

# L0 superclass grouping
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

# Map each L1 class to its L0 integer ID
L1_TO_L0_ID = {L1_TO_ID[l1]: L0_TO_ID[l0] for l1, l0 in L1_TO_L0.items()}

PREFIX_DIMS = [64, 128, 192, 256]
TOTAL_DIM = 256
INPUT_DIM = 384  # bge-small hidden dim
NUM_SCALES = 4
SCALE_DIM = 64
SEEDS = [42, 123, 456]


class EmbeddingDataset(Dataset):
    """Dataset of pre-computed embeddings + hierarchy labels."""
    def __init__(self, embeddings, l0_labels, l1_labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.l0_labels = torch.tensor(l0_labels, dtype=torch.long)
        self.l1_labels = torch.tensor(l1_labels, dtype=torch.long)

    def __len__(self):
        return len(self.l0_labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.l0_labels[idx], self.l1_labels[idx]


class ProjectionHead(nn.Module):
    """V5/MRL projection head: 384d -> 256d with classification heads."""
    def __init__(self, input_dim, total_dim, num_l0, num_l1, scale_dim=64):
        super().__init__()
        self.projection = nn.Linear(input_dim, total_dim)
        self.head_top = nn.Linear(scale_dim, num_l0)
        self.head_full = nn.Linear(total_dim, num_l1)
        self.scale_dim = scale_dim
        self.total_dim = total_dim

    def forward(self, x):
        z = self.projection(x)  # (B, 256)
        return z

    def classify(self, z, prefix_len=None):
        """Get L0 and L1 logits from embedding z."""
        if prefix_len is not None:
            prefix = z[:, :prefix_len * self.scale_dim]
            l0_logits = self.head_top(prefix[:, :self.scale_dim])
        else:
            l0_logits = self.head_top(z[:, :self.scale_dim])
        l1_logits = self.head_full(z)
        return l0_logits, l1_logits


def encode_dbpedia14(split="train", max_samples=50000):
    """Encode dbpedia_14 texts using bge-small and return embeddings + labels."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading dbpedia_14 {split}...")
    ds = load_dataset("dbpedia_14", split=split)

    # Subsample
    if max_samples and len(ds) > max_samples:
        indices = np.random.RandomState(42).choice(len(ds), max_samples, replace=False)
        ds = ds.select(indices)

    texts = [row["content"][:512] for row in ds]
    l1_labels = [row["label"] for row in ds]
    l0_labels = [L1_TO_L0_ID[l1] for l1 in l1_labels]

    print(f"  {len(texts)} texts, {len(L0_NAMES)} L0, {len(DBPEDIA14_CLASSES)} L1")

    # Encode with bge-small
    cache_path = DATA_DIR / f"dbpedia14_{split}_{len(texts)}_embeddings.npy"
    cache_labels = DATA_DIR / f"dbpedia14_{split}_{len(texts)}_labels.npz"

    if cache_path.exists() and cache_labels.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
        lab = np.load(cache_labels)
        return embeddings, lab["l0"], lab["l1"]

    print(f"  Encoding {len(texts)} texts with bge-small...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.eval().half()
    if torch.cuda.is_available():
        model.cuda()

    all_embs = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            if i % (batch_size * 10) == 0:
                print(f"    {i}/{len(texts)}")
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            out = model(**enc)
            embs = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_embs.append(embs)

    embeddings = np.vstack(all_embs)
    l0_arr = np.array(l0_labels)
    l1_arr = np.array(l1_labels)

    np.save(cache_path, embeddings)
    np.savez(cache_labels, l0=l0_arr, l1=l1_arr)
    print(f"  Saved to {cache_path}")

    return embeddings, l0_arr, l1_arr


def train_v5_head(train_embs, train_l0, train_l1,
                  val_embs, val_l0, val_l1,
                  seed=42, epochs=10, lr=1e-3, batch_size=256):
    """Train V5 projection head with hierarchy-aligned prefix supervision."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_l0 = len(L0_NAMES)
    num_l1 = len(DBPEDIA14_CLASSES)

    head = ProjectionHead(INPUT_DIM, TOTAL_DIM, num_l0, num_l1, SCALE_DIM)
    if torch.cuda.is_available():
        head.cuda()

    train_ds = EmbeddingDataset(train_embs, train_l0, train_l1)
    val_ds = EmbeddingDataset(val_embs, val_l0, val_l1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    PREFIX_PROBS = [0.4, 0.3, 0.2, 0.1]
    BLOCK_KEEP = [0.95, 0.9, 0.8, 0.7]
    PREFIX_WEIGHT = 0.6

    best_score = -1
    best_state = None

    for epoch in range(epochs):
        head.train()
        total_loss = 0
        n_batches = 0

        for embs, l0, l1 in train_loader:
            if torch.cuda.is_available():
                embs, l0, l1 = embs.cuda(), l0.cuda(), l1.cuda()

            z = head(embs)  # (B, 256)
            B = z.shape[0]

            # Sample prefix lengths
            probs = torch.tensor(PREFIX_PROBS)
            j = torch.multinomial(probs.expand(B, -1), 1).squeeze(-1) + 1  # 1-4

            # Create prefix embeddings with block dropout
            prefix = z.clone()
            for b in range(B):
                jb = j[b].item()
                # Zero blocks > j
                prefix[b, jb * SCALE_DIM:] = 0
                # Block dropout on kept blocks
                for k in range(jb):
                    if np.random.random() > BLOCK_KEEP[k]:
                        prefix[b, k * SCALE_DIM:(k+1) * SCALE_DIM] = 0

            # L_prefix: L0 classification from first 64d of prefix
            l0_logits = head.head_top(prefix[:, :SCALE_DIM])
            loss_prefix = F.cross_entropy(l0_logits, l0)

            # L_full: L1 classification from full 256d
            l1_logits = head.head_full(z)
            loss_full = F.cross_entropy(l1_logits, l1)

            loss = loss_full + PREFIX_WEIGHT * loss_prefix

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        head.eval()
        l0_correct, l1_correct, total = 0, 0, 0
        with torch.no_grad():
            for embs, l0, l1 in val_loader:
                if torch.cuda.is_available():
                    embs, l0, l1 = embs.cuda(), l0.cuda(), l1.cuda()
                z = head(embs)
                l0_logits = head.head_top(z[:, :SCALE_DIM])
                l1_logits = head.head_full(z)
                l0_correct += (l0_logits.argmax(1) == l0).sum().item()
                l1_correct += (l1_logits.argmax(1) == l1).sum().item()
                total += l0.shape[0]

        l0_acc = l0_correct / total
        l1_acc = l1_correct / total
        score = l0_acc + l1_acc

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        print(f"  V5 Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, "
              f"L0={l0_acc:.4f}, L1={l1_acc:.4f}")

    head.load_state_dict(best_state)
    return head


def train_mrl_head(train_embs, train_l0, train_l1,
                   val_embs, val_l0, val_l1,
                   seed=42, epochs=10, lr=1e-3, batch_size=256):
    """Train MRL projection head (flat L1 loss at all prefix lengths)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_l0 = len(L0_NAMES)
    num_l1 = len(DBPEDIA14_CLASSES)

    head = ProjectionHead(INPUT_DIM, TOTAL_DIM, num_l0, num_l1, SCALE_DIM)
    if torch.cuda.is_available():
        head.cuda()

    train_ds = EmbeddingDataset(train_embs, train_l0, train_l1)
    val_ds = EmbeddingDataset(val_embs, val_l0, val_l1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    PREFIX_PROBS = [0.4, 0.3, 0.2, 0.1]
    MRL_WEIGHT = 0.6

    best_score = -1
    best_state = None

    for epoch in range(epochs):
        head.train()
        total_loss = 0
        n_batches = 0

        for embs, l0, l1 in train_loader:
            if torch.cuda.is_available():
                embs, l0, l1 = embs.cuda(), l0.cuda(), l1.cuda()

            z = head(embs)
            B = z.shape[0]

            # MRL: L1 loss at ALL prefix lengths (flat supervision)
            probs = torch.tensor(PREFIX_PROBS)
            j = torch.multinomial(probs.expand(B, -1), 1).squeeze(-1) + 1

            # Create prefix with zero-out
            prefix = z.clone()
            for b in range(B):
                jb = j[b].item()
                prefix[b, jb * SCALE_DIM:] = 0

            # MRL loss: L1 classification from prefix (FLAT — same target at all scales)
            # Build per-prefix-length classifier output
            # Use head_full on the full embedding (padded with zeros)
            loss_mrl = F.cross_entropy(head.head_full(prefix), l1)

            # Full loss
            loss_full = F.cross_entropy(head.head_full(z), l1)

            loss = loss_full + MRL_WEIGHT * loss_mrl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        head.eval()
        l0_correct, l1_correct, total = 0, 0, 0
        with torch.no_grad():
            for embs, l0, l1 in val_loader:
                if torch.cuda.is_available():
                    embs, l0, l1 = embs.cuda(), l0.cuda(), l1.cuda()
                z = head(embs)
                l0_logits = head.head_top(z[:, :SCALE_DIM])
                l1_logits = head.head_full(z)
                l0_correct += (l0_logits.argmax(1) == l0).sum().item()
                l1_correct += (l1_logits.argmax(1) == l1).sum().item()
                total += l0.shape[0]

        l0_acc = l0_correct / total
        l1_acc = l1_correct / total
        score = l0_acc + l1_acc

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        print(f"  MRL Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, "
              f"L0={l0_acc:.4f}, L1={l1_acc:.4f}")

    head.load_state_dict(best_state)
    return head


def apply_projection(embeddings_fp16, projection_weight, projection_bias):
    """Apply learned projection to full corpus embeddings.

    embeddings_fp16: np.memmap of shape (N, 384), dtype=float16
    projection_weight: (256, 384) numpy array
    projection_bias: (256,) numpy array
    Returns: (N, 256) float16 array
    """
    N = embeddings_fp16.shape[0]
    chunk_size = 100000
    result = np.zeros((N, TOTAL_DIM), dtype=np.float16)

    print(f"  Applying projection to {N} embeddings...")
    for i in range(0, N, chunk_size):
        if i % (chunk_size * 5) == 0:
            print(f"    {i}/{N} ({100*i/N:.1f}%)")
        end = min(i + chunk_size, N)
        chunk = embeddings_fp16[i:end].astype(np.float32)
        projected = chunk @ projection_weight.T + projection_bias
        result[i:end] = projected.astype(np.float16)

    return result


def evaluate_projected_retrieval_fast(doc_ids, corpus_256d, query_ids, query_256d,
                                       qrels, prefix_dims=PREFIX_DIMS):
    """Evaluate retrieval with pre-projected embeddings (no model loading needed)."""
    import faiss

    n_docs = len(doc_ids)
    n_queries = len(query_ids)
    print(f"    Evaluating ({n_docs} docs, {n_queries} queries)...")

    results = {}
    for dim in prefix_dims:
        corpus_prefix = np.ascontiguousarray(corpus_256d[:, :dim].astype(np.float32))
        query_prefix = np.ascontiguousarray(query_256d[:, :dim].astype(np.float32))

        faiss.normalize_L2(corpus_prefix)
        faiss.normalize_L2(query_prefix)

        index = faiss.IndexFlatIP(dim)
        index.add(corpus_prefix)

        t0 = time.time()
        scores, indices = index.search(query_prefix, 100)
        lat = (time.time() - t0) / n_queries * 1000

        ndcg_list, recall_list, map_list = [], [], []
        for qi, qid in enumerate(query_ids):
            if qid not in qrels:
                continue
            rel = qrels[qid]
            retrieved = [doc_ids[idx] for idx in indices[qi] if idx < n_docs]

            dcg = sum((2**rel.get(did, 0) - 1) / np.log2(r + 2)
                      for r, did in enumerate(retrieved[:10]))
            ideal = sorted(rel.values(), reverse=True)[:10]
            idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
            ndcg_list.append(dcg / idcg if idcg > 0 else 0)

            n_rel = sum(1 for r in rel.values() if r > 0)
            n_found = sum(1 for did in retrieved[:100] if rel.get(did, 0) > 0)
            recall_list.append(n_found / n_rel if n_rel > 0 else 0)

            ap, n_rf = 0.0, 0
            for r, did in enumerate(retrieved[:100]):
                if rel.get(did, 0) > 0:
                    n_rf += 1
                    ap += n_rf / (r + 1)
            map_list.append(ap / n_rel if n_rel > 0 else 0)

        ndcg = float(np.mean(ndcg_list))
        results[dim] = {
            "ndcg_at_10": ndcg,
            "recall_at_100": float(np.mean(recall_list)),
            "map": float(np.mean(map_list)),
            "latency_ms": float(lat),
        }
        print(f"      {dim}d: nDCG@10={ndcg:.4f}, R@100={results[dim]['recall_at_100']:.4f}")

        del corpus_prefix, query_prefix, index
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true",
                       help="Skip training, just evaluate from cached projections")
    args = parser.parse_args()

    print("=" * 70)
    print("BEIR DBPedia-Entity V5/MRL Retrieval Benchmark")
    print("=" * 70)

    # Phase 1: Prepare training data
    print("\n[1] Preparing training data from dbpedia_14...")
    train_embs, train_l0, train_l1 = encode_dbpedia14("train", args.train_samples)

    # Split 80/20 for train/val
    n = len(train_l0)
    n_val = n // 5
    perm = np.random.RandomState(42).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr_embs, tr_l0, tr_l1 = train_embs[tr_idx], train_l0[tr_idx], train_l1[tr_idx]
    va_embs, va_l0, va_l1 = train_embs[val_idx], train_l0[val_idx], train_l1[val_idx]

    print(f"  Train: {len(tr_l0)}, Val: {len(va_l0)}")
    print(f"  L0 classes: {len(L0_NAMES)} ({L0_NAMES})")
    print(f"  L1 classes: {len(DBPEDIA14_CLASSES)}")

    # Phase 2: Load BEIR data + encode queries once
    print("\n[2] Loading BEIR corpus + embeddings...")
    from beir.datasets.data_loader import GenericDataLoader
    data_path = str(DATA_DIR / "dbpedia-entity")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    doc_ids = list(corpus.keys())
    n_docs = len(doc_ids)

    memmap_path = DATA_DIR / "corpus_embeddings_fp16.npy"
    corpus_embs = np.memmap(memmap_path, dtype=np.float16, mode='r',
                           shape=(n_docs, INPUT_DIM))
    print(f"  Corpus: {n_docs} docs, Queries: {len(queries)}")

    # Encode queries once (avoids loading model 6 times)
    query_cache_path = DATA_DIR / "query_embeddings_384.npy"
    if query_cache_path.exists():
        print(f"  Loading cached query embeddings from {query_cache_path}")
        query_embs_384 = np.load(query_cache_path)
    else:
        print(f"  Encoding {len(queries)} queries with bge-small...")
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        qmodel = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
        qmodel.eval().half()
        if torch.cuda.is_available():
            qmodel.cuda()

        query_ids_list = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids_list]
        q_embs = []
        with torch.no_grad():
            for i in range(0, len(query_texts), 64):
                batch = query_texts[i:i+64]
                enc = tokenizer(batch, padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
                if torch.cuda.is_available():
                    enc = {k: v.cuda() for k, v in enc.items()}
                out = qmodel(**enc)
                embs = out.last_hidden_state[:, 0, :].cpu().numpy()
                q_embs.append(embs)
        query_embs_384 = np.vstack(q_embs).astype(np.float32)
        np.save(query_cache_path, query_embs_384)
        print(f"  Cached to {query_cache_path}")

        del qmodel
        torch.cuda.empty_cache()
        gc.collect()

    query_ids_ordered = list(queries.keys())

    # Phase 3: Train V5 and MRL heads
    all_results = {"v5": {}, "mrl": {}, "base": {}}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"[3] Seed {seed}: Training projection heads")
        print(f"{'='*70}")

        # V5
        print(f"\n  --- V5 (hierarchy-aligned) ---")
        v5_head = train_v5_head(tr_embs, tr_l0, tr_l1, va_embs, va_l0, va_l1,
                                seed=seed, epochs=args.epochs)

        # Extract projection
        v5_W = v5_head.projection.weight.detach().cpu().numpy()  # (256, 384)
        v5_b = v5_head.projection.bias.detach().cpu().numpy()    # (256,)

        # Apply to BEIR corpus
        print(f"\n  Applying V5 projection to {n_docs} documents...")
        v5_corpus = apply_projection(corpus_embs, v5_W, v5_b)

        # Project queries with same projection
        v5_query_256 = (query_embs_384 @ v5_W.T + v5_b).astype(np.float16)

        # Evaluate
        print(f"\n  Evaluating V5 retrieval...")
        v5_results = evaluate_projected_retrieval_fast(
            doc_ids, v5_corpus, query_ids_ordered, v5_query_256, qrels)
        all_results["v5"][seed] = v5_results

        del v5_head, v5_corpus
        torch.cuda.empty_cache()
        gc.collect()

        # MRL
        print(f"\n  --- MRL (flat supervision) ---")
        mrl_head = train_mrl_head(tr_embs, tr_l0, tr_l1, va_embs, va_l0, va_l1,
                                  seed=seed, epochs=args.epochs)

        mrl_W = mrl_head.projection.weight.detach().cpu().numpy()
        mrl_b = mrl_head.projection.bias.detach().cpu().numpy()

        print(f"\n  Applying MRL projection to {n_docs} documents...")
        mrl_corpus = apply_projection(corpus_embs, mrl_W, mrl_b)

        # Project queries with same projection
        mrl_query_256 = (query_embs_384 @ mrl_W.T + mrl_b).astype(np.float16)

        print(f"\n  Evaluating MRL retrieval...")
        mrl_results = evaluate_projected_retrieval_fast(
            doc_ids, mrl_corpus, query_ids_ordered, mrl_query_256, qrels)
        all_results["mrl"][seed] = mrl_results

        del mrl_head, mrl_corpus
        torch.cuda.empty_cache()
        gc.collect()

    # Save all results
    out = {
        "experiment": "beir_v5_mrl_retrieval",
        "model": "bge-small",
        "dataset": "dbpedia-entity",
        "training_data": "dbpedia_14",
        "n_corpus": n_docs,
        "n_queries": len(queries),
        "seeds": SEEDS,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    results_file = RESULTS_DIR / "beir_v5_mrl_retrieval.json"
    with open(results_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for method in ["v5", "mrl"]:
        print(f"\n  {method.upper()}:")
        for dim in PREFIX_DIMS:
            ndcgs = [all_results[method][s][dim]["ndcg_at_10"] for s in SEEDS]
            print(f"    {dim}d: nDCG@10 = {np.mean(ndcgs):.4f} +/- {np.std(ndcgs):.4f}")




if __name__ == "__main__":
    main()
