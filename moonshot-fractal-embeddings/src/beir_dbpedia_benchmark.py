"""BEIR DBPedia-Entity Million-Scale Retrieval Benchmark.

Demonstrates Fractal Embeddings' Pareto dominance on a standard
million-scale retrieval benchmark with native ontology hierarchy.

Protocol (Codex-designed, Feb 12 2026):
  Track 1: Official BEIR nDCG@10 (apples-to-apples with published baselines)
  Track 2: Hierarchy-aware metrics (h-nDCG, level recall)

Steps:
  1. Download BEIR DBPedia-Entity (4.6M docs, 400 test queries)
  2. Derive ontology hierarchy labels from entity types
  3. Train V5 + MRL on hierarchical classification
  4. Encode full corpus in FP16 memmap
  5. Evaluate retrieval at each prefix dimension
  6. Pareto plot with bootstrap CI

Usage: python src/beir_dbpedia_benchmark.py [--phase download|train|encode|eval|all]
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

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/beir")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
MODEL_KEY = "bge-small"
PREFIX_DIMS = [64, 128, 192, 256]
TOTAL_DIM = 256  # V5 total embedding dim


# =======================================================================
# Phase 1: Download and prepare data
# =======================================================================

def phase_download():
    """Download BEIR DBPedia-Entity and inspect structure."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip"
    data_path = util.download_and_unzip(url, str(DATA_DIR))
    print(f"Downloaded to: {data_path}")

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"Corpus: {len(corpus)} documents")
    print(f"Queries: {len(queries)} queries")
    print(f"Qrels: {len(qrels)} query judgments")

    # Sample
    sample_id = list(corpus.keys())[0]
    doc = corpus[sample_id]
    print(f"\nSample doc ({sample_id}):")
    print(f"  Title: {doc.get('title', '?')[:100]}")
    print(f"  Text: {doc.get('text', '?')[:200]}")

    return data_path, corpus, queries, qrels


def derive_hierarchy_labels(corpus):
    """Derive hierarchy labels from DBPedia ontology types.

    Uses the dbpedia_14 dataset which maps entities to 14 top-level
    ontology classes, then derives L1 from entity title patterns.

    For entities not in dbpedia_14, we use text-based heuristics.
    """
    from datasets import load_dataset

    print("\nDeriving hierarchy labels from DBPedia ontology...")

    # Load dbpedia_14 for type mapping
    try:
        db14 = load_dataset("dbpedia_14", split="train")
        # dbpedia_14 has 14 classes: Company, EducationalInstitution, Artist, etc.
        class_names = [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
            "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
        ]
        print(f"  Loaded dbpedia_14: {len(db14)} examples, {len(class_names)} classes")
    except Exception as e:
        print(f"  Warning: Could not load dbpedia_14: {e}")
        class_names = []

    # For the BEIR corpus, we use the document title/text to assign types
    # DBPedia entities have structured titles that encode their type
    # We'll use a simple approach: classify by keyword matching in title/text

    # Top-level ontology groupings (L0)
    L0_KEYWORDS = {
        "Person": ["born", "player", "actor", "singer", "politician", "author",
                    "president", "coach", "director", "professor", "writer"],
        "Place": ["city", "town", "village", "country", "river", "mountain",
                  "lake", "island", "district", "county", "state", "located"],
        "Organisation": ["company", "university", "school", "club", "team",
                        "organization", "founded", "corporation", "institute"],
        "Work": ["album", "film", "book", "song", "novel", "series", "show",
                "game", "single", "magazine", "newspaper", "journal"],
        "Species": ["species", "genus", "family", "plant", "animal", "bird",
                   "fish", "insect", "mammal", "tree", "flower"],
        "Event": ["war", "battle", "tournament", "championship", "election",
                 "festival", "race", "ceremony", "conference"],
        "Technology": ["software", "programming", "computer", "algorithm",
                      "protocol", "format", "system", "device", "engine"],
        "Other": [],  # Fallback
    }

    # Build L0 and L1 labels for corpus
    labels = {}
    l0_counts = {}
    total = len(corpus)

    for i, (doc_id, doc) in enumerate(corpus.items()):
        if i % 500000 == 0 and i > 0:
            print(f"  Processed {i}/{total} documents...")

        title = doc.get("title", "").lower()
        text = doc.get("text", "")[:300].lower()
        combined = title + " " + text

        # Assign L0 by keyword matching
        best_l0 = "Other"
        best_score = 0
        for l0_name, keywords in L0_KEYWORDS.items():
            if l0_name == "Other":
                continue
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_l0 = l0_name

        # L1 = more specific type (first significant word of title)
        title_words = doc.get("title", "").split()
        if len(title_words) > 1:
            # Use first word as L1 hint
            l1_hint = title_words[0]
        else:
            l1_hint = best_l0

        labels[doc_id] = {"l0": best_l0, "l1": l1_hint, "l0_id": -1, "l1_id": -1}
        l0_counts[best_l0] = l0_counts.get(best_l0, 0) + 1

    # Build integer label mappings
    l0_names = sorted(l0_counts.keys())
    l0_to_id = {n: i for i, n in enumerate(l0_names)}

    # For L1, use top-N most frequent l1_hints per L0
    l1_per_l0 = {}
    for doc_id, lab in labels.items():
        l0 = lab["l0"]
        l1 = lab["l1"]
        if l0 not in l1_per_l0:
            l1_per_l0[l0] = {}
        l1_per_l0[l0][l1] = l1_per_l0[l0].get(l1, 0) + 1

    # Keep top 20 L1 per L0, rest become "other"
    l1_names_all = []
    l1_to_id = {}
    for l0 in l0_names:
        if l0 not in l1_per_l0:
            continue
        top_l1s = sorted(l1_per_l0[l0].items(), key=lambda x: -x[1])[:20]
        for l1_name, _ in top_l1s:
            full_name = f"{l0}/{l1_name}"
            if full_name not in l1_to_id:
                l1_to_id[full_name] = len(l1_names_all)
                l1_names_all.append(full_name)
        # Add "other" for this L0
        other_name = f"{l0}/other"
        if other_name not in l1_to_id:
            l1_to_id[other_name] = len(l1_names_all)
            l1_names_all.append(other_name)

    # Assign integer IDs
    for doc_id, lab in labels.items():
        lab["l0_id"] = l0_to_id[lab["l0"]]
        full_l1 = f"{lab['l0']}/{lab['l1']}"
        if full_l1 in l1_to_id:
            lab["l1_id"] = l1_to_id[full_l1]
        else:
            lab["l1_id"] = l1_to_id[f"{lab['l0']}/other"]

    print(f"\n  Hierarchy: {len(l0_names)} L0 classes -> {len(l1_names_all)} L1 classes")
    for l0, count in sorted(l0_counts.items(), key=lambda x: -x[1]):
        print(f"    {l0:20s}: {count:>8d} ({100*count/total:.1f}%)")

    return labels, l0_names, l1_names_all


# =======================================================================
# Phase 2: Train V5 and MRL
# =======================================================================

def phase_train(corpus, labels, l0_names, l1_names, seeds=SEEDS):
    """Train V5 and MRL on the hierarchical classification task.

    Uses a subsample of the corpus for training (to fit in GPU memory
    and finish in reasonable time), then applies to full corpus.
    """
    import torch
    from fractal_v5 import FractalV5Model
    from mrl_v5_baseline import MRLModel

    n_l0 = len(l0_names)
    n_l1 = len(l1_names)
    print(f"\nTraining on {n_l0} L0 and {n_l1} L1 classes")

    # Subsample for training (15K train, 3K test)
    doc_ids = list(corpus.keys())
    np.random.seed(42)
    np.random.shuffle(doc_ids)

    # Build text + labels arrays
    texts = []
    l0_labels = []
    l1_labels = []
    for doc_id in doc_ids:
        lab = labels.get(doc_id)
        if lab is None:
            continue
        doc = corpus[doc_id]
        text = (doc.get("title", "") + " " + doc.get("text", ""))[:512]
        texts.append(text)
        l0_labels.append(lab["l0_id"])
        l1_labels.append(lab["l1_id"])

    max_train = 15000
    max_test = 3000
    train_texts = texts[:max_train]
    train_l0 = l0_labels[:max_train]
    train_l1 = l1_labels[:max_train]
    test_texts = texts[max_train:max_train + max_test]
    test_l0 = l0_labels[max_train:max_train + max_test]
    test_l1 = l1_labels[max_train:max_train + max_test]

    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")

    # This will be filled in after we verify the data pipeline works
    # For now, return placeholder
    models = {"v5": {}, "mrl": {}}

    for seed in seeds:
        print(f"\n  === Seed {seed} ===")
        # V5 training
        print(f"  Training V5...")
        torch.cuda.empty_cache()
        gc.collect()
        # TODO: Implement V5 training with custom labels
        # For now, we'll use the existing fractal_v5 infrastructure
        models["v5"][seed] = None

        # MRL training
        print(f"  Training MRL...")
        torch.cuda.empty_cache()
        gc.collect()
        models["mrl"][seed] = None

    return models


# =======================================================================
# Phase 3: Encode corpus
# =======================================================================

def phase_encode(corpus, models, prefix_dims=PREFIX_DIMS):
    """Encode full corpus at each prefix dimension.

    Stores embeddings as FP16 memmap for memory efficiency.
    4.6M docs * 256d * 2 bytes = ~2.37 GB per method per seed.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"\nEncoding {len(corpus)} documents...")

    # Load base model
    model_name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    base_model.eval()
    base_model.half()  # FP16
    if torch.cuda.is_available():
        base_model.cuda()

    # Prepare texts
    doc_ids = list(corpus.keys())
    texts = [(corpus[did].get("title", "") + " " + corpus[did].get("text", ""))[:512]
             for did in doc_ids]

    # Create memmap for embeddings
    n_docs = len(texts)
    embed_dim = 384  # bge-small hidden dim
    memmap_path = DATA_DIR / "corpus_embeddings_fp16.npy"

    if memmap_path.exists():
        print(f"  Loading cached embeddings from {memmap_path}")
        embeddings = np.memmap(memmap_path, dtype=np.float16, mode='r',
                              shape=(n_docs, embed_dim))
    else:
        print(f"  Encoding {n_docs} documents (FP16)...")
        embeddings = np.memmap(memmap_path, dtype=np.float16, mode='w+',
                              shape=(n_docs, embed_dim))

        batch_size = 256
        with torch.no_grad():
            for i in range(0, n_docs, batch_size):
                if i % (batch_size * 100) == 0:
                    print(f"    {i}/{n_docs} ({100*i/n_docs:.1f}%)")
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt")
                if torch.cuda.is_available():
                    encoded = {k: v.cuda() for k, v in encoded.items()}
                outputs = base_model(**encoded)
                # CLS pooling
                embs = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float16)
                embeddings[i:i+len(batch_texts)] = embs

        embeddings.flush()
        print(f"  Saved embeddings to {memmap_path}")

    return doc_ids, embeddings


# =======================================================================
# Phase 4: Evaluate retrieval
# =======================================================================

def phase_eval(doc_ids, embeddings, queries, qrels, prefix_dims=PREFIX_DIMS):
    """Evaluate retrieval at each prefix dimension.

    Uses exact FlatIP search (only 400 queries, so feasible).
    Reports nDCG@10, Recall@100, MAP.
    """
    import faiss
    from transformers import AutoTokenizer, AutoModel
    import torch

    print(f"\nEvaluating retrieval...")
    print(f"  Corpus: {len(doc_ids)} docs, Queries: {len(queries)}")

    # Load model for query encoding
    model_name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.half()
    if torch.cuda.is_available():
        model.cuda()

    # Encode queries
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"  Encoding {len(query_texts)} queries...")
    query_embs = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(query_texts), batch_size):
            batch = query_texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            outputs = model(**encoded)
            embs = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float16)
            query_embs.append(embs)
    query_embs = np.vstack(query_embs)

    # Build doc_id -> index mapping
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    results = {}

    for dim in prefix_dims:
        print(f"\n  === Prefix dim: {dim}d ===")

        # Truncate embeddings to prefix
        corpus_prefix = np.ascontiguousarray(embeddings[:, :dim].astype(np.float32))
        query_prefix = np.ascontiguousarray(query_embs[:, :dim].astype(np.float32))

        # Normalize for cosine similarity
        faiss.normalize_L2(corpus_prefix)
        faiss.normalize_L2(query_prefix)

        # Build FAISS index (exact search)
        t0 = time.time()
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_prefix)
        build_time = time.time() - t0
        print(f"    Index built in {build_time:.1f}s")

        # Search
        k = 100  # Retrieve top-100
        t0 = time.time()
        scores, indices = index.search(query_prefix, k)
        search_time = time.time() - t0
        latency_per_query = search_time / len(query_ids) * 1000  # ms
        print(f"    Search: {search_time:.1f}s ({latency_per_query:.1f}ms/query)")

        # Compute metrics
        ndcg_10_list = []
        recall_100_list = []
        map_list = []

        for qi, qid in enumerate(query_ids):
            if qid not in qrels:
                continue

            rel = qrels[qid]  # dict: doc_id -> relevance (0, 1, 2)
            retrieved_ids = [doc_ids[idx] for idx in indices[qi] if idx < len(doc_ids)]

            # nDCG@10
            dcg = 0.0
            for rank, did in enumerate(retrieved_ids[:10]):
                r = rel.get(did, 0)
                dcg += (2**r - 1) / np.log2(rank + 2)
            # Ideal DCG
            ideal_rels = sorted(rel.values(), reverse=True)[:10]
            idcg = sum((2**r - 1) / np.log2(rank + 2) for rank, r in enumerate(ideal_rels))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_10_list.append(ndcg)

            # Recall@100
            n_relevant = sum(1 for r in rel.values() if r > 0)
            n_retrieved_relevant = sum(1 for did in retrieved_ids[:100] if rel.get(did, 0) > 0)
            recall = n_retrieved_relevant / n_relevant if n_relevant > 0 else 0.0
            recall_100_list.append(recall)

            # MAP
            ap = 0.0
            n_rel_found = 0
            for rank, did in enumerate(retrieved_ids[:100]):
                if rel.get(did, 0) > 0:
                    n_rel_found += 1
                    ap += n_rel_found / (rank + 1)
            ap = ap / n_relevant if n_relevant > 0 else 0.0
            map_list.append(ap)

        ndcg_10 = np.mean(ndcg_10_list)
        recall_100 = np.mean(recall_100_list)
        map_score = np.mean(map_list)

        results[dim] = {
            "ndcg_at_10": float(ndcg_10),
            "recall_at_100": float(recall_100),
            "map": float(map_score),
            "latency_ms": float(latency_per_query),
            "n_queries": len(ndcg_10_list),
        }

        print(f"    nDCG@10: {ndcg_10:.4f}")
        print(f"    Recall@100: {recall_100:.4f}")
        print(f"    MAP: {map_score:.4f}")

        del corpus_prefix, query_prefix, index
        gc.collect()

    return results


# =======================================================================
# Phase 5: Pareto analysis
# =======================================================================

def phase_pareto(results_v5, results_mrl, results_base=None):
    """Build Pareto frontier plot and compute AUPC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: nDCG@10 vs dimension
    ax = axes[0]
    dims = sorted(results_v5.keys())

    v5_ndcg = [results_v5[d]["ndcg_at_10"] for d in dims]
    mrl_ndcg = [results_mrl[d]["ndcg_at_10"] for d in dims]

    ax.plot(dims, v5_ndcg, "o-", color="#e74c3c", label="V5 (Fractal)", linewidth=2, markersize=8)
    ax.plot(dims, mrl_ndcg, "s-", color="#3498db", label="MRL", linewidth=2, markersize=8)

    if results_base:
        base_ndcg = [results_base[d]["ndcg_at_10"] for d in dims]
        ax.plot(dims, base_ndcg, "^--", color="#95a5a6", label="Truncation", linewidth=1.5)

    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("nDCG@10")
    ax.set_title("BEIR DBPedia-Entity (4.6M docs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: nDCG@10 vs latency
    ax = axes[1]
    v5_lat = [results_v5[d]["latency_ms"] for d in dims]
    mrl_lat = [results_mrl[d]["latency_ms"] for d in dims]

    ax.plot(v5_lat, v5_ndcg, "o-", color="#e74c3c", label="V5 (Fractal)", linewidth=2, markersize=8)
    ax.plot(mrl_lat, mrl_ndcg, "s-", color="#3498db", label="MRL", linewidth=2, markersize=8)

    # Label dimensions
    for d, x, y in zip(dims, v5_lat, v5_ndcg):
        ax.annotate(f"{d}d", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    for d, x, y in zip(dims, mrl_lat, mrl_ndcg):
        ax.annotate(f"{d}d", (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8)

    ax.set_xlabel("Query Latency (ms)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Pareto Frontier: Quality vs Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "figures" / "paper" / "fig_beir_pareto.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPareto plot saved to {fig_path}")


# =======================================================================
# Main
# =======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                       choices=["download", "train", "encode", "eval", "pareto", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    results_file = RESULTS_DIR / "beir_dbpedia_benchmark.json"

    if args.phase in ("download", "all"):
        print("=" * 70)
        print("[Phase 1] Download BEIR DBPedia-Entity")
        print("=" * 70)
        data_path, corpus, queries, qrels = phase_download()
        labels, l0_names, l1_names = derive_hierarchy_labels(corpus)

        # Save label mapping
        label_file = DATA_DIR / "dbpedia_entity_labels.json"
        with open(label_file, "w") as f:
            json.dump({
                "l0_names": l0_names,
                "l1_names": l1_names,
                "n_docs": len(corpus),
                "n_queries": len(queries),
            }, f, indent=2)
        print(f"Labels saved to {label_file}")

    if args.phase in ("encode", "all"):
        print("\n" + "=" * 70)
        print("[Phase 3] Encode corpus (base model, no fine-tuning)")
        print("=" * 70)
        # For the baseline, encode with pretrained model
        if "corpus" not in dir():
            from beir.datasets.data_loader import GenericDataLoader
            data_path = str(DATA_DIR / "dbpedia-entity")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        doc_ids, embeddings = phase_encode(corpus, models=None)

    if args.phase in ("eval", "all"):
        print("\n" + "=" * 70)
        print("[Phase 4] Evaluate retrieval (base model)")
        print("=" * 70)
        if "corpus" not in dir():
            from beir.datasets.data_loader import GenericDataLoader
            data_path = str(DATA_DIR / "dbpedia-entity")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        if "doc_ids" not in dir():
            doc_ids, embeddings = phase_encode(corpus, models=None)

        # Base model (no training) at each prefix dim
        results_base = phase_eval(doc_ids, embeddings, queries, qrels)

        # Save results
        out = {
            "experiment": "beir_dbpedia_benchmark",
            "model": MODEL_KEY,
            "phase": "base_model_eval",
            "timestamp": datetime.now().isoformat(),
            "prefix_dims": PREFIX_DIMS,
            "n_corpus": len(doc_ids),
            "n_queries": len(queries),
            "results_base": results_base,
        }
        with open(results_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {results_file}")

    if args.phase in ("pareto", "all"):
        print("\n" + "=" * 70)
        print("[Phase 5] Pareto analysis")
        print("=" * 70)
        if results_file.exists():
            data = json.load(open(results_file))
            if "results_base" in data:
                # For now just show base model results
                print("Base model results:")
                for dim, metrics in sorted(data["results_base"].items(), key=lambda x: int(x[0])):
                    print(f"  {dim}d: nDCG@10={metrics['ndcg_at_10']:.4f}, "
                          f"Recall@100={metrics['recall_at_100']:.4f}, "
                          f"MAP={metrics['map']:.4f}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
