"""Build a real training corpus from local files + HuggingFace.

Ingests: .py, .md, .txt, .js, .ts files from the laptop
Plus: wikitext-2 from HuggingFace for natural language baseline

Deduplicates, shuffles, and creates train/test splits.
Target: ~50MB of clean, diverse text for training probes and MVP.
"""

import hashlib
import json
import os
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
SEARCH_ROOTS = [
    Path("C:/Users/devan/OneDrive/Desktop"),
    Path("C:/Users/devan/Downloads"),
    Path("C:/Users/devan/OneDrive/Documents"),
]

# Extensions to ingest, ordered by quality for training
TEXT_EXTENSIONS = {
    ".py": "code",
    ".md": "prose",
    ".txt": "prose",
    ".js": "code",
    ".ts": "code",
    ".tex": "prose",
    ".sh": "code",
    ".sql": "code",
    ".yaml": "config",
    ".toml": "config",
}

# Skip patterns
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
             ".next", "dist", "build", ".cache", ".ollama"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "Cargo.lock"}
MAX_FILE_SIZE = 500_000  # 500KB per file max
MIN_FILE_SIZE = 100  # Skip tiny files


def collect_local_files():
    """Collect text files from the laptop."""
    files = []
    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue
        for root, dirs, filenames in os.walk(search_root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in filenames:
                if fname in SKIP_FILES:
                    continue
                ext = Path(fname).suffix.lower()
                if ext not in TEXT_EXTENSIONS:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    size = os.path.getsize(fpath)
                    if MIN_FILE_SIZE <= size <= MAX_FILE_SIZE:
                        files.append((fpath, ext, TEXT_EXTENSIONS[ext]))
                except (OSError, PermissionError):
                    continue
    return files


def read_file_safe(path):
    """Read a file, handling encoding errors."""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, PermissionError, OSError):
            continue
    return None


def deduplicate(texts):
    """Remove duplicate texts using content hashing."""
    seen = set()
    unique = []
    for text, category in texts:
        h = hashlib.md5(text[:1000].encode("utf-8", errors="ignore")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append((text, category))
    return unique


def get_wikitext():
    """Download wikitext-2 from HuggingFace."""
    try:
        from datasets import load_dataset
        print("  Downloading wikitext-2 from HuggingFace...", flush=True)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(ds["text"])
        print(f"  WikiText-2: {len(text):,} chars")
        return text
    except Exception as e:
        print(f"  WikiText-2 failed: {e}")
        return ""


def main():
    print("Building Training Corpus for Sutra")
    print("=" * 60)

    # 1. Collect local files
    print("\n1. Collecting local files...", flush=True)
    files = collect_local_files()
    print(f"   Found {len(files):,} files")

    # Categorize
    by_category = {}
    for fpath, ext, cat in files:
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(fpath)

    for cat, fpaths in by_category.items():
        print(f"   {cat}: {len(fpaths):,} files")

    # 2. Read and collect texts
    print("\n2. Reading files...", flush=True)
    texts = []
    total_chars = 0
    errors = 0

    for fpath, ext, cat in files:
        content = read_file_safe(fpath)
        if content and len(content.strip()) > 50:
            texts.append((content, cat))
            total_chars += len(content)
        else:
            errors += 1

        if len(texts) % 1000 == 0 and len(texts) > 0:
            print(f"   Read {len(texts):,} files, {total_chars/1024/1024:.1f}MB...", flush=True)

    print(f"   Total: {len(texts):,} files, {total_chars/1024/1024:.1f}MB, {errors} errors")

    # 3. Deduplicate
    print("\n3. Deduplicating...", flush=True)
    before = len(texts)
    texts = deduplicate(texts)
    print(f"   {before} -> {len(texts)} ({before - len(texts)} duplicates removed)")

    # 4. Add WikiText
    print("\n4. Adding WikiText-2...", flush=True)
    wiki_text = get_wikitext()
    if wiki_text:
        # Split into chunks
        chunk_size = 2000
        for i in range(0, len(wiki_text) - chunk_size, chunk_size):
            chunk = wiki_text[i:i + chunk_size]
            if len(chunk.strip()) > 100:
                texts.append((chunk, "wiki"))

    print(f"   Total texts after wiki: {len(texts):,}")

    # 5. Shuffle and split
    print("\n5. Shuffling and splitting...", flush=True)
    random.shuffle(texts)

    split_idx = int(len(texts) * 0.95)
    train_texts = texts[:split_idx]
    test_texts = texts[split_idx:]

    # 6. Write to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "corpus_train.txt"
    test_path = DATA_DIR / "corpus_test.txt"

    train_chars = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for text, cat in train_texts:
            f.write(text + "\n\n")
            train_chars += len(text)

    test_chars = 0
    with open(test_path, "w", encoding="utf-8") as f:
        for text, cat in test_texts:
            f.write(text + "\n\n")
            test_chars += len(text)

    # 7. Save metadata
    meta = {
        "train_files": len(train_texts),
        "test_files": len(test_texts),
        "train_chars": train_chars,
        "test_chars": test_chars,
        "train_mb": round(train_chars / 1024 / 1024, 2),
        "test_mb": round(test_chars / 1024 / 1024, 2),
        "categories": dict(
            (cat, sum(1 for _, c in texts if c == cat))
            for cat in set(c for _, c in texts)
        ),
        "source": "local files + wikitext-2",
    }
    with open(DATA_DIR / "corpus_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CORPUS BUILT")
    print(f"  Train: {len(train_texts):,} files, {train_chars/1024/1024:.1f}MB")
    print(f"  Test: {len(test_texts):,} files, {test_chars/1024/1024:.1f}MB")
    print(f"  Categories: {json.dumps(meta['categories'], indent=2)}")
    print(f"  Saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
