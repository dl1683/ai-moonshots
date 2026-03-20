"""Tokenize FineWeb-Edu with GPT-2 BPE, chunked for memory efficiency.

Usage: python code/tokenize_fineweb.py
Output: data/fineweb_tokens.pt (~10B tokens)
"""

import torch
import time
from pathlib import Path

REPO = Path(__file__).parent.parent


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    src = REPO / "data" / "fineweb_edu_10bt.txt"
    out = REPO / "data" / "fineweb_tokens.pt"

    file_size = src.stat().st_size
    print(f"Tokenizing {file_size/1e9:.1f}GB FineWeb-Edu...")

    all_tokens = []
    chunk_size = 50_000_000  # 50MB chunks
    offset = 0
    chunk_num = 0
    t0 = time.time()

    with open(src, "r", encoding="utf-8", errors="replace") as f:
        while True:
            text = f.read(chunk_size)
            if not text:
                break
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            offset += len(text.encode("utf-8", errors="replace"))
            chunk_num += 1
            elapsed = time.time() - t0
            rate = offset / elapsed / 1e6
            print(f"  Chunk {chunk_num}: {len(tokens):,} tokens, "
                  f"total={len(all_tokens):,}, "
                  f"{offset/file_size*100:.1f}%, "
                  f"{rate:.1f}MB/s", flush=True)

    print(f"Total: {len(all_tokens):,} tokens ({len(all_tokens)/1e9:.2f}B)")
    t = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(t, out)
    print(f"Saved to {out} ({out.stat().st_size/1e9:.1f}GB)")


if __name__ == "__main__":
    main()
