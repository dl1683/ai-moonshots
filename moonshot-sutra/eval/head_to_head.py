"""Head-to-head BPB comparison: Sutra vs competitors on same test set.

Runs ALL models on the SAME test text and computes byte-level perplexity.
Supports both byte-level Sutra and token-level Combo 5.

Usage:
    python eval/head_to_head.py --sutra-model results/combo5_best.pt --mode token
    python eval/head_to_head.py --competitors EleutherAI/pythia-160m,EleutherAI/pythia-410m
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))


def compute_sutra_bpb_byte(model_path, test_text, dim=5120, seq_len=512):
    """Compute BPB for byte-level Sutra model."""
    from sutra_v04 import SutraV04
    model = SutraV04(dim=dim, patch_size=4, max_rounds=6, k_retrieval=16,
                     max_seq=seq_len, adaptive_halt=False)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cuda"))
    model = model.cuda().eval()

    data = torch.tensor(list(test_text.encode("utf-8")), dtype=torch.long)
    total_loss = total_bytes = 0

    with torch.no_grad():
        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i:i + seq_len].unsqueeze(0).cuda()
            y = data[i + 1:i + seq_len + 1].unsqueeze(0).cuda()
            logits, _ = model(x)
            Tc = min(logits.size(1), y.size(1))
            loss = F.cross_entropy(logits[:, :Tc].reshape(-1, 256),
                                   y[:, :Tc].reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_bytes += y[:, :Tc].numel()

    del model
    torch.cuda.empty_cache()
    return total_loss / (total_bytes * math.log(2))


def compute_sutra_bpb_token(model_path, test_text, dim=768, seq_len=512,
                            tie_weights=True, n_gru_layers=2):
    """Compute BPB for token-level Combo 5 model."""
    from sutra_v04 import SutraV04
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    model = SutraV04(vocab_size=vocab_size, dim=dim, patch_size=4, max_rounds=4,
                     k_retrieval=8, max_seq=seq_len, use_kan=False,
                     adaptive_halt=False, tie_weights=tie_weights,
                     n_gru_layers=n_gru_layers)
    state = torch.load(model_path, weights_only=True, map_location="cuda")
    model.load_state_dict(state)
    model = model.cuda().eval()

    # Tokenize test text
    tokens = tokenizer.encode(test_text)
    data = torch.tensor(tokens, dtype=torch.long)

    total_loss = total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i:i + seq_len].unsqueeze(0).cuda()
            y = data[i + 1:i + seq_len + 1].unsqueeze(0).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            Tc = min(logits.size(1), y.size(1))
            loss = F.cross_entropy(logits[:, :Tc].float().reshape(-1, vocab_size),
                                   y[:, :Tc].reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y[:, :Tc].numel()

    # Convert token-level CE to byte-level BPB
    text_bytes = len(test_text.encode("utf-8"))
    text_tokens = len(tokens)
    bytes_per_token = text_bytes / text_tokens
    bpt = total_loss / (total_tokens * math.log(2))
    bpb = bpt / bytes_per_token

    del model
    torch.cuda.empty_cache()
    return {"bpb": bpb, "bpt": bpt, "bytes_per_token": bytes_per_token}


def compute_hf_bpb(model_name, test_text, max_chars=50000):
    """Compute BPB for a HuggingFace model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  transformers not installed")
        return None

    print(f"  Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    text = test_text[:max_chars]
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"].to(model.device)

    total_loss = 0
    total_tokens = 0
    seq_len = 512
    model.eval()

    with torch.no_grad():
        for i in range(0, input_ids.size(1) - seq_len, seq_len):
            chunk = input_ids[:, i:i + seq_len + 1]
            x = chunk[:, :-1]
            y = chunk[:, 1:]
            outputs = model(x)
            logits = outputs.logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()

    # Convert token-level CE to byte-level BPB
    text_bytes = len(text.encode("utf-8"))
    text_tokens = total_tokens
    bytes_per_token = text_bytes / text_tokens if text_tokens > 0 else 1
    bpt = total_loss / (total_tokens * math.log(2)) if total_tokens > 0 else 0
    bpb = bpt / bytes_per_token

    del model
    torch.cuda.empty_cache()
    return {"bpb": bpb, "bpt": bpt, "bytes_per_token": bytes_per_token}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sutra-model", help="Path to Sutra checkpoint")
    parser.add_argument("--mode", default="token", choices=["byte", "token"])
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--test-text", default=str(REPO / "data" / "corpus_test.txt"))
    parser.add_argument("--competitors",
                        default="EleutherAI/pythia-160m,EleutherAI/pythia-410m",
                        help="Comma-separated HF model names")
    args = parser.parse_args()

    # Load test text
    with open(args.test_text, "r", encoding="utf-8") as f:
        test_text = f.read()[:50000]
    print(f"Test text: {len(test_text):,} chars, {len(test_text.encode('utf-8')):,} bytes")

    results = {}

    # Sutra
    if args.sutra_model and Path(args.sutra_model).exists():
        print(f"\nEvaluating Sutra ({args.mode}-level)...", flush=True)
        if args.mode == "byte":
            bpb = compute_sutra_bpb_byte(args.sutra_model, test_text, dim=args.dim)
            results["Sutra (byte)"] = {"bpb": bpb}
        else:
            r = compute_sutra_bpb_token(args.sutra_model, test_text, dim=args.dim)
            results["Sutra Combo5 (token)"] = r
        print(f"  BPB: {results[list(results.keys())[0]]['bpb']:.4f}")

    # Competitors
    for model_name in args.competitors.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        print(f"\nEvaluating {model_name}...", flush=True)
        r = compute_hf_bpb(model_name, test_text)
        if r is not None:
            results[model_name] = r
            print(f"  BPB: {r['bpb']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD BPB COMPARISON (lower is better)")
    print(f"{'='*60}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["bpb"]):
        bpb = r["bpb"]
        bpt = r.get("bpt", "N/A")
        bpt_str = f"{bpt:.4f}" if isinstance(bpt, float) else bpt
        print(f"  {name:40s}: BPB={bpb:.4f}  BPT={bpt_str}")

    with open(REPO / "results" / "head_to_head.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/head_to_head.json")


if __name__ == "__main__":
    main()
