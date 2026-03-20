"""Clean evaluation: Sutra v0.5 vs Pythia on leakage-free benchmark.

Uses eval/clean_benchmark.txt (hand-written, not in any training data).
Reports BPB (bits per byte) for fair cross-model comparison.

Usage:
    python eval/clean_eval.py --model results/v05_best.pt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))


def eval_sutra_v05(model_path, text, dim=768):
    """Evaluate Sutra v0.5 SSM on text."""
    from sutra_v05_ssm import SutraV05
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    text_bytes = len(text.encode("utf-8"))

    model = SutraV05(vocab_size=50257, dim=dim, ff_dim=dim * 2, max_steps=8,
                     window=4, k_retrieval=8).cuda()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cuda"))
    model.eval()

    total_loss = total_tokens = 0
    seq_len = 512
    with torch.no_grad():
        for i in range(0, len(tokens) - seq_len, seq_len):
            x = tokens[i:i + seq_len].unsqueeze(0).cuda()
            y = tokens[i + 1:i + seq_len + 1].unsqueeze(0).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            Tc = min(logits.size(1), y.size(1))
            loss = F.cross_entropy(logits[:, :Tc].float().reshape(-1, 50257),
                                   y[:, :Tc].reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += Tc
        # Handle remainder
        if len(tokens) > seq_len:
            x = tokens[-seq_len:].unsqueeze(0).cuda()
            y_end = tokens[-seq_len + 1:].unsqueeze(0).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            # Only count the last few tokens not already counted
            already = (len(tokens) - seq_len) // seq_len * seq_len + seq_len
            new_start = max(0, already - (len(tokens) - seq_len))

    bpb = total_loss / (text_bytes * math.log(2)) if text_bytes > 0 else 0
    bpt = total_loss / (total_tokens * math.log(2)) if total_tokens > 0 else 0
    del model; torch.cuda.empty_cache()
    return {"bpb": round(bpb, 4), "bpt": round(bpt, 4), "tokens": total_tokens, "bytes": text_bytes}


def eval_hf_model(model_name, text):
    """Evaluate HuggingFace model on text."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    text_bytes = len(text.encode("utf-8"))

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda()
    model.eval()

    total_loss = total_tokens = 0
    seq_len = 512
    with torch.no_grad():
        for i in range(0, len(tokens) - seq_len, seq_len):
            x = tokens[i:i + seq_len].unsqueeze(0).cuda()
            y = tokens[i + 1:i + seq_len + 1].unsqueeze(0).cuda()
            logits = model(x).logits
            Tc = min(logits.size(1), y.size(1))
            loss = F.cross_entropy(logits[:, :Tc].reshape(-1, logits.size(-1)),
                                   y[:, :Tc].reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += Tc

    bpb = total_loss / (text_bytes * math.log(2)) if text_bytes > 0 else 0
    bpt = total_loss / (total_tokens * math.log(2)) if total_tokens > 0 else 0
    del model; torch.cuda.empty_cache()
    return {"bpb": round(bpb, 4), "bpt": round(bpt, 4), "tokens": total_tokens, "bytes": text_bytes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(REPO / "results" / "v05_best.pt"))
    parser.add_argument("--competitors", default="EleutherAI/pythia-70m")
    args = parser.parse_args()

    with open(REPO / "eval" / "clean_benchmark.txt", "r") as f:
        text = f.read()
    print(f"Clean benchmark: {len(text):,} chars, {len(text.encode('utf-8')):,} bytes")

    results = {}

    if Path(args.model).exists():
        print("Evaluating Sutra v0.5...", flush=True)
        results["Sutra-v0.5"] = eval_sutra_v05(args.model, text)
        print(f"  BPB={results['Sutra-v0.5']['bpb']}")

    for name in args.competitors.split(","):
        name = name.strip()
        if name:
            print(f"Evaluating {name}...", flush=True)
            results[name] = eval_hf_model(name, text)
            print(f"  BPB={results[name]['bpb']}")

    print(f"\n{'='*50}")
    print(f"CLEAN BENCHMARK (no leakage risk)")
    print(f"{'='*50}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["bpb"]):
        print(f"  {name:30s}: BPB={r['bpb']:.4f}")

    json.dump(results, open(REPO / "results" / "clean_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
