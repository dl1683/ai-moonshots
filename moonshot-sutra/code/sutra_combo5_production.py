"""Sutra Combo 5 Production: Token-level + bf16 + full MiniPile.

THE competitive model. Based on validated experiments:
- Token-level BPE (33% better than byte-level, validated)
- bf16 mixed precision (2x throughput, half VRAM)
- v0.4 architecture (GRU patches + msg passing + sparse retrieval + PonderNet)
- Full MiniPile ~1.4B tokens

Target: approach Pythia-410M quality with ~127M params.
"""

import json, math, os, random, time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = Path(__file__).parent.parent

# --- Hyperparameters ---
# Codex review: 50K steps = 819M tokens, doesn't cover 1.7B corpus once.
# Fix: 200K steps = 3.3B tokens seen (2 epochs of ~1.7B).
# Tokens/step = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN = 16 * 4 * 512 = 32K
DIM = 768            # ~88M params (closer to Chinchilla-optimal for 1.7B tokens)
PATCH_SIZE = 4
MAX_ROUNDS = 4
K_RETRIEVAL = 8
SEQ_LEN = 512        # Longer context (Codex: 256 too short for routing architecture)
BATCH_SIZE = 8        # Smaller batch to fit longer seq in VRAM with bf16
GRAD_ACCUM = 8        # Effective batch = 64
LR = 3e-4
WARMUP_STEPS = 1000   # Longer warmup for stability
MAX_STEPS = 100000    # 100K steps * 32K tok/step = 3.2B tokens (~2 epochs)
EVAL_EVERY = 5000
SAVE_EVERY = 5000
LOG_EVERY = 100
VOCAB_SIZE = 50257    # GPT-2 BPE
TEST_FRAC = 0.005     # 0.5% held out for eval

import sys
sys.path.insert(0, str(REPO / "code"))
from sutra_v04 import SutraV04


def load_tokens():
    """Load tokenized data, preferring full MiniPile if available."""
    full_path = REPO / "data" / "minipile_full_tokens.pt"
    subset_path = REPO / "data" / "minipile_tokens.pt"

    if full_path.exists():
        print(f"Loading FULL MiniPile tokens...")
        tokens = torch.load(full_path, weights_only=True)
        print(f"  {len(tokens):,} tokens ({len(tokens)/1e9:.2f}B)")
    elif subset_path.exists():
        print(f"Loading subset tokens (full not ready)...")
        tokens = torch.load(subset_path, weights_only=True)
        print(f"  {len(tokens):,} tokens ({len(tokens)/1e6:.1f}M)")
    else:
        raise FileNotFoundError("No tokenized data found. Run tokenization first.")

    # Train/test split
    n_test = max(1000, int(len(tokens) * TEST_FRAC))
    test_tokens = tokens[-n_test:]
    train_tokens = tokens[:-n_test]
    print(f"  Train: {len(train_tokens):,}, Test: {len(test_tokens):,}")
    return train_tokens, test_tokens


def sample_batch(tokens, batch_size, seq_len):
    """Random batch from token tensor."""
    max_start = len(tokens) - seq_len - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[i:i + seq_len] for i in idx])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)


def evaluate(model, test_tokens, n_batches=20):
    """Compute test metrics over multiple batches.

    Returns dict with:
    - bpt: bits per token (CE / ln2)
    - bpb: bits per byte (bpt / avg_bytes_per_token) for fair cross-vocab comparison
    - loss: raw cross-entropy loss
    """
    # GPT-2 BPE averages ~3.7 bytes per token on English text
    AVG_BYTES_PER_TOKEN = 3.7
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_batch(test_tokens, min(BATCH_SIZE, 8), SEQ_LEN)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                Tc = min(logits.size(1), y.size(1))
                loss = F.cross_entropy(
                    logits[:, :Tc].reshape(-1, VOCAB_SIZE),
                    y[:, :Tc].reshape(-1)
                )
            total_loss += loss.item()
    model.train()
    avg_loss = total_loss / n_batches
    bpt = avg_loss / math.log(2)
    bpb = bpt / AVG_BYTES_PER_TOKEN
    return {"bpt": bpt, "bpb": bpb, "loss": avg_loss}


def generate_sample(model, prompt_tokens, max_new=100, temperature=0.8, top_k=50):
    """Autoregressive generation from a prompt."""
    model.eval()
    tokens = prompt_tokens.clone().unsqueeze(0).to(DEVICE)  # (1, T)

    with torch.no_grad():
        for _ in range(max_new):
            # Use last SEQ_LEN tokens as context
            ctx = tokens[:, -SEQ_LEN:] if tokens.size(1) > SEQ_LEN else tokens
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ctx)
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = next_logits.topk(top_k, dim=-1)
                threshold = topk_vals[:, -1:]
                next_logits[next_logits < threshold] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

    model.train()
    return tokens[0].cpu()


def log_generation(model, test_tokens, step, tokenizer):
    """Generate and print sample text at eval checkpoints."""
    # Use first 32 test tokens as prompt
    prompt = test_tokens[:32]
    output = generate_sample(model, prompt, max_new=128)
    prompt_text = tokenizer.decode(prompt.tolist())
    generated_text = tokenizer.decode(output[32:].tolist())
    print(f"\n{'='*60}")
    print(f"GENERATION @ step {step}")
    print(f"PROMPT: {prompt_text[:200]}")
    print(f"OUTPUT: {generated_text[:400]}")
    print(f"{'='*60}\n", flush=True)
    return {"prompt": prompt_text[:200], "output": generated_text[:400]}


def main():
    print(f"SUTRA COMBO 5 PRODUCTION")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Config: dim={DIM}, patch={PATCH_SIZE}, rounds={MAX_ROUNDS}, k={K_RETRIEVAL}")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}, lr={LR}")
    print(f"{'='*60}")

    # Load data
    train_tokens, test_tokens = load_tokens()
    tok_per_param = len(train_tokens) / (DIM * DIM * 10)  # rough estimate

    # Tokenizer for generation
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Model (fixed rounds per Codex review: PonderNet halt was broken)
    # Weight tying: 9.3% lower BPB at 48% fewer params (validated)
    model = SutraV04(
        vocab_size=VOCAB_SIZE, dim=DIM, patch_size=PATCH_SIZE,
        max_rounds=MAX_ROUNDS, k_retrieval=K_RETRIEVAL, max_seq=SEQ_LEN,
        use_kan=False,         # MLP for token-level (KAN neutral at token scale)
        adaptive_halt=False,   # Fixed rounds (Codex: halt math was wrong)
        tie_weights=True,      # Share emb/head: 9.3% better BPB, 48% fewer params
        n_gru_layers=1,        # 1-layer GRU (2-layer didn't help at current training budget)
    ).to(DEVICE)

    params = model.count_params()
    print(f"Params: {params:,} ({params/1e6:.1f}M)")
    print(f"Tokens/param ratio: {len(train_tokens)/params:.1f}")
    print()

    # Checkpoint resume
    ckpt_dir = REPO / "results" / "checkpoints_combo5"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "combo5_log.txt"
    metrics_file = REPO / "results" / "combo5_metrics.json"

    start_step = 0
    best_bpb = float("inf")
    metrics_history = []

    latest = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if latest:
        ckpt = torch.load(latest[-1], weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        best_bpb = ckpt.get("best_bpb", float("inf"))
        metrics_history = ckpt.get("metrics", [])
        print(f"RESUMED from step {start_step}, best BPB={best_bpb:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    if latest and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    # Training loop
    model.train()
    step = start_step
    running_loss = 0
    loss_count = 0
    start = time.time()

    while step < MAX_STEPS:
        x, y = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN)

        # Cosine LR with warmup
        lr = LR * min(1.0, (step + 1) / WARMUP_STEPS) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)))
        )
        for pg in opt.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, kl = model(x)
            Tc = min(logits.size(1), y.size(1))
            ce = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE), y[:, :Tc].reshape(-1))
            loss = (ce + 0.01 * kl) / GRAD_ACCUM

        loss.backward()
        running_loss += ce.item()
        loss_count += 1

        if loss_count % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1

            # Logging
            if step % LOG_EVERY == 0:
                avg = running_loss / loss_count
                elapsed = time.time() - start
                tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                msg = f"Step {step:>6d}/{MAX_STEPS}: loss={avg:.4f} lr={lr:.2e} {tps:.0f}tok/s {mem:.1f}GB"
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_loss = 0
                loss_count = 0

            # Eval + generation
            if step % EVAL_EVERY == 0:
                metrics = evaluate(model, test_tokens)
                bpb = metrics["bpb"]
                is_best = bpb < best_bpb
                if is_best:
                    best_bpb = bpb
                    torch.save(model.state_dict(), REPO / "results" / "combo5_best.pt")

                gen = log_generation(model, test_tokens, step, tokenizer)

                entry = {
                    "step": step,
                    "test_bpt": round(metrics["bpt"], 4),
                    "test_bpb": round(bpb, 4),
                    "test_loss": round(metrics["loss"], 4),
                    "best_bpb": round(best_bpb, 4),
                    "is_best": is_best,
                    "lr": lr,
                    "generation_sample": gen,
                    "timestamp": datetime.now().isoformat()
                }
                metrics_history.append(entry)
                json.dump(metrics_history, open(metrics_file, "w"), indent=2)

                best_marker = " *BEST*" if is_best else ""
                eval_msg = (f"  EVAL Step {step}: BPT={metrics['bpt']:.4f} "
                           f"BPB={bpb:.4f}{best_marker}")
                print(eval_msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(eval_msg + "\n")

            # Checkpoint
            if step % SAVE_EVERY == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "best_bpb": best_bpb,
                    "metrics": metrics_history,
                }
                torch.save(ckpt, ckpt_dir / f"step_{step}.pt")
                # Keep only last 3 checkpoints
                old = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
                for o in old[:-3]:
                    o.unlink()

    elapsed_h = (time.time() - start) / 3600
    print(f"\nDone. {step} steps, {elapsed_h:.1f}h, best test BPB={best_bpb:.4f}")

    # Final generation
    log_generation(model, test_tokens, step, tokenizer)


if __name__ == "__main__":
    main()
