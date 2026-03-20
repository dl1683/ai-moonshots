"""Sutra v0.5 SSM Production Training.

Stage-Superposition State Machine trained on full MiniPile (1.7B tokens).
4-stage curriculum per Codex design:
1. 0-20%: Stages 1-3-5-7 only (learn basics)
2. 20-60%: Enable Stage 4 routing
3. 60-85%: Enable Stage 6 compute penalty + Stage 7 verifier
4. 85-100%: Full loopback training

Must pass Pre-Training Gate before launching.
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
DIM = 768
FF_DIM = 1536
MAX_STEPS_PER_POSITION = 8
WINDOW = 4
K_RETRIEVAL = 8
SEQ_LEN = 512
BATCH_SIZE = 8         # v0.5 needs more VRAM than v0.4 (6 recurrent steps)
GRAD_ACCUM = 8         # Effective batch = 64 (same, more accumulation)
LR = 8e-4                # v0.5.2: gain clamp eliminates NaN, safe at 8e-4
WARMUP_STEPS = 1000
MAX_STEPS = 100000
EVAL_EVERY = 5000
SAVE_EVERY = 5000
LOG_EVERY = 100
VOCAB_SIZE = 50257

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v052 import create_v052 as _create_model


def load_tokens():
    """Load tokenized MiniPile."""
    full_path = REPO / "data" / "minipile_full_tokens.pt"
    subset_path = REPO / "data" / "minipile_tokens.pt"

    path = full_path if full_path.exists() else subset_path
    print(f"Loading tokens from {path.name}...")
    tokens = torch.load(path, weights_only=True)
    print(f"  {len(tokens):,} tokens ({len(tokens)/1e9:.3f}B)")

    n_test = max(1000, int(len(tokens) * 0.005))
    return tokens[:-n_test], tokens[-n_test:]


def sample_batch(tokens, batch_size, seq_len):
    """Random batch."""
    max_start = len(tokens) - seq_len - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[i:i + seq_len] for i in idx])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)


def evaluate(model, test_tokens, n_batches=20):
    """Test BPT on held-out data."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_batch(test_tokens, min(BATCH_SIZE, 8), SEQ_LEN)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = model(x)
                Tc = min(logits.size(1), y.size(1))
                loss = F.cross_entropy(
                    logits[:, :Tc].reshape(-1, VOCAB_SIZE),
                    y[:, :Tc].reshape(-1)
                )
            total_loss += loss.item()
    model.train()
    avg_loss = total_loss / n_batches
    bpt = avg_loss / math.log(2)
    return {"bpt": bpt, "loss": avg_loss}


def generate_sample(model, test_tokens, tokenizer, max_new=100):
    """Generate text for quality check."""
    model.eval()
    prompt = test_tokens[:32]
    tokens = prompt.clone().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for _ in range(max_new):
            ctx = tokens[:, -SEQ_LEN:] if tokens.size(1) > SEQ_LEN else tokens
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ctx)
            next_logits = logits[:, -1, :].float() / 0.9
            topk_vals, topk_idx = next_logits.topk(40)
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered.scatter_(-1, topk_idx, topk_vals)
            next_token = torch.multinomial(F.softmax(filtered, dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)

    model.train()
    prompt_text = tokenizer.decode(prompt.tolist())
    gen_text = tokenizer.decode(tokens[0, 32:].cpu().tolist())
    return {"prompt": prompt_text[:200], "output": gen_text[:400]}


def main():
    print(f"SUTRA v0.5 STAGE-SUPERPOSITION TRAINING")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Config: dim={DIM}, ff={FF_DIM}, max_steps={MAX_STEPS_PER_POSITION}")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}")
    print(f"{'='*60}")

    train_tokens, test_tokens = load_tokens()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = _create_model(
        dim=DIM, ff_dim=FF_DIM,
        max_steps=MAX_STEPS_PER_POSITION, window=WINDOW,
        k_retrieval=K_RETRIEVAL,
    ).to(DEVICE)

    params = model.count_params()
    print(f"Params: {params:,} ({params/1e6:.1f}M)")
    print(f"Tokens/param: {len(train_tokens)/params:.1f}")
    print()

    # Checkpoint resume
    ckpt_dir = REPO / "results" / "checkpoints_v052"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "v052_log.txt"
    metrics_file = REPO / "results" / "v052_metrics.json"

    start_step = 0
    best_bpt = float("inf")
    metrics_history = []

    latest = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    warmstart = REPO / "results" / "v052_warmstart.pt"
    if latest:
        ckpt = torch.load(latest[-1], weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        best_bpt = ckpt.get("best_bpt", float("inf"))
        metrics_history = ckpt.get("metrics", [])
        print(f"RESUMED from step {start_step}, best BPT={best_bpt:.4f}")
    elif warmstart.exists():
        state = torch.load(warmstart, weights_only=True, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"WARM-START from {warmstart.name}")
    else:
        print("Training from SCRATCH")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    if latest and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    # Training
    model.train()
    step = start_step
    running_loss = 0
    loss_count = 0
    start = time.time()

    while step < MAX_STEPS:
        x, y = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN)

        lr = LR * min(1.0, (step + 1) / WARMUP_STEPS) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - WARMUP_STEPS)
                                 / max(1, MAX_STEPS - WARMUP_STEPS)))
        )
        for pg in opt.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = model(x)
            Tc = min(logits.size(1), y.size(1))
            ce = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE),
                                 y[:, :Tc].reshape(-1))
            # Auxiliary losses (curriculum-gated)
            compute_penalty = 0.01 * aux["compute_cost"] if step > MAX_STEPS * 0.6 else 0
            loss = (ce + compute_penalty) / GRAD_ACCUM

        loss.backward()
        running_loss += ce.item()
        loss_count += 1

        if loss_count % GRAD_ACCUM == 0:
            # NaN guard (learned from LR=1e-3 divergence at step 3900)
            if torch.isnan(loss) or any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            ):
                print(f"WARNING: NaN at step {step}, skipping", flush=True)
                opt.zero_grad()
                loss_count = 0
                running_loss = 0
                continue
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                avg = running_loss / loss_count
                elapsed = time.time() - start
                tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                msg = (f"Step {step:>6d}/{MAX_STEPS}: loss={avg:.4f} "
                       f"lr={lr:.2e} {tps:.0f}tok/s {mem:.1f}GB "
                       f"avg_steps={aux['avg_steps']}")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_loss = 0
                loss_count = 0

            if step % EVAL_EVERY == 0:
                try:
                    metrics = evaluate(model, test_tokens)
                    bpt = metrics["bpt"]
                    is_best = bpt < best_bpt
                    if is_best:
                        best_bpt = bpt
                        torch.save(model.state_dict(), REPO / "results" / "v052_best.pt")

                    gen = {"prompt": "(eval)", "output": "(eval)"}
                    try:
                        gen = generate_sample(model, test_tokens, tokenizer)
                        # Sanitize for Windows cp1252
                        gen = {k: v.encode("ascii", errors="replace").decode("ascii") for k, v in gen.items()}
                        print(f"\nGENERATION @ step {step}: {gen['output'][:200]}", flush=True)
                    except Exception as e:
                        print(f"\nGeneration failed: {e}", flush=True)

                    entry = {
                        "step": step, "test_bpt": round(bpt, 4),
                        "best_bpt": round(best_bpt, 4), "is_best": is_best,
                        "lr": lr, "avg_steps": aux["avg_steps"],
                        "generation": gen,
                        "timestamp": datetime.now().isoformat(),
                    }
                    metrics_history.append(entry)
                    json.dump(metrics_history, open(metrics_file, "w"), indent=2)

                    best_marker = " *BEST*" if is_best else ""
                    eval_msg = f"  EVAL Step {step}: BPT={bpt:.4f}{best_marker}"
                    print(eval_msg, flush=True)
                    with open(log_file, "a") as f:
                        f.write(eval_msg + "\n")
                except Exception as e:
                    print(f"EVAL FAILED at step {step}: {e}", flush=True)

            if step % SAVE_EVERY == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step, "best_bpt": best_bpt,
                    "metrics": metrics_history,
                }
                torch.save(ckpt, ckpt_dir / f"step_{step}.pt")
                old = sorted(ckpt_dir.glob("step_*.pt"),
                             key=lambda p: int(p.stem.split("_")[1]))
                for o in old[:-3]:
                    o.unlink()

    elapsed_h = (time.time() - start) / 3600
    print(f"\nDone. {step} steps, {elapsed_h:.1f}h, best BPT={best_bpt:.4f}")
    generate_sample(model, test_tokens, tokenizer)


if __name__ == "__main__":
    main()
