"""Sutra v0.5.4 Production Training.

v0.5.4 = v0.5.3 + Peri-LN + Delayed Pheromone + Grokfast.
Chrome-validated: +13.6% combined BPT improvement.

Changes from sutra_v05_train.py:
  1. Imports from launch_v054 (Peri-LN + Delayed Pheromone)
  2. Grokfast EMA gradient filter (alpha=0.95, lambda=2.0)
  3. 500-step re-warmup after warm-start (Codex design)
  4. Grokfast only on matrix params (skip norms/biases)
  5. Grokfast delayed 200 steps after warm-start

Usage:
    python code/train_v054.py
    (automatically warm-starts from latest v0.5.3 checkpoint)
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
BATCH_SIZE = 8
GRAD_ACCUM = 8
LR = 8e-4
WARMUP_STEPS = 500        # v0.5.4: shorter re-warmup (Codex design)
REWARMUP_LR_START = 3e-4  # v0.5.4: start LR lower for norm adaptation
MAX_STEPS = 100000
EVAL_EVERY = 1000         # v0.5.4: more frequent eval (Codex design)
SAVE_EVERY = 5000
LOG_EVERY = 100
VOCAB_SIZE = 50257

# Grokfast config (Chrome-validated: +11% BPT)
GROKFAST_ALPHA = 0.95
GROKFAST_LAMBDA = 2.0
GROKFAST_DELAY = 200      # Enable after 200 warm-start steps

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v054 import create_v054 as _create_model, warmstart_v054
from grokfast import GrokfastFilter


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
    max_start = len(tokens) - seq_len - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[i:i + seq_len] for i in idx])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)


def evaluate(model, test_tokens, n_batches=20):
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
    return {"bpt": avg_loss / math.log(2), "loss": avg_loss}


def generate_sample(model, test_tokens, tokenizer, max_new=100):
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
    print(f"SUTRA v0.5.4 PRODUCTION TRAINING")
    print(f"  Peri-LN + Delayed Pheromone + Grokfast(a={GROKFAST_ALPHA}, l={GROKFAST_LAMBDA})")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Config: dim={DIM}, ff={FF_DIM}, max_steps={MAX_STEPS_PER_POSITION}")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}")
    print(f"{'='*60}")

    train_tokens, test_tokens = load_tokens()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Checkpoint paths
    ckpt_dir = REPO / "results" / "checkpoints_v054"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "v054_log.txt"
    metrics_file = REPO / "results" / "v054_metrics.json"

    start_step = 0
    best_bpt = float("inf")
    metrics_history = []

    # Try resume from v0.5.4 checkpoint first
    latest_054 = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if latest_054:
        ckpt = torch.load(latest_054[-1], weights_only=False, map_location=DEVICE)
        model = _create_model(dim=DIM, ff_dim=FF_DIM,
                              max_steps=MAX_STEPS_PER_POSITION,
                              window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        best_bpt = ckpt.get("best_bpt", float("inf"))
        metrics_history = ckpt.get("metrics", [])
        print(f"RESUMED v0.5.4 from step {start_step}, best BPT={best_bpt:.4f}")
    else:
        # Warm-start from v0.5.3 checkpoint
        v053_ckpt_dir = REPO / "results" / "checkpoints_v053"
        v053_latest = sorted(v053_ckpt_dir.glob("step_*.pt"),
                             key=lambda p: int(p.stem.split("_")[1]))
        if v053_latest:
            print(f"Warm-starting v0.5.4 from v0.5.3: {v053_latest[-1].name}")
            model = warmstart_v054(str(v053_latest[-1]),
                                   dim=DIM, ff_dim=FF_DIM,
                                   max_steps=MAX_STEPS_PER_POSITION,
                                   window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)
        else:
            print("No v0.5.3 checkpoint found. Training v0.5.4 from SCRATCH.")
            model = _create_model(dim=DIM, ff_dim=FF_DIM,
                                  max_steps=MAX_STEPS_PER_POSITION,
                                  window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)

    params = model.count_params()
    print(f"Params: {params:,} ({params/1e6:.1f}M)")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    if latest_054 and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    # v0.5.4: Grokfast filter (only on matrix params per Codex design)
    gf = GrokfastFilter(model, alpha=GROKFAST_ALPHA, lam=GROKFAST_LAMBDA)
    if latest_054 and "grokfast" in ckpt:
        gf.load_state_dict(ckpt["grokfast"])
        print("Loaded Grokfast EMA state")

    # Training
    model.train()
    step = start_step
    running_loss = 0
    loss_count = 0
    start = time.time()

    while step < MAX_STEPS:
        x, y = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN)

        # v0.5.4: Re-warmup schedule (500 steps from REWARMUP_LR_START to LR)
        if step < WARMUP_STEPS:
            lr = REWARMUP_LR_START + (LR - REWARMUP_LR_START) * (step + 1) / WARMUP_STEPS
        else:
            lr = LR * 0.5 * (1 + math.cos(math.pi * (step - WARMUP_STEPS)
                                            / max(1, MAX_STEPS - WARMUP_STEPS)))
        for pg in opt.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = model(x)
            Tc = min(logits.size(1), y.size(1))
            ce = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE),
                                 y[:, :Tc].reshape(-1))
            compute_penalty = 0.01 * aux["compute_cost"] if step > MAX_STEPS * 0.6 else 0
            loss = (ce + compute_penalty) / GRAD_ACCUM

        loss.backward()
        running_loss += ce.item()
        loss_count += 1

        if loss_count % GRAD_ACCUM == 0:
            # NaN guard
            if torch.isnan(loss) or any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            ):
                print(f"WARNING: NaN at step {step}, skipping", flush=True)
                opt.zero_grad()
                loss_count = 0
                running_loss = 0
                continue

            # v0.5.4: Grokfast (delayed, matrix params only)
            if step >= GROKFAST_DELAY:
                gf.apply()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                avg = running_loss / loss_count
                elapsed = time.time() - start
                tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                gf_status = "ON" if step >= GROKFAST_DELAY else "warmup"
                msg = (f"Step {step:>6d}/{MAX_STEPS}: loss={avg:.4f} "
                       f"lr={lr:.2e} {tps:.0f}tok/s {mem:.1f}GB "
                       f"avg_steps={aux['avg_steps']} gf={gf_status}")
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
                        torch.save(model.state_dict(), REPO / "results" / "v054_best.pt")

                    gen = {"prompt": "(eval)", "output": "(eval)"}
                    try:
                        gen = generate_sample(model, test_tokens, tokenizer)
                        gen = {k: v.encode("ascii", errors="replace").decode("ascii") for k, v in gen.items()}
                        print(f"\nGENERATION @ step {step}: {gen['output'][:200]}", flush=True)
                    except Exception as e:
                        print(f"\nGeneration failed: {e}", flush=True)

                    entry = {
                        "step": step, "test_bpt": round(bpt, 4),
                        "best_bpt": round(best_bpt, 4), "is_best": is_best,
                        "lr": lr, "avg_steps": aux["avg_steps"],
                        "grokfast": step >= GROKFAST_DELAY,
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
                    "grokfast": gf.state_dict(),
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


if __name__ == "__main__":
    main()
