"""Sutra v0.5.1 Production Training.

Stage-Superposition with switching kernel + lambda halting.
Warm-starts from v0.5 checkpoint (94% param transfer).

Key changes from v0.5 training:
- LR: 1e-3 (was 3e-4, +15.2% BPT improvement validated)
- max_steps: 6 (was 8, 37% faster, 91% of benefit retained)
- Inter-step loss: 0.75*final + 0.25*intermediate
- Passes targets to forward() for intermediate loss computation
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

# --- v0.5.1 Hyperparameters ---
DIM = 768
FF_DIM = 1536
MAX_STEPS = 6           # Down from 8 (Chrome: 91% benefit in 4, 2 recovery steps)
WINDOW = 4
K_RETRIEVAL = 8
SEQ_LEN = 512
BATCH_SIZE = 8
GRAD_ACCUM = 8           # Effective batch = 64
LR = 1e-3                # Up from 3e-4 (Chrome LR sweep: +15.2%)
WARMUP_STEPS = 1000
MAX_TRAIN_STEPS = 100000
EVAL_EVERY = 5000
SAVE_EVERY = 5000
LOG_EVERY = 100
VOCAB_SIZE = 50257

import sys
sys.path.insert(0, str(REPO / "code"))
from sutra_v051 import SutraV051


def load_tokens():
    full_path = REPO / "data" / "minipile_full_tokens.pt"
    subset_path = REPO / "data" / "minipile_tokens.pt"
    path = full_path if full_path.exists() else subset_path
    tokens = torch.load(path, weights_only=True)
    n_test = max(1000, int(len(tokens) * 0.005))
    print(f"Loaded {len(tokens):,} tokens, train: {len(tokens)-n_test:,}, test: {n_test:,}")
    return tokens[:-n_test], tokens[-n_test:]


def sample_batch(tokens, batch_size, seq_len):
    max_start = len(tokens) - seq_len - 1
    idx = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([tokens[i:i + seq_len] for i in idx])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)


def evaluate(model, test_tokens, n_batches=20):
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_batch(test_tokens, min(BATCH_SIZE, 8), SEQ_LEN)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                Tc = min(logits.size(1), y.size(1))
                loss = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE),
                                       y[:, :Tc].reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += Tc * x.size(0)
    model.train()
    return total_loss / (total_tokens * math.log(2))


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
            topk_v, topk_i = next_logits.topk(40)
            filt = torch.full_like(next_logits, float("-inf"))
            filt.scatter_(-1, topk_i, topk_v)
            next_token = torch.multinomial(F.softmax(filt, dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)
    model.train()
    return {"prompt": tokenizer.decode(prompt.tolist())[:200],
            "output": tokenizer.decode(tokens[0, 32:].cpu().tolist())[:400]}


def main():
    print(f"SUTRA v0.5.1 PRODUCTION TRAINING")
    print(f"Switching Kernel + Lambda Halting + Inter-Step Loss")
    print(f"Device: {DEVICE}, bf16, LR={LR}, max_steps={MAX_STEPS}")
    print(f"{'='*60}")

    train_tokens, test_tokens = load_tokens()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = SutraV051(vocab_size=VOCAB_SIZE, dim=DIM, ff_dim=FF_DIM,
                      max_steps=MAX_STEPS, window=WINDOW, k_retrieval=K_RETRIEVAL
                      ).to(DEVICE)
    print(f"Params: {model.count_params():,} ({model.count_params()/1e6:.1f}M)")

    # Checkpoint resume or warm-start
    ckpt_dir = REPO / "results" / "checkpoints_v051"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "v051_log.txt"
    metrics_file = REPO / "results" / "v051_metrics.json"

    start_step = 0
    best_bpt = float("inf")
    metrics_history = []

    # Check for v0.5.1 checkpoint first, then warm-start
    latest = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    warmstart = REPO / "results" / "v051_warmstart.pt"

    if latest:
        ckpt = torch.load(latest[-1], weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        best_bpt = ckpt.get("best_bpt", float("inf"))
        metrics_history = ckpt.get("metrics", [])
        print(f"RESUMED v0.5.1 from step {start_step}")
    elif warmstart.exists():
        ckpt = torch.load(warmstart, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        start_step = 0  # fresh training schedule, warm weights
        print(f"WARM-START from v0.5 (step {ckpt.get('step', '?')})")
    else:
        print("Training from SCRATCH (no checkpoint or warm-start found)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    if latest and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    model.train()
    step = start_step
    running_loss = 0
    loss_count = 0
    start = time.time()

    while step < MAX_TRAIN_STEPS:
        x, y = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN)

        lr = LR * min(1.0, (step + 1) / WARMUP_STEPS) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - WARMUP_STEPS)
                                 / max(1, MAX_TRAIN_STEPS - WARMUP_STEPS)))
        )
        for pg in opt.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = model(x, targets=y)
            if "loss" in aux:
                loss = aux["loss"] / GRAD_ACCUM
            else:
                Tc = min(logits.size(1), y.size(1))
                ce = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE),
                                     y[:, :Tc].reshape(-1))
                loss = ce / GRAD_ACCUM

        loss.backward()
        running_loss += loss.item() * GRAD_ACCUM
        loss_count += 1

        if loss_count % GRAD_ACCUM == 0:
            # NaN guard
            if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                print(f"WARNING: NaN gradient at step {step}, skipping update", flush=True)
                opt.zero_grad()
                continue

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                avg = running_loss / loss_count
                elapsed = time.time() - start
                tps = (step - start_step) * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                avg_s = aux.get("avg_steps", MAX_STEPS)
                msg = (f"Step {step:>6d}/{MAX_TRAIN_STEPS}: loss={avg:.4f} "
                       f"lr={lr:.2e} {tps:.0f}tok/s {mem:.1f}GB avg_steps={avg_s}")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_loss = 0
                loss_count = 0

            if step % EVAL_EVERY == 0:
                bpt = evaluate(model, test_tokens)
                is_best = bpt < best_bpt
                if is_best:
                    best_bpt = bpt
                    torch.save(model.state_dict(), REPO / "results" / "v051_best.pt")

                gen = generate_sample(model, test_tokens, tokenizer)
                print(f"\n{'='*60}\nGENERATION @ step {step}")
                try:
                    print(f"PROMPT: {gen['prompt'][:150]}")
                    print(f"OUTPUT: {gen['output'][:300]}")
                except UnicodeEncodeError:
                    print("(non-ASCII generation)")
                print(f"{'='*60}\n")

                entry = {"step": step, "test_bpt": round(bpt, 4),
                         "best_bpt": round(best_bpt, 4), "is_best": is_best,
                         "lr": lr, "avg_steps": aux.get("avg_steps", MAX_STEPS),
                         "mean_lambda": aux.get("mean_lambda", 0),
                         "generation": gen,
                         "timestamp": datetime.now().isoformat()}
                metrics_history.append(entry)
                json.dump(metrics_history, open(metrics_file, "w"), indent=2)

                best_marker = " *BEST*" if is_best else ""
                eval_msg = f"  EVAL Step {step}: BPT={bpt:.4f}{best_marker}"
                print(eval_msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(eval_msg + "\n")

            if step % SAVE_EVERY == 0:
                ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "step": step, "best_bpt": best_bpt, "metrics": metrics_history,
                        "config": {"dim": DIM, "ff_dim": FF_DIM, "max_steps": MAX_STEPS,
                                   "lr": LR, "vocab_size": VOCAB_SIZE}}
                torch.save(ckpt, ckpt_dir / f"step_{step}.pt")
                old = sorted(ckpt_dir.glob("step_*.pt"),
                             key=lambda p: int(p.stem.split("_")[1]))
                for o in old[:-3]:
                    o.unlink()

    print(f"\nDone. {step} steps, {(time.time()-start)/3600:.1f}h, best BPT={best_bpt:.4f}")


if __name__ == "__main__":
    main()
