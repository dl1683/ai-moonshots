#!/usr/bin/env python
"""
LAMBDA SWEEP: Fix perplexity confound + Pythia-410m replication

Run eta_preserve with lambda=0.01, 0.02 to find perplexity-matched point.
Then replicate on Pythia-410m for external validity.
"""

import json
import sys
import time
import gc
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 256
BATCH_SIZE = 16
NUM_TRAIN_STEPS = 2000
EVAL_EVERY = 250
LR = 6e-4
CLINC_MAX = 2000
HIDDEN_SAMPLE = 256


class PreTokenizedDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        n = len(token_ids) // seq_len
        self.chunks = token_ids[:n * seq_len].reshape(n, seq_len)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def pretokenize(tokenizer, texts, max_tokens=2_000_000):
    print("  Pre-tokenizing...", flush=True)
    all_ids = []
    total = 0
    for text in texts:
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        total += len(ids)
        if total >= max_tokens:
            break
    print(f"  {total:,} tokens", flush=True)
    return np.array(all_ids, dtype=np.int64)


def load_clinc_data(max_samples=CLINC_MAX):
    from hierarchical_datasets import load_hierarchical_dataset
    ds = load_hierarchical_dataset("clinc", split="train", max_samples=max_samples)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])
    return texts, labels


def compute_kappa_and_eta(X, labels):
    d = X.shape[1]
    unique_labels = np.unique(labels)
    grand_mean = X.mean(0)
    Z_parts, trace_sb, trace_sw = [], 0.0, 0.0
    for lbl in unique_labels:
        m = labels == lbl
        n_k = m.sum()
        if n_k < 2:
            continue
        X_k = X[m]
        mu_k = X_k.mean(0)
        trace_sb += n_k * np.sum((mu_k - grand_mean) ** 2)
        c = X_k - mu_k
        trace_sw += np.sum(c ** 2)
        Z_parts.append(c)
    if trace_sw < 1e-12:
        return 0.0, 0.0
    kappa = float(min(trace_sb / trace_sw, 100.0))
    Z = np.concatenate(Z_parts, axis=0)
    s = np.linalg.svd(Z, compute_uv=False)
    s2, s4 = s**2, s**4
    tsq = float(s4.sum())
    eta = float((trace_sw**2) / (d * tsq)) if tsq > 1e-20 else 0.0
    return kappa, eta


def compute_knn(X, labels, frac=0.7):
    n = len(labels)
    n_train = int(frac * n)
    if n_train < 5 or n - n_train < 5:
        return 0.0
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X[:n_train], labels[:n_train])
    return float(knn.score(X[n_train:], labels[n_train:]))


@torch.no_grad()
def extract_reps(model, tokenizer, texts, device=DEVICE, batch_size=32):
    all_reps = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1].float()
        mask = enc.get("attention_mask",
                       torch.ones(enc["input_ids"].shape, device=device))
        m = mask.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        all_reps.append(pooled.cpu().numpy())
    return np.concatenate(all_reps, axis=0)


def evaluate(model, tokenizer, texts, labels, device=DEVICE):
    model.eval()
    X = extract_reps(model, tokenizer, texts, device)
    knn_acc = compute_knn(X, labels)
    kappa, eta = compute_kappa_and_eta(X, labels)
    K = len(np.unique(labels))
    q = (knn_acc - 1.0/K) / (1.0 - 1.0/K)
    return {"knn": knn_acc, "kappa": kappa, "eta": eta, "q": q, "K": K}


def compute_isotropy_reg(hidden_states, sample_n=HIDDEN_SAMPLE):
    H = hidden_states.reshape(-1, hidden_states.shape[-1])
    if H.shape[0] > sample_n:
        idx = torch.randperm(H.shape[0], device=H.device)[:sample_n]
        H = H[idx]
    H = H - H.mean(0, keepdim=True)
    N, d = H.shape
    C = H.T @ H / N
    tr_C = torch.trace(C)
    tr_C2 = torch.trace(C @ C)
    return tr_C ** 2 / (d * tr_C2 + 1e-10)


def compute_ppl(model, tokenizer, eval_texts, device=DEVICE):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in eval_texts[:50]:
            if not text.strip():
                continue
            enc = tokenizer(text, truncation=True, max_length=SEQ_LEN,
                            return_tensors="pt").to(device)
            if enc["input_ids"].shape[1] < 2:
                continue
            out = model(**enc, labels=enc["input_ids"])
            n_tok = enc["input_ids"].shape[1] - 1
            total_loss += out.loss.item() * n_tok
            total_tokens += n_tok
    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float("inf")


def train_branch(name, model, tokenizer, dataloader, clinc_texts,
                 clinc_labels, ppl_texts, lam, device=DEVICE):
    print(f"\n  === {name} (lambda={lam}) ===", flush=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    trajectory = []
    step = 0

    # Initial eval
    geo = evaluate(model, tokenizer, clinc_texts, clinc_labels, device)
    ppl = compute_ppl(model, tokenizer, ppl_texts, device)
    geo["step"] = 0
    geo["ppl"] = ppl
    trajectory.append(geo)
    print(f"    [0] kNN={geo['knn']:.4f} eta={geo['eta']:.4f} ppl={ppl:.1f}",
          flush=True)

    model.train()
    t0 = time.time()
    data_iter = iter(dataloader)
    accum_lm, accum_reg, n_a = 0.0, 0.0, 0

    while step < NUM_TRAIN_STEPS:
        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids = next(data_iter).to(device)

        need_hidden = (name != "baseline")
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, labels=input_ids,
                            output_hidden_states=need_hidden, return_dict=True)
            lm_loss = outputs.loss

        if name == "eta_preserve":
            hidden = outputs.hidden_states[-1].float()
            eta = compute_isotropy_reg(hidden)
            reg_loss = -torch.log(eta + 1e-8) * lam
            total_loss = lm_loss + reg_loss
        elif name == "eta_collapse":
            hidden = outputs.hidden_states[-1].float()
            eta = compute_isotropy_reg(hidden)
            reg_loss = torch.log(eta + 1e-8) * lam
            total_loss = lm_loss + reg_loss
        else:
            reg_loss = torch.tensor(0.0, device=device)
            total_loss = lm_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        accum_lm += lm_loss.item()
        accum_reg += reg_loss.item()
        n_a += 1
        step += 1

        if step % EVAL_EVERY == 0 or step == NUM_TRAIN_STEPS:
            geo = evaluate(model, tokenizer, clinc_texts, clinc_labels, device)
            ppl = compute_ppl(model, tokenizer, ppl_texts, device)
            geo["step"] = step
            geo["ppl"] = ppl
            geo["lm_loss"] = accum_lm / n_a
            geo["reg_loss"] = accum_reg / n_a
            trajectory.append(geo)
            elapsed = time.time() - t0
            print(f"    [{step}] kNN={geo['knn']:.4f} eta={geo['eta']:.4f} "
                  f"ppl={ppl:.1f} lm={accum_lm/n_a:.3f} {elapsed:.0f}s",
                  flush=True)
            accum_lm, accum_reg, n_a = 0.0, 0.0, 0
            model.train()

        elif step % 200 == 0:
            elapsed = time.time() - t0
            rate = step / elapsed if elapsed > 0 else 0
            print(f"    step {step}/{NUM_TRAIN_STEPS} "
                  f"({rate:.1f} steps/s)", flush=True)

    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    return trajectory


def main():
    print("=" * 70)
    print("LAMBDA SWEEP + PYTHIA-410M REPLICATION")
    print("=" * 70)

    # Load CLINC
    print("\nLoading CLINC...", flush=True)
    clinc_texts, clinc_labels = load_clinc_data()
    print(f"  {len(clinc_texts)} samples, {len(np.unique(clinc_labels))} classes")

    all_results = {}

    # ============================================================
    # PHASE 1: Lambda sweep on Pythia-160m
    # ============================================================
    for model_id, start_step in [
        ("EleutherAI/pythia-160m", 4000),
        ("EleutherAI/pythia-410m", 4000),
    ]:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id} @ step {start_step}")
        print(f"{'='*70}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Pre-tokenize
        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        wiki_texts = [x["text"] for x in wiki]
        token_ids = pretokenize(tokenizer, wiki_texts)
        train_ds = PreTokenizedDataset(token_ids, SEQ_LEN)
        dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0, pin_memory=True)

        ppl_eval = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        ppl_texts = [x["text"] for x in ppl_eval if x["text"].strip()][:100]

        model_key = model_id.split("/")[-1]
        all_results[model_key] = {}

        # Run branches
        configs = [
            ("baseline", 0.0),
            ("eta_preserve_l01", 0.01),
            ("eta_preserve_l02", 0.02),
            ("eta_collapse_l05", 0.05),
        ]

        for branch_name, lam in configs:
            print(f"\nLoading fresh {model_id} step {start_step}...", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=f"step{start_step}",
                torch_dtype=torch.float32,
            ).to(DEVICE)

            # Determine branch type
            if "preserve" in branch_name:
                btype = "eta_preserve"
            elif "collapse" in branch_name:
                btype = "eta_collapse"
            else:
                btype = "baseline"

            traj = train_branch(btype, model, tokenizer, dataloader,
                                clinc_texts, clinc_labels, ppl_texts, lam, DEVICE)
            all_results[model_key][branch_name] = traj

            del model
            gc.collect()
            torch.cuda.empty_cache()

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    for model_key in all_results:
        print(f"\n  --- {model_key} ---")
        branches = all_results[model_key]
        base_final = branches["baseline"][-1]

        for bname, traj in branches.items():
            final = traj[-1]
            ppl_ratio = final["ppl"] / base_final["ppl"] if base_final["ppl"] > 0 else 0
            knn_delta = final["knn"] - base_final["knn"]
            eta_delta = final["eta"] - base_final["eta"]
            match = "MATCH" if abs(ppl_ratio - 1.0) < 0.05 else "miss"
            print(f"    {bname:>25}: kNN={final['knn']:.4f} ({knn_delta:+.4f}) "
                  f"eta={final['eta']:.4f} ({eta_delta:+.4f}) "
                  f"ppl={final['ppl']:.1f} ({ppl_ratio:.3f}) [{match}]")

    # Save
    out_path = RESULTS_DIR / "cti_lambda_sweep.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
