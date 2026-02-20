#!/usr/bin/env python
"""
CAUSAL FORKED-TRAINING: ETA CONTROL EXPERIMENT

Codex design (8/10 Nobel rating):
  Fork Pythia-160m from step 4000 (just before quality peak at step 8000).
  Continue training in 3 branches:
    1. BASELINE: standard LM loss only
    2. ETA-PRESERVE: LM loss + lambda * (-log(eta)) -- maintain isotropy
    3. ETA-COLLAPSE: LM loss + lambda * log(eta) -- accelerate collapse

  Key prediction (pre-registered):
    q_t = f(kappa_t, eta_t) from the 2D law should predict ALL 3 trajectories.

  Success criteria:
    1. At matched perplexity (+/- 5%), quality diverges by branch
    2. eta-preserve branch maintains/improves kNN over baseline
    3. eta-collapse branch degrades kNN vs baseline
    4. 2D law q=f(kappa,eta) predicts all trajectories (R^2 > 0.9)
    5. Perplexity within 5% across branches (geometry doesn't hurt LM)
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

# ============================================================
# HYPERPARAMETERS
# ============================================================
MODEL_ID = "EleutherAI/pythia-160m"
START_STEP = 4000
NUM_TRAIN_STEPS = 2000
EVAL_EVERY = 250
BATCH_SIZE = 16
SEQ_LEN = 256
LR = 6e-4
LAMBDA_REG = 0.05
CLINC_MAX_SAMPLES = 2000
HIDDEN_SAMPLE = 256


# ============================================================
# PRE-TOKENIZED DATASET
# ============================================================
class PreTokenizedDataset(Dataset):
    """Pre-tokenized chunks for fast training."""

    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        # Chunk into fixed-length sequences
        n = len(token_ids) // seq_len
        self.chunks = token_ids[:n * seq_len].reshape(n, seq_len)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def pretokenize(tokenizer, texts, max_tokens=2_000_000):
    """Tokenize all texts into one long array, then chunk."""
    print("  Pre-tokenizing (this is one-time)...", flush=True)
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
    print(f"  Tokenized {total:,} tokens", flush=True)
    return np.array(all_ids, dtype=np.int64)


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================
def load_clinc_data(max_samples=CLINC_MAX_SAMPLES):
    """Load CLINC-150 dataset for geometry evaluation."""
    from hierarchical_datasets import load_hierarchical_dataset
    ds = load_hierarchical_dataset("clinc", split="train", max_samples=max_samples)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])
    return texts, labels


def compute_kappa_and_eta(X, labels):
    """Compute kappa and eta from representations."""
    d = X.shape[1]
    unique_labels = np.unique(labels)
    grand_mean = X.mean(0)

    Z_parts = []
    trace_sb = 0.0
    trace_sw = 0.0

    for lbl in unique_labels:
        lbl_mask = labels == lbl
        n_k = lbl_mask.sum()
        if n_k < 2:
            continue
        X_k = X[lbl_mask]
        mu_k = X_k.mean(0)
        trace_sb += n_k * np.sum((mu_k - grand_mean) ** 2)
        centered = X_k - mu_k
        trace_sw += np.sum(centered ** 2)
        Z_parts.append(centered)

    if trace_sw < 1e-12:
        return 0.0, 0.0

    kappa = float(min(trace_sb / trace_sw, 100.0))
    Z = np.concatenate(Z_parts, axis=0)
    s = np.linalg.svd(Z, compute_uv=False)
    s2 = s ** 2
    s4 = s2 ** 2
    trace_sw_sq = float(s4.sum())
    if trace_sw_sq < 1e-20:
        eta = 0.0
    else:
        eta = float((trace_sw ** 2) / (d * trace_sw_sq))

    return kappa, eta


def compute_knn(X, labels, n_train_frac=0.7):
    """kNN accuracy."""
    n = len(labels)
    n_train = int(n_train_frac * n)
    if n_train < 5 or n - n_train < 5:
        return 0.0
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X[:n_train], labels[:n_train])
    return float(knn.score(X[n_train:], labels[n_train:]))


def compute_dist_ratio(X, labels):
    """Distance ratio: mean(nearest diff-class) / mean(nearest same-class)."""
    from sklearn.metrics import pairwise_distances
    n = min(len(X), 400)
    idx = np.random.choice(len(X), n, replace=False)
    X_sub = X[idx]
    labels_sub = labels[idx]

    D = pairwise_distances(X_sub, metric="cosine")
    np.fill_diagonal(D, np.inf)

    same_dists = []
    diff_dists = []
    for i in range(n):
        same_mask = labels_sub == labels_sub[i]
        same_mask[i] = False
        diff_mask = ~same_mask
        diff_mask[i] = False

        if same_mask.any():
            same_dists.append(D[i, same_mask].min())
        if diff_mask.any():
            diff_dists.append(D[i, diff_mask].min())

    if not same_dists or not diff_dists:
        return 1.0
    return float(np.mean(diff_dists) / max(np.mean(same_dists), 1e-10))


@torch.no_grad()
def extract_reps(model, tokenizer, texts, device=DEVICE, batch_size=32):
    """Extract last-layer representations (mean-pool, L2-norm)."""
    all_reps = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = texts[i * batch_size:(i + 1) * batch_size]
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


def evaluate_geometry(model, tokenizer, texts, labels, device=DEVICE):
    """Full geometry evaluation on CLINC."""
    model.eval()
    X = extract_reps(model, tokenizer, texts, device)
    knn_acc = compute_knn(X, labels)
    kappa, eta = compute_kappa_and_eta(X, labels)
    dist_ratio = compute_dist_ratio(X, labels)
    K = len(np.unique(labels))
    q = (knn_acc - 1.0 / K) / (1.0 - 1.0 / K)
    return {
        "knn": knn_acc,
        "kappa": kappa,
        "eta": eta,
        "dist_ratio": dist_ratio,
        "q": q,
        "K": K,
    }


def compute_isotropy_reg(hidden_states, sample_n=HIDDEN_SAMPLE):
    """Differentiable isotropy: eta = tr(C)^2 / (d * tr(C^2))."""
    H = hidden_states.reshape(-1, hidden_states.shape[-1])
    if H.shape[0] > sample_n:
        idx = torch.randperm(H.shape[0], device=H.device)[:sample_n]
        H = H[idx]
    H = H - H.mean(0, keepdim=True)
    N, d = H.shape
    C = H.T @ H / N
    tr_C = torch.trace(C)
    tr_C2 = torch.trace(C @ C)
    eta = tr_C ** 2 / (d * tr_C2 + 1e-10)
    return eta


def compute_perplexity(model, tokenizer, eval_texts, device=DEVICE):
    """Compute perplexity on a small text sample."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

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

    if total_tokens == 0:
        return float("inf")
    return float(np.exp(total_loss / total_tokens))


# ============================================================
# TRAINING LOOP
# ============================================================
def train_branch(branch_name, model, tokenizer, dataloader, clinc_texts,
                 clinc_labels, ppl_eval_texts, device=DEVICE):
    """Train one branch and collect trajectory data."""
    print(f"\n{'='*60}")
    print(f"  BRANCH: {branch_name}")
    print(f"{'='*60}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    trajectory = []
    step = 0

    # Initial evaluation
    print(f"  [Step 0] Evaluating geometry...", flush=True)
    geo = evaluate_geometry(model, tokenizer, clinc_texts, clinc_labels, device)
    ppl = compute_perplexity(model, tokenizer, ppl_eval_texts, device)
    geo["step"] = 0
    geo["ppl"] = ppl
    geo["lm_loss"] = 0.0
    geo["reg_loss"] = 0.0
    trajectory.append(geo)
    print(f"    kNN={geo['knn']:.4f}, kappa={geo['kappa']:.4f}, "
          f"eta={geo['eta']:.4f}, dr={geo['dist_ratio']:.4f}, ppl={ppl:.2f}",
          flush=True)

    model.train()
    t0 = time.time()
    data_iter = iter(dataloader)
    accum_lm = 0.0
    accum_reg = 0.0
    n_accum = 0

    while step < NUM_TRAIN_STEPS:
        # Get pre-tokenized batch
        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids = next(data_iter).to(device)

        # Forward with AMP
        need_hidden = (branch_name != "baseline")
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, labels=input_ids,
                            output_hidden_states=need_hidden, return_dict=True)
            lm_loss = outputs.loss

        # Regularizer (in float32)
        if branch_name == "eta_preserve":
            hidden = outputs.hidden_states[-1].float()
            eta = compute_isotropy_reg(hidden)
            reg_loss = -torch.log(eta + 1e-8) * LAMBDA_REG
            total_loss = lm_loss + reg_loss
        elif branch_name == "eta_collapse":
            hidden = outputs.hidden_states[-1].float()
            eta = compute_isotropy_reg(hidden)
            reg_loss = torch.log(eta + 1e-8) * LAMBDA_REG
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
        n_accum += 1
        step += 1

        # Periodic evaluation
        if step % EVAL_EVERY == 0 or step == NUM_TRAIN_STEPS:
            avg_lm = accum_lm / n_accum
            avg_reg = accum_reg / n_accum

            elapsed = time.time() - t0
            print(f"  [Step {step}] lm={avg_lm:.4f}, reg={avg_reg:.4f}, "
                  f"{elapsed:.0f}s", flush=True)

            geo = evaluate_geometry(model, tokenizer, clinc_texts,
                                    clinc_labels, device)
            ppl = compute_perplexity(model, tokenizer, ppl_eval_texts, device)
            geo["step"] = step
            geo["ppl"] = ppl
            geo["lm_loss"] = avg_lm
            geo["reg_loss"] = avg_reg
            trajectory.append(geo)
            print(f"    kNN={geo['knn']:.4f}, kappa={geo['kappa']:.4f}, "
                  f"eta={geo['eta']:.4f}, dr={geo['dist_ratio']:.4f}, "
                  f"ppl={ppl:.2f}", flush=True)

            accum_lm = 0.0
            accum_reg = 0.0
            n_accum = 0
            model.train()

        elif step % 50 == 0:
            elapsed = time.time() - t0
            rate = step / elapsed if elapsed > 0 else 0
            rem = (NUM_TRAIN_STEPS - step) / rate if rate > 0 else 0
            print(f"    step {step}/{NUM_TRAIN_STEPS} "
                  f"({rate:.1f} steps/s, ~{rem:.0f}s left)", flush=True)

    total_time = time.time() - t0
    print(f"\n  Branch '{branch_name}' done in {total_time:.0f}s", flush=True)
    return trajectory


def main():
    print("=" * 70)
    print("CAUSAL FORKED-TRAINING: ETA CONTROL EXPERIMENT")
    print("=" * 70)
    print(f"Model: {MODEL_ID} @ step {START_STEP}")
    print(f"Training: {NUM_TRAIN_STEPS} steps, eval every {EVAL_EVERY}")
    print(f"Lambda: {LAMBDA_REG}, Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}")
    print(f"Branches: baseline, eta_preserve, eta_collapse")

    # Load CLINC for evaluation
    print("\nLoading CLINC evaluation data...")
    clinc_texts, clinc_labels = load_clinc_data()
    print(f"  CLINC: {len(clinc_texts)} samples, "
          f"{len(np.unique(clinc_labels))} classes")

    # Load and pre-tokenize training data
    print("\nLoading and pre-tokenizing WikiText-103...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    wiki_texts = [x["text"] for x in wiki]
    token_ids = pretokenize(tokenizer, wiki_texts, max_tokens=2_000_000)
    train_dataset = PreTokenizedDataset(token_ids, SEQ_LEN)
    print(f"  {len(train_dataset)} training chunks of length {SEQ_LEN}")

    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True)

    # PPL eval texts
    ppl_eval = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    ppl_eval_texts = [x["text"] for x in ppl_eval if x["text"].strip()][:100]
    print(f"  PPL eval: {len(ppl_eval_texts)} samples")

    # Run 3 branches
    all_results = {}
    branches = ["baseline", "eta_preserve", "eta_collapse"]

    for branch in branches:
        print(f"\n{'='*70}")
        print(f"Loading fresh model from step {START_STEP} for {branch}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=f"step{START_STEP}",
            torch_dtype=torch.float32,
        ).to(DEVICE)

        trajectory = train_branch(
            branch, model, tokenizer, dataloader,
            clinc_texts, clinc_labels, ppl_eval_texts, DEVICE
        )
        all_results[branch] = trajectory

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("ANALYSIS: COMPARING BRANCHES")
    print(f"{'='*70}")

    for branch in branches:
        traj = all_results[branch]
        final = traj[-1]
        initial = traj[0]
        print(f"\n  {branch}:")
        print(f"    kNN:  {initial['knn']:.4f} -> {final['knn']:.4f} "
              f"(delta={final['knn']-initial['knn']:+.4f})")
        print(f"    kappa: {initial['kappa']:.4f} -> {final['kappa']:.4f}")
        print(f"    eta:  {initial['eta']:.4f} -> {final['eta']:.4f} "
              f"(delta={final['eta']-initial['eta']:+.4f})")
        print(f"    dr:   {initial['dist_ratio']:.4f} -> {final['dist_ratio']:.4f}")
        print(f"    ppl:  {initial['ppl']:.2f} -> {final['ppl']:.2f}")

    # Perplexity matching
    print(f"\n  --- PERPLEXITY MATCHING ---")
    final_ppls = {b: all_results[b][-1]["ppl"] for b in branches}
    base_ppl = final_ppls["baseline"]
    for branch in branches:
        ppl_ratio = final_ppls[branch] / base_ppl if base_ppl > 0 else 0
        match = "MATCH" if abs(ppl_ratio - 1.0) < 0.05 else "MISMATCH"
        print(f"    {branch}: ppl={final_ppls[branch]:.2f} "
              f"(ratio={ppl_ratio:.4f}) [{match}]")

    # Quality divergence
    print(f"\n  --- QUALITY DIVERGENCE ---")
    final_knns = {b: all_results[b][-1]["knn"] for b in branches}
    base_knn = final_knns["baseline"]
    for branch in branches:
        diff = final_knns[branch] - base_knn
        print(f"    {branch}: kNN={final_knns[branch]:.4f} "
              f"(vs baseline: {diff:+.4f})")

    # 2D law prediction
    print(f"\n  --- 2D LAW PREDICTION ---")
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr

    all_kappa = []
    all_eta = []
    all_q = []
    for branch in branches:
        for pt in all_results[branch]:
            if pt["kappa"] > 0 and pt["eta"] > 0:
                all_kappa.append(pt["kappa"])
                all_eta.append(pt["eta"])
                all_q.append(pt["q"])

    all_kappa = np.array(all_kappa)
    all_eta = np.array(all_eta)
    all_q = np.array(all_q)

    def sigmoid_2d(X, a, b, c):
        kappa, eta = X
        z = a * kappa * eta ** b + c
        return 1.0 / (1.0 + np.exp(np.clip(-z, -500, 500)))

    try:
        popt, _ = curve_fit(sigmoid_2d, (all_kappa, all_eta), all_q,
                            p0=[5.0, 0.5, -2.0], maxfev=5000)
        q_pred = sigmoid_2d((all_kappa, all_eta), *popt)
        r, p = pearsonr(all_q, q_pred)
        ss_res = np.sum((all_q - q_pred) ** 2)
        ss_tot = np.sum((all_q - all_q.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mae = np.mean(np.abs(all_q - q_pred))
        print(f"    2D law fit: R^2={r2:.4f}, r={r:.4f}, MAE={mae:.4f}")
        print(f"    Params: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
    except Exception as e:
        print(f"    2D law fit failed: {e}")
        r2 = 0.0

    # ============================================================
    # SCORECARD
    # ============================================================
    print(f"\n{'='*70}")
    print("SCORECARD")
    print(f"{'='*70}")

    ppl_matched = all(abs(final_ppls[b] / base_ppl - 1.0) < 0.05
                      for b in branches if b != "baseline")
    preserve_better = final_knns.get("eta_preserve", 0) > base_knn
    collapse_worse = final_knns.get("eta_collapse", 0) < base_knn
    eta_preserve_maintained = (all_results["eta_preserve"][-1]["eta"] >
                                all_results["baseline"][-1]["eta"])
    eta_collapse_lower = (all_results["eta_collapse"][-1]["eta"] <
                           all_results["baseline"][-1]["eta"])

    checks = [
        ("Perplexity matched within 5% across branches",
         ppl_matched,
         f"preserve={final_ppls.get('eta_preserve', 0):.1f}, "
         f"collapse={final_ppls.get('eta_collapse', 0):.1f}, "
         f"base={base_ppl:.1f}"),
        ("eta_preserve maintains/improves kNN over baseline",
         preserve_better,
         f"preserve={final_knns.get('eta_preserve', 0):.4f} vs "
         f"base={base_knn:.4f}"),
        ("eta_collapse degrades kNN vs baseline",
         collapse_worse,
         f"collapse={final_knns.get('eta_collapse', 0):.4f} vs "
         f"base={base_knn:.4f}"),
        ("eta directions correct (preserve>base>collapse)",
         eta_preserve_maintained and eta_collapse_lower,
         f"preserve_eta={all_results['eta_preserve'][-1]['eta']:.4f}, "
         f"base_eta={all_results['baseline'][-1]['eta']:.4f}, "
         f"collapse_eta={all_results['eta_collapse'][-1]['eta']:.4f}"),
        ("2D law predicts all trajectories (R^2 > 0.9)",
         r2 > 0.9,
         f"R^2={r2:.4f}"),
    ]

    passes = sum(1 for _, p, _ in checks if p)
    for criterion, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}")
        print(f"         {val}")
    print(f"\n  TOTAL: {passes}/{len(checks)}")

    # Save
    results = {
        "experiment": "forked_training_eta_control",
        "model": MODEL_ID,
        "start_step": START_STEP,
        "num_steps": NUM_TRAIN_STEPS,
        "lambda_reg": LAMBDA_REG,
        "eval_every": EVAL_EVERY,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "branches": {b: all_results[b] for b in branches},
        "scorecard": {
            "passes": passes,
            "total": len(checks),
            "checks": [{
                "criterion": c,
                "passed": bool(p),
                "value": v,
            } for c, p, v in checks],
        },
    }

    out_path = RESULTS_DIR / "cti_forked_training.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
