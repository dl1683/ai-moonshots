#!/usr/bin/env python -u
"""
TRAINING DYNAMICS: Track kappa_nearest (true order parameter) during training.

KEY PREDICTION: kappa_spec can increase while q DECREASES (observed in training data).
  This means kappa_spec is NOT monotone with q during training.

THEOREM PREDICTION: kappa_nearest = kappa_spec * h(rank(S_B), K) IS monotone with q.
  If rank(S_B) decreases during neural collapse, h decreases, compensating for kappa_spec increase.

EXPERIMENT: At key checkpoints (steps 4000, 8000, 16000, 32000, 64000, 143000):
  1. Extract representations
  2. Compute rank(S_B) via effective rank = tr(S_B)^2 / tr(S_B^2)
  3. Compute kappa_nearest = kappa_spec * h(eff_rank, K)
  4. Compare kappa_spec and kappa_nearest as predictors of q

Evidence for: kappa_nearest is monotone with q even when kappa_spec is not.
"""

import json
import sys
import gc
import numpy as np
from pathlib import Path
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))


def h_rK(r, K, n_mc=5000, seed=42):
    """h(r, K) = 2 * E[chi^2(r)_min_(K-1)] / r."""
    m = K - 1
    if m < 1:
        return 2.0
    x_max = float(stats.chi2.ppf(1 - 1e-10, df=r))
    xs = np.linspace(0, x_max, 2000)
    dx = xs[1] - xs[0]
    survival = (1.0 - stats.chi2.cdf(xs, df=r)) ** m
    return 2.0 * float(np.sum(survival) * dx) / r


def effective_rank(matrix):
    """Effective rank = tr(A)^2 / tr(A^2) (von Neumann entropy normalization).
    Also known as participation ratio. Returns a float in [1, d].
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    tr_A = float(np.sum(eigenvalues))
    tr_A2 = float(np.sum(eigenvalues ** 2))
    if tr_A2 < 1e-15:
        return 1.0
    return float(tr_A ** 2 / tr_A2)


print("=" * 70)
print("TRAINING RANK TRACK: kappa_nearest vs kappa_spec during training")
print("=" * 70)

# Check if we have GPU for this
try:
    import torch
    has_gpu = torch.cuda.is_available()
    print(f"GPU available: {has_gpu}")
except ImportError:
    has_gpu = False
    print("No PyTorch available")

# Strategy: use the EXISTING cached training dynamics data
# and enrich it with rank estimates from the structure of S_B
# We can estimate effective rank from the existing kappa data
# BUT: we need actual representations to get S_B rank.

# Check if we can load the training geometry cache (has more info)
try:
    geom = json.load(open(RESULTS_DIR / "cti_training_geometry.json"))
    print("Training geometry loaded:", list(geom.keys())[:5])
except:
    geom = None
    print("No training geometry cache")

try:
    geom_cache = json.load(open(RESULTS_DIR / "cti_training_geometry_cache.json"))
    print("Training geometry cache loaded:", list(geom_cache.keys())[:5])
    if geom_cache:
        first_model = list(geom_cache.keys())[0]
        first_val = list(geom_cache[first_model].values())[0]
        print("Fields per checkpoint:", list(first_val.keys()))
except Exception as e:
    geom_cache = None
    print(f"No training geometry cache: {e}")


# ============================================================
# If we have geometry cache with rank, use it directly
# ============================================================

K = 150  # CLINC
results_by_model = {}

if geom_cache:
    for model_key in geom_cache:
        model_rows = []
        for step_str, row in sorted(geom_cache[model_key].items(), key=lambda x: int(x[0])):
            step = int(step_str)
            kappa = row.get("kappa", 0)
            q = row.get("q", 0)
            knn = row.get("knn", 0)
            rank = row.get("rank", None)
            rank_eff = row.get("rank_eff", None) or row.get("eff_dim", None)
            eta = row.get("eta", None)

            if rank_eff is not None:
                h = h_rK(max(int(rank_eff), 1), K)
                kappa_near = kappa * h
            elif rank is not None:
                h = h_rK(max(int(rank), 1), K)
                kappa_near = kappa * h
            else:
                kappa_near = None

            model_rows.append({
                "step": step, "kappa_spec": kappa, "q": q, "knn": knn,
                "rank": rank, "rank_eff": rank_eff, "eta": eta,
                "kappa_nearest": kappa_near, "h": h if kappa_near else None,
            })
        results_by_model[model_key] = model_rows
        print(f"  {model_key}: {len(model_rows)} steps")


# ============================================================
# ANALYSIS: kappa_spec vs kappa_nearest vs q during training
# ============================================================

if not results_by_model:
    # Fall back: use existing training dynamics and try to load model for rank
    dyn = json.load(open(RESULTS_DIR / "cti_training_dynamics.json"))

    if has_gpu:
        print("\nAttempting GPU rank extraction from key checkpoints...")
        # Import needed libraries
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from hierarchical_datasets import load_hierarchical_dataset
        except ImportError as e:
            print(f"Cannot import: {e}")
            has_gpu = False

    if has_gpu:
        # Key steps where discrepancy is visible
        target_steps = [4000, 8000, 16000, 32000, 64000, 143000]
        model_id = "EleutherAI/pythia-160m"
        dataset_name = "clinc"

        print(f"Extracting representations from {model_id} at key steps...")

        def extract_reps_fast(model_id, step, texts, labels, n_classes, device="cuda", batch_size=64):
            """Extract final-layer mean pooled representations."""
            revision = f"step{step}" if step > 0 else "step0"
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                       trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision,
                torch_dtype=torch.float16, trust_remote_code=True, device_map=device,
            )
            model.eval()

            all_reps = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                enc = tokenizer(batch_texts, padding=True, truncation=True,
                                max_length=128, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states[-1]
                mask = enc.attention_mask.unsqueeze(-1).float()
                reps = (hs * mask).sum(1) / mask.sum(1)
                all_reps.append(reps.float().cpu().numpy())

            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return np.vstack(all_reps)

        def compute_geometry(reps, labels, n_classes):
            """Compute kappa, rank_eff, kappa_nearest from representations."""
            from sklearn.neighbors import KNeighborsClassifier
            d = reps.shape[1]
            K = n_classes

            # Centroids
            centroids = np.array([reps[labels == k].mean(0) for k in range(K)])
            grand_mean = reps.mean(0)

            # S_B
            n_k = np.array([np.sum(labels == k) for k in range(K)])
            S_B = np.zeros((d, d))
            for k in range(K):
                diff = (centroids[k] - grand_mean).reshape(-1, 1)
                S_B += n_k[k] * diff @ diff.T

            # S_W
            S_W = np.zeros((d, d))
            for k in range(K):
                Xk = reps[labels == k]
                diff = Xk - centroids[k]
                S_W += diff.T @ diff

            kappa_spec = float(np.trace(S_B) / max(np.trace(S_W), 1e-10))
            rank_eff = effective_rank(S_B)
            h = h_rK(max(int(rank_eff), 1), K)
            kappa_nearest = kappa_spec * h

            # kNN quality
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(reps, labels)
            knn_acc = knn.score(reps, labels)
            q = (knn_acc - 1.0/K) / (1.0 - 1.0/K)

            return {
                "kappa_spec": float(kappa_spec),
                "rank_eff": float(rank_eff),
                "h": float(h),
                "kappa_nearest": float(kappa_nearest),
                "knn_acc": float(knn_acc),
                "q": float(max(q, 0.0)),
            }

        # Load dataset (small subset for speed)
        try:
            texts, labels, label_names = load_hierarchical_dataset(
                dataset_name, max_samples_per_class=10
            )
            n_classes = len(set(labels))
            print(f"Dataset: {len(texts)} samples, {n_classes} classes")

            key_steps_results = []
            for step in target_steps:
                try:
                    print(f"  Step {step}...", end=" ", flush=True)
                    reps = extract_reps_fast(model_id, step, texts, labels, n_classes)
                    geo = compute_geometry(reps, labels, n_classes)
                    geo["step"] = step
                    key_steps_results.append(geo)
                    print(f"kappa_spec={geo['kappa_spec']:.4f}, rank_eff={geo['rank_eff']:.1f}, "
                          f"kappa_near={geo['kappa_nearest']:.4f}, q={geo['q']:.4f}")
                    del reps
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"ERROR: {e}")
                sys.stdout.flush()

            results_by_model[model_id + "_key_steps"] = key_steps_results
        except Exception as e:
            print(f"Dataset load failed: {e}")

    # Merge with existing dynamics data
    dyn_rows = []
    for step_str, row in sorted(dyn.get("models", {}).get("pythia-160m", {}).items(),
                                  key=lambda x: int(x[0])):
        step = int(step_str)
        dyn_rows.append({
            "step": step,
            "kappa_spec": row["kappa"],
            "q": row["q"],
            "kappa_nearest": None,
        })
    results_by_model["pythia-160m_existing"] = dyn_rows


# ============================================================
# COMPUTE CORRELATIONS: kappa_spec vs kappa_nearest with q
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS: Which is more monotone with q during training?")
print("=" * 70)

for model_key, rows in results_by_model.items():
    if not rows:
        continue

    kappa_spec_arr = np.array([r.get("kappa_spec", 0) for r in rows])
    q_arr = np.array([r.get("q", 0) for r in rows])

    # Filter valid
    mask = (kappa_spec_arr > 0) & (q_arr > 0)
    if mask.sum() < 5:
        continue

    kappa_spec_valid = kappa_spec_arr[mask]
    q_valid = q_arr[mask]

    rho_spec, _ = stats.spearmanr(kappa_spec_valid, q_valid)

    kappa_near_arr = np.array([r.get("kappa_nearest", None) or 0 for r in rows])
    if kappa_near_arr[mask].sum() > 0:
        kappa_near_valid = kappa_near_arr[mask]
        rho_near, _ = stats.spearmanr(kappa_near_valid, q_valid)
        print(f"\n{model_key}:")
        print(f"  rho(kappa_spec, q) = {rho_spec:.4f}")
        print(f"  rho(kappa_nearest, q) = {rho_near:.4f}")
        print(f"  kappa_nearest more monotone: {rho_near > rho_spec}")
    else:
        print(f"\n{model_key}:")
        print(f"  rho(kappa_spec, q) = {rho_spec:.4f}")
        print(f"  kappa_nearest not available")

    # Print table: step, kappa_spec, kappa_near, q
    if len(rows) <= 30:
        print(f"  {'step':>8} {'kappa_spec':>12} {'kappa_near':>12} {'q':>8}")
        for r in rows:
            kn = r.get("kappa_nearest")
            kn_str = f"{kn:>12.4f}" if kn is not None else "         N/A"
            print(f"  {r.get('step', '?'):>8} {r.get('kappa_spec', 0):>12.4f} "
                  f"{kn_str} {r.get('q', 0):>8.4f}")
    sys.stdout.flush()


# ============================================================
# THEORY PREDICTION: for the existing dynamics data
# ============================================================

print("\n" + "=" * 70)
print("THEORY PREDICTION: if rank_eff changes during training")
print("=" * 70)
print("""
Theory: kappa_nearest = kappa_spec * h(rank_eff, K)
  - If rank_eff DECREASES during late training (neural collapse):
    h decreases -> kappa_nearest decreases even though kappa_spec increases
  - This would explain: kappa_spec up but q down after step 8000

To verify: need rank_eff at each checkpoint.
Predicted pattern:
  Step 0-8000: rank_eff grows (learning useful features) -> h grows -> kappa_nearest grows faster than kappa_spec
  Step 8000+: rank_eff decreases (neural collapse) -> h shrinks -> kappa_nearest grows slower or decreases

This is a TESTABLE PREDICTION of the theorem.
Expected result: rho(kappa_nearest, q) > rho(kappa_spec, q) over full training trajectory.
""")

# Save
out = {
    "theory_prediction": "kappa_nearest = kappa_spec * h(rank_eff, K) is monotone with q",
    "models": results_by_model,
}
out_path = RESULTS_DIR / "cti_training_rank_track.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"Saved: {out_path.name}")
