#!/usr/bin/env python
"""
2x2 ID-vs-NOISE DISENTANGLEMENT

Separates whether the causal driver of quality onset is:
  (a) intrinsic dimensionality (ID) itself, or
  (b) signal-to-noise ratio (SNR) destroyed by noise injection

Design:
  A. baseline       — natural representations
  B. noise-only     — isotropic noise (degrades SNR, minimal ID change)
  C. ID-up-clean    — spectral reweighting (increases ID, no external noise)
  D. ID-up-noise    — noise in low-var dims (increases ID AND degrades SNR)

Pre-registered prediction:
  If ID is causal: C should delay transition (not just D)
  If SNR is causal: B should delay transition similarly to D, C should not

Model: Qwen/Qwen2-0.5B (from MODEL_DIRECTORY.md)
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from cti_residual_surgery import load_model, ResidualScaler
from hierarchical_datasets import load_hierarchical_dataset


def extract_all_layer_reps(model, tokenizer, texts, alpha, device="cuda", batch_size=32):
    """Extract all layer representations with residual scaling."""
    all_hidden = {}
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with ResidualScaler(model, alpha):
        for i in range(n_batches):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**enc, output_hidden_states=True, return_dict=True)
            mask = enc.get("attention_mask",
                           torch.ones(enc["input_ids"].shape, device=device))
            for idx, hs in enumerate(outputs.hidden_states):
                hs_f = hs.float()
                m = mask.unsqueeze(-1).float()
                pooled = (hs_f * m).sum(1) / m.sum(1).clamp(min=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if idx not in all_hidden:
                    all_hidden[idx] = []
                all_hidden[idx].append(pooled.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in all_hidden.items()}


def compute_id(X):
    """Participation ratio."""
    try:
        Xc = X - X.mean(0)
        if np.isnan(Xc).any() or np.std(Xc) < 1e-10:
            return 1.0
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        ev = S ** 2 / max(X.shape[0] - 1, 1)
        denom = (ev ** 2).sum()
        if denom < 1e-20:
            return 1.0
        return float((ev.sum() ** 2) / denom)
    except np.linalg.LinAlgError:
        return 1.0


def compute_knn(X, labels, n_train_frac=0.7):
    """kNN accuracy."""
    n = len(labels)
    n_train = int(n_train_frac * n)
    if n_train < 5 or n - n_train < 5:
        return 0.0
    try:
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X[:n_train], labels[:n_train])
        return float(knn.score(X[n_train:], labels[n_train:]))
    except Exception:
        return 0.0


def normalize(X):
    """L2-normalize rows."""
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(norms, 1e-8)


# ============================================================
# INTERVENTION CONDITIONS
# ============================================================

def condition_B_noise_only(X, energy_frac=0.3, rng=None):
    """Add ISOTROPIC noise matched to a fraction of signal energy.
    This degrades SNR but barely changes ID (isotropic noise has max ID).
    Key: noise is in ALL directions equally, so the eigenvalue distribution
    gets a uniform additive shift, barely changing participation ratio
    relative to the signal's peaked spectrum.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    signal_energy = np.mean(np.sum(X ** 2, axis=1))
    noise_std = np.sqrt(energy_frac * signal_energy / X.shape[1])
    noise = rng.randn(*X.shape) * noise_std
    return normalize(X + noise)


def condition_C_id_up_clean(X):
    """Increase ID by spectral reweighting WITHOUT adding external noise.
    Redistribute variance from top PCs to bottom PCs.
    This flattens the eigenvalue spectrum, increasing participation ratio,
    while keeping the exact same subspace and no random noise.
    """
    try:
        Xc = X - X.mean(0)
        if np.isnan(Xc).any() or np.std(Xc) < 1e-10:
            return X.copy()
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # Flatten the spectrum: S_new = S^(1/3) * (total_energy / new_total_energy)^(1/2)
        # This makes the spectrum more uniform (higher ID) while preserving total energy
        S_new = S ** (1.0 / 3.0)
        # Match total energy
        old_energy = np.sum(S ** 2)
        new_energy = np.sum(S_new ** 2)
        if new_energy > 0:
            S_new *= np.sqrt(old_energy / new_energy)
        X_new = (U * S_new[None, :]) @ Vt + X.mean(0)
        return normalize(X_new)
    except np.linalg.LinAlgError:
        return X.copy()


def condition_D_id_up_noise(X, noise_scale=0.3, rng=None):
    """Original expansion: noise in low-variance dims (increases ID + degrades SNR)."""
    if rng is None:
        rng = np.random.RandomState(42)
    try:
        Xc = X - X.mean(0)
        if np.isnan(Xc).any() or np.std(Xc) < 1e-10:
            return X.copy()
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        weights = 1.0 / (S + 1e-8)
        weights = weights / weights.max()
        noise = rng.randn(X.shape[0], len(S))
        noise = noise * weights[None, :] * noise_scale
        X_noisy = X + noise @ Vt
        return normalize(X_noisy)
    except np.linalg.LinAlgError:
        return X.copy()


def find_knn_transition(alphas, knn_values):
    """Find alpha where kNN crosses 50% of its range."""
    alphas = np.array(alphas)
    knn = np.array(knn_values)
    knn_min, knn_max = knn.min(), knn.max()
    if knn_max - knn_min < 0.01:
        return None
    knn_norm = (knn - knn_min) / (knn_max - knn_min)
    for i in range(len(alphas) - 1):
        if knn_norm[i] <= 0.5 and knn_norm[i + 1] > 0.5:
            frac = (0.5 - knn_norm[i]) / (knn_norm[i + 1] - knn_norm[i])
            return float(alphas[i] + frac * (alphas[i + 1] - alphas[i]))
    return None


def main():
    print("=" * 70)
    print("2x2 ID-vs-NOISE DISENTANGLEMENT")
    print("Does ID itself shift quality onset, or is it just SNR?")
    print("=" * 70)

    model_id = "Qwen/Qwen2-0.5B"  # From MODEL_DIRECTORY.md
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [0.0, 0.3, 0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

    ds = load_hierarchical_dataset("clinc", split="test", max_samples=2000)
    texts = [s.text for s in ds.samples]
    labels = np.array([s.level1_label for s in ds.samples])

    model, tokenizer, n_layers, n_params = load_model(model_id, device)

    conditions = ["A_baseline", "B_noise_only", "C_id_up_clean", "D_id_up_noise"]
    all_results = {}
    rng = np.random.RandomState(42)

    for alpha in alphas:
        print(f"\nalpha={alpha:.2f}", flush=True)

        # Extract representations once
        reps = extract_all_layer_reps(model, tokenizer, texts, alpha, device)
        result = {}

        for cond in conditions:
            t0 = time.time()
            knn_vals, id_vals = [], []

            for layer_idx in sorted(reps.keys()):
                X = reps[layer_idx]
                if X.shape[0] < 20:
                    continue

                # Apply intervention
                if cond == "A_baseline":
                    X_mod = X
                elif cond == "B_noise_only":
                    X_mod = condition_B_noise_only(X, energy_frac=0.3, rng=rng)
                elif cond == "C_id_up_clean":
                    X_mod = condition_C_id_up_clean(X)
                elif cond == "D_id_up_noise":
                    X_mod = condition_D_id_up_noise(X, noise_scale=0.3, rng=rng)
                else:
                    X_mod = X

                knn_vals.append(compute_knn(X_mod, labels))
                id_vals.append(compute_id(X_mod))

            mean_knn = float(np.mean(knn_vals)) if knn_vals else 0
            mean_id = float(np.mean(id_vals)) if id_vals else 0
            elapsed = time.time() - t0

            print(f"  {cond:20s}: kNN={mean_knn:.3f}  ID={mean_id:.1f}  ({elapsed:.1f}s)")
            result[cond] = {"knn_acc": mean_knn, "intrinsic_dim": mean_id}

        all_results[str(alpha)] = result
        sys.stdout.flush()

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("TRANSITION ANALYSIS")
    print(f"{'='*70}")

    transitions = {}
    for cond in conditions:
        knn_vals = [all_results[str(a)][cond]["knn_acc"] for a in alphas]
        id_vals = [all_results[str(a)][cond]["intrinsic_dim"] for a in alphas]
        alpha_knn = find_knn_transition(alphas, knn_vals)
        id_peak_idx = np.argmax(id_vals)

        transitions[cond] = {
            "alpha_knn": alpha_knn,
            "alpha_id_peak": float(alphas[id_peak_idx]),
            "peak_id": float(id_vals[id_peak_idx]),
            "knn_at_1": float(knn_vals[-1]),
            "mean_id": float(np.mean(id_vals)),
        }

        print(f"\n  {cond:20s}:")
        print(f"    ID peak={alphas[id_peak_idx]:.2f} (mean ID={np.mean(id_vals):.1f})")
        print(f"    kNN transition={alpha_knn}")
        print(f"    kNN at alpha=1: {knn_vals[-1]:.3f}")

    # ============================================================
    # KEY COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("2x2 DISENTANGLEMENT VERDICT")
    print(f"{'='*70}")

    a_trans = transitions["A_baseline"]["alpha_knn"]
    b_trans = transitions["B_noise_only"]["alpha_knn"]
    c_trans = transitions["C_id_up_clean"]["alpha_knn"]
    d_trans = transitions["D_id_up_noise"]["alpha_knn"]

    print(f"\n  A (baseline):      transition = {a_trans}")
    print(f"  B (noise-only):    transition = {b_trans}")
    print(f"  C (ID-up clean):   transition = {c_trans}")
    print(f"  D (ID-up + noise): transition = {d_trans}")

    if a_trans is not None:
        print(f"\n  Shifts from baseline (alpha={a_trans:.3f}):")
        for name, trans in [("B_noise_only", b_trans),
                            ("C_id_up_clean", c_trans),
                            ("D_id_up_noise", d_trans)]:
            if trans is not None:
                shift = trans - a_trans
                print(f"    {name:20s}: shift = {shift:+.3f}")
            else:
                print(f"    {name:20s}: NO TRANSITION (blocked)")

    # Determine verdict
    print(f"\n  INTERPRETATION:")
    b_blocks = b_trans is None
    c_blocks = c_trans is None
    d_blocks = d_trans is None

    b_shift = (b_trans - a_trans) if (b_trans and a_trans) else None
    c_shift = (c_trans - a_trans) if (c_trans and a_trans) else None

    if c_blocks and not b_blocks:
        print("    ID-UP (clean) BLOCKS quality but NOISE-ONLY does NOT")
        print("    => ID itself is the causal driver, not SNR")
        print("    => STRONG support for geometric/spectral mechanism")
        verdict = "ID_CAUSAL"
    elif c_blocks and b_blocks:
        print("    BOTH block quality")
        print("    => Cannot separate ID from SNR with this design")
        verdict = "AMBIGUOUS"
    elif not c_blocks and not b_blocks:
        if c_shift is not None and b_shift is not None:
            if abs(c_shift) > abs(b_shift) + 0.02:
                print("    ID-UP (clean) delays MORE than noise-only")
                print("    => ID has INDEPENDENT causal effect beyond SNR")
                verdict = "ID_CAUSAL_PARTIAL"
            elif abs(b_shift) > abs(c_shift) + 0.02:
                print("    NOISE delays MORE than clean ID-up")
                print("    => SNR is the primary driver, not ID")
                verdict = "SNR_CAUSAL"
            else:
                print("    Both shift similarly")
                print("    => Cannot cleanly separate ID from SNR")
                verdict = "AMBIGUOUS"
        else:
            verdict = "UNCLEAR"
            print("    Insufficient transition data")
    elif not c_blocks and b_blocks:
        print("    NOISE blocks but clean ID-up does NOT")
        print("    => SNR is the driver, ID change alone is insufficient")
        verdict = "SNR_CAUSAL"
    else:
        verdict = "UNCLEAR"
        print("    Unclear pattern")

    print(f"\n  VERDICT: {verdict}")

    # Also compare mean IDs to verify interventions worked
    print(f"\n  INTERVENTION VERIFICATION (mean ID across all alphas):")
    for cond in conditions:
        mean_ids = [all_results[str(a)][cond]["intrinsic_dim"] for a in alphas]
        print(f"    {cond:20s}: mean ID = {np.mean(mean_ids):.1f}")

    # Save
    out = {
        "experiment": "2x2_id_noise_disentanglement",
        "model_id": model_id,
        "alphas": alphas,
        "conditions": conditions,
        "results": all_results,
        "transitions": transitions,
        "verdict": verdict,
    }
    out_path = RESULTS_DIR / "cti_id_noise_disentangle.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
