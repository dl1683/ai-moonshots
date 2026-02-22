#!/usr/bin/env python -u
"""
CTI Audio/Speech Modality Test
================================
Tests whether logit(q_norm) = alpha * kappa_nearest + C holds for SPEECH embeddings,
extending the CTI Universal Law from text+vision to audio (3rd modality).

Dataset: MINDS14 (K=14 intent classes, en-US split, n=563)
  - K=14 matches DBpedia text dataset (controlled K comparison)
Models: wav2vec2-base-960h, hubert-base-ls960

Pre-hypothesis: The FUNCTIONAL FORM should hold (r > 0.7) even if alpha differs from NLP.
"""

import json
import numpy as np
from pathlib import Path
from scipy.special import logit as sp_logit
from scipy.stats import pearsonr

import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio speech models
AUDIO_MODELS = [
    "facebook/wav2vec2-base-960h",
    "facebook/hubert-base-ls960",
]

# Reference NLP alpha (per-dataset-intercept form, from LOAO 12-arch)
ALPHA_NLP = 1.477
ALPHA_NLP_CV = 0.023


def load_minds14():
    """Load MINDS14 en-US, return (audio_arrays, labels, sampling_rate)."""
    import soundfile as sf
    import io
    import datasets as _datasets

    ds = _datasets.load_dataset("PolyAI/minds14", "en-US", split="train")
    ds_raw = ds.cast_column("audio", _datasets.Audio(decode=False))
    # Read labels from ds_raw (not ds) to avoid triggering audio decoder
    labels = np.array([x["intent_class"] for x in ds_raw])

    audio_arrays = []
    source_sr = None
    for item in ds_raw:
        audio_bytes = item["audio"]["bytes"]
        arr, sr = sf.read(io.BytesIO(audio_bytes))
        if arr.ndim > 1:
            arr = arr.mean(axis=1)  # stereo -> mono
        audio_arrays.append(arr.astype(np.float32))
        source_sr = sr

    print(f"  MINDS14 en-US: n={len(audio_arrays)}, K={len(np.unique(labels))}, sr={source_sr}")
    return audio_arrays, labels, int(source_sr)


def get_audio_embeddings_at_layers(model_name, audio_arrays, sampling_rate, batch_size=8):
    """Extract embeddings at proportional-depth layers from a wav2vec2/HuBERT model."""
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    from transformers import HubertModel

    model_short = model_name.split("/")[-1]

    if "hubert" in model_name.lower():
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = HubertModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(DEVICE).eval()
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(
            model_name, output_hidden_states=True
        ).to(DEVICE).eval()

    n_layers = model.config.num_hidden_layers
    layer_indices = sorted(set([
        max(1, round(n_layers * 0.25)),
        max(1, round(n_layers * 0.50)),
        max(1, round(n_layers * 0.75)),
        n_layers,
    ]))
    print(f"  {model_short}: {n_layers} transformer layers, testing at {layer_indices}")

    all_layer_embs = {l: [] for l in layer_indices}

    for i in range(0, len(audio_arrays), batch_size):
        batch = audio_arrays[i:i+batch_size]

        # Resample if needed (processor expects 16kHz)
        # Resample from source_sr to 16000 if needed
        TARGET_SR = 16000
        if sampling_rate != TARGET_SR:
            import resampy
            batch = [resampy.resample(x, sampling_rate, TARGET_SR) for x in batch]
            effective_sr = TARGET_SR
        else:
            effective_sr = sampling_rate

        inputs = processor(
            batch,
            sampling_rate=effective_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # (n_layers+1, B, T, d)

        for li in layer_indices:
            if li < len(hidden_states):
                h = hidden_states[li].float()  # (B, T, d)
                # Mean pool over time dimension
                # Use attention_mask for proper masking if available
                if "attention_mask" in inputs:
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    # HuBERT/wav2vec2 use feature extractor, mask has different length
                    # Use simple mean over all time steps
                emb = h.mean(dim=1)  # (B, d)
                all_layer_embs[li].append(
                    np.nan_to_num(emb.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                )

    del model
    torch.cuda.empty_cache()

    for l in layer_indices:
        if all_layer_embs[l]:
            all_layer_embs[l] = np.vstack(all_layer_embs[l])
        else:
            all_layer_embs[l] = None

    return all_layer_embs, layer_indices


def compute_kappa_nearest(X, y):
    """kappa_nearest = mean_class(delta_min / (sigma_W * sqrt(d)))."""
    classes = np.unique(y)
    if len(classes) < 2:
        return None
    mu = {c: X[y == c].mean(0) for c in classes if (y == c).sum() >= 2}
    if len(mu) < 2:
        return None
    # Pooled within-class std (per dimension)
    within_var = sum(np.sum((X[y == c] - mu[c])**2) for c in mu)
    n_total = sum((y == c).sum() for c in mu)
    sigma_W = float(np.sqrt(within_var / (n_total * X.shape[1])))
    if sigma_W < 1e-10:
        return None
    kappas = []
    for c in mu:
        dists = [np.linalg.norm(mu[c] - mu[j]) for j in mu if j != c]
        kappas.append(min(dists) / (sigma_W * np.sqrt(X.shape[1])))
    return float(np.mean(kappas))


def compute_knn_q(X, y):
    """q_norm = (1NN_acc - 1/K) / (1 - 1/K)."""
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr_idx, te_idx = idx[:split], idx[split:]
    classes_tr = set(np.unique(y[tr_idx]))
    classes_te = set(np.unique(y[te_idx]))
    if len(classes_tr) < 2 or not classes_te.issubset(classes_tr):
        return None
    K_eff = len(classes_tr)
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
    knn.fit(X[tr_idx], y[tr_idx])
    acc = float(knn.score(X[te_idx], y[te_idx]))
    return (acc - 1.0/K_eff) / (1.0 - 1.0/K_eff)


def main():
    print("=" * 65)
    print("CTI Audio/Speech Modality Test")
    print("=" * 65)
    print(f"Reference: NLP alpha_NLP = {ALPHA_NLP} (CV={ALPHA_NLP_CV})")
    print(f"Hypothesis: functional form holds for SPEECH embeddings")
    print()

    # Load dataset
    print("Loading MINDS14 dataset...")
    audio_arrays, labels, sr = load_minds14()
    K = len(np.unique(labels))
    print(f"  K={K}, matching DBpedia K=14")
    print()

    all_points = []

    for model_name in AUDIO_MODELS:
        model_short = model_name.split("/")[-1]
        print(f"Model: {model_short}")
        print("-" * 40)

        try:
            layer_embs, layer_indices = get_audio_embeddings_at_layers(
                model_name, audio_arrays, sr)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        for li in layer_indices:
            embs = layer_embs.get(li)
            if embs is None:
                continue
            kappa = compute_kappa_nearest(embs, labels)
            q = compute_knn_q(embs, labels)
            if kappa is None or q is None:
                continue
            all_points.append({
                "model": model_short,
                "layer": li,
                "kappa": float(kappa),
                "q": float(q),
                "K": K,
            })
            print(f"  Layer {li:2d}: kappa={kappa:.4f}, q={q:.4f}")
        print()

    # FIT
    print("=" * 65)
    print("RESULTS: Fit logit(q) = A_audio * kappa + C")
    print("=" * 65)

    if len(all_points) < 3:
        print(f"INSUFFICIENT DATA (n={len(all_points)})")
        return

    kappas = np.array([p["kappa"] for p in all_points])
    qs = np.clip(np.array([p["q"] for p in all_points]), 0.01, 0.99)
    logit_qs = np.array([sp_logit(float(q)) for q in qs])

    valid = np.isfinite(kappas) & np.isfinite(logit_qs) & (kappas > 0)
    kappas, logit_qs = kappas[valid], logit_qs[valid]

    if len(kappas) < 3:
        print("INSUFFICIENT VALID POINTS")
        return

    coeffs = np.polyfit(kappas, logit_qs, deg=1)
    A_audio = float(coeffs[0])
    C = float(coeffs[1])
    r, p = pearsonr(kappas, logit_qs)

    print(f"  A_audio = {A_audio:.4f}")
    print(f"  C = {C:.4f}")
    print(f"  Pearson r = {r:.4f} (p={p:.4f}, n={len(kappas)})")
    print()
    print(f"  Alpha comparison:")
    print(f"    NLP (text):  alpha = {ALPHA_NLP} (CV={ALPHA_NLP_CV})")
    print(f"    Audio:       A = {A_audio:.4f}")
    print(f"    Ratio audio/NLP = {A_audio/ALPHA_NLP:.3f}")
    print()

    form_holds = r > 0.70
    print(f"  Functional form (logit-linear in kappa): {'CONFIRMED' if form_holds else 'FAILS'} (r={r:.4f}, threshold 0.70)")

    # Save results
    out = {
        "experiment": "audio_modality_cti_law",
        "dataset": "MINDS14_en-US",
        "K": K,
        "n_samples": len(audio_arrays),
        "models": AUDIO_MODELS,
        "A_audio": A_audio,
        "C": C,
        "r": float(r),
        "p": float(p),
        "n_points": len(kappas),
        "alpha_nlp": ALPHA_NLP,
        "alpha_ratio": A_audio / ALPHA_NLP,
        "functional_form_r_threshold": 0.70,
        "functional_form_holds": bool(form_holds),
        "points": all_points,
        "summary": (
            f"Audio (wav2vec2/HuBERT) on MINDS14 (K=14): A_audio={A_audio:.3f}, "
            f"r={r:.3f}. {'Functional form CONFIRMED' if form_holds else 'Functional form FAILS'}. "
            f"NLP alpha_NLP={ALPHA_NLP} for reference."
        ),
    }

    out_path = RESULTS_DIR / "cti_audio_modality.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path.name}")


if __name__ == "__main__":
    main()
