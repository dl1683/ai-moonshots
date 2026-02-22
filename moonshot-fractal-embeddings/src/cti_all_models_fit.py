"""
Comprehensive CTI law fit across ALL models with existing kappa caches.
Tests encoder-only (BERT, DeBERTa, Electra), SSM (Mamba), contrastive (BGE),
and decoder-only (GPT-2, phi2) in addition to the 12 LOAO models.
"""
import json
import os
import numpy as np
from scipy.stats import pearsonr
from numpy.linalg import lstsq

DATASETS = ['agnews', 'dbpedia', '20newsgroups', 'go_emotions']
DATASET_IDX = {d: i for i, d in enumerate(DATASETS)}

# All models that have caches
MODELS = [
    ('bert-base-uncased', 'Encoder-only (MLM)', 'bert-base-uncased'),
    ('bge-base-v1-5', 'Contrastive (InfoNCE)', 'bge-base-v1-5'),
    ('deberta-base', 'Encoder-only (replaced-token)', 'deberta-base'),
    ('electra-small', 'Encoder-only (discriminator)', 'electra-small'),
    ('Falcon-H1-0.5B-Base', 'Transformer+SSM hybrid', 'Falcon-H1-0.5B-Base'),
    ('gpt2', 'Decoder-only (small)', 'gpt2'),
    ('gpt-neo-125m', 'Decoder-only (2021)', 'gpt-neo-125m'),
    ('mamba-130m', 'Pure SSM (no attention)', 'mamba-130m'),
    ('Mistral-7B-v0.3', 'Decoder-only (2023)', 'Mistral-7B-v0.3'),
    ('OLMo-1B-hf', 'Decoder-only (2024)', 'OLMo-1B-hf'),
    ('phi2', 'Decoder-only (compact)', 'phi2'),
    ('pythia-160m', 'Decoder-only (2021)', 'pythia-160m'),
    ('pythia-1b', 'Decoder-only (2021)', 'pythia-1b'),
    ('pythia-410m', 'Decoder-only (2021)', 'pythia-410m'),
    ('Qwen2.5-0.5B', 'Decoder-only (2024)', 'Qwen2.5-0.5B'),
    ('Qwen3-0.6B', 'Decoder-only (2025)', 'Qwen3-0.6B'),
    ('Qwen3-1.7B', 'Decoder-only (2025)', 'Qwen3-1.7B'),
    ('rwkv-4-169m-pile', 'Pure linear RNN (no attention)', 'rwkv-4-169m-pile'),
    ('TinyLlama-1.1B-intermediate-step-1431k-3T', 'Decoder-only (2024)', 'TinyLlama-1.1B-intermediate-step-1431k-3T'),
]

def load_model_data(model_key):
    """Load all dataset caches for a model."""
    rows = []
    for ds in DATASETS:
        fpath = f'results/kappa_near_cache_{ds}_{model_key}.json'
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            for r in data:
                r['dataset_name'] = ds
                rows.append(r)
    return rows

def fit_cti_law(rows):
    """Fit logit(q_norm) = alpha * kappa - beta * log(K-1) + C0_ds using OLS."""
    if len(rows) < 8:
        return None
    kappas = np.array([r['kappa_nearest'] for r in rows])
    logit_qs = np.array([r['logit_q'] for r in rows])
    Ks = np.array([r['K'] for r in rows])
    logKm1 = np.log(Ks - 1)
    ds_names = [r['dataset_name'] for r in rows]

    # Which datasets are present?
    present_ds = sorted(set(ds_names))
    ds_to_col = {d: i for i, d in enumerate(present_ds)}
    n_ds = len(present_ds)

    # Design matrix: [kappa, log(K-1), d0, d1, ...]
    N = len(rows)
    X = np.zeros((N, 2 + n_ds))
    X[:, 0] = kappas
    X[:, 1] = logKm1
    for i, r in enumerate(rows):
        X[i, 2 + ds_to_col[r['dataset_name']]] = 1

    coefs, _, _, _ = lstsq(X, logit_qs, rcond=None)
    alpha, beta = coefs[0], coefs[1]
    C0s = coefs[2:]

    pred = X @ coefs
    ss_res = np.sum((logit_qs - pred)**2)
    ss_tot = np.sum((logit_qs - logit_qs.mean())**2)
    r2 = 1 - ss_res / ss_tot

    # Overall Pearson r between kappa and ds-adjusted logit
    adj_logit = logit_qs - logKm1 * beta
    for i, r_row in enumerate(rows):
        adj_logit[i] -= C0s[ds_to_col[r_row['dataset_name']]]
    r_pearson, p_pearson = pearsonr(kappas, adj_logit)

    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'r2': float(r2),
        'r_pearson': float(r_pearson),
        'p_pearson': float(p_pearson),
        'n': N,
        'n_datasets': n_ds,
        'C0': {d: float(C0s[i]) for i, d in enumerate(present_ds)},
    }

# Fit all models
print(f"{'Model':<50} {'Type':<35} {'alpha':>8} {'r':>8} {'R^2':>8} {'N':>5}")
print("-" * 120)

results = []
for model_key, model_type, cache_key in MODELS:
    rows = load_model_data(cache_key)
    if len(rows) == 0:
        print(f"{model_key:<50} {'No data':35}")
        continue

    fit = fit_cti_law(rows)
    if fit is None:
        print(f"{model_key:<50} {'Too few points':35}")
        continue

    print(f"{model_key:<50} {model_type:<35} {fit['alpha']:>8.4f} {fit['r_pearson']:>8.4f} {fit['r2']:>8.4f} {fit['n']:>5}")
    results.append({
        'model': model_key,
        'type': model_type,
        **fit,
    })

# Analysis
print("\n\n=== Analysis by Architecture Family ===")
generative = [r for r in results if 'Decoder' in r['type']]
encoder = [r for r in results if 'Encoder' in r['type']]
ssm = [r for r in results if 'SSM' in r['type'] or 'RNN' in r['type']]
contrastive = [r for r in results if 'Contrastive' in r['type']]
hybrid = [r for r in results if 'hybrid' in r['type']]

for family, members in [('Generative Decoder', generative), ('Encoder-only', encoder),
                         ('SSM/RNN', ssm), ('Contrastive', contrastive), ('Hybrid', hybrid)]:
    if members:
        alphas = [m['alpha'] for m in members]
        rs = [m['r_pearson'] for m in members]
        print(f"\n{family} ({len(members)} models):")
        print(f"  alpha: mean={np.mean(alphas):.4f} +/- {np.std(alphas):.4f}, CV={np.std(alphas)/np.mean(alphas):.4f}")
        print(f"  r:     mean={np.mean(rs):.4f} +/- {np.std(rs):.4f}")

# Alpha universality across ALL models
all_alphas = [r['alpha'] for r in results]
print(f"\n=== GLOBAL SUMMARY ===")
print(f"All {len(results)} models: alpha mean={np.mean(all_alphas):.4f}, std={np.std(all_alphas):.4f}, CV={np.std(all_alphas)/np.mean(all_alphas):.4f}")
print(f"NLP-only (12 arch): alpha=1.477, CV=0.023")
print(f"All types: alpha CV={np.std(all_alphas)/np.mean(all_alphas):.4f}")

# Save
out = {
    'experiment': 'cti_all_models_fit',
    'description': 'CTI law fit for all cached models (encoder, decoder, SSM, contrastive)',
    'n_models': len(results),
    'global_alpha_mean': float(np.mean(all_alphas)),
    'global_alpha_std': float(np.std(all_alphas)),
    'global_alpha_cv': float(np.std(all_alphas)/np.mean(all_alphas)),
    'models': results,
}
with open('results/cti_all_models_fit.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/cti_all_models_fit.json")
