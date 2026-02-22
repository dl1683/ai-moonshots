"""
CTI Universal Law: Quick fit for BGE-base-v1.5 (contrastive/retrieval model).
BGE is trained with InfoNCE objective (discriminative) vs generative LLMs.
Tests whether the law holds for non-autoregressive embedding models.
Uses pre-existing kappa_near_cache files.
"""
import json
import numpy as np
from scipy.stats import linregress, pearsonr
from scipy.special import logit

DATASETS = ['agnews', 'dbpedia', '20newsgroups', 'go_emotions']
MODEL = 'bge-base-v1-5'

# Load all cached data
all_data = []
for ds in DATASETS:
    fpath = f'results/kappa_near_cache_{ds}_{MODEL}.json'
    try:
        with open(fpath) as f:
            rows = json.load(f)
        for r in rows:
            r['dataset'] = ds
        all_data.extend(rows)
    except FileNotFoundError:
        print(f"  Missing: {fpath}")

print(f"Loaded {len(all_data)} data points for BGE-base-v1.5")
for d in all_data:
    print(f"  {d['dataset']:20s} layer={d['layer']:2d} K={d['K']:2d} "
          f"kappa={d['kappa_nearest']:.4f} q={d['q']:.4f} logit_q={d['logit_q']:.4f}")

# Fit CTI law: logit(q_norm) = alpha * kappa + beta * log(K-1) + C
# q_norm = (q - 1/K) / (1 - 1/K)
kappas = np.array([d['kappa_nearest'] for d in all_data])
logit_qs = np.array([d['logit_q'] for d in all_data])
Ks = np.array([d['K'] for d in all_data])
logKm1 = np.log(Ks - 1)
datasets_list = [d['dataset'] for d in all_data]

# Per-dataset intercept model
DATASET_MAP = {'agnews': 0, 'dbpedia': 1, '20newsgroups': 2, 'go_emotions': 3}
from numpy.linalg import lstsq

# Build design matrix: [kappa, log(K-1), d_agnews, d_dbpedia, d_20news, d_goemo]
N = len(all_data)
X_design = np.zeros((N, 2 + 4))
X_design[:, 0] = kappas
X_design[:, 1] = logKm1
for i, d in enumerate(all_data):
    X_design[i, 2 + DATASET_MAP[d['dataset']]] = 1

coefs, residuals, rank, sv = lstsq(X_design, logit_qs, rcond=None)
alpha = coefs[0]
beta = coefs[1]
C0_per_dataset = coefs[2:]

pred = X_design @ coefs
r2 = 1 - np.sum((logit_qs - pred)**2) / np.sum((logit_qs - logit_qs.mean())**2)
r_pearson, _ = pearsonr(kappas, logit_qs - logKm1 * beta - X_design[:, 2:] @ C0_per_dataset)

print(f"\n=== BGE-base-v1.5 CTI Law Fit (per-dataset intercepts) ===")
print(f"alpha = {alpha:.4f} (NLP reference: 1.477)")
print(f"beta  = {beta:.4f} (NLP reference: 0.326)")
print(f"R^2   = {r2:.4f}")
print(f"C0 per dataset: agnews={C0_per_dataset[0]:.3f}, dbpedia={C0_per_dataset[1]:.3f}, "
      f"20news={C0_per_dataset[2]:.3f}, go_emotions={C0_per_dataset[3]:.3f}")

# Pearson r (kappa vs adjusted logit)
adjusted_logit = logit_qs - logKm1 * beta
for ds_idx, ds_name in enumerate(['agnews', 'dbpedia', '20newsgroups', 'go_emotions']):
    mask = np.array([d['dataset'] == ds_name for d in all_data])
    r_ds, p_ds = pearsonr(kappas[mask], adjusted_logit[mask] - C0_per_dataset[ds_idx])
    print(f"  r({ds_name}) = {r_ds:.4f} (p={p_ds:.4f})")

# Compare to NLP alpha
print(f"\nBGE alpha = {alpha:.4f} vs NLP alpha = 1.477")
print(f"Ratio = {alpha/1.477:.4f}")
print(f"Theory: alpha = sqrt(4/pi) * sqrt(d_eff) => d_eff = (alpha/1.128)^2 = {(alpha/1.128)**2:.4f}")
print(f"BGE model_type: Contrastive fine-tuned (InfoNCE) - discriminative objective")

# Save
out = {
    'experiment': 'cti_bge_fit',
    'description': 'CTI law fit for BGE-base-v1.5 (contrastive embedding model)',
    'model': 'bge-base-v1-5',
    'model_type': 'Contrastive/InfoNCE (discriminative embedding)',
    'n_points': N,
    'alpha': float(alpha),
    'beta': float(beta),
    'r2': float(r2),
    'C0_per_dataset': {
        'agnews': float(C0_per_dataset[0]),
        'dbpedia': float(C0_per_dataset[1]),
        '20newsgroups': float(C0_per_dataset[2]),
        'go_emotions': float(C0_per_dataset[3]),
    },
    'reference': {'NLP_alpha': 1.477, 'NLP_r2': 0.955},
}
with open('results/cti_bge_fit.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/cti_bge_fit.json")
