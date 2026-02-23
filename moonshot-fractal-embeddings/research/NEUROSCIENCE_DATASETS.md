# Neuroscience Datasets for CTI Law Biological Validation

**Date**: February 23, 2026
**Strategic purpose**: Test CTI law `logit(q) = A * kappa_nearest * sqrt(d_eff) + C` on biological
neural population recordings. If A_bio ≈ A_artificial (~1.05), law is substrate-independent.
This is the path to Nobel Physics/Medicine (current score: Physics 4/10, needs multi-domain replication).

---

## Tier 1: Recommended Datasets

### 1. MajajHong2015 — Macaque IT Cortex (START HERE)
- 256 IT cortex multi-units, macaque, awake passive fixation
- K=8 categories: Cars, Faces, Fruits, Bodies, Animals, Planes, Chairs, Tables
- 148,480 stimulus presentations, 47 repetitions
- Format: BrainIO/xarray, ~400MB download
- Install: `pip install git+https://github.com/brain-score/brainio.git`
- Load: `from brainio.fetch import get_assembly; assembly = get_assembly("dicarlo.MajajHong2015.public")`
- Paper: Majaj et al. 2015, J. Neuroscience https://pubmed.ncbi.nlm.nih.gov/26424887/
- Sub-regions available: aIT, pIT

### 2. Cadieu/DiCarlo 2014 — Macaque IT + V4 (SECOND)
- 168 IT multi-unit sites, 64 objects, K=8 categories
- Direct download: `https://s3.amazonaws.com/cadieu-etal-ploscb2014/PLoSCB2014_data_20141216.zip`
- Format: .mat files, ~400MB
- GitHub: https://github.com/dicarlolab/Cadieu_etal_PLoSCB_2014
- V4 comparison available (same data, different area)

### 3. Stringer/Pachitariu 2018b — Mouse V1, 10K Neurons (STRESS TEST)
- ~10,000 V1 neurons, calcium imaging
- 2,800 natural images, K=15 categories
- Figshare: https://janelia.figshare.com/articles/dataset/.../6845348
- Size: 6.71 GB total (multiple sessions)
- Note: V1 encodes edges not objects → kappa likely low → tests whether law predicts near-chance

### 4. Allen Brain Observatory Neuropixels — Mouse Visual Cortex
- 500-1500 units across 6 visual areas (VISp, VISl, VISam, VISrl, VISpm, VISal)
- 118 natural scene images (K=118 images as "categories", 50 trials each)
- AllenSDK: `from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache`
- Docs: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
- Size: 1.7-3.3 GB per session, 58 sessions total

---

## CTI Law Computation for Neural Data

```python
import numpy as np
from scipy.spatial.distance import cdist

def compute_cti_inputs(responses, labels):
    """
    responses: (N_stimuli, N_neurons) — mean firing rates
    labels: (N_stimuli,) — category labels
    Returns: kappa_nearest, d_eff per class
    """
    classes = np.unique(labels)
    K = len(classes)
    centroids = np.array([responses[labels == c].mean(axis=0) for c in classes])
    W_list = [np.cov(responses[labels == c].T) for c in classes]
    Sigma_W = np.mean(W_list, axis=0)
    tr_W = np.trace(Sigma_W)
    d = responses.shape[1]
    sigma_W_global = np.sqrt(tr_W / d)

    centroid_dists = cdist(centroids, centroids)
    np.fill_diagonal(centroid_dists, np.inf)
    nearest_j = np.argmin(centroid_dists, axis=1)

    kappas, d_effs = [], []
    for i, c in enumerate(classes):
        j = nearest_j[i]
        delta = centroids[i] - centroids[j]
        delta_norm = delta / np.linalg.norm(delta)
        delta_min = np.linalg.norm(delta)
        kappa_i = delta_min / (sigma_W_global * np.sqrt(d))
        sigma_cdir = np.sqrt(delta_norm @ Sigma_W @ delta_norm)
        d_eff_i = tr_W / (sigma_cdir ** 2)
        kappas.append(kappa_i)
        d_effs.append(d_eff_i)

    return np.array(kappas), np.array(d_effs), K
```

---

## Key Hypothesis

**If A_bio ≈ A_artificial ≈ 1.05**: The CTI law is substrate-independent — holds for both
biological and artificial neural systems. This is the strongest possible universality claim
and the direct path to Nobel Physics/Medicine score increase (from ~4/10 to ~7+/10).

**Prediction**: 
- IT cortex (high categorical representation): A_bio close to A_NLP range
- V1 (edge-based): A_bio may differ (different d_eff regime)
- Cross-area comparison: A should correlate with categorical coding strength

---

## Next Steps

1. Download MajajHong2015 (~400MB) 
2. Run CTI law on it: compute kappa_nearest, q (1-NN on held-out trials)
3. Compare A_bio vs A_NLP=1.477 (canonical)
4. Pre-register prediction BEFORE running: predict A_bio range from theory
