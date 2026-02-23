"""
LOAO analysis only - all SmolLM2 caches already exist.
Loads kappa_near_cache_{dataset}_SmolLM2-1.7B.json for 3 datasets,
plus all existing architecture caches, does LOAO analysis.
"""
import json
import os
import glob
import numpy as np
from scipy.stats import pearsonr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "cti_smollm2_loao_replication.json")

MODEL_SHORT = "SmolLM2-1.7B"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
ALPHA_LOW = 2.43
ALPHA_HIGH = 3.29
DATASETS_ALL = ["dbpedia", "agnews", "20newsgroups"]

all_points = []
for ds_name in DATASETS_ALL:
    pattern = os.path.join(RESULTS_DIR, f"kappa_near_cache_{ds_name}_*.json")
    for cache_file in glob.glob(pattern):
        with open(cache_file) as f:
            pts = json.load(f)
        for pt in pts:
            if "logKm1" not in pt and "K" in pt:
                pt["logKm1"] = float(np.log(pt["K"] - 1))
        all_points.extend(pts)

print(f"Total points: {len(all_points)}")
models = sorted(set(p["model"] for p in all_points))
print(f"Architectures ({len(models)}): {models}")

train_pts = [p for p in all_points if p["model"] != MODEL_SHORT]
test_pts = [p for p in all_points if p["model"] == MODEL_SHORT]
print(f"\nTrain: {len(train_pts)} pts ({len(set(p['model'] for p in train_pts))} archs)")
print(f"Test (SmolLM2): {len(test_pts)} pts")

# Fit on train
kappa_tr = np.array([p["kappa_nearest"] for p in train_pts])
logKm1_tr = np.array([p["logKm1"] for p in train_pts])
logit_tr = np.array([p["logit_q"] for p in train_pts])
X_tr = np.column_stack([kappa_tr, logKm1_tr, np.ones(len(train_pts))])
coeffs, _, _, _ = np.linalg.lstsq(X_tr, logit_tr, rcond=None)
alpha_loao, beta_loao, C0_loao = coeffs
print(f"\nTrain fit: alpha={alpha_loao:.4f}, beta={beta_loao:.4f}, C={C0_loao:.4f}")

# Fit SmolLM2 independently
kappa_te = np.array([p["kappa_nearest"] for p in test_pts])
logKm1_te = np.array([p["logKm1"] for p in test_pts])
logit_te = np.array([p["logit_q"] for p in test_pts])
X_te = np.column_stack([kappa_te, logKm1_te, np.ones(len(test_pts))])
coeffs_smol, _, _, _ = np.linalg.lstsq(X_te, logit_te, rcond=None)
alpha_smol, beta_smol, C_smol = coeffs_smol
print(f"\nSmolLM2 fit: alpha={alpha_smol:.4f}, beta={beta_smol:.4f}, C={C_smol:.4f}")

# Prediction r using train-fit coefficients on SmolLM2
logit_pred_frozen = X_te @ coeffs
r_pred, p_pred = pearsonr(logit_te, logit_pred_frozen)
mae_frozen = float(np.mean(np.abs(logit_te - logit_pred_frozen)))

pr1 = ALPHA_LOW <= alpha_smol <= ALPHA_HIGH
pr2 = r_pred >= 0.80

print(f"\nPR1 (SmolLM2 alpha in [{ALPHA_LOW},{ALPHA_HIGH}]): {'PASS' if pr1 else 'FAIL'}")
print(f"     alpha_smol={alpha_smol:.4f}")
print(f"PR2 (r(pred,obs) >= 0.80): {'PASS' if pr2 else 'FAIL'}")
print(f"     r={r_pred:.4f}, p={p_pred:.4f}, MAE={mae_frozen:.4f}")
print(f"\nOVERALL: {'PASS' if (pr1 and pr2) else 'FAIL'}")

print(f"\nSmolLM2 data points:")
for p in sorted(test_pts, key=lambda x: (x['dataset'], x['layer'])):
    pred = coeffs[0]*p['kappa_nearest'] + coeffs[1]*p['logKm1'] + coeffs[2]
    print(f"  {p['dataset']} L{p['layer']}: kappa={p['kappa_nearest']:.4f}, "
          f"q={p['q']:.3f}, logit={p['logit_q']:.3f}, pred={pred:.3f}")

output = {
    "experiment": "smollm2_loao_replication",
    "model": MODEL_NAME,
    "design": "LOAO-equivalent: mean-pool, logit(q_raw), 4 proportional layers (6,12,18,23)",
    "convention": "logit(q_raw) -- matches original kappa_near_cache format",
    "pre_reg_alpha_interval": [ALPHA_LOW, ALPHA_HIGH],
    "datasets_used": DATASETS_ALL,
    "layers_used": [6, 12, 18, 23],
    "n_train_points": len(train_pts),
    "n_train_architectures": len(set(p["model"] for p in train_pts)),
    "n_smollm2_points": len(test_pts),
    "smollm2_points": test_pts,
    "train_fit": {"alpha": float(alpha_loao), "beta": float(beta_loao), "C0": float(C0_loao)},
    "smollm2_fit": {"alpha": float(alpha_smol), "beta": float(beta_smol), "C": float(C_smol)},
    "evaluation": {
        "mae_frozen": mae_frozen,
        "pearson_r_pred": float(r_pred),
        "pearson_p_pred": float(p_pred),
    },
    "pr1_alpha_pass": bool(pr1),
    "pr2_r_pass": bool(pr2),
    "overall_pass": bool(pr1 and pr2),
    "verdict": "PASS" if (pr1 and pr2) else "FAIL",
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}")
