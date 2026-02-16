#!/usr/bin/env python
"""
parse_week2_output.py

Parse the Week 2 control study output from the log file to reconstruct
the results JSON, avoiding a 2+ hour re-run.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_log(log_path):
    """Parse the experiment log to extract all condition results."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    conditions = []
    # Pattern for condition header
    cond_pattern = re.compile(
        r'\[(\d+)/49\] obj=(\w+) lam_sep=([\d.]+) lam_uni=([\d.]+) seed=(\d+)'
    )
    # Pattern for baseline
    baseline_pattern = re.compile(
        r'\[(\d+)/49\] BASELINE seed=(\d+)'
    )
    # Pattern for metric line
    metric_pattern = re.compile(
        r'\s+(\w+) L(\d+): knn_l0=([\d.]+) knn_l1=([\d.]+) sep_l1=([\d.]+) margin=([-\d.]+)'
    )

    lines = text.split('\n')
    current_cond = None
    current_metrics = {}

    for line in lines:
        # Check for baseline header
        bm = baseline_pattern.search(line)
        if bm:
            # Save previous condition
            if current_cond and current_metrics:
                current_cond["metrics"] = current_metrics
                conditions.append(current_cond)

            idx, seed = bm.groups()
            current_cond = {
                "condition_idx": int(idx),
                "objective": "baseline",
                "lambda_sep": 0.0,
                "lambda_uni": 0.0,
                "seed": int(seed),
            }
            current_metrics = {}
            continue

        # Check for condition header
        cm = cond_pattern.search(line)
        if cm:
            # Save previous condition
            if current_cond and current_metrics:
                current_cond["metrics"] = current_metrics
                conditions.append(current_cond)

            idx, obj, lam_sep, lam_uni, seed = cm.groups()
            current_cond = {
                "condition_idx": int(idx),
                "objective": obj,
                "lambda_sep": float(lam_sep),
                "lambda_uni": float(lam_uni),
                "seed": int(seed),
            }
            current_metrics = {}
            continue

        # Check for metric line
        mm = metric_pattern.search(line)
        if mm and current_cond:
            ds_name, layer, knn_l0, knn_l1, sep_l1, margin = mm.groups()
            if ds_name not in current_metrics:
                current_metrics[ds_name] = {}
            current_metrics[ds_name][int(layer)] = {
                "knn_l0": float(knn_l0),
                "knn_l1": float(knn_l1),
                "class_sep_l1": float(sep_l1),
                "mean_local_margin": float(margin),
            }

    # Save last condition
    if current_cond and current_metrics:
        current_cond["metrics"] = current_metrics
        conditions.append(current_cond)

    return conditions


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else (
        r"C:\Users\devan\AppData\Local\Temp\claude\C--Users-devan-OneDrive-Desktop-Projects-AI-Moonshots-moonshot-fractal-embeddings\tasks\bc0eeaa.output"
    )

    print(f"Parsing: {log_path}")
    conditions = parse_log(log_path)
    print(f"Parsed {len(conditions)} conditions")

    if not conditions:
        print("ERROR: No conditions found!")
        sys.exit(1)

    # Verify we have all 49+baseline conditions
    cond_indices = [c["condition_idx"] for c in conditions]
    print(f"Condition indices: {sorted(set(cond_indices))}")
    print(f"Unique conditions: {len(set(cond_indices))}")

    # Build output
    output = {
        "experiment": "CGP Week 2: Orthogonal Control Study",
        "model": "pythia-160m",
        "design": {
            "objectives": ["contrastive", "lm"],
            "lambda_sep_values": [0.0, 0.1, 0.3, 1.0],
            "lambda_uni_values": [0.0, 0.3],
            "seeds": [42, 123, 456],
            "steps": 500,
            "eval_datasets": ["clinc", "dbpedia_classes"],
        },
        "conditions": conditions,
    }

    out_path = RESULTS_DIR / "cgp_week2_control_study.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    print(f"Saved to {out_path}")

    # Quick sanity check: print dose-response for contrastive
    print("\n--- Dose-response (contrastive, lambda_uni=0.0) ---")
    for lam in [0.0, 0.1, 0.3, 1.0]:
        vals = []
        for c in conditions:
            if c["objective"] == "contrastive" and c["lambda_sep"] == lam and c["lambda_uni"] == 0.0:
                for ds_name, ds_metrics in c.get("metrics", {}).items():
                    if ds_name == "clinc":
                        for layer_key, m in ds_metrics.items():
                            vals.append(m["knn_l1"])
        if vals:
            print(f"  lambda_sep={lam}: CLINC knn_l1 mean={sum(vals)/len(vals):.3f} ({vals})")


if __name__ == "__main__":
    main()
