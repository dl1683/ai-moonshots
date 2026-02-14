#!/usr/bin/env python
"""
cti_pilot.py

Compute Thermodynamics of Intelligence (CTI) pilot experiment.

Protocol:
- Models: bge-base, e5-base, bge-small
- Fit datasets: dbpedia_classes, clinc
- Holdout dataset: agnews
- Core measured layers: 2,4,6,8,10,12
- Prospective layers: 3,9
- Seeds: 42,123,456

Experiment:
1) Train V5 head-only at each chosen backbone layer (backbone frozen)
2) Measure steerability S=(L0@j1-L0@j4)+(L1@j4-L1@j1)
3) Compute compute proxy C(L)=L/12 and distortion D(L)=1-S(L)/S(12)
4) Fit D(C)=D_inf + k*C^{-alpha}
5) Compare universality models:
   - shared alpha across models
   - per-model alpha
   using AIC.

Resume support:
- State is saved incrementally after each seed/combo.
- Frozen prospective predictions are generated before any target runs
  (L={3,9} and holdout AGNews).
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit, least_squares

REPO_ROOT = Path(__file__).resolve().parent.parent  # src/ -> repo root
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS

SCRIPT_VERSION = "1.0.0"
EPS = 1e-12
EXPECTED_MODEL_DIMS = {"bge-base": 768, "e5-base": 768, "bge-small": 384}


@dataclass(frozen=True)
class RunSpec:
    model: str
    dataset: str
    layer: int
    seed: int

    def key(self) -> str:
        return f"{self.model}|{self.dataset}|L{self.layer}|seed{self.seed}"


class TempDataset:
    def __init__(self, samples, level0_names, level1_names):
        self.samples = samples
        self.level0_names = level0_names
        self.level1_names = level1_names


class DatasetCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, str, Optional[int]], Any] = {}

    def get(self, name: str, split: str, max_samples: Optional[int]) -> Any:
        key = (name, split, max_samples)
        if key not in self._cache:
            log(f"Loading dataset {name} split={split} max_samples={max_samples}")
            self._cache[key] = load_hierarchical_dataset(name, split=split, max_samples=max_samples)
        return self._cache[key]


class LayerwiseFractalModel(FractalModelV5):
    """
    Same V5 architecture, but reads backbone hidden_states[layer_idx] instead of last layer.
    """

    def __init__(self, target_layer: int, *args, **kwargs):
        self.target_layer = int(target_layer)
        super().__init__(*args, **kwargs)
        self.total_backbone_layers = self._infer_total_layers()
        if self.target_layer < 1 or self.target_layer > self.total_backbone_layers:
            raise ValueError(
                f"Invalid target_layer={self.target_layer}; model has {self.total_backbone_layers} layers"
            )

    def _infer_total_layers(self) -> int:
        cfg = getattr(self.backbone, "config", None)
        if cfg is not None and getattr(cfg, "num_hidden_layers", None) is not None:
            return int(cfg.num_hidden_layers)
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return int(len(self.backbone.encoder.layer))
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
            return int(len(self.backbone.model.layers))
        if hasattr(self.backbone, "layers"):
            return int(len(self.backbone.layers))
        raise RuntimeError("Could not infer number of backbone layers")

    def forward(self, input_ids, attention_mask, block_dropout_mask: Optional[torch.Tensor] = None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Backbone did not return hidden_states; required for layerwise CTI experiment")
        if self.target_layer >= len(hidden_states):
            raise RuntimeError(
                f"target_layer={self.target_layer} out of range for hidden_states len={len(hidden_states)}"
            )
        pooled = self.pool(hidden_states[self.target_layer], attention_mask).float()
        return self.fractal_head(pooled, block_dropout_mask)


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_str(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_int(value: str) -> List[int]:
    return [int(x) for x in parse_csv_str(value)]


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def atomic_json_dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=False)
    tmp.replace(path)


def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def entropy_from_labels(labels: Sequence[int]) -> float:
    arr = np.asarray(labels, dtype=np.int64)
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr)
    probs = counts[counts > 0].astype(np.float64) / float(arr.size)
    return float(-np.sum(probs * np.log2(probs + EPS)))


def conditional_entropy(fine_labels: Sequence[int], coarse_labels: Sequence[int]) -> float:
    fine = np.asarray(fine_labels, dtype=np.int64)
    coarse = np.asarray(coarse_labels, dtype=np.int64)
    if fine.size == 0 or coarse.size == 0:
        return 0.0
    total = float(len(fine))
    h = 0.0
    for c in np.unique(coarse):
        idx = coarse == c
        p_c = float(np.sum(idx)) / total
        h += p_c * entropy_from_labels(fine[idx])
    return float(h)


def dataset_profile(dataset_name: str, dataset_obj: Any) -> Dict[str, Any]:
    l0 = [s.level0_label for s in dataset_obj.samples]
    l1 = [s.level1_label for s in dataset_obj.samples]
    return {
        "dataset": dataset_name,
        "n_samples": int(len(dataset_obj.samples)),
        "n_l0": int(len(set(l0))),
        "n_l1": int(len(set(l1))),
        "h_l1_given_l0": float(conditional_entropy(l1, l0)),
    }


def get_run(state: Dict[str, Any], spec: RunSpec) -> Optional[Dict[str, Any]]:
    return (
        state.get("results", {})
        .get(spec.model, {})
        .get(spec.dataset, {})
        .get(str(spec.layer), {})
        .get(str(spec.seed))
    )


def set_run(state: Dict[str, Any], spec: RunSpec, payload: Dict[str, Any]) -> None:
    state.setdefault("results", {}).setdefault(spec.model, {}).setdefault(spec.dataset, {}).setdefault(
        str(spec.layer), {}
    )[str(spec.seed)] = payload


def run_ok(payload: Optional[Dict[str, Any]]) -> bool:
    return isinstance(payload, dict) and payload.get("status") == "ok"


def parse_iso_datetime(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def config_compatible(existing: Dict[str, Any], current: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    keys = [
        "models",
        "fit_datasets",
        "holdout_datasets",
        "fit_layers",
        "prospective_layers",
        "seeds",
        "total_layers",
        "max_train_samples",
        "max_test_samples",
    ]
    for k in keys:
        if existing.get(k) != current.get(k):
            return False, k
    return True, None


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, EPS, None)
    return x / norms


def fast_knn_accuracy(emb: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    k = int(max(1, min(k, n - 1)))

    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)

    if k < n - 1:
        nn_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    else:
        nn_idx = np.argsort(-sim, axis=1)[:, :k]

    nn_labels = labels[nn_idx]
    correct = 0
    for i in range(n):
        vals, counts = np.unique(nn_labels[i], return_counts=True)
        pred = vals[np.argmax(counts)]
        if pred == labels[i]:
            correct += 1
    return float(correct / n)


def evaluate_prefix_accuracy_fast(
    model: LayerwiseFractalModel,
    samples: Sequence[Any],
    max_eval_samples: int = 500,
    batch_size: int = 64,
    k: int = 5,
) -> Dict[str, float]:
    n = min(len(samples), int(max_eval_samples))
    eval_samples = list(samples[:n])
    if len(eval_samples) < k + 1:
        raise ValueError(f"Need at least {k + 1} eval samples, got {len(eval_samples)}")

    texts = [s.text for s in eval_samples]
    l0 = np.asarray([s.level0_label for s in eval_samples], dtype=np.int64)
    l1 = np.asarray([s.level1_label for s in eval_samples], dtype=np.int64)

    results: Dict[str, float] = {}
    for j in (1, 2, 3, 4):
        prefix_len = j if j < 4 else None
        with torch.no_grad():
            emb = model.encode(texts, batch_size=batch_size, prefix_len=prefix_len).cpu().numpy()
        emb = l2_normalize(emb)
        results[f"j{j}_l0"] = fast_knn_accuracy(emb, l0, k=k)
        results[f"j{j}_l1"] = fast_knn_accuracy(emb, l1, k=k)
    return results


def steerability_from_prefix(prefix_accuracy: Dict[str, float]) -> float:
    return float(
        (prefix_accuracy.get("j1_l0", 0.0) - prefix_accuracy.get("j4_l0", 0.0))
        + (prefix_accuracy.get("j4_l1", 0.0) - prefix_accuracy.get("j1_l1", 0.0))
    )


def run_single_config(spec: RunSpec, cfg: Dict[str, Any], dataset_cache: DatasetCache) -> Dict[str, Any]:
    t0 = time.monotonic()
    model = None
    trainer = None
    try:
        set_seed(spec.seed)

        train_data = dataset_cache.get(spec.dataset, "train", cfg["max_train_samples"])
        test_data = dataset_cache.get(spec.dataset, "test", cfg["max_test_samples"])

        train_samples, val_samples = split_train_val(
            list(train_data.samples), val_ratio=cfg["val_ratio"], seed=spec.seed
        )
        train_ds = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)
        val_ds = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)

        num_l0 = len(train_data.level0_names)
        num_l1 = len(train_data.level1_names)
        model_cfg = MODELS[spec.model]

        model = LayerwiseFractalModel(
            target_layer=spec.layer,
            config=model_cfg,
            num_l0_classes=num_l0,
            num_l1_classes=num_l1,
            num_scales=cfg["num_scales"],
            scale_dim=cfg["scale_dim"],
            device=cfg["device"],
        ).to(cfg["device"])

        if model.total_backbone_layers != cfg["total_layers"]:
            raise ValueError(
                f"{spec.model} reports {model.total_backbone_layers} layers; expected {cfg['total_layers']}"
            )

        trainer = V5Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=cfg["device"],
            stage1_epochs=cfg["stage1_epochs"],
            stage2_epochs=0,
        )
        history = trainer.train(batch_size=cfg["batch_size"], patience=cfg["patience"])

        prefix_acc = evaluate_prefix_accuracy_fast(
            model=model,
            samples=test_data.samples,
            max_eval_samples=cfg["max_eval_samples"],
            batch_size=cfg["eval_batch_size"],
            k=cfg["knn_k"],
        )
        steer = steerability_from_prefix(prefix_acc)

        runtime_sec = float(time.monotonic() - t0)
        return {
            "status": "ok",
            "model": spec.model,
            "dataset": spec.dataset,
            "layer": int(spec.layer),
            "seed": int(spec.seed),
            "train_size": int(len(train_samples)),
            "val_size": int(len(val_samples)),
            "test_size": int(len(test_data.samples)),
            "backbone_hidden_dim": int(model_cfg.hidden_dim),
            "backbone_layers": int(model.total_backbone_layers),
            "prefix_accuracy": {k: float(v) for k, v in prefix_acc.items()},
            "steerability": float(steer),
            "compute": {
                "layer_count": int(spec.layer),
                "compute_proxy_relative": float(spec.layer / cfg["total_layers"]),
                "compute_proxy_absolute": float(spec.layer),
            },
            "history": history,
            "runtime_sec": runtime_sec,
            "finished_at": utc_now_iso(),
        }
    except Exception as exc:
        runtime_sec = float(time.monotonic() - t0)
        return {
            "status": "error",
            "model": spec.model,
            "dataset": spec.dataset,
            "layer": int(spec.layer),
            "seed": int(spec.seed),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_sec": runtime_sec,
            "finished_at": utc_now_iso(),
        }
    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_specs(models: Sequence[str], datasets: Sequence[str], layers: Sequence[int], seeds: Sequence[int]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for m in models:
        for d in datasets:
            for l in layers:
                for s in seeds:
                    specs.append(RunSpec(model=m, dataset=d, layer=int(l), seed=int(s)))
    return specs


def make_combos(models: Sequence[str], datasets: Sequence[str], layers: Sequence[int]) -> List[Tuple[str, str, int]]:
    combos: List[Tuple[str, str, int]] = []
    for m in models:
        for d in datasets:
            for l in layers:
                combos.append((m, d, int(l)))
    return combos


def specs_complete(state: Dict[str, Any], specs: Sequence[RunSpec]) -> bool:
    return all(run_ok(get_run(state, spec)) for spec in specs)


def summarize_values(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "se": None, "ci95": None}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(std / math.sqrt(arr.size)) if arr.size > 0 else None
    ci95 = float(1.96 * se) if se is not None else None
    return {"n": int(arr.size), "mean": mean, "std": std, "se": se, "ci95": ci95}


def seed_steerability_map(state: Dict[str, Any], model: str, dataset: str, layer: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    by_seed = state.get("results", {}).get(model, {}).get(dataset, {}).get(str(layer), {})
    for seed_s, payload in by_seed.items():
        if run_ok(payload):
            out[int(seed_s)] = float(payload["steerability"])
    return out


def aggregate_results(
    state: Dict[str, Any],
    models: Sequence[str],
    datasets: Sequence[str],
    layers: Sequence[int],
    seeds: Sequence[int],
    total_layers: int,
) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for m in models:
        agg[m] = {}
        for ds in datasets:
            agg[m][ds] = {}
            s12_by_seed = seed_steerability_map(state, m, ds, total_layers)

            for l in layers:
                s_by_seed = seed_steerability_map(state, m, ds, l)
                s_vals = [s_by_seed[s] for s in seeds if s in s_by_seed]
                s_stats = summarize_values(s_vals)

                d_by_seed: Dict[int, float] = {}
                for s in seeds:
                    if s not in s_by_seed or s not in s12_by_seed:
                        continue
                    denom = s12_by_seed[s]
                    if abs(denom) < EPS:
                        continue
                    if l == total_layers:
                        d_by_seed[s] = 0.0
                    else:
                        d_by_seed[s] = float(1.0 - (s_by_seed[s] / denom))

                d_vals = [d_by_seed[s] for s in seeds if s in d_by_seed]
                d_stats = summarize_values(d_vals)

                agg[m][ds][str(l)] = {
                    "layer": int(l),
                    "compute_proxy_relative": float(l / total_layers),
                    "compute_proxy_absolute": float(l),
                    "steerability_by_seed": {str(k): float(v) for k, v in s_by_seed.items()},
                    "steerability": s_stats,
                    "distortion_by_seed": {str(k): float(v) for k, v in d_by_seed.items()},
                    "distortion": d_stats,
                    "s12_available_seeds": sorted([int(k) for k in s12_by_seed.keys()]),
                }
    return agg


def build_distortion_points(
    state: Dict[str, Any],
    models: Sequence[str],
    fit_datasets: Sequence[str],
    fit_layers: Sequence[int],
    seeds: Sequence[int],
    total_layers: int,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for m in models:
        for ds in fit_datasets:
            s12_by_seed = seed_steerability_map(state, m, ds, total_layers)
            if not s12_by_seed:
                continue
            for s in seeds:
                s12 = s12_by_seed.get(s)
                if s12 is None or abs(s12) < EPS:
                    continue
                for l in fit_layers:
                    s_by_seed = seed_steerability_map(state, m, ds, l)
                    if s not in s_by_seed:
                        continue
                    s_l = s_by_seed[s]
                    d_l = 0.0 if l == total_layers else float(1.0 - (s_l / s12))
                    points.append(
                        {
                            "model": m,
                            "dataset": ds,
                            "layer": int(l),
                            "seed": int(s),
                            "compute": float(l / total_layers),
                            "steerability": float(s_l),
                            "steerability_l12": float(s12),
                            "distortion": float(d_l),
                        }
                    )
    return points


def distortion_law(c: np.ndarray, d_inf: float, k: float, alpha: float) -> np.ndarray:
    return d_inf + k * np.power(c, -alpha)


def aic_from_residuals(residuals: np.ndarray, n_params: int) -> float:
    n = residuals.size
    rss = float(np.sum(np.square(residuals)))
    rss = max(rss, EPS)
    return float(n * np.log(rss / max(n, 1)) + 2 * n_params)


def fit_per_model_distortion(points: Sequence[Dict[str, Any]], models: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in models:
        pts = [p for p in points if p["model"] == m]
        if len(pts) < 4:
            out[m] = {"status": "insufficient_data", "n": len(pts)}
            continue

        c = np.asarray([p["compute"] for p in pts], dtype=np.float64)
        d = np.asarray([p["distortion"] for p in pts], dtype=np.float64)

        bounds = (np.array([-2.0, -5.0, 0.01]), np.array([2.0, 5.0, 5.0]))
        p0 = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        method = "curve_fit"
        try:
            popt, _ = curve_fit(
                distortion_law,
                c,
                d,
                p0=p0,
                bounds=bounds,
                maxfev=200000,
            )
        except Exception:
            method = "least_squares_fallback"

            def residual_fn(x: np.ndarray) -> np.ndarray:
                return distortion_law(c, x[0], x[1], x[2]) - d

            ls = least_squares(residual_fn, x0=p0, bounds=bounds, max_nfev=200000)
            popt = ls.x

        pred = distortion_law(c, popt[0], popt[1], popt[2])
        residuals = d - pred
        rss = float(np.sum(residuals ** 2))
        aic = aic_from_residuals(residuals, n_params=3)
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        denom = float(np.sum((d - np.mean(d)) ** 2))
        r2 = float(1.0 - rss / denom) if denom > EPS else None
        residual_sd = float(np.sqrt(rss / max(len(d) - 3, 1)))

        out[m] = {
            "status": "ok",
            "method": method,
            "n": int(len(d)),
            "params": {"d_inf": float(popt[0]), "k": float(popt[1]), "alpha": float(popt[2])},
            "rss": rss,
            "rmse": rmse,
            "r2": r2,
            "aic": aic,
            "residual_sd": residual_sd,
        }
    return out


def fit_universality(points: Sequence[Dict[str, Any]], models: Sequence[str]) -> Dict[str, Any]:
    model_to_idx = {m: i for i, m in enumerate(models)}
    pts = [p for p in points if p["model"] in model_to_idx]
    if len(pts) == 0:
        return {"status": "insufficient_data", "n": 0}

    idx = np.asarray([model_to_idx[p["model"]] for p in pts], dtype=np.int64)
    c = np.asarray([p["compute"] for p in pts], dtype=np.float64)
    y = np.asarray([p["distortion"] for p in pts], dtype=np.float64)
    n = int(y.size)
    mcount = len(models)

    def fit_shared_alpha() -> Dict[str, Any]:
        # params = [alpha, d_inf_0..d_inf_M-1, k_0..k_M-1]
        x0 = np.array([1.0] + [0.0] * mcount + [0.5] * mcount, dtype=np.float64)
        lb = np.array([0.01] + [-2.0] * mcount + [-5.0] * mcount, dtype=np.float64)
        ub = np.array([5.0] + [2.0] * mcount + [5.0] * mcount, dtype=np.float64)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            alpha = x[0]
            d_inf = x[1 : 1 + mcount]
            k = x[1 + mcount : 1 + 2 * mcount]
            pred = d_inf[idx] + k[idx] * np.power(c, -alpha)
            return pred - y

        ls = least_squares(residual_fn, x0=x0, bounds=(lb, ub), max_nfev=200000)
        residuals = ls.fun.astype(np.float64)
        rss = float(np.sum(residuals ** 2))
        residual_sd = float(np.sqrt(rss / max(n - (1 + 2 * mcount), 1)))
        params = {
            "alpha": float(ls.x[0]),
            "d_inf": {models[i]: float(ls.x[1 + i]) for i in range(mcount)},
            "k": {models[i]: float(ls.x[1 + mcount + i]) for i in range(mcount)},
        }
        return {
            "status": "ok",
            "n": n,
            "p": int(1 + 2 * mcount),
            "params": params,
            "rss": rss,
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "aic": aic_from_residuals(residuals, n_params=1 + 2 * mcount),
            "residual_sd": residual_sd,
        }

    def fit_per_model_alpha() -> Dict[str, Any]:
        # params per model: [d_inf, k, alpha]
        x0 = []
        lb = []
        ub = []
        for _ in models:
            x0.extend([0.0, 0.5, 1.0])
            lb.extend([-2.0, -5.0, 0.01])
            ub.extend([2.0, 5.0, 5.0])
        x0 = np.asarray(x0, dtype=np.float64)
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            params = x.reshape(mcount, 3)
            d_inf = params[:, 0]
            k = params[:, 1]
            alpha = params[:, 2]
            pred = d_inf[idx] + k[idx] * np.power(c, -alpha[idx])
            return pred - y

        ls = least_squares(residual_fn, x0=x0, bounds=(lb, ub), max_nfev=200000)
        residuals = ls.fun.astype(np.float64)
        rss = float(np.sum(residuals ** 2))
        params_m = ls.x.reshape(mcount, 3)

        per_model_params: Dict[str, Any] = {}
        residual_sd_by_model: Dict[str, float] = {}
        for mname, midx in model_to_idx.items():
            mask = idx == midx
            resid_m = residuals[mask]
            rss_m = float(np.sum(resid_m ** 2))
            residual_sd_by_model[mname] = float(np.sqrt(rss_m / max(np.sum(mask) - 3, 1)))
            per_model_params[mname] = {
                "d_inf": float(params_m[midx, 0]),
                "k": float(params_m[midx, 1]),
                "alpha": float(params_m[midx, 2]),
                "residual_sd_model": residual_sd_by_model[mname],
            }

        return {
            "status": "ok",
            "n": n,
            "p": int(3 * mcount),
            "params": per_model_params,
            "rss": rss,
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "aic": aic_from_residuals(residuals, n_params=3 * mcount),
            "residual_sd": float(np.sqrt(rss / max(n - 3 * mcount, 1))),
            "residual_sd_by_model": residual_sd_by_model,
        }

    shared = fit_shared_alpha()
    per_model = fit_per_model_alpha()

    selected = "shared_alpha" if shared["aic"] <= per_model["aic"] else "per_model_alpha"
    delta = float(per_model["aic"] - shared["aic"])  # >0 favors shared
    return {
        "status": "ok",
        "n_points": n,
        "shared_alpha": shared,
        "per_model_alpha": per_model,
        "selected_model": selected,
        "aic_delta_per_model_minus_shared": delta,
    }


def predict_distortion(
    universality_fit: Dict[str, Any],
    selected_model: str,
    model: str,
    compute_c: float,
) -> Tuple[float, float]:
    if selected_model == "shared_alpha":
        shared = universality_fit["shared_alpha"]
        p = shared["params"]
        d_inf = p["d_inf"][model]
        k = p["k"][model]
        alpha = p["alpha"]
        d_pred = float(d_inf + k * (compute_c ** (-alpha)))
        d_sd = float(shared.get("residual_sd", 0.0))
        return d_pred, d_sd

    per_m = universality_fit["per_model_alpha"]
    p = per_m["params"][model]
    d_pred = float(p["d_inf"] + p["k"] * (compute_c ** (-p["alpha"])))
    d_sd = float(p.get("residual_sd_model", per_m.get("residual_sd", 0.0)))
    return d_pred, d_sd


def build_s12_predictors(
    aggregated: Dict[str, Any],
    models: Sequence[str],
    fit_datasets: Sequence[str],
    dataset_profiles: Dict[str, Dict[str, Any]],
    total_layers: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in models:
        x = []
        y = []
        for ds in fit_datasets:
            h = dataset_profiles.get(ds, {}).get("h_l1_given_l0")
            s12 = (
                aggregated.get(m, {})
                .get(ds, {})
                .get(str(total_layers), {})
                .get("steerability", {})
                .get("mean")
            )
            if h is None or s12 is None:
                continue
            x.append(float(h))
            y.append(float(s12))

        if len(x) >= 2 and abs(x[0] - x[-1]) > EPS:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            residuals = np.asarray(y) - (slope * np.asarray(x) + intercept)
            resid_sd = float(np.std(residuals, ddof=1)) if len(y) > 2 else 0.0
        elif len(x) == 1:
            slope = 0.0
            intercept = float(y[0])
            resid_sd = 0.0
        else:
            slope = 0.0
            intercept = 0.0
            resid_sd = float("nan")

        out[m] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "residual_sd": float(resid_sd),
            "fit_points": [{"h": float(xx), "s12": float(yy)} for xx, yy in zip(x, y)],
        }
    return out


def build_prediction_targets(
    models: Sequence[str],
    fit_datasets: Sequence[str],
    holdout_datasets: Sequence[str],
    fit_layers: Sequence[int],
    prospective_layers: Sequence[int],
) -> List[Dict[str, Any]]:
    holdout_layers = sorted(set(int(x) for x in fit_layers).union(set(int(x) for x in prospective_layers)))
    targets: List[Dict[str, Any]] = []
    seen = set()

    for m in models:
        for ds in fit_datasets:
            for l in prospective_layers:
                key = (m, ds, int(l))
                if key in seen:
                    continue
                seen.add(key)
                targets.append({"model": m, "dataset": ds, "layer": int(l)})

        for ds in holdout_datasets:
            for l in holdout_layers:
                key = (m, ds, int(l))
                if key in seen:
                    continue
                seen.add(key)
                targets.append({"model": m, "dataset": ds, "layer": int(l)})

    targets.sort(key=lambda t: (t["model"], t["dataset"], t["layer"]))
    return targets


def create_frozen_predictions(
    freeze_path: Path,
    config: Dict[str, Any],
    universality_fit: Dict[str, Any],
    aggregated: Dict[str, Any],
    dataset_profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    selected_model = universality_fit["selected_model"]
    total_layers = int(config["total_layers"])
    fit_datasets = config["fit_datasets"]
    holdout_datasets = config["holdout_datasets"]
    models = config["models"]

    s12_predictors = build_s12_predictors(
        aggregated=aggregated,
        models=models,
        fit_datasets=fit_datasets,
        dataset_profiles=dataset_profiles,
        total_layers=total_layers,
    )

    targets = build_prediction_targets(
        models=models,
        fit_datasets=fit_datasets,
        holdout_datasets=holdout_datasets,
        fit_layers=config["fit_layers"],
        prospective_layers=config["prospective_layers"],
    )

    target_preds: List[Dict[str, Any]] = []
    for t in targets:
        model = t["model"]
        dataset = t["dataset"]
        layer = int(t["layer"])
        c_rel = float(layer / total_layers)

        d_pred, d_sd = predict_distortion(
            universality_fit=universality_fit,
            selected_model=selected_model,
            model=model,
            compute_c=c_rel,
        )
        if layer == total_layers:
            d_pred = 0.0

        s12_obs = (
            aggregated.get(model, {})
            .get(dataset, {})
            .get(str(total_layers), {})
            .get("steerability", {})
            .get("mean")
        )
        if dataset in fit_datasets and s12_obs is not None:
            s12_ref = float(s12_obs)
            s12_sd = (
                aggregated.get(model, {})
                .get(dataset, {})
                .get(str(total_layers), {})
                .get("steerability", {})
                .get("std")
            )
            s12_sd = 0.0 if s12_sd is None else float(s12_sd)
            s12_source = "observed_fit_layer12_mean"
        else:
            h = float(dataset_profiles[dataset]["h_l1_given_l0"])
            pred = s12_predictors.get(model, {"slope": 0.0, "intercept": 0.0, "residual_sd": float("nan")})
            s12_ref = float(pred["intercept"] + pred["slope"] * h)
            s12_sd = float(pred.get("residual_sd", float("nan")))
            s12_source = "predicted_from_h_l1_given_l0"

        s_pred = float(s12_ref * (1.0 - d_pred))

        d_ci = [float(d_pred - 1.96 * d_sd), float(d_pred + 1.96 * d_sd)] if np.isfinite(d_sd) else None

        s_var = 0.0
        if np.isfinite(s12_sd):
            s_var += ((1.0 - d_pred) ** 2) * (s12_sd ** 2)
        if np.isfinite(d_sd):
            s_var += (s12_ref ** 2) * (d_sd ** 2)
        s_sd = float(math.sqrt(max(s_var, 0.0))) if s_var > 0 else 0.0
        s_ci = [float(s_pred - 1.96 * s_sd), float(s_pred + 1.96 * s_sd)]

        tags = []
        if dataset in holdout_datasets:
            tags.append("holdout_dataset")
        if layer in config["prospective_layers"]:
            tags.append("prospective_layer")

        target_preds.append(
            {
                "model": model,
                "dataset": dataset,
                "layer": layer,
                "compute_proxy_relative": c_rel,
                "predicted_distortion": float(d_pred),
                "predicted_distortion_sd": float(d_sd),
                "prediction_interval_95_D": d_ci,
                "predicted_s12": float(s12_ref),
                "s12_source": s12_source,
                "predicted_steerability": float(s_pred),
                "predicted_steerability_sd": float(s_sd),
                "prediction_interval_95_S": s_ci,
                "tags": tags,
            }
        )

    payload = {
        "type": "cti_pilot_frozen_predictions",
        "created_at": utc_now_iso(),
        "script_version": SCRIPT_VERSION,
        "note": "Frozen BEFORE running target configs: all L={3,9} runs and all holdout AGNews runs.",
        "config_snapshot": {
            "models": config["models"],
            "fit_datasets": config["fit_datasets"],
            "holdout_datasets": config["holdout_datasets"],
            "fit_layers": config["fit_layers"],
            "prospective_layers": config["prospective_layers"],
            "seeds": config["seeds"],
            "total_layers": config["total_layers"],
        },
        "dataset_profiles": dataset_profiles,
        "universality_fit": universality_fit,
        "selected_law_model": selected_model,
        "s12_predictors": s12_predictors,
        "targets": target_preds,
    }
    atomic_json_dump(freeze_path, payload)
    return payload


def load_or_create_frozen_predictions(
    freeze_path: Path,
    config: Dict[str, Any],
    state: Dict[str, Any],
    universality_fit: Dict[str, Any],
    aggregated: Dict[str, Any],
    dataset_profiles: Dict[str, Dict[str, Any]],
    force: bool = False,
) -> Dict[str, Any]:
    if freeze_path.exists():
        log(f"Using existing frozen predictions: {freeze_path}")
        return load_json(freeze_path)

    if universality_fit.get("status") != "ok":
        raise RuntimeError("Cannot freeze predictions: universality fit is unavailable.")

    calib_specs = make_specs(
        config["models"],
        config["fit_datasets"],
        config["fit_layers"],
        config["seeds"],
    )
    if not specs_complete(state, calib_specs) and not force:
        raise RuntimeError(
            "Calibration runs are incomplete. Refusing to freeze predictions early."
        )

    log("Creating frozen predictions file before target runs...")
    return create_frozen_predictions(
        freeze_path=freeze_path,
        config=config,
        universality_fit=universality_fit,
        aggregated=aggregated,
        dataset_profiles=dataset_profiles,
    )


def validate_freeze_protocol(
    state: Dict[str, Any],
    freeze_path: Path,
    target_specs: Sequence[RunSpec],
    force: bool = False,
) -> None:
    completed_target_runs = []
    for spec in target_specs:
        payload = get_run(state, spec)
        if run_ok(payload):
            completed_target_runs.append((spec, payload))

    if completed_target_runs and not freeze_path.exists() and not force:
        example = completed_target_runs[0][0]
        raise RuntimeError(
            "Protocol violation: target runs already exist but frozen predictions file is missing. "
            f"Example completed target: {example.key()}"
        )

    if freeze_path.exists() and completed_target_runs and not force:
        frozen = load_json(freeze_path)
        freeze_time = parse_iso_datetime(frozen.get("created_at"))
        if freeze_time is None:
            log("Warning: could not parse freeze timestamp; skipping temporal protocol check.")
            return

        violated = []
        for spec, payload in completed_target_runs:
            run_time = parse_iso_datetime(payload.get("finished_at"))
            if run_time is not None and run_time <= freeze_time:
                violated.append(spec.key())

        if violated:
            raise RuntimeError(
                "Protocol violation: found target runs that finished before/at frozen prediction timestamp. "
                f"Examples: {violated[:3]}"
            )


def evaluate_frozen_predictions(frozen: Dict[str, Any], aggregated: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    residuals_s = []
    residuals_d = []
    in_pi_s = 0
    in_pi_d = 0
    n_pi_s = 0
    n_pi_d = 0

    for t in frozen.get("targets", []):
        model = t["model"]
        dataset = t["dataset"]
        layer = str(t["layer"])
        actual_obj = aggregated.get(model, {}).get(dataset, {}).get(layer, {})
        actual_s = actual_obj.get("steerability", {}).get("mean")
        actual_d = actual_obj.get("distortion", {}).get("mean")

        row = copy.deepcopy(t)
        row["actual_steerability_mean"] = actual_s
        row["actual_distortion_mean"] = actual_d

        if actual_s is not None:
            res_s = float(actual_s - t["predicted_steerability"])
            row["residual_S"] = res_s
            residuals_s.append(res_s)
            pi_s = t.get("prediction_interval_95_S")
            if isinstance(pi_s, list) and len(pi_s) == 2:
                n_pi_s += 1
                if pi_s[0] <= actual_s <= pi_s[1]:
                    in_pi_s += 1

        if actual_d is not None and t.get("predicted_distortion") is not None:
            res_d = float(actual_d - t["predicted_distortion"])
            row["residual_D"] = res_d
            residuals_d.append(res_d)
            pi_d = t.get("prediction_interval_95_D")
            if isinstance(pi_d, list) and len(pi_d) == 2:
                n_pi_d += 1
                if pi_d[0] <= actual_d <= pi_d[1]:
                    in_pi_d += 1

        rows.append(row)

    summary: Dict[str, Any] = {"n_targets": len(rows)}

    if residuals_s:
        arr = np.asarray(residuals_s, dtype=np.float64)
        summary["S"] = {
            "n": int(arr.size),
            "mae": float(np.mean(np.abs(arr))),
            "rmse": float(np.sqrt(np.mean(arr ** 2))),
            "bias": float(np.mean(arr)),
            "coverage_95": float(in_pi_s / n_pi_s) if n_pi_s > 0 else None,
            "n_with_pi": int(n_pi_s),
        }

    if residuals_d:
        arr = np.asarray(residuals_d, dtype=np.float64)
        summary["D"] = {
            "n": int(arr.size),
            "mae": float(np.mean(np.abs(arr))),
            "rmse": float(np.sqrt(np.mean(arr ** 2))),
            "bias": float(np.mean(arr)),
            "coverage_95": float(in_pi_d / n_pi_d) if n_pi_d > 0 else None,
            "n_with_pi": int(n_pi_d),
        }

    pred_s = [r["predicted_steerability"] for r in rows if r.get("actual_steerability_mean") is not None]
    act_s = [r["actual_steerability_mean"] for r in rows if r.get("actual_steerability_mean") is not None]
    if len(pred_s) >= 3 and "S" in summary:
        rho, p_rho = stats.spearmanr(pred_s, act_s)
        r, p_r = stats.pearsonr(pred_s, act_s)
        summary["S"]["spearman_rho"] = float(rho)
        summary["S"]["spearman_p"] = float(p_rho)
        summary["S"]["pearson_r"] = float(r)
        summary["S"]["pearson_p"] = float(p_r)

    return {"rows": rows, "summary": summary}


def execute_phase(
    phase_name: str,
    combos: Sequence[Tuple[str, str, int]],
    seeds: Sequence[int],
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    dataset_cache: DatasetCache,
    output_path: Path,
    rerun_errors: bool = False,
    fail_fast: bool = False,
    dry_run: bool = False,
) -> None:
    log(f"Starting phase '{phase_name}' with {len(combos)} model/dataset/layer combos")

    for combo_idx, (model, dataset, layer) in enumerate(combos, start=1):
        combo_key = f"{model}|{dataset}|L{layer}"
        log(f"[{phase_name}] combo {combo_idx}/{len(combos)}: {combo_key}")

        ok_count = 0
        err_count = 0
        completed_seeds: List[int] = []

        for seed in seeds:
            spec = RunSpec(model=model, dataset=dataset, layer=layer, seed=seed)
            existing = get_run(state, spec)

            if run_ok(existing):
                log(f"  seed={seed}: skip (already complete)")
                ok_count += 1
                completed_seeds.append(seed)
                continue

            if isinstance(existing, dict) and existing.get("status") == "error" and not rerun_errors:
                log(f"  seed={seed}: skip (previous error, rerun disabled)")
                err_count += 1
                continue

            if dry_run:
                log(f"  seed={seed}: dry-run planned")
                continue

            log(f"  seed={seed}: running")
            payload = run_single_config(spec=spec, cfg=cfg, dataset_cache=dataset_cache)
            set_run(state, spec, payload)

            if payload.get("status") == "ok":
                log(f"  seed={seed}: done S={payload['steerability']:+.5f}")
                ok_count += 1
                completed_seeds.append(seed)
            else:
                log(f"  seed={seed}: ERROR {payload.get('error', 'unknown')}")
                err_count += 1
                if fail_fast:
                    atomic_json_dump(output_path, state)
                    raise RuntimeError(f"Fail-fast enabled; aborting on {spec.key()}")

            state["updated_at"] = utc_now_iso()
            atomic_json_dump(output_path, state)

        state.setdefault("combo_progress", {}).setdefault(phase_name, {})[combo_key] = {
            "ok_count": int(ok_count),
            "error_count": int(err_count),
            "completed_seeds": sorted(completed_seeds),
            "updated_at": utc_now_iso(),
        }
        state["updated_at"] = utc_now_iso()
        atomic_json_dump(output_path, state)


def count_runs(state: Dict[str, Any]) -> Tuple[int, int]:
    total = 0
    ok = 0
    for m_dict in state.get("results", {}).values():
        for d_dict in m_dict.values():
            for l_dict in d_dict.values():
                for payload in l_dict.values():
                    total += 1
                    if run_ok(payload):
                        ok += 1
    return total, ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CTI pilot: layerwise compute-distortion law for steerability.")
    p.add_argument("--output", type=str, default="results/cti_pilot_results.json")
    p.add_argument("--frozen-predictions-path", type=str, default="results/cti_pilot_frozen_predictions.json")
    p.add_argument("--models", type=str, default="bge-base,e5-base,bge-small")
    p.add_argument("--fit-datasets", type=str, default="dbpedia_classes,clinc")
    p.add_argument("--holdout-datasets", type=str, default="agnews")
    p.add_argument("--fit-layers", type=str, default="2,4,6,8,10,12")
    p.add_argument("--prospective-layers", type=str, default="3,9")
    p.add_argument("--seeds", type=str, default="42,123,456")
    p.add_argument("--total-layers", type=int, default=12)
    p.add_argument("--stage1-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-train-samples", type=int, default=10000)
    p.add_argument("--max-test-samples", type=int, default=2000)
    p.add_argument("--max-eval-samples", type=int, default=500)
    p.add_argument("--num-scales", type=int, default=4)
    p.add_argument("--scale-dim", type=int, default=64)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--phase", choices=["all", "calibration", "targets", "analysis"], default="all")
    p.add_argument("--rerun-errors", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    models = parse_csv_str(args.models)
    fit_datasets = parse_csv_str(args.fit_datasets)
    holdout_datasets = parse_csv_str(args.holdout_datasets)
    fit_layers = sorted(set(parse_csv_int(args.fit_layers)))
    prospective_layers = sorted(set(parse_csv_int(args.prospective_layers)))
    seeds = [int(s) for s in parse_csv_int(args.seeds)]

    if args.total_layers not in fit_layers:
        raise ValueError(f"fit_layers must include total layer {args.total_layers} for D(L)=1-S(L)/S(L12).")
    if any(l <= 0 for l in fit_layers + prospective_layers):
        raise ValueError("All layers must be positive integers.")

    for m in models:
        if m not in MODELS:
            raise ValueError(f"Unknown model '{m}'.")
        if m in EXPECTED_MODEL_DIMS:
            got = int(MODELS[m].hidden_dim)
            exp = int(EXPECTED_MODEL_DIMS[m])
            if got != exp:
                raise ValueError(f"Model {m} hidden_dim mismatch: expected {exp}, got {got}")

    config = {
        "script": "cti_pilot.py",
        "script_version": SCRIPT_VERSION,
        "models": models,
        "fit_datasets": fit_datasets,
        "holdout_datasets": holdout_datasets,
        "fit_layers": fit_layers,
        "prospective_layers": prospective_layers,
        "seeds": seeds,
        "total_layers": int(args.total_layers),
        "stage1_epochs": int(args.stage1_epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "knn_k": int(args.knn_k),
        "patience": int(args.patience),
        "val_ratio": float(args.val_ratio),
        "max_train_samples": int(args.max_train_samples),
        "max_test_samples": int(args.max_test_samples),
        "max_eval_samples": int(args.max_eval_samples),
        "num_scales": int(args.num_scales),
        "scale_dim": int(args.scale_dim),
        "device": args.device,
    }

    output_path = Path(args.output)
    freeze_path = Path(args.frozen_predictions_path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if not freeze_path.is_absolute():
        freeze_path = REPO_ROOT / freeze_path

    if output_path.exists():
        state = load_json(output_path)
        old_cfg = state.get("config", {})
        ok_cfg, key = config_compatible(old_cfg, config)
        if not ok_cfg and not args.force:
            raise RuntimeError(f"Existing state config mismatch at '{key}'. Use --force to override.")
        if not ok_cfg and args.force:
            log(f"Warning: forcing config override (mismatch key: {key})")
            state["config"] = config
    else:
        state = {
            "meta": {
                "created_at": utc_now_iso(),
                "script": "cti_pilot.py",
                "script_version": SCRIPT_VERSION,
            },
            "config": config,
            "results": {},
            "combo_progress": {},
            "phase": {},
            "analysis": {},
        }
        atomic_json_dump(output_path, state)

    all_layers = sorted(set(fit_layers).union(set(prospective_layers)))
    all_datasets = sorted(set(fit_datasets).union(set(holdout_datasets)))

    calibration_specs = make_specs(models, fit_datasets, fit_layers, seeds)
    calibration_combos = make_combos(models, fit_datasets, fit_layers)

    target_combos_set = set(make_combos(models, fit_datasets, prospective_layers))
    holdout_layers = sorted(set(fit_layers).union(set(prospective_layers)))
    target_combos_set.update(make_combos(models, holdout_datasets, holdout_layers))
    target_combos = sorted(list(target_combos_set), key=lambda x: (x[0], x[1], x[2]))
    target_specs = [
        RunSpec(model=m, dataset=d, layer=l, seed=s)
        for (m, d, l) in target_combos
        for s in seeds
    ]

    validate_freeze_protocol(state, freeze_path, target_specs, force=args.force)

    dataset_cache = DatasetCache()
    dataset_profiles: Dict[str, Dict[str, Any]] = {}
    for ds in all_datasets:
        train_ds = dataset_cache.get(ds, "train", config["max_train_samples"])
        dataset_profiles[ds] = dataset_profile(ds, train_ds)

    if args.phase in ("all", "calibration"):
        execute_phase(
            phase_name="calibration",
            combos=calibration_combos,
            seeds=seeds,
            cfg=config,
            state=state,
            dataset_cache=dataset_cache,
            output_path=output_path,
            rerun_errors=args.rerun_errors,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
        )
        state.setdefault("phase", {})["calibration_completed"] = specs_complete(state, calibration_specs)
        state["updated_at"] = utc_now_iso()
        atomic_json_dump(output_path, state)

    aggregated = aggregate_results(
        state=state,
        models=models,
        datasets=all_datasets,
        layers=all_layers,
        seeds=seeds,
        total_layers=config["total_layers"],
    )
    fit_points = build_distortion_points(
        state=state,
        models=models,
        fit_datasets=fit_datasets,
        fit_layers=fit_layers,
        seeds=seeds,
        total_layers=config["total_layers"],
    )
    per_model_fit = fit_per_model_distortion(fit_points, models=models)
    universality_fit = fit_universality(fit_points, models=models) if fit_points else {"status": "insufficient_data", "n": 0}

    frozen_predictions = None
    if args.phase in ("all", "targets", "analysis"):
        frozen_predictions = load_or_create_frozen_predictions(
            freeze_path=freeze_path,
            config=config,
            state=state,
            universality_fit=universality_fit,
            aggregated=aggregated,
            dataset_profiles=dataset_profiles,
            force=args.force,
        )

    if args.phase in ("all", "targets"):
        execute_phase(
            phase_name="targets",
            combos=target_combos,
            seeds=seeds,
            cfg=config,
            state=state,
            dataset_cache=dataset_cache,
            output_path=output_path,
            rerun_errors=args.rerun_errors,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
        )
        state.setdefault("phase", {})["targets_completed"] = specs_complete(state, target_specs)
        state["updated_at"] = utc_now_iso()
        atomic_json_dump(output_path, state)

    aggregated = aggregate_results(
        state=state,
        models=models,
        datasets=all_datasets,
        layers=all_layers,
        seeds=seeds,
        total_layers=config["total_layers"],
    )
    fit_points = build_distortion_points(
        state=state,
        models=models,
        fit_datasets=fit_datasets,
        fit_layers=fit_layers,
        seeds=seeds,
        total_layers=config["total_layers"],
    )
    per_model_fit = fit_per_model_distortion(fit_points, models=models)
    universality_fit = fit_universality(fit_points, models=models) if fit_points else {"status": "insufficient_data", "n": 0}

    if frozen_predictions is None and freeze_path.exists():
        frozen_predictions = load_json(freeze_path)

    prediction_eval = evaluate_frozen_predictions(frozen_predictions, aggregated) if isinstance(frozen_predictions, dict) else None

    state["analysis"] = {
        "updated_at": utc_now_iso(),
        "dataset_profiles": dataset_profiles,
        "aggregated": aggregated,
        "fit_points_count": int(len(fit_points)),
        "fit_points_scope": {
            "datasets": fit_datasets,
            "layers": fit_layers,
            "seeds": seeds,
        },
        "per_model_distortion_fit": per_model_fit,
        "universality_fit": universality_fit,
        "prediction_evaluation": prediction_eval,
    }
    state["frozen_predictions_path"] = str(freeze_path)
    state["updated_at"] = utc_now_iso()

    total_runs, ok_runs = count_runs(state)
    state["run_counts"] = {"total_saved": int(total_runs), "ok": int(ok_runs)}
    state["completed"] = bool(specs_complete(state, calibration_specs) and specs_complete(state, target_specs))
    if state["completed"]:
        state["completed_at"] = utc_now_iso()

    atomic_json_dump(output_path, state)

    log(f"Saved state: {output_path}")
    log(f"Run count: {ok_runs}/{total_runs} successful")
    if universality_fit.get("status") == "ok":
        log(
            "Universality AIC: "
            f"shared={universality_fit['shared_alpha']['aic']:.3f}, "
            f"per_model={universality_fit['per_model_alpha']['aic']:.3f}, "
            f"selected={universality_fit['selected_model']}"
        )
    if prediction_eval and "summary" in prediction_eval:
        s_summary = prediction_eval["summary"].get("S", {})
        if s_summary:
            log(
                "Frozen prediction eval (S): "
                f"n={s_summary.get('n')}, "
                f"MAE={s_summary.get('mae'):.5f}, "
                f"RMSE={s_summary.get('rmse'):.5f}"
            )


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
cti_pilot.py

Compute Thermodynamics of Intelligence (CTI) pilot experiment.

Protocol:
- Models: bge-base, e5-base, bge-small
- Fit datasets: dbpedia_classes, clinc
- Holdout dataset: agnews
- Core measured layers: 2,4,6,8,10,12
- Prospective layers: 3,9
- Seeds: 42,123,456

Experiment:
1) Train V5 head-only at each chosen backbone layer (backbone frozen)
2) Measure steerability S=(L0@j1-L0@j4)+(L1@j4-L1@j1)
3) Compute compute proxy C(L)=L/12 and distortion D(L)=1-S(L)/S(12)
4) Fit D(C)=D_inf + k*C^{-alpha}
5) Compare universality models:
   - shared alpha across models
   - per-model alpha
   using AIC.

Resume support:
- State is saved incrementally after each seed/combo.
- Frozen prospective predictions are generated before any target runs
  (L={3,9} and holdout AGNews).
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit, least_squares

REPO_ROOT = Path(__file__).resolve().parent.parent  # src/ -> repo root
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fractal_v5 import FractalModelV5, V5Trainer, split_train_val
from hierarchical_datasets import load_hierarchical_dataset
from multi_model_pipeline import MODELS

SCRIPT_VERSION = "1.0.0"
EPS = 1e-12
EXPECTED_MODEL_DIMS = {"bge-base": 768, "e5-base": 768, "bge-small": 384}


@dataclass(frozen=True)
class RunSpec:
    model: str
    dataset: str
    layer: int
    seed: int

    def key(self) -> str:
        return f"{self.model}|{self.dataset}|L{self.layer}|seed{self.seed}"


class TempDataset:
    def __init__(self, samples, level0_names, level1_names):
        self.samples = samples
        self.level0_names = level0_names
        self.level1_names = level1_names


class DatasetCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, str, Optional[int]], Any] = {}

    def get(self, name: str, split: str, max_samples: Optional[int]) -> Any:
        key = (name, split, max_samples)
        if key not in self._cache:
            log(f"Loading dataset {name} split={split} max_samples={max_samples}")
            self._cache[key] = load_hierarchical_dataset(name, split=split, max_samples=max_samples)
        return self._cache[key]


class LayerwiseFractalModel(FractalModelV5):
    """
    Same V5 architecture, but reads backbone hidden_states[layer_idx] instead of last layer.
    """

    def __init__(self, target_layer: int, *args, **kwargs):
        self.target_layer = int(target_layer)
        super().__init__(*args, **kwargs)
        self.total_backbone_layers = self._infer_total_layers()
        if self.target_layer < 1 or self.target_layer > self.total_backbone_layers:
            raise ValueError(
                f"Invalid target_layer={self.target_layer}; model has {self.total_backbone_layers} layers"
            )

    def _infer_total_layers(self) -> int:
        cfg = getattr(self.backbone, "config", None)
        if cfg is not None and getattr(cfg, "num_hidden_layers", None) is not None:
            return int(cfg.num_hidden_layers)
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return int(len(self.backbone.encoder.layer))
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
            return int(len(self.backbone.model.layers))
        if hasattr(self.backbone, "layers"):
            return int(len(self.backbone.layers))
        raise RuntimeError("Could not infer number of backbone layers")

    def forward(self, input_ids, attention_mask, block_dropout_mask: Optional[torch.Tensor] = None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Backbone did not return hidden_states; required for layerwise CTI experiment")
        if self.target_layer >= len(hidden_states):
            raise RuntimeError(
                f"target_layer={self.target_layer} out of range for hidden_states len={len(hidden_states)}"
            )
        pooled = self.pool(hidden_states[self.target_layer], attention_mask).float()
        return self.fractal_head(pooled, block_dropout_mask)


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_str(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_int(value: str) -> List[int]:
    return [int(x) for x in parse_csv_str(value)]


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def atomic_json_dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=False)
    tmp.replace(path)


def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def entropy_from_labels(labels: Sequence[int]) -> float:
    arr = np.asarray(labels, dtype=np.int64)
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr)
    probs = counts[counts > 0].astype(np.float64) / float(arr.size)
    return float(-np.sum(probs * np.log2(probs + EPS)))


def conditional_entropy(fine_labels: Sequence[int], coarse_labels: Sequence[int]) -> float:
    fine = np.asarray(fine_labels, dtype=np.int64)
    coarse = np.asarray(coarse_labels, dtype=np.int64)
    if fine.size == 0 or coarse.size == 0:
        return 0.0
    total = float(len(fine))
    h = 0.0
    for c in np.unique(coarse):
        idx = coarse == c
        p_c = float(np.sum(idx)) / total
        h += p_c * entropy_from_labels(fine[idx])
    return float(h)


def dataset_profile(dataset_name: str, dataset_obj: Any) -> Dict[str, Any]:
    l0 = [s.level0_label for s in dataset_obj.samples]
    l1 = [s.level1_label for s in dataset_obj.samples]
    return {
        "dataset": dataset_name,
        "n_samples": int(len(dataset_obj.samples)),
        "n_l0": int(len(set(l0))),
        "n_l1": int(len(set(l1))),
        "h_l1_given_l0": float(conditional_entropy(l1, l0)),
    }


def get_run(state: Dict[str, Any], spec: RunSpec) -> Optional[Dict[str, Any]]:
    return (
        state.get("results", {})
        .get(spec.model, {})
        .get(spec.dataset, {})
        .get(str(spec.layer), {})
        .get(str(spec.seed))
    )


def set_run(state: Dict[str, Any], spec: RunSpec, payload: Dict[str, Any]) -> None:
    state.setdefault("results", {}).setdefault(spec.model, {}).setdefault(spec.dataset, {}).setdefault(
        str(spec.layer), {}
    )[str(spec.seed)] = payload


def run_ok(payload: Optional[Dict[str, Any]]) -> bool:
    return isinstance(payload, dict) and payload.get("status") == "ok"


def parse_iso_datetime(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def config_compatible(existing: Dict[str, Any], current: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    keys = [
        "models",
        "fit_datasets",
        "holdout_datasets",
        "fit_layers",
        "prospective_layers",
        "seeds",
        "total_layers",
        "max_train_samples",
        "max_test_samples",
    ]
    for k in keys:
        if existing.get(k) != current.get(k):
            return False, k
    return True, None


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, EPS, None)
    return x / norms


def fast_knn_accuracy(emb: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    k = int(max(1, min(k, n - 1)))

    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)

    if k < n - 1:
        nn_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    else:
        nn_idx = np.argsort(-sim, axis=1)[:, :k]

    nn_labels = labels[nn_idx]
    correct = 0
    for i in range(n):
        vals, counts = np.unique(nn_labels[i], return_counts=True)
        pred = vals[np.argmax(counts)]
        if pred == labels[i]:
            correct += 1
    return float(correct / n)


def evaluate_prefix_accuracy_fast(
    model: LayerwiseFractalModel,
    samples: Sequence[Any],
    max_eval_samples: int = 500,
    batch_size: int = 64,
    k: int = 5,
) -> Dict[str, float]:
    n = min(len(samples), int(max_eval_samples))
    eval_samples = list(samples[:n])
    if len(eval_samples) < k + 1:
        raise ValueError(f"Need at least {k + 1} eval samples, got {len(eval_samples)}")

    texts = [s.text for s in eval_samples]
    l0 = np.asarray([s.level0_label for s in eval_samples], dtype=np.int64)
    l1 = np.asarray([s.level1_label for s in eval_samples], dtype=np.int64)

    results: Dict[str, float] = {}
    for j in (1, 2, 3, 4):
        prefix_len = j if j < 4 else None
        with torch.no_grad():
            emb = model.encode(texts, batch_size=batch_size, prefix_len=prefix_len).cpu().numpy()
        emb = l2_normalize(emb)
        results[f"j{j}_l0"] = fast_knn_accuracy(emb, l0, k=k)
        results[f"j{j}_l1"] = fast_knn_accuracy(emb, l1, k=k)
    return results


def steerability_from_prefix(prefix_accuracy: Dict[str, float]) -> float:
    return float(
        (prefix_accuracy.get("j1_l0", 0.0) - prefix_accuracy.get("j4_l0", 0.0))
        + (prefix_accuracy.get("j4_l1", 0.0) - prefix_accuracy.get("j1_l1", 0.0))
    )


def run_single_config(spec: RunSpec, cfg: Dict[str, Any], dataset_cache: DatasetCache) -> Dict[str, Any]:
    t0 = time.monotonic()
    model = None
    trainer = None
    try:
        set_seed(spec.seed)

        train_data = dataset_cache.get(spec.dataset, "train", cfg["max_train_samples"])
        test_data = dataset_cache.get(spec.dataset, "test", cfg["max_test_samples"])

        train_samples, val_samples = split_train_val(
            list(train_data.samples), val_ratio=cfg["val_ratio"], seed=spec.seed
        )
        train_ds = TempDataset(train_samples, train_data.level0_names, train_data.level1_names)
        val_ds = TempDataset(val_samples, train_data.level0_names, train_data.level1_names)

        num_l0 = len(train_data.level0_names)
        num_l1 = len(train_data.level1_names)
        model_cfg = MODELS[spec.model]

        model = LayerwiseFractalModel(
            target_layer=spec.layer,
            config=model_cfg,
            num_l0_classes=num_l0,
            num_l1_classes=num_l1,
            num_scales=cfg["num_scales"],
            scale_dim=cfg["scale_dim"],
            device=cfg["device"],
        ).to(cfg["device"])

        if model.total_backbone_layers != cfg["total_layers"]:
            raise ValueError(
                f"{spec.model} reports {model.total_backbone_layers} layers; expected {cfg['total_layers']}"
            )

        trainer = V5Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=cfg["device"],
            stage1_epochs=cfg["stage1_epochs"],
            stage2_epochs=0,
        )
        history = trainer.train(batch_size=cfg["batch_size"], patience=cfg["patience"])

        prefix_acc = evaluate_prefix_accuracy_fast(
            model=model,
            samples=test_data.samples,
            max_eval_samples=cfg["max_eval_samples"],
            batch_size=cfg["eval_batch_size"],
            k=cfg["knn_k"],
        )
        steer = steerability_from_prefix(prefix_acc)

        runtime_sec = float(time.monotonic() - t0)
        return {
            "status": "ok",
            "model": spec.model,
            "dataset": spec.dataset,
            "layer": int(spec.layer),
            "seed": int(spec.seed),
            "train_size": int(len(train_samples)),
            "val_size": int(len(val_samples)),
            "test_size": int(len(test_data.samples)),
            "backbone_hidden_dim": int(model_cfg.hidden_dim),
            "backbone_layers": int(model.total_backbone_layers),
            "prefix_accuracy": {k: float(v) for k, v in prefix_acc.items()},
            "steerability": float(steer),
            "compute": {
                "layer_count": int(spec.layer),
                "compute_proxy_relative": float(spec.layer / cfg["total_layers"]),
                "compute_proxy_absolute": float(spec.layer),
            },
            "history": history,
            "runtime_sec": runtime_sec,
            "finished_at": utc_now_iso(),
        }
    except Exception as exc:
        runtime_sec = float(time.monotonic() - t0)
        return {
            "status": "error",
            "model": spec.model,
            "dataset": spec.dataset,
            "layer": int(spec.layer),
            "seed": int(spec.seed),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_sec": runtime_sec,
            "finished_at": utc_now_iso(),
        }
    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_specs(models: Sequence[str], datasets: Sequence[str], layers: Sequence[int], seeds: Sequence[int]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for m in models:
        for d in datasets:
            for l in layers:
                for s in seeds:
                    specs.append(RunSpec(model=m, dataset=d, layer=int(l), seed=int(s)))
    return specs


def make_combos(models: Sequence[str], datasets: Sequence[str], layers: Sequence[int]) -> List[Tuple[str, str, int]]:
    combos: List[Tuple[str, str, int]] = []
    for m in models:
        for d in datasets:
            for l in layers:
                combos.append((m, d, int(l)))
    return combos


def specs_complete(state: Dict[str, Any], specs: Sequence[RunSpec]) -> bool:
    return all(run_ok(get_run(state, spec)) for spec in specs)


def summarize_values(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "se": None, "ci95": None}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(std / math.sqrt(arr.size)) if arr.size > 0 else None
    ci95 = float(1.96 * se) if se is not None else None
    return {"n": int(arr.size), "mean": mean, "std": std, "se": se, "ci95": ci95}


def seed_steerability_map(state: Dict[str, Any], model: str, dataset: str, layer: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    by_seed = state.get("results", {}).get(model, {}).get(dataset, {}).get(str(layer), {})
    for seed_s, payload in by_seed.items():
        if run_ok(payload):
            out[int(seed_s)] = float(payload["steerability"])
    return out


def aggregate_results(
    state: Dict[str, Any],
    models: Sequence[str],
    datasets: Sequence[str],
    layers: Sequence[int],
    seeds: Sequence[int],
    total_layers: int,
) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for m in models:
        agg[m] = {}
        for ds in datasets:
            agg[m][ds] = {}
            s12_by_seed = seed_steerability_map(state, m, ds, total_layers)

            for l in layers:
                s_by_seed = seed_steerability_map(state, m, ds, l)
                s_vals = [s_by_seed[s] for s in seeds if s in s_by_seed]
                s_stats = summarize_values(s_vals)

                d_by_seed: Dict[int, float] = {}
                for s in seeds:
                    if s not in s_by_seed or s not in s12_by_seed:
                        continue
                    denom = s12_by_seed[s]
                    if abs(denom) < EPS:
                        continue
                    if l == total_layers:
                        d_by_seed[s] = 0.0
                    else:
                        d_by_seed[s] = float(1.0 - (s_by_seed[s] / denom))

                d_vals = [d_by_seed[s] for s in seeds if s in d_by_seed]
                d_stats = summarize_values(d_vals)

                agg[m][ds][str(l)] = {
                    "layer": int(l),
                    "compute_proxy_relative": float(l / total_layers),
                    "compute_proxy_absolute": float(l),
                    "steerability_by_seed": {str(k): float(v) for k, v in s_by_seed.items()},
                    "steerability": s_stats,
                    "distortion_by_seed": {str(k): float(v) for k, v in d_by_seed.items()},
                    "distortion": d_stats,
                    "s12_available_seeds": sorted([int(k) for k in s12_by_seed.keys()]),
                }
    return agg


def build_distortion_points(
    state: Dict[str, Any],
    models: Sequence[str],
    fit_datasets: Sequence[str],
    fit_layers: Sequence[int],
    seeds: Sequence[int],
    total_layers: int,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for m in models:
        for ds in fit_datasets:
            s12_by_seed = seed_steerability_map(state, m, ds, total_layers)
            if not s12_by_seed:
                continue
            for s in seeds:
                s12 = s12_by_seed.get(s)
                if s12 is None or abs(s12) < EPS:
                    continue
                for l in fit_layers:
                    s_by_seed = seed_steerability_map(state, m, ds, l)
                    if s not in s_by_seed:
                        continue
                    s_l = s_by_seed[s]
                    d_l = 0.0 if l == total_layers else float(1.0 - (s_l / s12))
                    points.append(
                        {
                            "model": m,
                            "dataset": ds,
                            "layer": int(l),
                            "seed": int(s),
                            "compute": float(l / total_layers),
                            "steerability": float(s_l),
                            "steerability_l12": float(s12),
                            "distortion": float(d_l),
                        }
                    )
    return points


def distortion_law(c: np.ndarray, d_inf: float, k: float, alpha: float) -> np.ndarray:
    return d_inf + k * np.power(c, -alpha)


def aic_from_residuals(residuals: np.ndarray, n_params: int) -> float:
    n = residuals.size
    rss = float(np.sum(np.square(residuals)))
    rss = max(rss, EPS)
    return float(n * np.log(rss / max(n, 1)) + 2 * n_params)


def fit_per_model_distortion(points: Sequence[Dict[str, Any]], models: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in models:
        pts = [p for p in points if p["model"] == m]
        if len(pts) < 4:
            out[m] = {"status": "insufficient_data", "n": len(pts)}
            continue

        c = np.asarray([p["compute"] for p in pts], dtype=np.float64)
        d = np.asarray([p["distortion"] for p in pts], dtype=np.float64)

        bounds = (np.array([-2.0, -5.0, 0.01]), np.array([2.0, 5.0, 5.0]))
        p0 = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        method = "curve_fit"
        try:
            popt, _ = curve_fit(
                distortion_law,
                c,
                d,
                p0=p0,
                bounds=bounds,
                maxfev=200000,
            )
        except Exception:
            method = "least_squares_fallback"

            def residual_fn(x: np.ndarray) -> np.ndarray:
                return distortion_law(c, x[0], x[1], x[2]) - d

            ls = least_squares(residual_fn, x0=p0, bounds=bounds, max_nfev=200000)
            popt = ls.x

        pred = distortion_law(c, popt[0], popt[1], popt[2])
        residuals = d - pred
        rss = float(np.sum(residuals ** 2))
        aic = aic_from_residuals(residuals, n_params=3)
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        denom = float(np.sum((d - np.mean(d)) ** 2))
        r2 = float(1.0 - rss / denom) if denom > EPS else None
        residual_sd = float(np.sqrt(rss / max(len(d) - 3, 1)))

        out[m] = {
            "status": "ok",
            "method": method,
            "n": int(len(d)),
            "params": {"d_inf": float(popt[0]), "k": float(popt[1]), "alpha": float(popt[2])},
            "rss": rss,
            "rmse": rmse,
            "r2": r2,
            "aic": aic,
            "residual_sd": residual_sd,
        }
    return out


def fit_universality(points: Sequence[Dict[str, Any]], models: Sequence[str]) -> Dict[str, Any]:
    model_to_idx = {m: i for i, m in enumerate(models)}
    pts = [p for p in points if p["model"] in model_to_idx]
    if len(pts) == 0:
        return {"status": "insufficient_data", "n": 0}

    idx = np.asarray([model_to_idx[p["model"]] for p in pts], dtype=np.int64)
    c = np.asarray([p["compute"] for p in pts], dtype=np.float64)
    y = np.asarray([p["distortion"] for p in pts], dtype=np.float64)
    n = int(y.size)
    mcount = len(models)

    def fit_shared_alpha() -> Dict[str, Any]:
        # params = [alpha, d_inf_0..d_inf_M-1, k_0..k_M-1]
        x0 = np.array([1.0] + [0.0] * mcount + [0.5] * mcount, dtype=np.float64)
        lb = np.array([0.01] + [-2.0] * mcount + [-5.0] * mcount, dtype=np.float64)
        ub = np.array([5.0] + [2.0] * mcount + [5.0] * mcount, dtype=np.float64)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            alpha = x[0]
            d_inf = x[1 : 1 + mcount]
            k = x[1 + mcount : 1 + 2 * mcount]
            pred = d_inf[idx] + k[idx] * np.power(c, -alpha)
            return pred - y

        ls = least_squares(residual_fn, x0=x0, bounds=(lb, ub), max_nfev=200000)
        residuals = ls.fun.astype(np.float64)
        rss = float(np.sum(residuals ** 2))
        residual_sd = float(np.sqrt(rss / max(n - (1 + 2 * mcount), 1)))
        params = {
            "alpha": float(ls.x[0]),
            "d_inf": {models[i]: float(ls.x[1 + i]) for i in range(mcount)},
            "k": {models[i]: float(ls.x[1 + mcount + i]) for i in range(mcount)},
        }
        return {
            "status": "ok",
            "n": n,
            "p": int(1 + 2 * mcount),
            "params": params,
            "rss": rss,
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "aic": aic_from_residuals(residuals, n_params=1 + 2 * mcount),
            "residual_sd": residual_sd,
        }

    def fit_per_model_alpha() -> Dict[str, Any]:
        # params per model: [d_inf, k, alpha]
        x0 = []
        lb = []
        ub = []
        for _ in models:
            x0.extend([0.0, 0.5, 1.0])
            lb.extend([-2.0, -5.0, 0.01])
            ub.extend([2.0, 5.0, 5.0])
        x0 = np.asarray(x0, dtype=np.float64)
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            params = x.reshape(mcount, 3)
            d_inf = params[:, 0]
            k = params[:, 1]
            alpha = params[:, 2]
            pred = d_inf[idx] + k[idx] * np.power(c, -alpha[idx])
            return pred - y

        ls = least_squares(residual_fn, x0=x0, bounds=(lb, ub), max_nfev=200000)
        residuals = ls.fun.astype(np.float64)
        rss = float(np.sum(residuals ** 2))
        params_m = ls.x.reshape(mcount, 3)

        per_model_params: Dict[str, Any] = {}
        residual_sd_by_model: Dict[str, float] = {}
        for mname, midx in model_to_idx.items():
            mask = idx == midx
            resid_m = residuals[mask]
            rss_m = float(np.sum(resid_m ** 2))
            residual_sd_by_model[mname] = float(np.sqrt(rss_m / max(np.sum(mask) - 3, 1)))
            per_model_params[mname] = {
                "d_inf": float(params_m[midx, 0]),
                "k": float(params_m[midx, 1]),
                "alpha": float(params_m[midx, 2]),
                "residual_sd_model": residual_sd_by_model[mname],
            }

        return {
            "status": "ok",
            "n": n,
            "p": int(3 * mcount),
            "params": per_model_params,
            "rss": rss,
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "aic": aic_from_residuals(residuals, n_params=3 * mcount),
            "residual_sd": float(np.sqrt(rss / max(n - 3 * mcount, 1))),
            "residual_sd_by_model": residual_sd_by_model,
        }

    shared = fit_shared_alpha()
    per_model = fit_per_model_alpha()

    selected = "shared_alpha" if shared["aic"] <= per_model["aic"] else "per_model_alpha"
    delta = float(per_model["aic"] - shared["aic"])  # >0 favors shared
    return {
        "status": "ok",
        "n_points": n,
        "shared_alpha": shared,
        "per_model_alpha": per_model,
        "selected_model": selected,
        "aic_delta_per_model_minus_shared": delta,
    }


def predict_distortion(
    universality_fit: Dict[str, Any],
    selected_model: str,
    model: str,
    compute_c: float,
) -> Tuple[float, float]:
    if selected_model == "shared_alpha":
        shared = universality_fit["shared_alpha"]
        p = shared["params"]
        d_inf = p["d_inf"][model]
        k = p["k"][model]
        alpha = p["alpha"]
        d_pred = float(d_inf + k * (compute_c ** (-alpha)))
        d_sd = float(shared.get("residual_sd", 0.0))
        return d_pred, d_sd

    per_m = universality_fit["per_model_alpha"]
    p = per_m["params"][model]
    d_pred = float(p["d_inf"] + p["k"] * (compute_c ** (-p["alpha"])))
    d_sd = float(p.get("residual_sd_model", per_m.get("residual_sd", 0.0)))
    return d_pred, d_sd


def build_s12_predictors(
    aggregated: Dict[str, Any],
    models: Sequence[str],
    fit_datasets: Sequence[str],
    dataset_profiles: Dict[str, Dict[str, Any]],
    total_layers: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in models:
        x = []
        y = []
        for ds in fit_datasets:
            h = dataset_profiles.get(ds, {}).get("h_l1_given_l0")
            s12 = (
                aggregated.get(m, {})
                .get(ds, {})
                .get(str(total_layers), {})
                .get("steerability", {})
                .get("mean")
            )
            if h is None or s12 is None:
                continue
            x.append(float(h))
            y.append(float(s12))

        if len(x) >= 2 and abs(x[0] - x[-1]) > EPS:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            residuals = np.asarray(y) - (slope * np.asarray(x) + intercept)
            resid_sd = float(np.std(residuals, ddof=1)) if len(y) > 2 else 0.0
        elif len(x) == 1:
            slope = 0.0
            intercept = float(y[0])
            resid_sd = 0.0
        else:
            slope = 0.0
            intercept = 0.0
            resid_sd = float("nan")

        out[m] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "residual_sd": float(resid_sd),
            "fit_points": [{"h": float(xx), "s12": float(yy)} for xx, yy in zip(x, y)],
        }
    return out


def build_prediction_targets(
    models: Sequence[str],
    fit_datasets: Sequence[str],
    holdout_datasets: Sequence[str],
    fit_layers: Sequence[int],
    prospective_layers: Sequence[int],
) -> List[Dict[str, Any]]:
    holdout_layers = sorted(set(int(x) for x in fit_layers).union(set(int(x) for x in prospective_layers)))
    targets: List[Dict[str, Any]] = []
    seen = set()

    for m in models:
        for ds in fit_datasets:
            for l in prospective_layers:
                key = (m, ds, int(l))
                if key in seen:
                    continue
                seen.add(key)
                targets.append({"model": m, "dataset": ds, "layer": int(l)})

        for ds in holdout_datasets:
            for l in holdout_layers:
                key = (m, ds, int(l))
                if key in seen:
                    continue
                seen.add(key)
                targets.append({"model": m, "dataset": ds, "layer": int(l)})

    targets.sort(key=lambda t: (t["model"], t["dataset"], t["layer"]))
    return targets


def create_frozen_predictions(
    freeze_path: Path,
    config: Dict[str, Any],
    universality_fit: Dict[str, Any],
    aggregated: Dict[str, Any],
    dataset_profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    selected_model = universality_fit["selected_model"]
    total_layers = int(config["total_layers"])
    fit_datasets = config["fit_datasets"]
    holdout_datasets = config["holdout_datasets"]
    models = config["models"]

    s12_predictors = build_s12_predictors(
        aggregated=aggregated,
        models=models,
        fit_datasets=fit_datasets,
        dataset_profiles=dataset_profiles,
        total_layers=total_layers,
    )

    targets = build_prediction_targets(
        models=models,
        fit_datasets=fit_datasets,
        holdout_datasets=holdout_datasets,
        fit_layers=config["fit_layers"],
        prospective_layers=config["prospective_layers"],
    )

    target_preds: List[Dict[str, Any]] = []
    for t in targets:
        model = t["model"]
        dataset = t["dataset"]
        layer = int(t["layer"])
        c_rel = float(layer / total_layers)

        d_pred, d_sd = predict_distortion(
            universality_fit=universality_fit,
            selected_model=selected_model,
            model=model,
            compute_c=c_rel,
        )
        if layer == total_layers:
            d_pred = 0.0

        s12_obs = (
            aggregated.get(model, {})
            .get(dataset, {})
            .get(str(total_layers), {})
            .get("steerability", {})
            .get("mean")
        )
        if dataset in fit_datasets and s12_obs is not None:
            s12_ref = float(s12_obs)
            s12_sd = (
                aggregated.get(model, {})
                .get(dataset, {})
                .get(str(total_layers), {})
                .get("steerability", {})
                .get("std")
            )
            s12_sd = 0.0 if s12_sd is None else float(s12_sd)
            s12_source = "observed_fit_layer12_mean"
        else:
            h = float(dataset_profiles[dataset]["h_l1_given_l0"])
            pred = s12_predictors.get(model, {"slope": 0.0, "intercept": 0.0, "residual_sd": float("nan")})
            s12_ref = float(pred["intercept"] + pred["slope"] * h)
            s12_sd = float(pred.get("residual_sd", float("nan")))
            s12_source = "predicted_from_h_l1_given_l0"

        s_pred = float(s12_ref * (1.0 - d_pred))

        d_ci = [float(d_pred - 1.96 * d_sd), float(d_pred + 1.96 * d_sd)] if np.isfinite(d_sd) else None

        s_var = 0.0
        if np.isfinite(s12_sd):
            s_var += ((1.0 - d_pred) ** 2) * (s12_sd ** 2)
        if np.isfinite(d_sd):
            s_var += (s12_ref ** 2) * (d_sd ** 2)
        s_sd = float(math.sqrt(max(s_var, 0.0))) if s_var > 0 else 0.0
        s_ci = [float(s_pred - 1.96 * s_sd), float(s_pred + 1.96 * s_sd)]

        tags = []
        if dataset in holdout_datasets:
            tags.append("holdout_dataset")
        if layer in config["prospective_layers"]:
            tags.append("prospective_layer")

        target_preds.append(
            {
                "model": model,
                "dataset": dataset,
                "layer": layer,
                "compute_proxy_relative": c_rel,
                "predicted_distortion": float(d_pred),
                "predicted_distortion_sd": float(d_sd),
                "prediction_interval_95_D": d_ci,
                "predicted_s12": float(s12_ref),
                "s12_source": s12_source,
                "predicted_steerability": float(s_pred),
                "predicted_steerability_sd": float(s_sd),
                "prediction_interval_95_S": s_ci,
                "tags": tags,
            }
        )

    payload = {
        "type": "cti_pilot_frozen_predictions",
        "created_at": utc_now_iso(),
        "script_version": SCRIPT_VERSION,
        "note": "Frozen BEFORE running target configs: all L={3,9} runs and all holdout AGNews runs.",
        "config_snapshot": {
            "models": config["models"],
            "fit_datasets": config["fit_datasets"],
            "holdout_datasets": config["holdout_datasets"],
            "fit_layers": config["fit_layers"],
            "prospective_layers": config["prospective_layers"],
            "seeds": config["seeds"],
            "total_layers": config["total_layers"],
        },
        "dataset_profiles": dataset_profiles,
        "universality_fit": universality_fit,
        "selected_law_model": selected_model,
        "s12_predictors": s12_predictors,
        "targets": target_preds,
    }
    atomic_json_dump(freeze_path, payload)
    return payload


def load_or_create_frozen_predictions(
    freeze_path: Path,
    config: Dict[str, Any],
    state: Dict[str, Any],
    universality_fit: Dict[str, Any],
    aggregated: Dict[str, Any],
    dataset_profiles: Dict[str, Dict[str, Any]],
    force: bool = False,
) -> Dict[str, Any]:
    if freeze_path.exists():
        log(f"Using existing frozen predictions: {freeze_path}")
        return load_json(freeze_path)

    if universality_fit.get("status") != "ok":
        raise RuntimeError("Cannot freeze predictions: universality fit is unavailable.")

    calib_specs = make_specs(
        config["models"],
        config["fit_datasets"],
        config["fit_layers"],
        config["seeds"],
    )
    if not specs_complete(state, calib_specs) and not force:
        raise RuntimeError(
            "Calibration runs are incomplete. Refusing to freeze predictions early."
        )

    log("Creating frozen predictions file before target runs...")
    return create_frozen_predictions(
        freeze_path=freeze_path,
        config=config,
        universality_fit=universality_fit,
        aggregated=aggregated,
        dataset_profiles=dataset_profiles,
    )


def validate_freeze_protocol(
    state: Dict[str, Any],
    freeze_path: Path,
    target_specs: Sequence[RunSpec],
    force: bool = False,
) -> None:
    completed_target_runs = []
    for spec in target_specs:
        payload = get_run(state, spec)
        if run_ok(payload):
            completed_target_runs.append((spec, payload))

    if completed_target_runs and not freeze_path.exists() and not force:
        example = completed_target_runs[0][0]
        raise RuntimeError(
            "Protocol violation: target runs already exist but frozen predictions file is missing. "
            f"Example completed target: {example.key()}"
        )

    if freeze_path.exists() and completed_target_runs and not force:
        frozen = load_json(freeze_path)
        freeze_time = parse_iso_datetime(frozen.get("created_at"))
        if freeze_time is None:
            log("Warning: could not parse freeze timestamp; skipping temporal protocol check.")
            return

        violated = []
        for spec, payload in completed_target_runs:
            run_time = parse_iso_datetime(payload.get("finished_at"))
            if run_time is not None and run_time <= freeze_time:
                violated.append(spec.key())

        if violated:
            raise RuntimeError(
                "Protocol violation: found target runs that finished before/at frozen prediction timestamp. "
                f"Examples: {violated[:3]}"
            )


def evaluate_frozen_predictions(frozen: Dict[str, Any], aggregated: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    residuals_s = []
    residuals_d = []
    in_pi_s = 0
    in_pi_d = 0
    n_pi_s = 0
    n_pi_d = 0

    for t in frozen.get("targets", []):
        model = t["model"]
        dataset = t["dataset"]
        layer = str(t["layer"])
        actual_obj = aggregated.get(model, {}).get(dataset, {}).get(layer, {})
        actual_s = actual_obj.get("steerability", {}).get("mean")
        actual_d = actual_obj.get("distortion", {}).get("mean")

        row = copy.deepcopy(t)
        row["actual_steerability_mean"] = actual_s
        row["actual_distortion_mean"] = actual_d

        if actual_s is not None:
            res_s = float(actual_s - t["predicted_steerability"])
            row["residual_S"] = res_s
            residuals_s.append(res_s)
            pi_s = t.get("prediction_interval_95_S")
            if isinstance(pi_s, list) and len(pi_s) == 2:
                n_pi_s += 1
                if pi_s[0] <= actual_s <= pi_s[1]:
                    in_pi_s += 1

        if actual_d is not None and t.get("predicted_distortion") is not None:
            res_d = float(actual_d - t["predicted_distortion"])
            row["residual_D"] = res_d
            residuals_d.append(res_d)
            pi_d = t.get("prediction_interval_95_D")
            if isinstance(pi_d, list) and len(pi_d) == 2:
                n_pi_d += 1
                if pi_d[0] <= actual_d <= pi_d[1]:
                    in_pi_d += 1

        rows.append(row)

    summary: Dict[str, Any] = {"n_targets": len(rows)}

    if residuals_s:
        arr = np.asarray(residuals_s, dtype=np.float64)
        summary["S"] = {
            "n": int(arr.size),
            "mae": float(np.mean(np.abs(arr))),
            "rmse": float(np.sqrt(np.mean(arr ** 2))),
            "bias": float(np.mean(arr)),
            "coverage_95": float(in_pi_s / n_pi_s) if n_pi_s > 0 else None,
            "n_with_pi": int(n_pi_s),
        }

    if residuals_d:
        arr = np.asarray(residuals_d, dtype=np.float64)
        summary["D"] = {
            "n": int(arr.size),
            "mae": float(np.mean(np.abs(arr))),
            "rmse": float(np.sqrt(np.mean(arr ** 2))),
            "bias": float(np.mean(arr)),
            "coverage_95": float(in_pi_d / n_pi_d) if n_pi_d > 0 else None,
            "n_with_pi": int(n_pi_d),
        }

    pred_s = [r["predicted_steerability"] for r in rows if r.get("actual_steerability_mean") is not None]
    act_s = [r["actual_steerability_mean"] for r in rows if r.get("actual_steerability_mean") is not None]
    if len(pred_s) >= 3 and "S" in summary:
        rho, p_rho = stats.spearmanr(pred_s, act_s)
        r, p_r = stats.pearsonr(pred_s, act_s)
        summary["S"]["spearman_rho"] = float(rho)
        summary["S"]["spearman_p"] = float(p_rho)
        summary["S"]["pearson_r"] = float(r)
        summary["S"]["pearson_p"] = float(p_r)

    return {"rows": rows, "summary": summary}


def execute_phase(
    phase_name: str,
    combos: Sequence[Tuple[str, str, int]],
    seeds: Sequence[int],
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    dataset_cache: DatasetCache,
    output_path: Path,
    rerun_errors: bool = False,
    fail_fast: bool = False,
    dry_run: bool = False,
) -> None:
    log(f"Starting phase '{phase_name}' with {len(combos)} model/dataset/layer combos")

    for combo_idx, (model, dataset, layer) in enumerate(combos, start=1):
        combo_key = f"{model}|{dataset}|L{layer}"
        log(f"[{phase_name}] combo {combo_idx}/{len(combos)}: {combo_key}")

        ok_count = 0
        err_count = 0
        completed_seeds: List[int] = []

        for seed in seeds:
            spec = RunSpec(model=model, dataset=dataset, layer=layer, seed=seed)
            existing = get_run(state, spec)

            if run_ok(existing):
                log(f"  seed={seed}: skip (already complete)")
                ok_count += 1
                completed_seeds.append(seed)
                continue

            if isinstance(existing, dict) and existing.get("status") == "error" and not rerun_errors:
                log(f"  seed={seed}: skip (previous error, rerun disabled)")
                err_count += 1
                continue

            if dry_run:
                log(f"  seed={seed}: dry-run planned")
                continue

            log(f"  seed={seed}: running")
            payload = run_single_config(spec=spec, cfg=cfg, dataset_cache=dataset_cache)
            set_run(state, spec, payload)

            if payload.get("status") == "ok":
                log(f"  seed={seed}: done S={payload['steerability']:+.5f}")
                ok_count += 1
                completed_seeds.append(seed)
            else:
                log(f"  seed={seed}: ERROR {payload.get('error', 'unknown')}")
                err_count += 1
                if fail_fast:
                    atomic_json_dump(output_path, state)
                    raise RuntimeError(f"Fail-fast enabled; aborting on {spec.key()}")

            state["updated_at"] = utc_now_iso()
            atomic_json_dump(output_path, state)

        state.setdefault("combo_progress", {}).setdefault(phase_name, {})[combo_key] = {
            "ok_count": int(ok_count),
            "error_count": int(err_count),
            "completed_seeds": sorted(completed_seeds),
            "updated_at": utc_now_iso(),
        }
        state["updated_at"] = utc_now_iso()
        atomic_json_dump(output_path, state)


def count_runs(state: Dict[str, Any]) -> Tuple[int, int]:
    total = 0
    ok = 0
    for m_dict in state.get("results", {}).values():
        for d_dict in m_dict.values():
            for l_dict in d_dict.values():
                for payload in l_dict.values():
                    total += 1
                    if run_ok(payload):
                        ok += 1
    return total, ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CTI pilot: layerwise compute-distortion law for steerability.")
    p.add_argument("--output", type=str, default="results/cti_pilot_results.json")
    p.add_argument("--frozen-predictions-path", type=str, default="results/cti_pilot_frozen_predictions.json")
    p.add_argument("--models", type=str, default="bge-base,e5-base,bge-small")
    p.add_argument("--fit-datasets", type=str, default="dbpedia_classes,clinc")
    p.add_argument("--holdout-datasets", type=str, default="agnews")
    p.add_argument("--fit-layers", type=str, default="2,4,6,8,10,12")
    p.add_argument("--prospective-layers", type=str, default="3,9")
    p.add_argument("--seeds", type=str, default="42,123,456")
    p.add_argument("--total-layers", type=int, default=12)
    p.add_argument("--stage1-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-train-samples", type=int, default=10000)
    p.add_argument("--max-test-samples", type=int, default=2000)
    p.add_argument("--max-eval-samples", type=int, default=500)
    p.add_argument("--num-scales", type=int, default=4)
    p.add_argument("--scale-dim", type=int, default=64)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--phase", choices=["all", "calibration", "targets", "analysis"], default="all")
    p.add_argument("--rerun-errors", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    models = parse_csv_str(args.models)
    fit_datasets = parse_csv_str(args.fit_datasets)
    holdout_datasets = parse_csv_str(args.holdout_datasets)
    fit_layers = sorted(set(parse_csv_int(args.fit_layers)))
    prospective_layers = sorted(set(parse_csv_int(args.prospective_layers)))
    seeds = [int(s) for s in parse_csv_int(args.seeds)]

    if args.total_layers not in fit_layers:
        raise ValueError(f"fit_layers must include total layer {args.total_layers} for D(L)=1-S(L)/S(L12).")
    if any(l <= 0 for l in fit_layers + prospective_layers):
        raise ValueError("All layers must be positive integers.")

    for m in models:
        if m not in MODELS:
            raise ValueError(f"Unknown model '{m}'.")
        if m in EXPECTED_MODEL_DIMS:
            got = int(MODELS[m].hidden_dim)
            exp = int(EXPECTED_MODEL_DIMS[m])
            if got != exp:
                raise ValueError(f"Model {m} hidden_dim mismatch: expected {exp}, got {got}")

    config = {
        "script": "cti_pilot.py",
        "script_version": SCRIPT_VERSION,
        "models": models,
        "fit_datasets": fit_datasets,
        "holdout_datasets": holdout_datasets,
        "fit_layers": fit_layers,
        "prospective_layers": prospective_layers,
        "seeds": seeds,
        "total_layers": int(args.total_layers),
        "stage1_epochs": int(args.stage1_epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "knn_k": int(args.knn_k),
        "patience": int(args.patience),
        "val_ratio": float(args.val_ratio),
        "max_train_samples": int(args.max_train_samples),
        "max_test_samples": int(args.max_test_samples),
        "max_eval_samples": int(args.max_eval_samples),
        "num_scales": int(args.num_scales),
        "scale_dim": int(args.scale_dim),
        "device": args.device,
    }

    output_path = Path(args.output)
    freeze_path = Path(args.frozen_predictions_path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if not freeze_path.is_absolute():
        freeze_path = REPO_ROOT / freeze_path

    if output_path.exists():
        state = load_json(output_path)
        old_cfg = state.get("config", {})
        ok_cfg, key = config_compatible(old_cfg, config)
        if not ok_cfg and not args.force:
            raise RuntimeError(f"Existing state config mismatch at '{key}'. Use --force to override.")
        if not ok_cfg and args.force:
            log(f"Warning: forcing config override (mismatch key: {key})")
            state["config"] = config
    else:
        state = {
            "meta": {
                "created_at": utc_now_iso(),
                "script": "cti_pilot.py",
                "script_version": SCRIPT_VERSION,
            },
            "config": config,
            "results": {},
            "combo_progress": {},
            "phase": {},
            "analysis": {},
        }
        atomic_json_dump(output_path, state)

    all_layers = sorted(set(fit_layers).union(set(prospective_layers)))
    all_datasets = sorted(set(fit_datasets).union(set(holdout_datasets)))

    calibration_specs = make_specs(models, fit_datasets, fit_layers, seeds)
    calibration_combos = make_combos(models, fit_datasets, fit_layers)

    target_combos_set = set(make_combos(models, fit_datasets, prospective_layers))
    holdout_layers = sorted(set(fit_layers).union(set(prospective_layers)))
    target_combos_set.update(make_combos(models, holdout_datasets, holdout_layers))
    target_combos = sorted(list(target_combos_set), key=lambda x: (x[0], x[1], x[2]))
    target_specs = [
        RunSpec(model=m, dataset=d, layer=l, seed=s)
        for (m, d, l) in target_combos
        for s in seeds
    ]

    validate_freeze_protocol(state, freeze_path, target_specs, force=args.force)

    dataset_cache = DatasetCache()
    dataset_profiles: Dict[str, Dict[str, Any]] = {}
    for ds in all_datasets:
        train_ds = dataset_cache.get(ds, "train", config["max_train_samples"])
        dataset_profiles[ds] = dataset_profile(ds, train_ds)

    if args.phase in ("all", "calibration"):
        execute_phase(
            phase_name="calibration",
            combos=calibration_combos,
            seeds=seeds,
            cfg=config,
            state=state,
            dataset_cache=dataset_cache,
            output_path=output_path,
            rerun_errors=args.rerun_errors,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
        )
        state.setdefault("phase", {})["calibration_completed"] = specs_complete(state, calibration_specs)
        state["updated_at"] = utc_now_iso()
        atomic_json_dump(output_path, state)

    aggregated = aggregate_results(
        state=state,
        models=models,
        datasets=all_datasets,
        layers=all_layers,
        seeds=seeds,
        total_layers=config["total_layers"],
    )
    fit_points = build_distortion_points(
        state=state,
        models=models,
        fit_datasets=fit_datasets,
        fit_layers=fit_layers,
        seeds=seeds,
        total_layers=config["total_layers"],
    )
    per_model_fit = fit_per_model_distortion(fit_points, models=models)
    universality_fit = fit_universality(fit_points, models=models) if fit_points else {"status": "insufficient_data", "n": 0}

    frozen_predictions = None
    if args.phase in ("all", "targets", "analysis"):
        frozen_predictions = load_or_create_frozen_predictions(
            freeze_path=freeze_path,
            config=config,
            state=state,
            universality_fit=universality_fit,
            aggregated=aggregated,
            dataset_profiles=dataset_profiles,
            force=args.force,
        )

    if args.phase in ("all", "targets"):
        execute_phase(
            phase_name="targets",
            combos=target_combos,
            seeds=seeds,
            cfg=config,
            state=state,
            dataset_cache=dataset_cache,
            output_path=output_path,
            rerun_errors=args.rerun_errors,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
        )
        state.setdefault("phase", {})["targets_completed"] = specs_complete(state, target_specs)
        state["updated_at"] = utc_now_iso()
        atomic_json_dump(output_path, state)

    aggregated = aggregate_results(
        state=state,
        models=models,
        datasets=all_datasets,
        layers=all_layers,
        seeds=seeds,
        total_layers=config["total_layers"],
    )
    fit_points = build_distortion_points(
        state=state,
        models=models,
        fit_datasets=fit_datasets,
        fit_layers=fit_layers,
        seeds=seeds,
        total_layers=config["total_layers"],
    )
    per_model_fit = fit_per_model_distortion(fit_points, models=models)
    universality_fit = fit_universality(fit_points, models=models) if fit_points else {"status": "insufficient_data", "n": 0}

    if frozen_predictions is None and freeze_path.exists():
        frozen_predictions = load_json(freeze_path)

    prediction_eval = evaluate_frozen_predictions(frozen_predictions, aggregated) if isinstance(frozen_predictions, dict) else None

    state["analysis"] = {
        "updated_at": utc_now_iso(),
        "dataset_profiles": dataset_profiles,
        "aggregated": aggregated,
        "fit_points_count": int(len(fit_points)),
        "fit_points_scope": {
            "datasets": fit_datasets,
            "layers": fit_layers,
            "seeds": seeds,
        },
        "per_model_distortion_fit": per_model_fit,
        "universality_fit": universality_fit,
        "prediction_evaluation": prediction_eval,
    }
    state["frozen_predictions_path"] = str(freeze_path)
    state["updated_at"] = utc_now_iso()

    total_runs, ok_runs = count_runs(state)
    state["run_counts"] = {"total_saved": int(total_runs), "ok": int(ok_runs)}
    state["completed"] = bool(specs_complete(state, calibration_specs) and specs_complete(state, target_specs))
    if state["completed"]:
        state["completed_at"] = utc_now_iso()

    atomic_json_dump(output_path, state)

    log(f"Saved state: {output_path}")
    log(f"Run count: {ok_runs}/{total_runs} successful")
    if universality_fit.get("status") == "ok":
        log(
            "Universality AIC: "
            f"shared={universality_fit['shared_alpha']['aic']:.3f}, "
            f"per_model={universality_fit['per_model_alpha']['aic']:.3f}, "
            f"selected={universality_fit['selected_model']}"
        )
    if prediction_eval and "summary" in prediction_eval:
        s_summary = prediction_eval["summary"].get("S", {})
        if s_summary:
            log(
                "Frozen prediction eval (S): "
                f"n={s_summary.get('n')}, "
                f"MAE={s_summary.get('mae'):.5f}, "
                f"RMSE={s_summary.get('rmse'):.5f}"
            )


if __name__ == "__main__":
    main()