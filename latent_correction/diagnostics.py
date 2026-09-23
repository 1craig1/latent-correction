"""Held-out paired-feature and fixed-readout diagnostics."""

from __future__ import annotations

import numpy as np

from .alignment import METHODS, fit_map


REQUIRED = ("anchor_id", "anchor_source", "anchor_target", "anchor_label",
            "eval_id", "eval_source", "eval_target", "eval_label", "eval_benchmark")


def _strs(value: np.ndarray) -> np.ndarray:
    return np.asarray(value).astype(str)


def validate(data: dict[str, np.ndarray]) -> None:
    missing = [name for name in REQUIRED if name not in data]
    if missing:
        raise ValueError(f"Missing arrays: {missing}")
    anchors = np.asarray(data["anchor_source"])
    target = np.asarray(data["anchor_target"])
    evaluation = np.asarray(data["eval_source"])
    eval_target = np.asarray(data["eval_target"])
    if anchors.ndim != 2 or anchors.shape != target.shape:
        raise ValueError("anchor_source and anchor_target must be paired 2D arrays")
    if evaluation.ndim != 2 or evaluation.shape != eval_target.shape:
        raise ValueError("eval_source and eval_target must be paired 2D arrays")
    if anchors.shape[1] != evaluation.shape[1] or not len(anchors) or not len(evaluation):
        raise ValueError("Anchor/eval dimensions must match and splits must be nonempty")
    for prefix, count, fields in (
        ("anchor", len(anchors), ("id", "label")),
        ("eval", len(evaluation), ("id", "label", "benchmark")),
    ):
        for field in fields:
            if len(data[f"{prefix}_{field}"]) != count:
                raise ValueError(f"{prefix}_{field} length differs from its features")
    anchor_ids, eval_ids = _strs(data["anchor_id"]), _strs(data["eval_id"])
    if len(set(anchor_ids)) != len(anchor_ids) or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("Item IDs must be unique within each split")
    if set(anchor_ids) & set(eval_ids):
        raise ValueError("Anchor and evaluation item IDs overlap")
    if not set(_strs(data["eval_label"])).issubset(set(_strs(data["anchor_label"]))):
        raise ValueError("Some evaluation labels have no anchor example")
    if any(not value for value in _strs(data["eval_benchmark"])):
        raise ValueError("Each evaluation row needs a benchmark")
    for name in ("anchor_source", "anchor_target", "eval_source", "eval_target"):
        if not np.isfinite(data[name]).all():
            raise ValueError(f"{name} contains nonfinite values")


def ncm_predict(features: np.ndarray, prototypes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    distances = np.sum((features[:, None, :] - prototypes[None, :, :]) ** 2, axis=-1)
    return classes[distances.argmin(axis=1)]


def evaluate(data: dict[str, np.ndarray], methods: tuple[str, ...] = METHODS,
             ridge_alpha: float = 1.0) -> dict:
    """Compare operators using a *fixed* NCM readout trained on base anchors."""
    validate(data)
    source = np.asarray(data["anchor_source"], dtype=np.float64)
    target = np.asarray(data["anchor_target"], dtype=np.float64)
    eval_source = np.asarray(data["eval_source"], dtype=np.float64)
    eval_target = np.asarray(data["eval_target"], dtype=np.float64)
    anchor_label = _strs(data["anchor_label"])
    eval_label = _strs(data["eval_label"])
    benchmark = _strs(data["eval_benchmark"])
    classes = np.unique(anchor_label)
    prototypes = np.stack([target[anchor_label == label].mean(axis=0) for label in classes])

    def scores(features: np.ndarray) -> dict:
        predicted = ncm_predict(features, prototypes, classes)
        correct = predicted == eval_label
        return {name: {"n": int(mask.sum()), "accuracy": float(correct[mask].mean())}
                for name in sorted(set(benchmark))
                for mask in (benchmark == name,)}

    base_scores = scores(eval_target)
    result = {"n_anchors": len(source), "n_eval": len(eval_source),
              "feature_dimension": source.shape[1],
              "readout": "nearest class mean fitted on baseline anchor features",
              "baseline_readout_by_benchmark": base_scores, "methods": {}}
    for method in methods:
        fitted = fit_map(source, target, method, ridge_alpha=ridge_alpha)
        mapped = fitted.transform(eval_source)
        denominator = float(np.sum((eval_target - target.mean(axis=0)) ** 2))
        mapped_norm = np.linalg.norm(mapped, axis=1)
        target_norm = np.linalg.norm(eval_target, axis=1)
        cosine = np.sum(mapped * eval_target, axis=1) / np.maximum(
            mapped_norm * target_norm, 1e-12
        )
        per_benchmark = scores(mapped)
        deltas = {name: 100 * (item["accuracy"] - base_scores[name]["accuracy"])
                  for name, item in per_benchmark.items()}
        result["methods"][method] = {
            "relative_feature_mse": float(np.sum((mapped - eval_target) ** 2) /
                                          max(denominator, 1e-12)),
            "mean_cosine_to_baseline": float(cosine.mean()),
            "readout_by_benchmark": per_benchmark,
            "delta_to_baseline_pp": deltas,
            "benchmark_drop_count": sum(value < 0 for value in deltas.values()),
            "worst_delta_pp": min(deltas.values()),
            "macro_delta_pp": float(np.mean(list(deltas.values()))),
        }
    return result
