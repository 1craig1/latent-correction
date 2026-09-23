#!/usr/bin/env python3
"""Pair enhanced-method source features with original-Qwen target features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from same_input_probe import atomic_json, sha256_file


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        required = ("id", "split", "benchmark", "gold", "video_sha256",
                    "processed_input_sha256", "feature")
        missing = [name for name in required if name not in bundle]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        return {name: bundle[name] for name in required}


def pair(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> tuple[dict, dict]:
    source_ids = source["id"].astype(str)
    target_ids = target["id"].astype(str)
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("Duplicate item ID")
    if set(source_ids) != set(target_ids):
        raise ValueError("Source and target item ID sets differ")
    source_index = {item_id: index for index, item_id in enumerate(source_ids)}
    order = np.array([source_index[item_id] for item_id in target_ids])
    for field in ("split", "benchmark", "gold", "video_sha256"):
        if not np.array_equal(source[field][order], target[field]):
            raise ValueError(f"Paired item metadata differs: {field}")
    if source["feature"].shape != target["feature"].shape:
        raise ValueError("Paired feature shapes differ")
    splits = target["split"].astype(str)
    if set(splits) != {"anchor", "eval"}:
        raise ValueError("Both anchor and eval splits are required")
    video_hashes = target["video_sha256"].astype(str)
    if set(video_hashes[splits == "anchor"]) & set(video_hashes[splits == "eval"]):
        raise ValueError("A video appears in both anchor and eval splits")
    result = {}
    for split in ("anchor", "eval"):
        mask = splits == split
        result[f"{split}_id"] = target["id"][mask]
        result[f"{split}_source"] = source["feature"][order][mask]
        result[f"{split}_target"] = target["feature"][mask]
        result[f"{split}_label"] = target["gold"][mask]
        if split == "eval":
            result["eval_benchmark"] = target["benchmark"][mask]
    equal_input = source["processed_input_sha256"][order] == target["processed_input_sha256"]
    return result, {"n_items": len(target_ids), "n_anchors": int((splits == "anchor").sum()),
                    "n_eval": int((splits == "eval").sum()),
                    "identical_processed_input_fraction": float(equal_input.mean())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True,
                   help="Method-enhanced Qwen features.npz")
    p.add_argument("--target", type=Path, required=True,
                   help="Original Qwen baseline features.npz")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data, report = pair(load_features(args.source), load_features(args.target))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(args.output)
    report.update({"source_sha256": sha256_file(args.source),
                   "target_sha256": sha256_file(args.target),
                   "direction": "source(method) -> target(original baseline)"})
    atomic_json(args.output.with_suffix(".json"), report)
    print(f"Paired {report['n_items']} items: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
