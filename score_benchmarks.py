#!/usr/bin/env python3
"""Measure cross-benchmark negative transfer from actual answer records.

Predictions JSONL: one {"id": ..., "method": ..., "answer": "A"} per
item/method. The manifest supplies benchmark and gold. The baseline method
must be named "baseline" and every method must cover exactly the same items.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from same_input_probe import atomic_json, read_jsonl


def score(manifest: list[dict], predictions: list[dict]) -> dict:
    items = {}
    for row in manifest:
        item_id = str(row["id"])
        if item_id in items:
            raise ValueError(f"Duplicate manifest ID: {item_id}")
        gold = str(row["gold"]).upper()
        if gold not in "ABCDE" or not row.get("benchmark"):
            raise ValueError(f"{item_id}: valid gold and benchmark required")
        items[item_id] = (str(row["benchmark"]), gold)
    if not items:
        raise ValueError("Manifest is empty")
    answers = defaultdict(dict)
    for row in predictions:
        item_id, method = str(row["id"]), str(row["method"])
        if item_id not in items or not method:
            raise ValueError(f"Unknown item or empty method: {item_id}")
        if item_id in answers[method]:
            raise ValueError(f"Duplicate prediction: {method}/{item_id}")
        answers[method][item_id] = str(row["answer"]).strip().upper()
    if "baseline" not in answers:
        raise ValueError("Predictions need a baseline method")
    for method, rows in answers.items():
        if set(rows) != set(items):
            raise ValueError(f"{method}: predictions must cover every manifest item")
    benchmarks = sorted({value[0] for value in items.values()})
    result = {"n_items": len(items), "benchmarks": {}, "methods": {}}
    for name in benchmarks:
        ids = [item_id for item_id, (benchmark, _) in items.items() if benchmark == name]
        result["benchmarks"][name] = {"n": len(ids)}
        for method, rows in answers.items():
            accuracy = sum(rows[item_id] == items[item_id][1] for item_id in ids) / len(ids)
            result["benchmarks"][name][method] = accuracy
    for method, rows in answers.items():
        if method == "baseline":
            continue
        deltas = {name: 100 * (entry[method] - entry["baseline"])
                  for name, entry in result["benchmarks"].items()}
        flips = {"gained": 0, "lost": 0}
        for item_id, (_, gold) in items.items():
            base_correct = answers["baseline"][item_id] == gold
            method_correct = rows[item_id] == gold
            flips["gained"] += int(not base_correct and method_correct)
            flips["lost"] += int(base_correct and not method_correct)
        result["methods"][method] = {
            "delta_pp_by_benchmark": deltas,
            "benchmark_drop_count": sum(delta < 0 for delta in deltas.values()),
            "worst_delta_pp": min(deltas.values()),
            "macro_delta_pp": sum(deltas.values()) / len(deltas),
            "item_flips": flips,
        }
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = score(read_jsonl(args.manifest), read_jsonl(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    for name, row in report["methods"].items():
        print(f"{name}: drops={row['benchmark_drop_count']} "
              f"worst={row['worst_delta_pp']:+.2f} pp "
              f"macro={row['macro_delta_pp']:+.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
