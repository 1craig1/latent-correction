#!/usr/bin/env python3
"""Evaluate CL maps on held-out, paired VideoQA features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from latent_correction.alignment import METHODS
from latent_correction.diagnostics import evaluate
from same_input_probe import atomic_json, sha256_file


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    args = p.parse_args()
    with np.load(args.paired, allow_pickle=False) as bundle:
        data = {name: bundle[name] for name in bundle.files}
    report = evaluate(data, tuple(args.methods), ridge_alpha=args.ridge_alpha)
    report.update({"paired_sha256": sha256_file(args.paired),
                   "ridge_alpha": args.ridge_alpha,
                   "scope": "fixed NCM readout diagnostic; not generated VideoQA accuracy"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(f"Saved held-out diagnostic: {args.output}")
    for method, result in report["methods"].items():
        print(f"{method:>14}  drops={result['benchmark_drop_count']} "
              f"worst={result['worst_delta_pp']:+.1f} pp "
              f"macro={result['macro_delta_pp']:+.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
