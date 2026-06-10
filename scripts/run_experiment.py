#!/usr/bin/env python3
"""Run a recommendation experiment over a processed split (U12).

Example:
    PYTHONPATH=src python3 scripts/run_experiment.py \
        --config configs/default.yaml --method svd \
        --split data/processed/Beauty_and_Personal_Care \
        --out results/svd.json

Methods runnable today: popularity, itemknn, svd (alias base_cf), sequential.
Aspect-aware and the full system require ABSA (U4) and the KG (U5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emorecagent.config import load_config
from emorecagent.eval.runner import (
    build_recommender,
    evaluate,
    load_split_jsonl,
    write_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EmoRecAgent experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--split",
        required=True,
        help="directory with train.jsonl / test.jsonl (from build_dataset.py)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--n-negatives",
        type=int,
        default=None,
        help="sampled-metric protocol; omit to rank the full catalog",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_dir = Path(args.split)
    train = load_split_jsonl(split_dir / "train.jsonl")
    test = load_split_jsonl(split_dir / "test.jsonl")

    recommender = build_recommender(
        args.method, {"factors": cfg.cf.factors}, seed=cfg.experiment.seed
    )
    result = evaluate(
        recommender,
        train,
        test,
        k_values=cfg.eval.k_values,
        method=args.method,
        n_negatives=args.n_negatives,
        seed=cfg.experiment.seed,
    )
    out = write_results(args.out, result)
    print(f"[run_experiment] {args.method}: {result.n_test_users} test users")
    for key in sorted(result.means):
        print(f"  {key}: {result.means[key]:.4f}")
    print(f"[run_experiment] results written to {out}")


if __name__ == "__main__":
    main()
