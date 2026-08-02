#!/usr/bin/env python3
"""Sample reviews for ABSA gold labeling."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from emorecagent.config import load_config
from emorecagent.eval.runner import load_split_jsonl


def _review_id(row: dict) -> str:
    return str(row.get("review_id") or row.get("asin") or "")


def _rating_bucket(rating: float) -> str:
    if rating <= 2:
        return "low"
    if rating <= 3:
        return "mid"
    return "high"


def _length_bucket(text: str, quartiles: tuple[int, int, int]) -> str:
    n = len(text.split())
    q1, q2, q3 = quartiles
    if n <= q1:
        return "short"
    if n <= q2:
        return "medium"
    if n <= q3:
        return "long"
    return "very_long"


def _load_train_pairs(train_path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if not train_path.exists():
        return pairs
    for row in load_split_jsonl(train_path):
        pairs.add((row.user_id, row.item))
    return pairs


def _stream_reviews(path: Path, train_pairs: set[tuple[str, str]], train_only: bool):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if not text:
                continue
            rid = _review_id(row)
            if not rid:
                continue
            user_id = str(row.get("user_id") or "")
            item_id = str(row.get("parent_asin") or row.get("asin") or "")
            if train_only and (user_id, item_id) not in train_pairs:
                continue
            yield {
                "review_id": rid,
                "user_id": user_id,
                "parent_asin": item_id,
                "text": text,
                "rating": float(row.get("rating", 0.0)),
            }


def _stratified_sample(
    rows: list[dict],
    n: int,
    rng: random.Random,
    stratify_length: bool,
) -> list[dict]:
    if not rows or n <= 0:
        return []
    if len(rows) <= n:
        return rows

    buckets: dict[str, list[dict]] = defaultdict(list)
    lengths = [len(r["text"].split()) for r in rows]
    lengths_sorted = sorted(lengths)
    q1 = lengths_sorted[len(lengths_sorted) // 4]
    q2 = lengths_sorted[len(lengths_sorted) // 2]
    q3 = lengths_sorted[(3 * len(lengths_sorted)) // 4]

    for row in rows:
        rating_key = _rating_bucket(row["rating"])
        if stratify_length:
            key = f"{rating_key}:{_length_bucket(row['text'], (q1, q2, q3))}"
        else:
            key = rating_key
        buckets[key].append(row)

    keys = sorted(buckets)
    per_bucket = max(1, n // len(keys))
    selected: list[dict] = []
    for key in keys:
        pool = buckets[key]
        rng.shuffle(pool)
        selected.extend(pool[:per_bucket])

    if len(selected) < n:
        remaining = [r for r in rows if r not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])
    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample ABSA gold candidates.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/labeled/absa_gold_candidates.jsonl")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--pilot", type=int, default=None, help="pilot subset size")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.experiment.seed
    n_samples = args.pilot or args.n_samples or cfg.absa.gold_n_samples
    rng = random.Random(seed)

    review_path = Path(cfg.data.review_path)
    train_path = Path(cfg.data.out_dir) / "train.jsonl"
    train_pairs = _load_train_pairs(train_path)

    rows = list(
        _stream_reviews(
            review_path,
            train_pairs,
            train_only=cfg.absa.gold_train_only,
        )
    )
    selected = _stratified_sample(
        rows,
        n_samples,
        rng,
        stratify_length=cfg.absa.gold_stratify_length,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(
                json.dumps(
                    {
                        "review_id": row["review_id"],
                        "user_id": row["user_id"],
                        "parent_asin": row["parent_asin"],
                        "text": row["text"],
                        "rating": row["rating"],
                        "triples": [],
                    }
                )
                + "\n"
            )
    print(f"[sample_absa_gold] wrote {len(selected)} candidates to {out}")


if __name__ == "__main__":
    main()
