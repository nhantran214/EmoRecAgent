#!/usr/bin/env python3
"""Run offline ABSA extraction over review JSONL with caching (U4).

Reads review text from the raw category file (or a JSONL with review_id+text),
runs extract→judge, and writes/updates the SQLite cache. Re-runs are near-instant
on cache hits.

Example:
    PYTHONPATH=src python3 scripts/run_absa.py \
        --config configs/default.yaml \
        --reviews data/amazon-reviews-2023/raw/review_categories/Beauty_and_Personal_Care.jsonl \
        --max-reviews 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from emorecagent.absa.cache import AbsaCache
from emorecagent.absa.extractor import AbsaExtractor
from emorecagent.absa.judge import AbsaJudge
from emorecagent.absa.pipeline import AbsaPipeline, ReviewRecord
from emorecagent.config import load_config
from emorecagent.llm.client import LLMClient


def _review_id(row: dict) -> str:
    return str(row.get("review_id") or row.get("asin") or row.get("user_id", ""))


def stream_reviews(path: Path, max_reviews: int | None):
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if max_reviews is not None and n >= max_reviews:
                break
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
            yield ReviewRecord(review_id=rid, text=text)
            n += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ABSA extraction with cache.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--reviews", required=True, help="source review JSONL")
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = LLMClient.from_config(cfg)
    cache = AbsaCache(cfg.absa.cache_path)
    pipeline = AbsaPipeline(
        AbsaExtractor(client),
        AbsaJudge(client, min_confidence=cfg.absa.min_confidence),
        cache=cache,
    )

    hits = processed = 0
    for rec in tqdm(
        stream_reviews(Path(args.reviews), args.max_reviews),
        desc="ABSA",
    ):
        if not args.no_cache and cache.contains(rec.review_id):
            hits += 1
            continue
        pipeline.process(rec, use_cache=not args.no_cache)
        processed += 1

    print(
        f"[run_absa] cache_hits={hits} newly_processed={processed} "
        f"cache_path={cfg.absa.cache_path}"
    )
    cache.close()


if __name__ == "__main__":
    main()
