#!/usr/bin/env python3
"""Precompute $T_u$ preference text cache for train/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emorecagent.agents.profiling_agent import DynamicUserProfilingAgent
from emorecagent.config import load_config
from emorecagent.eval.runner import load_split_jsonl
from emorecagent.llm.client import LLMClient
from emorecagent.tisasrec_align.absa_signal_source import AbsaCacheSignalSource
from emorecagent.tisasrec_align.item_metadata import load_item_metadata
from emorecagent.tisasrec_align.preference_text import (
    generate_preference_text,
    generate_preference_text_from_metadata,
)
from emorecagent.tisasrec_align.review_context import load_review_text_index
from emorecagent.tisasrec_align.tu_cache import (
    TuCacheRow,
    append_tu_cache,
    cache_key,
    load_tu_cache,
)
from emorecagent.utils.run_log import configure_run_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=("train", "test", "valid"), default="train")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="template-only T_u")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    logger, _ = configure_run_logging("precompute_tu_cache", log_dir=args.log_dir)
    cfg = load_config(args.config)
    ta = cfg.tisasrec_align
    base = Path(cfg.data.out_dir)

    train = load_split_jsonl(base / "train.jsonl")
    split_path = base / f"{args.split}.jsonl"
    if not split_path.exists():
        logger.error("split missing: %s", split_path)
        return 1
    target = load_split_jsonl(split_path)
    all_interactions = train + target

    existing = load_tu_cache(ta.tu_cache_path)
    use_metadata = ta.preference_source == "item_metadata"

    signal_source = None
    review_index: dict = {}
    profiling = None
    llm = None
    item_meta = None
    if use_metadata:
        meta_root = cfg.data.meta_path or cfg.data.inter_path or cfg.data.review_path
        item_meta = load_item_metadata(meta_root)
        logger.info(
            "preference_source=item_metadata items=%s",
            f"{len(item_meta):,}",
        )
    else:
        train_users = {it.user_id for it in train}
        signal_source = AbsaCacheSignalSource(
            cfg.absa.cache_path, cfg.data.review_path, train_users=train_users
        )
        review_index = load_review_text_index(cfg.data.review_path)
        profiling = DynamicUserProfilingAgent(signal_source, ta.lambda_decay)
        llm = None if args.no_llm else LLMClient.from_config(cfg)

    done = 0
    for it in target:
        key = cache_key(it.user_id, it.timestamp)
        if key in existing:
            continue
        if use_metadata:
            assert item_meta is not None
            result = generate_preference_text_from_metadata(
                user_id=it.user_id,
                query_ts_ms=it.timestamp,
                user_interactions=all_interactions,
                item_meta=item_meta,
            )
        else:
            assert signal_source is not None and profiling is not None
            result = generate_preference_text(
                user_id=it.user_id,
                query_ts_ms=it.timestamp,
                user_interactions=all_interactions,
                signal_source=signal_source,
                review_index=review_index,
                profiling=profiling,
                llm=llm,
                top_k_aspects=ta.top_k_aspects,
            )
        row = TuCacheRow(
            user_id=it.user_id,
            query_ts_ms=it.timestamp,
            T_u=result.T_u,
            has_reviews=result.has_reviews,
        )
        append_tu_cache(ta.tu_cache_path, row)
        existing[key] = row
        done += 1
        if args.max_rows is not None and done >= args.max_rows:
            break
        if done % 50 == 0:
            logger.info("precomputed %s rows", done)

    logger.info("wrote %s new rows -> %s", done, ta.tu_cache_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
