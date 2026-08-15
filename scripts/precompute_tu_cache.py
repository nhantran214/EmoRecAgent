#!/usr/bin/env python3
"""Precompute $T_u$ preference text cache for train/test/valid splits.

History for $T_u$ follows Protocol B / ``tisasrec_align.test_history``:
train-only (or train+valid), **never** the eval split itself — otherwise
earlier test items leak into the manifesto under ``user_batch``.

Supports multi-process sharding: each process owns ``idx % num_shards == shard_id``
and writes ``tu_cache.shard{id}.jsonl``. Merge shards into the main cache after::

  python3 scripts/precompute_tu_cache.py --config CFG --merge-shards --num-shards N

Recompute leaked test keys (after upgrading this script)::

  python3 scripts/precompute_tu_cache.py --config CFG --split test --no-llm --overwrite
"""

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
    merge_tu_cache_shards,
    purge_tu_cache_keys,
    shard_tu_cache_path,
)
from emorecagent.tisasrec_align.tu_history import (
    interaction_keys,
    resolve_tu_history_interactions,
)
from emorecagent.utils.run_log import configure_run_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=("train", "test", "valid"), default="train")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="template-only T_u")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total shard count (default: 1 = write directly to tu_cache_path)",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="this process shard id in [0, num_shards)",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="merge tu_cache.shard*.jsonl into tu_cache_path and exit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute keys for this split (purge existing keys first)",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        print("--num-shards must be >= 1", file=sys.stderr)
        return 2
    if not (0 <= args.shard_id < args.num_shards):
        print(
            f"--shard-id must be in [0, {args.num_shards})",
            file=sys.stderr,
        )
        return 2

    logger, _ = configure_run_logging("precompute_tu_cache", log_dir=args.log_dir)
    cfg = load_config(args.config)
    ta = cfg.tisasrec_align
    cache_path = Path(ta.tu_cache_path)

    if args.merge_shards:
        added, total = merge_tu_cache_shards(
            cache_path, args.num_shards, overwrite=args.overwrite
        )
        logger.info(
            "merged shards=0..%s added_or_replaced=%s total_keys=%s overwrite=%s -> %s",
            args.num_shards - 1,
            added,
            total,
            args.overwrite,
            cache_path,
        )
        return 0

    base = Path(cfg.data.out_dir)

    train = load_split_jsonl(base / "train.jsonl")
    split_path = base / f"{args.split}.jsonl"
    if not split_path.exists():
        logger.error("split missing: %s", split_path)
        return 1
    target = load_split_jsonl(split_path)

    valid = None
    valid_path = base / "valid.jsonl"
    if valid_path.exists():
        valid = load_split_jsonl(valid_path)

    history = resolve_tu_history_interactions(
        split=args.split,
        train=train,
        valid=valid,
        test_history=ta.test_history,
    )
    history_keys = interaction_keys(history)
    logger.info(
        "T_u history scope: split=%s test_history=%s history_rows=%s "
        "(Protocol B: never includes eval-split interactions)",
        args.split,
        ta.test_history,
        f"{len(history):,}",
    )

    # Single-shard keeps legacy path; multi-shard writes sidecar files.
    out_path = (
        cache_path
        if args.num_shards == 1
        else shard_tu_cache_path(cache_path, args.shard_id)
    )

    assigned_keys = {
        cache_key(it.user_id, it.timestamp)
        for idx, it in enumerate(target)
        if idx % args.num_shards == args.shard_id
    }
    if args.overwrite:
        if args.num_shards == 1:
            removed = purge_tu_cache_keys(cache_path, assigned_keys)
            logger.info("overwrite: purged %s keys from %s", removed, cache_path)
        else:
            # Shard workers must not skip keys that still sit in the main cache;
            # otherwise overwrite writes 0 rows and merge is a no-op (leak remains).
            if out_path.exists():
                out_path.unlink()
                logger.info("overwrite: truncated shard file %s", out_path)
            if args.shard_id == 0:
                # Drop stale split keys from main so a mid-run eval cannot reuse them.
                all_split_keys = {
                    cache_key(it.user_id, it.timestamp) for it in target
                }
                removed = purge_tu_cache_keys(cache_path, all_split_keys)
                logger.info(
                    "overwrite: purged %s split keys from main cache %s",
                    removed,
                    cache_path,
                )

    # Skip keys already in main cache or this shard (resume-safe).
    existing = load_tu_cache(cache_path)
    if out_path != cache_path:
        existing.update(load_tu_cache(out_path))
    if args.overwrite:
        # Ensure this shard recomputes its assigned keys even if purge raced.
        for key in assigned_keys:
            existing.pop(key, None)
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
        train_users = {it.user_id for it in history}
        signal_source = AbsaCacheSignalSource(
            cfg.absa.cache_path,
            cfg.data.review_path,
            train_users=train_users,
            allowed_reviews=history_keys,
        )
        review_index = load_review_text_index(cfg.data.review_path)
        profiling = DynamicUserProfilingAgent(signal_source, ta.lambda_decay)
        llm = None if args.no_llm else LLMClient.from_config(cfg)

    assigned = len(assigned_keys)
    logger.info(
        "split=%s rows=%s shard=%s/%s assigned=%s out=%s existing_keys=%s overwrite=%s",
        args.split,
        len(target),
        args.shard_id,
        args.num_shards,
        assigned,
        out_path,
        len(existing),
        args.overwrite,
    )

    done = 0
    skipped = 0
    for idx, it in enumerate(target):
        if idx % args.num_shards != args.shard_id:
            continue
        key = cache_key(it.user_id, it.timestamp)
        if key in existing:
            skipped += 1
            continue
        if use_metadata:
            assert item_meta is not None
            result = generate_preference_text_from_metadata(
                user_id=it.user_id,
                query_ts_ms=it.timestamp,
                user_interactions=history,
                item_meta=item_meta,
            )
        else:
            assert signal_source is not None and profiling is not None
            result = generate_preference_text(
                user_id=it.user_id,
                query_ts_ms=it.timestamp,
                user_interactions=history,
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
        append_tu_cache(out_path, row)
        existing[key] = row
        done += 1
        if args.max_rows is not None and done >= args.max_rows:
            break
        if done % 50 == 0:
            logger.info(
                "precomputed %s rows (shard %s/%s)",
                done,
                args.shard_id,
                args.num_shards,
            )

    logger.info(
        "wrote %s new rows (skipped_existing=%s) -> %s",
        done,
        skipped,
        out_path,
    )
    if args.num_shards > 1:
        merge_extra = " --overwrite" if args.overwrite else ""
        logger.info(
            "when all shards finish, merge with:\n"
            "  python3 scripts/precompute_tu_cache.py --config %s "
            "--merge-shards --num-shards %s%s --log-dir %s",
            args.config,
            args.num_shards,
            merge_extra,
            args.log_dir,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
