"""Unit tests for T_u cache shard helpers and history scoping."""

from __future__ import annotations

from pathlib import Path

from emorecagent.data.types import Interaction
from emorecagent.tisasrec_align.tu_cache import (
    TuCacheRow,
    append_tu_cache,
    cache_key,
    load_tu_cache,
    merge_tu_cache_shards,
    purge_tu_cache_keys,
    shard_tu_cache_path,
)
from emorecagent.tisasrec_align.tu_history import resolve_tu_history_interactions


def test_shard_tu_cache_path() -> None:
    p = Path("data/processed/Yelp/tisasrec_option_b/tu_cache.jsonl")
    assert shard_tu_cache_path(p, 0) == Path(
        "data/processed/Yelp/tisasrec_option_b/tu_cache.shard0.jsonl"
    )
    assert shard_tu_cache_path(p, 7).name == "tu_cache.shard7.jsonl"


def test_merge_tu_cache_shards(tmp_path: Path) -> None:
    main = tmp_path / "tu_cache.jsonl"
    append_tu_cache(
        main,
        TuCacheRow(user_id="u0", query_ts_ms=1, T_u="keep", has_reviews=True),
    )
    for sid, uid in ((0, "u1"), (1, "u2"), (0, "u3")):
        append_tu_cache(
            shard_tu_cache_path(main, sid),
            TuCacheRow(user_id=uid, query_ts_ms=10, T_u=uid, has_reviews=True),
        )
    # Duplicate key in shard1 should not overwrite / double-count.
    append_tu_cache(
        shard_tu_cache_path(main, 1),
        TuCacheRow(user_id="u0", query_ts_ms=1, T_u="dup", has_reviews=True),
    )

    added, total = merge_tu_cache_shards(main, num_shards=2)
    assert added == 3
    assert total == 4
    cache = load_tu_cache(main)
    assert cache[cache_key("u0", 1)].T_u == "keep"
    assert cache[cache_key("u1", 10)].T_u == "u1"
    assert cache[cache_key("u2", 10)].T_u == "u2"
    assert cache[cache_key("u3", 10)].T_u == "u3"


def test_merge_tu_cache_shards_overwrite(tmp_path: Path) -> None:
    main = tmp_path / "tu_cache.jsonl"
    append_tu_cache(
        main,
        TuCacheRow(user_id="u0", query_ts_ms=1, T_u="old", has_reviews=True),
    )
    append_tu_cache(
        shard_tu_cache_path(main, 0),
        TuCacheRow(user_id="u0", query_ts_ms=1, T_u="new", has_reviews=True),
    )
    added, total = merge_tu_cache_shards(main, num_shards=1, overwrite=True)
    assert added == 1
    assert total == 1
    assert load_tu_cache(main)[cache_key("u0", 1)].T_u == "new"


def test_purge_tu_cache_keys(tmp_path: Path) -> None:
    main = tmp_path / "tu_cache.jsonl"
    append_tu_cache(
        main, TuCacheRow(user_id="u0", query_ts_ms=1, T_u="a", has_reviews=True)
    )
    append_tu_cache(
        main, TuCacheRow(user_id="u1", query_ts_ms=2, T_u="b", has_reviews=True)
    )
    removed = purge_tu_cache_keys(main, {cache_key("u0", 1)})
    assert removed == 1
    cache = load_tu_cache(main)
    assert cache_key("u0", 1) not in cache
    assert cache[cache_key("u1", 2)].T_u == "b"


def test_resolve_tu_history_never_includes_test_split() -> None:
    train = [
        Interaction(user_id="u", item="i1", rating=5.0, timestamp=1, verified_purchase=True),
    ]
    valid = [
        Interaction(user_id="u", item="i2", rating=5.0, timestamp=2, verified_purchase=True),
    ]
    # Even if caller has test rows, resolve ignores them — only train / train+valid.
    hist = resolve_tu_history_interactions(
        split="test", train=train, valid=valid, test_history="train"
    )
    assert hist == train
    hist_tv = resolve_tu_history_interactions(
        split="test", train=train, valid=valid, test_history="train_valid"
    )
    assert hist_tv == train + valid
    hist_train = resolve_tu_history_interactions(
        split="train", train=train, valid=valid, test_history="train_valid"
    )
    assert hist_train == train
