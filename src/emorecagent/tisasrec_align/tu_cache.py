"""$T_u$ preference text cache (JSONL)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TuCacheRow:
    user_id: str
    query_ts_ms: int
    T_u: str
    has_reviews: bool


def cache_key(user_id: str, query_ts_ms: int) -> str:
    return f"{user_id}|{query_ts_ms}"


def shard_tu_cache_path(path: str | Path, shard_id: int) -> Path:
    """Sidecar path for multi-process precompute: ``tu_cache.shard{id}.jsonl``."""
    p = Path(path)
    return p.parent / f"{p.stem}.shard{shard_id}{p.suffix}"


def load_tu_cache(path: str | Path) -> dict[str, TuCacheRow]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, TuCacheRow] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            row = TuCacheRow(
                user_id=str(d["user_id"]),
                query_ts_ms=int(d["query_ts_ms"]),
                T_u=str(d.get("T_u") or ""),
                has_reviews=bool(d.get("has_reviews", True)),
            )
            out[cache_key(row.user_id, row.query_ts_ms)] = row
    return out


def append_tu_cache(path: str | Path, row: TuCacheRow) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_tu_cache(path: str | Path, rows: dict[str, TuCacheRow]) -> None:
    """Rewrite cache file from a key→row map (deterministic key order)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for key in sorted(rows):
            fh.write(json.dumps(asdict(rows[key]), ensure_ascii=False) + "\n")
    tmp.replace(p)


def purge_tu_cache_keys(path: str | Path, keys: set[str]) -> int:
    """Remove ``keys`` from the cache file. Returns number of keys removed."""
    if not keys:
        return 0
    p = Path(path)
    if not p.exists():
        return 0
    existing = load_tu_cache(p)
    before = len(existing)
    for key in keys:
        existing.pop(key, None)
    removed = before - len(existing)
    if removed:
        write_tu_cache(p, existing)
    return removed


def merge_tu_cache_shards(
    path: str | Path,
    num_shards: int,
    *,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Append missing keys from ``*.shard{i}.jsonl`` into the main cache.

    When ``overwrite`` is true, shard rows replace existing keys (rewrite).

    Returns ``(added_or_replaced, total_keys_after)``.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    main = Path(path)
    existing = load_tu_cache(main)
    added = 0
    if overwrite:
        shard_rows: dict[str, TuCacheRow] = {}
        for shard_id in range(num_shards):
            shard_rows.update(load_tu_cache(shard_tu_cache_path(main, shard_id)))
        for key, row in shard_rows.items():
            if key in existing and existing[key] == row:
                continue
            existing[key] = row
            added += 1
        if added:
            write_tu_cache(main, existing)
        return added, len(existing)

    for shard_id in range(num_shards):
        shard_path = shard_tu_cache_path(main, shard_id)
        for key, row in load_tu_cache(shard_path).items():
            if key in existing:
                continue
            append_tu_cache(main, row)
            existing[key] = row
            added += 1
    return added, len(existing)
