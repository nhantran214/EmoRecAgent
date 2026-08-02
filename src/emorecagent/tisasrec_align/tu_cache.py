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
