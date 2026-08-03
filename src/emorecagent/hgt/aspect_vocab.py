"""Frozen aspect vocabulary from ABSA cache (read-only)."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..absa.normalize import normalize_aspect
from .schema import OTHER_ASPECT


@dataclass(frozen=True, slots=True)
class AspectVocab:
    """Canonical aspect list with stable ids; rare aspects map to ``OTHER_ASPECT``."""

    aspects: tuple[str, ...]
    other_id: int

    @property
    def size(self) -> int:
        return len(self.aspects)

    def id_for(self, raw_aspect: str) -> int:
        key = normalize_aspect(raw_aspect)
        try:
            return self.aspects.index(key)
        except ValueError:
            return self.other_id

    def to_dict(self) -> dict:
        return {"aspects": list(self.aspects), "other_id": self.other_id}

    @classmethod
    def from_dict(cls, data: dict) -> "AspectVocab":
        aspects = tuple(data["aspects"])
        other_id = int(data.get("other_id", aspects.index(OTHER_ASPECT)))
        return cls(aspects=aspects, other_id=other_id)


def open_absa_cache_readonly(path: str | Path) -> sqlite3.Connection:
    """Open the ABSA SQLite cache without write access."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"ABSA cache not found: {p}. Run `make absa` first."
        )
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def count_aspects_from_cache(
    cache_path: str | Path,
    *,
    min_support: int = 1,
) -> Counter[str]:
    """Scan cached triples and count canonical aspect strings."""
    conn = open_absa_cache_readonly(cache_path)
    counts: Counter[str] = Counter()
    try:
        rows = conn.execute("SELECT triples_json FROM absa_cache").fetchall()
    finally:
        conn.close()
    for (payload,) in rows:
        data = json.loads(payload)
        for t in data.get("triples", []):
            aspect = normalize_aspect(str(t.get("aspect", "")))
            if aspect:
                counts[aspect] += 1
    if min_support > 1:
        return Counter({a: c for a, c in counts.items() if c >= min_support})
    return counts


def build_aspect_vocab(
    cache_path: str | Path,
    *,
    top_k: int = 100,
    min_support: int = 5,
) -> AspectVocab:
    """Select top-supported aspects; append ``OTHER_ASPECT`` bucket."""
    counts = count_aspects_from_cache(cache_path, min_support=min_support)
    if not counts:
        raise ValueError(f"No aspects found in ABSA cache: {cache_path}")
    ranked = [a for a, _ in counts.most_common(top_k)]
    if OTHER_ASPECT not in ranked:
        ranked.append(OTHER_ASPECT)
    other_id = ranked.index(OTHER_ASPECT)
    return AspectVocab(aspects=tuple(ranked), other_id=other_id)


def save_aspect_vocab(vocab: AspectVocab, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab.to_dict(), indent=2), encoding="utf-8")
    return out


def load_aspect_vocab(path: str | Path) -> AspectVocab:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AspectVocab.from_dict(data)
