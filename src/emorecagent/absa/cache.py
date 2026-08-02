"""SQLite cache for ABSA triples keyed by review id."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..llm.schemas import TripleSet


class AbsaCache:
    def __init__(self, path: str | Path, *, readonly: bool = False) -> None:
        self.path = Path(path)
        self.readonly = readonly
        self._lock = threading.Lock()
        if readonly:
            if not self.path.is_file():
                raise FileNotFoundError(f"ABSA cache not found (read-only): {self.path}")
            self._conn = sqlite3.connect(
                f"file:{self.path.resolve()}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS absa_cache (
                    review_id TEXT PRIMARY KEY,
                    triples_json TEXT NOT NULL,
                    updated_at INTEGER DEFAULT (strftime('%s','now'))
                )
                """
            )
            self._conn.commit()

    def get(self, review_id: str) -> TripleSet | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT triples_json FROM absa_cache WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return TripleSet.model_validate(data)

    def put(self, review_id: str, triples: TripleSet) -> None:
        if self.readonly:
            raise RuntimeError("AbsaCache is read-only")
        payload = triples.model_dump_json()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO absa_cache (review_id, triples_json)
                VALUES (?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    triples_json = excluded.triples_json,
                    updated_at = strftime('%s','now')
                """,
                (review_id, payload),
            )
            self._conn.commit()

    def contains(self, review_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM absa_cache WHERE review_id = ? LIMIT 1",
                (review_id,),
            ).fetchone()
        return row is not None

    def delete(self, review_id: str) -> bool:
        if self.readonly:
            raise RuntimeError("AbsaCache is read-only")
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM absa_cache WHERE review_id = ?",
                (review_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def iter_all(self) -> list[tuple[str, TripleSet]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT review_id, triples_json FROM absa_cache ORDER BY review_id"
            ).fetchall()
        return [(rid, TripleSet.model_validate(json.loads(payload))) for rid, payload in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
