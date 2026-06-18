"""Neo4j constraints and indexes.

Idempotent: safe to re-run on an existing database.
"""

from __future__ import annotations

from typing import Any

SCHEMA_STATEMENTS: list[str] = [
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT item_asin IF NOT EXISTS FOR (i:Item) REQUIRE i.asin IS UNIQUE",
    "CREATE CONSTRAINT aspect_name IF NOT EXISTS FOR (a:Aspect) REQUIRE a.name IS UNIQUE",
    "CREATE INDEX reviewed_ts IF NOT EXISTS FOR ()-[r:REVIEWED]-() ON (r.ts)",
    "CREATE INDEX has_sentiment_score IF NOT EXISTS FOR ()-[r:HAS_SENTIMENT]-() ON (r.score)",
]


def ensure_schema(driver: Any) -> None:
    with driver.session() as session:
        for stmt in SCHEMA_STATEMENTS:
            session.run(stmt)
