#!/usr/bin/env python3
"""Hydrate Neo4j from train split interactions and ABSA cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from emorecagent.config import ConfigError, load_config
from emorecagent.kg.loaders import load_train_kg
from emorecagent.kg.schema import ensure_schema


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Load train KG into Neo4j.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--split",
        default=None,
        help="processed split directory (default: data.out_dir from config)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete all nodes before loading",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[load_kg] config error: {exc}", file=sys.stderr)
        return 1

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print(
            "[load_kg] install driver: pip install 'neo4j>=5.20,<6'",
            file=sys.stderr,
        )
        return 1

    split_dir = Path(args.split or cfg.data.out_dir)
    train_path = split_dir / "train.jsonl"
    if not train_path.exists():
        print(f"[load_kg] missing train split: {train_path}", file=sys.stderr)
        return 1

    cache_path = Path(cfg.absa.cache_path)
    if not cache_path.exists():
        print(
            f"[load_kg] ABSA cache not found: {cache_path} "
            "(run make absa first)",
            file=sys.stderr,
        )
        return 1

    uri = cfg.neo4j.uri
    auth = (cfg.neo4j.user, cfg.neo4j.password)
    print(f"[load_kg] connecting to {uri} ...")
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        with driver.session() as session:
            ensure_schema(driver)
            if args.fresh:
                session.run("MATCH (n) DETACH DELETE n")
                print("[load_kg] cleared existing graph (--fresh)")

        from emorecagent.kg.repository import KGRepository

        repo = KGRepository(driver)
        stats = load_train_kg(
            repo,
            train_path=train_path,
            raw_review_path=cfg.data.review_path,
            cache_path=cache_path,
            helpful_cap=cfg.scoring.helpful_vote_cap,
        )
        print(json.dumps({"status": "ok", **stats}, indent=2))
        if stats["reviews_with_triples"] == 0:
            print(
                "[load_kg] WARNING: no ABSA triples written — SIGNAL edges missing. "
                "Run make absa first, then make load-kg -- --fresh",
                file=sys.stderr,
            )
            return 1
    except Exception as exc:
        print(f"[load_kg] FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        driver.close()

    print("[load_kg] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
