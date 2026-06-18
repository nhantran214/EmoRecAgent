#!/usr/bin/env python3
"""Verify Neo4j connectivity using .env credentials."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from emorecagent.config import ConfigError, load_config
from emorecagent.kg.schema import ensure_schema


def main() -> int:
    load_dotenv()
    try:
        cfg = load_config("configs/default.yaml")
    except ConfigError as exc:
        print(f"[verify_neo4j] config error: {exc}", file=sys.stderr)
        return 1

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print(
            "[verify_neo4j] install driver: pip install 'neo4j>=5.20,<6'",
            file=sys.stderr,
        )
        return 1

    uri = cfg.neo4j.uri
    auth = (cfg.neo4j.user, cfg.neo4j.password)
    print(f"[verify_neo4j] connecting to {uri} as {auth[0]} ...")

    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        with driver.session() as session:
            row = session.run(
                "CALL dbms.components() YIELD name, versions "
                "RETURN name, versions[0] AS version LIMIT 1"
            ).single()
            print(f"[verify_neo4j] server: {row['name']} {row['version']}")
            ping = session.run("RETURN 1 AS ok").single()["ok"]
            print(f"[verify_neo4j] bolt ping: {ping}")
            ensure_schema(driver)
            print("[verify_neo4j] schema constraints OK")
        driver.close()
    except Exception as exc:
        print(f"[verify_neo4j] FAILED: {exc}", file=sys.stderr)
        print(
            "[verify_neo4j] If Docker permission denied, run: "
            "bash scripts/setup_docker_neo4j.sh",
            file=sys.stderr,
        )
        return 1

    print("[verify_neo4j] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
