#!/usr/bin/env python3
"""Build train-scoped heterogeneous graph for HGT (read-only ABSA cache)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emorecagent.config import load_config
from emorecagent.hgt.graph_builder import build_and_save_hgt_graph
from emorecagent.utils.run_log import configure_run_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    logger, log_path = configure_run_logging(
        "build_hgt_graph",
        log_dir=args.log_dir,
    )
    cfg = load_config(args.config)
    split_dir = Path(cfg.data.out_dir)
    train_path = split_dir / "train.jsonl"
    valid_path = split_dir / "valid.jsonl"
    cache_path = Path(cfg.absa.cache_path)

    if not train_path.exists():
        logger.error("train split missing: %s (run make data)", train_path)
        return 1
    if not cache_path.exists():
        logger.error("ABSA cache missing: %s (run make absa)", cache_path)
        return 1

    logger.info("log file: %s", log_path.resolve())
    logger.info(
        "building HGT graph train=%s cache=%s (read-only)",
        train_path,
        cache_path,
    )

    stats = build_and_save_hgt_graph(
        train_path=train_path,
        valid_path=valid_path if valid_path.exists() else None,
        cache_path=cache_path,
        meta_path=cfg.data.meta_path,
        raw_review_path=cfg.data.review_path,
        graph_path=cfg.hgt.graph_path,
        aspect_vocab_path=cfg.hgt.aspect_vocab_path,
        aspect_top_k=cfg.hgt.aspect_top_k,
        min_aspect_support=cfg.absa.min_aspect_support,
        text_encoder=cfg.hgt.text_encoder,
        feature_dim=cfg.hgt.feature_dim,
        seed=cfg.experiment.seed,
        lambda_decay=cfg.scoring.lambda_decay,
        max_users=args.max_users,
        max_items=args.max_items,
    )
    logger.info(
        "graph saved: users=%s items=%s aspects=%s edges=%s train_pairs=%s valid_pairs=%s",
        stats.n_users,
        stats.n_items,
        stats.n_aspects,
        stats.n_edges,
        stats.n_train_pairs,
        stats.n_valid_pairs,
    )
    print(
        f"[build_hgt_graph] wrote {cfg.hgt.graph_path} "
        f"({stats.n_edges} edges, log={log_path})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
