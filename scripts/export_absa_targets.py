#!/usr/bin/env python3
"""Export scoped ABSA targets from processed train split + raw reviews.

Writes ``absa.targets_path`` (default: ``{out_dir}/absa_targets.jsonl``).
Run automatically at the end of ``make data``; can also be invoked alone
after the train split exists.

Example:
    PYTHONPATH=src python3 scripts/export_absa_targets.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emorecagent.absa.targets import export_absa_targets
from emorecagent.config import load_config
from emorecagent.utils.run_log import configure_run_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger, log_path = configure_run_logging(
        "absa_targets",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )

    cfg = load_config(args.config)
    train_path = Path(cfg.data.out_dir) / "train.jsonl"
    if not train_path.exists():
        logger.error("train split missing: %s (run make data first)", train_path)
        return 1

    raw_path = Path(cfg.data.review_path)
    if not raw_path.exists():
        logger.error("raw review file missing: %s", raw_path)
        return 1

    targets_path = Path(cfg.absa.targets_path)
    logger.info("log file: %s", log_path.resolve())
    logger.info("train=%s raw=%s out=%s", train_path, raw_path, targets_path)

    stats = export_absa_targets(
        train_path=train_path,
        raw_review_path=raw_path,
        out_path=targets_path,
    )
    logger.info(
        "train_interactions=%s targets_written=%s raw_scanned=%s "
        "raw_matched=%s skipped_no_text=%s skipped_duplicate=%s",
        stats.n_train_interactions,
        stats.n_targets_written,
        stats.n_raw_scanned,
        stats.n_raw_matched,
        stats.n_skipped_no_text,
        stats.n_skipped_duplicate,
    )
    print(
        f"[export_absa_targets] wrote {stats.n_targets_written} reviews to {targets_path} "
        f"(train interactions={stats.n_train_interactions})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
