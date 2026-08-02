#!/usr/bin/env python3
"""Build the processed dataset: stream -> [optional dedup] -> k-core -> sample -> split.

Usage:
  PYTHONPATH=src python scripts/build_dataset.py --config configs/default.yaml
  PYTHONPATH=src python scripts/build_dataset.py --config configs/default.yaml --max-scan 200000
  PYTHONPATH=src python scripts/build_dataset.py --log-dir logs
  # RecBole / AC-TSR Yelp (no user-item dedup):
  PYTHONPATH=src python scripts/build_dataset.py --config configs/categories/Yelp_AC_tisasrec_paper.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emorecagent.absa.targets import export_absa_targets  # noqa: E402
from emorecagent.config import load_config  # noqa: E402
from emorecagent.data.kcore import k_core_filter, k_core_summary  # noqa: E402
from emorecagent.data.loader import dedup_earliest, load_interactions  # noqa: E402
from emorecagent.data.recbole_inter import (  # noqa: E402
    load_recbole_inter,
    resolve_inter_path,
)
from emorecagent.data.yelp import resolve_review_source  # noqa: E402
from emorecagent.data.split import (  # noqa: E402
    chronological_split,
    leave_last_out,
    sample_subset,
    write_split,
)
from emorecagent.utils.run_log import configure_run_logging  # noqa: E402


def _log_config(logger, cfg) -> None:
    d = cfg.data
    logger.info("=== EmoRecAgent dataset build ===")
    logger.info("config: %s", cfg.experiment.name)
    logger.info("seed: %s", cfg.experiment.seed)
    logger.info("review_path: %s", d.review_path)
    logger.info("inter_path: %s", d.inter_path)
    logger.info("out_dir: %s", d.out_dir)
    logger.info(
        "k_core=%s max_users=%s max_items=%s min_history=%s split_method=%s",
        d.k_core,
        d.max_users,
        d.max_items,
        d.min_history,
        d.split_method,
    )
    logger.info(
        "timestamp_s bounds: min=%s max=%s",
        d.min_timestamp_s,
        d.max_timestamp_s,
    )
    logger.info("dedup_user_item: %s", d.dedup_user_item)
    logger.info(
        "split ratios: train=%.2f valid=%.2f test=%.2f",
        d.split_train_ratio,
        d.split_valid_ratio,
        d.split_test_ratio,
    )
    logger.info("absa.enabled: %s", cfg.absa.enabled)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Cap raw lines read from the review/.inter file (smoke testing).",
    )
    ap.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for timestamped log files (default: logs/).",
    )
    ap.add_argument(
        "--log-file",
        default=None,
        help="Explicit log file path (overrides --log-dir auto naming).",
    )
    args = ap.parse_args()

    logger, log_path = configure_run_logging(
        "data",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )
    logger.info("log file: %s", log_path.resolve())
    t0 = time.monotonic()

    cfg = load_config(args.config)
    d = cfg.data
    _log_config(logger, cfg)

    use_inter = bool(d.inter_path)
    review_path: Path | None = None
    inter_path: Path | None = None

    if use_inter:
        try:
            inter_path = resolve_inter_path(d.inter_path)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("resolved inter_path: %s", inter_path)
        logger.info(
            "inter file size: %.2f MB",
            inter_path.stat().st_size / (1024 * 1024),
        )
        logger.info("--- stage: load RecBole .inter (max_scan=%s) ---", args.max_scan)
        t_stage = time.monotonic()
        raw = load_recbole_inter(
            inter_path,
            max_scan=args.max_scan,
            min_timestamp_s=d.min_timestamp_s,
            max_timestamp_s=d.max_timestamp_s,
        )
        logger.info(
            "raw interactions (after date filter): %s (%.1fs)",
            f"{len(raw):,}",
            time.monotonic() - t_stage,
        )
    else:
        try:
            review_path = resolve_review_source(d.review_path)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        if not review_path.exists():
            logger.error("review file not found: %s", review_path)
            return 1
        if str(review_path) != str(d.review_path):
            logger.info("resolved review_path: %s -> %s", d.review_path, review_path)
        logger.info(
            "review file size: %.2f MB",
            review_path.stat().st_size / (1024 * 1024),
        )

        logger.info("--- stage: load raw reviews (max_scan=%s) ---", args.max_scan)
        t_stage = time.monotonic()
        raw = load_interactions(review_path, max_scan=args.max_scan)
        logger.info(
            "raw interactions: %s (%.1fs)",
            f"{len(raw):,}",
            time.monotonic() - t_stage,
        )
        if d.min_timestamp_s is not None or d.max_timestamp_s is not None:
            from emorecagent.data.recbole_inter import filter_by_timestamp_s

            before = len(raw)
            raw = filter_by_timestamp_s(
                raw,
                min_timestamp_s=d.min_timestamp_s,
                max_timestamp_s=d.max_timestamp_s,
            )
            logger.info(
                "after timestamp filter: %s (removed %s)",
                f"{len(raw):,}",
                f"{before - len(raw):,}",
            )

    if d.dedup_user_item:
        logger.info("--- stage: dedup (earliest per user,item) ---")
        t_stage = time.monotonic()
        deduped = dedup_earliest(raw)
        logger.info(
            "after dedup: %s (%.1fs, removed %s)",
            f"{len(deduped):,}",
            time.monotonic() - t_stage,
            f"{len(raw) - len(deduped):,}",
        )
    else:
        # RecBole / AC-TSR keep multi-visit (user, item) rows via review_id.
        logger.info("--- stage: dedup skipped (dedup_user_item=false) ---")
        deduped = raw

    logger.info("--- stage: %s-core filter ---", d.k_core)
    t_stage = time.monotonic()
    kcore_data = k_core_filter(deduped, d.k_core)
    kcore_stats = k_core_summary(kcore_data, d.k_core)
    logger.info(
        "after %s-core: %s interactions, %s users, %s items "
        "(min user/item degree: %s/%s) (%.1fs)",
        d.k_core,
        f"{kcore_stats.n_interactions:,}",
        f"{kcore_stats.n_users:,}",
        f"{kcore_stats.n_items:,}",
        kcore_stats.min_user_degree,
        kcore_stats.min_item_degree,
        time.monotonic() - t_stage,
    )

    logger.info("--- stage: agentic sampling ---")
    t_stage = time.monotonic()
    sampled = sample_subset(
        kcore_data,
        k_core=d.k_core,
        max_users=d.max_users,
        max_items=d.max_items,
        seed=cfg.experiment.seed,
    )
    if d.max_users is not None or d.max_items is not None:
        post_sample = k_core_summary(sampled, d.k_core)
        logger.info(
            "after sampling + %s-core restore: %s interactions, %s users, %s items "
            "(%.1fs)",
            d.k_core,
            f"{post_sample.n_interactions:,}",
            f"{post_sample.n_users:,}",
            f"{post_sample.n_items:,}",
            time.monotonic() - t_stage,
        )
        if post_sample.n_interactions == 0:
            logger.error(
                "agentic sampling produced 0 interactions after %s-core restore. "
                "Try raising max_users/max_items or set them to null in config.",
                d.k_core,
            )
            return 1
    else:
        post_sample = kcore_stats
        logger.info(
            "sampling skipped (no max_users/max_items caps) (%.1fs)",
            time.monotonic() - t_stage,
        )

    logger.info("--- stage: split (%s) ---", d.split_method)
    t_stage = time.monotonic()
    if d.split_method == "leave_last_out":
        split = leave_last_out(sampled, min_history=d.min_history)
    else:
        split = chronological_split(
            sampled,
            train_ratio=d.split_train_ratio,
            valid_ratio=d.split_valid_ratio,
            test_ratio=d.split_test_ratio,
            min_history=d.min_history,
        )
    logger.info("split computed (%.1fs)", time.monotonic() - t_stage)
    for key, val in split.manifest.items():
        logger.info("  manifest[%s] = %s", key, val)

    logger.info("--- stage: write outputs ---")
    t_stage = time.monotonic()
    extra = {
        **kcore_stats.as_manifest(),
        "post_sample_min_user_degree": post_sample.min_user_degree,
        "post_sample_min_item_degree": post_sample.min_item_degree,
        "split_method": d.split_method,
        "min_timestamp_s": d.min_timestamp_s,
        "max_timestamp_s": d.max_timestamp_s,
        "dedup_user_item": d.dedup_user_item,
        "inter_path": str(inter_path) if inter_path is not None else None,
        "source": "recbole_inter" if use_inter else "review_jsonl",
    }
    manifest_path = write_split(
        d.out_dir,
        split,
        cfg.experiment.seed,
        d.k_core,
        extra_manifest=extra,
    )
    out_dir = manifest_path.parent
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl", "manifest.json"):
        p = out_dir / name
        if p.exists():
            logger.info("  wrote %s (%.2f MB)", p, p.stat().st_size / (1024 * 1024))

    if cfg.absa.enabled:
        if review_path is None:
            logger.error(
                "absa.enabled=true but no review JSONL source is available "
                "(inter-only builds must set absa.enabled=false)"
            )
            return 1
        logger.info("--- stage: export ABSA targets (train scope) ---")
        t_stage = time.monotonic()
        targets_path = Path(cfg.absa.targets_path)
        absa_stats = export_absa_targets(
            train_path=out_dir / "train.jsonl",
            raw_review_path=review_path,
            out_path=targets_path,
        )
        logger.info(
            "ABSA targets: %s reviews (train interactions=%s, raw scanned=%s) (%.1fs)",
            f"{absa_stats.n_targets_written:,}",
            f"{absa_stats.n_train_interactions:,}",
            f"{absa_stats.n_raw_scanned:,}",
            time.monotonic() - t_stage,
        )
        if targets_path.exists():
            logger.info(
                "  wrote %s (%.2f MB)",
                targets_path,
                targets_path.stat().st_size / (1024 * 1024),
            )
    else:
        logger.info("--- stage: export ABSA targets skipped (absa.enabled=false) ---")

    elapsed = time.monotonic() - t0
    logger.info("=== done in %.1fs ===", elapsed)
    logger.info("outputs: %s", out_dir.resolve())
    logger.info("manifest: %s", manifest_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
