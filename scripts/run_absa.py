#!/usr/bin/env python3
"""Run offline ABSA extraction over scoped review targets with caching."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from emorecagent.absa.classical import MockClassicalAbsaTool
from emorecagent.absa.pipeline import ReviewRecord, build_absa_pipeline
from emorecagent.absa.preview import load_target_texts, write_absa_report
from emorecagent.config import ConfigError, load_config, resolve_llm_model
from emorecagent.llm.client import LLMClient, LLMError
from emorecagent.llm.schemas import TripleSet
from emorecagent.utils.run_log import configure_run_logging


def stream_targets(path: Path, max_reviews: int | None, only_ids: set[str] | None = None):
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if max_reviews is not None and n >= max_reviews:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            rid = str(row.get("review_id") or "")
            if not text or not rid:
                continue
            if only_ids is not None and rid not in only_ids:
                continue
            yield ReviewRecord(review_id=rid, text=text)
            n += 1


def load_error_review_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("review_id") or "")
            if rid:
                ids.add(rid)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ABSA extraction with cache.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--targets",
        default=None,
        help="scoped targets JSONL (default: absa.targets_path from config)",
    )
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="skip writing results/absa_summary.json and preview HTML",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="reprocess review_ids from absa_errors.jsonl (clears their cache entries first)",
    )
    args = parser.parse_args()

    logger, log_path = configure_run_logging(
        "absa",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )

    cfg = load_config(args.config)
    targets_path = Path(args.targets or cfg.absa.targets_path)
    if not targets_path.exists():
        raise SystemExit(
            f"ABSA targets not found: {targets_path}\n"
            "Build the processed split first (exports train-scoped targets):\n"
            "  make data"
        )

    client = LLMClient.from_config(cfg, for_absa=True)
    try:
        mock_tool = (
            MockClassicalAbsaTool([])
            if os.environ.get("EMOREC_ABSA_MOCK") == "1"
            else None
        )
        pipeline = build_absa_pipeline(cfg, client, classical_tool=mock_tool)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    error_log = Path(args.log_dir) / "absa_errors.jsonl"
    error_log.parent.mkdir(parents=True, exist_ok=True)

    retry_ids: set[str] | None = None
    if args.retry_errors:
        retry_ids = load_error_review_ids(error_log)
        if not retry_ids:
            raise SystemExit(f"No review_ids found in {error_log}")

    logger.info("log file: %s", log_path.resolve())
    absa_model = resolve_llm_model(cfg.llm, for_absa=True)
    logger.info(
        "backend=%s pipeline_version=%s llm_model=%s classical_checkpoint=%s "
        "targets=%s cache=%s",
        cfg.absa.backend,
        cfg.absa.pipeline_version,
        absa_model,
        cfg.absa.classical_checkpoint,
        targets_path,
        cfg.absa.cache_path,
    )
    logger.info("error log: %s", error_log.resolve())

    cache = pipeline.cache
    if retry_ids is not None and cache is not None:
        cleared = sum(1 for rid in retry_ids if cache.delete(rid))
        logger.info(
            "retry_errors: %d ids from %s, cleared %d cache entries",
            len(retry_ids),
            error_log,
            cleared,
        )

    hits = processed = failed = 0
    for rec in tqdm(
        stream_targets(targets_path, args.max_reviews, only_ids=retry_ids),
        desc="ABSA",
    ):
        if (
            not args.no_cache
            and cache is not None
            and cache.contains(rec.review_id)
        ):
            hits += 1
            continue
        try:
            pipeline.process(rec, use_cache=not args.no_cache)
            processed += 1
        except LLMError as exc:
            failed += 1
            logger.error("review_id=%s LLMError: %s", rec.review_id, exc)
            with error_log.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "review_id": rec.review_id,
                            "error": str(exc),
                            "text_preview": rec.text[:200],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if not args.no_cache and cache is not None:
                cache.put(rec.review_id, TripleSet(triples=[]))

    logger.info(
        "cache_hits=%s newly_processed=%s failed=%s cache_path=%s",
        hits,
        processed,
        failed,
        cfg.absa.cache_path,
    )
    run_stats = {
        "cache_hits": hits,
        "newly_processed": processed,
        "failed": failed,
    }
    print(
        f"[run_absa] backend={cfg.absa.backend} cache_hits={hits} "
        f"newly_processed={processed} failed={failed} log={log_path} "
        f"errors={error_log if failed else 'none'}"
    )

    if cache is not None and not args.no_report:
        entries = list(cache.iter_all())
        texts = load_target_texts(targets_path)
        report = write_absa_report(
            entries,
            texts,
            json_path=cfg.absa.summary_path,
            html_path=cfg.absa.preview_html_path,
            samples_n=8,
            seed=cfg.experiment.seed,
            backend=cfg.absa.backend,
            pipeline_version=cfg.absa.pipeline_version,
            cache_path=cfg.absa.cache_path,
            run_stats=run_stats,
        )
        logger.info("summary json: %s", report.paths.json_path.resolve())
        if report.paths.html_path is not None:
            logger.info("preview html: %s", report.paths.html_path.resolve())
        print(
            f"[run_absa] summary={report.paths.json_path.resolve()} "
            f"preview={report.paths.html_path.resolve() if report.paths.html_path else 'none'}"
        )

    if cache is not None:
        cache.close()


if __name__ == "__main__":
    main()
