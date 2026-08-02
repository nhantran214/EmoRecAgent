#!/usr/bin/env python3
"""Run offline ABSA extraction over scoped review targets with caching."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from emorecagent.absa.classical import MockClassicalAbsaTool
from emorecagent.absa.pipeline import ReviewRecord, build_absa_pipeline
from emorecagent.absa.preview import load_target_texts, write_absa_report
from emorecagent.config import ConfigError, load_config, resolve_llm_model
from emorecagent.llm.client import LLMClient
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


def count_targets(
    path: Path, max_reviews: int | None, only_ids: set[str] | None = None
) -> int:
    return sum(1 for _ in stream_targets(path, max_reviews, only_ids=only_ids))


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Concurrent reviews (default 1). For hybrid/llm_only use 4–32 so TGI "
            "can batch. Ignored for --backend classical (uses --batch-size on GPU)."
        ),
    )
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--classical-only",
        action="store_true",
        help="PyABSA only (no LLM). Sets backend=classical and pipeline_version=classical-v1.",
    )
    backend_group.add_argument(
        "--backend",
        choices=("hybrid", "llm_only", "classical"),
        default=None,
        help="Override absa.backend from config (classical = no LLM).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help=(
            "Classical PyABSA device (overrides absa.classical_device). "
            "Classical-only defaults to cuda."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Classical PyABSA eval_batch_size (overrides absa.classical_batch_size, "
            "default 32). Only used with --backend classical / --classical-only."
        ),
    )
    args = parser.parse_args()
    workers = max(1, int(args.workers))

    logger, log_path = configure_run_logging(
        "absa",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )

    cfg = load_config(args.config)
    if args.classical_only or args.backend == "classical":
        device = args.device or "cuda"
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else cfg.absa.classical_batch_size
        )
        absa_update: dict = {
            "backend": "classical",
            "pipeline_version": "classical-v1",
            "classical_device": device,
            "classical_batch_size": max(1, int(batch_size)),
        }
        # Keep hybrid cache intact: write classical results to a sibling DB.
        cache_path = Path(cfg.absa.cache_path)
        if "classical" not in cache_path.stem.lower():
            absa_update["cache_path"] = str(
                cache_path.with_name(f"{cache_path.stem}.classical{cache_path.suffix}")
            )
        cfg = cfg.model_copy(
            update={"absa": cfg.absa.model_copy(update=absa_update)}
        )
    else:
        absa_update = {}
        if args.backend is not None:
            absa_update["backend"] = args.backend
        if args.device is not None:
            absa_update["classical_device"] = args.device
        if args.batch_size is not None:
            absa_update["classical_batch_size"] = max(1, int(args.batch_size))
        if absa_update:
            cfg = cfg.model_copy(
                update={"absa": cfg.absa.model_copy(update=absa_update)}
            )

    targets_path = Path(args.targets or cfg.absa.targets_path)
    if not targets_path.exists():
        raise SystemExit(
            f"ABSA targets not found: {targets_path}\n"
            "Build the processed split first (exports train-scoped targets):\n"
            "  make data"
        )

    client = None
    if cfg.absa.backend != "classical":
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
    absa_model = (
        "none (classical-only)"
        if cfg.absa.backend == "classical"
        else resolve_llm_model(cfg.llm, for_absa=True)
    )
    logger.info(
        "backend=%s pipeline_version=%s llm_model=%s classical_checkpoint=%s "
        "classical_device=%s classical_batch_size=%s targets=%s cache=%s workers=%s",
        cfg.absa.backend,
        cfg.absa.pipeline_version,
        absa_model,
        cfg.absa.classical_checkpoint,
        cfg.absa.classical_device,
        cfg.absa.classical_batch_size,
        targets_path,
        cfg.absa.cache_path,
        workers,
    )
    logger.info("error log: %s", error_log.resolve())
    if cfg.absa.backend == "classical":
        tool = getattr(pipeline.processor, "tool", None)
        logger.info(
            "classical runtime device=%s batch_size=%s",
            getattr(tool, "device", cfg.absa.classical_device),
            cfg.absa.classical_batch_size,
        )

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
    pending: list[ReviewRecord] = []
    for rec in stream_targets(targets_path, args.max_reviews, only_ids=retry_ids):
        if (
            not args.no_cache
            and cache is not None
            and cache.contains(rec.review_id)
        ):
            hits += 1
            continue
        pending.append(rec)

    total = hits + len(pending)
    progress = tqdm(total=total, desc="ABSA")
    progress.update(hits)
    progress.set_postfix(hit=hits, new=processed, fail=failed, refresh=False)

    def _process_one(rec: ReviewRecord) -> tuple[str, Exception | None]:
        try:
            pipeline.process(rec, use_cache=not args.no_cache)
            return rec.review_id, None
        except Exception as exc:  # noqa: BLE001 — per-review resilience
            return rec.review_id, exc

    def _on_result(rec: ReviewRecord, exc: Exception | None) -> None:
        nonlocal processed, failed
        if exc is None:
            processed += 1
        else:
            failed += 1
            err_name = type(exc).__name__
            logger.error("review_id=%s %s: %s", rec.review_id, err_name, exc)
            with error_log.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "review_id": rec.review_id,
                            "error": f"{err_name}: {exc}",
                            "text_preview": rec.text[:200],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        progress.update(1)
        progress.set_postfix(hit=hits, new=processed, fail=failed, refresh=False)

    if cfg.absa.backend == "classical":
        batch_size = max(1, int(cfg.absa.classical_batch_size))
        logger.info(
            "Classical GPU batching: batch_size=%d (workers ignored; PyABSA "
            "eval_batch_size)",
            batch_size,
        )
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            try:
                pipeline.process_batch(chunk, use_cache=not args.no_cache)
                for rec in chunk:
                    _on_result(rec, None)
            except Exception as exc:  # noqa: BLE001 — fall back per review
                logger.warning(
                    "classical batch failed (%s); retrying %d reviews one-by-one",
                    exc,
                    len(chunk),
                )
                for rec in chunk:
                    _, one_exc = _process_one(rec)
                    _on_result(rec, one_exc)
    elif workers == 1:
        for rec in pending:
            _, exc = _process_one(rec)
            _on_result(rec, exc)
    else:
        logger.info(
            "Parallel ABSA: %d workers (TGI can batch; expect higher VRAM/util)",
            workers,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, rec): rec for rec in pending}
            for fut in as_completed(futures):
                rec = futures[fut]
                _, exc = fut.result()
                _on_result(rec, exc)

    progress.close()

    logger.info(
        "cache_hits=%s newly_processed=%s failed=%s cache_path=%s workers=%s",
        hits,
        processed,
        failed,
        cfg.absa.cache_path,
        workers,
    )
    run_stats = {
        "cache_hits": hits,
        "newly_processed": processed,
        "failed": failed,
        "workers": workers,
        "classical_device": cfg.absa.classical_device,
        "classical_batch_size": cfg.absa.classical_batch_size,
    }
    print(
        f"[run_absa] backend={cfg.absa.backend} "
        f"device={cfg.absa.classical_device} batch_size={cfg.absa.classical_batch_size} "
        f"workers={workers} "
        f"cache_hits={hits} newly_processed={processed} failed={failed} "
        f"log={log_path} errors={error_log if failed else 'none'}"
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
