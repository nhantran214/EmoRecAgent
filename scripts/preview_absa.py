#!/usr/bin/env python3
"""Preview ABSA cache: terminal summary and optional HTML report."""

from __future__ import annotations

import argparse
from pathlib import Path

from emorecagent.absa.cache import AbsaCache
from emorecagent.absa.preview import format_terminal, load_target_texts, write_absa_report
from emorecagent.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview ABSA cache extractions.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", default=None, help="override absa.cache_path")
    parser.add_argument("--targets", default=None, help="override absa.targets_path")
    parser.add_argument("--samples", type=int, default=8, help="random sample size")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--json",
        default=None,
        help="override summary JSON path (default: absa.summary_path)",
    )
    parser.add_argument(
        "--html",
        default=None,
        help="override HTML path (default: absa.preview_html_path; pass '' to skip)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="allow empty-triple reviews in random samples",
    )
    parser.add_argument(
        "--require-text",
        action="store_true",
        help="only sample reviews whose text is in targets (skip stale cache keys)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_path = Path(args.cache or cfg.absa.cache_path)
    targets_path = Path(args.targets or cfg.absa.targets_path)
    seed = cfg.experiment.seed if args.seed is None else args.seed

    if not cache_path.exists():
        raise SystemExit(
            f"ABSA cache not found: {cache_path}\n"
            "Run extraction first:\n"
            "  make absa\n"
            "Dev smoke:\n"
            "  python3 scripts/run_absa.py --config configs/default.yaml --max-reviews 20"
        )

    cache = AbsaCache(cache_path)
    entries = list(cache.iter_all())
    cache.close()

    texts = load_target_texts(targets_path)
    n_with_text = sum(1 for rid, _ in entries if rid in texts)
    if texts and n_with_text == 0:
        print(
            "[preview_absa] warning: no cache review_id matches absa_targets.jsonl — "
            "re-run `make clean-absa && make absa` after `make data` to align ids."
        )
    elif texts and n_with_text < len(entries):
        print(
            f"[preview_absa] note: {n_with_text}/{len(entries)} cached ids match "
            "targets (showing samples with text when available)."
        )

    json_path = args.json if args.json is not None else cfg.absa.summary_path
    if args.html is None:
        html_path: str | None = cfg.absa.preview_html_path
    elif args.html == "":
        html_path = None
    else:
        html_path = args.html

    report = write_absa_report(
        entries,
        texts,
        json_path=json_path,
        html_path=html_path,
        samples_n=args.samples,
        seed=seed,
        prefer_non_empty=not args.include_empty,
        require_text=args.require_text,
        backend=cfg.absa.backend,
        pipeline_version=cfg.absa.pipeline_version,
        cache_path=cache_path,
    )
    print(format_terminal(report.summary, report.samples))
    print(f"\n[preview_absa] summary JSON: {report.paths.json_path.resolve()}")
    if report.paths.html_path is not None:
        print(f"[preview_absa] HTML report: {report.paths.html_path.resolve()}")


if __name__ == "__main__":
    main()
