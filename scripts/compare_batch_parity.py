#!/usr/bin/env python3
"""Dev A/B gate: batch vs per-row sampled eval parity and speed."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from emorecagent.config import load_config
from emorecagent.eval.runner import build_recommender, evaluate, load_split_jsonl
from emorecagent.utils.run_log import configure_run_logging
from emorecagent.utils.seeding import set_global_seed

PARITY_KEYS = ("hr@1", "hr@3", "hr@5")
PARITY_TOL_PP = 0.1
MIN_SPEEDUP = 3.0


def _runner_cfg(cfg, train, parallel_workers: int) -> dict:
    return {
        "factors": cfg.cf.factors,
        "cf_backend": cfg.cf.backend,
        "alpha": cfg.scoring.alpha,
        "lambda_decay": cfg.scoring.lambda_decay,
        "helpful_cap": cfg.scoring.helpful_vote_cap,
        "affective_rescaled": cfg.scoring.affective_rescaled,
        "absa_cache_path": cfg.absa.cache_path,
        "review_path": cfg.data.review_path,
        "max_reflection_iters": cfg.agents.max_reflection_iters,
        "pool_size": cfg.agents.candidate_pool_size,
        "top_k_aspects": cfg.agents.top_k_aspects,
        "llm_rank_prefix": cfg.agents.llm_rank_prefix,
        "aspect_recall_tau": cfg.agents.aspect_recall_tau,
        "aspect_recall_max": cfg.agents.aspect_recall_max,
        "use_reflection": cfg.ablation.reflection,
        "use_dynamic_weights": cfg.ablation.dynamic_weights,
        "use_aspect_term": cfg.ablation.aspect_term,
        "train_interactions": train,
        "seed": cfg.experiment.seed,
        "use_llm_cot": cfg.experiment.use_llm_cot,
        "parallel_workers": parallel_workers,
        "kg_backend": "memory",
        "app_config": cfg,
    }


def _hr_means(result) -> dict[str, float]:
    return {k: result.means[k] for k in PARITY_KEYS if k in result.means}


def _check_parity(baseline: dict[str, float], batch: dict[str, float]) -> list[str]:
    errors: list[str] = []
    for key in PARITY_KEYS:
        if key not in baseline or key not in batch:
            continue
        drift_pp = abs(batch[key] - baseline[key]) * 100.0
        if drift_pp > PARITY_TOL_PP:
            errors.append(
                f"{key}: baseline={baseline[key]:.4f} batch={batch[key]:.4f} "
                f"drift={drift_pp:.2f}pp > {PARITY_TOL_PP}pp"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch vs per-row parity gate.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="emorecagent_hettisasrec")
    parser.add_argument(
        "--split",
        default="data/processed/Beauty_and_Personal_Care",
    )
    parser.add_argument("--max-test-rows", type=int, default=100)
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=None,
        help="override llm.request_timeout_s (use 300+ for real TGI batch runs)",
    )
    parser.add_argument(
        "--skip-speed-gate",
        action="store_true",
        help="parity only (use with FakeLLM / CI)",
    )
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--log-file",
        default=None,
        help="log path (default: logs/compare_batch_parity_latest.log)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="log every N completed test rows (default: 25)",
    )
    args = parser.parse_args()

    log_file = args.log_file or Path(args.log_dir) / "compare_batch_parity_latest.log"
    logger, log_path = configure_run_logging(
        "batch_parity",
        log_file=log_file,
        log_dir=args.log_dir,
    )
    logger.info("=== batch parity gate ===")
    logger.info("log file: %s", log_path.resolve())

    cfg = load_config(args.config)
    if args.request_timeout is not None:
        cfg.llm.request_timeout_s = args.request_timeout
    set_global_seed(cfg.experiment.seed)
    split_dir = (
        args.split
        if isinstance(args.split, Path)
        else Path(args.split)
    )
    train = load_split_jsonl(split_dir / "train.jsonl")
    test = load_split_jsonl(split_dir / "test.jsonl")
    test = [t for t in test if t.verified_purchase][: args.max_test_rows]

    logger.info(
        "config: rows=%s n_negatives=%s batch_size=%s timeout=%ss",
        len(test),
        args.n_negatives,
        args.batch_size,
        cfg.llm.request_timeout_s,
    )

    runner_cfg = _runner_cfg(cfg, train, parallel_workers=1)
    eval_common = dict(
        k_values=cfg.eval.k_values,
        method=args.method,
        n_negatives=args.n_negatives,
        seed=cfg.experiment.seed,
        verified_only=True,
        max_test_rows=args.max_test_rows,
        progress_logger=logger,
        progress_interval=args.progress_interval,
    )

    logger.info("--- pass 1/2: per-row baseline (llm_batch=false) ---")
    t0 = time.monotonic()
    rec_base = build_recommender(args.method, runner_cfg, seed=cfg.experiment.seed)
    baseline = evaluate(
        rec_base,
        train,
        test,
        llm_batch=False,
        **eval_common,
    )
    t_baseline = time.monotonic() - t0
    logger.info("baseline pass finished (%.1fs)", t_baseline)

    logger.info("--- pass 2/2: batched eval (llm_batch=true) ---")
    t1 = time.monotonic()
    rec_batch = build_recommender(args.method, runner_cfg, seed=cfg.experiment.seed)
    batch = evaluate(
        rec_batch,
        train,
        test,
        llm_batch=True,
        batch_size=args.batch_size,
        batch_token_budget=cfg.eval.batch_token_budget,
        **eval_common,
    )
    t_batch = time.monotonic() - t1
    logger.info("batch pass finished (%.1fs)", t_batch)

    base_hr = _hr_means(baseline)
    batch_hr = _hr_means(batch)
    parity_errors = _check_parity(base_hr, batch_hr)

    def _emit(msg: str) -> None:
        logger.info(msg)

    _emit("=== parity (HR@K, pp tolerance 0.1) ===")
    for key in PARITY_KEYS:
        _emit(
            f"  {key}: baseline={base_hr.get(key, 0):.4f} "
            f"batch={batch_hr.get(key, 0):.4f}"
        )

    speedup = t_baseline / t_batch if t_batch > 0 else 0.0
    _emit(
        f"=== wall-clock: baseline={t_baseline:.1f}s "
        f"batch={t_batch:.1f}s speedup={speedup:.2f}x ==="
    )

    failed = False
    if parity_errors:
        failed = True
        _emit("PARITY GATE FAILED:")
        for err in parity_errors:
            _emit(f"  - {err}")
    else:
        _emit("PARITY GATE PASSED")

    if not args.skip_speed_gate:
        if speedup < MIN_SPEEDUP:
            failed = True
            _emit(
                f"SPEED GATE FAILED: {speedup:.2f}x < {MIN_SPEEDUP}x "
                "(use --skip-speed-gate for FakeLLM CI)"
            )
        else:
            _emit("SPEED GATE PASSED")

    logger.info("=== done (exit %s) ===", 1 if failed else 0)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
