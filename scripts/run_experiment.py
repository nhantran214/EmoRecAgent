#!/usr/bin/env python3
"""Run a recommendation experiment over a processed split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from emorecagent.config import load_config, resolve_llm_model
from emorecagent.eval.bootstrap import bootstrap_user_mean_ci
from emorecagent.eval import metrics as M
from emorecagent.eval.checkpoint import default_checkpoint_stem
from emorecagent.eval.runner import (
    build_recommender,
    evaluate,
    load_split_jsonl,
    write_results,
)
from emorecagent.utils.logging import RunLogger
from emorecagent.utils.run_log import configure_run_logging
from emorecagent.utils.seeding import set_global_seed


def _eval_pass_protocol(args: argparse.Namespace) -> str:
    if args.no_sampled_eval and args.eval_pass == "both":
        return "full_catalog"
    if args.eval_pass == "both":
        n = args.n_negatives if args.n_negatives is not None else "config"
        return f"full_catalog + sampled_negatives({n})"
    if args.eval_pass == "sampled":
        n = args.n_negatives if args.n_negatives is not None else "config"
        return f"sampled_negatives({n})"
    return "full_catalog"


def _resolve_eval_pass(
    args: argparse.Namespace, cfg
) -> tuple[str, int | None, int | None]:
    """Return (mode, n_negatives single-pass, sampled_n_negatives dual-pass)."""
    eval_pass = args.eval_pass
    if args.no_sampled_eval:
        if eval_pass == "both":
            eval_pass = "full"
        elif eval_pass == "sampled":
            raise SystemExit(
                "--no-sampled-eval conflicts with --eval-pass sampled"
            )

    n_neg = args.n_negatives if args.n_negatives is not None else cfg.eval.n_negatives

    if eval_pass == "both":
        if n_neg is None:
            raise SystemExit(
                "eval.n_negatives is required when --eval-pass both "
                "(set in config or pass --n-negatives)"
            )
        return "both", None, n_neg
    if eval_pass == "sampled":
        if n_neg is None:
            raise SystemExit(
                "eval.n_negatives is required for --eval-pass sampled "
                "(set in config or pass --n-negatives)"
            )
        return "sampled", n_neg, None
    return "full", None, None


def _log_config(
    logger, cfg, args, cumulative: bool, verified_only: bool
) -> None:
    logger.info("=== EmoRecAgent experiment ===")
    logger.info("config: %s", cfg.experiment.name)
    logger.info("method: %s", args.method)
    logger.info("seed: %s", cfg.experiment.seed)
    logger.info("split: %s", args.split)
    logger.info("out: %s", args.out)
    logger.info(
        "eval_protocol: %s aggregation: %s",
        cfg.eval.protocol,
        cfg.eval.aggregation,
    )
    logger.info(
        "llm_batch: %s batch_size: %s batch_token_budget: %s",
        cfg.eval.llm_batch,
        cfg.eval.batch_size,
        cfg.eval.batch_token_budget,
    )
    logger.info(
        "eval_pass: %s",
        args.eval_pass,
    )
    logger.info(
        "candidate_protocol: %s",
        _eval_pass_protocol(args),
    )
    logger.info("k_values: %s hr_avg_k: %s", cfg.eval.k_values, cfg.eval.hr_avg_k)
    logger.info("verified_only: %s", verified_only)
    logger.info(
        "llm: model=%s endpoint=%s",
        resolve_llm_model(cfg.llm),
        cfg.tgi.base_url,
    )
    if args.method in (
        "emorecagent",
        "emorecagent_fast",
        "emorecagent_align",
        "aspect_aware",
    ):
        logger.info(
            "ablation: reflection=%s dynamic_weights=%s aspect_term=%s",
            cfg.ablation.reflection,
            cfg.ablation.dynamic_weights,
            cfg.ablation.aspect_term,
        )
    if args.method == "emorecagent":
        logger.info(
            "graph: use_llm_cot=%s max_test_rows=%s verified_only=%s",
            cfg.experiment.use_llm_cot,
            args.max_test_rows if args.max_test_rows is not None else cfg.experiment.max_test_rows,
            verified_only,
        )
    if args.method == "emorecagent_align":
        ta = cfg.tisasrec_align
        logger.info(
            "tisasrec_align: fusion_alpha=%s tu_mode=%s stage1_only=%s "
            "use_hash_encoder=%s stage1=%s alignment=%s tu_cache=%s",
            ta.fusion_alpha,
            ta.tu_mode,
            ta.stage1_only,
            ta.use_hash_encoder,
            ta.stage1_checkpoint_path,
            ta.alignment_checkpoint_path,
            ta.tu_cache_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EmoRecAgent experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--split",
        required=True,
        help="directory with train.jsonl / test.jsonl (from build_dataset.py)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--n-negatives",
        type=int,
        default=None,
        help="override eval.n_negatives for the additional 1+N sampled pass",
    )
    parser.add_argument(
        "--no-sampled-eval",
        action="store_true",
        help="legacy alias for --eval-pass full (skip sampled 1+N pass)",
    )
    parser.add_argument(
        "--eval-pass",
        choices=["sampled", "full", "both"],
        default="both",
        help=(
            "candidate ranking pass: sampled (1+N negatives only), "
            "full (full catalog only), or both (default for generic experiments)"
        ),
    )
    parser.add_argument(
        "--cumulative-history",
        action="store_true",
        help="exclude prior test interactions from candidate pool per user",
    )
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="log every N test rows (default: 100)",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="disable tqdm progress bar on stderr",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="include test rows without verified_purchase (overrides eval.verified_only)",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="cap number of test rows (overrides config experiment.max_test_rows)",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help=(
            "concurrent eval units (overrides eval.parallel_workers); "
            "rows under per_row, users under user_batch full-catalog. "
            "The user_batch sampled pass stays serial."
        ),
    )
    parser.add_argument(
        "--checkpoint-stem",
        default=None,
        help=(
            "checkpoint base path (default: <out>.checkpoint); "
            "creates .full.jsonl / .sampled.jsonl siblings"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore existing checkpoint and re-evaluate all rows",
    )
    parser.add_argument(
        "--fresh-checkpoint",
        action="store_true",
        help="delete checkpoint files before this run",
    )
    parser.add_argument(
        "--llm-batch",
        action="store_true",
        help="group per-row rankings into batched LLM calls (overrides eval.llm_batch)",
    )
    parser.add_argument(
        "--no-llm-batch",
        action="store_true",
        help="disable batched LLM eval even if enabled in config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="max rows per batched LLM call (8–16, overrides eval.batch_size)",
    )
    parser.add_argument(
        "--batch-token-budget",
        type=int,
        default=None,
        help="approx prompt token budget per batch (overrides eval.batch_token_budget)",
    )
    args = parser.parse_args()

    logger, log_path = configure_run_logging(
        "experiment",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )
    logger.info("log file: %s", log_path.resolve())
    t0 = time.monotonic()

    cfg = load_config(args.config)
    set_global_seed(cfg.experiment.seed)
    split_dir = Path(args.split)
    cumulative = args.cumulative_history or cfg.eval.cumulative_history
    hr_avg_k = tuple(cfg.eval.hr_avg_k)
    verified_only = cfg.eval.verified_only and not args.include_unverified
    _log_config(logger, cfg, args, cumulative, verified_only)

    logger.info("--- stage: load split ---")
    t_stage = time.monotonic()
    train = load_split_jsonl(split_dir / "train.jsonl")
    test = load_split_jsonl(split_dir / "test.jsonl")
    # RecBole LOO / Yelp_AC paper: history = train+valid when scoring test.
    history_train = train
    if (
        args.method == "emorecagent_align"
        and getattr(cfg.tisasrec_align, "test_history", "train") == "train_valid"
    ):
        valid_path = split_dir / "valid.jsonl"
        if valid_path.is_file():
            valid = load_split_jsonl(valid_path)
            history_train = train + valid
            logger.info(
                "test_history=train_valid → fit history train+valid "
                "(%s + %s rows)",
                f"{len(train):,}",
                f"{len(valid):,}",
            )
        else:
            logger.warning(
                "test_history=train_valid but %s missing; using train-only",
                valid_path,
            )
    n_verified_test = sum(1 for t in test if t.verified_purchase)
    logger.info(
        "loaded train=%s test=%s verified_test=%s (%.1fs)",
        f"{len(train):,}",
        f"{len(test):,}",
        f"{n_verified_test:,}",
        time.monotonic() - t_stage,
    )
    if verified_only:
        logger.info(
            "eval will use verified_purchase rows only (%s of %s test rows)",
            f"{n_verified_test:,}",
            f"{len(test):,}",
        )

    parallel_workers = (
        args.parallel_workers
        if args.parallel_workers is not None
        else cfg.eval.parallel_workers
    )
    llm_batch = cfg.eval.llm_batch
    if args.llm_batch:
        llm_batch = True
    if args.no_llm_batch:
        llm_batch = False
    batch_size = (
        args.batch_size if args.batch_size is not None else cfg.eval.batch_size
    )
    batch_token_budget = (
        args.batch_token_budget
        if args.batch_token_budget is not None
        else cfg.eval.batch_token_budget
    )
    runner_cfg = {
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
        "train_interactions": history_train,
        "seed": cfg.experiment.seed,
        "use_llm_cot": cfg.experiment.use_llm_cot,
        "parallel_workers": parallel_workers,
        "kg_backend": "neo4j" if args.method == "emorecagent" else "memory",
        "app_config": cfg,
    }

    logger.info("--- stage: build recommender ---")
    t_stage = time.monotonic()
    recommender = build_recommender(args.method, runner_cfg, seed=cfg.experiment.seed)
    logger.info(
        "recommender=%s (%.1fs)",
        recommender.name,
        time.monotonic() - t_stage,
    )

    logger.info("--- stage: evaluate ---")
    t_stage = time.monotonic()
    max_test_rows = args.max_test_rows
    if max_test_rows is None:
        max_test_rows = cfg.experiment.max_test_rows
    method_variant = "langgraph" if args.method == "emorecagent" else None
    eval_mode, n_negatives, sampled_n = _resolve_eval_pass(args, cfg)
    logger.info("eval_mode: %s", eval_mode)
    checkpoint_stem = (
        Path(args.checkpoint_stem)
        if args.checkpoint_stem
        else default_checkpoint_stem(args.out)
    )
    resume = not args.no_resume
    logger.info(
        "checkpoint stem: %s (resume=%s fresh=%s)",
        checkpoint_stem.resolve(),
        resume,
        args.fresh_checkpoint,
    )
    eval_kwargs = dict(
        checkpoint_stem=checkpoint_stem,
        resume=resume,
        fresh_checkpoint=args.fresh_checkpoint,
        llm_batch=llm_batch,
        batch_size=batch_size,
        batch_token_budget=batch_token_budget,
    )
    if eval_mode == "both":
        result = evaluate(
            recommender,
            train,
            test,
            k_values=cfg.eval.k_values,
            method=args.method,
            seed=cfg.experiment.seed,
            cumulative_history=cumulative,
            hr_avg_k=hr_avg_k,
            show_progress=not args.no_progress_bar,
            progress_logger=logger,
            progress_interval=args.progress_interval,
            verified_only=verified_only,
            max_test_rows=max_test_rows,
            method_variant=method_variant,
            use_llm_cot=cfg.experiment.use_llm_cot if args.method == "emorecagent" else None,
            eval_protocol=cfg.eval.protocol,
            aggregation=cfg.eval.aggregation,
            sampled_n_negatives=sampled_n,
            sampled_k_values=cfg.eval.sampled_k_values,
            parallel_workers=parallel_workers,
            **eval_kwargs,
        )
    else:
        result = evaluate(
            recommender,
            train,
            test,
            k_values=cfg.eval.k_values,
            method=args.method,
            seed=cfg.experiment.seed,
            cumulative_history=cumulative,
            hr_avg_k=hr_avg_k,
            show_progress=not args.no_progress_bar,
            progress_logger=logger,
            progress_interval=args.progress_interval,
            verified_only=verified_only,
            max_test_rows=max_test_rows,
            method_variant=method_variant,
            use_llm_cot=cfg.experiment.use_llm_cot if args.method == "emorecagent" else None,
            eval_protocol=cfg.eval.protocol,
            aggregation=cfg.eval.aggregation,
            n_negatives=n_negatives,
            sampled_k_values=cfg.eval.sampled_k_values,
            parallel_workers=parallel_workers,
            **eval_kwargs,
        )
    logger.info("evaluate finished (%.1fs)", time.monotonic() - t_stage)

    if args.method == "emorecagent_align":
        ta = cfg.tisasrec_align
        result.metadata = {
            "tisasrec_align": {
                "fusion_alpha": ta.fusion_alpha,
                "stage1_only": ta.stage1_only,
                "stage2_mode": ta.stage2_mode,
                "use_hash_encoder": ta.use_hash_encoder,
                "stage1_checkpoint": ta.stage1_checkpoint_path,
                "alignment_checkpoint": ta.alignment_checkpoint_path,
                "tu_cache": ta.tu_cache_path,
                "cross_user_lookup": ta.cross_user_lookup_path,
                "rerank_pool_k": ta.rerank_pool_k,
                "llm_pool_cap": ta.llm_pool_cap,
            }
        }
        rec = recommender
        if hasattr(rec, "n_fallback"):
            result.metadata["tisasrec_align"].update(
                {
                    "n_fallback": rec.n_fallback,
                    "n_llm_calls": rec.n_llm_calls,
                    "n_stage1_only": rec.n_stage1_only,
                }
            )

    logger.info("--- stage: bootstrap CI ---")
    t_stage = time.monotonic()
    ci_keys = [M.AVG_HR_KEY, "hr@10", "ndcg@10"]
    if cfg.eval.protocol == "user_batch" and cfg.eval.k_values:
        ci_keys.append(f"recall@{cfg.eval.k_values[-1]}")
    ci_keys = list(dict.fromkeys(ci_keys))

    def _add_bootstrap(
        per_user: dict,
        user_ids: list[str],
        ci_store: dict,
        label: str,
    ) -> None:
        for key in ci_keys:
            if key in per_user and user_ids:
                ci = bootstrap_user_mean_ci(
                    per_user,
                    user_ids,
                    key,
                    n_bootstrap=cfg.eval.n_bootstrap,
                    seed=cfg.experiment.seed,
                )
                ci_store[key] = {"low": ci.low, "high": ci.high}
                logger.info(
                    "  [%s] %s user-mean CI [%.4f, %.4f]",
                    label,
                    key,
                    ci.low,
                    ci.high,
                )

    _add_bootstrap(
        result.per_user,
        result.user_ids,
        result.ci_per_user,
        "sampled" if result.protocol == "sampled_negatives" else "full",
    )
    if result.sampled:
        result.sampled["ci_per_user"] = {}
        _add_bootstrap(
            result.sampled["per_user"],
            result.sampled["user_ids"],
            result.sampled["ci_per_user"],
            "sampled",
        )
    logger.info("bootstrap complete (%.1fs)", time.monotonic() - t_stage)

    logger.info("--- stage: write results ---")
    out = write_results(args.out, result)
    run_id = f"{args.method}_seed{cfg.experiment.seed}"
    RunLogger(Path(args.out).parent).log_run(
        run_id=run_id,
        config=cfg,
        result_path=out,
        manifest_paths=[
            split_dir / "train.jsonl",
            split_dir / "test.jsonl",
            split_dir / "manifest.json",
            cfg.absa.cache_path,
        ],
        extra={
            "method": args.method,
            "eval_pass": eval_mode,
            "n_negatives": n_negatives if eval_mode == "sampled" else sampled_n,
            "cumulative_history": cumulative,
            "protocol": result.protocol,
            "eval_protocol": result.eval_protocol,
            "aggregation": result.aggregation,
            "log_file": str(log_path),
        },
    )
    logger.info("wrote %s", out.resolve())

    elapsed = time.monotonic() - t0
    logger.info(
        "=== done in %.1fs: %s rows, %s users ===",
        elapsed,
        result.n_test_rows,
        result.n_test_users,
    )
    primary = result.means
    if result.aggregation == "user_mean":
        primary = result.means_per_user
    for key in sorted(primary):
        logger.info("  %s: %.4f", key, primary[key])
    if result.aggregation == "row_mean" and result.means_per_user:
        logger.info(
            "  (user-mean %s: %.4f)",
            M.AVG_HR_KEY,
            result.means_per_user.get(M.AVG_HR_KEY, 0),
        )

    print(
        f"[run_experiment] {args.method}: "
        f"{result.n_test_rows} rows, {result.n_test_users} users "
        f"eval_protocol={result.eval_protocol} aggregation={result.aggregation} "
        f"({elapsed:.1f}s) log={log_path}"
    )
    for key in sorted(primary):
        print(f"  {key}: {primary[key]:.4f}")
    if result.aggregation == "row_mean" and result.means_per_user:
        print(
            f"  (user-mean {M.AVG_HR_KEY}: "
            f"{result.means_per_user.get(M.AVG_HR_KEY, 0):.4f})"
        )
    if result.sampled:
        sampled_primary = result.sampled["means"]
        if result.sampled.get("aggregation") == "user_mean":
            sampled_primary = result.sampled["means_per_user"]
        logger.info("--- sampled (1+%s negs) ---", result.sampled.get("n_negatives"))
        for key in sorted(sampled_primary):
            if key in ("hr@10", "ndcg@10") or key.endswith("@20"):
                logger.info("  [sampled] %s: %.4f", key, sampled_primary[key])

    print(f"[run_experiment] results written to {out}")


if __name__ == "__main__":
    main()
