#!/usr/bin/env python3
"""Run a recommendation experiment over a processed split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from emorecagent.config import load_config
from emorecagent.eval.bootstrap import bootstrap_user_mean_ci
from emorecagent.eval import metrics as M
from emorecagent.eval.runner import (
    build_recommender,
    evaluate,
    load_split_jsonl,
    write_results,
)
from emorecagent.utils.logging import RunLogger
from emorecagent.utils.run_log import configure_run_logging
from emorecagent.utils.seeding import set_global_seed


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
        "candidate_protocol: full_catalog + sampled_negatives=%s",
        "off" if args.no_sampled_eval else (args.n_negatives or cfg.eval.n_negatives),
    )
    logger.info("k_values: %s hr_avg_k: %s", cfg.eval.k_values, cfg.eval.hr_avg_k)
    logger.info("verified_only: %s", verified_only)
    if args.method in ("emorecagent", "emorecagent_fast", "emorecagent_hgt", "aspect_aware"):
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
    if args.method == "emorecagent_hgt":
        logger.info(
            "hgt: pool_size=%s embeddings=%s checkpoint=%s verified_only=%s",
            cfg.hgt.pool_size,
            cfg.hgt.embeddings_dir,
            cfg.hgt.checkpoint_path,
            verified_only,
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
        help="skip the additional 1+N negative-sampling eval pass",
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
        "use_reflection": cfg.ablation.reflection,
        "use_dynamic_weights": cfg.ablation.dynamic_weights,
        "use_aspect_term": cfg.ablation.aspect_term,
        "train_interactions": train,
        "seed": cfg.experiment.seed,
        "use_llm_cot": cfg.experiment.use_llm_cot,
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
    sampled_n = None
    if not args.no_sampled_eval:
        sampled_n = args.n_negatives if args.n_negatives is not None else cfg.eval.n_negatives
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
    )
    logger.info("evaluate finished (%.1fs)", time.monotonic() - t_stage)

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

    _add_bootstrap(result.per_user, result.user_ids, result.ci_per_user, "full")
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
            "n_negatives": sampled_n,
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
