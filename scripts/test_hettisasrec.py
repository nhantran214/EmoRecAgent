#!/usr/bin/env python3
"""Evaluate trained HetTiSASRec on the test split (retriever-only, no agents)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from emorecagent.config import load_config, resolve_llm_model
from emorecagent.eval import metrics as M
from emorecagent.eval.bootstrap import bootstrap_user_mean_ci
from emorecagent.eval.runner import (
    build_recommender,
    evaluate,
    load_split_jsonl,
    write_results,
)
from emorecagent.hettisasrec.eval_recommender import HetTiSASRecEvalRecommender
from emorecagent.hettisasrec.sequence_data import (
    build_test_eval_cases,
    build_train_pairs,
)
from emorecagent.hettisasrec.valid_eval import evaluate_valid_cases
from emorecagent.utils.logging import RunLogger
from emorecagent.utils.run_log import configure_run_logging
from emorecagent.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HetTiSASRec checkpoint on test (no LangGraph agents)."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--split",
        default="data/processed/Beauty_and_Personal_Care",
        help="directory with train.jsonl / valid.jsonl / test.jsonl",
    )
    parser.add_argument(
        "--out",
        default="results/hettisasrec_test.json",
        help="JSON output path (same schema as run_experiment.py)",
    )
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
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="concurrent eval rows (overrides eval.parallel_workers)",
    )
    args = parser.parse_args()

    logger, log_path = configure_run_logging(
        "test_hettisasrec",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )
    logger.info("log file: %s", log_path.resolve())
    t0 = time.monotonic()

    cfg = load_config(args.config)
    set_global_seed(cfg.experiment.seed)
    ht = cfg.hettisasrec
    split_dir = Path(args.split)
    cumulative = args.cumulative_history or cfg.eval.cumulative_history
    hr_avg_k = tuple(cfg.eval.hr_avg_k)
    verified_only = cfg.eval.verified_only and not args.include_unverified
    parallel_workers = (
        args.parallel_workers
        if args.parallel_workers is not None
        else cfg.eval.parallel_workers
    )

    logger.info("=== HetTiSASRec test eval (retriever-only) ===")
    logger.info("config: %s", cfg.experiment.name)
    logger.info("split: %s", split_dir)
    logger.info("out: %s", args.out)
    logger.info(
        "eval_protocol: %s aggregation: %s verified_only: %s parallel_workers: %s",
        cfg.eval.protocol,
        cfg.eval.aggregation,
        verified_only,
        parallel_workers,
    )
    logger.info(
        "hettisasrec: pool_size=%s checkpoint=%s aspect_graph=%s",
        ht.pool_size,
        ht.checkpoint_path,
        ht.aspect_graph_path,
    )
    logger.info(
        "llm: model=%s endpoint=%s (not used in this eval)",
        resolve_llm_model(cfg.llm),
        cfg.tgi.base_url,
    )

    logger.info("--- stage: load split ---")
    t_stage = time.monotonic()
    train = load_split_jsonl(split_dir / "train.jsonl")
    valid = load_split_jsonl(split_dir / "valid.jsonl")
    test = load_split_jsonl(split_dir / "test.jsonl")
    n_verified_test = sum(1 for t in test if t.verified_purchase)
    logger.info(
        "loaded train=%s valid=%s test=%s verified_test=%s (%.1fs)",
        f"{len(train):,}",
        f"{len(valid):,}",
        f"{len(test):,}",
        f"{n_verified_test:,}",
        time.monotonic() - t_stage,
    )

    runner_cfg = {
        "train_interactions": train,
        "app_config": cfg,
    }

    logger.info("--- stage: build recommender ---")
    t_stage = time.monotonic()
    recommender = build_recommender("hettisasrec", runner_cfg, seed=cfg.experiment.seed)
    assert isinstance(recommender, HetTiSASRecEvalRecommender)
    logger.info(
        "recommender=%s checkpoint=%s (%.1fs)",
        recommender.name,
        ht.checkpoint_path,
        time.monotonic() - t_stage,
    )

    logger.info("--- stage: evaluate ---")
    t_stage = time.monotonic()
    max_test_rows = args.max_test_rows
    if max_test_rows is None:
        max_test_rows = cfg.experiment.max_test_rows
    sampled_n = None
    if not args.no_sampled_eval:
        sampled_n = args.n_negatives if args.n_negatives is not None else cfg.eval.n_negatives
    result = evaluate(
        recommender,
        train,
        test,
        k_values=cfg.eval.k_values,
        method="hettisasrec",
        seed=cfg.experiment.seed,
        cumulative_history=cumulative,
        hr_avg_k=hr_avg_k,
        show_progress=not args.no_progress_bar,
        progress_logger=logger,
        progress_interval=args.progress_interval,
        verified_only=verified_only,
        max_test_rows=max_test_rows,
        eval_protocol=cfg.eval.protocol,
        aggregation=cfg.eval.aggregation,
        sampled_n_negatives=sampled_n,
        sampled_k_values=cfg.eval.sampled_k_values,
        parallel_workers=parallel_workers,
    )
    logger.info("evaluate finished (%.1fs)", time.monotonic() - t_stage)

    logger.info("--- stage: retriever pool@%s on test ---", ht.pool_size)
    t_stage = time.monotonic()
    retriever = recommender.retriever
    id_maps = retriever.id_maps
    test_cases = build_test_eval_cases(train, valid, test, id_maps)
    if verified_only:
        verified_items = {
            (it.user_id, it.item)
            for it in test
            if it.verified_purchase
        }
        filtered = []
        for case in test_cases:
            uid = id_maps.idx_to_user.get(case.user_local)
            iid = id_maps.idx_to_item.get(case.gold_local)
            if uid is not None and iid is not None and (uid, iid) in verified_items:
                filtered.append(case)
        test_cases = filtered
    train_pairs = build_train_pairs(train + valid, id_maps)
    pool_metrics = evaluate_valid_cases(
        retriever.model,
        test_cases,
        retriever.item_ids,
        train_pairs,
        device=retriever.device,
        pool_size=ht.pool_size,
        max_pairs=ht.valid_eval_max_pairs,
        seed=cfg.experiment.seed,
        mask_train_seen=ht.valid_mask_train_seen,
        maxlen=ht.maxlen,
        time_span=ht.time_span,
        eval_batch_size=ht.valid_eval_batch_size,
    )
    pool_block = {
        "pool_size": ht.pool_size,
        "pool_recall": pool_metrics.pool_recall,
        "link_mrr": pool_metrics.link_mrr,
        "link_recall_at_20": pool_metrics.link_recall_at_20,
        "link_ndcg_at_10": pool_metrics.link_ndcg_at_10,
        "n_pairs_eval": pool_metrics.n_pairs_eval,
        "n_test_pairs_total": pool_metrics.n_valid_pairs_total,
    }
    logger.info(
        "test pool@%s=%.4f mrr=%.4f recall@20=%.4f ndcg@10=%.4f "
        "n_pairs=%s/%s (%.1fs)",
        ht.pool_size,
        pool_metrics.pool_recall,
        pool_metrics.link_mrr,
        pool_metrics.link_recall_at_20,
        pool_metrics.link_ndcg_at_10,
        pool_metrics.n_pairs_eval,
        pool_metrics.n_valid_pairs_total,
        time.monotonic() - t_stage,
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
    payload = result.to_json()
    payload["retriever_pool"] = pool_block
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    run_id = f"hettisasrec_test_seed{cfg.experiment.seed}"
    RunLogger(out.parent).log_run(
        run_id=run_id,
        config=cfg,
        result_path=out,
        manifest_paths=[
            split_dir / "train.jsonl",
            split_dir / "valid.jsonl",
            split_dir / "test.jsonl",
            split_dir / "manifest.json",
            cfg.absa.cache_path,
            Path(ht.checkpoint_path),
            Path(ht.aspect_graph_path),
        ],
        extra={
            "method": "hettisasrec",
            "mode": "retriever_only",
            "n_negatives": sampled_n,
            "cumulative_history": cumulative,
            "protocol": result.protocol,
            "eval_protocol": result.eval_protocol,
            "aggregation": result.aggregation,
            "retriever_pool": pool_block,
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

    print(
        f"[test_hettisasrec] pool@{ht.pool_size}={pool_metrics.pool_recall:.4f} "
        f"full_catalog avg_hr@1,3,5={primary.get(M.AVG_HR_KEY, 0):.4f} "
        f"rows={result.n_test_rows} out={out} log={log_path}"
    )


if __name__ == "__main__":
    main()
