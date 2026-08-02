#!/usr/bin/env python3
"""Evaluate Stage 1 TiSASRec on the test split (after make train-emorecagent)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from emorecagent.config import load_config
from emorecagent.eval.runner import load_split_jsonl
from emorecagent.tisasrec_align.checkpoint import load_stage1, load_stage1_id_maps
from emorecagent.tisasrec_align.sequence_data import (
    build_train_pairs,
    build_user_batch_eval_cases,
)
from emorecagent.tisasrec_align.stage1_test_eval import resolve_valid_eval_max_pairs
from emorecagent.tisasrec_align.valid_eval import evaluate_user_batch_cases
from emorecagent.utils.run_log import configure_run_logging


def _resolve_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--out",
        default="results/emorecagent_stage1_test.json",
        help="JSON metrics output path",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="subsample cap on users (default: all; config valid_eval_max_pairs if --use-config-max-pairs)",
    )
    parser.add_argument(
        "--use-config-max-pairs",
        action="store_true",
        help="cap at tisasrec_align.valid_eval_max_pairs (same as valid early-stop subsample)",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="include test rows without verified_purchase",
    )
    args = parser.parse_args()

    logger, log_path = configure_run_logging(
        "test_emorecagent",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )
    logger.info("log file: %s", log_path.resolve())
    t0 = time.monotonic()

    cfg = load_config(args.config)
    ta = cfg.tisasrec_align
    verified_only = cfg.eval.verified_only and not args.include_unverified
    device = _resolve_device(ta.device)

    base = Path(cfg.data.out_dir)
    logger.info("--- stage: load split ---")
    train = load_split_jsonl(base / "train.jsonl")
    valid = load_split_jsonl(base / "valid.jsonl")
    test = load_split_jsonl(base / "test.jsonl")
    logger.info(
        "loaded train=%s valid=%s test=%s verified_test=%s",
        f"{len(train):,}",
        f"{len(valid):,}",
        f"{len(test):,}",
        f"{sum(1 for t in test if t.verified_purchase):,}",
    )

    logger.info("--- stage: load Stage 1 checkpoint ---")
    model, item_ids, _e_i, targs = load_stage1(
        ta.stage1_checkpoint_path,
        ta.e_i_matrix_path,
        device,
    )
    id_maps = load_stage1_id_maps(ta.stage1_checkpoint_path)
    logger.info(
        "checkpoint=%s device=%s items=%s users=%s",
        ta.stage1_checkpoint_path,
        device,
        len(item_ids),
        len(id_maps.user_to_idx),
    )
    history_src = train if ta.test_history == "train" else train + valid
    logger.info("test_history=%s (history rows=%s)", ta.test_history, f"{len(history_src):,}")
    test_cases = build_user_batch_eval_cases(
        history_src,
        test,
        id_maps,
        verified_only=verified_only,
        time_unit_seconds=targs.time_unit_seconds,
    )
    logger.info(
        "user_batch eval: %s users, %s test rows, verified_only=%s",
        f"{len(test_cases):,}",
        f"{sum(c.n_target_rows for c in test_cases):,}",
        verified_only,
    )

    max_users = args.max_users
    if max_users is None and args.use_config_max_pairs:
        max_users = resolve_valid_eval_max_pairs(
            valid_eval_all=False,
            valid_eval_max_pairs=ta.valid_eval_max_pairs,
            n_valid_cases=len(test_cases),
        )
    if max_users is None:
        max_users = len(test_cases)

    train_pairs = build_train_pairs(history_src, id_maps)
    logger.info(
        "--- stage: evaluate test (max_users=%s pool_size=%s) ---",
        f"{max_users:,}",
        ta.pool_size,
    )
    t_eval = time.monotonic()
    metrics = evaluate_user_batch_cases(
        model,
        test_cases,
        item_ids,
        train_pairs,
        device=device,
        pool_size=ta.pool_size,
        max_users=max_users,
        seed=cfg.experiment.seed,
        mask_train_seen=ta.valid_mask_train_seen,
        maxlen=ta.maxlen,
        time_span=ta.time_span,
        eval_batch_size=ta.valid_eval_batch_size,
    )
    logger.info("evaluate finished (%.1fs)", time.monotonic() - t_eval)
    logger.info("test metrics: %s", metrics.format_line())
    logger.info(
        "test hr@1=%.4f hr@3=%.4f hr@5=%.4f hr@10=%.4f hr@20=%.4f "
        "recall@10=%.4f recall@20=%.4f mrr@10=%.4f mrr@20=%.4f "
        "ndcg@10=%.4f ndcg@20=%.4f pool@%s=%.4f n_users=%s/%s n_rows=%s",
        metrics.link_hr_at_1,
        metrics.link_hr_at_3,
        metrics.link_hr_at_5,
        metrics.link_hr_at_10,
        metrics.link_hr_at_20,
        metrics.link_recall_at_10,
        metrics.link_recall_at_20,
        metrics.link_mrr_at_10,
        metrics.link_mrr_at_20,
        metrics.link_ndcg_at_10,
        metrics.link_ndcg_at_20,
        ta.pool_size,
        metrics.pool_recall,
        metrics.n_pairs_eval,
        len(test_cases),
        metrics.n_valid_pairs_total,
    )

    payload = {
        "method": "tisasrec_stage1",
        "eval_protocol": "user_batch",
        "split": str(base),
        "verified_only": verified_only,
        "max_users": max_users,
        "pool_size": ta.pool_size,
        "metrics": asdict(metrics),
        "checkpoint": ta.stage1_checkpoint_path,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s (total %.1fs)", out_path.resolve(), time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
