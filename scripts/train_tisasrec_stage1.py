#!/usr/bin/env python3
"""Train pure TiSASRec (Stage 1)."""

from __future__ import annotations

import argparse
import sys
import time

from emorecagent.config import load_config
from emorecagent.tisasrec_align.schema import TiSASRecArgs
from emorecagent.tisasrec_align.train_stage1 import (
    load_test_from_config,
    load_train_valid_from_config,
    train_stage1,
)
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
    args = parser.parse_args()

    logger, log_path = configure_run_logging("train_tisasrec_stage1", log_dir=args.log_dir)
    logger.info("log file: %s", log_path.resolve())
    print(f"[train_tisasrec] log file: {log_path.resolve()}", flush=True)

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    ta = cfg.tisasrec_align
    logger.info("config loaded from %s (%.1fs)", args.config, time.perf_counter() - t0)
    print(f"[train_tisasrec] config loaded ({time.perf_counter() - t0:.1f}s)", flush=True)

    if getattr(ta, "stage1_backend", "era") == "recbole":
        msg = (
            "This config uses tisasrec_align.stage1_backend=recbole "
            f"(category={cfg.data.category}). Do NOT use train_tisasrec_stage1.py "
            "(ERA in-repo trainer). Train/export RecBole Stage-1 with:\n"
            "  python3 scripts/train_yelp_ac_recbole_stage1.py "
            "--python $ERA_PY --log-dir logs/Yelp_AC\n"
            "Or re-export an existing RecBole .pth:\n"
            "  python3 scripts/train_yelp_ac_recbole_stage1.py --skip-train "
            "--checkpoint baseline/RecBole-TiSASRec/checkpoints/TiSASRec-….pth"
        )
        logger.error(msg)
        print(f"[train_tisasrec] ERROR: {msg}", flush=True)
        return 2

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error("torch required (pip install -e '.[torch]')")
        return 1

    t_load = time.perf_counter()
    out_dir = cfg.data.out_dir
    print(f"[train_tisasrec] loading splits from {out_dir} ...", flush=True)
    train, valid = load_train_valid_from_config(out_dir)
    test = load_test_from_config(out_dir)
    load_s = time.perf_counter() - t_load
    msg = (
        f"[train_tisasrec] splits loaded in {load_s:.1f}s "
        f"(train={len(train):,} valid={len(valid):,} test={len(test):,})"
    )
    logger.info(msg)
    print(msg, flush=True)

    targs = TiSASRecArgs(
        maxlen=ta.maxlen,
        hidden_units=ta.hidden_units,
        num_blocks=ta.num_blocks,
        num_heads=ta.num_heads,
        dropout_rate=ta.dropout_rate,
        l2_emb=ta.l2_emb,
        time_span=ta.time_span,
        inner_size=ta.inner_size,
        time_unit_seconds=ta.time_unit_seconds,
    )
    device = _resolve_device(ta.device)
    dev_msg = f"[train_tisasrec] device={device}"
    logger.info(dev_msg)
    print(dev_msg, flush=True)

    result = train_stage1(
        train=train,
        valid=valid,
        args=targs,
        checkpoint_path=ta.stage1_checkpoint_path,
        e_i_matrix_path=ta.e_i_matrix_path,
        device=device,
        epochs=ta.stage1_epochs,
        batch_size=ta.batch_size,
        steps_per_epoch=ta.steps_per_epoch,
        lr=ta.lr,
        early_stop_patience=ta.early_stop_patience,
        early_stop_metric=ta.early_stop_metric,
        pool_size=ta.pool_size,
        valid_mask_train_seen=ta.valid_mask_train_seen,
        require_valid=ta.require_valid,
        valid_eval_all=ta.valid_eval_all,
        valid_eval_max_pairs=ta.valid_eval_max_pairs,
        valid_eval_batch_size=ta.valid_eval_batch_size,
        lr_scheduler_enabled=ta.lr_scheduler_enabled,
        lr_scheduler_patience=ta.lr_scheduler_patience,
        lr_scheduler_factor=ta.lr_scheduler_factor,
        min_lr=ta.min_lr,
        optimizer_name=ta.optimizer,
        weight_decay=ta.weight_decay,
        stage1_loss=ta.stage1_loss,
        num_train_negatives=ta.num_train_negatives,
        test=test,
        verified_only=cfg.eval.verified_only,
        test_history=ta.test_history,
        seed=cfg.experiment.seed,
        run_logger=logger,
    )
    logger.info(
        "done best_epoch=%s metric=%.4f ckpt=%s",
        result.best_epoch,
        result.best_metric,
        result.checkpoint_path,
    )
    if result.post_train_test is not None:
        pt = result.post_train_test
        logger.info(
            "post-train test hr@10=%.4f ndcg@10=%.4f ratio=%.4f",
            pt.test_metrics.link_hr_at_10,
            pt.test_metrics.link_ndcg_at_10,
            pt.hr_ratio,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
