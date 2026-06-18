#!/usr/bin/env python3
"""Train HGT link-prediction model (BPR) on a pre-built graph."""

from __future__ import annotations

import argparse
import sys

from emorecagent.config import load_config
from emorecagent.hgt.graph_data import HgtGraphBundle
from emorecagent.hgt.train import train_hgt
from emorecagent.utils.run_log import configure_run_logging


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logger, log_path = configure_run_logging("train_hgt", log_dir=args.log_dir)
    cfg = load_config(args.config)
    graph_path = cfg.hgt.graph_path
    try:
        bundle = HgtGraphBundle.load(graph_path)
    except FileNotFoundError:
        logger.error("graph missing: %s (run make build-hgt-graph)", graph_path)
        return 1
    except ImportError as exc:
        logger.error("%s (install: pip install -e '.[hgt]')", exc)
        return 1

    device = _resolve_device(args.device or cfg.hgt.device)
    epochs = args.epochs if args.epochs is not None else cfg.hgt.epochs
    logger.info(
        "training HGT graph=%s device=%s epochs=%s log=%s",
        graph_path,
        device,
        epochs,
        log_path.resolve(),
    )

    result = train_hgt(
        bundle,
        checkpoint_path=cfg.hgt.checkpoint_path,
        embeddings_dir=cfg.hgt.embeddings_dir,
        n_hid=cfg.hgt.n_hid,
        n_heads=cfg.hgt.n_heads,
        n_layers=cfg.hgt.n_layers,
        dropout=cfg.hgt.dropout,
        use_RTE=cfg.hgt.use_RTE,
        lr=cfg.hgt.lr,
        epochs=epochs,
        batch_size=cfg.hgt.batch_size,
        neg_samples=cfg.hgt.neg_samples,
        early_stop_patience=cfg.hgt.early_stop_patience,
        device=device,
        seed=cfg.experiment.seed,
    )
    logger.info(
        "done: best_valid_mrr=%.4f epochs=%s checkpoint=%s embeddings=%s",
        result.best_valid_mrr,
        result.epochs_run,
        result.checkpoint_path,
        result.embeddings_dir,
    )
    print(
        f"[train_hgt] mrr={result.best_valid_mrr:.4f} "
        f"checkpoint={result.checkpoint_path} log={log_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
