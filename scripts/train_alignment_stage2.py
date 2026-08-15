#!/usr/bin/env python3
"""Train alignment MLP with InfoNCE (Stage 2)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from emorecagent.config import load_config
from emorecagent.tisasrec_align.text_encoder import HashEncoder
from emorecagent.tisasrec_align.train_stage1 import load_train_valid_from_config
from emorecagent.tisasrec_align.train_stage2 import train_stage2
from emorecagent.utils.run_log import configure_run_logging


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _attach_package_logging(script_logger: logging.Logger) -> None:
    for name in (
        "emorecagent.tisasrec_align",
        "emorecagent.tisasrec_align.train_stage2",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        lg.propagate = False
        for handler in script_logger.handlers:
            lg.addHandler(handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--use-hash-encoder", action="store_true")
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=None,
        help="Progress log interval within an epoch (default: ~10%% of batches)",
    )
    args = parser.parse_args()

    logger, log_path = configure_run_logging(
        "train_alignment_stage2", log_dir=args.log_dir
    )
    _attach_package_logging(logger)
    logger.info("log_file=%s", log_path.resolve())
    cfg = load_config(args.config)
    ta = cfg.tisasrec_align
    device = _resolve_device(ta.device)

    if not Path(ta.e_i_matrix_path).exists():  # noqa: F821
        logger.error("missing E_I matrix: %s (run Stage 1 first)", ta.e_i_matrix_path)
        return 1
    e_i = torch.load(ta.e_i_matrix_path, map_location="cpu", weights_only=True)

    train, _ = load_train_valid_from_config(cfg.data.out_dir)
    encoder = HashEncoder(dim=ta.text_encoder_dim) if args.use_hash_encoder else None
    result = train_stage2(
        train=train,
        tu_cache_path=ta.tu_cache_path,
        e_i_matrix=e_i,
        hidden_dim=ta.hidden_units,
        alignment_ckpt_path=ta.alignment_checkpoint_path,
        device=device,
        text_encoder=encoder,
        tau_grid=tuple(ta.infonce_tau_grid),
        epochs=ta.stage2_epochs,
        batch_size=ta.alignment_batch_size,
        lr=ta.alignment_lr,
        seed=cfg.experiment.seed,
        activation=ta.alignment_activation,
        log_every_batches=args.log_every_batches,
    )
    logger.info(
        "done tau=%.3f loss=%.4f ckpt=%s log_file=%s",
        result.best_tau,
        result.best_loss,
        result.checkpoint_path,
        log_path.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
