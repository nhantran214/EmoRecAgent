#!/usr/bin/env python3
"""Train InfoNCE Alignment MLP for Option B (paper §III.F Eqs. 14–15).

Uses RecBole E_I + item token map exported by
``scripts/train_recbole_stage1_option_b.py``. Does not touch Option A
checkpoints under ``tisasrec_align/`` (see ``configs/legacy/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from emorecagent.config import load_config
from emorecagent.tisasrec_align.text_encoder import HashEncoder, SentenceTransformerEncoder
from emorecagent.tisasrec_align.train_stage1 import load_train_valid_from_config
from emorecagent.tisasrec_align.train_stage2 import train_stage2
from emorecagent.utils.run_log import configure_run_logging


def _resolve_device(name: str) -> torch.device:
    """Alignment training always uses CUDA (no silent CPU fallback)."""
    requested = (name or "cuda").strip().lower()
    if requested in ("auto", "cuda", "gpu"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "Alignment training requires a CUDA GPU "
                f"(tisasrec_align.device={name!r}, torch.cuda.is_available()=False). "
                "Free GPU memory / check drivers, then re-run."
            )
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"Alignment training requires CUDA (requested {name!r}, unavailable)."
            )
        return torch.device(requested)
    raise SystemExit(
        f"Alignment training refuses device={name!r}; use cuda/auto "
        "(CPU training is disabled for Option B)."
    )


def _attach_package_logging(script_logger: logging.Logger) -> None:
    """Route train_stage2 / text_encoder logs to the same file + stdout handlers."""
    for name in (
        "emorecagent.tisasrec_align",
        "emorecagent.tisasrec_align.train_stage2",
        "emorecagent.tisasrec_align.text_encoder",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        lg.propagate = False
        for handler in script_logger.handlers:
            lg.addHandler(handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/categories/Yelp.yaml",
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=None,
        help="Progress log interval within an epoch (default: ~10%% of batches)",
    )
    parser.add_argument(
        "--use-hash-encoder",
        action="store_true",
        help="Override config: train with HashEncoder (ablation)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override tisasrec_align.device (default: cuda; CPU refused)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = args.log_dir or f"logs/{cfg.data.category}"
    logger, log_path = configure_run_logging(
        "train_alignment_stage2_option_b", log_dir=log_dir
    )
    _attach_package_logging(logger)
    logger.info("log_file=%s", log_path.resolve())

    ta = cfg.tisasrec_align
    if args.device:
        ta = ta.model_copy(update={"device": args.device})
        cfg = cfg.model_copy(update={"tisasrec_align": ta})
    allowed = {
        "Beauty_and_Personal_Care",
        "Sports_and_Outdoors",
        "Toys_and_Games",
        "Yelp",
        "Yelp_AC",
    }
    if ta.stage1_backend != "recbole" or cfg.data.category not in allowed:
        logger.error(
            "Option B alignment requires stage1_backend=recbole and category in %s "
            "(got category=%r backend=%r)",
            sorted(allowed),
            cfg.data.category,
            ta.stage1_backend,
        )
        return 1
    if ta.guardrail_mode != "context_dependent":
        logger.warning(
            "config guardrail_mode=%s (expected context_dependent for §III.F)",
            ta.guardrail_mode,
        )

    e_i_path = Path(ta.e_i_matrix_path)
    item_map_path = Path(ta.recbole_bundle_path).with_name("item_token_to_idx.json")
    if not e_i_path.is_file():
        # Fallback beside the RecBole bundle.
        alt = Path(ta.recbole_bundle_path).with_name("e_i_matrix.pt")
        if alt.is_file():
            e_i_path = alt
        else:
            logger.error(
                "missing E_I (%s); re-run train_recbole_stage1_option_b.py",
                e_i_path,
            )
            return 1
    if not item_map_path.is_file():
        logger.error(
            "missing item map %s; re-run Stage-1 Option B export", item_map_path
        )
        return 1

    device = _resolve_device(ta.device or "cuda")
    logger.info(
        "config=%s device=%s cuda_name=%s epochs=%s batch=%s tau=%s e_i=%s tu_cache=%s ckpt=%s",
        args.config,
        device,
        torch.cuda.get_device_name(0) if device.type == "cuda" else "n/a",
        ta.stage2_epochs,
        ta.alignment_batch_size,
        ta.infonce_tau_grid,
        e_i_path,
        ta.tu_cache_path,
        ta.alignment_checkpoint_path,
    )
    logger.info("loading E_I / item map / train split…")
    e_i = torch.load(e_i_path, map_location="cpu", weights_only=True)
    item_to_idx = {
        str(k): int(v) for k, v in json.loads(item_map_path.read_text(encoding="utf-8")).items()
    }
    train, _ = load_train_valid_from_config(cfg.data.out_dir)
    logger.info(
        "loaded train=%s items=%s e_i_shape=%s",
        len(train),
        len(item_to_idx),
        tuple(e_i.shape),
    )
    use_hash = bool(args.use_hash_encoder or ta.use_hash_encoder)
    logger.info(
        "loading text encoder (%s)…",
        "hash" if use_hash else "sentence_transformer",
    )
    encoder = (
        HashEncoder(dim=ta.text_encoder_dim)
        if use_hash
        else SentenceTransformerEncoder(model_device=device)
    )
    logger.info(
        "text encoder ready (st_device=%s); starting InfoNCE…",
        "hash" if use_hash else device,
    )
    try:
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
            item_to_idx=item_to_idx,
            log_every_batches=args.log_every_batches,
        )
    except ValueError as exc:
        msg = str(exc)
        if "no alignment rows" in msg:
            logger.error(
                "%s\n"
                "InfoNCE joins train interactions to T_u by (user_id, timestamp). "
                "Your cache looks test-only — precompute the train split first "
                "(sharded example):\n"
                "  N=8; for i in $(seq 0 $((N-1))); do\n"
                "    python3 scripts/precompute_tu_cache.py \\\n"
                "      --config %s --split train --no-llm \\\n"
                "      --num-shards $N --shard-id $i --log-dir %s &\n"
                "  done; wait\n"
                "  python3 scripts/precompute_tu_cache.py --config %s \\\n"
                "    --merge-shards --num-shards $N --log-dir %s\n"
                "Then re-run this script. (Keep --split test for Stage-2 eval.)",
                msg,
                args.config,
                log_dir,
                args.config,
                log_dir,
            )
            return 1
        raise
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
