#!/usr/bin/env python3
"""Train HetTiSASRec on chronological train/valid splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emorecagent.config import load_config
from emorecagent.hettisasrec.aspect_graph import build_and_save_aspect_graph
from emorecagent.hettisasrec.aspect_vocab import build_aspect_vocab, save_aspect_vocab
from emorecagent.hettisasrec.model import HetTiSASRecArgs
from emorecagent.hettisasrec.train import load_train_valid_from_config, train_hettisasrec
from emorecagent.sequential.id_maps import build_id_maps_from_interactions
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-aspect-enrichment",
        action="store_true",
        help="Ablation: pure TiSASRec (disable item–aspect MP)",
    )
    args = parser.parse_args()

    logger, log_path = configure_run_logging("train_hettisasrec", log_dir=args.log_dir)
    cfg = load_config(args.config)
    ht = cfg.hettisasrec

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error("torch required (pip install -e '.[torch]')")
        return 1

    train_path = Path(cfg.data.out_dir) / "train.jsonl"
    if not train_path.exists():
        logger.error("train split missing: %s (run make data)", train_path)
        return 1

    train, valid = load_train_valid_from_config(cfg.data.out_dir)
    id_maps = build_id_maps_from_interactions(train, valid)

    vocab_path = Path(ht.aspect_vocab_path)
    if not vocab_path.exists():
        vocab = build_aspect_vocab(
            cfg.absa.cache_path,
            top_k=ht.aspect_top_k,
            min_support=cfg.absa.min_aspect_support,
        )
        save_aspect_vocab(vocab, vocab_path)
        logger.info("wrote aspect vocab: %s", vocab_path)
    else:
        from emorecagent.hettisasrec.aspect_vocab import load_aspect_vocab

        vocab = load_aspect_vocab(vocab_path)

    graph_path = Path(ht.aspect_graph_path)
    if not graph_path.exists():
        graph = build_and_save_aspect_graph(
            train,
            id_maps,
            vocab,
            cfg.absa.cache_path,
            cfg.data.review_path,
            graph_path,
        )
        logger.info("wrote aspect graph: %s", graph_path)
    else:
        from emorecagent.hettisasrec.aspect_graph import AspectGraphBundle

        graph = AspectGraphBundle.load(graph_path)

    device = _resolve_device(args.device or ht.device)
    epochs = args.epochs if args.epochs is not None else ht.epochs
    use_aspect = ht.use_aspect_enrichment and not args.no_aspect_enrichment

    model_args = HetTiSASRecArgs(
        maxlen=ht.maxlen,
        hidden_units=ht.hidden_units,
        num_blocks=ht.num_blocks,
        num_heads=ht.num_heads,
        dropout_rate=ht.dropout_rate,
        l2_emb=ht.l2_emb,
        time_span=ht.time_span,
        use_aspect_enrichment=use_aspect,
        aspect_mp_layers=ht.aspect_mp_layers,
        aspect_loss_weight=ht.aspect_loss_weight,
    )

    logger.info(
        "training HetTiSASRec graph=%s device=%s epochs=%s aspect=%s log=%s",
        graph_path,
        device,
        epochs,
        use_aspect,
        log_path.resolve(),
    )

    result = train_hettisasrec(
        train=train,
        valid=valid,
        graph=graph,
        args=model_args,
        checkpoint_path=ht.checkpoint_path,
        device=device,
        epochs=epochs,
        batch_size=ht.batch_size,
        steps_per_epoch=ht.steps_per_epoch,
        lr=ht.lr,
        early_stop_patience=ht.early_stop_patience,
        early_stop_metric=ht.early_stop_metric,
        pool_size=ht.pool_size,
        valid_mask_train_seen=ht.valid_mask_train_seen,
        require_valid=ht.require_valid,
        valid_eval_max_pairs=ht.valid_eval_max_pairs,
        valid_eval_batch_size=ht.valid_eval_batch_size,
        seed=cfg.experiment.seed,
        run_logger=logger,
    )
    logger.info(
        "done: best_%s=%.4f pool@50=%.4f epoch=%s checkpoint=%s",
        ht.early_stop_metric,
        result.best_metric,
        result.best_metrics.pool_recall,
        result.best_epoch,
        result.checkpoint_path,
    )
    print(
        f"[train_hettisasrec] best_{ht.early_stop_metric}={result.best_metric:.4f} "
        f"pool@50={result.best_metrics.pool_recall:.4f} "
        f"checkpoint={result.checkpoint_path} log={log_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
