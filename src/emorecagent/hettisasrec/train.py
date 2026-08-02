"""Training loop for HetTiSASRec."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ..data.types import Interaction
from ..eval.runner import load_split_jsonl
from ..sequential.id_maps import build_id_maps_from_interactions
from .aspect_graph import AspectGraphBundle
from .model import HetTiSASRecArgs, HetTiSASRecModel
from .sequence_data import (
    build_train_pairs,
    build_user_sequences,
    build_valid_eval_cases,
    sample_batch,
)
from .valid_eval import (
    ValidMetrics,
    evaluate_valid_cases,
    normalize_early_stop_metric,
    popularity_pool_recall,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainResult:
    checkpoint_path: Path
    best_epoch: int
    best_metric: float
    best_metrics: ValidMetrics


def _bce_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)
    loss_fn = nn.BCEWithLogitsLoss()
    valid = pos_logits != 0
    if not valid.any():
        return pos_logits.sum() * 0.0
    return loss_fn(pos_logits[valid], pos_labels[valid]) + loss_fn(
        neg_logits[valid], neg_labels[valid]
    )


def _log_epoch(
    log: logging.Logger,
    *,
    epoch: int,
    mean_loss: float,
    metrics: ValidMetrics,
    early_stop_metric: str,
    stop_val: float,
    best_metric: float,
    stale: int,
    patience: int,
    train_s: float,
    eval_s: float,
) -> None:
    msg = (
        f"[train_hettisasrec] epoch={epoch} loss={mean_loss:.4f} "
        f"{early_stop_metric}={stop_val:.4f} {metrics.format_line()} "
        f"best={best_metric:.4f} stale={stale}/{patience} "
        f"n_eval={metrics.n_pairs_eval}/{metrics.n_valid_pairs_total} "
        f"train={train_s:.1f}s eval={eval_s:.1f}s"
    )
    log.info(msg)
    print(msg, flush=True)


def train_hettisasrec(
    *,
    train: list[Interaction],
    valid: list[Interaction],
    graph: AspectGraphBundle,
    args: HetTiSASRecArgs,
    checkpoint_path: str | Path,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 256,
    steps_per_epoch: int = 800,
    lr: float = 0.001,
    early_stop_patience: int = 10,
    early_stop_metric: str = "valid_pool_recall@50",
    pool_size: int = 50,
    valid_mask_train_seen: bool = True,
    require_valid: bool = True,
    valid_eval_max_pairs: int = 2048,
    valid_eval_batch_size: int = 64,
    seed: int = 42,
    run_logger: logging.Logger | None = None,
) -> TrainResult:
    log = run_logger or logger
    normalize_early_stop_metric(early_stop_metric)
    id_maps = build_id_maps_from_interactions(train, valid)
    user_num = len(id_maps.user_to_idx)
    item_num = len(id_maps.item_to_idx)

    user_train = build_user_sequences(train, id_maps)
    valid_cases = build_valid_eval_cases(train, valid, id_maps)
    train_pairs = build_train_pairs(train, id_maps)

    if require_valid and not valid_cases:
        raise ValueError(
            "valid split has no chronological eval cases; rebuild dataset with "
            "valid.jsonl or set hettisasrec.require_valid=false"
        )

    pop_recall = popularity_pool_recall(
        valid_cases,
        train_pairs,
        list(graph.item_ids),
        pool_size=pool_size,
        max_pairs=valid_eval_max_pairs,
        seed=seed,
        mask_train_seen=valid_mask_train_seen,
    )
    baseline_msg = (
        f"[train_hettisasrec] baseline popularity pool@{pool_size}={pop_recall:.4f} "
        f"(n_cases={len(valid_cases)} users_train={len(user_train)})"
    )
    log.info(baseline_msg)
    print(baseline_msg, flush=True)

    model = HetTiSASRecModel(user_num, item_num, args, graph).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    best_epoch = 0
    best_metrics = ValidMetrics.empty(pool_size=pool_size)
    stale = 0

    torch.manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        n_steps = max(1, steps_per_epoch)
        t_train = time.perf_counter()
        for _ in range(n_steps):
            _users, seqs, time_mats, poss, negs = sample_batch(
                user_train,
                item_num=item_num,
                maxlen=args.maxlen,
                time_span=args.time_span,
                batch_size=batch_size,
            )
            seq_t = torch.tensor(seqs, dtype=torch.long, device=device)
            time_t = torch.tensor(time_mats, dtype=torch.long, device=device)
            pos_t = torch.tensor(poss, dtype=torch.long, device=device)
            neg_t = torch.tensor(negs, dtype=torch.long, device=device)

            pos_logits, neg_logits = model(seq_t, time_t, pos_t, neg_t)
            loss = _bce_loss(pos_logits, neg_logits) + model.l2_regularization()

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_s = time.perf_counter() - t_train

        t_eval = time.perf_counter()
        metrics = evaluate_valid_cases(
            model,
            valid_cases,
            list(graph.item_ids),
            train_pairs,
            device=device,
            pool_size=pool_size,
            max_pairs=valid_eval_max_pairs,
            seed=seed + epoch,
            mask_train_seen=valid_mask_train_seen,
            maxlen=args.maxlen,
            time_span=args.time_span,
            eval_batch_size=valid_eval_batch_size,
        )
        eval_s = time.perf_counter() - t_eval
        stop_val = metrics.early_stop_value(early_stop_metric)
        mean_loss = sum(losses) / len(losses)

        if stop_val > best_metric:
            best_metric = stop_val
            best_epoch = epoch + 1
            best_metrics = metrics
            stale = 0
            payload = {
                "model": model.state_dict(),
                "meta": {
                    "user_num": user_num,
                    "item_num": item_num,
                    "args": asdict(args),
                    "item_ids": list(graph.item_ids),
                    "aspect_ids": list(graph.aspect_ids),
                    "id_maps": {
                        "user_to_idx": id_maps.user_to_idx,
                        "item_to_idx": id_maps.item_to_idx,
                    },
                },
            }
            torch.save(payload, ckpt_path)
            saved = " *best*"
        else:
            stale += 1
            saved = ""

        _log_epoch(
            log,
            epoch=epoch + 1,
            mean_loss=mean_loss,
            metrics=metrics,
            early_stop_metric=early_stop_metric,
            stop_val=stop_val,
            best_metric=best_metric,
            stale=stale,
            patience=early_stop_patience,
            train_s=train_s,
            eval_s=eval_s,
        )
        if saved:
            save_msg = f"[train_hettisasrec] saved checkpoint epoch={epoch + 1} -> {ckpt_path}"
            log.info(save_msg)
            print(save_msg, flush=True)

        if stale >= early_stop_patience:
            stop_msg = (
                f"[train_hettisasrec] early stop epoch={epoch + 1} "
                f"best_{early_stop_metric}={best_metric:.4f} at epoch={best_epoch}"
            )
            log.info(stop_msg)
            print(stop_msg, flush=True)
            break

    return TrainResult(
        checkpoint_path=ckpt_path,
        best_epoch=best_epoch,
        best_metric=best_metric,
        best_metrics=best_metrics,
    )


def load_train_valid_from_config(
    out_dir: str | Path,
) -> tuple[list[Interaction], list[Interaction]]:
    base = Path(out_dir)
    train = load_split_jsonl(base / "train.jsonl")
    valid = load_split_jsonl(base / "valid.jsonl")
    return train, valid
