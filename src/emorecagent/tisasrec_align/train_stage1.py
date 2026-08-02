"""Stage 1 training loop for pure TiSASRec."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from ..data.loader import load_split_jsonl
from ..data.types import Interaction
from ..sequential.id_maps import IdMaps, build_id_maps_from_interactions
from .checkpoint import load_stage1
from .model import TiSASRecModel, init_model_weights
from .schema import TiSASRecArgs
from .sequence_data import (
    build_train_pairs,
    build_user_batch_eval_cases,
    build_user_sequences,
    sample_batch,
)
from .stage1_test_eval import (
    PostTrainTestSummary,
    resolve_steps_per_epoch,
    resolve_valid_eval_max_pairs,
    run_post_train_test_eval,
)
from .valid_eval import (
    ValidMetrics,
    evaluate_user_batch_cases,
    normalize_early_stop_metric,
    popularity_pool_recall_user_batch,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainStage1Result:
    checkpoint_path: Path
    e_i_matrix_path: Path
    best_epoch: int
    best_metric: float
    best_metrics: ValidMetrics
    post_train_test: PostTrainTestSummary | None = None


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


def _multi_bce_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    valid = pos_logits != 0
    if not valid.any():
        return pos_logits.sum() * 0.0
    pos_loss = loss_fn(pos_logits, torch.ones_like(pos_logits))
    neg_loss = loss_fn(neg_logits, torch.zeros_like(neg_logits))
    total = pos_loss + neg_loss
    return total[valid].mean()


def _bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    if neg_logits.dim() == 2:
        neg_logits = neg_logits.unsqueeze(-1)
    pos_exp = pos_logits.unsqueeze(-1)
    valid = pos_logits != 0
    if not valid.any():
        return pos_logits.sum() * 0.0
    diff = pos_exp - neg_logits
    loss = -torch.nn.functional.logsigmoid(diff)
    return loss[valid.unsqueeze(-1).expand_as(loss)].mean()


def compute_stage1_loss(
    stage1_loss: Literal["bce", "multi_bce", "bpr", "ce"],
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
) -> torch.Tensor:
    if stage1_loss == "ce":
        raise ValueError(
            "stage1_loss='ce' requires compute_ce_loss(model, seq, time, pos); "
            "do not call compute_stage1_loss for CE"
        )
    if stage1_loss == "bce":
        return _bce_loss(pos_logits, neg_logits)
    if neg_logits.dim() == 2:
        neg_logits = neg_logits.unsqueeze(-1)
    if stage1_loss == "multi_bce":
        pos_exp = pos_logits.unsqueeze(-1).expand_as(neg_logits)
        return _multi_bce_loss(pos_exp, neg_logits)
    if stage1_loss == "bpr":
        return _bpr_loss(pos_logits, neg_logits)
    raise ValueError(f"unknown stage1_loss={stage1_loss!r}")


def compute_ce_loss(
    model: TiSASRecModel,
    log_seqs: torch.Tensor,
    time_matrices: torch.Tensor,
    pos_seqs: torch.Tensor,
) -> torch.Tensor:
    """Full-softmax CE over the item catalog at every non-pad position.

    Equivalent to RecBole ``loss_type: CE``: RecBole augments each user
    sequence into one row per prefix and takes CE at that row's last position,
    so its target count equals the interaction count. This repo packs one row
    per user (Kang-style) with a target at every timestep, which yields the
    same set of (prefix -> next item) targets in a single pass. Supervising
    only the last position here would drop ~8x of that signal.
    """
    log_feats = model.seq2feats(log_seqs, time_matrices)
    # [B, T, item_num+1] including padding row 0
    logits = log_feats @ model.item_emb.weight.transpose(0, 1)
    logits = logits.clone()
    logits[:, :, 0] = -1.0e9
    valid = pos_seqs.ne(0)
    if not valid.any():
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[valid], pos_seqs[valid])


def apply_lr_scheduler_step(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    optimizer: torch.optim.Optimizer,
    metric: float,
    stale: int,
) -> tuple[int, bool, float]:
    """Step plateau scheduler; reset stale when LR drops."""
    if scheduler is None:
        return stale, False, float(optimizer.param_groups[0]["lr"])
    old_lr = float(optimizer.param_groups[0]["lr"])
    scheduler.step(metric)
    new_lr = float(optimizer.param_groups[0]["lr"])
    if new_lr < old_lr - 1e-12:
        return 0, True, new_lr
    return stale, False, new_lr


def _emit(log: logging.Logger, msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


def _train_step_log_interval(n_steps: int) -> int:
    if n_steps <= 5:
        return 1
    return max(1, n_steps // 10)


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
    lr: float,
) -> None:
    msg = (
        f"[train_tisasrec] epoch={epoch} loss={mean_loss:.4f} lr={lr:.2e} "
        f"{early_stop_metric}={stop_val:.4f} {metrics.format_line()} "
        f"best={best_metric:.4f} stale={stale}/{patience} "
        f"n_eval={metrics.n_pairs_eval}/{metrics.n_valid_pairs_total} "
        f"train={train_s:.1f}s eval={eval_s:.1f}s"
    )
    _emit(log, msg)


def _item_ids_from_maps(id_maps: IdMaps) -> list[str]:
    idx_to_item = {v: k for k, v in id_maps.item_to_idx.items()}
    return [idx_to_item[i] for i in range(1, len(idx_to_item) + 1)]


def _build_optimizer(
    model: TiSASRecModel,
    *,
    optimizer_name: Literal["adam", "adamw"],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))


def train_stage1(
    *,
    train: list[Interaction],
    valid: list[Interaction],
    args: TiSASRecArgs,
    checkpoint_path: str | Path,
    e_i_matrix_path: str | Path,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 2048,
    steps_per_epoch: int | None = None,
    lr: float = 0.001,
    early_stop_patience: int = 10,
    early_stop_metric: str = "valid_link_hr@10",
    pool_size: int = 50,
    valid_mask_train_seen: bool = True,
    require_valid: bool = True,
    valid_eval_all: bool = True,
    valid_eval_max_pairs: int = 2048,
    valid_eval_batch_size: int = 64,
    lr_scheduler_enabled: bool = True,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.5,
    min_lr: float = 1e-5,
    optimizer_name: Literal["adam", "adamw"] = "adam",
    weight_decay: float = 0.0,
    stage1_loss: Literal["bce", "multi_bce", "bpr", "ce"] = "multi_bce",
    num_train_negatives: int = 5,
    test: list[Interaction] | None = None,
    verified_only: bool = True,
    test_history: str = "train",
    seed: int = 42,
    run_logger: logging.Logger | None = None,
) -> TrainStage1Result:
    log = run_logger or logger
    t_setup = time.perf_counter()
    _emit(log, "[train_tisasrec] preparing data (id maps, sequences, valid cases)...")
    normalize_early_stop_metric(early_stop_metric)
    # Id maps cover train/valid/(test) so held-out items stay in the catalog.
    # Fit sequences are train-only so valid remains a true early-stop holdout.
    map_inputs = [train, valid]
    if test is not None:
        map_inputs.append(test)
    id_maps = build_id_maps_from_interactions(*map_inputs)
    item_num = len(id_maps.item_to_idx)
    item_ids = _item_ids_from_maps(id_maps)

    user_train = build_user_sequences(
        train, id_maps, time_unit_seconds=args.time_unit_seconds
    )
    valid_cases = build_user_batch_eval_cases(
        train, valid, id_maps, time_unit_seconds=args.time_unit_seconds
    )
    train_pairs = build_train_pairs(train, id_maps)
    eval_max_users = resolve_valid_eval_max_pairs(
        valid_eval_all=valid_eval_all,
        valid_eval_max_pairs=valid_eval_max_pairs,
        n_valid_cases=len(valid_cases),
    )
    n_steps = resolve_steps_per_epoch(len(user_train), batch_size, steps_per_epoch)
    train_k = 1 if stage1_loss in ("bce", "ce") else num_train_negatives
    _emit(
        log,
        f"[train_tisasrec] data ready in {time.perf_counter() - t_setup:.1f}s "
        f"(items={item_num} users_train={len(user_train)} "
        f"valid_users={len(valid_cases)} eval_users={eval_max_users} "
        f"fit=train_only)",
    )

    if require_valid and not valid_cases:
        raise ValueError(
            "valid split has no user-batch eval cases; rebuild dataset with "
            "valid.jsonl or set tisasrec_align.require_valid=false"
        )

    pop_max_users = min(eval_max_users, 2048)
    _emit(
        log,
        f"[train_tisasrec] computing popularity baseline "
        f"({pop_max_users}/{len(valid_cases)} users)...",
    )
    t_pop = time.perf_counter()
    pop_recall = popularity_pool_recall_user_batch(
        valid_cases,
        train_pairs,
        pool_size=pool_size,
        max_users=pop_max_users,
        seed=seed,
        mask_train_seen=valid_mask_train_seen,
    )
    _emit(log, f"[train_tisasrec] baseline done in {time.perf_counter() - t_pop:.1f}s")
    baseline_msg = (
        f"[train_tisasrec] baseline popularity pool@{pool_size}={pop_recall:.4f} "
        f"steps/epoch={n_steps} batch={batch_size} loss={stage1_loss} K={train_k} "
        f"device={device}"
    )
    _emit(log, baseline_msg)

    t_model = time.perf_counter()
    model = TiSASRecModel(item_num, args).to(device)
    init_model_weights(model)
    opt = _build_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if lr_scheduler_enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=min_lr,
        )

    _emit(log, f"[train_tisasrec] model ready in {time.perf_counter() - t_model:.1f}s")

    ckpt_path = Path(checkpoint_path)
    e_i_path = Path(e_i_matrix_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    e_i_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    best_epoch = 0
    best_metrics = ValidMetrics.empty(pool_size=pool_size)
    stale = 0

    torch.manual_seed(seed)
    step_log_every = _train_step_log_interval(n_steps)

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        t_train = time.perf_counter()
        _emit(
            log,
            f"[train_tisasrec] epoch {epoch + 1}/{epochs} training "
            f"({n_steps} steps, log every {step_log_every})...",
        )
        for step in range(n_steps):
            _users, seqs, time_mats, poss, negs = sample_batch(
                user_train,
                item_num=item_num,
                maxlen=args.maxlen,
                time_span=args.time_span,
                batch_size=batch_size,
                num_negatives=train_k,
            )
            seq_t = torch.tensor(seqs, dtype=torch.long, device=device)
            time_t = torch.tensor(time_mats, dtype=torch.long, device=device)
            pos_t = torch.tensor(poss, dtype=torch.long, device=device)
            if stage1_loss == "ce":
                loss = compute_ce_loss(model, seq_t, time_t, pos_t)
            else:
                neg_t = torch.tensor(negs, dtype=torch.long, device=device)
                pos_logits, neg_logits = model(seq_t, time_t, pos_t, neg_t)
                loss = compute_stage1_loss(stage1_loss, pos_logits, neg_logits)
            loss = loss + model.l2_regularization()

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            step_no = step + 1
            if step_no == 1 or step_no == n_steps or step_no % step_log_every == 0:
                _emit(
                    log,
                    f"[train_tisasrec] epoch {epoch + 1} step {step_no}/{n_steps} "
                    f"loss={losses[-1]:.4f}",
                )
        train_s = time.perf_counter() - t_train

        _emit(
            log,
            f"[train_tisasrec] epoch {epoch + 1} valid eval "
            f"({eval_max_users}/{len(valid_cases)} users)...",
        )
        t_eval = time.perf_counter()
        metrics = evaluate_user_batch_cases(
            model,
            valid_cases,
            item_ids,
            train_pairs,
            device=device,
            pool_size=pool_size,
            max_users=eval_max_users,
            seed=seed + epoch,
            mask_train_seen=valid_mask_train_seen,
            maxlen=args.maxlen,
            time_span=args.time_span,
            eval_batch_size=valid_eval_batch_size,
        )
        eval_s = time.perf_counter() - t_eval
        stop_val = metrics.early_stop_value(early_stop_metric)
        mean_loss = sum(losses) / len(losses)
        current_lr = float(opt.param_groups[0]["lr"])

        if stop_val > best_metric:
            best_metric = stop_val
            best_epoch = epoch + 1
            best_metrics = metrics
            stale = 0
            e_i = model.all_item_embeddings().detach().cpu()
            torch.save(e_i, e_i_path)
            payload = {
                "model": model.state_dict(),
                "meta": {
                    "item_num": item_num,
                    "args": asdict(args),
                    "item_ids": item_ids,
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

        stale, lr_reduced, current_lr = apply_lr_scheduler_step(
            scheduler, opt, stop_val, stale
        )
        if lr_reduced:
            lr_msg = f"[train_tisasrec] lr reduced -> {current_lr:.2e}"
            log.info(lr_msg)
            print(lr_msg, flush=True)

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
            lr=current_lr,
        )
        if saved:
            save_msg = (
                f"[train_tisasrec] saved checkpoint epoch={epoch + 1} -> {ckpt_path}"
            )
            log.info(save_msg)
            print(save_msg, flush=True)

        if stale >= early_stop_patience:
            stop_msg = (
                f"[train_tisasrec] early stop epoch={epoch + 1} "
                f"best_{early_stop_metric}={best_metric:.4f} at epoch={best_epoch}"
            )
            log.info(stop_msg)
            print(stop_msg, flush=True)
            break

    post_train_test: PostTrainTestSummary | None = None
    if test is not None and ckpt_path.is_file():
        eval_model, eval_item_ids, _e_i, _targs = load_stage1(
            ckpt_path, e_i_path, device
        )
        post_train_test = run_post_train_test_eval(
            eval_model,
            train=train,
            valid=valid,
            test=test,
            id_maps=id_maps,
            item_ids=eval_item_ids,
            device=device,
            verified_only=verified_only,
            valid_hr_at_10=best_metrics.link_hr_at_10,
            pool_size=pool_size,
            mask_train_seen=valid_mask_train_seen,
            maxlen=args.maxlen,
            time_span=args.time_span,
            eval_batch_size=valid_eval_batch_size,
            seed=seed,
            test_history=test_history,
            time_unit_seconds=args.time_unit_seconds,
        )
        tm = post_train_test.test_metrics
        summary_msg = (
            f"[train_tisasrec] post-train test hr@10={tm.link_hr_at_10:.4f} "
            f"ndcg@10={tm.link_ndcg_at_10:.4f} "
            f"valid_hr@10={post_train_test.valid_hr_at_10:.4f} "
            f"ratio={post_train_test.hr_ratio:.4f}"
        )
        log.info(summary_msg)
        print(summary_msg, flush=True)

    return TrainStage1Result(
        checkpoint_path=ckpt_path,
        e_i_matrix_path=e_i_path,
        best_epoch=best_epoch,
        best_metric=best_metric,
        best_metrics=best_metrics,
        post_train_test=post_train_test,
    )


def load_train_valid_from_config(
    out_dir: str | Path,
) -> tuple[list[Interaction], list[Interaction]]:
    base = Path(out_dir)
    train = load_split_jsonl(base / "train.jsonl")
    valid = load_split_jsonl(base / "valid.jsonl")
    return train, valid


def load_test_from_config(out_dir: str | Path) -> list[Interaction]:
    return load_split_jsonl(Path(out_dir) / "test.jsonl")
