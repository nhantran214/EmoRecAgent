"""Post-train Stage 1 test evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.types import Interaction
from ..sequential.id_maps import IdMaps
from .model import TiSASRecModel
from .sequence_data import UserBatchEvalCase, build_train_pairs, build_user_batch_eval_cases
from .valid_eval import ValidMetrics, evaluate_user_batch_cases


@dataclass(frozen=True, slots=True)
class PostTrainTestSummary:
    test_metrics: ValidMetrics
    valid_hr_at_10: float
    hr_ratio: float


def filter_verified_test_cases(
    test_cases: list[UserBatchEvalCase],
    test: list[Interaction],
    id_maps: IdMaps,
) -> list[UserBatchEvalCase]:
    """Legacy helper; prefer ``build_user_batch_eval_cases(..., verified_only=True)``."""
    verified_items = {(it.user_id, it.item) for it in test if it.verified_purchase}
    filtered: list[UserBatchEvalCase] = []
    idx_to_user = id_maps.idx_to_user
    idx_to_item = id_maps.idx_to_item
    for case in test_cases:
        uid = idx_to_user.get(case.user_local)
        if uid is None:
            continue
        keep_relevant: set[int] = set()
        for loc in case.relevant_locals:
            iid = idx_to_item.get(loc)
            if iid is not None and (uid, iid) in verified_items:
                keep_relevant.add(loc)
        if keep_relevant:
            filtered.append(
                UserBatchEvalCase(
                    user_local=case.user_local,
                    relevant_locals=frozenset(keep_relevant),
                    history=case.history,
                    n_target_rows=len(keep_relevant),
                )
            )
    return filtered


def resolve_valid_eval_max_pairs(
    *,
    valid_eval_all: bool,
    valid_eval_max_pairs: int,
    n_valid_cases: int,
) -> int:
    """Cap for user-batch eval units (users when ``valid_eval_all=false``)."""
    if valid_eval_all:
        return n_valid_cases
    return valid_eval_max_pairs


def resolve_steps_per_epoch(
    n_users: int,
    batch_size: int,
    steps_per_epoch: int | None,
) -> int:
    if steps_per_epoch is not None:
        return max(1, steps_per_epoch)
    return max(n_users // batch_size, 1)


def run_post_train_test_eval(
    model: TiSASRecModel,
    *,
    train: list[Interaction],
    valid: list[Interaction],
    test: list[Interaction],
    id_maps: IdMaps,
    item_ids: list[str],
    device: torch.device,
    verified_only: bool,
    valid_hr_at_10: float,
    pool_size: int,
    mask_train_seen: bool,
    maxlen: int,
    time_span: int,
    eval_batch_size: int,
    seed: int,
    test_history: str = "train",
    time_unit_seconds: int | None = None,
) -> PostTrainTestSummary:
    # ``train_valid`` matches RecBole LOO (history = train + valid when ranking test).
    history_src = train if test_history == "train" else train + valid
    test_cases = build_user_batch_eval_cases(
        history_src,
        test,
        id_maps,
        verified_only=verified_only,
        time_unit_seconds=time_unit_seconds,
    )
    train_pairs = build_train_pairs(history_src, id_maps)
    metrics = evaluate_user_batch_cases(
        model,
        test_cases,
        item_ids,
        train_pairs,
        device=device,
        pool_size=pool_size,
        max_users=len(test_cases),
        seed=seed,
        mask_train_seen=mask_train_seen,
        maxlen=maxlen,
        time_span=time_span,
        eval_batch_size=eval_batch_size,
    )
    ratio = (
        metrics.link_hr_at_10 / valid_hr_at_10 if valid_hr_at_10 > 0 else 0.0
    )
    return PostTrainTestSummary(
        test_metrics=metrics,
        valid_hr_at_10=valid_hr_at_10,
        hr_ratio=ratio,
    )
