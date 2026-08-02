"""Tests for user-batch Stage 1 eval (baseline Protocol B alignment)."""

from __future__ import annotations

import torch

from emorecagent.data.types import Interaction
from emorecagent.sequential.id_maps import build_id_maps_from_interactions
from emorecagent.tisasrec_align.model import TiSASRecModel, init_model_weights
from emorecagent.tisasrec_align.schema import TiSASRecArgs
from emorecagent.tisasrec_align.sequence_data import (
    build_train_pairs,
    build_user_batch_eval_cases,
)
from emorecagent.tisasrec_align.valid_eval import evaluate_user_batch_cases


def test_build_user_batch_uses_train_only_history():
    train = [
        Interaction("u1", "i1", 5.0, 1000),
        Interaction("u1", "i2", 5.0, 2000),
    ]
    valid = [Interaction("u1", "i3", 5.0, 3000)]
    id_maps = build_id_maps_from_interactions(train, valid)
    cases = build_user_batch_eval_cases(train, valid, id_maps)
    assert len(cases) == 1
    case = cases[0]
    assert case.n_target_rows == 1
    assert len(case.history) == 2
    assert {loc for loc, _ in case.history} == {
        id_maps.item_to_idx["i1"],
        id_maps.item_to_idx["i2"],
    }


def test_l2_regularization_includes_all_embeddings():
    args = TiSASRecArgs(maxlen=4, hidden_units=8, num_blocks=1, time_span=4)
    model = TiSASRecModel(3, args)
    reg = model.l2_regularization()
    assert reg.ndim == 0
    assert reg.item() > 0


def test_init_model_weights_runs():
    args = TiSASRecArgs(maxlen=4, hidden_units=8, num_blocks=1, time_span=4)
    model = TiSASRecModel(3, args)
    init_model_weights(model)


def test_evaluate_user_batch_smoke():
    train = [
        Interaction("u1", "i1", 5.0, 1000),
        Interaction("u1", "i2", 5.0, 2000),
        Interaction("u2", "i1", 4.0, 1500),
    ]
    test = [
        Interaction("u1", "i3", 5.0, 4000),
        Interaction("u2", "i2", 4.0, 4500),
    ]
    id_maps = build_id_maps_from_interactions(train, test)
    cases = build_user_batch_eval_cases(train, test, id_maps)
    args = TiSASRecArgs(maxlen=5, hidden_units=8, num_blocks=1, time_span=4)
    model = TiSASRecModel(len(id_maps.item_to_idx), args)
    init_model_weights(model)
    item_ids = [item for item, _ in sorted(id_maps.item_to_idx.items(), key=lambda kv: kv[1])]
    train_pairs = build_train_pairs(train, id_maps)
    metrics = evaluate_user_batch_cases(
        model,
        cases,
        item_ids,
        train_pairs,
        device=torch.device("cpu"),
        pool_size=50,
        max_users=len(cases),
        seed=0,
        eval_batch_size=2,
        maxlen=args.maxlen,
        time_span=args.time_span,
    )
    assert metrics.n_pairs_eval == len(cases)
    assert metrics.n_valid_pairs_total == sum(c.n_target_rows for c in cases)
    assert 0.0 <= metrics.link_hr_at_10 <= 1.0
