"""Stage-1 LOO test history: train vs train+valid."""

from __future__ import annotations

from emorecagent.data.types import Interaction
from emorecagent.sequential.id_maps import IdMaps
from emorecagent.tisasrec_align.sequence_data import build_user_batch_eval_cases


def _id_maps() -> IdMaps:
    return IdMaps(
        user_to_idx={"u1": 1},
        item_to_idx={"a": 1, "b": 2, "c": 3, "d": 4},
    )


def test_train_only_history_excludes_valid() -> None:
    train = [
        Interaction(user_id="u1", item="a", rating=5.0, timestamp=100),
        Interaction(user_id="u1", item="b", rating=5.0, timestamp=200),
    ]
    valid = [Interaction(user_id="u1", item="c", rating=5.0, timestamp=300)]
    test = [Interaction(user_id="u1", item="d", rating=5.0, timestamp=400)]
    cases = build_user_batch_eval_cases(train, test, _id_maps())
    assert len(cases) == 1
    hist_items = {loc for loc, _ in cases[0].history}
    assert hist_items == {1, 2}  # a,b only
    assert 3 not in hist_items


def test_train_valid_history_includes_valid_item() -> None:
    train = [
        Interaction(user_id="u1", item="a", rating=5.0, timestamp=100),
        Interaction(user_id="u1", item="b", rating=5.0, timestamp=200),
    ]
    valid = [Interaction(user_id="u1", item="c", rating=5.0, timestamp=300)]
    test = [Interaction(user_id="u1", item="d", rating=5.0, timestamp=400)]
    history_src = train + valid
    cases = build_user_batch_eval_cases(history_src, test, _id_maps())
    assert len(cases) == 1
    hist_items = {loc for loc, _ in cases[0].history}
    assert hist_items == {1, 2, 3}  # a,b,c
    assert cases[0].relevant_locals == frozenset({4})
