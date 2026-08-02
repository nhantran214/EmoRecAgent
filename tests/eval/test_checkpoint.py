"""Checkpoint resume tests."""

from __future__ import annotations

import pytest

from emorecagent.baselines.popularity import PopularityRecommender
from emorecagent.data.types import Interaction
from emorecagent.eval.checkpoint import (
    CheckpointError,
    EvalCheckpoint,
    build_fingerprint,
    pass_checkpoint_path,
    row_key,
)
from emorecagent.eval.runner import evaluate

DAY = 86_400_000


def _train() -> list[Interaction]:
    data = []
    for u in range(4):
        data.append(Interaction(f"u{u}", "i_pop", 5.0, 1 * DAY))
        data.append(Interaction(f"u{u}", f"i_tail{u}", 5.0, 2 * DAY))
    return data


def _test_rows() -> list[Interaction]:
    return [
        Interaction("u0", "i_held1", 5.0, 3 * DAY, verified_purchase=True),
        Interaction("u1", "i_held2", 5.0, 3 * DAY, verified_purchase=True),
    ]


def test_row_key_is_stable() -> None:
    it = Interaction("u0", "i1", 5.0, 123, verified_purchase=True)
    assert row_key(it) == "u0\ti1\t123"


def test_checkpoint_resume_skips_rank_calls(tmp_path) -> None:
    stem = tmp_path / "exp.checkpoint"
    train = _train()
    test = _test_rows()

    evaluate(
        PopularityRecommender(),
        train,
        test,
        [5],
        method="pop",
        n_negatives=2,
        seed=0,
        checkpoint_stem=stem,
        resume=True,
    )

    class CountingPopularity(PopularityRecommender):
        calls = 0

        def rank(self, user_id, candidates, **kwargs):  # type: ignore[override]
            CountingPopularity.calls += 1
            return super().rank(user_id, candidates, **kwargs)

    CountingPopularity.calls = 0
    res = evaluate(
        CountingPopularity(),
        train,
        test,
        [5],
        method="pop",
        n_negatives=2,
        seed=0,
        checkpoint_stem=stem,
        resume=True,
    )
    assert CountingPopularity.calls == 0
    assert res.n_test_rows == 2
    assert "ndcg@5" in res.means


def test_fingerprint_mismatch_raises(tmp_path) -> None:
    stem = tmp_path / "exp.checkpoint"
    path = pass_checkpoint_path(stem, sampled=True)
    fp1 = build_fingerprint(
        method="pop",
        seed=0,
        protocol="sampled_negatives",
        n_negatives=2,
        verified_only=True,
        cumulative_history=False,
        max_test_rows=None,
        eval_protocol="per_row",
        aggregation="row_mean",
        k_values=[5],
        hr_avg_k=(1, 3, 5),
        parallel_workers=1,
        llm_batch=False,
        batch_size=12,
    )
    EvalCheckpoint(path, fp1, resume=False)
    path.write_text(
        '{"row_key": "u0\\ti\\t1", "ranked": ["a"]}\n',
        encoding="utf-8",
    )
    fp2 = dict(fp1)
    fp2["seed"] = 99
    with pytest.raises(CheckpointError, match="fingerprint"):
        EvalCheckpoint(path, fp2, resume=True)
