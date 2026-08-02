"""Batch orchestrator unit tests."""

from __future__ import annotations

from emorecagent.agents.reasoning_agent import BatchRowContext
from emorecagent.data.types import Interaction
from emorecagent.eval.batch_orchestrator import (
    RepairItem,
    RowEvalContext,
    _job_indices,
    form_batch_jobs,
)

DAY = 86_400_000


def _job(idx: int, user: str, ts: int) -> tuple[int, Interaction, list[str]]:
    it = Interaction(user, f"i{idx}", 5.0, ts, verified_purchase=True)
    return idx, it, [f"i{idx}", "n1", "n2"]


def test_form_batch_prefers_same_user_timestamp() -> None:
    pending = [
        _job(0, "u0", 3 * DAY),
        _job(1, "u0", 3 * DAY),
        _job(2, "u1", 3 * DAY),
    ]
    batch, leftover = form_batch_jobs(
        pending, batch_size=2, batch_token_budget=50_000
    )
    assert len(batch) == 2
    assert batch[0][1].user_id == batch[1][1].user_id == "u0"
    assert len(leftover) == 1


def test_job_indices_works_with_list_candidates() -> None:
    jobs = [_job(0, "u0", 3 * DAY), _job(1, "u1", 4 * DAY)]
    assert _job_indices(jobs) == {0, 1}


def test_repair_queue_cleanup_by_row_index() -> None:
    """RowJob tuples hold list candidates and cannot be placed in a set."""
    job = _job(3, "u0", 5 * DAY)
    ctx = RowEvalContext(
        row_id="u0\ti3\t432000000",
        user_id="u0",
        candidates=job[2],
        query_ts_ms=5 * DAY,
        weights={},
        filtered_pool=["a", "b"],
        breakdowns={},
        numeric_order=["a", "b"],
        item_e_hat={},
        recent_complaint_aspects=[],
    )
    queue = [RepairItem(job=job, ctx=ctx)]
    job_idxs = _job_indices([job])
    remaining = [item for item in queue if item.job[0] not in job_idxs]
    assert remaining == []


def test_batch_row_context_fields() -> None:
    ctx = BatchRowContext(
        row_id="r1",
        user_id="u0",
        weights={"comfort": 1.0},
        pool=["a", "b"],
        breakdowns={},
        numeric_order=["a", "b"],
    )
    assert ctx.row_id == "r1"
