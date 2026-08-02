"""Staged batch eval orchestrator for sampled per-row ranking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from ..agents.reasoning_agent import (
    BatchRowContext,
    ReasoningConstraints,
    Recommendation,
)
from ..agents.reflection_agent import ReflectionAgent, ReflectionInput
from ..data.types import Interaction
from ..scoring.score import ScoreBreakdown
from .checkpoint import row_key

if TYPE_CHECKING:
    from ..recommend.graph_recommender import GraphRecommender

logger = logging.getLogger(__name__)

RowJob = tuple[int, Interaction, list[str]]


def _job_indices(jobs: list[RowJob]) -> set[int]:
    return {job[0] for job in jobs}


@dataclass
class RowEvalContext:
    row_id: str
    user_id: str
    candidates: list[str]
    query_ts_ms: int
    weights: dict[str, float]
    filtered_pool: list[str]
    breakdowns: dict[str, ScoreBreakdown]
    numeric_order: list[str]
    item_e_hat: dict[str, dict[str, float]]
    recent_complaint_aspects: list[str]
    ranked_pool_order: list[str] = field(default_factory=list)
    reflection_iters: int = 0
    constraints: ReasoningConstraints | None = None


@dataclass
class RepairItem:
    job: RowJob
    ctx: RowEvalContext


def estimate_row_tokens(ctx: RowEvalContext) -> int:
    """Rough prompt tokens for one row (cards + metadata)."""
    return 180 + len(ctx.filtered_pool) * 42


def form_batch_jobs(
    pending: list[RowJob],
    *,
    batch_size: int,
    batch_token_budget: int,
    precomputed: dict[str, RowEvalContext] | None = None,
) -> tuple[list[RowJob], list[RowJob]]:
    """Greedy batch pack: same (user, timestamp) first, then fill by order."""
    if not pending:
        return [], []
    precomputed = precomputed or {}
    batch: list[RowJob] = []
    remaining = list(pending)
    budget = batch_token_budget

    anchor = remaining[0]
    uid, ts = anchor[1].user_id, anchor[1].timestamp
    same_user: list[RowJob] = []
    rest: list[RowJob] = []
    for job in remaining:
        if job[1].user_id == uid and job[1].timestamp == ts:
            same_user.append(job)
        else:
            rest.append(job)

    for job in same_user + rest:
        if len(batch) >= batch_size:
            break
        rid = row_key(job[1])
        ctx = precomputed.get(rid)
        row_cost = estimate_row_tokens(ctx) if ctx else 800
        if batch and budget - row_cost < 0:
            break
        batch.append(job)
        budget -= row_cost

    batch_ids = {id(j) for j in batch}
    leftover = [j for j in pending if id(j) not in batch_ids]
    return batch, leftover


class BatchEvalOrchestrator:
    def __init__(
        self,
        recommender: GraphRecommender,
        *,
        batch_size: int = 12,
        batch_token_budget: int = 28_000,
        max_reflection_iters: int = 2,
        use_reflection: bool = True,
        use_llm_cot: bool = True,
        top_k: int = 5,
        parity_check: bool = False,
        rank_fallback: Callable[[str, list[str], int], list[str]] | None = None,
    ) -> None:
        self._rec = recommender
        self._batch_size = batch_size
        self._batch_token_budget = batch_token_budget
        self._max_reflection_iters = max_reflection_iters if use_reflection else 0
        self._use_llm_cot = use_llm_cot
        self._top_k = top_k
        self._parity_check = parity_check
        self._rank_fallback = rank_fallback
        deps = recommender._ctx.graph_deps
        self._reasoning = deps.reasoning
        self._reflection: ReflectionAgent = deps.reflection
        self._llm_rank_prefix = recommender._ctx.llm_rank_prefix
        self._repair_queue: list[RepairItem] = []

    @property
    def repair_pending(self) -> int:
        return len(self._repair_queue)

    def precompute(self, job: RowJob) -> RowEvalContext:
        _idx, interaction, candidates = job
        row_id = row_key(interaction)
        (
            _t_query,
            weights,
            filtered,
            breakdowns,
            numeric_order,
            item_e_hat,
            complaints,
        ) = self._rec.batch_precompute(
            interaction.user_id,
            candidates,
            interaction.timestamp,
        )
        return RowEvalContext(
            row_id=row_id,
            user_id=interaction.user_id,
            candidates=candidates,
            query_ts_ms=interaction.timestamp,
            weights=weights,
            filtered_pool=filtered,
            breakdowns=breakdowns,
            numeric_order=numeric_order,
            item_e_hat=item_e_hat,
            recent_complaint_aspects=complaints,
        )

    def run_batch(
        self,
        jobs: list[RowJob],
        *,
        repair: bool = False,
    ) -> dict[int, list[str]]:
        """Process one batch; returns idx -> final ranked list for completed rows."""
        if not jobs:
            return {}

        contexts: list[RowEvalContext] = []
        if repair:
            job_idxs = _job_indices(jobs)
            repair_map = {
                item.job[0]: item.ctx
                for item in self._repair_queue
                if item.job[0] in job_idxs
            }
            for job in jobs:
                ctx = repair_map[job[0]]
                ctx.ranked_pool_order = self._reasoning.repair_ranked_pool(
                    list(ctx.ranked_pool_order),
                    ctx.filtered_pool,
                    ctx.breakdowns,
                    constraints=ctx.constraints,
                    price_lookup={},
                )
                contexts.append(ctx)
        else:
            for job in jobs:
                contexts.append(self.precompute(job))
            batch_inputs = [
                BatchRowContext(
                    row_id=ctx.row_id,
                    user_id=ctx.user_id,
                    weights=ctx.weights,
                    pool=ctx.filtered_pool,
                    breakdowns=ctx.breakdowns,
                    numeric_order=ctx.numeric_order,
                )
                for ctx in contexts
            ]
            orders = self._reasoning.llm_rerank_pool_batch(
                batch_inputs,
                use_llm_cot=self._use_llm_cot,
            )
            for ctx in contexts:
                ctx.ranked_pool_order = orders.get(
                    ctx.row_id, list(ctx.numeric_order)
                )

        completed: dict[int, list[str]] = {}
        deferred: list[RepairItem] = []

        for job, ctx in zip(jobs, contexts):
            idx, interaction, candidates = job
            verdict = self._reflection.evaluate(self._reflection_input(ctx))

            if verdict.approved:
                ranked = self._complete_row(ctx, interaction, candidates)
                completed[idx] = ranked
                continue

            ctx.reflection_iters += 1
            ctx.constraints = self._reflection.constraints_from_verdict(verdict)

            if ctx.reflection_iters >= self._max_reflection_iters:
                ranked = self._complete_row(ctx, interaction, candidates)
                completed[idx] = ranked
            else:
                deferred.append(RepairItem(job=job, ctx=ctx))

        job_idxs = _job_indices(jobs)
        self._repair_queue = [
            item for item in self._repair_queue if item.job[0] not in job_idxs
        ]
        self._repair_queue.extend(deferred)

        return completed

    def process_repair_batch(self) -> dict[int, list[str]]:
        """Run one repair batch when the queue is full enough."""
        if not self._repair_queue:
            return {}
        chunk = self._repair_queue[: self._batch_size]
        jobs = [item.job for item in chunk]
        return self.run_batch(jobs, repair=True)

    def flush_repair_queue(self) -> dict[int, list[str]]:
        """Process any remaining repair rows (partial final batch)."""
        out: dict[int, list[str]] = {}
        while self._repair_queue:
            out.update(self.process_repair_batch())
        return out

    def _reflection_input(self, ctx: RowEvalContext) -> ReflectionInput:
        recs = self._top_recommendations(ctx.ranked_pool_order, ctx.breakdowns)
        return ReflectionInput(
            recommendations=recs,
            breakdowns=ctx.breakdowns,
            item_e_hat=ctx.item_e_hat,
            recent_complaint_aspects=ctx.recent_complaint_aspects,
        )

    def _top_recommendations(
        self,
        pool_order: list[str],
        breakdowns: dict[str, ScoreBreakdown],
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        for i, item_id in enumerate(pool_order[: self._top_k]):
            if item_id not in breakdowns:
                continue
            recs.append(
                Recommendation(
                    item_id=item_id,
                    breakdown=breakdowns[item_id],
                    rank=i + 1,
                )
            )
        return recs

    def _complete_row(
        self,
        ctx: RowEvalContext,
        interaction: Interaction,
        candidates: list[str],
    ) -> list[str]:
        ranked = self._rec.batch_merge_rank(
            ctx.user_id,
            candidates,
            ctx.weights,
            ctx.query_ts_ms,
            ctx.ranked_pool_order,
        )
        if self._parity_check and self._rank_fallback is not None:
            baseline = self._rank_fallback(
                interaction.user_id,
                candidates,
                interaction.timestamp,
            )
            prefix = self._llm_rank_prefix
            if ranked[:prefix] != baseline[:prefix]:
                logger.warning("batch_parity_fallback row_id=%s", ctx.row_id)
                ranked = baseline
        return ranked


def run_batched_eval_loop(
    orchestrator: BatchEvalOrchestrator,
    pending_jobs: list[RowJob],
    *,
    progress_logger: logging.Logger | None = None,
    progress_interval: int = 25,
    done_offset: int = 0,
    total: int = 0,
    t_start: float | None = None,
    on_rows_completed: Callable[[dict[int, list[str]]], None] | None = None,
) -> dict[int, list[str]]:
    """Drive batch formation, reasoning, repair flush until pending is empty."""
    from .runner import _log_eval_progress

    ranked_by_idx: dict[int, list[str]] = {}
    remaining = list(pending_jobs)
    batch_num = 0
    if t_start is None:
        t_start = time.monotonic()
    if total <= 0:
        total = done_offset + len(pending_jobs)

    def _report_batch_done(label: str) -> None:
        done = done_offset + len(ranked_by_idx)
        if progress_logger:
            progress_logger.info(
                "%s: %s rows complete (%s/%s)",
                label,
                len(ranked_by_idx),
                f"{done:,}",
                f"{total:,}",
            )
        _log_eval_progress(
            progress_logger,
            done=done,
            total=total,
            t_start=t_start,
            progress_interval=progress_interval,
        )

    while remaining or orchestrator.repair_pending:
        if remaining:
            batch, remaining = form_batch_jobs(
                remaining,
                batch_size=orchestrator._batch_size,
                batch_token_budget=orchestrator._batch_token_budget,
            )
            if not batch:
                batch, remaining = [remaining[0]], remaining[1:]
            batch_num += 1
            if progress_logger:
                progress_logger.info(
                    "batch %s/%s: row indices %s–%s (%s rows, %s pending, "
                    "%s repair queued)",
                    batch_num,
                    max(1, (len(pending_jobs) + orchestrator._batch_size - 1)
                        // orchestrator._batch_size),
                    batch[0][0] + 1,
                    batch[-1][0] + 1,
                    len(batch),
                    len(remaining),
                    orchestrator.repair_pending,
                )
            completed = orchestrator.run_batch(batch)
            ranked_by_idx.update(completed)
            if on_rows_completed is not None:
                on_rows_completed(completed)
            _report_batch_done(f"batch {batch_num}")

        if orchestrator.repair_pending and (
            orchestrator.repair_pending >= orchestrator._batch_size or not remaining
        ):
            if progress_logger:
                progress_logger.info(
                    "repair batch: %s rows queued",
                    orchestrator.repair_pending,
                )
            completed = orchestrator.process_repair_batch()
            ranked_by_idx.update(completed)
            if on_rows_completed is not None:
                on_rows_completed(completed)
            _report_batch_done("repair batch")

    if orchestrator.repair_pending:
        if progress_logger:
            progress_logger.info(
                "repair flush: %s rows remaining",
                orchestrator.repair_pending,
            )
        completed = orchestrator.flush_repair_queue()
        ranked_by_idx.update(completed)
        if on_rows_completed is not None:
            on_rows_completed(completed)
        _report_batch_done("repair flush")

    if progress_logger and ranked_by_idx:
        progress_logger.info(
            "batch eval complete: %s rows ranked",
            f"{len(ranked_by_idx):,}",
        )

    return ranked_by_idx
