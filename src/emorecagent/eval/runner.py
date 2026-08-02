"""Experiment runner for ranking evaluation over train/test splits."""

from __future__ import annotations

import inspect
import json
import logging
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..baselines.base import Recommender
from ..baselines.itemknn import ItemKNNRecommender
from ..baselines.popularity import PopularityRecommender
from ..baselines.sequential import SequentialRecommender
from ..baselines.svd import SVDRecommender
from ..data.types import Interaction
from .checkpoint import (
    EvalCheckpoint,
    build_fingerprint,
    default_checkpoint_stem,
    pass_checkpoint_path,
)
from . import metrics as M
from .significance import PairedResult, paired_bootstrap

_AGENTIC_METHODS = frozenset(
    {
        "aspect_aware",
        "emorecagent",
        "emorecagent_fast",
    }
)


def _rank_supports_query_ts(recommender: Recommender) -> bool:
    return "query_ts_ms" in inspect.signature(recommender.rank).parameters


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _should_log_progress(done: int, total: int, progress_interval: int) -> bool:
    if total <= 0:
        return False
    if done == 1 or done >= total:
        return True
    return progress_interval > 0 and done % progress_interval == 0


def _log_eval_progress(
    progress_logger: logging.Logger | None,
    *,
    done: int,
    total: int,
    t_start: float,
    progress_interval: int,
    unit: str = "rows",
) -> None:
    if not progress_logger or not _should_log_progress(done, total, progress_interval):
        return
    elapsed = time.monotonic() - t_start
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - done)
    eta_s = remaining / rate if rate > 0 else 0.0
    progress_logger.info(
        "evaluated %s/%s %s (%.1f%%) elapsed=%s rate=%.2f %s/s eta=%s",
        f"{done:,}",
        f"{total:,}",
        unit,
        100.0 * done / total,
        _fmt_duration(elapsed),
        rate,
        unit,
        _fmt_duration(eta_s),
    )


def _rank_row(
    recommender: Recommender,
    user_id: str,
    candidates: list[str],
    query_ts_ms: int | None,
) -> list[str]:
    if query_ts_ms is not None and _rank_supports_query_ts(recommender):
        return recommender.rank(user_id, candidates, query_ts_ms=query_ts_ms)
    if query_ts_ms is not None and hasattr(recommender, "prepare_user_query"):
        recommender.prepare_user_query(user_id, query_ts_ms)
    return recommender.rank(user_id, candidates)


def _check_parallel_eval(
    parallel_workers: int,
    cumulative_history: bool,
    *,
    llm_batch: bool = False,
) -> None:
    if parallel_workers > 1 and cumulative_history:
        raise ValueError(
            "parallel_workers > 1 is incompatible with cumulative_history=True"
        )
    if llm_batch and parallel_workers > 1:
        raise ValueError(
            "llm_batch and parallel_workers > 1 are mutually exclusive "
            "(use llm_batch for batched LLM calls or parallel_workers for row-parallel eval)"
        )


def load_split_jsonl(path: str | Path) -> list[Interaction]:
    """Read interactions written by data.split.write_split."""
    from ..data.loader import load_split_jsonl as _load

    return _load(path)


def build_recommender(method: str, cfg: dict, seed: int) -> Recommender:
    if method in _AGENTIC_METHODS:
        train = cfg.get("train_interactions")
        if train is None:
            raise ValueError(
                f"method '{method}' requires cfg['train_interactions'] "
                "(pass the train split when building the recommender)"
            )
        from ..recommend.context import build_recommend_context
        from ..recommend.aspect_eval import AspectAwareEvalRecommender
        from ..recommend.emorec import EmoRecRecommender
        from ..recommend.graph_recommender import GraphRecommender

        if method == "emorecagent":
            return GraphRecommender.from_runner_cfg(
                cfg, config=cfg.get("app_config")
            )

        ctx = build_recommend_context(
            train,
            seed=seed,
            cf_backend=str(cfg.get("cf_backend", "svd")),
            cf_factors=int(cfg.get("factors", 64)),
            alpha=float(cfg.get("alpha", 0.5)),
            lambda_decay=float(cfg.get("lambda_decay", 0.01)),
            helpful_cap=int(cfg.get("helpful_cap", 10)),
            affective_rescaled=bool(cfg.get("affective_rescaled", True)),
            absa_cache_path=cfg.get("absa_cache_path"),
            review_path=cfg.get("review_path"),
            max_reflection_iters=int(cfg.get("max_reflection_iters", 2)),
            pool_size=int(cfg.get("pool_size", 200)),
            top_k_aspects=int(cfg.get("top_k_aspects", 5)),
            use_reflection=bool(cfg.get("use_reflection", True)),
            use_dynamic_weights=bool(cfg.get("use_dynamic_weights", True)),
            use_aspect_term=bool(cfg.get("use_aspect_term", True)),
        )
        if method == "aspect_aware":
            return AspectAwareEvalRecommender(ctx)
        if method == "emorecagent_fast":
            return EmoRecRecommender(ctx)
        raise ValueError(f"Unknown agentic method: {method}")

    if method == "popularity":
        return PopularityRecommender()
    if method == "itemknn":
        return ItemKNNRecommender(seed=seed)
    if method in ("svd", "base_cf"):
        return SVDRecommender(factors=int(cfg.get("factors", 64)), seed=seed)
    if method == "sequential":
        return SequentialRecommender()
    if method == "emorecagent_align":
        train = cfg.get("train_interactions")
        app_cfg = cfg.get("app_config")
        if train is None or app_cfg is None:
            raise ValueError(
                "method 'emorecagent_align' requires cfg['train_interactions'] "
                "and cfg['app_config']"
            )
        ta = app_cfg.tisasrec_align
        if ta.stage2_mode == "rerank" and not ta.stage1_only:
            from ..tisasrec_align.rerank_recommender import RerankAlignRecommender

            return RerankAlignRecommender.from_config(
                app_cfg, train, seed=seed
            )
        from ..tisasrec_align.stage1_factory import build_stage1_recommender

        return build_stage1_recommender(app_cfg, train, seed=seed)
    raise ValueError(f"Unknown method: {method}")


def aggregate_per_user(
    per_user: dict[str, list[float]],
    user_ids: list[str],
) -> dict[str, float]:
    """User-mean: average each metric per user, then mean across users."""
    if not user_ids:
        return {}
    by_user: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx, uid in enumerate(user_ids):
        for key, vec in per_user.items():
            by_user[uid][key].append(vec[idx])
    user_means = {
        uid: {k: sum(v) / len(v) for k, v in metrics.items()}
        for uid, metrics in by_user.items()
    }
    return {
        k: sum(u[k] for u in user_means.values()) / len(user_means)
        for k in per_user
    }


@dataclass
class EvalResult:
    method: str
    k_values: list[int]
    n_test_rows: int
    n_test_users: int
    protocol: str
    cumulative_history: bool = False
    hr_avg_k: list[int] = field(default_factory=lambda: list(M.HR_AVG_KS))
    means: dict[str, float] = field(default_factory=dict)
    means_per_user: dict[str, float] = field(default_factory=dict)
    per_user: dict[str, list[float]] = field(default_factory=dict)
    user_ids: list[str] = field(default_factory=list)
    ci_per_user: dict[str, dict[str, float]] = field(default_factory=dict)
    verified_only: bool = False
    n_verified_rows: int = 0
    eval_protocol: str = "per_row"
    aggregation: str = "row_mean"
    n_negatives: int | None = None
    sampled: dict | None = None
    method_variant: str | None = None
    max_test_rows: int | None = None
    use_llm_cot: bool | None = None
    metadata: dict[str, object] | None = None

    def to_json(self) -> dict:
        payload: dict = {
            "method": self.method,
            "k_values": self.k_values,
            "hr_avg_k": self.hr_avg_k,
            "n_test_rows": self.n_test_rows,
            "n_test_users": self.n_test_users,
            "protocol": self.protocol,
            "cumulative_history": self.cumulative_history,
            "verified_only": self.verified_only,
            "n_verified_rows": self.n_verified_rows,
            "eval_protocol": self.eval_protocol,
            "aggregation": self.aggregation,
            "means": self.means,
            "means_per_user": self.means_per_user,
            "per_user": self.per_user,
            "user_ids": self.user_ids,
        }
        if self.method_variant is not None:
            payload["method_variant"] = self.method_variant
        if self.max_test_rows is not None:
            payload["max_test_rows"] = self.max_test_rows
        if self.use_llm_cot is not None:
            payload["use_llm_cot"] = self.use_llm_cot
        if self.ci_per_user:
            payload["ci_per_user"] = self.ci_per_user
        if self.sampled is not None:
            payload["sampled"] = self.sampled
        if self.metadata is not None:
            payload["metadata"] = self.metadata
        return payload

    def to_sampled_payload(self) -> dict:
        """JSON-serializable block for the 1+N negative-sampling pass."""
        return {
            "protocol": self.protocol,
            "n_negatives": self.n_negatives,
            "k_values": self.k_values,
            "eval_protocol": self.eval_protocol,
            "aggregation": self.aggregation,
            "n_test_rows": self.n_test_rows,
            "n_test_users": self.n_test_users,
            "means": self.means,
            "means_per_user": self.means_per_user,
            "per_user": self.per_user,
            "user_ids": self.user_ids,
            "ci_per_user": self.ci_per_user,
        }


def _metric_keys(k_values: list[int], hr_avg_k: tuple[int, ...]) -> list[str]:
    keys = [f"{m}@{k}" for k in k_values for m in M.METRIC_NAMES]
    for k in hr_avg_k:
        hk = f"hr@{k}"
        if hk not in keys:
            keys.append(hk)
    if M.AVG_HR_KEY not in keys:
        keys.append(M.AVG_HR_KEY)
    return keys


def _effective_k_values(
    k_values: list[int],
    n_negatives: int | None,
    sampled_k_values: list[int] | None,
) -> list[int]:
    """Merge extra K for negative-sampling passes; full-catalog pass unchanged."""
    if n_negatives is None or not sampled_k_values:
        return list(k_values)
    return sorted(set(k_values) | set(sampled_k_values))


def _group_test_by_user(
    test: list[Interaction],
    *,
    verified_only: bool,
) -> dict[str, list[Interaction]]:
    grouped: dict[str, list[Interaction]] = defaultdict(list)
    for t in sorted(test, key=lambda x: (x.user_id, x.timestamp)):
        if verified_only and not t.verified_purchase:
            continue
        grouped[t.user_id].append(t)
    return dict(grouped)


def _finalize_means(
    per_user: dict[str, list[float]],
    user_ids: list[str],
    *,
    aggregation: str,
) -> tuple[dict[str, float], dict[str, float]]:
    row_means = {
        key: (sum(vec) / len(vec) if vec else 0.0) for key, vec in per_user.items()
    }
    user_means = aggregate_per_user(per_user, user_ids)
    if aggregation == "user_mean":
        return user_means, user_means
    return row_means, user_means


def _append_rank_scores(
    per_user: dict[str, list[float]],
    user_ids: list[str],
    metric_keys: list[str],
    ranked: list[str],
    relevant: set[str],
    k_values: list[int],
    hr_avg_k: tuple[int, ...],
    user_id: str,
) -> None:
    row_scores: dict[str, float] = {}
    for k in k_values:
        scored = M.evaluate_ranking(ranked, relevant, k)
        for m, v in scored.items():
            row_scores[f"{m}@{k}"] = v
    row_scores.update(M.evaluate_hr_avg(ranked, relevant, hr_avg_k))
    for key in metric_keys:
        per_user[key].append(row_scores[key])
    user_ids.append(user_id)


def _evaluate_user_batch(
    recommender: Recommender,
    train: list[Interaction],
    test: list[Interaction],
    k_values: list[int],
    method: str,
    seed: int,
    hr_avg_k: tuple[int, ...],
    *,
    show_progress: bool,
    progress_logger: logging.Logger | None,
    progress_interval: int,
    verified_only: bool,
    max_test_rows: int | None,
    method_variant: str | None,
    use_llm_cot: bool | None,
    aggregation: str,
    n_negatives: int | None = None,
    parallel_workers: int = 1,
) -> EvalResult:
    """LightGCN-style eval; optional 1+N negative sampling per test item.

    ``parallel_workers`` only applies to the full-catalog pass. The sampled pass
    draws negatives from a shared ``rng`` in row order, so parallelising it
    would change which negatives each row gets.
    """
    if parallel_workers > 1:
        show_progress = False
    train_items: dict[str, set[str]] = {}
    for it in train:
        train_items.setdefault(it.user_id, set()).add(it.item)
    if hasattr(recommender, "catalog_items"):
        all_items = sorted(recommender.catalog_items())
    else:
        all_items = sorted({it.item for it in train} | {it.item for it in test})
    rng = random.Random(seed)

    grouped = _group_test_by_user(test, verified_only=verified_only)
    if max_test_rows is not None:
        capped: dict[str, list[Interaction]] = {}
        n_rows = 0
        for uid in sorted(grouped):
            rows = grouped[uid]
            if n_rows >= max_test_rows:
                break
            take = rows[: max(0, max_test_rows - n_rows)]
            if take:
                capped[uid] = take
                n_rows += len(take)
        grouped = capped

    n_verified_rows = sum(1 for t in test if t.verified_purchase)
    metric_keys = _metric_keys(k_values, hr_avg_k)
    per_user: dict[str, list[float]] = {k: [] for k in metric_keys}
    user_ids: list[str] = []
    n_test_rows = 0

    users = sorted(grouped)
    total = len(users) if n_negatives is None else sum(len(v) for v in grouped.values())
    candidate_protocol = (
        "sampled_negatives" if n_negatives is not None else "full_catalog"
    )
    if progress_logger:
        progress_logger.info(
            "user-batch eval: %s users, %s test rows in scope, protocol=%s, "
            "n_negatives=%s, verified_only=%s, aggregation=%s, parallel_workers=%s",
            f"{len(users):,}",
            f"{sum(len(v) for v in grouped.values()):,}",
            candidate_protocol,
            n_negatives,
            verified_only,
            aggregation,
            parallel_workers if n_negatives is None else 1,
        )

    if n_negatives is not None:
        flat_rows: list[tuple[str, Interaction]] = []
        for uid in users:
            for t in grouped[uid]:
                flat_rows.append((uid, t))
        if show_progress and flat_rows:
            from tqdm import tqdm

            row_iter: Iterator[tuple[str, Interaction]] = tqdm(
                flat_rows, desc="Evaluating (sampled)", unit="row", total=len(flat_rows)
            )
        else:
            row_iter = iter(flat_rows)

        for i, (uid, t) in enumerate(row_iter):
            n_test_rows += 1
            seen = set(train_items.get(uid, set()))
            negs = [item for item in all_items if item not in seen and item != t.item]
            rng.shuffle(negs)
            candidates = [t.item, *negs[:n_negatives]]
            if hasattr(recommender, "prepare_user_query"):
                recommender.prepare_user_query(uid, t.timestamp)
            ranked = recommender.rank(uid, candidates)
            _append_rank_scores(
                per_user,
                user_ids,
                metric_keys,
                ranked,
                {t.item},
                k_values,
                hr_avg_k,
                uid,
            )
            if progress_logger and flat_rows and (i + 1) % progress_interval == 0:
                progress_logger.info(
                    "sampled user-batch: %s/%s rows (%.1f%%)",
                    f"{i + 1:,}",
                    f"{len(flat_rows):,}",
                    100.0 * (i + 1) / len(flat_rows),
                )
    else:

        def _rank_user(uid: str) -> tuple[str, set[str], list[str]]:
            user_tests = grouped[uid]
            relevant = {t.item for t in user_tests}
            seen = set(train_items.get(uid, set()))
            candidates = [item for item in all_items if item not in seen]
            for item in relevant:
                if item not in candidates:
                    candidates.append(item)
            t_query = max(t.timestamp for t in user_tests)
            return uid, relevant, _rank_row(recommender, uid, candidates, t_query)

        def _collect(
            uid: str, relevant: set[str], ranked: list[str], done: int
        ) -> None:
            nonlocal n_test_rows
            n_test_rows += len(relevant)
            _append_rank_scores(
                per_user,
                user_ids,
                metric_keys,
                ranked,
                relevant,
                k_values,
                hr_avg_k,
                uid,
            )
            if progress_logger and total and done % progress_interval == 0:
                progress_logger.info(
                    "evaluated %s/%s users (%.1f%%)",
                    f"{done:,}",
                    f"{len(users):,}",
                    100.0 * done / len(users),
                )

        if parallel_workers > 1:
            # Chunked so at most a few hundred full-catalog rankings are held in
            # memory at once; ``map`` keeps user order, so metrics match serial.
            chunk = max(1, parallel_workers * 4)
            done = 0
            with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                for start in range(0, len(users), chunk):
                    for uid, relevant, ranked in pool.map(
                        _rank_user, users[start : start + chunk]
                    ):
                        done += 1
                        _collect(uid, relevant, ranked, done)
        else:
            user_iter: Iterator[str] = users
            if show_progress and total:
                from tqdm import tqdm

                user_iter = tqdm(
                    users, desc="Evaluating users", unit="user", total=total
                )
            for i, uid in enumerate(user_iter):
                _collect(*_rank_user(uid), i + 1)

    if progress_logger and total:
        progress_logger.info(
            "user-batch evaluation complete (%s)", candidate_protocol
        )

    means, means_per_user = _finalize_means(per_user, user_ids, aggregation=aggregation)
    return EvalResult(
        method=method,
        k_values=k_values,
        n_test_rows=n_test_rows,
        n_test_users=len(set(user_ids)),
        protocol=candidate_protocol,
        cumulative_history=False,
        hr_avg_k=list(hr_avg_k),
        means=means,
        means_per_user=means_per_user,
        per_user=per_user,
        user_ids=user_ids,
        verified_only=verified_only,
        n_verified_rows=n_verified_rows,
        eval_protocol="user_batch",
        aggregation=aggregation,
        n_negatives=n_negatives,
        method_variant=method_variant,
        max_test_rows=max_test_rows,
        use_llm_cot=use_llm_cot,
    )


def _evaluate_per_row(
    recommender: Recommender,
    train: list[Interaction],
    test: list[Interaction],
    k_values: list[int],
    method: str,
    n_negatives: int | None,
    seed: int,
    cumulative_history: bool,
    hr_avg_k: tuple[int, ...],
    *,
    show_progress: bool,
    progress_logger: logging.Logger | None,
    progress_interval: int,
    verified_only: bool,
    max_test_rows: int | None,
    method_variant: str | None,
    use_llm_cot: bool | None,
    aggregation: str,
    parallel_workers: int = 1,
    checkpoint: EvalCheckpoint | None = None,
    llm_batch: bool = False,
    batch_size: int = 12,
    batch_token_budget: int = 28_000,
) -> EvalResult:
    """Per test-row ranking (Protocol A — EmoRecAgent ablations)."""
    _check_parallel_eval(
        parallel_workers, cumulative_history, llm_batch=llm_batch
    )
    if parallel_workers > 1:
        show_progress = False

    train_items: dict[str, set[str]] = {}
    for it in train:
        train_items.setdefault(it.user_id, set()).add(it.item)
    if hasattr(recommender, "catalog_items"):
        all_items = sorted(recommender.catalog_items())
    else:
        all_items = sorted({it.item for it in train} | {it.item for it in test})
    rng = random.Random(seed)

    protocol = "sampled_negatives" if n_negatives is not None else "full_catalog"
    test_order = sorted(test, key=lambda t: (t.user_id, t.timestamp))
    n_verified_rows = sum(1 for t in test_order if t.verified_purchase)
    if verified_only:
        test_order = [t for t in test_order if t.verified_purchase]
    if max_test_rows is not None:
        test_order = test_order[:max_test_rows]
    prior_test: dict[str, list[tuple[int, str]]] = defaultdict(list)

    metric_keys = _metric_keys(k_values, hr_avg_k)
    per_user: dict[str, list[float]] = {k: [] for k in metric_keys}
    user_ids: list[str] = []

    total = len(test_order)
    if progress_logger:
        progress_logger.info(
            "ranking %s test rows (%s users, protocol=%s, cumulative_history=%s, "
            "verified_only=%s, verified_in_test=%s/%s, parallel_workers=%s)",
            f"{total:,}",
            f"{len({t.user_id for t in test_order}):,}",
            protocol,
            cumulative_history,
            verified_only,
            f"{n_verified_rows:,}",
            f"{len(test):,}",
            parallel_workers,
        )

    row_jobs: list[tuple[int, Interaction, list[str]]] = []
    for idx, t in enumerate(test_order):
        seen = set(train_items.get(t.user_id, set()))
        if cumulative_history:
            for ts, item in prior_test[t.user_id]:
                if ts < t.timestamp:
                    seen.add(item)

        pool = [i for i in all_items if i not in seen]
        if t.item not in pool:
            pool.append(t.item)
        if n_negatives is not None:
            negs = [i for i in pool if i != t.item]
            rng.shuffle(negs)
            candidates = [t.item, *negs[:n_negatives]]
        else:
            candidates = pool
        row_jobs.append((idx, t, candidates))
        if cumulative_history:
            prior_test[t.user_id].append((t.timestamp, t.item))

    ranked_by_idx: dict[int, list[str]] = {}
    if checkpoint is not None:
        for idx, t, _candidates in row_jobs:
            if checkpoint.has(t):
                ranked_by_idx[idx] = checkpoint.get(t)
    n_cached = len(ranked_by_idx)
    pending_jobs = [
        (idx, t, candidates)
        for idx, t, candidates in row_jobs
        if idx not in ranked_by_idx
    ]
    if progress_logger and checkpoint is not None and n_cached:
        progress_logger.info(
            "resume: %s/%s rows from checkpoint (%s remaining)",
            f"{n_cached:,}",
            f"{total:,}",
            f"{len(pending_jobs):,}",
        )

    t_eval = time.monotonic()
    if llm_batch:
        from ..recommend.graph_recommender import GraphRecommender
        from .batch_orchestrator import BatchEvalOrchestrator, run_batched_eval_loop

        if not isinstance(recommender, GraphRecommender):
            raise ValueError("llm_batch requires GraphRecommender (emorecagent*)")
        if progress_logger:
            progress_logger.info(
                "llm_batch=true batch_size=%s batch_token_budget=%s",
                batch_size,
                batch_token_budget,
            )
        orchestrator = BatchEvalOrchestrator(
            recommender,
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            max_reflection_iters=recommender._ctx.graph_deps.max_reflection_iters,
            use_reflection=recommender._ctx.graph_deps.max_reflection_iters > 0,
            use_llm_cot=recommender._ctx.use_llm_cot,
            top_k=recommender._ctx.top_k_aspects,
        )
        def _save_batch_rows(completed: dict[int, list[str]]) -> None:
            for idx, ranked in completed.items():
                _idx, interaction, _cands = row_jobs[idx]
                if checkpoint is not None:
                    checkpoint.save(interaction, ranked)
                ranked_by_idx[idx] = ranked

        run_batched_eval_loop(
            orchestrator,
            pending_jobs,
            progress_logger=progress_logger,
            progress_interval=progress_interval,
            done_offset=n_cached,
            total=total,
            t_start=t_eval,
            on_rows_completed=_save_batch_rows,
        )
    elif parallel_workers <= 1:
        rows: Iterator[tuple[int, Interaction, list[str]]] = pending_jobs
        if show_progress and pending_jobs:
            from tqdm import tqdm

            rows = tqdm(
                pending_jobs,
                desc="Evaluating",
                unit="row",
                total=len(pending_jobs),
                initial=0,
            )

        for i, (idx, t, candidates) in enumerate(rows):
            ranked = _rank_row(recommender, t.user_id, candidates, t.timestamp)
            if checkpoint is not None:
                checkpoint.save(t, ranked)
            ranked_by_idx[idx] = ranked

            done = n_cached + i + 1
            _log_eval_progress(
                progress_logger,
                done=done,
                total=total,
                t_start=t_eval,
                progress_interval=progress_interval,
            )
            if (
                progress_logger
                and _should_log_progress(done, total, progress_interval)
                and hasattr(recommender, "log_stage_timing")
            ):
                recommender.log_stage_timing(progress_logger)
    else:
        if pending_jobs:
            def _run_job(
                job: tuple[int, Interaction, list[str]],
            ) -> tuple[int, str, str, list[str], Interaction]:
                idx, t, candidates = job
                ranked = _rank_row(recommender, t.user_id, candidates, t.timestamp)
                return idx, t.user_id, t.item, ranked, t

            with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                futures = [pool.submit(_run_job, job) for job in pending_jobs]
                done_new = 0
                completed = as_completed(futures)
                if show_progress and pending_jobs:
                    from tqdm import tqdm

                    completed = tqdm(
                        completed,
                        desc="Evaluating",
                        unit="row",
                        total=len(futures),
                    )
                for fut in completed:
                    idx, _uid, _item, ranked, t = fut.result()
                    if checkpoint is not None:
                        checkpoint.save(t, ranked)
                    ranked_by_idx[idx] = ranked
                    done_new += 1
                    _log_eval_progress(
                        progress_logger,
                        done=n_cached + done_new,
                        total=total,
                        t_start=t_eval,
                        progress_interval=progress_interval,
                    )

    for idx, t, _candidates in row_jobs:
        ranked = ranked_by_idx[idx]
        _append_rank_scores(
            per_user,
            user_ids,
            metric_keys,
            ranked,
            {t.item},
            k_values,
            hr_avg_k,
            t.user_id,
        )

    if checkpoint is not None:
        checkpoint.close()

    if progress_logger and total:
        progress_logger.info("evaluation complete: %s test rows", f"{total:,}")
        if hasattr(recommender, "log_stage_timing"):
            progress_logger.info("--- graph stage timing (final) ---")
            recommender.log_stage_timing(progress_logger)

    means, means_per_user = _finalize_means(per_user, user_ids, aggregation=aggregation)
    return EvalResult(
        method=method,
        k_values=k_values,
        n_test_rows=len(user_ids),
        n_test_users=len(set(user_ids)),
        protocol=protocol,
        cumulative_history=cumulative_history,
        hr_avg_k=list(hr_avg_k),
        means=means,
        means_per_user=means_per_user,
        per_user=per_user,
        user_ids=user_ids,
        verified_only=verified_only,
        n_verified_rows=n_verified_rows,
        eval_protocol="per_row",
        aggregation=aggregation,
        n_negatives=n_negatives,
        method_variant=method_variant,
        max_test_rows=max_test_rows,
        use_llm_cot=use_llm_cot,
    )


def _run_eval_pass(
    recommender: Recommender,
    train: list[Interaction],
    test: list[Interaction],
    k_values: list[int],
    method: str,
    n_negatives: int | None,
    seed: int,
    cumulative_history: bool,
    hr_avg_k: tuple[int, ...],
    *,
    show_progress: bool,
    progress_logger: logging.Logger | None,
    progress_interval: int,
    verified_only: bool,
    max_test_rows: int | None,
    method_variant: str | None,
    use_llm_cot: bool | None,
    eval_protocol: str,
    aggregation: str,
    sampled_k_values: list[int] | None = None,
    parallel_workers: int = 1,
    checkpoint_stem: Path | None = None,
    resume: bool = True,
    fresh_checkpoint: bool = False,
    llm_batch: bool = False,
    batch_size: int = 12,
    batch_token_budget: int = 28_000,
) -> EvalResult:
    metric_k_values = _effective_k_values(k_values, n_negatives, sampled_k_values)
    protocol = "sampled_negatives" if n_negatives is not None else "full_catalog"

    checkpoint: EvalCheckpoint | None = None
    if checkpoint_stem is not None and eval_protocol == "per_row":
        ckpt_path = pass_checkpoint_path(
            checkpoint_stem, sampled=n_negatives is not None
        )
        fingerprint = build_fingerprint(
            method=method,
            seed=seed,
            protocol=protocol,
            n_negatives=n_negatives,
            verified_only=verified_only,
            cumulative_history=cumulative_history,
            max_test_rows=max_test_rows,
            eval_protocol=eval_protocol,
            aggregation=aggregation,
            k_values=metric_k_values,
            hr_avg_k=hr_avg_k,
            parallel_workers=parallel_workers,
            llm_batch=llm_batch,
            batch_size=batch_size,
        )
        checkpoint = EvalCheckpoint(
            ckpt_path,
            fingerprint,
            resume=resume,
            fresh=fresh_checkpoint,
            logger=progress_logger,
        )
        if progress_logger:
            progress_logger.info("checkpoint file: %s", ckpt_path.resolve())

    if eval_protocol == "user_batch":
        if cumulative_history and progress_logger:
            progress_logger.warning(
                "user_batch protocol ignores cumulative_history=True"
            )
        return _evaluate_user_batch(
            recommender,
            train,
            test,
            metric_k_values,
            method,
            seed,
            hr_avg_k,
            show_progress=show_progress,
            progress_logger=progress_logger,
            progress_interval=progress_interval,
            verified_only=verified_only,
            max_test_rows=max_test_rows,
            method_variant=method_variant,
            use_llm_cot=use_llm_cot,
            aggregation=aggregation,
            n_negatives=n_negatives,
            parallel_workers=parallel_workers,
        )
    return _evaluate_per_row(
        recommender,
        train,
        test,
        metric_k_values,
        method,
        n_negatives,
        seed,
        cumulative_history,
        hr_avg_k,
        show_progress=show_progress,
        progress_logger=progress_logger,
        progress_interval=progress_interval,
        verified_only=verified_only,
        max_test_rows=max_test_rows,
        method_variant=method_variant,
        use_llm_cot=use_llm_cot,
        aggregation=aggregation,
        parallel_workers=parallel_workers,
        checkpoint=checkpoint,
        llm_batch=llm_batch,
        batch_size=batch_size,
        batch_token_budget=batch_token_budget,
    )


def evaluate(
    recommender: Recommender,
    train: list[Interaction],
    test: list[Interaction],
    k_values: list[int],
    method: str = "method",
    n_negatives: int | None = None,
    seed: int = 42,
    cumulative_history: bool = False,
    hr_avg_k: tuple[int, ...] = M.HR_AVG_KS,
    *,
    show_progress: bool = False,
    progress_logger: logging.Logger | None = None,
    progress_interval: int = 100,
    verified_only: bool = False,
    max_test_rows: int | None = None,
    method_variant: str | None = None,
    use_llm_cot: bool | None = None,
    eval_protocol: str = "per_row",
    aggregation: str = "row_mean",
    sampled_n_negatives: int | None = None,
    sampled_k_values: list[int] | None = None,
    parallel_workers: int = 1,
    checkpoint_stem: Path | None = None,
    resume: bool = True,
    fresh_checkpoint: bool = False,
    llm_batch: bool = False,
    batch_size: int = 12,
    batch_token_budget: int = 28_000,
) -> EvalResult:
    """Dispatch eval; optional second pass with 1+N negative sampling."""
    _check_parallel_eval(
        parallel_workers, cumulative_history, llm_batch=llm_batch
    )
    if parallel_workers > 1:
        show_progress = False

    if fresh_checkpoint and checkpoint_stem is not None:
        for sampled in (False, True):
            path = pass_checkpoint_path(checkpoint_stem, sampled=sampled)
            path.unlink(missing_ok=True)
            path.with_name(path.stem + ".meta.json").unlink(missing_ok=True)
        fresh_checkpoint = False

    if progress_logger:
        progress_logger.info("fitting recommender on %s train interactions", f"{len(train):,}")
    recommender.fit(train)
    if progress_logger:
        progress_logger.info("fit complete")

    pass_kwargs = dict(
        show_progress=show_progress,
        progress_logger=progress_logger,
        progress_interval=progress_interval,
        verified_only=verified_only,
        max_test_rows=max_test_rows,
        method_variant=method_variant,
        use_llm_cot=use_llm_cot,
        eval_protocol=eval_protocol,
        aggregation=aggregation,
        parallel_workers=parallel_workers,
        checkpoint_stem=checkpoint_stem,
        resume=resume,
        fresh_checkpoint=fresh_checkpoint,
        llm_batch=llm_batch,
        batch_size=batch_size,
        batch_token_budget=batch_token_budget,
    )

    if sampled_n_negatives is not None and n_negatives is not None:
        raise ValueError("Pass only one of n_negatives or sampled_n_negatives")

    if sampled_n_negatives is not None:
        primary = _run_eval_pass(
            recommender,
            train,
            test,
            k_values,
            method,
            None,
            seed,
            cumulative_history,
            hr_avg_k,
            sampled_k_values=None,
            **pass_kwargs,
        )
        if progress_logger:
            progress_logger.info(
                "--- sampled eval: 1 pos + %s negatives, k=%s ---",
                sampled_n_negatives,
                _effective_k_values(k_values, sampled_n_negatives, sampled_k_values),
            )
        sampled = _run_eval_pass(
            recommender,
            train,
            test,
            k_values,
            method,
            sampled_n_negatives,
            seed,
            cumulative_history,
            hr_avg_k,
            sampled_k_values=sampled_k_values,
            **pass_kwargs,
        )
        primary.sampled = sampled.to_sampled_payload()
        return primary

    return _run_eval_pass(
        recommender,
        train,
        test,
        k_values,
        method,
        n_negatives,
        seed,
        cumulative_history,
        hr_avg_k,
        sampled_k_values=sampled_k_values if n_negatives is not None else None,
        **pass_kwargs,
    )


def paired_compare(
    full: EvalResult,
    other: EvalResult,
    metric_key: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> PairedResult:
    """Paired-bootstrap delta on one metric between two runs."""
    return paired_bootstrap(
        full.per_user[metric_key],
        other.per_user[metric_key],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def write_results(out_path: str | Path, result: EvalResult) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return out
