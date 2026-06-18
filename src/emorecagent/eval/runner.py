"""Experiment runner for ranking evaluation over train/test splits."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..baselines.base import Recommender
from ..baselines.itemknn import ItemKNNRecommender
from ..baselines.popularity import PopularityRecommender
from ..baselines.sequential import SequentialRecommender
from ..baselines.svd import SVDRecommender
from ..data.types import Interaction
from . import metrics as M
from .significance import PairedResult, paired_bootstrap

_AGENTIC_METHODS = frozenset(
    {"aspect_aware", "emorecagent", "emorecagent_fast", "emorecagent_hgt"}
)


def load_split_jsonl(path: str | Path) -> list[Interaction]:
    """Read interactions written by data.split.write_split."""
    out: list[Interaction] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                Interaction(
                    user_id=d["user_id"],
                    item=d["item"],
                    rating=float(d.get("rating", 0.0)),
                    timestamp=int(d.get("timestamp", 0)),
                    helpful_vote=int(d.get("helpful_vote", 0)),
                    verified_purchase=bool(d.get("verified_purchase", False)),
                )
            )
    return out


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
        if method == "emorecagent_hgt":
            hgt_cfg = dict(cfg)
            hgt_cfg["cf_backend"] = "hgt"
            hgt_cfg["kg_backend"] = "memory"
            return GraphRecommender.from_runner_cfg(
                hgt_cfg, config=hgt_cfg.get("app_config")
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
) -> EvalResult:
    """LightGCN-style eval; optional 1+N negative sampling per test item."""
    train_items: dict[str, set[str]] = {}
    for it in train:
        train_items.setdefault(it.user_id, set()).add(it.item)
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
            "n_negatives=%s, verified_only=%s, aggregation=%s",
            f"{len(users):,}",
            f"{sum(len(v) for v in grouped.values()):,}",
            candidate_protocol,
            n_negatives,
            verified_only,
            aggregation,
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
        user_iter: Iterator[str] = users
        if show_progress and total:
            from tqdm import tqdm

            user_iter = tqdm(users, desc="Evaluating users", unit="user", total=total)

        for i, uid in enumerate(user_iter):
            user_tests = grouped[uid]
            relevant = {t.item for t in user_tests}
            n_test_rows += len(relevant)
            seen = set(train_items.get(uid, set()))
            candidates = [item for item in all_items if item not in seen]
            for item in relevant:
                if item not in candidates:
                    candidates.append(item)

            t_query = max(t.timestamp for t in user_tests)
            if hasattr(recommender, "prepare_user_query"):
                recommender.prepare_user_query(uid, t_query)
            ranked = recommender.rank(uid, candidates)

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

            if progress_logger and total and (i + 1) % progress_interval == 0:
                progress_logger.info(
                    "evaluated %s/%s users (%.1f%%)",
                    f"{i + 1:,}",
                    f"{len(users):,}",
                    100.0 * (i + 1) / len(users),
                )

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
) -> EvalResult:
    """Per test-row ranking (Protocol A — EmoRecAgent ablations)."""
    train_items: dict[str, set[str]] = {}
    for it in train:
        train_items.setdefault(it.user_id, set()).add(it.item)
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
            "verified_only=%s, verified_in_test=%s/%s)",
            f"{total:,}",
            f"{len({t.user_id for t in test_order}):,}",
            protocol,
            cumulative_history,
            verified_only,
            f"{n_verified_rows:,}",
            f"{len(test):,}",
        )

    rows: Iterator[Interaction] = test_order
    if show_progress and total:
        from tqdm import tqdm

        rows = tqdm(test_order, desc="Evaluating", unit="row", total=total)

    for i, t in enumerate(rows):
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

        if hasattr(recommender, "prepare_user_query"):
            recommender.prepare_user_query(t.user_id, t.timestamp)
        ranked = recommender.rank(t.user_id, candidates)
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
        if cumulative_history:
            prior_test[t.user_id].append((t.timestamp, t.item))

        if progress_logger and total and (i + 1) % progress_interval == 0:
            progress_logger.info(
                "evaluated %s/%s test rows (%.1f%%)",
                f"{i + 1:,}",
                f"{total:,}",
                100.0 * (i + 1) / total,
            )
            if hasattr(recommender, "log_stage_timing"):
                recommender.log_stage_timing(progress_logger)

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
) -> EvalResult:
    metric_k_values = _effective_k_values(k_values, n_negatives, sampled_k_values)
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
) -> EvalResult:
    """Dispatch eval; optional second pass with 1+N negative sampling."""
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
