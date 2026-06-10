"""Experiment runner (U12 / KTD8).

Loads a resolved config, reads a train/test split, builds the selected method
behind the shared `Recommender` interface, ranks each test user's candidates
(everything not in their train history — the held-out test item is among them by
construction), and reports Recall/NDCG/HR/MRR@K with per-user vectors retained so
ablation/baseline deltas can be significance-tested.

Methods runnable without LLM/KG infrastructure (CF family, popularity, sequential)
run end-to-end here today. Aspect-aware and the full agentic system plug into the
same interface once ABSA (U4) and the KG (U5) provide item-aspect sentiment.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..baselines.base import Recommender
from ..baselines.itemknn import ItemKNNRecommender
from ..baselines.popularity import PopularityRecommender
from ..baselines.sequential import SequentialRecommender
from ..baselines.svd import SVDRecommender
from ..data.types import Interaction
from . import metrics as M
from .significance import PairedResult, paired_bootstrap

_INFRA_METHODS = {
    "aspect_aware": "needs ABSA item-aspect sentiment (U4) + user signals (U5/U6)",
    "emorecagent": "needs the full LangGraph pipeline (U4, U5, U8, U9)",
}


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
                )
            )
    return out


def build_recommender(method: str, cfg: dict, seed: int) -> Recommender:
    if method in _INFRA_METHODS:
        raise NotImplementedError(
            f"method '{method}' is not runnable in the numeric harness yet: "
            f"{_INFRA_METHODS[method]}"
        )
    if method == "popularity":
        return PopularityRecommender()
    if method == "itemknn":
        return ItemKNNRecommender(seed=seed)
    if method in ("svd", "base_cf"):
        return SVDRecommender(factors=int(cfg.get("factors", 64)), seed=seed)
    if method == "sequential":
        return SequentialRecommender()
    raise ValueError(f"Unknown method: {method}")


@dataclass
class EvalResult:
    method: str
    k_values: list[int]
    n_test_users: int
    means: dict[str, float] = field(default_factory=dict)         # "metric@k" -> mean
    per_user: dict[str, list[float]] = field(default_factory=dict)  # "metric@k" -> vec

    def to_json(self) -> dict:
        return {
            "method": self.method,
            "k_values": self.k_values,
            "n_test_users": self.n_test_users,
            "means": self.means,
        }


def evaluate(
    recommender: Recommender,
    train: list[Interaction],
    test: list[Interaction],
    k_values: list[int],
    method: str = "method",
    n_negatives: int | None = None,
    seed: int = 42,
) -> EvalResult:
    """Rank each test user's candidates and aggregate metrics.

    `n_negatives`: if set, score the held-out item against that many sampled
    negatives (the common sampled-metric protocol) instead of the full catalog.
    """
    recommender.fit(train)

    train_items: dict[str, set[str]] = {}
    for it in train:
        train_items.setdefault(it.user_id, set()).add(it.item)
    all_items = sorted({it.item for it in train} | {it.item for it in test})
    rng = random.Random(seed)

    per_user: dict[str, list[float]] = {
        f"{m}@{k}": [] for k in k_values for m in M.METRIC_NAMES
    }
    n_users = 0
    for t in test:
        seen = train_items.get(t.user_id, set())
        pool = [i for i in all_items if i not in seen]
        if t.item not in pool:
            pool.append(t.item)
        if n_negatives is not None:
            negs = [i for i in pool if i != t.item]
            rng.shuffle(negs)
            candidates = [t.item, *negs[:n_negatives]]
        else:
            candidates = pool
        ranked = recommender.rank(t.user_id, candidates)
        relevant = {t.item}
        for k in k_values:
            scored = M.evaluate_ranking(ranked, relevant, k)
            for m, v in scored.items():
                per_user[f"{m}@{k}"].append(v)
        n_users += 1

    means = {
        key: (sum(vec) / len(vec) if vec else 0.0) for key, vec in per_user.items()
    }
    return EvalResult(
        method=method,
        k_values=k_values,
        n_test_users=n_users,
        means=means,
        per_user=per_user,
    )


def paired_compare(
    full: EvalResult,
    other: EvalResult,
    metric_key: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> PairedResult:
    """Paired-bootstrap delta on one metric@k between two runs over the same users."""
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
