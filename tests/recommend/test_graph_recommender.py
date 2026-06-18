"""GraphRecommender ranking smoke tests."""

from __future__ import annotations

from emorecagent.data.types import Interaction
from emorecagent.eval.runner import build_recommender, evaluate
from emorecagent.kg.memory import InMemoryKG
from emorecagent.llm.schemas import AbsaTriple
from emorecagent.recommend.graph_recommender import GraphRecommender

DAY = 86_400_000


def _train() -> list[Interaction]:
    return [
        Interaction("u1", "i_a", 5.0, 1 * DAY),
        Interaction("u1", "i_b", 5.0, 2 * DAY),
        Interaction("u2", "i_a", 5.0, 1 * DAY),
        Interaction("u2", "i_c", 5.0, 2 * DAY),
    ]


def test_graph_rank_returns_full_candidate_list() -> None:
    rec = build_recommender(
        "emorecagent",
        {
            "train_interactions": _train(),
            "factors": 4,
            "kg_backend": "memory",
            "use_llm_cot": False,
            "use_reflection": False,
            "pool_size": 10,
        },
        seed=0,
    )
    rec.fit(_train())
    rec.prepare_user_query("u1", 3 * DAY)
    candidates = ["i_a", "i_b", "i_c", "i_held"]
    ranked = rec.rank("u1", candidates)
    assert len(ranked) == len(candidates)
    assert set(ranked) == set(candidates)


def test_graph_rank_prefers_graph_head_item() -> None:
    """Graph top item should appear before numeric-only ordering when signaled."""
    kg = InMemoryKG()
    for it in _train():
        kg.upsert_interaction(it)
    kg.upsert_triples(
        "u1",
        "i_b",
        [AbsaTriple(aspect="comfort", opinion="hurts", sentiment="negative")],
        ts=int(1.5 * DAY),
    )
    kg.upsert_item_sentiment("i_held", "comfort", 0.9, 5, ts=500)
    kg.upsert_item_sentiment("i_c", "comfort", -0.5, 5, ts=500)

    rec = GraphRecommender.from_runner_cfg(
        {
            "train_interactions": _train(),
            "factors": 4,
            "kg_backend": "memory",
            "use_llm_cot": False,
            "use_reflection": False,
            "pool_size": 10,
            "alpha": 0.2,
        },
        memory_kg=kg,
    )
    rec.fit(_train())
    rec.prepare_user_query("u1", 3 * DAY)
    ranked = rec.rank("u1", ["i_c", "i_held"])
    assert ranked[0] == "i_held"


def test_evaluate_graph_protocol_full_catalog() -> None:
    test = [Interaction("u1", "i_c", 5.0, 3 * DAY)]
    res = evaluate(
        build_recommender(
            "emorecagent",
            {
                "train_interactions": _train(),
                "factors": 4,
                "kg_backend": "memory",
                "use_llm_cot": False,
                "use_reflection": False,
            },
            seed=0,
        ),
        _train(),
        test,
        [5],
        method="emorecagent",
        method_variant="langgraph",
    )
    assert res.protocol == "full_catalog"
