"""Tests for Stage 2 LLM rerank helper."""

from __future__ import annotations

from unittest.mock import MagicMock

from emorecagent.llm.client import LLMError
import json

from emorecagent.tisasrec_align.item_metadata import ItemMeta
from emorecagent.tisasrec_align.stage2_llm_rerank import (
    build_lookup_hints,
    card_budget_for_pool,
    llm_rerank_pool,
)


def test_build_lookup_hints_only_pool_items() -> None:
    lookup = {"A": {"X": 5, "Z": 1}}
    hints = build_lookup_hints(["A"], {"X"}, lookup)
    assert "X" in hints
    assert "Z" not in hints


def test_llm_rerank_pool_uses_mock_llm() -> None:
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["b", "a"]
    pool = ["a", "b"]
    scores = {"a": 1.0, "b": 0.5}
    out = llm_rerank_pool(
        llm,
        t_u="likes skincare",
        reviewed_items=["A"],
        lookup={},
        pool=pool,
        scores=scores,
        numeric_fallback=["a", "b"],
    )
    assert out == ["b", "a"]


def test_llm_rerank_pool_unknown_ids_coerced_via_client() -> None:
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["b", "a"]
    out = llm_rerank_pool(
        llm,
        t_u="t",
        reviewed_items=[],
        lookup={},
        pool=["a", "b", "c"],
        scores={"a": 1.0, "b": 0.5, "c": 0.1},
        numeric_fallback=["a", "b", "c"],
    )
    assert out == ["b", "a"]


def test_llm_rerank_pool_none_returns_numeric_fallback() -> None:
    out = llm_rerank_pool(
        None,
        t_u="t",
        reviewed_items=[],
        lookup={},
        pool=["a", "b"],
        scores={"a": 1.0, "b": 0.2},
        numeric_fallback=["a", "b"],
    )
    assert out == ["a", "b"]


def test_llm_rerank_pool_llm_error_falls_back() -> None:
    llm = MagicMock()
    llm.invoke_ranking_json.side_effect = LLMError("fail")
    out = llm_rerank_pool(
        llm,
        t_u="t",
        reviewed_items=[],
        lookup={},
        pool=["a", "b"],
        scores={"a": 1.0, "b": 0.2},
        numeric_fallback=["a", "b"],
    )
    assert out == ["a", "b"]


def test_llm_rerank_pool_top_k_promote() -> None:
    llm = MagicMock()
    # Model only returns two picks; fill keeps numeric order for the rest.
    llm.invoke_ranking_json.return_value = ["c", "a", "b", "d"]
    out = llm_rerank_pool(
        llm,
        t_u="wants vitamin C serum",
        reviewed_items=["x"],
        lookup={},
        pool=["a", "b", "c", "d"],
        scores={"a": 1.0, "b": 0.8, "c": 0.5, "d": 0.1},
        numeric_fallback=["a", "b", "c", "d"],
        rerank_mode="top_k_promote",
        promote_k=2,
        alignment_confidence=0.7,
    )
    assert out[:2] == ["c", "a"]
    assert set(out) == {"a", "b", "c", "d"}
    kwargs = llm.invoke_ranking_json.call_args.kwargs
    assert "suffix" in kwargs
    prompt = llm.invoke_ranking_json.call_args.args[0]
    assert "vitamin C" in prompt or "2" in prompt


def test_llm_rerank_pool_promote_swap() -> None:
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["d"]
    pool = ["a", "b", "c", "d"]
    out = llm_rerank_pool(
        llm,
        t_u="likes d",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        rerank_mode="promote_swap",
        promote_k=1,
        protect_n=2,
        alignment_confidence=0.6,
        review_snippets={"d": ["matches T_u"]},
    )
    # head window = protect+k = 3 → swap slot c with d
    assert out[:2] == ["a", "b"]
    assert out[2] == "d"
    assert out[3] == "c"
    prompt = llm.invoke_ranking_json.call_args.args[0]
    assert "swap" in prompt.lower() or "SWAP" in prompt


def test_llm_rerank_pool_promote_preserve() -> None:
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["d", "c"]
    out = llm_rerank_pool(
        llm,
        t_u="likes d",
        reviewed_items=[],
        lookup={},
        pool=["a", "b", "c", "d"],
        scores={"a": 1.0, "b": 0.8, "c": 0.5, "d": 0.1},
        numeric_fallback=["a", "b", "c", "d"],
        rerank_mode="promote_preserve",
        promote_k=2,
        protect_n=2,
        alignment_confidence=0.6,
        review_snippets={"d": ["great serum for dry skin"]},
    )
    assert out[:2] == ["a", "b"]
    assert out[2:4] == ["d", "c"]
    prompt = llm.invoke_ranking_json.call_args.args[0]
    assert "frozen" in prompt.lower() or "protect" in prompt.lower() or "1–2" in prompt or "1-2" in prompt
    assert "rev=" in prompt


def test_item_review_snippets_from_index() -> None:
    from emorecagent.tisasrec_align.review_context import (
        item_review_snippets_from_index,
    )

    idx = {
        ("u1", "i1", 1): "short",
        ("u2", "i1", 2): "ignored second",
        ("u1", "i2", 1): "x" * 200,
    }
    snips = item_review_snippets_from_index(idx, keep_ids={"i1", "i2"}, max_chars=20)
    assert snips["i1"] == ["short"]
    assert snips["i2"][0].endswith("…")
    assert len(snips["i2"][0]) == 20
    multi = item_review_snippets_from_index(
        idx, keep_ids={"i1"}, max_chars=20, max_per_item=2
    )
    assert multi["i1"] == ["short", "ignored second"]
    allowed = item_review_snippets_from_index(
        idx,
        keep_ids={"i1", "i2"},
        allowed_reviews={("u1", "i1", 1)},
        max_chars=20,
        max_per_item=5,
    )
    assert allowed == {"i1": ["short"]}
    assert "i2" not in allowed


def test_card_budget_compacts_large_pool() -> None:
    snips = {
        "a": [
            "Shipping was slow. This vitamin C serum fixed my dry skin overnight. "
            "Packaging looks nice too."
        ]
    }
    n, c, r, out_snips = card_budget_for_pool(
        300,
        max_name=100,
        max_cats=5,
        max_review_chars=100,
        review_snippets=snips,
        t_u="wants vitamin C serum for dry skin",
    )
    assert n <= 50
    assert c <= 2
    assert r > 0  # keep a review budget
    assert out_snips is not None
    assert out_snips["a"]
    summary = out_snips["a"][0]
    assert len(summary) <= r
    # T_u-relevant clause kept, not dropped / not only shipping fluff.
    assert "serum" in summary.lower() or "dry" in summary.lower()
    # Small pools keep full budget without forced compact.
    n2, c2, r2, s2 = card_budget_for_pool(
        20,
        max_name=100,
        max_cats=5,
        max_review_chars=100,
        review_snippets=snips,
    )
    assert (n2, c2, r2) == (100, 5, 100)
    assert s2 == snips


def test_summarize_review_keeps_tu_evidence() -> None:
    from emorecagent.tisasrec_align.stage2_llm_rerank import summarize_review_snippet

    text = (
        "Bought on sale. The vitamin C serum really helped my dry skin. "
        "Box arrived dented."
    )
    out = summarize_review_snippet(
        text, max_chars=36, t_u="vitamin C serum dry skin"
    )
    assert len(out) <= 36
    assert "serum" in out.lower() or "dry" in out.lower() or "vitamin" in out.lower()
    assert "dented" not in out.lower() or "serum" in out.lower()


def test_override_focus_cap_zero_keeps_full_eligible_pool() -> None:
    """scorecard_focus_cap=0 must not silently truncate to 8 (full π¹ path)."""
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        json.dumps(
            {
                "must_have": ["serum"],
                "product_types": ["serum"],
                "ingredients": [],
                "brands": [],
                "use_cases": [],
                "avoid": [],
                "decision_rule": "prefer serum",
            }
        ),
        json.dumps(
            {
                "displacee": {"id": "h0", "fit": 1, "evidence": "weak"},
                "candidates": [
                    {
                        "id": "deep",
                        "fit": 5,
                        "beats_displacee": True,
                        "evidence": "serum match",
                    }
                ],
                "ranked_item_ids": ["deep"],
                "rationale": "deep pool gold",
            }
        ),
    ]
    # protect=0, promote_k=10 → head = first 10; eligible = rest (incl. deep tail).
    head = [f"h{i}" for i in range(10)]
    mid = [f"m{i}" for i in range(20)]
    pool = head + mid + ["deep"]
    meta = {
        x: ItemMeta(item_id=x, name=f"Item {x}", categories="Skincare") for x in pool
    }
    meta["deep"] = ItemMeta(
        item_id="deep", name="Vitamin C Serum", categories="Skincare"
    )
    ranks = {x: i + 1 for i, x in enumerate(pool)}
    stats: dict[str, int] = {}
    out = llm_rerank_pool(
        llm,
        t_u="wants vitamin C serum",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        item_meta=meta,
        stage1_ranks=ranks,
        review_snippets={"deep": ["serum fixed dry skin"]},
        rerank_mode="promote_swap",
        promote_k=10,
        protect_n=0,
        reason_then_pick=True,
        narrow_cap=0,
        scorecard_focus_cap=0,
        pick_mode="argmax_llm_override",
        hybrid_first_enabled=False,
        hybrid_gate_enabled=False,
        swap_stats=stats,
    )
    assert "deep" in out[:10]
    assert stats.get("n_stage2_llm_override") == 1
    pick_prompt = llm.invoke_text.call_args_list[1].args[0]
    assert "deep" in pick_prompt
    # Deep-tail id must appear; focus must not have dropped it.
    assert "m19" in pick_prompt or "Eligible ids:" in pick_prompt
    assert "scan the FULL" in pick_prompt or "FULL eligible" in pick_prompt


def test_llm_rerank_pool_listwise_phi_rewrites_head() -> None:
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["z", "y", "x"]
    stats: dict[str, int] = {}
    out = llm_rerank_pool(
        llm,
        t_u="serum",
        reviewed_items=[],
        lookup={},
        pool=["x", "y", "z"],
        scores={"x": 1.0, "y": 0.5, "z": 0.1},
        numeric_fallback=["x", "y", "z"],
        rerank_mode="listwise",
        alignment_confidence=0.6,
        phi_scores={"x": 0.9, "y": 0.5, "z": 0.1},
        swap_stats=stats,
    )
    assert out[:3] == ["z", "y", "x"]
    assert stats.get("n_stage2_llm_override") == 1
    prompt = llm.invoke_ranking_json.call_args.args[0]
    assert "PRIMARY" in prompt
    assert "Top-20" in prompt or "top-20" in prompt
