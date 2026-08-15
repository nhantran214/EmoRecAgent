"""Tests for Stage-2 3+4+1 helpers and reason-then-pick path."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from emorecagent.tisasrec_align.item_metadata import ItemMeta
from emorecagent.tisasrec_align.review_context import item_review_snippets_from_index
from emorecagent.tisasrec_align.stage2_llm_rerank import (
    build_candidate_cards,
    llm_rerank_pool,
)
from emorecagent.tisasrec_align.stage2_reason_promote import (
    _scorecard_allowed_ids,
    annotate_cards_with_overlap,
    build_listwise_window,
    build_scorecard_focus,
    extract_preference_facts,
    filter_eligible_hybrid_first,
    filter_picks_hybrid_gate,
    format_preference_facts,
    hybrid_lexical_allows,
    match_snippets_to_tu,
    narrow_llm_shortlist,
    reason_pick_promotions,
    select_lexical_argmax,
    select_lexical_first_promotion,
    select_tu_matched_snippet,
)


def test_parse_json_object_salvages_broken_pref_lists() -> None:
    from emorecagent.tisasrec_align.stage2_reason_promote import _parse_json_object

    # Unescaped inner quotes + trailing comma — common TGI glitch.
    raw = (
        '{"brands":["Neutrogena"],"product_types":["vitamin C serum"],'
        '"ingredients":["vitamin C"],"use_cases":["for "dry" skin"],'
        '"avoid":[],"keywords":["serum","dry"],}'
    )
    data = _parse_json_object(raw)
    assert "Neutrogena" in data["brands"]
    assert any("serum" in x.lower() for x in data["product_types"] + data["keywords"])
    assert "vitamin C" in data["ingredients"]


def test_select_tu_matched_snippet_prefers_overlap() -> None:
    t_u = "I want a vitamin C serum for dry skin"
    snip = select_tu_matched_snippet(
        [
            "fast shipping, nice packaging",
            "this vitamin C serum helped my dry skin",
            "smells floral",
        ],
        t_u,
        max_chars=80,
    )
    assert snip is not None
    assert "vitamin" in snip.lower()
    assert "dry" in snip.lower()


def test_match_snippets_to_tu_one_per_item() -> None:
    matched = match_snippets_to_tu(
        {
            "a": ["unrelated review", "great retinol for acne"],
            "b": ["moisturizer for dry skin"],
        },
        "prefers retinol for acne-prone skin",
        item_ids=["a", "b"],
        max_chars=100,
    )
    assert matched["a"][0].lower().find("retinol") >= 0
    assert len(matched["a"]) == 1
    assert "dry" in matched["b"][0].lower() or "moisturizer" in matched["b"][0].lower()


def test_narrow_llm_shortlist_keeps_head_and_top_overlap() -> None:
    meta = {
        "x": ItemMeta(item_id="x", name="Vitamin C Serum", categories="Skincare"),
        "y": ItemMeta(item_id="y", name="Garden Hose", categories="Outdoor"),
        "z": ItemMeta(item_id="z", name="Dry Skin Cream", categories="Skincare"),
        "w": ItemMeta(item_id="w", name="Phone Case", categories="Electronics"),
    }
    # protect=2, promote=1 → head = a,b,c; outside = x,y,z,w
    pool = ["a", "b", "c", "x", "y", "z", "w"]
    ranks = {item: i + 1 for i, item in enumerate(pool)}
    out = narrow_llm_shortlist(
        pool,
        t_u="vitamin C serum for dry skin",
        stage1_ranks=ranks,
        item_meta=meta,
        review_snippets=None,
        protect_n=2,
        promote_k=1,
        narrow_cap=2,
    )
    assert out[:3] == ["a", "b", "c"]
    assert set(out[3:]) <= {"x", "y", "z", "w"}
    assert len(out) == 5  # head 3 + narrow 2
    # Skincare overlap items should beat hose/phone.
    assert "y" not in out or "w" not in out
    assert "x" in out and "z" in out


def test_item_review_snippets_keeps_multiple_candidates() -> None:
    idx = {
        ("u1", "i1", 1): "first review about packaging",
        ("u2", "i1", 2): "second review vitamin C serum",
        ("u3", "i1", 3): "third ignored when cap=2",
    }
    snips = item_review_snippets_from_index(
        idx, keep_ids={"i1"}, max_chars=40, max_per_item=2
    )
    assert len(snips["i1"]) == 2
    assert "packaging" in snips["i1"][0]
    assert "vitamin" in snips["i1"][1].lower()


def test_llm_rerank_pool_lexical_argmax_skips_llm() -> None:
    llm = MagicMock()
    meta = {
        "c": ItemMeta(item_id="c", name="Random Lipstick", categories="Makeup"),
        "d": ItemMeta(item_id="d", name="Vitamin C Serum", categories="Skincare"),
        "e": ItemMeta(
            item_id="e",
            name="Vitamin C Serum for Dry Skin",
            categories="Skincare Serum",
        ),
        "f": ItemMeta(item_id="f", name="Phone Case", categories="Electronics"),
    }
    pool = ["a", "b", "c", "d", "e", "f"]
    ranks = {x: i + 1 for i, x in enumerate(pool)}
    ranks["d"] = 15
    ranks["e"] = 12
    ranks["f"] = 25
    stats: dict[str, int] = {}
    out = llm_rerank_pool(
        llm,
        t_u="wants vitamin C serum for dry skin",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        item_meta=meta,
        stage1_ranks=ranks,
        rerank_mode="promote_swap",
        promote_k=1,
        protect_n=2,
        reason_then_pick=True,
        narrow_cap=5,
        hybrid_first_enabled=True,
        hybrid_overlap_delta=0,
        hybrid_overlap_delta_out_of_band=1,
        pick_mode="lexical_argmax",
        lexical_first_enabled=False,
        swap_stats=stats,
    )
    assert out[:2] == ["a", "b"]
    assert out[2] == "e"  # higher overlap + better π¹ than d
    assert out[3] == "c"
    assert llm.invoke_text.call_count == 0
    assert stats.get("n_stage2_lexical_argmax") == 1
    assert stats.get("n_stage2_swaps") == 1
    assert stats.get("n_stage2_empty_picks", 0) == 0


def test_select_lexical_argmax_prefers_overlap_then_rank() -> None:
    meta = {
        "good": ItemMeta(item_id="good", name="Vitamin Serum", categories="Skincare"),
        "better": ItemMeta(
            item_id="better",
            name="Vitamin C Serum Dry Skin",
            categories="Skincare",
        ),
    }
    pick = select_lexical_argmax(
        ["good", "better"],
        t_u="wants vitamin C serum for dry skin",
        item_meta=meta,
        review_snippets=None,
        stage1_ranks={"good": 15, "better": 18},
    )
    assert pick == "better"


def test_select_lexical_argmax_top_k_orders_by_overlap() -> None:
    from emorecagent.tisasrec_align.stage2_reason_promote import (
        select_lexical_argmax_top_k,
    )

    meta = {
        "weak": ItemMeta(item_id="weak", name="Phone Case", categories="Electronics"),
        "good": ItemMeta(item_id="good", name="Vitamin Serum", categories="Skincare"),
        "best": ItemMeta(
            item_id="best",
            name="Vitamin C Serum Dry Skin",
            categories="Skincare",
        ),
    }
    picks = select_lexical_argmax_top_k(
        ["weak", "good", "best"],
        promote_k=2,
        t_u="wants vitamin C serum for dry skin",
        item_meta=meta,
        review_snippets=None,
        stage1_ranks={"weak": 11, "good": 15, "best": 18},
    )
    assert picks == ["best", "good"]


def test_llm_rerank_pool_lexical_argmax_promote_k2() -> None:
    llm = MagicMock()
    meta = {
        "c": ItemMeta(item_id="c", name="Random Lipstick", categories="Makeup"),
        "d": ItemMeta(item_id="d", name="Vitamin Serum", categories="Skincare"),
        "e": ItemMeta(
            item_id="e",
            name="Vitamin C Serum for Dry Skin",
            categories="Skincare Serum",
        ),
        "x": ItemMeta(item_id="x", name="Other Makeup", categories="Makeup"),
    }
    # protect_n=2, promote_k=2 → head a,b,c,x; displacee = c,x (slots 3–4)
    pool = ["a", "b", "c", "x", "d", "e"]
    ranks = {x: i + 1 for i, x in enumerate(pool)}
    ranks["d"] = 15
    ranks["e"] = 12
    stats: dict[str, int] = {}
    out = llm_rerank_pool(
        llm,
        t_u="wants vitamin C serum for dry skin",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        item_meta=meta,
        stage1_ranks=ranks,
        rerank_mode="promote_swap",
        promote_k=2,
        protect_n=2,
        reason_then_pick=True,
        narrow_cap=5,
        hybrid_first_enabled=True,
        hybrid_overlap_delta=0,
        hybrid_overlap_delta_out_of_band=1,
        pick_mode="lexical_argmax",
        lexical_first_enabled=False,
        swap_stats=stats,
    )
    assert out[:2] == ["a", "b"]
    assert out[2] == "e"
    assert out[3] == "d"
    assert llm.invoke_text.call_count == 0
    assert stats.get("n_stage2_lexical_argmax") == 1


def test_build_scorecard_focus_keeps_near_miss() -> None:
    from emorecagent.tisasrec_align.stage2_reason_promote import build_scorecard_focus

    meta = {
        "near": ItemMeta(item_id="near", name="Vitamin Serum", categories="Skincare"),
        "far": ItemMeta(item_id="far", name="Vitamin C Serum Dry", categories="Skincare"),
        "weak": ItemMeta(item_id="weak", name="Phone", categories="Electronics"),
    }
    ranks = {"near": 15, "far": 35, "weak": 40}
    focus = build_scorecard_focus(
        ["weak", "far", "near"],
        t_u="wants vitamin C serum for dry skin",
        preference_facts=None,
        item_meta=meta,
        review_snippets=None,
        stage1_ranks=ranks,
        focus_cap=2,
        near_miss_lo=11,
        near_miss_hi=20,
    )
    assert "near" in focus
    assert focus[0] == "near"


def test_argmax_llm_override_prefers_llm_pick() -> None:
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        json.dumps(
            {
                "must_have": ["serum"],
                "nice_to_have": [],
                "avoid": [],
                "brands": [],
                "product_types": ["serum"],
                "ingredients": ["vitamin C"],
                "keywords": ["serum"],
                "decision_rule": "prefer serum",
            }
        ),
        json.dumps(
            {
                "displacee": {"id": "c", "fit": 1, "evidence": "lipstick"},
                "candidates": [
                    {
                        "id": "d",
                        "fit": 5,
                        "beats_displacee": True,
                        "evidence": "serum match",
                    }
                ],
                "ranked_item_ids": ["d"],
                "rationale": "override low ov with evidence",
            }
        ),
    ]
    meta = {
        "c": ItemMeta(item_id="c", name="Random Lipstick", categories="Makeup"),
        "d": ItemMeta(item_id="d", name="Night Cream", categories="Skincare"),
        "e": ItemMeta(
            item_id="e",
            name="Vitamin C Serum for Dry Skin",
            categories="Skincare",
        ),
    }
    pool = ["a", "b", "c", "x", "d", "e"]
    ranks = {x: i + 1 for i, x in enumerate(pool)}
    ranks["d"] = 15
    ranks["e"] = 12
    stats: dict[str, int] = {}
    out = llm_rerank_pool(
        llm,
        t_u="wants vitamin C serum for dry skin",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        item_meta=meta,
        stage1_ranks=ranks,
        review_snippets={"d": ["this serum fixed my dry skin"]},
        rerank_mode="promote_swap",
        promote_k=2,
        protect_n=2,
        reason_then_pick=True,
        narrow_cap=5,
        hybrid_first_enabled=True,
        hybrid_min_overlap=2,
        hybrid_overlap_delta=0,
        scorecard_focus_cap=5,
        pick_mode="argmax_llm_override",
        swap_stats=stats,
    )
    assert out[:2] == ["a", "b"]
    # One LLM pick → swaps into last replaceable slot (index 3 with protect=2,k=2).
    assert out[3] == "d"
    assert stats.get("n_stage2_llm_override") == 1
    assert llm.invoke_text.call_count == 2
    pick_prompt = llm.invoke_text.call_args_list[1].args[0]
    assert "quality-first" in pick_prompt.lower() or "ADVISORY" in pick_prompt


def test_hybrid_min_overlap_rejects_weak_hits() -> None:
    meta = {
        "disp": ItemMeta(item_id="disp", name="Random Lipstick", categories="Makeup"),
        "one": ItemMeta(item_id="one", name="Vitamin Balm", categories="Skincare"),
        "two": ItemMeta(
            item_id="two", name="Vitamin C Serum", categories="Skincare"
        ),
    }
    t_u = "wants vitamin C serum"
    ranks = {"disp": 10, "one": 15, "two": 16}
    assert not hybrid_lexical_allows(
        "one",
        displacee_ids=["disp"],
        t_u=t_u,
        preference_facts=None,
        item_meta=meta,
        review_snippets=None,
        stage1_ranks=ranks,
        overlap_delta=0,
        overlap_delta_out_of_band=1,
        min_overlap=2,
    )
    assert hybrid_lexical_allows(
        "two",
        displacee_ids=["disp"],
        t_u=t_u,
        preference_facts=None,
        item_meta=meta,
        review_snippets=None,
        stage1_ranks=ranks,
        overlap_delta=0,
        overlap_delta_out_of_band=1,
        min_overlap=2,
    )


def test_reason_pick_overlap_grounded_uses_v3() -> None:
    llm = MagicMock()
    llm.invoke_text.return_value = json.dumps(
        {
            "displacee": {"id": "c", "fit": 1, "evidence": "lipstick"},
            "candidates": [
                {
                    "id": "d",
                    "fit": 5,
                    "beats_displacee": True,
                    "evidence": "serum match",
                }
            ],
            "ranked_item_ids": ["d"],
            "rationale": "higher ov and must_have",
        }
    )
    picks = reason_pick_promotions(
        llm,
        t_u="wants vitamin C serum",
        preference_facts={"must_have": ["serum"], "keywords": ["vitamin"]},
        candidate_cards="d | name=Vitamin C Serum | ov=3/0",
        displacee_cards="c | name=Lipstick | ov=0/0",
        promote_k=1,
        eligible_ids=["d"],
        depth="deep",
        overlap_grounded=True,
    )
    assert picks == ["d"]
    prompt = llm.invoke_text.call_args.args[0]
    assert "overlap-grounded" in prompt.lower() or "ov=" in prompt.lower()
    assert "DEFAULT" in prompt or "abstain" in prompt.lower()


def test_filter_eligible_hybrid_first_drops_weak() -> None:
    meta = {
        "disp": ItemMeta(item_id="disp", name="Random Lipstick", categories="Makeup"),
        "good": ItemMeta(
            item_id="good", name="Vitamin C Serum", categories="Skincare"
        ),
        "weak": ItemMeta(item_id="weak", name="Phone Case", categories="Electronics"),
    }
    kept, n_filtered = filter_eligible_hybrid_first(
        ["weak", "good"],
        displacee_ids=["disp"],
        t_u="wants vitamin C serum for dry skin",
        preference_facts={"must_have": ["serum"], "keywords": ["vitamin"]},
        item_meta=meta,
        review_snippets=None,
        stage1_ranks={"disp": 10, "good": 15, "weak": 80},
        overlap_delta=1,
        overlap_delta_out_of_band=2,
        rank_lo=11,
        rank_hi=40,
    )
    assert kept == ["good"]
    assert n_filtered == 1


def test_annotate_cards_with_overlap_appends_ov() -> None:
    meta = {
        "disp": ItemMeta(item_id="disp", name="Random Lipstick", categories="Makeup"),
        "good": ItemMeta(
            item_id="good", name="Vitamin C Serum", categories="Skincare"
        ),
    }
    cards = "good | S=1.0000\ndisp | S=0.5000"
    out = annotate_cards_with_overlap(
        cards,
        ["good", "disp"],
        t_u="wants vitamin C serum",
        preference_facts={"keywords": ["vitamin", "serum"]},
        displacee_ids=["disp"],
        item_meta=meta,
        review_snippets=None,
    )
    assert "ov=" in out
    assert out.splitlines()[0].startswith("good")


def test_llm_rerank_pool_hybrid_first_constrains_pick() -> None:
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        json.dumps(
            {
                "must_have": ["serum"],
                "nice_to_have": [],
                "avoid": [],
                "brands": [],
                "product_types": ["serum"],
                "ingredients": ["vitamin C"],
                "keywords": ["serum", "vitamin"],
                "decision_rule": "prefer serum",
            }
        ),
        json.dumps(
            {
                "displacee": {"id": "c", "fit": 1, "evidence": "lipstick"},
                "candidates": [
                    {
                        "id": "d",
                        "fit": 5,
                        "beats_displacee": True,
                        "evidence": "serum",
                    }
                ],
                "ranked_item_ids": ["d"],
                "rationale": "d is serum",
            }
        ),
    ]
    meta = {
        "c": ItemMeta(item_id="c", name="Random Lipstick", categories="Makeup"),
        "d": ItemMeta(item_id="d", name="Vitamin C Serum", categories="Skincare"),
        "e": ItemMeta(item_id="e", name="Phone Case", categories="Electronics"),
    }
    pool = ["a", "b", "c", "d", "e"]
    ranks = {x: i + 1 for i, x in enumerate(pool)}
    ranks["d"] = 15
    ranks["e"] = 25
    stats: dict[str, int] = {}
    out = llm_rerank_pool(
        llm,
        t_u="wants vitamin C serum for dry skin",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        item_meta=meta,
        stage1_ranks=ranks,
        rerank_mode="promote_swap",
        promote_k=1,
        protect_n=2,
        reason_then_pick=True,
        reason_depth="deep",
        narrow_cap=5,
        hybrid_first_enabled=True,
        hybrid_gate_enabled=True,
        hybrid_overlap_delta=1,
        hybrid_overlap_delta_out_of_band=2,
        lexical_first_enabled=False,
        swap_stats=stats,
    )
    assert out[:2] == ["a", "b"]
    assert out[2] == "d"
    assert llm.invoke_text.call_count == 2
    pick_prompt = llm.invoke_text.call_args_list[1].args[0]
    assert "ov=" in pick_prompt
    # Weak phone case should be filtered before pick.
    assert stats.get("n_stage2_hybrid_first_filtered", 0) >= 1
    assert stats.get("n_stage2_swaps") == 1
    assert stats.get("n_stage2_lexical_first", 0) == 0


def test_select_lexical_first_promotion_picks_best_margin() -> None:
    meta = {
        "disp": ItemMeta(item_id="disp", name="Random Lipstick", categories="Makeup"),
        "good": ItemMeta(
            item_id="good", name="Vitamin C Serum", categories="Skincare"
        ),
        "better": ItemMeta(
            item_id="better",
            name="Vitamin C Serum for Dry Skin",
            categories="Skincare Serum",
        ),
        "weak": ItemMeta(item_id="weak", name="Phone Case", categories="Electronics"),
    }
    t_u = "wants vitamin C serum for dry skin"
    ranks = {"disp": 10, "good": 15, "better": 12, "weak": 18}
    pick = select_lexical_first_promotion(
        ["weak", "good", "better"],
        displacee_ids=["disp"],
        t_u=t_u,
        item_meta=meta,
        review_snippets=None,
        stage1_ranks=ranks,
        overlap_delta=1,
    )
    assert pick == "better"


def test_select_lexical_first_promotion_none_when_no_margin() -> None:
    meta = {
        "disp": ItemMeta(
            item_id="disp", name="Vitamin Serum", categories="Skincare"
        ),
        "cand": ItemMeta(item_id="cand", name="Phone Case", categories="Electronics"),
    }
    assert (
        select_lexical_first_promotion(
            ["cand"],
            displacee_ids=["disp"],
            t_u="wants vitamin C serum",
            item_meta=meta,
            review_snippets=None,
            stage1_ranks={"disp": 10, "cand": 15},
            overlap_delta=1,
        )
        is None
    )


def test_llm_rerank_pool_lexical_first_skips_llm() -> None:
    llm = MagicMock()
    meta = {
        "c": ItemMeta(item_id="c", name="Random Lipstick", categories="Makeup"),
        "d": ItemMeta(item_id="d", name="Vitamin C Serum", categories="Skincare"),
        "e": ItemMeta(item_id="e", name="Phone Case", categories="Electronics"),
    }
    pool = ["a", "b", "c", "d", "e"]
    ranks = {x: i + 1 for i, x in enumerate(pool)}
    ranks["d"] = 15
    ranks["e"] = 25
    stats: dict[str, int] = {}
    out = llm_rerank_pool(
        llm,
        t_u="wants vitamin C serum for dry skin",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        item_meta=meta,
        stage1_ranks=ranks,
        rerank_mode="promote_swap",
        promote_k=1,
        protect_n=2,
        reason_then_pick=True,
        narrow_cap=2,
        lexical_first_enabled=True,
        lexical_first_rank_lo=11,
        lexical_first_rank_hi=20,
        lexical_first_overlap_delta=1,
        swap_stats=stats,
    )
    assert out[:2] == ["a", "b"]
    assert out[2] == "d"
    assert out[3] == "c"
    assert llm.invoke_text.call_count == 0
    assert stats.get("n_stage2_lexical_first") == 1
    assert stats.get("n_stage2_swaps") == 1


def test_hybrid_lexical_gate_requires_overlap_margin() -> None:
    meta = {
        "disp": ItemMeta(item_id="disp", name="Random Lipstick", categories="Makeup"),
        "good": ItemMeta(
            item_id="good", name="Vitamin C Serum", categories="Skincare"
        ),
        "weak": ItemMeta(item_id="weak", name="Phone Case", categories="Electronics"),
    }
    t_u = "wants vitamin C serum for dry skin"
    ranks = {"disp": 10, "good": 15, "weak": 80}
    assert hybrid_lexical_allows(
        "good",
        displacee_ids=["disp"],
        t_u=t_u,
        preference_facts={"must_have": ["serum"], "keywords": ["vitamin"]},
        item_meta=meta,
        review_snippets=None,
        stage1_ranks=ranks,
        overlap_delta=1,
        overlap_delta_out_of_band=2,
        rank_lo=11,
        rank_hi=40,
    )
    # Out-of-band weak item needs larger margin; lipstick vs phone both ~0 → fail.
    assert not hybrid_lexical_allows(
        "weak",
        displacee_ids=["disp"],
        t_u=t_u,
        preference_facts=None,
        item_meta=meta,
        review_snippets=None,
        stage1_ranks=ranks,
        overlap_delta=1,
        overlap_delta_out_of_band=2,
        rank_lo=11,
        rank_hi=40,
    )
    kept, blocked = filter_picks_hybrid_gate(
        ["good", "weak"],
        displacee_ids=["disp"],
        t_u=t_u,
        preference_facts={"must_have": ["serum"]},
        item_meta=meta,
        review_snippets=None,
        stage1_ranks=ranks,
    )
    assert kept == ["good"]
    assert blocked == 1


def test_format_preference_facts_includes_deep_fields() -> None:
    text = format_preference_facts(
        {
            "must_have": ["serum"],
            "nice_to_have": ["fragrance free"],
            "brands": [],
            "product_types": ["serum"],
            "ingredients": ["vitamin C"],
            "avoid": ["alcohol"],
            "keywords": ["dry"],
            "decision_rule": "swap when candidate is a vitamin C serum",
        }
    )
    assert "must_have: serum" in text
    assert "decision_rule: swap when candidate is a vitamin C serum" in text
    assert "nice_to_have: fragrance free" in text


def test_extract_preference_facts_deep_parses_v2_fields() -> None:
    llm = MagicMock()
    llm.invoke_text.return_value = json.dumps(
        {
            "must_have": ["serum"],
            "nice_to_have": ["dry skin"],
            "avoid": ["alcohol"],
            "brands": [],
            "product_types": ["serum"],
            "ingredients": ["vitamin C"],
            "keywords": ["serum", "dry"],
            "decision_rule": "swap when candidate matches serum and vitamin C",
        }
    )
    facts = extract_preference_facts(llm, "wants vitamin C serum", depth="deep")
    assert facts["must_have"] == ["serum"]
    assert facts["decision_rule"] == (
        "swap when candidate matches serum and vitamin C"
    )
    prompt = llm.invoke_text.call_args.args[0]
    assert "must_have" in prompt
    assert "decision_rule" in prompt


def test_resolve_truncated_asin_prefix() -> None:
    from emorecagent.tisasrec_align.stage2_reason_promote import _resolve_item_id

    eligible = ["B0BLABC123", "B0OTHER999"]
    assert _resolve_item_id("B0BLABC123", eligible) == "B0BLABC123"
    assert _resolve_item_id("B0BL", eligible) == "B0BLABC123"
    assert _resolve_item_id("B0", eligible) is None  # ambiguous / too short unique


def test_parse_scorecard_strips_user_role_prefix() -> None:
    from emorecagent.tisasrec_align.stage2_reason_promote import (
        _looks_like_prompt_echo,
        _looks_like_scorecard_json,
        _parse_scorecard_object,
        _scorecard_allowed_ids,
    )

    body = {
        "displacee": {"id": "c", "fit": 2, "evidence": "lipstick"},
        "candidates": [
            {
                "id": "d",
                "fit": 4,
                "beats_displacee": True,
                "evidence": "serum",
            }
        ],
        "ranked_item_ids": ["d"],
        "rationale": "d beats displacee",
    }
    raw = "user\n" + json.dumps(body)
    assert _looks_like_scorecard_json(raw)
    assert not _looks_like_prompt_echo(raw)
    data = _parse_scorecard_object(raw, eligible_ids=["d"])
    assert _scorecard_allowed_ids(data, eligible_ids=["d"], promote_k=1) == ["d"]


def test_parse_scorecard_truncated_runaway_evidence() -> None:
    from emorecagent.tisasrec_align.stage2_reason_promote import (
        _parse_scorecard_object,
    )

    evidence = ", ".join(["concealer"] * 80)
    # Truncated mid-string — the live TGI failure mode.
    raw = (
        '{"displacee":{"id":"B07YT2VKTG","fit":2,"evidence":"' + evidence
    )
    data = _parse_scorecard_object(raw, eligible_ids=["B0C4M5DV5S"])
    assert data["displacee"]["id"] == "B07YT2VKTG"
    assert data["displacee"]["fit"] == 2

    # Truncated after candidates started — keep displacee, salvage candidate if possible.
    raw2 = (
        '{"displacee":{"id":"B00NR1YQHM","fit":2,"evidence":"Hydrating Water Gel"},'
        '"candidates":[{"id":"B0C4M5DV5S","fit":4,"beats_displacee":true,'
        '"evidence":"My hair l'
    )
    data2 = _parse_scorecard_object(raw2, eligible_ids=["B0C4M5DV5S"])
    assert data2["displacee"]["id"] == "B00NR1YQHM"
    cand_ids = {str(r["id"]) for r in data2.get("candidates") or []}
    assert "B0C4M5DV5S" in cand_ids


def test_reason_pick_accepts_user_prefixed_scorecard() -> None:
    llm = MagicMock()
    llm.invoke_text.return_value = "user\n" + json.dumps(
        {
            "displacee": {"id": "c", "fit": 1, "evidence": "lipstick"},
            "candidates": [
                {
                    "id": "d",
                    "fit": 5,
                    "beats_displacee": True,
                    "evidence": "serum title match",
                }
            ],
            "ranked_item_ids": ["d"],
            "rationale": "d beats displacee",
        }
    )
    picks = reason_pick_promotions(
        llm,
        t_u="serum",
        preference_facts={"must_have": ["serum"]},
        candidate_cards="d | Serum",
        displacee_cards="c | Lipstick",
        promote_k=1,
        eligible_ids=["d"],
        depth="deep",
    )
    assert picks == ["d"]
    assert llm.invoke_text.call_count == 1


def test_salvage_scorecard_from_broken_json() -> None:
    from emorecagent.tisasrec_align.stage2_reason_promote import (
        _parse_scorecard_object,
        _scorecard_allowed_ids,
    )

    # Unescaped quote in evidence + missing closing brace — common TGI glitch.
    raw = (
        '{"displacee":{"id":"c","fit":2,"evidence":"lipstick title"},'
        '"candidates":['
        '{"id":"d","fit":5,"beats_displacee":true,"evidence":"serum for "dry" skin"},'
        '{"id":"e","fit":1,"beats_displacee":false,"evidence":"weak"}'
        '],'
        '"ranked_item_ids":["d"],'
        '"rationale":"d wins'
    )
    data = _parse_scorecard_object(raw, eligible_ids=["d", "e"])
    assert data["displacee"]["fit"] == 2
    ids = {str(r["id"]) for r in data["candidates"]}
    assert "d" in ids
    assert _scorecard_allowed_ids(
        data, eligible_ids=["d", "e"], promote_k=1
    ) == ["d"]


def test_scorecard_allowed_ids_requires_fit_delta() -> None:
    data = {
        "displacee": {"id": "c", "fit": 2, "evidence": "lipstick title"},
        "candidates": [
            {
                "id": "d",
                "fit": 4,
                "beats_displacee": True,
                "evidence": "vitamin C serum title",
            },
            {
                "id": "e",
                "fit": 3,
                "beats_displacee": True,
                "evidence": "weak cream",
            },
        ],
        "ranked_item_ids": ["d", "e"],
    }
    # d: 4 >= 2+2 OK; e: 3 < 2+2 drop
    assert _scorecard_allowed_ids(
        data, eligible_ids=["d", "e"], promote_k=2
    ) == ["d"]


def test_scorecard_drops_ranked_without_beats_or_evidence() -> None:
    data = {
        "displacee": {"id": "c", "fit": 1, "evidence": "ok"},
        "candidates": [
            {
                "id": "d",
                "fit": 5,
                "beats_displacee": False,
                "evidence": "serum",
            },
            {
                "id": "e",
                "fit": 5,
                "beats_displacee": True,
                "evidence": "",
            },
        ],
        "ranked_item_ids": ["d", "e"],
    }
    assert (
        _scorecard_allowed_ids(data, eligible_ids=["d", "e"], promote_k=2) == []
    )


def test_reason_pick_retries_on_prompt_echo() -> None:
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        (
            "user\nYou are the Guarded Reranking Agent (deep scorecard). "
            'Decide whether to SWAP any\nReturn JSON: {"displacee":{"id":"..."}'
        ),
        json.dumps(
            {
                "displacee": {"id": "c", "fit": 1, "evidence": "lipstick"},
                "candidates": [
                    {
                        "id": "d",
                        "fit": 5,
                        "beats_displacee": True,
                        "evidence": "serum title match",
                    }
                ],
                "ranked_item_ids": ["d"],
                "rationale": "d beats displacee",
            }
        ),
    ]
    picks = reason_pick_promotions(
        llm,
        t_u="serum",
        preference_facts={"must_have": ["serum"]},
        candidate_cards="d | Serum",
        displacee_cards="c | Lipstick",
        promote_k=1,
        eligible_ids=["d"],
        depth="deep",
    )
    assert picks == ["d"]
    assert llm.invoke_text.call_count == 2
    assert "Eligible ids" in llm.invoke_text.call_args_list[1].args[0]


def test_reason_pick_retries_on_card_regurgitation() -> None:
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        # First reply echoes a candidate card (no JSON).
        "user\nB088PPXPS6 | S=6.0889 | name=Paul Mitchell Spray | cats=Beauty",
        # Repair reply is a valid scorecard.
        json.dumps(
            {
                "displacee": {"id": "c", "fit": 1, "evidence": "lipstick"},
                "candidates": [
                    {
                        "id": "d",
                        "fit": 5,
                        "beats_displacee": True,
                        "evidence": "serum title match",
                    }
                ],
                "ranked_item_ids": ["d"],
                "rationale": "d beats displacee",
            }
        ),
    ]
    picks = reason_pick_promotions(
        llm,
        t_u="serum",
        preference_facts={"must_have": ["serum"], "keywords": ["serum"]},
        candidate_cards="d | Serum\nc | Lipstick",
        displacee_cards="c | Lipstick",
        promote_k=1,
        eligible_ids=["d"],
        depth="deep",
    )
    assert picks == ["d"]
    assert llm.invoke_text.call_count == 2


def test_reason_pick_deep_applies_local_gate() -> None:
    llm = MagicMock()
    llm.invoke_text.return_value = json.dumps(
        {
            "displacee": {"id": "c", "fit": 2, "evidence": "lipstick"},
            "candidates": [
                {
                    "id": "d",
                    "fit": 5,
                    "beats_displacee": True,
                    "evidence": "vitamin C serum for dry skin",
                },
                {
                    "id": "e",
                    "fit": 2,
                    "beats_displacee": False,
                    "evidence": "unrelated",
                },
            ],
            # Model tries to promote both; gate keeps only d.
            "ranked_item_ids": ["d", "e"],
            "rationale": "d matches must_have serum with cited title evidence.",
        }
    )
    picks = reason_pick_promotions(
        llm,
        t_u="vitamin C serum",
        preference_facts={
            "must_have": ["serum"],
            "decision_rule": "swap for serum",
            "keywords": ["serum"],
        },
        candidate_cards="d | Vitamin C Serum\ne | Other",
        displacee_cards="c | Lipstick",
        promote_k=1,
        eligible_ids=["d", "e"],
        depth="deep",
    )
    assert picks == ["d"]
    assert "scorecard" in llm.invoke_text.call_args.args[0].lower() or (
        "fit" in llm.invoke_text.call_args.args[0].lower()
    )


def test_reason_pick_deep_rejects_naked_ids_without_scorecard() -> None:
    llm = MagicMock()
    # Missing candidates scorecard entirely.
    llm.invoke_text.return_value = json.dumps(
        {
            "displacee": {"id": "c", "fit": 1, "evidence": "x"},
            "ranked_item_ids": ["d"],
            "rationale": "guess",
        }
    )
    picks = reason_pick_promotions(
        llm,
        t_u="serum",
        preference_facts={"keywords": ["serum"]},
        candidate_cards="d | serum",
        displacee_cards="c | other",
        promote_k=1,
        eligible_ids=["d"],
        depth="deep",
    )
    assert picks == []


def test_llm_rerank_pool_reason_then_pick_swap() -> None:
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        json.dumps(
            {
                "must_have": ["serum"],
                "nice_to_have": ["dry skin"],
                "avoid": [],
                "brands": [],
                "product_types": ["serum"],
                "ingredients": ["vitamin C"],
                "keywords": ["serum", "vitamin", "dry"],
                "decision_rule": "swap when candidate is vitamin C serum",
            }
        ),
        json.dumps(
            {
                "displacee": {
                    "id": "c",
                    "fit": 1,
                    "evidence": "lipstick title",
                },
                "candidates": [
                    {
                        "id": "d",
                        "fit": 5,
                        "beats_displacee": True,
                        "evidence": "vitamin C serum rev match",
                    },
                    {
                        "id": "e",
                        "fit": 1,
                        "beats_displacee": False,
                        "evidence": "weak",
                    },
                ],
                "ranked_item_ids": ["d"],
                "rationale": "d cites serum evidence and beats displacee by 4.",
            }
        ),
    ]
    meta = {
        "d": ItemMeta(item_id="d", name="Vitamin C Serum", categories="Skincare"),
        "c": ItemMeta(item_id="c", name="Random Lipstick", categories="Makeup"),
    }
    pool = ["a", "b", "c", "d", "e"]
    stats: dict[str, int] = {}
    out = llm_rerank_pool(
        llm,
        t_u="wants vitamin C serum for dry skin",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        item_meta=meta,
        stage1_ranks={x: i + 1 for i, x in enumerate(pool)},
        review_snippets={
            "d": ["great packaging", "vitamin C serum fixed my dry skin"],
            "c": ["pretty color"],
        },
        rerank_mode="promote_swap",
        promote_k=1,
        protect_n=2,
        reason_then_pick=True,
        reason_depth="deep",
        narrow_cap=2,
        alignment_confidence=0.6,
        swap_stats=stats,
    )
    assert out[:2] == ["a", "b"]
    assert out[2] == "d"
    assert out[3] == "c"
    assert llm.invoke_text.call_count == 2
    assert llm.invoke_ranking_json.call_count == 0
    assert stats.get("n_stage2_swaps") == 1
    pick_prompt = llm.invoke_text.call_args_list[1].args[0]
    assert "rev=" in pick_prompt
    assert "vitamin" in pick_prompt.lower()
    assert "scorecard" in pick_prompt.lower() or "fit" in pick_prompt.lower()


def test_llm_rerank_pool_reason_then_pick_empty_keeps_stage1() -> None:
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        json.dumps(
            {
                "must_have": [],
                "nice_to_have": [],
                "avoid": [],
                "brands": [],
                "product_types": [],
                "ingredients": [],
                "keywords": ["serum"],
                "decision_rule": "only swap on clear serum match",
            }
        ),
        json.dumps(
            {
                "displacee": {"id": "c", "fit": 3, "evidence": "ok"},
                "candidates": [
                    {
                        "id": "d",
                        "fit": 3,
                        "beats_displacee": False,
                        "evidence": "not better",
                    }
                ],
                "ranked_item_ids": [],
                "rationale": "nobody beats displacee by 2 points.",
            }
        ),
    ]
    pool = ["a", "b", "c", "d"]
    stats: dict[str, int] = {}
    out = llm_rerank_pool(
        llm,
        t_u="serum",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        rerank_mode="promote_swap",
        promote_k=1,
        protect_n=2,
        reason_then_pick=True,
        reason_depth="deep",
        narrow_cap=4,
        swap_stats=stats,
    )
    assert out == pool
    assert stats.get("n_stage2_empty_picks") == 1


def test_llm_rerank_pool_shallow_still_accepts_simple_pick() -> None:
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        json.dumps(
            {
                "brands": [],
                "product_types": ["serum"],
                "ingredients": [],
                "use_cases": [],
                "avoid": [],
                "keywords": ["serum"],
            }
        ),
        json.dumps({"ranked_item_ids": ["d"], "rationale": "matches"}),
    ]
    pool = ["a", "b", "c", "d"]
    # Place d in near-miss band so hybrid delta=1 applies.
    ranks = {"a": 1, "b": 2, "c": 10, "d": 15}
    out = llm_rerank_pool(
        llm,
        t_u="serum",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=pool,
        item_meta={
            "c": ItemMeta(item_id="c", name="Lipstick", categories="Makeup"),
            "d": ItemMeta(item_id="d", name="Serum", categories="Skincare"),
        },
        stage1_ranks=ranks,
        rerank_mode="promote_swap",
        promote_k=1,
        protect_n=2,
        reason_then_pick=True,
        reason_depth="shallow",
        hybrid_gate_enabled=True,
        narrow_cap=4,
    )
    assert out[2] == "d"


def test_build_scorecard_focus_by_phi_not_overlap() -> None:
    meta = {
        "low": ItemMeta(item_id="low", name="Vitamin C Serum", categories="Skincare"),
        "mid": ItemMeta(item_id="mid", name="Hose", categories="Garden"),
        "high": ItemMeta(item_id="high", name="Phone", categories="Electronics"),
    }
    ranks = {"low": 21, "mid": 22, "high": 100}
    focus = build_scorecard_focus(
        ["low", "mid", "high"],
        t_u="wants vitamin C serum for dry skin",
        preference_facts=None,
        item_meta=meta,
        review_snippets=None,
        stage1_ranks=ranks,
        focus_cap=2,
        phi_scores={"low": 0.1, "mid": 0.5, "high": 0.9},
    )
    assert focus == ["high", "mid"]


def test_candidate_cards_include_phi_and_co() -> None:
    text = build_candidate_cards(
        ["a"],
        {"a": 1.0},
        None,
        stage1_ranks={"a": 25},
        phi_scores={"a": 0.42},
        co_scores={"a": 0.8},
    )
    assert "π¹_rank=25" in text
    assert "φ=0.420" in text
    assert "co=0.80" in text


def test_phi_llm_abstain_keeps_phi_order() -> None:
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        json.dumps(
            {
                "must_have": ["serum"],
                "nice_to_have": [],
                "avoid": [],
                "brands": [],
                "product_types": ["serum"],
                "ingredients": [],
                "keywords": ["serum"],
                "decision_rule": "prefer serum",
            }
        ),
        json.dumps(
            {
                "displacee": {"id": "c", "fit": 4, "evidence": "already good"},
                "candidates": [
                    {
                        "id": "e",
                        "fit": 4,
                        "beats_displacee": False,
                        "evidence": "not better",
                    }
                ],
                "ranked_item_ids": [],
                "rationale": "abstain keep phi head",
            }
        ),
    ]
    pool = ["a", "b", "c", "d", "e"]
    phi_order = ["e", "d", "c", "b", "a"]
    ranks = {x: i + 1 for i, x in enumerate(pool)}
    out = llm_rerank_pool(
        llm,
        t_u="wants serum",
        reviewed_items=[],
        lookup={},
        pool=pool,
        scores={x: 1.0 for x in pool},
        numeric_fallback=phi_order,
        item_meta={x: ItemMeta(item_id=x, name=x, categories="Skincare") for x in pool},
        stage1_ranks=ranks,
        rerank_mode="promote_swap",
        promote_k=2,
        protect_n=0,
        reason_then_pick=True,
        narrow_cap=0,
        hybrid_first_enabled=False,
        scorecard_focus_cap=2,
        pick_mode="argmax_llm_override",
        phi_scores={"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.8, "e": 0.9},
        co_scores={"e": 0.5},
    )
    assert out == phi_order
    pick_prompt = llm.invoke_text.call_args_list[1].args[0]
    assert "phi-grounded" in pick_prompt.lower() or "complementary potential" in pick_prompt
    assert "φ=" in pick_prompt


def test_phi_llm_swaps_high_phi_focus_into_head() -> None:
    llm = MagicMock()
    llm.invoke_text.side_effect = [
        json.dumps(
            {
                "must_have": ["serum"],
                "nice_to_have": [],
                "avoid": [],
                "brands": [],
                "product_types": ["serum"],
                "ingredients": ["vitamin C"],
                "keywords": ["serum"],
                "decision_rule": "prefer serum",
            }
        ),
        json.dumps(
            {
                "displacee": {"id": "c", "fit": 1, "evidence": "weak phi"},
                "candidates": [
                    {
                        "id": "e",
                        "fit": 5,
                        "beats_displacee": True,
                        "evidence": "high phi serum",
                    }
                ],
                "ranked_item_ids": ["e"],
                "rationale": "promote high phi serum",
            }
        ),
    ]
    pool = ["a", "b", "c", "d", "e"]
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
        item_meta={
            "a": ItemMeta(item_id="a", name="Soap", categories="Body"),
            "b": ItemMeta(item_id="b", name="Lipstick", categories="Makeup"),
            "c": ItemMeta(item_id="c", name="Brush", categories="Tools"),
            "d": ItemMeta(item_id="d", name="Cream", categories="Skincare"),
            "e": ItemMeta(item_id="e", name="Vitamin C Serum", categories="Skincare"),
        },
        stage1_ranks=ranks,
        rerank_mode="promote_swap",
        promote_k=3,
        protect_n=0,
        reason_then_pick=True,
        narrow_cap=0,
        hybrid_first_enabled=False,
        scorecard_focus_cap=2,
        pick_mode="argmax_llm_override",
        swap_stats=stats,
        phi_scores={"a": 0.2, "b": 0.1, "c": 0.05, "d": 0.4, "e": 0.9},
        co_scores={"e": 0.7},
    )
    assert "e" in out[:3]
    assert stats.get("n_stage2_llm_override") == 1
    pick_prompt = llm.invoke_text.call_args_list[1].args[0]
    assert "φ=0.900" in pick_prompt
    assert "co=0.70" in pick_prompt


def test_build_listwise_window_injects_overlap_tail() -> None:
    meta = {
        "a": ItemMeta(item_id="a", name="Random Lipstick", categories="Makeup"),
        "b": ItemMeta(item_id="b", name="Soap", categories="Bath"),
        "c": ItemMeta(item_id="c", name="Hose", categories="Outdoor"),
        "gold": ItemMeta(
            item_id="gold", name="Vitamin C Serum", categories="Skincare Serum"
        ),
    }
    phi_order = ["a", "b", "c", "gold"]
    window = build_listwise_window(
        phi_order,
        t_u="wants vitamin C serum for dry skin",
        item_meta=meta,
        review_snippets=None,
        stage1_ranks={x: i + 1 for i, x in enumerate(phi_order)},
        window_cap=3,
        overlap_inject=1,
    )
    assert len(window) == 3
    assert window[:2] == ["a", "b"]
    assert "gold" in window
    assert "c" not in window


def test_weighted_window_phi_dominates_tu() -> None:
    from emorecagent.tisasrec_align.stage2_reason_promote import build_weighted_window

    ids = ["a", "b", "c"]
    window = build_weighted_window(
        ids,
        phi_scores={"a": 3.0, "b": 2.0, "c": 0.1},
        tu_scores={"a": 0.0, "b": 0.0, "c": 1.0},
        co_scores={"a": 0.0, "b": 0.0, "c": 0.0},
        cap=2,
        w_phi=1.0,
        w_tu=0.2,
        w_co=0.0,
    )
    assert window[0] == "a"
    assert "c" not in window


def test_blend_window_ranks_keeps_high_phi_when_llm_demotes() -> None:
    from emorecagent.tisasrec_align.stage2_reason_promote import blend_window_ranks

    window = ["a", "b", "c"]
    # LLM puts the weakest-φ item first.
    out = blend_window_ranks(
        window,
        llm_order=["c", "b", "a"],
        phi_scores={"a": 3.0, "b": 2.0, "c": 0.1},
        tu_scores={"a": 0.0, "b": 0.0, "c": 0.0},
        co_scores={"a": 0.0, "b": 0.0, "c": 0.0},
        w_phi=1.0,
        w_llm=0.25,
        w_tu=0.0,
        w_co=0.0,
    )
    assert out[0] == "a"
    assert out[-1] == "c"
