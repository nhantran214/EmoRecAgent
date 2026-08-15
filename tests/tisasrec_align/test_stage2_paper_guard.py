"""Unit tests for paper §III.F context-dependent guardrail helpers."""

from __future__ import annotations

import numpy as np

from emorecagent.tisasrec_align.stage2_paper_guard import (
    compute_alignment_confidence,
    context_dependent_window,
    fuse_user_vector,
    fused_pool_scores,
)
from emorecagent.tisasrec_align.stage2_rerank import check_guardrail


def test_context_dependent_window_high_confidence_widens_m() -> None:
    n_lo, m_lo = context_dependent_window(0.0)
    n_hi, m_hi = context_dependent_window(1.0)
    # Low c_u → larger N_u (stricter head), smaller M_u (tighter window).
    assert n_lo >= n_hi
    assert m_hi >= m_lo
    assert 3 <= n_lo <= 8 and 3 <= n_hi <= 8
    assert 8 <= m_lo <= 15 and 8 <= m_hi <= 15


def test_fuse_user_vector_alpha_one_ignores_p() -> None:
    s = np.array([1.0, 0.0])
    p = np.array([0.0, 1.0])
    x = fuse_user_vector(s, p, alpha=1.0)
    np.testing.assert_allclose(x, s)


def test_fused_pool_scores_dot_product() -> None:
    x = np.array([1.0, 0.0])
    scores = fused_pool_scores(x, {"a": np.array([2.0, 3.0]), "b": np.array([0.0, 1.0])})
    assert scores["a"] == 2.0
    assert scores["b"] == 0.0


def test_alignment_confidence_without_p_uses_margin_only() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    scores = {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0}
    c = compute_alignment_confidence(
        s_u=None,
        p_u=None,
        stage1_ranked=ranked,
        stage1_scores=scores,
        n0=5,
        omega=0.7,
    )
    assert 0.0 <= c <= 1.0


def test_should_invoke_llm_gate() -> None:
    from emorecagent.tisasrec_align.stage2_paper_guard import should_invoke_llm

    ok, reason = should_invoke_llm(
        c_u=0.6, stage1_margin_conf=0.5, enabled=True, min_c_u=0.45, max_stage1_margin=0.85
    )
    assert ok and reason == "ok"
    skip_c, r_c = should_invoke_llm(
        c_u=0.2, stage1_margin_conf=0.5, enabled=True, min_c_u=0.45, max_stage1_margin=0.85
    )
    assert not skip_c and r_c == "low_c_u"
    skip_m, r_m = should_invoke_llm(
        c_u=0.6, stage1_margin_conf=0.95, enabled=True, min_c_u=0.45, max_stage1_margin=0.85
    )
    assert not skip_m and r_m == "high_stage1_margin"
    off, r_off = should_invoke_llm(
        c_u=0.0, stage1_margin_conf=1.0, enabled=False
    )
    assert off and r_off == "gate_off"


def test_guardrail_eq21_rejects_head_drop() -> None:
    stage1 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    # Head item "a" falls to rank 11 (> M=10).
    merged = ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "a"]
    assert check_guardrail(stage1, merged, top_n=5, max_drop_rank=10) is False
