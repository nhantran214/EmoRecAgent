"""Graph stage timing accumulator tests."""

from __future__ import annotations

from emorecagent.graph.timing import GraphStageTimer


def test_stage_timer_summary_includes_stages() -> None:
    timer = GraphStageTimer()
    timer.record_graph_invoke()
    timer.record("absa", 0.01)
    timer.record("profiling", 0.02)
    timer.record("reasoning", 1.5)
    timer.record("reflection", 0.3)
    lines = timer.summary_lines()
    assert len(lines) == 2
    assert "reasoning" in lines[0]
    assert "invokes=1" in lines[0]
    assert "total wall=" in lines[1]
