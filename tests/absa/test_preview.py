"""ABSA cache preview helpers."""

from __future__ import annotations

from emorecagent.absa.preview import (
    format_terminal,
    pick_samples,
    render_html,
    summarize_entries,
    summary_to_chart_json,
    write_absa_report,
)
from emorecagent.llm.schemas import AbsaTriple, TripleSet


def test_summarize_and_render_preview() -> None:
    entries = [
        (
            "r1",
            TripleSet(
                triples=[
                    AbsaTriple(
                        aspect="scent",
                        opinion="lovely",
                        sentiment="positive",
                    )
                ]
            ),
        ),
        ("r2", TripleSet(triples=[])),
    ]
    summary, materialized = summarize_entries(entries)
    assert summary.n_cached == 2
    assert summary.n_with_triples == 1
    assert summary.n_empty == 1
    assert summary.total_triples == 1

    samples = pick_samples(
        materialized,
        {"r1": "Smells great"},
        n=1,
        seed=0,
    )
    text = format_terminal(summary, samples)
    assert "scent" in text
    assert "Smells great" in text

    html = render_html(summary, samples)
    assert "<html" in html
    assert "scent" in html

    chart = summary_to_chart_json(summary, backend="llm_only", samples=samples)
    assert chart["sentiment"][0]["label"] == "positive"
    assert chart["aspects"][0]["aspect"] == "scent"


def test_write_absa_report(tmp_path) -> None:
    entries = [
        (
            "r1",
            TripleSet(
                triples=[
                    AbsaTriple(
                        aspect="scent",
                        opinion="lovely",
                        sentiment="positive",
                    )
                ]
            ),
        ),
    ]
    report = write_absa_report(
        entries,
        {"r1": "Smells great"},
        json_path=tmp_path / "summary.json",
        html_path=tmp_path / "preview.html",
    )
    assert report.paths.json_path.exists()
    assert report.paths.html_path is not None
    assert report.paths.html_path.exists()
