"""Summarize ABSA cache entries for terminal / HTML / JSON reports."""

from __future__ import annotations

import html
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..llm.schemas import AbsaTriple, TripleSet


@dataclass(frozen=True, slots=True)
class AbsaSample:
    review_id: str
    text: str
    triples: list[AbsaTriple]


@dataclass
class AbsaPreviewSummary:
    n_cached: int = 0
    n_with_triples: int = 0
    n_empty: int = 0
    total_triples: int = 0
    aspect_counts: Counter[str] = field(default_factory=Counter)
    sentiment_counts: Counter[str] = field(default_factory=Counter)

    @property
    def mean_triples(self) -> float:
        if self.n_cached == 0:
            return 0.0
        return self.total_triples / self.n_cached


def load_target_texts(path: Path) -> dict[str, str]:
    """Map review_id → review text from scoped targets JSONL."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("review_id") or "")
            text = (row.get("text") or "").strip()
            if rid and text:
                out[rid] = text
    return out


def summarize_entries(
    entries: Iterable[tuple[str, TripleSet]],
) -> tuple[AbsaPreviewSummary, list[tuple[str, TripleSet]]]:
    """Aggregate cache stats; return summary and materialized entry list."""
    summary = AbsaPreviewSummary()
    materialized: list[tuple[str, TripleSet]] = []
    for review_id, triple_set in entries:
        materialized.append((review_id, triple_set))
        summary.n_cached += 1
        n = len(triple_set.triples)
        summary.total_triples += n
        if n:
            summary.n_with_triples += 1
        else:
            summary.n_empty += 1
        for triple in triple_set.triples:
            summary.aspect_counts[triple.aspect] += 1
            summary.sentiment_counts[triple.sentiment] += 1
    return summary, materialized


def pick_samples(
    entries: list[tuple[str, TripleSet]],
    texts: dict[str, str],
    *,
    n: int,
    seed: int,
    prefer_non_empty: bool = True,
    require_text: bool = False,
) -> list[AbsaSample]:
    if not entries:
        return []
    pool = entries
    if texts:
        with_text = [(rid, ts) for rid, ts in pool if rid in texts]
        if with_text:
            pool = with_text
        elif require_text:
            return []
    if prefer_non_empty:
        non_empty = [(rid, ts) for rid, ts in pool if ts.triples]
        if non_empty:
            pool = non_empty
    rng = random.Random(seed)
    chosen = rng.sample(pool, k=min(n, len(pool)))
    out: list[AbsaSample] = []
    for review_id, triple_set in chosen:
        out.append(
            AbsaSample(
                review_id=review_id,
                text=texts.get(review_id, ""),
                triples=list(triple_set.triples),
            )
        )
    return out


def format_terminal(summary: AbsaPreviewSummary, samples: list[AbsaSample]) -> str:
    lines = [
        f"cached reviews: {summary.n_cached}",
        f"with triples: {summary.n_with_triples} | empty: {summary.n_empty}",
        f"total triples: {summary.total_triples} | mean/review: {summary.mean_triples:.2f}",
    ]
    if summary.sentiment_counts:
        sent = ", ".join(
            f"{k}={v}" for k, v in summary.sentiment_counts.most_common()
        )
        lines.append(f"sentiment: {sent}")
    if summary.aspect_counts:
        top = summary.aspect_counts.most_common(8)
        aspects = ", ".join(f"{a}({c})" for a, c in top)
        lines.append(f"top aspects: {aspects}")

    lines.append("")
    lines.append(f"sample reviews ({len(samples)}):")
    for idx, sample in enumerate(samples, start=1):
        lines.append(f"--- [{idx}] review_id={sample.review_id} ---")
        if sample.text:
            preview = sample.text.replace("\n", " ")
            if len(preview) > 280:
                preview = preview[:277] + "..."
            lines.append(preview)
        else:
            lines.append("(review text not in targets file)")
        if not sample.triples:
            lines.append("  (no triples)")
            continue
        for triple in sample.triples:
            lines.append(
                f"  • {triple.aspect} | {triple.opinion} | {triple.sentiment}"
                f" (conf={triple.confidence:.2f})"
            )
    return "\n".join(lines)


def summary_to_chart_json(
    summary: AbsaPreviewSummary,
    *,
    backend: str | None = None,
    pipeline_version: str | None = None,
    cache_path: str | Path | None = None,
    run_stats: dict[str, int] | None = None,
    samples: list[AbsaSample] | None = None,
) -> dict[str, Any]:
    """Chart-friendly JSON: sentiment + aspect bar series and optional samples."""
    payload: dict[str, Any] = {
        "backend": backend,
        "pipeline_version": pipeline_version,
        "cache_path": str(cache_path) if cache_path is not None else None,
        "summary": {
            "n_cached": summary.n_cached,
            "n_with_triples": summary.n_with_triples,
            "n_empty": summary.n_empty,
            "total_triples": summary.total_triples,
            "mean_triples_per_review": round(summary.mean_triples, 4),
        },
        "sentiment": [
            {"label": label, "count": count}
            for label, count in summary.sentiment_counts.most_common()
        ],
        "aspects": [
            {"aspect": aspect, "count": count}
            for aspect, count in summary.aspect_counts.most_common(30)
        ],
    }
    if run_stats:
        payload["run"] = run_stats
    if samples:
        payload["samples"] = [
            {
                "review_id": s.review_id,
                "text": s.text,
                "triples": [t.model_dump() for t in s.triples],
            }
            for s in samples
        ]
    return payload


@dataclass(frozen=True, slots=True)
class AbsaReportPaths:
    json_path: Path
    html_path: Path | None


@dataclass(frozen=True, slots=True)
class AbsaReportResult:
    summary: AbsaPreviewSummary
    samples: list[AbsaSample]
    paths: AbsaReportPaths


def write_absa_report(
    entries: list[tuple[str, TripleSet]],
    texts: dict[str, str],
    *,
    json_path: str | Path,
    html_path: str | Path | None = None,
    samples_n: int = 8,
    seed: int = 42,
    prefer_non_empty: bool = True,
    require_text: bool = False,
    backend: str | None = None,
    pipeline_version: str | None = None,
    cache_path: str | Path | None = None,
    run_stats: dict[str, int] | None = None,
) -> AbsaReportResult:
    """Write chart JSON (and optional HTML) from cache entries."""
    summary, materialized = summarize_entries(entries)
    samples = pick_samples(
        materialized,
        texts,
        n=samples_n,
        seed=seed,
        prefer_non_empty=prefer_non_empty,
        require_text=require_text,
    )
    payload = summary_to_chart_json(
        summary,
        backend=backend,
        pipeline_version=pipeline_version,
        cache_path=cache_path,
        run_stats=run_stats,
        samples=samples,
    )
    json_out = Path(json_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    html_out: Path | None = None
    if html_path:
        html_out = Path(html_path)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        title = f"ABSA preview — {Path(cache_path).name}" if cache_path else "ABSA preview"
        html_out.write_text(
            render_html(summary, samples, title=title),
            encoding="utf-8",
        )

    return AbsaReportResult(
        summary=summary,
        samples=samples,
        paths=AbsaReportPaths(json_path=json_out, html_path=html_out),
    )


_SENTIMENT_COLOR = {
    "positive": "#1a7f37",
    "negative": "#cf222e",
    "neutral": "#57606a",
}


def render_html(
    summary: AbsaPreviewSummary,
    samples: list[AbsaSample],
    *,
    title: str = "ABSA preview",
) -> str:
    def bar_rows(counter: Counter[str], total: int) -> str:
        if not counter or total <= 0:
            return "<p>No data.</p>"
        rows: list[str] = []
        for label, count in counter.most_common(12):
            pct = 100.0 * count / total
            color = _SENTIMENT_COLOR.get(label, "#0969da")
            rows.append(
                f'<div class="bar-row"><span class="label">{html.escape(label)}</span>'
                f'<div class="bar"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
                f'<span class="count">{count}</span></div>'
            )
        return "\n".join(rows)

    sample_blocks: list[str] = []
    for sample in samples:
        triple_rows = []
        for triple in sample.triples:
            color = _SENTIMENT_COLOR.get(triple.sentiment, "#57606a")
            triple_rows.append(
                "<tr>"
                f'<td>{html.escape(triple.aspect)}</td>'
                f'<td>{html.escape(triple.opinion)}</td>'
                f'<td style="color:{color};font-weight:600">{html.escape(triple.sentiment)}</td>'
                f"<td>{triple.confidence:.2f}</td>"
                "</tr>"
            )
        if not triple_rows:
            triple_rows.append(
                '<tr><td colspan="4" class="muted">(no triples)</td></tr>'
            )
        text_block = (
            f"<p class='review'>{html.escape(sample.text)}</p>"
            if sample.text
            else "<p class='muted'>(review text not in targets file)</p>"
        )
        sample_blocks.append(
            f"<section class='card'>"
            f"<h3>review_id: {html.escape(sample.review_id)}</h3>"
            f"{text_block}"
            "<table><thead><tr><th>aspect</th><th>opinion</th><th>sentiment</th><th>conf</th></tr></thead>"
            f"<tbody>{''.join(triple_rows)}</tbody></table>"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2328; background: #f6f8fa; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
    .stat {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 1rem; }}
    .stat .value {{ font-size: 1.5rem; font-weight: 700; }}
    .stat .label {{ color: #57606a; font-size: 0.9rem; }}
    .panel {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.5rem; }}
    .bar-row {{ display: grid; grid-template-columns: 120px 1fr 48px; gap: 0.75rem; align-items: center; margin: 0.35rem 0; }}
    .bar {{ background: #eaeef2; border-radius: 4px; height: 10px; overflow: hidden; }}
    .fill {{ height: 100%; }}
    .card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem; }}
    .review {{ line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0.75rem; }}
    th, td {{ text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid #eaeef2; }}
    th {{ color: #57606a; font-size: 0.85rem; }}
    .muted {{ color: #57606a; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="stats">
    <div class="stat"><div class="value">{summary.n_cached}</div><div class="label">cached reviews</div></div>
    <div class="stat"><div class="value">{summary.n_with_triples}</div><div class="label">with triples</div></div>
    <div class="stat"><div class="value">{summary.n_empty}</div><div class="label">empty</div></div>
    <div class="stat"><div class="value">{summary.mean_triples:.2f}</div><div class="label">mean triples / review</div></div>
  </div>
  <div class="panel">
    <h2>Sentiment distribution</h2>
    {bar_rows(summary.sentiment_counts, summary.total_triples)}
  </div>
  <div class="panel">
    <h2>Top aspects</h2>
    {bar_rows(summary.aspect_counts, summary.total_triples)}
  </div>
  <h2>Sample extractions</h2>
  {''.join(sample_blocks)}
</body>
</html>
"""
