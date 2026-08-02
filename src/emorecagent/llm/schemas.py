"""Pydantic schemas for structured LLM output.

Downstream agents bind these via `with_structured_output` so extraction,
judging, reasoning, and reflection all share typed contracts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TRIPLES_PER_REVIEW = 24
SENTIMENTS = frozenset({"positive", "negative", "neutral"})
_SENTIMENT_ALIASES: dict[str, str] = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AbsaTriple(_Strict):
    """One validated aspect–opinion–sentiment triple."""

    aspect: str = Field(description="Canonical aspect name, lowercase")
    opinion: str = Field(description="Opinion phrase from the review")
    sentiment: Literal["positive", "negative", "neutral"]
    # Clamp in validator; omit le=1.0 so JSON-schema parsers accept 0–100 LLM scales.
    confidence: float = Field(ge=0.0, default=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> float:
        return _coerce_confidence(value)


class TripleSet(_Strict):
    """ABSA extraction / judge output: a list of triples."""

    triples: list[AbsaTriple] = Field(default_factory=list)


class HybridAbsaVerdict(_Strict):
    """Hybrid agent validate-step output."""

    triples: list[AbsaTriple] = Field(default_factory=list)
    needs_repair: bool = False
    missing_aspect_hints: list[str] = Field(default_factory=list)


def _normalize_sentiment(raw: object) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return _SENTIMENT_ALIASES.get(key)


def _coerce_confidence(raw: object) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if val > 1.0:
        val /= 100.0
    return max(0.0, min(1.0, val))


def coerce_triple_set(data: TripleSet | dict[str, Any] | str) -> TripleSet:
    """Best-effort parse: drop invalid rows, dedupe, cap list length."""
    if isinstance(data, TripleSet):
        raw_triples = [t.model_dump() for t in data.triples]
    elif isinstance(data, str):
        raw_triples = json.loads(data).get("triples", [])
    else:
        raw_triples = data.get("triples", [])

    seen: set[tuple[str, str, str]] = set()
    cleaned: list[AbsaTriple] = []
    for row in raw_triples:
        if not isinstance(row, dict):
            continue
        aspect = str(row.get("aspect") or "").strip().lower()
        opinion = str(row.get("opinion") or "").strip()
        sentiment = _normalize_sentiment(row.get("sentiment"))
        if not aspect or not opinion or sentiment is None:
            continue
        if sentiment not in SENTIMENTS:
            continue
        key = (aspect, opinion, sentiment)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            AbsaTriple(
                aspect=aspect,
                opinion=opinion,
                sentiment=sentiment,  # type: ignore[arg-type]
                confidence=_coerce_confidence(row.get("confidence", 1.0)),
            )
        )
        if len(cleaned) >= MAX_TRIPLES_PER_REVIEW:
            break
    return TripleSet(triples=cleaned)


def coerce_hybrid_verdict(
    data: HybridAbsaVerdict | dict[str, Any] | str,
) -> HybridAbsaVerdict:
    """Parse hybrid validate output; normalize triple confidences."""
    if isinstance(data, HybridAbsaVerdict):
        raw = data.model_dump()
    elif isinstance(data, str):
        raw = json.loads(data)
    else:
        raw = data
    triples = coerce_triple_set({"triples": raw.get("triples", [])})
    hints = raw.get("missing_aspect_hints") or []
    return HybridAbsaVerdict(
        triples=triples.triples,
        needs_repair=bool(raw.get("needs_repair", False)),
        missing_aspect_hints=[str(h) for h in hints if str(h).strip()],
    )


def extract_triples_from_partial_json(text: str) -> list[dict[str, Any]]:
    """Recover complete triple objects from truncated / pretty-printed JSON."""
    triples: list[dict[str, Any]] = []
    for block in re.finditer(r"\{[^{}]+\}", text, re.DOTALL):
        chunk = block.group(0)
        if '"aspect"' not in chunk or '"sentiment"' not in chunk:
            continue
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("aspect") and obj.get("sentiment"):
            triples.append(obj)
    return triples


def _structured_text_from_exc(exc: BaseException) -> str:
    """Pull raw LLM JSON out of parser / pydantic errors when present."""
    llm_output = getattr(exc, "llm_output", None)
    if isinstance(llm_output, str) and llm_output.strip():
        return llm_output
    if isinstance(llm_output, dict):
        return json.dumps(llm_output)

    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            for err in errors():
                inp = err.get("input") if isinstance(err, dict) else None
                if isinstance(inp, str) and "{" in inp:
                    return inp
        except Exception:  # noqa: BLE001 — best-effort salvage only
            pass

    text = str(exc)
    # pydantic: input_value='{...truncated...', input_type=str
    match = re.search(r"input_value='((?:\\'|[^'])*)'", text, re.DOTALL)
    if match:
        try:
            return match.group(1).encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return match.group(1)
    match = re.search(r'input_value="((?:\\"|[^"])*)"', text, re.DOTALL)
    if match:
        try:
            return match.group(1).encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return match.group(1)
    return text


def salvage_triple_set_from_error(exc: BaseException) -> TripleSet | None:
    """Recover a partial TripleSet from a structured-output parse failure."""
    text = _structured_text_from_exc(exc)
    try:
        if text.lstrip().startswith(("{", "[")):
            return coerce_triple_set(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    match = re.search(r'\{\s*"triples"\s*:', text)
    if match:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[match.start() :])
            return coerce_triple_set(obj)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    partial = extract_triples_from_partial_json(text)
    if partial:
        return coerce_triple_set({"triples": partial})
    return None


def salvage_hybrid_verdict_from_error(exc: BaseException) -> HybridAbsaVerdict | None:
    """Recover HybridAbsaVerdict when structured parsing fails (e.g. bad confidence)."""
    text = _structured_text_from_exc(exc)
    try:
        if text.lstrip().startswith("{"):
            return coerce_hybrid_verdict(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    match = re.search(r'\{\s*"triples"\s*:', text)
    if match:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[match.start() :])
            return coerce_hybrid_verdict(obj)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    partial = extract_triples_from_partial_json(text)
    if not partial:
        return None
    needs_repair = bool(re.search(r'"needs_repair"\s*:\s*true', text, re.I))
    return coerce_hybrid_verdict(
        {"triples": partial, "needs_repair": needs_repair, "missing_aspect_hints": []}
    )

class ReflectionVerdict(_Strict):
    """Reflection Agent structured verdict."""

    approved: bool
    critique: str = ""
    violated_constraints: list[str] = Field(default_factory=list)


class BatchReasoningRow(_Strict):
    row_id: str
    ranked_item_ids: list[str] = Field(default_factory=list)


class BatchReasoningRankingVerdict(_Strict):
    """Listwise rerank output for multiple eval rows in one LLM call."""

    rows: list[BatchReasoningRow] = Field(default_factory=list)


class ReasoningRankingVerdict(_Strict):
    """Listwise rerank output for the reasoning agent."""

    ranked_item_ids: list[str] = Field(default_factory=list)


def coerce_ranking_verdict(
    data: ReasoningRankingVerdict | dict[str, Any] | str,
    *,
    pool_ids: list[str],
) -> list[str]:
    """Drop unknown IDs, dedupe, append missing pool items in numeric pool order."""
    if isinstance(data, ReasoningRankingVerdict):
        ranked = [str(x).strip() for x in data.ranked_item_ids]
    elif isinstance(data, str):
        ranked = [str(x).strip() for x in json.loads(data).get("ranked_item_ids", [])]
    else:
        ranked = [str(x).strip() for x in data.get("ranked_item_ids", [])]

    pool_set = set(pool_ids)
    seen: set[str] = set()
    out: list[str] = []
    for item in ranked:
        if not item or item not in pool_set or item in seen:
            continue
        seen.add(item)
        out.append(item)
    for item in pool_ids:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def coerce_batch_ranking_verdict(
    data: BatchReasoningRankingVerdict | dict[str, Any],
    *,
    pools_by_row: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Parse batched LLM output; coerce each row against its candidate pool."""
    if isinstance(data, BatchReasoningRankingVerdict):
        rows = data.rows
    else:
        rows = [
            BatchReasoningRow(**row)
            for row in data.get("rows", [])
            if isinstance(row, dict)
        ]
    out: dict[str, list[str]] = {}
    for row in rows:
        pool = pools_by_row.get(row.row_id)
        if pool is None:
            continue
        out[row.row_id] = coerce_ranking_verdict(
            {"ranked_item_ids": row.ranked_item_ids},
            pool_ids=pool,
        )
    return out


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_ranked_ids_from_partial_json(text: str) -> list[str]:
    """Best-effort recovery of item ids from truncated ranking JSON."""
    text = _strip_json_fences(text)
    anchor = text.find('"ranked_item_ids"')
    chunk = text[anchor:] if anchor >= 0 else text
    ids: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r'"([^"\\]{2,32})"', chunk):
        item = match.group(1).strip()
        if item in {"ranked_item_ids", "rows", "row_id"}:
            continue
        if item in seen:
            continue
        seen.add(item)
        ids.append(item)
    return ids


def extract_batch_rows_from_partial_json(text: str) -> list[dict[str, Any]]:
    """Recover row_id + ranked_item_ids pairs from truncated batch JSON."""
    text = _strip_json_fences(text)
    rows: list[dict[str, Any]] = []
    for block in re.finditer(
        r'"row_id"\s*:\s*"([^"]+)"\s*,\s*"ranked_item_ids"\s*:\s*\[([^\]]*)',
        text,
        re.DOTALL,
    ):
        row_id = block.group(1)
        ids = re.findall(r'"([^"\\]{2,32})"', block.group(2))
        if row_id and ids:
            rows.append({"row_id": row_id, "ranked_item_ids": ids})
    return rows


def parse_ranking_json(raw: str, *, pool_ids: list[str]) -> list[str]:
    """Parse ranking JSON; salvage partial output against pool_ids when truncated."""
    text = _strip_json_fences(raw)
    try:
        data = json.loads(text)
        return coerce_ranking_verdict(data, pool_ids=pool_ids)
    except json.JSONDecodeError:
        ids = extract_ranked_ids_from_partial_json(text)
        if not ids:
            raise
        return coerce_ranking_verdict({"ranked_item_ids": ids}, pool_ids=pool_ids)


def parse_batch_ranking_json(
    raw: str, *, pools_by_row: dict[str, list[str]]
) -> dict[str, list[str]]:
    text = _strip_json_fences(raw)
    try:
        data = json.loads(text)
        return coerce_batch_ranking_verdict(data, pools_by_row=pools_by_row)
    except json.JSONDecodeError:
        rows = extract_batch_rows_from_partial_json(text)
        if not rows:
            raise
        return coerce_batch_ranking_verdict({"rows": rows}, pools_by_row=pools_by_row)


def ranking_max_tokens(
    pool_sizes: list[int],
    *,
    cap: int = 4096,
    per_id: int = 18,
) -> int:
    """Completion budget for ranking JSON (ASIN list can be long)."""
    total_ids = sum(max(n, 1) for n in pool_sizes)
    return min(max(total_ids * per_id + 256, 2048), cap)


def salvage_ranking_verdict_from_error(
    exc: BaseException, *, pool_ids: list[str]
) -> list[str] | None:
    text = str(exc)
    ids = extract_ranked_ids_from_partial_json(text)
    if not ids:
        return None
    return coerce_ranking_verdict({"ranked_item_ids": ids}, pool_ids=pool_ids)


class ExplanationClaims(_Strict):
    """Rationalized explanation claims for faithfulness scoring."""

    summary: str
    cited_aspects: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
