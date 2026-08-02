#!/usr/bin/env python3
"""Compare ABSA quality (macro F1) across backends for the gold set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emorecagent.absa.classical import MockClassicalAbsaTool
from emorecagent.absa.pipeline import ReviewRecord, build_absa_pipeline
from emorecagent.absa.quality import build_absa_quality_report, load_gold
from emorecagent.config import Config, load_config
from emorecagent.llm.client import LLMClient, FakeLLM
from emorecagent.llm.schemas import AbsaTriple, HybridAbsaVerdict, TripleSet


def _with_backend(cfg: Config, backend: str, cache_path: str) -> Config:
    absa = cfg.absa.model_copy(
        update={"backend": backend, "cache_path": cache_path}  # type: ignore[arg-type]
    )
    return cfg.model_copy(update={"absa": absa})


def _load_texts(targets_path: Path, review_ids: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    if not targets_path.exists():
        return texts
    with targets_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rid = str(row.get("review_id") or "")
            if rid in review_ids and rid not in texts:
                texts[rid] = (row.get("text") or "").strip()
    return texts


def _mock_client() -> LLMClient:
    verdict = HybridAbsaVerdict(
        triples=[
            AbsaTriple(
                aspect="quality",
                opinion="nice",
                sentiment="positive",
                confidence=0.9,
            )
        ],
        needs_repair=False,
    ).model_dump_json()
    empty = TripleSet(triples=[]).model_dump_json()
    return LLMClient(FakeLLM([empty, empty, verdict] * 200))


def _predict_backend(
    cfg: Config,
    backend: str,
    gold_ids: list[str],
    texts: dict[str, str],
    *,
    mock: bool,
    tmp_dir: Path,
) -> dict[str, list[AbsaTriple]]:
    cache_path = str(tmp_dir / f"absa_quality_{backend}.sqlite")
    bench_cfg = _with_backend(cfg, backend, cache_path)
    client = _mock_client() if mock else LLMClient.from_config(bench_cfg, for_absa=True)
    mock_tool = (
        MockClassicalAbsaTool(
            [
                AbsaTriple(
                    aspect="quality",
                    opinion="",
                    sentiment="positive",
                    confidence=0.95,
                )
            ]
        )
        if mock and backend == "hybrid"
        else None
    )
    pipeline = build_absa_pipeline(
        bench_cfg,
        client,
        classical_tool=mock_tool,
        skip_manifest_check=True,
    )
    predictions: dict[str, list[AbsaTriple]] = {}
    for rid in gold_ids:
        text = texts.get(rid, "")
        if not text:
            continue
        out = pipeline.process(ReviewRecord(rid, text), use_cache=False)
        predictions[rid] = out.triples
    if pipeline.cache is not None:
        pipeline.cache.close()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ABSA quality per backend.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gold", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument(
        "--backends",
        default="llm_only,hybrid",
        help="comma-separated backends",
    )
    parser.add_argument("--out", default="results/absa_quality_comparison.json")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--tmp-dir", default="/tmp")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gold_path = Path(args.gold or cfg.absa.gold_path)
    if not gold_path.exists():
        raise SystemExit(f"Gold file not found: {gold_path}")

    gold = load_gold(gold_path)
    gold_ids = list(gold.keys())
    targets = Path(args.targets or cfg.absa.targets_path)
    texts = _load_texts(targets, set(gold_ids))

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    per_backend: dict[str, dict] = {}
    for backend in backends:
        preds = _predict_backend(
            cfg,
            backend,
            gold_ids,
            texts,
            mock=args.mock,
            tmp_dir=tmp_dir,
        )
        report = build_absa_quality_report(
            preds,
            gold,
            min_support=cfg.absa.min_aspect_support,
            min_sentiment_support=cfg.absa.min_sentiment_support,
        )
        per_backend[backend] = report.to_json()

    llm_macro = per_backend.get("llm_only", {}).get("macro_review", {}).get("f1")
    hybrid_macro = per_backend.get("hybrid", {}).get("macro_review", {}).get("f1")
    delta = None
    gate_pass = None
    if llm_macro is not None and hybrid_macro is not None:
        delta = llm_macro - hybrid_macro
        gate_pass = delta <= cfg.absa.quality_gate_max_f1_drop

    payload = {
        "gold_path": str(gold_path),
        "n_gold_reviews": len(gold),
        "quality_gate_max_f1_drop": cfg.absa.quality_gate_max_f1_drop,
        "backends": per_backend,
        "macro_f1_delta_llm_minus_hybrid": delta,
        "quality_gate_pass": gate_pass,
        "recommend_default_hybrid": gate_pass is True,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[compare_absa_quality] wrote {out}")
    if delta is not None:
        print(f"  macro F1 delta (llm_only - hybrid): {delta:.4f}")
        print(f"  quality_gate_pass: {gate_pass}")


if __name__ == "__main__":
    main()
