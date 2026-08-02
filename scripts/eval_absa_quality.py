#!/usr/bin/env python3
"""Evaluate ABSA quality on gold labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emorecagent.absa.cache import AbsaCache
from emorecagent.absa.quality import (
    build_absa_quality_report,
    jaccard_keys,
    load_gold,
)
from emorecagent.config import load_config
from emorecagent.eval.bootstrap import bootstrap_ci


def _load_predictions(cache: AbsaCache, review_ids: list[str]) -> dict:
    from emorecagent.llm.schemas import AbsaTriple

    out: dict[str, list[AbsaTriple]] = {}
    for rid in review_ids:
        hit = cache.get(rid)
        if hit is not None:
            out[rid] = hit.triples
    return out


def _dual_annotation_qa(paths: list[Path]) -> dict:
    if len(paths) < 2:
        return {}
    gold_a = load_gold(paths[0])
    gold_b = load_gold(paths[1])
    common = sorted(set(gold_a) & set(gold_b))
    if not common:
        return {"dual_annotation_n": 0, "mean_jaccard": 0.0}
    scores = [jaccard_keys(gold_a[rid], gold_b[rid]) for rid in common]
    return {
        "dual_annotation_n": len(common),
        "mean_jaccard": sum(scores) / len(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ABSA quality evaluation on gold set.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gold", default=None)
    parser.add_argument("--out", default="results/absa_quality.json")
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument(
        "--dual-annotation",
        nargs=2,
        metavar=("V1", "V2"),
        default=None,
        help="two gold JSONL files for Jaccard QA",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    gold_path = Path(args.gold or cfg.absa.gold_path)
    if not gold_path.exists():
        raise SystemExit(f"Gold file not found: {gold_path}")

    gold = load_gold(gold_path)
    cache = AbsaCache(cfg.absa.cache_path)
    predictions = _load_predictions(cache, list(gold))
    cache.close()

    labeling_qa = {}
    if args.dual_annotation:
        labeling_qa = _dual_annotation_qa([Path(p) for p in args.dual_annotation])

    report = build_absa_quality_report(
        predictions,
        gold,
        min_support=cfg.absa.min_aspect_support,
        min_sentiment_support=cfg.absa.min_sentiment_support,
        labeling_qa=labeling_qa,
    )

    payload = report.to_json()
    if args.bootstrap:
        scored_f1s: list[float] = []
        scored_ids = [
            rid
            for rid in gold
            if rid in predictions and (gold[rid] or predictions[rid])
        ]
        from emorecagent.absa.quality import triple_f1

        for rid in scored_ids:
            scored_f1s.append(triple_f1(predictions[rid], gold[rid]).f1)
        ci = bootstrap_ci(
            scored_f1s,
            n_bootstrap=args.bootstrap,
            seed=cfg.experiment.seed,
        )
        payload["macro_review"]["ci_low"] = ci.low
        payload["macro_review"]["ci_high"] = ci.high
        payload["macro_review"]["n_bootstrap"] = ci.n_bootstrap

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[eval_absa_quality] gold={len(gold)} scored={report.n_scored_reviews}")
    print(f"  coverage: {report.coverage:.3f}")
    print(f"  macro_review F1: {report.macro_review.f1:.4f}")
    print(f"  micro F1: {report.micro.f1:.4f}")
    print(f"[eval_absa_quality] wrote {out}")


if __name__ == "__main__":
    main()
