#!/usr/bin/env python3
"""Benchmark ABSA latency: llm_only vs hybrid on a review subset."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from emorecagent.absa.classical import MockClassicalAbsaTool
from emorecagent.absa.pipeline import ReviewRecord, build_absa_pipeline
from emorecagent.config import Config, load_config
from emorecagent.llm.client import LLMClient, FakeLLM
from emorecagent.llm.schemas import AbsaTriple, HybridAbsaVerdict, TripleSet


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _load_targets(path: Path, n: int) -> list[ReviewRecord]:
    rows: list[ReviewRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= n:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            rid = str(row.get("review_id") or "")
            if text and rid:
                rows.append(ReviewRecord(review_id=rid, text=text))
    return rows


def _with_backend(cfg: Config, backend: str, cache_path: str) -> Config:
    absa = cfg.absa.model_copy(
        update={"backend": backend, "cache_path": cache_path}  # type: ignore[arg-type]
    )
    return cfg.model_copy(update={"absa": absa})


def _mock_client() -> LLMClient:
    empty = TripleSet(triples=[]).model_dump_json()
    verdict = HybridAbsaVerdict(
        triples=[
            AbsaTriple(
                aspect="quality",
                opinion="good",
                sentiment="positive",
                confidence=0.9,
            )
        ],
        needs_repair=False,
    ).model_dump_json()
    # llm_only: extract + judge; hybrid: validate (+ optional repair uses TripleSet)
    responses = [empty, empty, verdict, empty]
    return LLMClient(FakeLLM(responses * 50))


def _bench_backend(
    cfg: Config,
    backend: str,
    records: list[ReviewRecord],
    *,
    warmup: int,
    mock: bool,
    tmp_dir: Path,
) -> dict:
    cache_path = str(tmp_dir / f"absa_bench_{backend}.sqlite")
    bench_cfg = _with_backend(cfg, backend, cache_path)
    client = _mock_client() if mock else LLMClient.from_config(bench_cfg, for_absa=True)
    mock_tool = MockClassicalAbsaTool(
        [
            AbsaTriple(
                aspect="quality",
                opinion="",
                sentiment="positive",
                confidence=0.95,
            )
        ]
    ) if mock and backend == "hybrid" else None

    pipeline = build_absa_pipeline(
        bench_cfg,
        client,
        classical_tool=mock_tool,
        skip_manifest_check=True,
    )

    for rec in records[:warmup]:
        pipeline.process(rec, use_cache=False)

    latencies: list[float] = []
    for rec in records:
        t0 = time.perf_counter()
        pipeline.process(rec, use_cache=False)
        latencies.append(time.perf_counter() - t0)

    hybrid_stats = {}
    proc = pipeline.processor
    if hasattr(proc, "stats"):
        st = proc.stats
        hybrid_stats = {
            "repair_rate": st.repair_rate,
            "validate_only_rate": st.validate_only_rate,
            "llm_calls_mean": st.llm_calls_mean,
            "fast_path_calls": st.fast_path_calls,
        }

    if pipeline.cache is not None:
        pipeline.cache.close()

    return {
        "backend": backend,
        "n_reviews": len(latencies),
        "warmup": warmup,
        "mock": mock,
        "cache_path": cache_path,
        "latency_s": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        **hybrid_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ABSA backends.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--n-reviews", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--backends",
        default="llm_only,hybrid",
        help="comma-separated backends",
    )
    parser.add_argument("--out", default="results/absa_latency.json")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use FakeLLM + MockClassicalAbsaTool (CI / dry-run)",
    )
    parser.add_argument("--tmp-dir", default="/tmp")
    args = parser.parse_args()

    cfg = load_config(args.config)
    targets = Path(args.targets or cfg.absa.targets_path)
    if not targets.exists():
        raise SystemExit(f"Targets not found: {targets}")

    records = _load_targets(targets, args.n_reviews)
    if not records:
        raise SystemExit(f"No reviews loaded from {targets}")

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for backend in backends:
        results.append(
            _bench_backend(
                cfg,
                backend,
                records,
                warmup=min(args.warmup, len(records)),
                mock=args.mock,
                tmp_dir=tmp_dir,
            )
        )

    baseline = next((r for r in results if r["backend"] == "llm_only"), None)
    hybrid = next((r for r in results if r["backend"] == "hybrid"), None)
    speedup = None
    if baseline and hybrid and hybrid["latency_s"]["mean"] > 0:
        speedup = baseline["latency_s"]["mean"] / hybrid["latency_s"]["mean"]

    payload = {
        "n_reviews": len(records),
        "targets": str(targets),
        "backends": results,
        "speedup_ratio": speedup,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[benchmark_absa_latency] wrote {out}")
    if speedup is not None:
        print(f"  speedup_ratio (llm_only/hybrid mean): {speedup:.2f}x")


if __name__ == "__main__":
    main()
