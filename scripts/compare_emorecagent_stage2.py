#!/usr/bin/env python3
"""Compare Stage 1-only vs fused Stage 2 emorecagent_align results (no-regression gate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_METRICS = ("hr@10", "ndcg@10", "recall@10")
DEFAULT_TOLERANCE = 0.005


def _load_means(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregation = payload.get("aggregation", "row_mean")
    if aggregation == "user_mean":
        return dict(payload.get("means_per_user") or {})
    return dict(payload.get("means") or {})


def compare_results(
    baseline: dict[str, float],
    fused: dict[str, float],
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[bool, list[str]]:
    ok = True
    lines: list[str] = []
    for key in metrics:
        b = float(baseline.get(key, 0.0))
        f = float(fused.get(key, 0.0))
        delta = f - b
        passed = f + tolerance >= b
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        lines.append(
            f"  {key}: baseline={b:.4f} fused={f:.4f} delta={delta:+.4f} [{status}]"
        )
    return ok, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="results/emorecagent_stage1_baseline.json",
        help="Stage 1-only experiment JSON (user_batch)",
    )
    parser.add_argument(
        "--fused",
        default="results/emorecagent_align.json",
        help="Fused Stage 2 experiment JSON (user_batch)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="allowed drop per metric (default: 0.005)",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    fused_path = Path(args.fused)
    if not baseline_path.is_file():
        print(f"Missing baseline results: {baseline_path}", file=sys.stderr)
        return 1
    if not fused_path.is_file():
        print(f"Missing fused results: {fused_path}", file=sys.stderr)
        return 1

    baseline = _load_means(baseline_path)
    fused = _load_means(fused_path)
    ok, lines = compare_results(
        baseline, fused, tolerance=args.tolerance
    )

    print("EmoRecAgent Stage 2 no-regression check")
    print(f"  baseline: {baseline_path}")
    print(f"  fused:    {fused_path}")
    print(f"  tolerance: {args.tolerance}")
    for line in lines:
        print(line)

    if ok:
        print("Overall: PASS (fused does not regress below baseline)")
        return 0
    print("Overall: FAIL (fused regresses below baseline on one or more metrics)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
