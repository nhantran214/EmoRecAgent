#!/usr/bin/env python3
"""E11: Stage-2 oracle / upper-bound from Stage-1 per-user hr@K (no re-rank dump).

If gold is already in Stage-1 top-K, a perfect LLM that always promotes it into
top-10 can reach hr@10 = 1 for that user. This script reports that ceiling and
the near-miss cohorts that A1 promote-only can target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Stage-1-only JSON")
    parser.add_argument(
        "--fused",
        default="",
        help="optional Stage-2 JSON (reports helped/hurt flips)",
    )
    args = parser.parse_args()

    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    pu = base.get("per_user") or {}
    n = len(base.get("user_ids") or pu.get("hr@10") or [])
    h10 = [float(x) for x in pu["hr@10"]]
    h20 = [float(x) for x in pu["hr@20"]]
    h50 = [float(x) for x in pu["hr@50"]]
    h100 = [float(x) for x in pu["hr@100"]]

    print(f"n_users={n}")
    print(
        f"Stage-1: hr@10={_mean(h10):.4f} hr@20={_mean(h20):.4f} "
        f"hr@50={_mean(h50):.4f} hr@100={_mean(h100):.4f}"
    )
    print(
        "\nOracle upper bound (force hr@10=1 iff Stage-1 hit@K; "
        "proxy for gold ∈ π¹[:K] ⊆ C when C≥K):"
    )
    for label, gate in (
        ("C≈20 (hr@20)", h20),
        ("C≈50 (hr@50)", h50),
        ("K=100 (hr@100)", h100),
    ):
        oracle = [1.0 if g > 0 else 0.0 for g in gate]
        print(
            f"  {label}: oracle_hr@10={_mean(oracle):.4f} "
            f"room_vs_S1={_mean(oracle) - _mean(h10):+.4f}"
        )

    already = sum(1 for x in h10 if x > 0)
    nm_11_20 = sum(1 for a, b in zip(h10, h20) if a == 0 and b > 0)
    nm_21_50 = sum(1 for a, b in zip(h20, h50) if a == 0 and b > 0)
    nm_51_100 = sum(1 for a, b in zip(h50, h100) if a == 0 and b > 0)
    out100 = sum(1 for x in h100 if x == 0)
    print("\nCohorts (Stage-1):")
    print(f"  already hr@10:     {already:5d} ({100 * already / n:.1f}%) — LLM can only hurt HR")
    print(f"  near-miss 11–20:   {nm_11_20:5d} ({100 * nm_11_20 / n:.1f}%) — promote into top-10")
    print(f"  near-miss 21–50:   {nm_21_50:5d} ({100 * nm_21_50 / n:.1f}%) — need gold ∈ C (≈40)")
    print(f"  near-miss 51–100:  {nm_51_100:5d} ({100 * nm_51_100 / n:.1f}%) — needs fusion→C")
    print(f"  outside top-100:   {out100:5d} ({100 * out100 / n:.1f}%) — Stage-2 cannot help")

    if args.fused:
        fused = json.loads(Path(args.fused).read_text(encoding="utf-8"))
        f10 = [float(x) for x in (fused.get("per_user") or {})["hr@10"]]
        helped = sum(1 for a, b in zip(h10, f10) if a == 0 and b > 0)
        hurt = sum(1 for a, b in zip(h10, f10) if a > 0 and b == 0)
        print(
            f"\nStage-2 flips vs Stage-1: helped={helped} hurt={hurt} "
            f"net={helped - hurt:+d}  hr@10={_mean(f10):.4f} "
            f"(Δ={_mean(f10) - _mean(h10):+.4f})"
        )
        meta = (fused.get("metadata") or {}).get("tisasrec_align") or {}
        if meta:
            print(
                f"  reject_rate={meta.get('guardrail_reject_rate')} "
                f"n_llm_calls={meta.get('n_llm_calls')} "
                f"llm_pool_cap={meta.get('llm_pool_cap')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
