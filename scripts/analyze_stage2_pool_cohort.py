#!/usr/bin/env python3
"""P2: compare Stage-1 vs Stage-2 on the cohort where Stage-1 can still help.

Full-catalog HR/Recall mixes two populations:
  - users with ≥1 relevant already in Stage-1 top-``pool_k`` (LLM can reorder),
  - users with zero relevants in that head (LLM cannot recover them).

This script reports metric deltas on the first cohort (and the complement) using
``per_user`` arrays from ``run_experiment`` JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _means(payload: dict) -> dict[str, float]:
    if payload.get("aggregation") == "user_mean":
        return {k: float(v) for k, v in (payload.get("means_per_user") or {}).items()}
    return {k: float(v) for k, v in (payload.get("means") or {}).items()}


def _per_user_metric(payload: dict, key: str) -> dict[str, float]:
    """Map user_id → metric value from parallel user_ids / per_user lists."""
    user_ids = list(payload.get("user_ids") or [])
    per_user = payload.get("per_user") or {}
    values = per_user.get(key)
    if not user_ids or values is None:
        return {}
    if len(values) != len(user_ids):
        raise SystemExit(
            f"{key}: len(per_user)={len(values)} != len(user_ids)={len(user_ids)}"
        )
    return {str(uid): float(v) for uid, v in zip(user_ids, values)}


def _cohort_mean(vals: dict[str, float], members: set[str]) -> float | None:
    xs = [vals[u] for u in members if u in vals]
    if not xs:
        return None
    return sum(xs) / len(xs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Stage-1-only JSON")
    parser.add_argument("--fused", required=True, help="Stage-2 JSON")
    parser.add_argument(
        "--pool-k",
        type=int,
        default=40,
        help="treat hr@pool_k>0 (or nearest available hr@K) as in-pool cohort",
    )
    parser.add_argument(
        "--metrics",
        default="hr@10,ndcg@10,recall@10,hr@20,recall@20",
        help="comma-separated metrics to report",
    )
    args = parser.parse_args()

    base = _load(Path(args.baseline))
    fused = _load(Path(args.fused))
    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())

    # Prefer exact hr@{pool_k}; else largest hr@K with K<=pool_k.
    hr_keys = sorted(
        (
            int(k.split("@", 1)[1])
            for k in (base.get("per_user") or {})
            if k.startswith("hr@") and k.split("@", 1)[1].isdigit()
        ),
        reverse=True,
    )
    pick_k = None
    if args.pool_k in hr_keys:
        pick_k = args.pool_k
    else:
        for k in hr_keys:
            if k <= args.pool_k:
                pick_k = k
                break
    if pick_k is None:
        print(
            "No hr@K in baseline per_user; available keys:",
            sorted((base.get("per_user") or {}).keys()),
            file=sys.stderr,
        )
        return 1

    hr_key = f"hr@{pick_k}"
    base_hr = _per_user_metric(base, hr_key)
    if not base_hr:
        print(f"missing per_user[{hr_key!r}] in baseline", file=sys.stderr)
        return 1

    in_pool = {u for u, v in base_hr.items() if v > 0.0}
    out_pool = set(base_hr) - in_pool
    print(
        f"cohort gate: Stage-1 {hr_key} > 0 → in_pool={len(in_pool):,} "
        f"out_pool={len(out_pool):,} (pool_k request={args.pool_k})"
    )
    meta = (fused.get("metadata") or {}).get("tisasrec_align") or {}
    if meta:
        print(
            "fused metadata: "
            f"n_llm_calls={meta.get('n_llm_calls')} "
            f"n_fallback={meta.get('n_fallback')} "
            f"reject_rate={meta.get('guardrail_reject_rate')} "
            f"mean_c_u={meta.get('mean_c_u')} "
            f"n_item_meta={meta.get('n_item_meta')}"
        )

    print("\nOverall (user_mean):")
    bm, fm = _means(base), _means(fused)
    for key in metrics:
        if key not in bm and key not in fm:
            continue
        b, f = float(bm.get(key, 0.0)), float(fm.get(key, 0.0))
        print(f"  {key}: stage1={b:.4f} stage2={f:.4f} delta={f - b:+.4f}")

    for label, members in (("in_pool", in_pool), ("out_pool", out_pool)):
        print(f"\nCohort {label} (n={len(members):,}):")
        for key in metrics:
            b_map = _per_user_metric(base, key)
            f_map = _per_user_metric(fused, key)
            if not b_map or not f_map:
                print(f"  {key}: (missing per_user)")
                continue
            b = _cohort_mean(b_map, members)
            f = _cohort_mean(f_map, members)
            if b is None or f is None:
                print(f"  {key}: (empty)")
                continue
            print(f"  {key}: stage1={b:.4f} stage2={f:.4f} delta={f - b:+.4f}")

    print(
        "\nNote: LLM uplift should appear mainly on in_pool. "
        "out_pool near-zero delta is expected (relevant outside Stage-1 head)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
