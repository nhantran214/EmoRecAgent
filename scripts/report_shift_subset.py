#!/usr/bin/env python3
"""Report ranking metrics on shift-subpopulation vs complement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emorecagent.eval.runner import aggregate_per_user
from emorecagent.eval.shift_eval import select_shift_users
from emorecagent.scoring.dynamic_weights import AspectSignal


def _load_signals(path: Path) -> dict[str, list[AspectSignal]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[AspectSignal]] = {}
    for user_id, signals in raw.items():
        out[user_id] = [
            AspectSignal(
                aspect=s["aspect"],
                polarity=float(s["polarity"]),
                timestamp_ms=int(s["timestamp_ms"]),
            )
            for s in signals
        ]
    return out


def _slice_means(
    per_user: dict[str, list[float]],
    user_ids: list[str],
    mask: list[bool],
    metric: str,
) -> tuple[float, int]:
    subset_ids = [uid for uid, keep in zip(user_ids, mask) if keep]
    if not subset_ids:
        return 0.0, 0
    sliced: dict[str, list[float]] = {metric: []}
    for idx, keep in enumerate(mask):
        if keep:
            sliced[metric].append(per_user[metric][idx])
    means = aggregate_per_user(sliced, subset_ids)
    return means.get(metric, 0.0), len(set(subset_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Shift-subpopulation report.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--signals", required=True)
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--out", default="results/shift_subset.json")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    user_ids = results.get("user_ids")
    if not user_ids:
        raise SystemExit(
            "results JSON missing user_ids; re-run experiment with updated runner"
        )
    per_user = results["per_user"]
    metric = args.metric
    if metric not in per_user:
        raise SystemExit(f"metric '{metric}' not in results")

    signals = _load_signals(Path(args.signals))
    shift_users = set(select_shift_users(signals))
    mask = [uid in shift_users for uid in user_ids]

    shift_mean, shift_n = _slice_means(per_user, user_ids, mask, metric)
    comp_mean, comp_n = _slice_means(
        per_user, user_ids, [not m for m in mask], metric
    )
    all_mean = results.get("means_per_user", {}).get(
        metric, results.get("means", {}).get(metric, 0.0)
    )

    payload = {
        "metric": metric,
        "shift_users": {"n_users": shift_n, "mean": shift_mean},
        "complement": {"n_users": comp_n, "mean": comp_mean},
        "all_users": {"n_users": results.get("n_test_users"), "mean": all_mean},
        "delta_shift_vs_all": shift_mean - all_mean,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[report_shift_subset] {metric} shift={shift_mean:.4f} (n={shift_n})")
    print(f"[report_shift_subset] wrote {out}")


if __name__ == "__main__":
    main()
