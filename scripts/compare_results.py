#!/usr/bin/env python3
"""Compare two stored experiment JSON files with paired significance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emorecagent.eval.significance import paired_bootstrap


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired significance between runs.")
    parser.add_argument("--a", required=True, help="baseline results JSON")
    parser.add_argument("--b", required=True, help="comparison results JSON")
    parser.add_argument("--metric", required=True, help="e.g. ndcg@10")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    a = _load(Path(args.a))
    b = _load(Path(args.b))
    key = args.metric
    if key not in a.get("per_user", {}):
        raise SystemExit(f"metric '{key}' not in --a per_user keys")
    if key not in b.get("per_user", {}):
        raise SystemExit(f"metric '{key}' not in --b per_user keys")

    vec_a = a["per_user"][key]
    vec_b = b["per_user"][key]
    result = paired_bootstrap(
        vec_b,
        vec_a,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    summary = {
        "metric": key,
        "method_a": a.get("method"),
        "method_b": b.get("method"),
        "mean_a": sum(vec_a) / len(vec_a),
        "mean_b": sum(vec_b) / len(vec_b),
        "mean_delta": result.mean_delta,
        "p_value": result.p_value,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "n": result.n,
    }

    print(f"[compare_results] {key}: {a.get('method')} vs {b.get('method')}")
    print(f"  mean delta (b-a): {result.mean_delta:.4f}")
    print(f"  p-value: {result.p_value:.4f}")
    print(f"  95% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[compare_results] wrote {out}")


if __name__ == "__main__":
    main()
