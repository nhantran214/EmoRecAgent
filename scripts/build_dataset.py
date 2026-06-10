#!/usr/bin/env python3
"""Build the processed dataset: stream -> dedup -> k-core -> sample -> split.

Usage:
  PYTHONPATH=src python scripts/build_dataset.py --config configs/default.yaml
  PYTHONPATH=src python scripts/build_dataset.py --config configs/default.yaml --max-scan 200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emorecagent.config import load_config  # noqa: E402
from emorecagent.data.loader import dedup_earliest, load_interactions  # noqa: E402
from emorecagent.data.split import (  # noqa: E402
    leave_last_out,
    sample_subset,
    write_split,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Cap raw lines read from the review file (smoke testing).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg.data

    print(f"Streaming reviews from {d.review_path} (max_scan={args.max_scan}) ...")
    raw = load_interactions(d.review_path, max_scan=args.max_scan)
    print(f"  raw interactions: {len(raw):,}")

    deduped = dedup_earliest(raw)
    print(f"  after dedup (earliest per user,item): {len(deduped):,}")

    sampled = sample_subset(
        deduped,
        k_core=d.k_core,
        max_users=d.max_users,
        max_items=d.max_items,
        seed=cfg.experiment.seed,
    )
    print(f"  after {d.k_core}-core + sampling: {len(sampled):,}")

    split = leave_last_out(sampled, min_history=d.min_history)
    manifest_path = write_split(d.out_dir, split, cfg.experiment.seed, d.k_core)

    print("  split:")
    for key, val in split.manifest.items():
        print(f"    {key}: {val}")
    print(f"Wrote split + manifest to {manifest_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
