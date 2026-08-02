#!/usr/bin/env python3
"""Build cross-user co-purchase lookup from train split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emorecagent.config import load_config
from emorecagent.eval.runner import load_split_jsonl
from emorecagent.tisasrec_align.cross_user_lookup import (
    build_lookup_from_config,
    save_lookup,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ta = cfg.tisasrec_align
    train_path = Path(cfg.data.out_dir) / "train.jsonl"
    if not train_path.is_file():
        print(f"Missing train split: {train_path}", file=sys.stderr)
        return 1

    train = load_split_jsonl(train_path)
    lookup = build_lookup_from_config(
        train,
        cfg.data.review_path,
        mode=ta.cross_user_mode,
    )
    out_path = Path(ta.cross_user_lookup_path)
    save_lookup(out_path, lookup)
    n_anchors = len(lookup)
    n_edges = sum(len(v) for v in lookup.values())
    print(f"Wrote cross-user lookup: {out_path.resolve()}")
    print(f"  mode={ta.cross_user_mode} anchors={n_anchors:,} co-purchase edges={n_edges:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
