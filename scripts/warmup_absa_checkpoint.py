#!/usr/bin/env python3
"""Prefetch PyABSA checkpoint (optional ML install)."""

from __future__ import annotations

import argparse
import sys

from emorecagent.absa.classical import PyAbsaClassicalTool, require_absa_ml
from emorecagent.config import ConfigError, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm up PyABSA classical ABSA model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    try:
        require_absa_ml()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    tool = PyAbsaClassicalTool.from_config(cfg.absa)
    print(
        f"[warmup_absa_checkpoint] loaded checkpoint={cfg.absa.classical_checkpoint} "
        f"warmup_seconds={tool.warmup_seconds:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
