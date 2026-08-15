#!/usr/bin/env python3
"""Shim → ``scripts/train_recbole_stage1_option_b.py`` (Yelp Option B defaults)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_recbole_stage1_option_b import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
