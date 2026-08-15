#!/usr/bin/env python3
"""Shim → ``scripts/train_recbole_stage1_option_b.py`` for Yelp_AC Option B.

Unified artifact root: ``data/processed/Yelp_AC/tisasrec_option_b/`` via
``configs/categories/Yelp_AC.yaml``. Legacy ``tisasrec_paper/`` paths live under
``configs/legacy/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_recbole_stage1_option_b import main  # noqa: E402

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", "configs/categories/Yelp_AC.yaml"]
    raise SystemExit(main())
