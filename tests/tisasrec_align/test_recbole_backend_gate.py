"""Yelp_AC-only gate for RecBole Stage-1 backend."""

from __future__ import annotations

import pytest

from emorecagent.config import load_config
from emorecagent.tisasrec_align.stage1_factory import build_stage1_recommender


def test_recbole_backend_rejected_for_non_yelp_ac() -> None:
    cfg = load_config("configs/default.yaml")
    ta = cfg.tisasrec_align.model_copy(update={"stage1_backend": "recbole"})
    cfg = cfg.model_copy(update={"tisasrec_align": ta})
    with pytest.raises(ValueError, match="Yelp_AC"):
        build_stage1_recommender(cfg, train=[])


def test_paper_config_is_recbole_backend() -> None:
    paper = load_config("configs/categories/Yelp_AC_tisasrec_paper.yaml")
    assert paper.data.category == "Yelp_AC"
    assert paper.tisasrec_align.stage1_backend == "recbole"
