"""Integration smoke: rerank recommender on user_batch fixture."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from emorecagent.config import load_config
from emorecagent.eval.runner import build_recommender, evaluate, load_split_jsonl
from emorecagent.tisasrec_align.rerank_recommender import RerankAlignRecommender


@pytest.fixture
def tiny_splits(tmp_path):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text(
        '{"user_id":"u1","item":"i1","timestamp":100}\n'
        '{"user_id":"u1","item":"i2","timestamp":200}\n',
        encoding="utf-8",
    )
    test_path.write_text(
        '{"user_id":"u1","item":"i3","timestamp":300}\n',
        encoding="utf-8",
    )
    return train_path, test_path


def test_build_recommender_rerank_mode(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EMOREC_DATA_OUT_DIR", str(tmp_path))
    cfg = load_config("configs/legacy/emorecagent_align.yaml")
    from emorecagent.data.types import Interaction

    train = [Interaction(user_id="u1", item="i1", rating=5.0, timestamp=100)]
    with patch(
        "emorecagent.tisasrec_align.rerank_recommender.RerankAlignRecommender.from_config"
    ) as mock_from:
        mock_from.return_value = RerankAlignRecommender.__new__(RerankAlignRecommender)
        rec = build_recommender(
            "emorecagent_align",
            {"train_interactions": train, "app_config": cfg},
            seed=0,
        )
        assert rec is mock_from.return_value
        mock_from.assert_called_once()


def test_rerank_user_batch_eval_smoke(monkeypatch, tmp_path, tiny_splits) -> None:
    """Smoke eval with mocked heavy artifacts; NO_LLM skips LLM calls."""
    train_path, test_path = tiny_splits
    monkeypatch.setenv("NO_LLM", "1")
    monkeypatch.setenv("EMOREC_DATA_OUT_DIR", str(tmp_path))

    with (
        patch(
            "emorecagent.tisasrec_align.rerank_recommender.build_stage1_recommender"
        ) as mock_stage1_cls,
        patch(
            "emorecagent.tisasrec_align.rerank_recommender.load_tu_cache",
            return_value={},
        ),
        patch(
            "emorecagent.tisasrec_align.rerank_recommender.load_lookup",
            return_value={},
        ),
        patch(
            "emorecagent.tisasrec_align.rerank_recommender.load_review_text_index",
            return_value={},
        ),
    ):
        from emorecagent.baselines.base import Recommender

        class _Stub(Recommender):
            name = "stub"

            def fit(self, interactions):
                return self

            def score(self, user_id, candidates, *, query_ts_ms=None):
                return {c: float(len(c)) for c in candidates}

            def rank(self, user_id, candidates, *, query_ts_ms=None):
                return sorted(candidates, reverse=True)

            def prepare_user_query(self, user_id, timestamp_ms):
                return None

            def catalog_items(self):
                return ["i1", "i2", "i3"]

        mock_stage1_cls.return_value = _Stub()
        cfg = load_config("configs/legacy/emorecagent_align.yaml")
        train = load_split_jsonl(train_path)
        test = load_split_jsonl(test_path)
        rec = RerankAlignRecommender.from_config(cfg, train, seed=0)
        result = evaluate(
            rec,
            train,
            test,
            k_values=[5],
            method="emorecagent_align",
            seed=0,
            eval_protocol="user_batch",
            aggregation="user_mean",
        )
        assert result.n_test_rows >= 1
        assert "hr@5" in result.means or "hr@5" in result.means_per_user
