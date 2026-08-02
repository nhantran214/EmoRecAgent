"""Tests for full vs subsampled valid eval during Stage 1 training."""

from __future__ import annotations

from emorecagent.tisasrec_align.stage1_test_eval import resolve_valid_eval_max_pairs


def test_valid_eval_all_uses_all_cases():
    assert (
        resolve_valid_eval_max_pairs(
            valid_eval_all=True,
            valid_eval_max_pairs=2048,
            n_valid_cases=100,
        )
        == 100
    )


def test_valid_eval_subsample_cap():
    assert (
        resolve_valid_eval_max_pairs(
            valid_eval_all=False,
            valid_eval_max_pairs=2048,
            n_valid_cases=16058,
        )
        == 2048
    )
