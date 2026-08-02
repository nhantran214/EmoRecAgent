"""Tests for auto steps/epoch resolution."""

from __future__ import annotations

from emorecagent.tisasrec_align.stage1_test_eval import resolve_steps_per_epoch


def test_auto_steps_per_epoch():
    assert resolve_steps_per_epoch(10, 4, None) == 2


def test_explicit_steps_override():
    assert resolve_steps_per_epoch(10, 4, 5) == 5
