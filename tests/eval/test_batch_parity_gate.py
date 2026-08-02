"""Parity gate helper tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "compare_batch_parity",
    Path(__file__).resolve().parents[2] / "scripts" / "compare_batch_parity.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def test_parity_gate_passes_identical_metrics() -> None:
    base = {"hr@1": 0.10, "hr@3": 0.20, "hr@5": 0.30}
    assert _mod._check_parity(base, dict(base)) == []


def test_parity_gate_fails_large_drift() -> None:
    base = {"hr@1": 0.10, "hr@3": 0.20, "hr@5": 0.30}
    batch = {"hr@1": 0.12, "hr@3": 0.20, "hr@5": 0.30}
    errors = _mod._check_parity(base, batch)
    assert any("hr@1" in e for e in errors)
