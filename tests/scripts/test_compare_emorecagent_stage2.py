"""No-regression compare script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_emorecagent_stage2 import compare_results


def test_compare_passes_when_fused_matches_baseline() -> None:
    metrics = {"hr@10": 0.05, "ndcg@10": 0.02, "recall@10": 0.05}
    ok, lines = compare_results(metrics, metrics, tolerance=0.005)
    assert ok
    assert any("PASS" in line for line in lines)


def test_compare_fails_on_regression() -> None:
    baseline = {"hr@10": 0.05, "ndcg@10": 0.02, "recall@10": 0.05}
    fused = {"hr@10": 0.04, "ndcg@10": 0.02, "recall@10": 0.05}
    ok, lines = compare_results(baseline, fused, tolerance=0.005)
    assert not ok
    assert any("hr@10" in line and "FAIL" in line for line in lines)


def test_compare_cli_reads_user_mean(tmp_path: Path) -> None:
    baseline = {
        "aggregation": "user_mean",
        "means_per_user": {"hr@10": 0.1, "ndcg@10": 0.05, "recall@10": 0.1},
    }
    fused = {
        "aggregation": "user_mean",
        "means_per_user": {"hr@10": 0.1, "ndcg@10": 0.05, "recall@10": 0.1},
    }
    b_path = tmp_path / "baseline.json"
    f_path = tmp_path / "fused.json"
    b_path.write_text(json.dumps(baseline), encoding="utf-8")
    f_path.write_text(json.dumps(fused), encoding="utf-8")

    from scripts.compare_emorecagent_stage2 import _load_means

    assert _load_means(b_path)["hr@10"] == 0.1
