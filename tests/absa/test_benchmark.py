"""Benchmark / quality-compare script smoke tests (mock mode)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(script: str, *args: str, tmp_path: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(repo / "scripts" / script),
        "--mock",
        "--n-reviews",
        "2",
        "--warmup",
        "0",
        "--tmp-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / f"{script}.json"),
    ]
    if script == "compare_absa_quality.py":
        cmd = [
            sys.executable,
            str(repo / "scripts" / script),
            "--mock",
            "--tmp-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "quality_cmp.json"),
        ]
    proc = subprocess.run(
        cmd,
        cwd=repo,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": "src",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "secret",
            "OLLAMA_HOST": "http://localhost:11434",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out_path = tmp_path / (
        "quality_cmp.json" if script == "compare_absa_quality.py" else f"{script}.json"
    )
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_benchmark_mock_writes_required_keys(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[2]
    targets = repo / "data/processed/Beauty_and_Personal_Care/absa_targets.jsonl"
    if not targets.exists() or targets.stat().st_size == 0:
        return
    payload = _run("benchmark_absa_latency.py", tmp_path=tmp_path)
    assert "backends" in payload
    assert payload["speedup_ratio"] is not None
    hybrid = next(b for b in payload["backends"] if b["backend"] == "hybrid")
    assert "repair_rate" in hybrid
    for row in payload["backends"]:
        assert str(tmp_path) in row["cache_path"]


def test_compare_quality_mock_skips_without_gold(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[2]
    gold = repo / "data/labeled/absa_gold.jsonl"
    if not gold.exists():
        return
    payload = _run("compare_absa_quality.py", tmp_path=tmp_path)
    assert "quality_gate_pass" in payload
    assert "llm_only" in payload["backends"]
