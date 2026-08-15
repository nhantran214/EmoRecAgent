"""Yelp_AC build_dataset path does not touch data/processed/Yelp."""

from __future__ import annotations

import json
from pathlib import Path

from emorecagent.config import load_config
from emorecagent.data.recbole_inter import (
    ACTSR_YELP_MAX_TIMESTAMP_S,
    ACTSR_YELP_MIN_TIMESTAMP_S,
)
from emorecagent.data.kcore import k_core_filter
from emorecagent.data.loader import dedup_earliest
from emorecagent.data.recbole_inter import load_recbole_inter
from emorecagent.data.split import leave_last_out, write_split


def test_yelp_ac_build_pipeline_isolated(tmp_path: Path, monkeypatch) -> None:
    yelp_dir = tmp_path / "processed" / "Yelp"
    yelp_dir.mkdir(parents=True)
    sentinel = yelp_dir / "manifest.json"
    sentinel.write_text('{"sentinel": true}\n', encoding="utf-8")

    out_dir = tmp_path / "processed" / "Yelp_AC"
    inter = tmp_path / "yelp.inter"
    base = ACTSR_YELP_MIN_TIMESTAMP_S
    header = (
        "user_id:token\titem_id:token\trating:float\ttimestamp:float\t"
        "useful:float\tfunny:float\tcool:float\treview_id:token\n"
    )
    rows = []
    # 5 users × 5 items shared → survives 5-core after filter.
    # Include a duplicate (user,item) with distinct review_id (RecBole keeps both).
    for u in range(5):
        for i in range(5):
            rows.append(
                f"u{u}\ti{i}\t5.0\t{base + u * 10 + i}\t0\t0\t0\tr{u}_{i}\n"
            )
    rows.append(f"u0\ti0\t4.0\t{base + 100}\t0\t0\t0\tr0_0_b\n")
    inter.write_text(header + "".join(rows), encoding="utf-8")

    raw = load_recbole_inter(
        inter,
        min_timestamp_s=ACTSR_YELP_MIN_TIMESTAMP_S,
        max_timestamp_s=ACTSR_YELP_MAX_TIMESTAMP_S,
    )
    # Paper / RecBole path: no user-item dedup.
    kcore = k_core_filter(raw, 5)
    assert len(kcore) >= len(dedup_earliest(raw))  # multi-visit retained upstream
    split = leave_last_out(kcore, min_history=0)
    write_split(
        out_dir,
        split,
        seed=42,
        k_core=5,
        extra_manifest={
            "split_method": "leave_last_out",
            "min_timestamp_s": ACTSR_YELP_MIN_TIMESTAMP_S,
            "max_timestamp_s": ACTSR_YELP_MAX_TIMESTAMP_S,
            "dedup_user_item": False,
            "source": "recbole_inter",
        },
    )

    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "valid.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()
    man = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["split_method"] == "leave_last_out"
    assert man["min_timestamp_s"] == ACTSR_YELP_MIN_TIMESTAMP_S
    assert man["dedup_user_item"] is False
    # Review track sentinel untouched.
    assert json.loads(sentinel.read_text(encoding="utf-8"))["sentinel"] is True


def test_yelp_ac_config_loads() -> None:
    cfg_path = Path("configs/categories/Yelp_AC.yaml")
    if not cfg_path.exists():
        return
    cfg = load_config(cfg_path)
    assert cfg.data.category == "Yelp_AC"
    assert cfg.data.split_method == "leave_last_out"
    assert cfg.data.dedup_user_item is False
    assert cfg.absa.enabled is False


def test_yelp_ac_paper_config_disables_dedup() -> None:
    paper = load_config("configs/legacy/categories/Yelp_AC_tisasrec_paper.yaml")
    assert paper.data.dedup_user_item is False
    assert paper.data.inter_path is not None
