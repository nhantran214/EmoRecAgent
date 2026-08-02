"""ABSA-free prereq checks for Yelp_AC."""

from __future__ import annotations

from emorecagent.config import load_config


def test_prereqs_skip_absa_when_disabled(tmp_path) -> None:
    from scripts import check_emorecagent_prereqs as prereqs

    out = tmp_path / "processed" / "Yelp_AC"
    out.mkdir(parents=True)
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        (out / name).write_text("", encoding="utf-8")
    inter_dir = tmp_path / "raw"
    inter_dir.mkdir()
    (inter_dir / "yelp.inter").write_text(
        "user_id:token\titem_id:token\trating:float\ttimestamp:float\n",
        encoding="utf-8",
    )

    cfg = load_config("configs/categories/Yelp_AC.yaml")
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(
                update={
                    "out_dir": str(out),
                    "inter_path": str(inter_dir),
                    "review_path": str(inter_dir),
                    "meta_path": str(inter_dir),
                }
            ),
            "absa": cfg.absa.model_copy(
                update={"cache_path": str(out / "absa_cache.sqlite")}
            ),
        }
    )
    missing: list[str] = []
    prereqs._check_data_absa(cfg, missing)
    assert missing == [], missing


def test_prereqs_still_require_absa_when_enabled(tmp_path) -> None:
    from scripts import check_emorecagent_prereqs as prereqs

    cfg = load_config("configs/categories/Yelp.yaml")
    out = tmp_path / "Yelp"
    out.mkdir()
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        (out / name).write_text("", encoding="utf-8")
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(
                update={
                    "out_dir": str(out),
                    "review_path": str(tmp_path / "missing_reviews.jsonl"),
                }
            ),
            "absa": cfg.absa.model_copy(
                update={"cache_path": str(tmp_path / "missing_absa.sqlite")}
            ),
        }
    )
    missing: list[str] = []
    prereqs._check_data_absa(cfg, missing)
    assert any("absa" in m.lower() for m in missing)
