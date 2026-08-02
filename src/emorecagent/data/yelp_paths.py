"""Resolve and locate Yelp Open Dataset JSON files."""

from __future__ import annotations

from pathlib import Path

# Official / legacy Yelp extract names.
REVIEW_FILENAMES = (
    "yelp_academic_dataset_review.json",
    "review.json",
)
BUSINESS_FILENAMES = (
    "yelp_academic_dataset_business.json",
    "business.json",
)

# Default unpack location used by download_yelp_open_dataset.py
DEFAULT_YELP_DATASET_DIR = Path("data/yelp-open-dataset/raw/yelp_dataset")


def find_yelp_json(root: Path, candidates: tuple[str, ...]) -> Path | None:
    """Locate a Yelp JSON file under ``root`` (shallow then recursive)."""
    root = Path(root)
    for name in candidates:
        direct = root / name
        if direct.is_file():
            return direct
    for name in candidates:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def resolve_review_source(path: str | Path) -> Path:
    """Return a readable review JSON/JSONL path.

    Accepts either a file or a directory (e.g. ``.../raw/yelp_dataset``),
    in which case the official Yelp review dump is resolved.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        found = find_yelp_json(p, REVIEW_FILENAMES)
        if found is not None:
            return found
        raise FileNotFoundError(
            f"No Yelp review JSON under {p} (looked for {REVIEW_FILENAMES})"
        )
    # Path may not exist yet; keep as-is for clearer caller errors.
    return p


def resolve_meta_source(path: str | Path) -> Path:
    """Return a readable business/meta JSON path (file or Yelp dataset dir)."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        found = find_yelp_json(p, BUSINESS_FILENAMES)
        if found is not None:
            return found
        raise FileNotFoundError(
            f"No Yelp business JSON under {p} (looked for {BUSINESS_FILENAMES})"
        )
    return p


def is_yelp_review_row(obj: dict) -> bool:
    """True when a JSON object looks like a native Yelp review."""
    if "business_id" in obj and ("stars" in obj or "date" in obj):
        return True
    if (
        "date" in obj
        and "stars" in obj
        and "parent_asin" not in obj
        and "asin" not in obj
    ):
        return True
    return False
