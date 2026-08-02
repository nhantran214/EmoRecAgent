"""Export EmoRecAgent JSONL splits to a single RecBole ``.inter`` file."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


_HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"


def _ts_seconds(raw: int | float) -> float:
    """JSONL stores ms; RecBole Yelp dumps use unix seconds."""
    value = float(raw)
    if value > 1e12:
        return value / 1000.0
    return value


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def export_combined_inter(
    processed_dir: Path,
    dataset_dir: Path,
    dataset_name: str,
) -> dict[str, int]:
    """Merge train/valid/test JSONL → ``{name}.inter`` (sorted per user by time).

    RecBole then applies ``LS: valid_and_test`` leave-last-out, matching
    AC-TSR / our processed LOO when the combined history is identical.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    by_user: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
    split_counts = {"train": 0, "valid": 0, "test": 0}

    for split in ("train", "valid", "test"):
        src = processed_dir / f"{split}.jsonl"
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}")
        for row in _load_jsonl(src):
            ts = _ts_seconds(row["timestamp"])
            rating = float(row.get("rating", 1.0))
            by_user[row["user_id"]].append((ts, row["item"], rating))
            split_counts[split] += 1

    out = dataset_dir / f"{dataset_name}.inter"
    n = 0
    with out.open("w", encoding="utf-8") as dst:
        dst.write(_HEADER)
        for user_id in sorted(by_user):
            events = sorted(by_user[user_id], key=lambda t: (t[0], t[1]))
            for ts, item, rating in events:
                dst.write(f"{user_id}\t{item}\t{rating}\t{ts}\n")
                n += 1

    return {**split_counts, "combined": n, "users": len(by_user)}
