"""RecBole ``*.inter`` TSV loader for ID-only sequential experiments.

Parses the typed header (``field:type`` columns), maps ``item_id`` → item,
and converts timestamps from unix **seconds** to the repo's millisecond
``Interaction.timestamp`` convention.
"""

from __future__ import annotations

from pathlib import Path

from .types import Interaction

# AC-TSR / RecBole yelp.yaml closed calendar-2019 window (raw unix seconds).
ACTSR_YELP_MIN_TIMESTAMP_S = 1_546_264_800
ACTSR_YELP_MAX_TIMESTAMP_S = 1_577_714_400


def resolve_inter_path(path: str | Path) -> Path:
    """Resolve a RecBole ``.inter`` file or a directory containing one."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RecBole inter path not found: {p}")
    if p.is_file():
        if p.suffix != ".inter":
            raise FileNotFoundError(
                f"Expected a .inter file, got {p} (suffix={p.suffix!r})"
            )
        return p
    candidates = sorted(p.glob("*.inter"))
    if not candidates:
        nested = sorted(p.rglob("*.inter"))
        candidates = nested
    if not candidates:
        raise FileNotFoundError(f"No *.inter file under {p}")
    if len(candidates) > 1:
        # Prefer a file whose stem matches the directory name (yelp/yelp.inter).
        preferred = [c for c in candidates if c.stem.lower() == p.name.lower()]
        if len(preferred) == 1:
            return preferred[0]
    return candidates[0]


def _parse_header(header_line: str) -> list[str]:
    cols: list[str] = []
    for cell in header_line.strip().split("\t"):
        name = cell.split(":", 1)[0].strip()
        if not name:
            raise ValueError(f"Empty RecBole header cell in {header_line!r}")
        cols.append(name)
    return cols


def load_recbole_inter(
    path: str | Path,
    *,
    max_scan: int | None = None,
    min_timestamp_s: int | None = None,
    max_timestamp_s: int | None = None,
) -> list[Interaction]:
    """Load interactions from a RecBole ``.inter`` TSV.

    Rows outside ``[min_timestamp_s, max_timestamp_s]`` (inclusive, unix
    seconds) are dropped when those bounds are set. Output timestamps are ms.
    """
    inter_path = resolve_inter_path(path)
    out: list[Interaction] = []
    with inter_path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            raise ValueError(f"Empty RecBole inter file: {inter_path}")
        cols = _parse_header(header)
        required = {"user_id", "item_id", "rating", "timestamp"}
        missing = required - set(cols)
        if missing:
            # Yelp native RecBole dumps may use business_id / stars / date.
            alias = {
                "business_id": "item_id",
                "stars": "rating",
                "date": "timestamp",
            }
            cols = [alias.get(c, c) for c in cols]
            missing = required - set(cols)
            if missing:
                raise ValueError(
                    f"RecBole inter {inter_path} missing columns {sorted(missing)}; "
                    f"got {cols}"
                )
        idx = {name: i for i, name in enumerate(cols)}
        for n, line in enumerate(fh):
            if max_scan is not None and n >= max_scan:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < len(cols):
                continue
            try:
                ts_s = float(parts[idx["timestamp"]])
            except ValueError:
                continue
            if min_timestamp_s is not None and ts_s < min_timestamp_s:
                continue
            if max_timestamp_s is not None and ts_s > max_timestamp_s:
                continue
            try:
                rating = float(parts[idx["rating"]])
            except ValueError:
                continue
            user = parts[idx["user_id"]].strip()
            item = parts[idx["item_id"]].strip()
            if not user or not item:
                continue
            out.append(
                Interaction(
                    user_id=user,
                    item=item,
                    rating=rating,
                    timestamp=int(ts_s * 1000),
                )
            )
    return out


def filter_by_timestamp_s(
    interactions: list[Interaction],
    *,
    min_timestamp_s: int | None = None,
    max_timestamp_s: int | None = None,
) -> list[Interaction]:
    """Filter Interaction rows whose ms timestamps fall outside second bounds."""
    if min_timestamp_s is None and max_timestamp_s is None:
        return list(interactions)
    lo_ms = None if min_timestamp_s is None else int(min_timestamp_s) * 1000
    hi_ms = None if max_timestamp_s is None else int(max_timestamp_s) * 1000
    out: list[Interaction] = []
    for it in interactions:
        if lo_ms is not None and it.timestamp < lo_ms:
            continue
        if hi_ms is not None and it.timestamp > hi_ms:
            continue
        out.append(it)
    return out
