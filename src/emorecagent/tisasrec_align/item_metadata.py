"""Load item metadata for Stage-2 LLM candidate cards / Yelp_AC T_u.

Supports:
  - RecBole ``*.item`` TSV (Yelp_AC)
  - Amazon Reviews 2023 ``meta_*.jsonl`` (title + categories)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ItemMeta:
    item_id: str
    name: str = ""
    categories: str = ""
    city: str = ""
    state: str = ""


def resolve_item_path(path: str | Path) -> Path:
    """Resolve a RecBole ``.item`` file or a directory containing one."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RecBole item path not found: {p}")
    if p.is_file():
        if p.suffix != ".item":
            raise FileNotFoundError(
                f"Expected a .item file, got {p} (suffix={p.suffix!r})"
            )
        return p
    candidates = sorted(p.glob("*.item"))
    if not candidates:
        candidates = sorted(p.rglob("*.item"))
    if not candidates:
        raise FileNotFoundError(f"No *.item file under {p}")
    if len(candidates) > 1:
        preferred = [c for c in candidates if c.stem.lower() == p.name.lower()]
        if len(preferred) == 1:
            return preferred[0]
    return candidates[0]


def _parse_header(header_line: str) -> list[str]:
    cols: list[str] = []
    for cell in header_line.strip().split("\t"):
        name = cell.split(":", 1)[0].strip()
        if not name:
            raise ValueError(f"Empty RecBole item header cell in {header_line!r}")
        cols.append(name)
    return cols


def _unescape(value: str) -> str:
    return value.replace("\\/", "/").strip()


def load_item_metadata(path: str | Path) -> dict[str, ItemMeta]:
    """Map item_id → name/categories/city/state from a RecBole ``.item`` TSV."""
    item_path = resolve_item_path(path)
    out: dict[str, ItemMeta] = {}
    with item_path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            raise ValueError(f"Empty RecBole item file: {item_path}")
        cols = _parse_header(header)
        # RecBole Yelp dump uses item_id / item_name; other dumps may use
        # business_id / name.
        aliases = {
            "business_id": "item_id",
            "name": "item_name",
        }
        cols = [aliases.get(c, c) for c in cols]
        if "item_id" not in cols:
            raise ValueError(
                f"RecBole item {item_path} missing item_id; got {cols}"
            )
        idx = {name: i for i, name in enumerate(cols)}
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < len(cols):
                continue
            item_id = parts[idx["item_id"]].strip()
            if not item_id:
                continue
            name = ""
            if "item_name" in idx:
                name = _unescape(parts[idx["item_name"]])
            categories = ""
            if "categories" in idx:
                categories = _unescape(parts[idx["categories"]])
            city = _unescape(parts[idx["city"]]) if "city" in idx else ""
            state = _unescape(parts[idx["state"]]) if "state" in idx else ""
            out[item_id] = ItemMeta(
                item_id=item_id,
                name=name,
                categories=categories,
                city=city,
                state=state,
            )
    return out


def _categories_to_str(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                parts.append(x.strip())
            elif isinstance(x, list):
                # nested Amazon category paths
                parts.extend(str(y).strip() for y in x if str(y).strip())
        # de-dupe, keep order
        seen: set[str] = set()
        out: list[str] = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return ", ".join(out)
    return str(raw).strip()


def load_amazon_meta_jsonl(
    path: str | Path,
    *,
    keep_ids: set[str] | None = None,
) -> dict[str, ItemMeta]:
    """Load Amazon Reviews 2023 meta JSONL (``parent_asin`` / ``asin`` + title).

    When ``keep_ids`` is set, only those ASINs are retained (stream-friendly).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Amazon meta JSONL not found: {p}")
    want = keep_ids
    remaining = set(want) if want is not None else None
    out: dict[str, ItemMeta] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = str(row.get("parent_asin") or row.get("asin") or "").strip()
            if not item_id:
                continue
            if remaining is not None and item_id not in remaining:
                continue
            title = str(row.get("title") or row.get("item_name") or "").strip()
            cats = _categories_to_str(row.get("categories"))
            if not cats:
                cats = str(row.get("main_category") or "").strip()
            out[item_id] = ItemMeta(
                item_id=item_id, name=title, categories=cats
            )
            if remaining is not None:
                remaining.discard(item_id)
                if not remaining:
                    break
    return out


def load_stage2_item_metadata(
    path: str | Path,
    *,
    keep_ids: set[str] | None = None,
) -> dict[str, ItemMeta]:
    """Auto-detect RecBole ``.item`` vs Amazon ``.jsonl`` for Stage-2 cards."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"item metadata path not found: {p}")
    if p.is_file() and p.suffix == ".jsonl":
        meta = load_amazon_meta_jsonl(p, keep_ids=keep_ids)
        logger.info(
            "loaded Amazon meta cards=%s from %s (keep_ids=%s)",
            f"{len(meta):,}",
            p,
            f"{len(keep_ids):,}" if keep_ids is not None else "all",
        )
        return meta
    if p.is_file() and p.suffix == ".item":
        return load_item_metadata(p)
    # Directory: prefer RecBole .item (Yelp_AC); else Amazon meta_*.jsonl
    try:
        return load_item_metadata(p)
    except FileNotFoundError:
        pass
    jsonl_hits = sorted(p.glob("meta_*.jsonl")) + sorted(p.glob("*.jsonl"))
    if not jsonl_hits:
        raise FileNotFoundError(
            f"No RecBole .item or Amazon meta JSONL under {p}"
        )
    return load_amazon_meta_jsonl(jsonl_hits[0], keep_ids=keep_ids)


def format_item_card(
    item_id: str,
    score: float,
    meta: ItemMeta | None,
    *,
    max_cats: int = 3,
    max_name: int = 40,
    include_score: bool = True,
) -> str:
    """Compact candidate card: ``id | S=… | name=… | cats=…``."""
    parts = [item_id]
    if include_score:
        parts[0] = f"{item_id} | S={float(score):.4f}"
    if meta is None:
        return parts[0] if include_score else item_id
    if meta.name:
        name = meta.name if len(meta.name) <= max_name else meta.name[: max_name - 1] + "…"
        parts.append(f"name={name}")
    if meta.categories:
        cats = [c.strip() for c in meta.categories.split(",") if c.strip()]
        if cats:
            parts.append(f"cats={', '.join(cats[:max_cats])}")
    return " | ".join(parts)


def format_anchor_card(
    item_id: str,
    meta: ItemMeta | None,
    *,
    max_name: int = 80,
    max_cats: int = 3,
) -> str:
    """Prefix-reviewed item with title (no Stage-1 score)."""
    return format_item_card(
        item_id,
        0.0,
        meta,
        max_cats=max_cats,
        max_name=max_name,
        include_score=False,
    )
