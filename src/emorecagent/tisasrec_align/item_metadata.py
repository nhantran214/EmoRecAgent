"""Load RecBole ``*.item`` metadata for ID-only Stage-2 preference/cards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


def format_item_card(
    item_id: str,
    score: float,
    meta: ItemMeta | None,
    *,
    max_cats: int = 3,
    max_name: int = 40,
) -> str:
    """Compact candidate card: ``id | S=… | name=… | cats=…``."""
    parts = [f"{item_id} | S={float(score):.4f}"]
    if meta is None:
        return parts[0]
    if meta.name:
        name = meta.name if len(meta.name) <= max_name else meta.name[: max_name - 1] + "…"
        parts.append(f"name={name}")
    if meta.categories:
        cats = [c.strip() for c in meta.categories.split(",") if c.strip()]
        if cats:
            parts.append(f"cats={', '.join(cats[:max_cats])}")
    return " | ".join(parts)
