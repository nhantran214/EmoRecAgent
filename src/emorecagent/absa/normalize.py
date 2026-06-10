"""Canonical aspect vocabulary (U4)."""

from __future__ import annotations

# Lightweight synonym map; extend from frequent aspects during dataset build.
SYNONYM_MAP: dict[str, str] = {
    "build": "build quality",
    "build quality": "build quality",
    "quality": "build quality",
    "smell": "scent",
    "fragrance": "scent",
    "odor": "scent",
    "texture": "feel",
    "consistency": "feel",
    "packaging": "packaging",
    "package": "packaging",
    "price": "value",
    "cost": "value",
    "value for money": "value",
    "comfort": "comfort",
    "fit": "comfort",
}


def normalize_aspect(raw: str) -> str:
    key = raw.strip().lower()
    return SYNONYM_MAP.get(key, key)
