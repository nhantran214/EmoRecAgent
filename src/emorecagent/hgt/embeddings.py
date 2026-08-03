"""Persist and load frozen HGT node embedding tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EmbeddingStore:
    user_ids: list[str]
    item_ids: list[str]
    aspect_ids: list[str]
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    aspect_embeddings: np.ndarray
    meta: dict

    @property
    def dim(self) -> int:
        return int(self.user_embeddings.shape[1])

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "user_embeddings.npy", self.user_embeddings)
        np.save(out / "item_embeddings.npy", self.item_embeddings)
        np.save(out / "aspect_embeddings.npy", self.aspect_embeddings)
        ids = {
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "aspect_ids": self.aspect_ids,
        }
        (out / "ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
        (out / "manifest.json").write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "EmbeddingStore":
        root = Path(directory)
        ids = json.loads((root / "ids.json").read_text(encoding="utf-8"))
        meta_path = root / "manifest.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(
            user_ids=list(ids["user_ids"]),
            item_ids=list(ids["item_ids"]),
            aspect_ids=list(ids["aspect_ids"]),
            user_embeddings=np.load(root / "user_embeddings.npy"),
            item_embeddings=np.load(root / "item_embeddings.npy"),
            aspect_embeddings=np.load(root / "aspect_embeddings.npy"),
            meta=meta,
        )

    def user_index(self) -> dict[str, int]:
        return {u: i for i, u in enumerate(self.user_ids)}

    def item_index(self) -> dict[str, int]:
        return {it: i for i, it in enumerate(self.item_ids)}

    def aspect_index(self) -> dict[str, int]:
        return {a: i for i, a in enumerate(self.aspect_ids)}
