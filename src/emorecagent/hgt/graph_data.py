"""Serializable heterogeneous graph bundle for HGT training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class HgtGraphBundle:
    """Train-scoped heterogeneous graph tensors and id maps."""

    node_feature: np.ndarray
    node_type: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_time: np.ndarray
    user_ids: list[str]
    item_ids: list[str]
    aspect_ids: list[str]
    train_pairs: list[tuple[int, int]]
    valid_pairs: list[tuple[int, int]]
    aspect_vocab: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Saving HGT graphs requires torch. Install with: pip install -e '.[hgt]'"
            ) from exc
        payload = {
            "node_feature": self.node_feature,
            "node_type": self.node_type,
            "edge_index": self.edge_index,
            "edge_type": self.edge_type,
            "edge_time": self.edge_time,
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "aspect_ids": self.aspect_ids,
            "train_pairs": self.train_pairs,
            "valid_pairs": self.valid_pairs,
            "aspect_vocab": self.aspect_vocab,
            "meta": self.meta,
        }
        torch.save(payload, out)
        manifest = out.with_suffix(".manifest.json")
        manifest.write_text(
            json.dumps(
                {
                    "graph_path": str(out),
                    "n_users": len(self.user_ids),
                    "n_items": len(self.item_ids),
                    "n_aspects": len(self.aspect_ids),
                    "n_edges": int(self.edge_index.shape[1]),
                    "meta": self.meta,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "HgtGraphBundle":
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Loading HGT graphs requires torch. Install with: pip install -e '.[hgt]'"
            ) from exc
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            node_feature=np.asarray(payload["node_feature"], dtype=np.float32),
            node_type=np.asarray(payload["node_type"], dtype=np.int64),
            edge_index=np.asarray(payload["edge_index"], dtype=np.int64),
            edge_type=np.asarray(payload["edge_type"], dtype=np.int64),
            edge_time=np.asarray(payload["edge_time"], dtype=np.int64),
            user_ids=list(payload["user_ids"]),
            item_ids=list(payload["item_ids"]),
            aspect_ids=list(payload["aspect_ids"]),
            train_pairs=[tuple(p) for p in payload["train_pairs"]],
            valid_pairs=[tuple(p) for p in payload["valid_pairs"]],
            aspect_vocab=dict(payload["aspect_vocab"]),
            meta=dict(payload.get("meta", {})),
        )

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_nodes(self) -> int:
        return int(self.node_type.shape[0])

    def user_offset(self) -> int:
        return 0

    def item_offset(self) -> int:
        return len(self.user_ids)

    def aspect_offset(self) -> int:
        return len(self.user_ids) + len(self.item_ids)

    def global_user_idx(self, local_u: int) -> int:
        return local_u

    def global_item_idx(self, local_i: int) -> int:
        return self.item_offset() + local_i

    def global_aspect_idx(self, local_a: int) -> int:
        return self.aspect_offset() + local_a
