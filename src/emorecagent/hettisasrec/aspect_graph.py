"""Item–aspect adjacency from ABSA cache (no user nodes)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from ..absa.normalize import normalize_aspect
from ..data.review_index import build_review_index_from_scope
from ..data.types import Interaction
from ..sequential.id_maps import IdMaps
from .aspect_vocab import AspectVocab, open_absa_cache_readonly


def _item_aspect_scores_from_cache(
    cache_path: Path,
    review_index: dict[str, tuple[str, str, int]],
) -> dict[str, dict[str, float]]:
    conn = open_absa_cache_readonly(cache_path)
    item_aspects: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    try:
        rows = conn.execute("SELECT review_id, triples_json FROM absa_cache").fetchall()
    finally:
        conn.close()
    idx_set = set(review_index)
    sentiment_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    for review_id, payload in rows:
        if review_id not in idx_set:
            continue
        _, item_id, _ = review_index[review_id]
        data = json.loads(payload)
        for t in data.get("triples", []):
            aspect = normalize_aspect(str(t.get("aspect", "")))
            if not aspect:
                continue
            pol = sentiment_map.get(str(t.get("sentiment", "neutral")), 0.0)
            item_aspects[item_id][aspect].append(pol)
    out: dict[str, dict[str, float]] = {}
    for item_id, asp_map in item_aspects.items():
        out[item_id] = {a: sum(v) / len(v) for a, v in asp_map.items()}
    return out


def _pad_neighbors(
    neighbors: dict[int, list[tuple[int, float]]],
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_deg = max((len(v) for v in neighbors.values()), default=1)
    max_deg = max(max_deg, 1)
    idx = torch.zeros(size, max_deg, dtype=torch.long)
    w = torch.zeros(size, max_deg, dtype=torch.float32)
    for node, pairs in neighbors.items():
        for j, (nbr, weight) in enumerate(pairs[:max_deg]):
            idx[node, j] = nbr
            w[node, j] = weight
    return idx, w


@dataclass(frozen=True, slots=True)
class AspectGraphBundle:
    """Sparse item↔aspect neighborhood tensors (1-based ids; row 0 unused)."""

    item_to_aspect_idx: torch.Tensor
    item_to_aspect_w: torch.Tensor
    aspect_to_item_idx: torch.Tensor
    aspect_to_item_w: torch.Tensor
    n_items: int
    n_aspects: int
    item_ids: tuple[str, ...]
    aspect_ids: tuple[str, ...]

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "item_to_aspect_idx": self.item_to_aspect_idx,
                "item_to_aspect_w": self.item_to_aspect_w,
                "aspect_to_item_idx": self.aspect_to_item_idx,
                "aspect_to_item_w": self.aspect_to_item_w,
                "n_items": self.n_items,
                "n_aspects": self.n_aspects,
                "item_ids": list(self.item_ids),
                "aspect_ids": list(self.aspect_ids),
            },
            out,
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> AspectGraphBundle:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            item_to_aspect_idx=payload["item_to_aspect_idx"],
            item_to_aspect_w=payload["item_to_aspect_w"],
            aspect_to_item_idx=payload["aspect_to_item_idx"],
            aspect_to_item_w=payload["aspect_to_item_w"],
            n_items=int(payload["n_items"]),
            n_aspects=int(payload["n_aspects"]),
            item_ids=tuple(payload["item_ids"]),
            aspect_ids=tuple(payload["aspect_ids"]),
        )


def build_aspect_graph(
    train: list[Interaction],
    id_maps: IdMaps,
    vocab: AspectVocab,
    cache_path: str | Path,
    review_path: str | Path,
) -> AspectGraphBundle:
    """Build typed item–aspect edges from train-scoped ABSA triples."""
    review_index = build_review_index_from_scope(train, review_path)
    raw_scores = _item_aspect_scores_from_cache(Path(cache_path), review_index)

    item_to_aspect: dict[int, list[tuple[int, float]]] = defaultdict(list)
    aspect_to_item: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for item_str, asp_map in raw_scores.items():
        item_local = id_maps.item_to_idx.get(item_str)
        if item_local is None:
            continue
        for asp_str, score in asp_map.items():
            asp_local = vocab.id_for(asp_str) + 1
            weight = abs(float(score)) + 0.1
            item_to_aspect[item_local].append((asp_local, weight))
            aspect_to_item[asp_local].append((item_local, weight))

    n_items = len(id_maps.item_to_idx)
    n_aspects = vocab.size
    item_idx, item_w = _pad_neighbors(item_to_aspect, n_items + 1)
    asp_idx, asp_w = _pad_neighbors(aspect_to_item, n_aspects + 1)

    item_ids = tuple(id_maps.idx_to_item[i] for i in range(1, n_items + 1))
    aspect_ids = vocab.aspects

    return AspectGraphBundle(
        item_to_aspect_idx=item_idx,
        item_to_aspect_w=item_w,
        aspect_to_item_idx=asp_idx,
        aspect_to_item_w=asp_w,
        n_items=n_items,
        n_aspects=n_aspects,
        item_ids=item_ids,
        aspect_ids=aspect_ids,
    )


def build_and_save_aspect_graph(
    train: list[Interaction],
    id_maps: IdMaps,
    vocab: AspectVocab,
    cache_path: str | Path,
    review_path: str | Path,
    out_path: str | Path,
) -> AspectGraphBundle:
    bundle = build_aspect_graph(train, id_maps, vocab, cache_path, review_path)
    bundle.save(out_path)
    return bundle
