"""TiSASRec candidate retriever (baseline checkpoint loader)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.types import Interaction
from .id_maps import IdMaps
from .seq_utils import TiSASRecArgs, compute_repos
from .tisasrec_loader import load_tisasrec_model

logger = logging.getLogger(__name__)


class TiSASRecRetriever:
    """Score and retrieve items with a trained TiSASRec checkpoint."""

    backend = "tisasrec"

    def __init__(
        self,
        model: Any,
        id_maps: IdMaps,
        device: torch.device,
        model_args: TiSASRecArgs,
        *,
        pool_size: int = 50,
        item_ids: list[str] | None = None,
    ) -> None:
        self._model = model
        self._id_maps = id_maps
        self._device = device
        self._args = model_args
        self.pool_size = pool_size
        self._items = item_ids or sorted(id_maps.item_to_idx.keys())
        self._user_history: dict[str, list[tuple[int, str]]] = {}
        self._query_context: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        id_maps: IdMaps,
        model_cfg: dict,
        *,
        device: str = "auto",
        pool_size: int = 50,
    ) -> TiSASRecRetriever:
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"TiSASRec checkpoint not found: {ckpt}")
        dev = torch.device(
            "cuda"
            if device == "cuda" and torch.cuda.is_available()
            else "cpu"
            if device == "cpu"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model, model_args = load_tisasrec_model(
            ckpt,
            id_maps,
            model_cfg,
            device=dev,
        )
        return cls(model, id_maps, dev, model_args, pool_size=pool_size)

    def fit(self, interactions: list[Interaction]) -> TiSASRecRetriever:
        self._user_history.clear()
        self._query_context.clear()
        for it in interactions:
            self._user_history.setdefault(it.user_id, []).append(
                (it.timestamp, it.item)
            )
        for uid in self._user_history:
            self._user_history[uid].sort(key=lambda t: (t[0], t[1]))
        return self

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None:
        self._query_context[user_id] = self._build_context(
            self._user_history.get(user_id, []),
            before_ts=timestamp_ms,
        )

    def _normalize_times(
        self, events: list[tuple[int, str]]
    ) -> list[tuple[int, int]]:
        times = [ts for ts, _ in events]
        if not times:
            return []
        time_diffs = {
            times[i + 1] - times[i]
            for i in range(len(times) - 1)
            if times[i + 1] - times[i] != 0
        }
        time_scale = min(time_diffs) if time_diffs else 1
        time_min = min(times)
        out: list[tuple[int, int]] = []
        for ts, item in events:
            idx = self._id_maps.item_to_idx.get(item)
            if idx is None:
                continue
            norm_ts = int(round((ts - time_min) / time_scale) + 1)
            out.append((idx, norm_ts))
        return out

    def _build_context(
        self,
        events: list[tuple[int, str]],
        *,
        before_ts: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        maxlen = self._args.maxlen
        if before_ts is not None:
            events = [(ts, item) for ts, item in events if ts < before_ts]
        pairs = self._normalize_times(events)

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for item_idx, norm_ts in reversed(pairs):
            seq[idx] = item_idx
            time_seq[idx] = norm_ts
            idx -= 1
            if idx == -1:
                break

        time_matrix = compute_repos(time_seq, self._args.time_span)
        return seq, time_seq, time_matrix

    def _context_for_user(
        self, user_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if user_id in self._query_context:
            return self._query_context[user_id]
        return self._build_context(self._user_history.get(user_id, []))

    def _raw_scores(
        self, user_id: str, candidates: list[str]
    ) -> dict[str, float]:
        seq, _time_seq, time_matrix = self._context_for_user(user_id)
        if not np.any(seq):
            return {item: 0.0 for item in candidates}

        user_idx = self._id_maps.user_to_idx.get(user_id, 1)

        item_indices: list[int] = []
        valid_candidates: list[str] = []
        for item in candidates:
            idx = self._id_maps.item_to_idx.get(item)
            if idx is not None:
                item_indices.append(idx)
                valid_candidates.append(item)

        if not item_indices:
            return {item: 0.0 for item in candidates}

        with torch.no_grad():
            logits = self._model.predict(
                np.array([user_idx]),
                np.array([seq]),
                np.array([time_matrix]),
                np.array([item_indices]),
            )
            scores = logits[0].cpu().tolist()

        out = {item: 0.0 for item in candidates}
        for item, score in zip(valid_candidates, scores, strict=True):
            out[item] = float(score)
        return out

    def score(
        self,
        user_id: str,
        item_ids: list[str],
        *,
        gammas: dict[str, float] | None = None,
    ) -> dict[str, float]:
        del gammas
        raw = self._raw_scores(user_id, item_ids)
        if not raw:
            return {}
        lo, hi = min(raw.values()), max(raw.values())
        if hi - lo < 1e-12:
            return {i: 0.0 for i in item_ids}
        return {i: (v - lo) / (hi - lo) for i, v in raw.items()}

    def retrieve(
        self,
        user_id: str,
        k: int,
        gammas: dict[str, float] | None,
        candidates: list[str],
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        del gammas
        exclude = exclude or set()
        pool = [c for c in candidates if c not in exclude]
        if len(pool) <= k:
            return list(pool)
        raw = self._raw_scores(user_id, pool)
        ranked = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))
        return [item for item, _ in ranked[:k]]

    def retrieve_top_m(
        self,
        user_id: str,
        m: int,
        candidates: list[str],
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Return top-M by raw TiSASRec score (for RRF fusion)."""
        exclude = exclude or set()
        pool = [c for c in candidates if c not in exclude]
        if len(pool) <= m:
            return list(pool)
        raw = self._raw_scores(user_id, pool)
        ranked = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))
        return [item for item, _ in ranked[:m]]
