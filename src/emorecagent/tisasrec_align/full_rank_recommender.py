"""Algorithm 1 full-rank recommender with neuro-symbolic fusion."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..baselines.base import Recommender
from ..config import Config
from ..data.types import Interaction
from ..sequential.id_maps import IdMaps
from ..sequential.seq_utils import compute_repos, compute_time_scale
from .checkpoint import AlignBundle, load_align_bundle, load_stage1_id_maps
from .text_encoder import HashEncoder, SentenceTransformerEncoder
from .tu_cache import TuCacheRow, cache_key, load_tu_cache
from .valid_eval import _build_user_context

logger = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    if name.lower() == "cpu":
        return torch.device("cpu")
    if name.lower() == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlignFullRankRecommender(Recommender):
    """Frozen TiSASRec + alignment MLP fusion; full-catalog scoring."""

    name = "emorecagent_align"

    def __init__(
        self,
        bundle: AlignBundle,
        id_maps: IdMaps,
        tu_cache: dict[str, TuCacheRow],
        *,
        fusion_alpha: float = 0.7,
        tu_mode: str = "cache",
        device: torch.device,
        use_hash_encoder: bool = False,
        stage1_only: bool = False,
    ) -> None:
        self._bundle = bundle
        self._id_maps = id_maps
        self._tu_cache = tu_cache
        self._fusion_alpha = fusion_alpha
        self._tu_mode = tu_mode
        self._device = device
        self._stage1_only = stage1_only
        if use_hash_encoder:
            self._encoder = HashEncoder(dim=bundle.text_encoder_dim)
        else:
            self._encoder = SentenceTransformerEncoder()
        self._user_events: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self._query_ctx: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self._local_to_item = {v: k for k, v in id_maps.item_to_idx.items()}

    @classmethod
    def from_config(
        cls,
        config: Config,
        train: list[Interaction],
        *,
        seed: int = 42,
    ) -> AlignFullRankRecommender:
        del seed
        cfg = config.tisasrec_align
        device = _resolve_device(cfg.device)
        need_alignment = not cfg.stage1_only and cfg.stage2_mode == "fusion"
        bundle = load_align_bundle(
            stage1_ckpt=cfg.stage1_checkpoint_path,
            e_i_path=cfg.e_i_matrix_path,
            alignment_ckpt=(
                cfg.alignment_checkpoint_path if need_alignment else None
            ),
            device=device,
            text_encoder_dim=cfg.text_encoder_dim,
        )
        tu_cache = load_tu_cache(cfg.tu_cache_path)
        id_maps = load_stage1_id_maps(cfg.stage1_checkpoint_path)
        rec = cls(
            bundle,
            id_maps,
            tu_cache,
            fusion_alpha=cfg.fusion_alpha,
            tu_mode=cfg.tu_mode,
            device=device,
            use_hash_encoder=cfg.use_hash_encoder,
            stage1_only=cfg.stage1_only,
        )
        rec.fit(train)
        return rec

    def fit(self, interactions: list[Interaction]) -> AlignFullRankRecommender:
        self._user_events.clear()
        for it in interactions:
            self._user_events[it.user_id].append((it.timestamp, it.item))
        for uid in self._user_events:
            self._user_events[uid].sort(key=lambda t: (t[0], t[1]))
        return self

    def catalog_items(self) -> list[str]:
        """Full Stage 1 item catalog (same order as checkpoint / E_I rows)."""
        return list(self._bundle.item_ids)

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None:
        events = [
            (ts, item)
            for ts, item in self._user_events.get(user_id, [])
            if ts < timestamp_ms
        ]
        history: list[tuple[int, int]] = []
        if events:
            times = [ts for ts, _ in events]
            time_scale = compute_time_scale(
                times, time_unit_seconds=self._bundle.args.time_unit_seconds
            )
            time_min = min(times)
            for ts, item in events:
                i_local = self._id_maps.item_to_idx.get(item)
                if i_local is None:
                    continue
                norm_ts = int(round((ts - time_min) / time_scale) + 1)
                history.append((i_local, norm_ts))
        args = self._bundle.args
        seq, time_mat = _build_user_context(
            history, maxlen=args.maxlen, time_span=args.time_span
        )
        self._query_ctx[user_id] = (seq, time_mat, timestamp_ms)

    def _alpha_eff(self, user_id: str, query_ts_ms: int) -> float:
        if self._stage1_only:
            return 1.0
        row = self._tu_cache.get(cache_key(user_id, query_ts_ms))
        if row is None or not row.has_reviews or not row.T_u.strip():
            return 1.0
        return self._fusion_alpha

    def _p_u(self, user_id: str, query_ts_ms: int) -> torch.Tensor | None:
        if self._tu_mode == "cache":
            row = self._tu_cache.get(cache_key(user_id, query_ts_ms))
            if row is None:
                raise KeyError(
                    f"Missing T_u cache for user={user_id!r} ts={query_ts_ms}; "
                    "run make precompute-tu-emorecagent SPLIT=test"
                )
            if not row.has_reviews or not row.T_u.strip():
                return None
            text = row.T_u
        else:
            row = self._tu_cache.get(cache_key(user_id, query_ts_ms))
            if row is None or not row.T_u:
                return None
            text = row.T_u
        if self._bundle.alignment_mlp is None:
            return None
        t_u = self._encoder.encode([text], device=self._device)
        return self._bundle.alignment_mlp(t_u).squeeze(0)

    def _x_u(self, user_id: str) -> torch.Tensor:
        if user_id not in self._query_ctx:
            raise RuntimeError(
                f"prepare_user_query not called for user_id={user_id!r}"
            )
        seq, time_mat, query_ts = self._query_ctx[user_id]
        seq_t = torch.as_tensor(seq, dtype=torch.long, device=self._device).unsqueeze(0)
        time_t = torch.as_tensor(time_mat, dtype=torch.long, device=self._device).unsqueeze(
            0
        )
        e_i = self._bundle.e_i_matrix
        s_u = self._bundle.tisasrec.user_repr(seq_t, time_t, item_table=e_i).squeeze(0)
        alpha = self._alpha_eff(user_id, query_ts)
        if alpha >= 1.0 - 1e-9:
            return s_u
        p_u = self._p_u(user_id, query_ts)
        if p_u is None:
            return s_u
        return alpha * s_u + (1.0 - alpha) * p_u

    def score(
        self,
        user_id: str,
        candidates: list[str],
        *,
        query_ts_ms: int | None = None,
    ) -> dict[str, float]:
        with torch.no_grad():
            return self._score_impl(user_id, candidates, query_ts_ms=query_ts_ms)

    def _score_impl(
        self,
        user_id: str,
        candidates: list[str],
        *,
        query_ts_ms: int | None = None,
    ) -> dict[str, float]:
        if query_ts_ms is not None:
            self.prepare_user_query(user_id, query_ts_ms)
        x_u = self._x_u(user_id)
        locs = []
        valid_cands = []
        for c in candidates:
            loc = self._id_maps.item_to_idx.get(c)
            if loc is None:
                continue
            locs.append(loc)
            valid_cands.append(c)
        if not locs:
            return {c: 0.0 for c in candidates}
        emb = self._bundle.e_i_matrix[torch.tensor(locs, device=self._device)]
        scores = emb @ x_u
        out = {c: 0.0 for c in candidates}
        for c, s in zip(valid_cands, scores.detach().cpu().tolist()):
            out[c] = float(s)
        return out

    def rank(
        self,
        user_id: str,
        candidates: list[str],
        *,
        query_ts_ms: int | None = None,
    ) -> list[str]:
        scores = self.score(user_id, candidates, query_ts_ms=query_ts_ms)
        return sorted(candidates, key=lambda i: (-scores.get(i, 0.0), i))
