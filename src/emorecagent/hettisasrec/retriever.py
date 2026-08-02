"""HetTiSASRec retriever for the EmoRecAgent graph pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..absa.normalize import normalize_aspect
from ..config import Config
from ..data.types import Interaction
from ..sequential.id_maps import IdMaps, build_id_maps_from_interactions
from ..sequential.seq_utils import compute_repos
from .aspect_graph import AspectGraphBundle
from .aspect_vocab import AspectVocab, load_aspect_vocab
from .model import HetTiSASRecArgs, HetTiSASRecModel


def _resolve_device(preference: str) -> torch.device:
    """Pick retriever device from config (``auto`` → CUDA when available)."""
    pref = preference.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("hettisasrec.device=cuda but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Peak VRAM for ``forward_all_items()`` (full-graph aspect MP); observed ~6.3 GiB.
_CATALOG_BUILD_MIN_FREE_GB = 6.5

_log = logging.getLogger("emorecagent.hettisasrec")


class HetTiSASRecRetriever:
    """Sequential retriever with item–aspect enrichment and gamma support."""

    backend = "hettisasrec"

    def __init__(
        self,
        model: HetTiSASRecModel,
        id_maps: IdMaps,
        graph: AspectGraphBundle,
        aspect_vocab: AspectVocab | None,
        device: torch.device,
        args: HetTiSASRecArgs,
        *,
        pool_size: int = 50,
    ) -> None:
        self._model = model
        self._id_maps = id_maps
        self._graph = graph
        self._aspect_vocab = aspect_vocab
        self._device = device
        self._args = args
        self.pool_size = pool_size
        self._items = list(graph.item_ids)
        self._aspect_idx = {a: i + 1 for i, a in enumerate(graph.aspect_ids)}
        self._user_history: dict[str, list[tuple[int, str]]] = {}
        self._query_context: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._catalog_cache: torch.Tensor | None = None
        self._model.eval()

    @property
    def id_maps(self) -> IdMaps:
        return self._id_maps

    @property
    def model(self) -> HetTiSASRecModel:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def item_ids(self) -> list[str]:
        return self._items

    @classmethod
    def from_config(
        cls,
        config: Config,
        train: list[Interaction],
        *,
        seed: int = 42,
    ) -> HetTiSASRecRetriever:
        del seed
        cfg = config.hettisasrec
        ckpt_path = Path(cfg.checkpoint_path)
        graph_path = Path(cfg.aspect_graph_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"HetTiSASRec checkpoint not found: {ckpt_path}. "
                "Run `make train-hettisasrec` first."
            )
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Aspect graph not found: {graph_path}. "
                "Run `make train-hettisasrec` first."
            )

        device = _resolve_device(cfg.device)

        graph = AspectGraphBundle.load(graph_path)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta = dict(payload["meta"])
        args = HetTiSASRecArgs(**dict(meta["args"]))
        model = HetTiSASRecModel(
            int(meta["user_num"]),
            int(meta["item_num"]),
            args,
            graph,
        )
        model.load_state_dict(payload["model"])
        model.to(device)

        id_maps = IdMaps(
            user_to_idx=meta["id_maps"]["user_to_idx"],
            item_to_idx=meta["id_maps"]["item_to_idx"],
        )
        vocab_path = Path(cfg.aspect_vocab_path)
        aspect_vocab = load_aspect_vocab(vocab_path) if vocab_path.exists() else None

        return cls(
            model,
            id_maps,
            graph,
            aspect_vocab,
            device,
            args,
            pool_size=cfg.pool_size,
        ).fit(train)

    def fit(self, interactions: list[Interaction]) -> HetTiSASRecRetriever:
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
    ) -> tuple[np.ndarray, np.ndarray]:
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
        return seq, compute_repos(time_seq, self._args.time_span)

    def _context_for_user(self, user_id: str) -> tuple[np.ndarray, np.ndarray]:
        if user_id in self._query_context:
            return self._query_context[user_id]
        return self._build_context(self._user_history.get(user_id, []))

    def _catalog(self) -> torch.Tensor:
        if self._catalog_cache is None:
            self._catalog_cache = self._build_catalog_table()
        return self._catalog_cache

    def _build_catalog_table(self) -> torch.Tensor:
        """Build full item embedding table; spill to CPU if GPU VRAM is tight."""
        with torch.no_grad():
            if self._device.type != "cuda":
                return self._model.all_item_embeddings()
            free_bytes, _total = torch.cuda.mem_get_info()
            if free_bytes >= int(_CATALOG_BUILD_MIN_FREE_GB * 1024**3):
                try:
                    return self._model.all_item_embeddings()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
            return self._build_catalog_table_on_cpu()

    def _build_catalog_table_on_cpu(self) -> torch.Tensor:
        """One-time catalog build on CPU; model returns to CUDA for scoring."""
        _log.info(
            "building HetTiSASRec item catalog on CPU "
            "(insufficient GPU VRAM for full-graph pass; inference stays on %s)",
            self._device,
        )
        self._model.cpu()
        try:
            table = self._model.all_item_embeddings()
        finally:
            self._model.to(self._device)
        return table.to(self._device)

    def _aspect_gamma_tensor(
        self, gammas: dict[str, float] | None
    ) -> torch.Tensor | None:
        if not gammas:
            return None
        asp_table = self._model.aspect_embeddings()
        delta = torch.zeros(
            asp_table.shape[1], device=self._device, dtype=asp_table.dtype
        )
        for aspect, gamma in gammas.items():
            if abs(gamma) < 1e-12:
                continue
            key = normalize_aspect(aspect)
            ai = self._aspect_idx.get(key)
            if ai is None and self._aspect_vocab is not None:
                other = self._aspect_vocab.aspects[self._aspect_vocab.other_id]
                ai = self._aspect_idx.get(other)
            if ai is None:
                continue
            delta = delta + gamma * asp_table[ai]
        if delta.abs().sum() < 1e-12:
            return None
        return delta.unsqueeze(0)

    def _raw_scores(
        self,
        user_id: str,
        candidates: list[str],
        gammas: dict[str, float] | None = None,
    ) -> dict[str, float]:
        seq, time_mat = self._context_for_user(user_id)
        if not np.any(seq):
            return {item: 0.0 for item in candidates}

        item_indices: list[int] = []
        valid_candidates: list[str] = []
        for item in candidates:
            idx = self._id_maps.item_to_idx.get(item)
            if idx is not None:
                item_indices.append(idx)
                valid_candidates.append(item)

        if not item_indices:
            return {item: 0.0 for item in candidates}

        aspect_gamma = self._aspect_gamma_tensor(gammas)
        with torch.no_grad():
            logits = self._model.predict(
                torch.as_tensor(seq, dtype=torch.long, device=self._device).unsqueeze(0),
                torch.as_tensor(time_mat, dtype=torch.long, device=self._device).unsqueeze(0),
                torch.tensor([item_indices], dtype=torch.long, device=self._device),
                item_table=self._catalog(),
                aspect_gamma=aspect_gamma,
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
        raw = self._raw_scores(user_id, item_ids, gammas)
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
        exclude = exclude or set()
        pool = [c for c in candidates if c not in exclude]
        if len(pool) <= k:
            return list(pool)
        raw = self._raw_scores(user_id, pool, gammas)
        ranked = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))
        return [item for item, _ in ranked[:k]]


def build_hettisasrec_retriever(
    config: Config,
    *,
    seed: int = 42,
    train_interactions: list[Interaction] | None = None,
) -> HetTiSASRecRetriever:
    if train_interactions is None:
        from .train import load_train_valid_from_config

        train, _ = load_train_valid_from_config(config.data.out_dir)
        train_interactions = train
    return HetTiSASRecRetriever.from_config(
        config, train_interactions, seed=seed
    )
