"""RecBole TiSASRec as Yelp_AC Stage-1 backbone (AC-TSR fork).

Amazon / Yelp-review tracks keep the in-repo ERA TiSASRec path. This module
loads a RecBole checkpoint exported into an EmoRecAgent bundle and scores via
the official RecBole ``TiSASRec.forward`` + item embeddings.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..baselines.base import Recommender
from ..config import Config
from ..data.types import Interaction

logger = logging.getLogger(__name__)

_DEFAULT_VENDOR = Path("baseline/RecBole-TiSASRec/vendor")
_BUNDLE_KEYS = (
    "state_dict",
    "item_tokens",
    "n_items",
    "max_seq_length",
    "model_config",
)


def _ensure_recbole_on_path(vendor_root: str | Path) -> Path:
    root = Path(vendor_root)
    if not root.is_absolute():
        # Prefer CWD, then repo root relative to this file.
        candidates = [
            Path.cwd() / root,
            Path(__file__).resolve().parents[3] / root,
        ]
        for cand in candidates:
            if cand.is_dir():
                root = cand.resolve()
                break
        else:
            root = (Path.cwd() / root).resolve()
    vendor = str(root)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return root


def _ts_seconds(raw_ms: int) -> float:
    value = float(raw_ms)
    if value > 1e12:
        return value / 1000.0
    return value


@dataclass(frozen=True)
class RecBoleBundle:
    state_dict: dict[str, torch.Tensor]
    item_tokens: list[str]
    n_items: int
    max_seq_length: int
    model_config: dict[str, Any]


class _StubDataset:
    def __init__(self, n_items: int) -> None:
        self._n_items = int(n_items)

    def num(self, _field: str) -> int:
        return self._n_items


class _DictConfig(dict):
    """Minimal Mapping that RecBole models index with ``config[key]``."""

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        return dict.__getitem__(self, key)


def _default_model_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": 64,
        "inner_size": 256,
        "hidden_dropout_prob": 0.5,
        "attn_dropout_prob": 0.5,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "loss_type": "CE",
        "time_span": 256,
        "MAX_ITEM_LIST_LENGTH": 50,
        "ITEM_ID_FIELD": "item_id",
        "USER_ID_FIELD": "user_id",
        "TIME_FIELD": "timestamp",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "NEG_PREFIX": "neg_",
        "device": "cpu",
    }
    if overrides:
        base.update(overrides)
    return base


def export_recbole_bundle(
    *,
    recbole_checkpoint: str | Path,
    out_path: str | Path,
    vendor_root: str | Path = _DEFAULT_VENDOR,
) -> Path:
    """Load a RecBole ``.pth`` + dataset maps → self-contained ERA bundle."""
    _ensure_recbole_on_path(vendor_root)
    from recbole.quick_start import load_data_and_model  # noqa: WPS433

    ckpt = Path(recbole_checkpoint).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"RecBole checkpoint not found: {ckpt}")

    config, model, dataset, _train, _valid, _test = load_data_and_model(str(ckpt))
    item_field = config["ITEM_ID_FIELD"]
    tokens = list(dataset.field2id_token[item_field])
    n_items = int(dataset.num(item_field))
    if len(tokens) != n_items:
        raise RuntimeError(
            f"item token table length {len(tokens)} != n_items {n_items}"
        )

    model_config = _default_model_config(
        {
            "n_layers": int(config["n_layers"]),
            "n_heads": int(config["n_heads"]),
            "hidden_size": int(config["hidden_size"]),
            "inner_size": int(config["inner_size"]),
            "hidden_dropout_prob": float(config["hidden_dropout_prob"]),
            "attn_dropout_prob": float(config["attn_dropout_prob"]),
            "hidden_act": str(config["hidden_act"]),
            "layer_norm_eps": float(config["layer_norm_eps"]),
            "initializer_range": float(config["initializer_range"]),
            "loss_type": str(config["loss_type"]),
            "time_span": int(config["time_span"]),
            "MAX_ITEM_LIST_LENGTH": int(config["MAX_ITEM_LIST_LENGTH"]),
            "ITEM_ID_FIELD": str(config["ITEM_ID_FIELD"]),
            "USER_ID_FIELD": str(config["USER_ID_FIELD"]),
            "TIME_FIELD": str(config["TIME_FIELD"]),
            "LIST_SUFFIX": str(config["LIST_SUFFIX"]),
            "ITEM_LIST_LENGTH_FIELD": str(config["ITEM_LIST_LENGTH_FIELD"]),
            "NEG_PREFIX": str(config["NEG_PREFIX"]),
        }
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "item_tokens": tokens,
        "n_items": n_items,
        "max_seq_length": int(config["MAX_ITEM_LIST_LENGTH"]),
        "model_config": model_config,
        "source_checkpoint": str(ckpt),
        "dataset": str(config["dataset"]),
    }
    torch.save(payload, out)
    logger.info(
        "exported RecBole Stage-1 bundle → %s (n_items=%s maxlen=%s)",
        out,
        n_items,
        model_config["MAX_ITEM_LIST_LENGTH"],
    )
    return out


def load_recbole_bundle(path: str | Path) -> RecBoleBundle:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    missing = [k for k in _BUNDLE_KEYS if k not in payload]
    if missing:
        raise ValueError(f"RecBole bundle missing keys {missing}: {path}")
    return RecBoleBundle(
        state_dict=dict(payload["state_dict"]),
        item_tokens=[str(t) for t in payload["item_tokens"]],
        n_items=int(payload["n_items"]),
        max_seq_length=int(payload["max_seq_length"]),
        model_config=dict(payload["model_config"]),
    )


def _build_model(
    bundle: RecBoleBundle,
    *,
    device: torch.device,
    vendor_root: str | Path,
) -> torch.nn.Module:
    _ensure_recbole_on_path(vendor_root)
    from recbole.model.sequential_recommender.tisasrec import TiSASRec  # noqa: WPS433

    cfg = _DictConfig(_default_model_config(bundle.model_config))
    cfg["device"] = str(device)
    model = TiSASRec(cfg, _StubDataset(bundle.n_items))
    model.load_state_dict(bundle.state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _resolve_device(name: str) -> torch.device:
    if name.lower() == "cpu":
        return torch.device("cpu")
    if name.lower() == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecBoleStage1Recommender(Recommender):
    """Frozen RecBole TiSASRec full-catalog scorer (Yelp_AC Stage-1 backend)."""

    name = "emorecagent_align"

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        item_tokens: list[str],
        max_seq_length: int,
        device: torch.device,
    ) -> None:
        self._model = model
        self._item_tokens = list(item_tokens)
        self._item_to_idx = {
            tok: i for i, tok in enumerate(self._item_tokens) if tok != "[PAD]"
        }
        # RecBole pad token at index 0 is usually "[PAD]".
        self._max_seq_length = int(max_seq_length)
        self._device = device
        self._user_events: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self._query_ctx: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

    @classmethod
    def from_config(
        cls,
        config: Config,
        train: list[Interaction],
        *,
        seed: int = 42,
    ) -> RecBoleStage1Recommender:
        del seed
        cfg = config.tisasrec_align
        device = _resolve_device(cfg.device)
        bundle = load_recbole_bundle(cfg.recbole_bundle_path)
        model = _build_model(
            bundle,
            device=device,
            vendor_root=cfg.recbole_vendor_root,
        )
        rec = cls(
            model=model,
            item_tokens=bundle.item_tokens,
            max_seq_length=bundle.max_seq_length,
            device=device,
        )
        rec.fit(train)
        return rec

    def fit(self, interactions: list[Interaction]) -> RecBoleStage1Recommender:
        self._user_events.clear()
        for it in interactions:
            self._user_events[it.user_id].append((it.timestamp, it.item))
        for uid in self._user_events:
            self._user_events[uid].sort(key=lambda t: (t[0], t[1]))
        return self

    def catalog_items(self) -> list[str]:
        return [t for t in self._item_tokens if t != "[PAD]"]

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None:
        events = [
            (ts, item)
            for ts, item in self._user_events.get(user_id, [])
            if ts < timestamp_ms
        ]
        item_ids: list[int] = []
        time_vals: list[float] = []
        for ts, item in events:
            loc = self._item_to_idx.get(item)
            if loc is None:
                continue
            item_ids.append(int(loc))
            time_vals.append(_ts_seconds(ts))
        if len(item_ids) > self._max_seq_length:
            item_ids = item_ids[-self._max_seq_length :]
            time_vals = time_vals[-self._max_seq_length :]
        seq = np.zeros(self._max_seq_length, dtype=np.int64)
        times = np.zeros(self._max_seq_length, dtype=np.float64)
        length = len(item_ids)
        if length:
            seq[:length] = np.asarray(item_ids, dtype=np.int64)
            times[:length] = np.asarray(time_vals, dtype=np.float64)
        self._query_ctx[user_id] = (seq, times, length)

    def _full_scores(self, user_id: str) -> torch.Tensor:
        if user_id not in self._query_ctx:
            raise RuntimeError(
                f"prepare_user_query not called for user_id={user_id!r}"
            )
        seq, times, length = self._query_ctx[user_id]
        if length <= 0:
            # No history → uniform zeros (unknown user prefix).
            return torch.zeros(len(self._item_tokens), device=self._device)
        item_seq = torch.as_tensor(seq, dtype=torch.long, device=self._device).unsqueeze(
            0
        )
        time_seq = torch.as_tensor(
            times, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        item_seq_len = torch.tensor([length], dtype=torch.long, device=self._device)
        time_matrix = self._model.get_time_matrix(time_seq)
        with torch.no_grad():
            seq_out = self._model.forward(item_seq, item_seq_len, time_matrix)
            emb = self._model.item_embedding.weight
            scores = torch.matmul(seq_out, emb.transpose(0, 1)).squeeze(0)
        return scores

    def score(
        self,
        user_id: str,
        candidates: list[str],
        *,
        query_ts_ms: int | None = None,
    ) -> dict[str, float]:
        if query_ts_ms is not None:
            self.prepare_user_query(user_id, query_ts_ms)
        scores = self._full_scores(user_id)
        out = {c: 0.0 for c in candidates}
        for c in candidates:
            loc = self._item_to_idx.get(c)
            if loc is None:
                continue
            out[c] = float(scores[loc].item())
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
