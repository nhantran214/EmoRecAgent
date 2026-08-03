"""Build train-scoped heterogeneous graph from interactions + ABSA cache."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..absa.normalize import normalize_aspect
from ..data.review_index import build_review_index_from_scope, build_train_scope
from ..data.types import Interaction
from ..eval.runner import load_split_jsonl
from ..scoring.dynamic_weights import AspectSignal, aspect_gammas
from .aspect_vocab import (
    AspectVocab,
    build_aspect_vocab,
    open_absa_cache_readonly,
    save_aspect_vocab,
)
from .features import TextEncoder, build_text_encoder
from .graph_data import HgtGraphBundle
from .schema import NodeType, RelationType
from .temporal import (
    RTE_MAX_LEN,
    assert_rte_edge_time,
    build_node_day_times,
    edge_time_from_nodes,
    timestamp_to_day_index,
)


@dataclass(frozen=True, slots=True)
class GraphBuildStats:
    n_users: int
    n_items: int
    n_aspects: int
    n_edges: int
    n_train_pairs: int
    n_valid_pairs: int


def _load_item_meta(meta_path: Path, item_ids: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    if not meta_path.exists():
        return texts
    with meta_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = str(row.get("parent_asin") or row.get("asin") or "")
            if asin not in item_ids:
                continue
            title = str(row.get("title") or "")
            desc = str(row.get("description") or "")
            if isinstance(desc, list):
                desc = " ".join(str(x) for x in desc)
            texts[asin] = f"{title}. {desc}".strip()
    return texts


def _load_user_review_texts(
    raw_review_path: Path,
    scope_users: set[str],
    scope_items: set[str],
    cutoff_ts: int,
) -> dict[str, list[str]]:
    per_user: dict[str, list[str]] = defaultdict(list)
    if not raw_review_path.exists():
        return per_user
    with raw_review_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(row.get("user_id") or "")
            item = str(row.get("parent_asin") or row.get("asin") or "")
            ts = int(row.get("timestamp") or 0)
            if uid not in scope_users or item not in scope_items or ts > cutoff_ts:
                continue
            text = str(row.get("text") or row.get("review_text") or "").strip()
            if text:
                per_user[uid].append(text)
    return per_user


def _item_aspect_scores_from_cache(
    cache_path: Path,
    review_index: dict[str, tuple[str, str, int]],
) -> dict[str, dict[str, float]]:
    conn = open_absa_cache_readonly(cache_path)
    item_aspects: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
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


def _user_pref_edges(
    train: list[Interaction],
    review_index: dict[str, tuple[str, str, int]],
    cache_path: Path,
    vocab: AspectVocab,
    lambda_decay: float,
) -> list[tuple[int, int, float, int]]:
    """Return (user_local, aspect_local, weight, ts) for prefers edges."""
    conn = open_absa_cache_readonly(cache_path)
    user_signals: dict[str, list[AspectSignal]] = defaultdict(list)
    sentiment_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    try:
        rows = conn.execute("SELECT review_id, triples_json FROM absa_cache").fetchall()
    finally:
        conn.close()
    idx_set = set(review_index)
    for review_id, payload in rows:
        if review_id not in idx_set:
            continue
        uid, _, ts = review_index[review_id]
        data = json.loads(payload)
        for t in data.get("triples", []):
            aspect = normalize_aspect(str(t.get("aspect", "")))
            if not aspect:
                continue
            pol = sentiment_map.get(str(t.get("sentiment", "neutral")), 0.0)
            user_signals[uid].append(AspectSignal(aspect, pol, ts))

    user_ids = sorted({it.user_id for it in train})
    user_idx = {u: i for i, u in enumerate(user_ids)}
    cutoff = max(it.timestamp for it in train) if train else 0
    edges: list[tuple[int, int, float, int]] = []
    for uid, signals in user_signals.items():
        if uid not in user_idx:
            continue
        gammas = aspect_gammas(signals, cutoff + 1, lambda_decay)
        for aspect, weight in gammas.items():
            if abs(weight) < 1e-9:
                continue
            asp_local = vocab.id_for(aspect)
            edges.append((user_idx[uid], asp_local, float(weight), cutoff))
    return edges


def build_hgt_graph(
    *,
    train: list[Interaction],
    valid: list[Interaction] | None = None,
    cache_path: str | Path,
    meta_path: str | Path,
    raw_review_path: str | Path,
    aspect_top_k: int = 100,
    min_aspect_support: int = 5,
    text_encoder: str = "hash",
    feature_dim: int = 64,
    seed: int = 42,
    lambda_decay: float = 0.01,
    max_users: int | None = None,
    max_items: int | None = None,
) -> tuple[HgtGraphBundle, AspectVocab, GraphBuildStats]:
    """Materialize graph tensors from train split + read-only ABSA cache."""
    del valid  # valid pairs built from valid interactions argument separately
    cache_path = Path(cache_path)
    vocab = build_aspect_vocab(
        cache_path,
        top_k=aspect_top_k,
        min_support=min_aspect_support,
    )

    user_ids = sorted({it.user_id for it in train})
    item_ids = sorted({it.item for it in train})
    if max_users is not None:
        user_ids = user_ids[:max_users]
        allowed = set(user_ids)
        train = [it for it in train if it.user_id in allowed]
    if max_items is not None:
        item_ids = item_ids[:max_items]
        allowed_items = set(item_ids)
        train = [it for it in train if it.item in allowed_items]

    aspect_ids = list(vocab.aspects)
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {it: i for i, it in enumerate(item_ids)}

    scope = build_train_scope(train)
    review_index = build_review_index_from_scope(train, raw_review_path)
    item_aspect_scores = _item_aspect_scores_from_cache(cache_path, review_index)
    user_texts = _load_user_review_texts(
        Path(raw_review_path),
        set(user_ids),
        set(item_ids),
        scope.cutoff_ts,
    )
    item_texts = _load_item_meta(Path(meta_path), set(item_ids))

    encoder: TextEncoder = build_text_encoder(text_encoder, dim=feature_dim, seed=seed)
    in_dim = encoder.dim

    n_u, n_i, n_a = len(user_ids), len(item_ids), len(aspect_ids)
    n_nodes = n_u + n_i + n_a
    node_type = np.zeros(n_nodes, dtype=np.int64)
    node_type[:n_u] = NodeType.USER
    node_type[n_u : n_u + n_i] = NodeType.ITEM
    node_type[n_u + n_i :] = NodeType.ASPECT

    user_feat_texts = [
        " ".join(user_texts.get(u, [])) or u for u in user_ids
    ]
    item_feat_texts = [item_texts.get(it, it) for it in item_ids]
    aspect_feat_texts = list(aspect_ids)

    feats = np.vstack(
        [
            encoder.encode(user_feat_texts),
            encoder.encode(item_feat_texts),
            encoder.encode(aspect_feat_texts),
        ]
    ).astype(np.float32)
    if feats.shape[1] != in_dim:
        raise RuntimeError("encoder dim mismatch")

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_rel: list[int] = []

    def _add_edge(src: int, dst: int, rel: RelationType) -> None:
        edge_src.append(src)
        edge_dst.append(dst)
        edge_rel.append(int(rel))

    item_off = n_u
    aspect_off = n_u + n_i

    min_ts = min(it.timestamp for it in train) if train else 0
    cutoff_day = timestamp_to_day_index(scope.cutoff_ts, min_ts)
    train_day_stamps: list[tuple[int, int, int]] = []

    for it in train:
        u = user_idx[it.user_id]
        i = item_idx[it.item]
        day = timestamp_to_day_index(it.timestamp, min_ts)
        train_day_stamps.append((u, i, day))
        g_u, g_i = u, item_off + i
        _add_edge(g_u, g_i, RelationType.BUYS)
        _add_edge(g_i, g_u, RelationType.BOUGHT_BY)

    for item_id, asp_scores in item_aspect_scores.items():
        if item_id not in item_idx:
            continue
        g_i = item_off + item_idx[item_id]
        for aspect, score in asp_scores.items():
            a_local = vocab.id_for(aspect)
            g_a = aspect_off + a_local
            _add_edge(g_i, g_a, RelationType.HAS_ASPECT)
            _add_edge(g_a, g_i, RelationType.APPEARS_IN)

    for u_local, a_local, weight, _ts in _user_pref_edges(
        train, review_index, cache_path, vocab, lambda_decay
    ):
        g_u = u_local
        g_a = aspect_off + a_local
        _add_edge(g_u, g_a, RelationType.PREFERS)
        _add_edge(g_a, g_u, RelationType.PREFERRED_BY)

    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    edge_type = np.array(edge_rel, dtype=np.int64)
    node_day_times = build_node_day_times(
        train_day_stamps,
        n_users=n_u,
        n_items=n_i,
        n_aspects=n_a,
        cutoff_day=cutoff_day,
    )
    edge_t = edge_time_from_nodes(edge_index, node_day_times, max_len=RTE_MAX_LEN)
    assert_rte_edge_time(edge_t)

    train_pairs = [(user_idx[it.user_id], item_idx[it.item]) for it in train]
    valid_pairs: list[tuple[int, int]] = []

    meta = {
        "text_encoder": text_encoder,
        "feature_dim": in_dim,
        "seed": seed,
        "lambda_decay": lambda_decay,
        "cutoff_ts": scope.cutoff_ts,
        "min_ts": min_ts,
        "rte_max_len": RTE_MAX_LEN,
    }
    bundle = HgtGraphBundle(
        node_feature=feats,
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_time=edge_t,
        user_ids=user_ids,
        item_ids=item_ids,
        aspect_ids=aspect_ids,
        train_pairs=train_pairs,
        valid_pairs=valid_pairs,
        aspect_vocab=vocab.to_dict(),
        meta=meta,
    )
    stats = GraphBuildStats(
        n_users=n_u,
        n_items=n_i,
        n_aspects=n_a,
        n_edges=int(edge_index.shape[1]),
        n_train_pairs=len(train_pairs),
        n_valid_pairs=len(valid_pairs),
    )
    return bundle, vocab, stats


def build_and_save_hgt_graph(
    *,
    train_path: str | Path,
    valid_path: str | Path | None,
    cache_path: str | Path,
    meta_path: str | Path,
    raw_review_path: str | Path,
    graph_path: str | Path,
    aspect_vocab_path: str | Path,
    **kwargs,
) -> GraphBuildStats:
    train = load_split_jsonl(train_path)
    valid = load_split_jsonl(valid_path) if valid_path else []
    bundle, vocab, stats = build_hgt_graph(
        train=train,
        valid=valid,
        cache_path=cache_path,
        meta_path=meta_path,
        raw_review_path=raw_review_path,
        **kwargs,
    )
    if valid:
        user_idx = {u: i for i, u in enumerate(bundle.user_ids)}
        item_idx = {it: i for i, it in enumerate(bundle.item_ids)}
        bundle.valid_pairs = [
            (user_idx[it.user_id], item_idx[it.item])
            for it in valid
            if it.user_id in user_idx and it.item in item_idx
        ]
    save_aspect_vocab(vocab, aspect_vocab_path)
    bundle.save(graph_path)
    return GraphBuildStats(
        n_users=stats.n_users,
        n_items=stats.n_items,
        n_aspects=stats.n_aspects,
        n_edges=stats.n_edges,
        n_train_pairs=stats.n_train_pairs,
        n_valid_pairs=len(bundle.valid_pairs),
    )
