"""Training data helpers for HetTiSASRec."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..data.types import Interaction
from ..sequential.id_maps import IdMaps
from ..sequential.seq_utils import compute_repos


@dataclass(frozen=True, slots=True)
class ValidEvalCase:
    """One chronological valid target with history strictly before the event."""

    user_local: int
    gold_local: int
    history: tuple[tuple[int, int], ...]


def _normalize_user_events(
    events: list[tuple[int, str]],
    id_maps: IdMaps,
) -> list[tuple[int, int]]:
    """Chronological (item_local, norm_ts) for one user."""
    events = sorted(events, key=lambda t: (t[0], t[1]))
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
        i_local = id_maps.item_to_idx.get(item)
        if i_local is None:
            continue
        norm_ts = int(round((ts - time_min) / time_scale) + 1)
        out.append((i_local, norm_ts))
    return out


def build_user_sequences(
    interactions: list[Interaction],
    id_maps: IdMaps,
) -> dict[int, list[tuple[int, int]]]:
    """Map user local id -> [(item_local, norm_ts), ...] chronological."""
    per_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for it in interactions:
        per_user[it.user_id].append((it.timestamp, it.item))

    out: dict[int, list[tuple[int, int]]] = {}
    for uid, events in per_user.items():
        u_local = id_maps.user_to_idx.get(uid)
        if u_local is None:
            continue
        seq = _normalize_user_events(events, id_maps)
        if len(seq) >= 2:
            out[u_local] = seq
    return out


def build_valid_eval_cases(
    train: list[Interaction],
    valid: list[Interaction],
    id_maps: IdMaps,
) -> list[ValidEvalCase]:
    """Build per-event valid cases: history = train + earlier valid only."""
    per_user: dict[str, list[tuple[int, str, bool]]] = defaultdict(list)
    for it in train:
        per_user[it.user_id].append((it.timestamp, it.item, False))
    for it in valid:
        per_user[it.user_id].append((it.timestamp, it.item, True))

    cases: list[ValidEvalCase] = []
    for uid, raw_events in per_user.items():
        u_local = id_maps.user_to_idx.get(uid)
        if u_local is None:
            continue
        raw_events.sort(key=lambda t: (t[0], t[1]))
        times = [ts for ts, _, _ in raw_events]
        time_diffs = {
            times[i + 1] - times[i]
            for i in range(len(times) - 1)
            if times[i + 1] - times[i] != 0
        }
        time_scale = min(time_diffs) if time_diffs else 1
        time_min = min(times) if times else 0

        history: list[tuple[int, int]] = []
        for ts, item, is_valid in raw_events:
            i_local = id_maps.item_to_idx.get(item)
            if i_local is None:
                continue
            norm_ts = int(round((ts - time_min) / time_scale) + 1)
            if is_valid and history:
                cases.append(ValidEvalCase(u_local, i_local, tuple(history)))
            history.append((i_local, norm_ts))
    return cases


def build_test_eval_cases(
    train: list[Interaction],
    valid: list[Interaction],
    test: list[Interaction],
    id_maps: IdMaps,
) -> list[ValidEvalCase]:
    """Build per-event test cases: history = train + valid + earlier test only."""
    per_user: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for it in train:
        per_user[it.user_id].append((it.timestamp, it.item, "train"))
    for it in valid:
        per_user[it.user_id].append((it.timestamp, it.item, "valid"))
    for it in test:
        per_user[it.user_id].append((it.timestamp, it.item, "test"))

    cases: list[ValidEvalCase] = []
    for uid, raw_events in per_user.items():
        u_local = id_maps.user_to_idx.get(uid)
        if u_local is None:
            continue
        raw_events.sort(key=lambda t: (t[0], t[1]))
        times = [ts for ts, _, _ in raw_events]
        time_diffs = {
            times[i + 1] - times[i]
            for i in range(len(times) - 1)
            if times[i + 1] - times[i] != 0
        }
        time_scale = min(time_diffs) if time_diffs else 1
        time_min = min(times) if times else 0

        history: list[tuple[int, int]] = []
        for ts, item, split in raw_events:
            i_local = id_maps.item_to_idx.get(item)
            if i_local is None:
                continue
            norm_ts = int(round((ts - time_min) / time_scale) + 1)
            if split == "test" and history:
                cases.append(ValidEvalCase(u_local, i_local, tuple(history)))
            history.append((i_local, norm_ts))
    return cases


def build_valid_pairs(
    valid: list[Interaction],
    id_maps: IdMaps,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for it in valid:
        u = id_maps.user_to_idx.get(it.user_id)
        i = id_maps.item_to_idx.get(it.item)
        if u is not None and i is not None:
            pairs.append((u, i))
    return pairs


def build_train_pairs(
    train: list[Interaction],
    id_maps: IdMaps,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for it in train:
        u = id_maps.user_to_idx.get(it.user_id)
        i = id_maps.item_to_idx.get(it.item)
        if u is not None and i is not None:
            pairs.append((u, i))
    return pairs


def random_neq(low: int, high: int, forbidden: set[int]) -> int:
    t = random.randint(low, high - 1)
    while t in forbidden:
        t = random.randint(low, high - 1)
    return t


def sample_batch(
    user_train: dict[int, list[tuple[int, int]]],
    *,
    item_num: int,
    maxlen: int,
    time_span: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    users: list[int] = []
    seqs: list[np.ndarray] = []
    time_mats: list[np.ndarray] = []
    poss: list[np.ndarray] = []
    negs: list[np.ndarray] = []

    user_ids = list(user_train.keys())
    for _ in range(batch_size):
        user = random.choice(user_ids)
        history = user_train[user]
        if len(history) <= 1:
            continue

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = history[-1][0]
        seen = {i for i, _ in history}
        idx = maxlen - 1
        for item_local, norm_ts in reversed(history[:-1]):
            seq[idx] = item_local
            time_seq[idx] = norm_ts
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, item_num + 1, seen)
            nxt = item_local
            idx -= 1
            if idx == -1:
                break

        users.append(user)
        seqs.append(seq)
        time_mats.append(compute_repos(time_seq, time_span))
        poss.append(pos)
        negs.append(neg)

    if not users:
        raise RuntimeError("empty training batch (no users with history>1)")

    return (
        np.array(users, dtype=np.int32),
        np.stack(seqs),
        np.stack(time_mats),
        np.stack(poss),
        np.stack(negs),
    )
