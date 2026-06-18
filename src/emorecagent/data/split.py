"""Sampling and chronological splitting (no random split — avoids temporal leakage).

Default protocol: per user, sort interactions by time and assign
80% earliest → train, next 10% → validation, last 10% → test.
Users without enough history for all three partitions are train-only.

`leave_last_out` is retained for legacy/smoke tests (1 valid + 1 test per user).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .kcore import k_core_filter
from .types import Interaction


@dataclass
class Split:
    train: list[Interaction]
    valid: list[Interaction]
    test: list[Interaction]
    manifest: dict = field(default_factory=dict)


def _group_by_user(interactions: list[Interaction]) -> dict[str, list[Interaction]]:
    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for it in interactions:
        by_user[it.user_id].append(it)
    # Deterministic order: timestamp, then item id as tiebreak.
    for items in by_user.values():
        items.sort(key=lambda x: (x.timestamp, x.item))
    return by_user


def _cap_entities_by_degree(
    entities: list[str],
    degrees: Counter[str],
    cap: int,
    rng: random.Random,
) -> set[str]:
    """Keep the highest-degree entities; break ties with a seeded shuffle."""
    if len(entities) <= cap:
        return set(entities)
    by_degree: dict[int, list[str]] = defaultdict(list)
    for entity in entities:
        by_degree[degrees[entity]].append(entity)
    ranked: list[str] = []
    for degree in sorted(by_degree.keys(), reverse=True):
        band = sorted(by_degree[degree])
        rng.shuffle(band)
        ranked.extend(band)
    return set(ranked[:cap])


def _cap_interactions(
    interactions: list[Interaction],
    *,
    key: str,
    cap: int | None,
    rng: random.Random,
) -> list[Interaction]:
    if cap is None:
        return interactions
    if key == "user":
        entities = sorted({it.user_id for it in interactions})
        degrees = Counter(it.user_id for it in interactions)
        keep = _cap_entities_by_degree(entities, degrees, cap, rng)
        return [it for it in interactions if it.user_id in keep]
    entities = sorted({it.item for it in interactions})
    degrees = Counter(it.item for it in interactions)
    keep = _cap_entities_by_degree(entities, degrees, cap, rng)
    return [it for it in interactions if it.item in keep]


def sample_subset(
    interactions: list[Interaction],
    k_core: int,
    max_users: int | None,
    max_items: int | None,
    seed: int,
) -> list[Interaction]:
    """Restrict to a capped user/item subset, then restore the k-core property.

  Caps prefer high-degree users/items (seeded tie-break) so a random item draw
  does not strip the hubs that keep everyone above ``k_core`` on sparse graphs.
  Items are capped before users; k-core is re-applied after each cap.
    """
    current = k_core_filter(interactions, k_core)
    if max_users is None and max_items is None:
        return current

    rng = random.Random(seed)

    if max_items is not None:
        current = k_core_filter(
            _cap_interactions(current, key="item", cap=max_items, rng=rng),
            k_core,
        )
    if max_users is not None:
        current = k_core_filter(
            _cap_interactions(current, key="user", cap=max_users, rng=rng),
            k_core,
        )
    return k_core_filter(current, k_core)


def _partition_sizes(
    n: int, train_ratio: float, valid_ratio: float
) -> tuple[int, int, int]:
    """Integer partition of n interactions into train / valid / test counts."""
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid
    return n_train, n_valid, n_test


def chronological_split(
    interactions: list[Interaction],
    *,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_history: int = 0,
) -> Split:
    """Per-user chronological ratio split (default 80% / 10% / 10%).

    Interactions are never shuffled randomly. For each user, timestamps increase
    monotonically across train → valid → test, so future interactions cannot
    leak into profile building or parameter tuning.
    """
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    by_user = _group_by_user(interactions)
    train: list[Interaction] = []
    valid: list[Interaction] = []
    test: list[Interaction] = []
    n_test_users = 0

    for items in by_user.values():
        n = len(items)
        n_train, n_valid, n_test = _partition_sizes(n, train_ratio, valid_ratio)
        eligible = (
            n_train >= min_history and n_valid >= 1 and n_test >= 1
        )
        if eligible:
            train.extend(items[:n_train])
            valid.extend(items[n_train : n_train + n_valid])
            test.extend(items[n_train + n_valid :])
            n_test_users += 1
        else:
            train.extend(items)

    test_ts = [it.timestamp for it in test]
    valid_ts = [it.timestamp for it in valid]
    train_ts = [it.timestamp for it in train]
    manifest = {
        "split_method": "chronological_ratio",
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "n_interactions": len(interactions),
        "n_users": len(by_user),
        "n_items": len({it.item for it in interactions}),
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_test_users": n_test_users,
        "min_history": min_history,
        "global_cutoff_max_train_ts": max(train_ts) if train_ts else None,
        "global_cutoff_min_valid_ts": min(valid_ts) if valid_ts else None,
        "global_cutoff_max_valid_ts": max(valid_ts) if valid_ts else None,
        "global_cutoff_min_test_ts": min(test_ts) if test_ts else None,
        "global_cutoff_max_test_ts": max(test_ts) if test_ts else None,
    }
    return Split(train=train, valid=valid, test=test, manifest=manifest)


def leave_last_out(
    interactions: list[Interaction],
    min_history: int = 0,
) -> Split:
    """Chronological leave-last-out split.

    A user is eval-eligible only if, after holding out test + valid, the
    remaining train history has at least `min_history` interactions (the
    dynamic-preference effect cannot exist for near-cold users). Ineligible
    users contribute all their interactions to train only.
    """
    by_user = _group_by_user(interactions)
    train: list[Interaction] = []
    valid: list[Interaction] = []
    test: list[Interaction] = []
    n_test_users = 0

    for items in by_user.values():
        eligible = len(items) >= 3 and (len(items) - 2) >= min_history
        if eligible:
            train.extend(items[:-2])
            valid.append(items[-2])
            test.append(items[-1])
            n_test_users += 1
        else:
            train.extend(items)

    test_ts = [it.timestamp for it in test]
    manifest = {
        "split_method": "leave_last_out",
        "n_interactions": len(interactions),
        "n_users": len(by_user),
        "n_items": len({it.item for it in interactions}),
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_test_users": n_test_users,
        "min_history": min_history,
        "global_cutoff_min_test_ts": min(test_ts) if test_ts else None,
        "global_cutoff_max_test_ts": max(test_ts) if test_ts else None,
    }
    return Split(train=train, valid=valid, test=test, manifest=manifest)


def _write_jsonl(path: Path, interactions: list[Interaction]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for it in interactions:
            fh.write(
                json.dumps(
                    {
                        "user_id": it.user_id,
                        "item": it.item,
                        "rating": it.rating,
                        "timestamp": it.timestamp,
                        "helpful_vote": it.helpful_vote,
                        "verified_purchase": it.verified_purchase,
                    }
                )
                + "\n"
            )


def write_split(
    out_dir: str | Path,
    split: Split,
    seed: int,
    k_core: int,
    *,
    extra_manifest: dict | None = None,
) -> Path:
    """Write train/valid/test JSONL files and a manifest.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", split.train)
    _write_jsonl(out / "valid.jsonl", split.valid)
    _write_jsonl(out / "test.jsonl", split.test)
    pct_verified_test = (
        sum(1 for it in split.test if it.verified_purchase) / len(split.test)
        if split.test
        else 0.0
    )
    manifest = {
        "seed": seed,
        "k_core": k_core,
        "pct_verified_test": round(pct_verified_test, 4),
        **(extra_manifest or {}),
        **split.manifest,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return out / "manifest.json"
