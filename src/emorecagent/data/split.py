"""Sampling and chronological leave-last-out splitting.

Split protocol (matches the AmazonReviews2023 benchmark convention): per user,
sort interactions by time; the latest is test, the second-latest is validation,
the rest are train. Users without enough history are kept as train-only.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
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


def sample_subset(
    interactions: list[Interaction],
    k_core: int,
    max_users: int | None,
    max_items: int | None,
    seed: int,
) -> list[Interaction]:
    """Restrict to a seeded user/item sample, then restore the k-core property.

    Sampling breaks the k-core invariant, so k-core is re-applied afterward.
    """
    current = k_core_filter(interactions, k_core)
    if max_users is None and max_items is None:
        return current

    rng = random.Random(seed)

    if max_users is not None:
        users = sorted({it.user_id for it in current})
        if len(users) > max_users:
            users = set(rng.sample(users, max_users))
            current = [it for it in current if it.user_id in users]

    if max_items is not None:
        items = sorted({it.item for it in current})
        if len(items) > max_items:
            keep = set(rng.sample(items, max_items))
            current = [it for it in current if it.item in keep]

    return k_core_filter(current, k_core)


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
                    }
                )
                + "\n"
            )


def write_split(out_dir: str | Path, split: Split, seed: int, k_core: int) -> Path:
    """Write train/valid/test JSONL files and a manifest.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", split.train)
    _write_jsonl(out / "valid.jsonl", split.valid)
    _write_jsonl(out / "test.jsonl", split.test)
    manifest = {"seed": seed, "k_core": k_core, **split.manifest}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return out / "manifest.json"
