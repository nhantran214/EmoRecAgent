"""String <-> integer id maps for sequential models."""

from __future__ import annotations

from dataclasses import dataclass

from ..data.types import Interaction


@dataclass(frozen=True)
class IdMaps:
    """Bidirectional string maps (1-based indices; 0 = pad)."""

    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]

    @property
    def idx_to_user(self) -> dict[int, str]:
        return {v: k for k, v in self.user_to_idx.items()}

    @property
    def idx_to_item(self) -> dict[int, str]:
        return {v: k for k, v in self.item_to_idx.items()}


def build_id_maps_from_interactions(
    *interaction_lists: list[Interaction],
) -> IdMaps:
    users = sorted(
        {it.user_id for group in interaction_lists for it in group}
    )
    items = sorted({it.item for group in interaction_lists for it in group})
    return IdMaps(
        user_to_idx={uid: i + 1 for i, uid in enumerate(users)},
        item_to_idx={iid: i + 1 for i, iid in enumerate(items)},
    )
