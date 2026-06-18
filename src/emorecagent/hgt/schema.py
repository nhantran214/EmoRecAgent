"""Node and relation type registry for the EmoRecAgent HGT graph."""

from __future__ import annotations

from enum import IntEnum


class NodeType(IntEnum):
    USER = 0
    ITEM = 1
    ASPECT = 2


class RelationType(IntEnum):
    BUYS = 0
    BOUGHT_BY = 1
    HAS_ASPECT = 2
    APPEARS_IN = 3
    PREFERS = 4
    PREFERRED_BY = 5


NUM_NODE_TYPES = len(NodeType)
NUM_RELATIONS = len(RelationType)

OTHER_ASPECT = "aspect:other"


def relation_name(rel: RelationType) -> str:
    return rel.name.lower()
