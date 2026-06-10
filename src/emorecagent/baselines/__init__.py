"""Comparison methods for the EmoRecAgent evaluation (U11).

All baselines share the `Recommender` interface so the U12 harness can run them
and the full system on the identical agentic subset.
"""

from .base import Recommender
from .popularity import PopularityRecommender
from .itemknn import ItemKNNRecommender
from .svd import SVDRecommender
from .aspect_aware import AspectAwareRecommender
from .sequential import SequentialRecommender

__all__ = [
    "Recommender",
    "PopularityRecommender",
    "ItemKNNRecommender",
    "SVDRecommender",
    "AspectAwareRecommender",
    "SequentialRecommender",
]
