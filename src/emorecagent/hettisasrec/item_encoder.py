"""Item–aspect heterogeneous message passing (plain PyTorch, no torch-geometric)."""

from __future__ import annotations

import torch
from torch import nn

from .aspect_graph import AspectGraphBundle


def _neighbor_mean(
    node_emb: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_w: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean of neighbor embeddings; index 0 is padding."""
    nbr = node_emb[neighbor_idx.clamp(min=0)]
    mask = (neighbor_idx > 0).unsqueeze(-1).float()
    w = neighbor_w.unsqueeze(-1) * mask
    denom = w.sum(dim=1).clamp(min=1e-8)
    return (nbr * w).sum(dim=1) / denom


class ItemAspectEncoder(nn.Module):
    """Refine item embeddings via typed item↔aspect message passing."""

    def __init__(
        self,
        n_items: int,
        n_aspects: int,
        hidden: int,
        graph: AspectGraphBundle | None,
        *,
        n_layers: int = 1,
        use_aspect_enrichment: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_aspects = n_aspects
        self.hidden = hidden
        self.use_aspect_enrichment = use_aspect_enrichment and graph is not None
        self.n_layers = n_layers if self.use_aspect_enrichment else 0

        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.aspect_emb = nn.Embedding(n_aspects + 1, hidden, padding_idx=0)

        self.w_has = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(max(n_layers, 1))]
        )
        self.w_appears = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(max(n_layers, 1))]
        )
        self.item_norm = nn.ModuleList(
            [nn.LayerNorm(hidden, eps=1e-8) for _ in range(max(n_layers, 1))]
        )
        self.aspect_norm = nn.ModuleList(
            [nn.LayerNorm(hidden, eps=1e-8) for _ in range(max(n_layers, 1))]
        )
        self.dropout = nn.Dropout(dropout)

        if graph is not None:
            self.register_buffer("item_to_aspect_idx", graph.item_to_aspect_idx)
            self.register_buffer("item_to_aspect_w", graph.item_to_aspect_w)
            self.register_buffer("aspect_to_item_idx", graph.aspect_to_item_idx)
            self.register_buffer("aspect_to_item_w", graph.aspect_to_item_w)
        else:
            self.item_to_aspect_idx = None
            self.item_to_aspect_w = None
            self.aspect_to_item_idx = None
            self.aspect_to_item_w = None

    def aspect_table(self) -> torch.Tensor:
        return self.aspect_emb.weight

    def _one_layer(
        self,
        items: torch.Tensor,
        aspects: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.item_to_aspect_idx is not None
        asp_msg = _neighbor_mean(
            aspects, self.item_to_aspect_idx, self.item_to_aspect_w
        )
        item_msg = _neighbor_mean(
            items, self.aspect_to_item_idx, self.aspect_to_item_w
        )
        items = self.item_norm[layer](
            items + self.dropout(self.w_has[layer](asp_msg))
        )
        aspects = self.aspect_norm[layer](
            aspects + self.dropout(self.w_appears[layer](item_msg))
        )
        return items, aspects

    def forward_all_items(self) -> torch.Tensor:
        items = self.item_emb.weight
        if not self.use_aspect_enrichment:
            return items
        aspects = self.aspect_emb.weight
        for layer in range(self.n_layers):
            items, aspects = self._one_layer(items, aspects, layer)
        return items

    def item_table(self) -> torch.Tensor:
        """Refined embeddings for all items (compute once per forward/eval pass)."""
        return self.forward_all_items()

    def lookup(
        self, item_ids: torch.Tensor, table: torch.Tensor | None = None
    ) -> torch.Tensor:
        if table is not None:
            return table[item_ids]
        return self.forward(item_ids)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_aspect_enrichment:
            return self.item_emb(item_ids)
        return self.item_table()[item_ids]
