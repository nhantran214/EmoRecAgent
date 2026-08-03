"""Ported pyHGT GNN + Matcher for link prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .conv import GeneralConv
from .schema import NUM_NODE_TYPES, NUM_RELATIONS


class Matcher(nn.Module):
    def __init__(self, n_hid: int) -> None:
        super().__init__()
        self.left_linear = nn.Linear(n_hid, n_hid)
        self.right_linear = nn.Linear(n_hid, n_hid)
        self.sqrt_hd = math.sqrt(n_hid)
        self.cache = None

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        infer: bool = False,
        pair: bool = False,
    ) -> torch.Tensor:
        ty = self.right_linear(y)
        if infer:
            if self.cache is not None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.left_linear(x)
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0, 1))
        return res / self.sqrt_hd


class GNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_hid: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.2,
        use_RTE: bool = True,
    ) -> None:
        super().__init__()
        self.num_types = NUM_NODE_TYPES
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.gcs = nn.ModuleList()

        for _ in range(NUM_NODE_TYPES):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for _ in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(
                    "hgt",
                    n_hid,
                    n_hid,
                    NUM_NODE_TYPES,
                    NUM_RELATIONS,
                    n_heads,
                    dropout,
                    use_norm=False,
                    use_RTE=use_RTE,
                )
            )
        self.gcs.append(
            GeneralConv(
                "hgt",
                n_hid,
                n_hid,
                NUM_NODE_TYPES,
                NUM_RELATIONS,
                n_heads,
                dropout,
                use_norm=True,
                use_RTE=use_RTE,
            )
        )

    def forward(
        self,
        node_feature: torch.Tensor,
        node_type: torch.Tensor,
        edge_time: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        res = torch.zeros(node_feature.size(0), self.n_hid, device=node_feature.device)
        for t_id in range(self.num_types):
            idx = node_type == int(t_id)
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs
