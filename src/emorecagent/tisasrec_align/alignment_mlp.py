"""Alignment MLP projecting SBERT embeddings into TiSASRec space."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class AlignmentMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        activation: Literal["elu", "gelu"] = "elu",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        if activation == "gelu":
            self.act: nn.Module = nn.GELU()
        else:
            self.act = nn.ELU()

    def forward(self, t_u: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(t_u))
