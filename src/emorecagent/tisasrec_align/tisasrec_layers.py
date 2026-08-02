"""TiSASRec layers (paper-faithful, device-agnostic tensors)."""

from __future__ import annotations

import sys

import torch
from torch import nn

FLOAT_MIN = -sys.float_info.max


class PointWiseFeedForward(nn.Module):
    def __init__(
        self,
        hidden_units: int,
        dropout_rate: float,
        *,
        inner_size: int | None = None,
    ) -> None:
        super().__init__()
        # RecBole Transformer FFN uses ``inner_size`` (default 256). Kang
        # TiSASRec uses hidden→hidden; pass ``inner_size=None`` / ``=hidden``
        # for that path.
        mid = hidden_units if inner_size is None else int(inner_size)
        self.conv1 = nn.Conv1d(hidden_units, mid, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(mid, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inputs.transpose(-1, -2))
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x)
        outputs = self.dropout2(x).transpose(-1, -2)
        return outputs + inputs


class TimeAwareMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, head_num: int, dropout_rate: float) -> None:
        super().__init__()
        self.Q_w = nn.Linear(hidden_size, hidden_size)
        self.K_w = nn.Linear(hidden_size, hidden_size)
        self.V_w = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        time_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        time_matrix_K: torch.Tensor,
        time_matrix_V: torch.Tensor,
        abs_pos_K: torch.Tensor,
        abs_pos_V: torch.Tensor,
    ) -> torch.Tensor:
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(
            torch.split(time_matrix_K, self.head_size, dim=3), dim=0
        )
        time_matrix_V_ = torch.cat(
            torch.split(time_matrix_V, self.head_size, dim=3), dim=0
        )
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        dev = queries.device
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape, device=dev) * (-2**32 + 1)
        attn_weights = torch.where(time_mask, paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        attn_weights = self.dropout(self.softmax(attn_weights))

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += (
            attn_weights.unsqueeze(2)
            .matmul(time_matrix_V_)
            .reshape(outputs.shape)
            .squeeze(2)
        )
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)
        return outputs
