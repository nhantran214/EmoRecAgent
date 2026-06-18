"""Ported pyHGT convolution layers (torch 2.x / modern PyG)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid: int, max_len: int = 240) -> None:
        super().__init__()
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x + self.lin(self.emb(t))


class HGTConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_types: int,
        num_relations: int,
        n_heads: int,
        dropout: float = 0.2,
        use_norm: bool = True,
        use_RTE: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(node_dim=0, aggr="add", **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_RTE = use_RTE
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        if self.use_RTE:
            self.emb = RelTemporalEncoding(in_dim)

        glorot(self.relation_att)
        glorot(self.relation_msg)

    def forward(
        self,
        node_inp: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_time: torch.Tensor,
    ) -> torch.Tensor:
        return self.propagate(
            edge_index,
            node_inp=node_inp,
            node_type=node_type,
            edge_type=edge_type,
            edge_time=edge_time,
        )

    def message(
        self,
        edge_index_i,
        node_inp_i,
        node_inp_j,
        node_type_i,
        node_type_j,
        edge_type,
        edge_time,
    ):
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads, device=node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k, device=node_inp_i.device)

        for source_type in range(self.num_types):
            sb = node_type_j == int(source_type)
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(
                        k_mat.transpose(1, 0), self.relation_att[relation_type]
                    ).transpose(1, 0)
                    res_att[idx] = (
                        (q_mat * k_mat).sum(dim=-1)
                        * self.relation_pri[relation_type]
                        / self.sqrt_dk
                    )
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(
                        v_mat.transpose(1, 0), self.relation_msg[relation_type]
                    ).transpose(1, 0)

        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim, device=node_inp.device)
        for target_type in range(self.num_types):
            idx = node_type == int(target_type)
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](
                    trans_out * alpha + node_inp[idx] * (1 - alpha)
                )
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res


class GeneralConv(nn.Module):
    def __init__(
        self,
        conv_name: str,
        in_hid: int,
        out_hid: int,
        num_types: int,
        num_relations: int,
        n_heads: int,
        dropout: float,
        use_norm: bool = True,
        use_RTE: bool = True,
    ) -> None:
        super().__init__()
        if conv_name != "hgt":
            raise ValueError(f"Unsupported conv: {conv_name}")
        self.base_conv = HGTConv(
            in_hid,
            out_hid,
            num_types,
            num_relations,
            n_heads,
            dropout,
            use_norm,
            use_RTE,
        )

    def forward(
        self,
        meta_xs: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_time: torch.Tensor,
    ) -> torch.Tensor:
        return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
