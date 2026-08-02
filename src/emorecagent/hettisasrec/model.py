"""HetTiSASRec: TiSASRec user tower + item–aspect enrichment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .aspect_graph import AspectGraphBundle
from .item_encoder import ItemAspectEncoder
from .tisasrec_layers import PointWiseFeedForward, TimeAwareMultiHeadAttention


@dataclass(frozen=True, slots=True)
class HetTiSASRecArgs:
    maxlen: int = 50
    hidden_units: int = 64
    num_blocks: int = 2
    num_heads: int = 1
    dropout_rate: float = 0.2
    l2_emb: float = 1e-4
    time_span: int = 256
    use_aspect_enrichment: bool = True
    aspect_mp_layers: int = 1
    aspect_loss_weight: float = 0.2


class HetTiSASRecModel(nn.Module):
    """Unified sequential retriever with optional item–aspect graph enrichment."""

    def __init__(
        self,
        user_num: int,
        item_num: int,
        args: HetTiSASRecArgs,
        graph: AspectGraphBundle | None = None,
    ) -> None:
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.args = args

        self.item_encoder = ItemAspectEncoder(
            item_num,
            graph.n_aspects if graph else 0,
            args.hidden_units,
            graph,
            n_layers=args.aspect_mp_layers,
            use_aspect_enrichment=args.use_aspect_enrichment,
            dropout=args.dropout_rate,
        )

        self.abs_pos_K_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_K_emb = nn.Embedding(args.time_span + 1, args.hidden_units)
        self.time_matrix_V_emb = nn.Embedding(args.time_span + 1, args.hidden_units)

        self.item_dropout = nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_dropout = nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_dropout = nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(
                nn.LayerNorm(args.hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                TimeAwareMultiHeadAttention(
                    args.hidden_units, args.num_heads, args.dropout_rate
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(
                PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            )

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

    def all_item_embeddings(self) -> torch.Tensor:
        """One graph-refinement pass over the full catalog."""
        return self.item_encoder.item_table()

    def item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_encoder.lookup(item_ids, self.all_item_embeddings())

    def aspect_embeddings(self) -> torch.Tensor:
        return self.item_encoder.aspect_table()

    def seq2feats(
        self,
        log_seqs: torch.Tensor,
        time_matrices: torch.Tensor,
        *,
        item_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        table = item_table if item_table is not None else self.all_item_embeddings()
        seqs = table[log_seqs]
        seqs = seqs * (self.args.hidden_units**0.5)
        seqs = self.item_dropout(seqs)

        positions = torch.arange(log_seqs.shape[1], device=log_seqs.device)
        positions = positions.unsqueeze(0).expand(log_seqs.shape[0], -1)
        abs_pos_K = self.abs_pos_K_dropout(self.abs_pos_K_emb(positions))
        abs_pos_V = self.abs_pos_V_dropout(self.abs_pos_V_emb(positions))

        time_matrix_K = self.time_matrix_K_dropout(
            self.time_matrix_K_emb(time_matrices)
        )
        time_matrix_V = self.time_matrix_V_dropout(
            self.time_matrix_V_emb(time_matrices)
        )

        timeline_mask = log_seqs == 0
        seqs = seqs * (~timeline_mask).unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device)
        )

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](
                Q,
                seqs,
                timeline_mask,
                attention_mask,
                time_matrix_K,
                time_matrix_V,
                abs_pos_K,
                abs_pos_V,
            )
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask).unsqueeze(-1)

        return self.last_layernorm(seqs)

    def user_repr(
        self,
        log_seqs: torch.Tensor,
        time_matrices: torch.Tensor,
        *,
        item_table: torch.Tensor | None = None,
        aspect_gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_feats = self.seq2feats(
            log_seqs, time_matrices, item_table=item_table
        )
        hu = log_feats[:, -1, :]
        if aspect_gamma is not None:
            hu = hu + aspect_gamma
        return hu

    def forward(
        self,
        log_seqs: torch.Tensor,
        time_matrices: torch.Tensor,
        pos_seqs: torch.Tensor,
        neg_seqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table = self.all_item_embeddings()
        log_feats = self.seq2feats(log_seqs, time_matrices, item_table=table)
        pos_embs = table[pos_seqs]
        neg_embs = table[neg_seqs]
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(
        self,
        log_seqs: torch.Tensor,
        time_matrices: torch.Tensor,
        item_indices: torch.Tensor,
        *,
        item_table: torch.Tensor | None = None,
        aspect_gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        table = item_table if item_table is not None else self.all_item_embeddings()
        hu = self.user_repr(
            log_seqs,
            time_matrices,
            item_table=table,
            aspect_gamma=aspect_gamma,
        )
        item_embs = table[item_indices]
        return item_embs.matmul(hu.unsqueeze(-1)).squeeze(-1)

    def l2_regularization(self) -> torch.Tensor:
        l2 = self.item_encoder.item_emb.weight.norm(2)
        if self.args.use_aspect_enrichment:
            l2 = l2 + self.item_encoder.aspect_emb.weight.norm(2)
        return self.args.l2_emb * l2
