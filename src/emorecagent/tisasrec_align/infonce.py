"""InfoNCE loss for alignment MLP training."""

from __future__ import annotations

import torch
from torch import nn


def infonce_loss(
    query: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Contrastive loss: query (B,d), pos (B,d), neg (B,N,d)."""
    pos_logits = (query * pos).sum(dim=-1, keepdim=True) / temperature
    neg_logits = torch.bmm(neg, query.unsqueeze(-1)).squeeze(-1) / temperature
    logits = torch.cat([pos_logits, neg_logits], dim=-1)
    labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    return nn.functional.cross_entropy(logits, labels)
