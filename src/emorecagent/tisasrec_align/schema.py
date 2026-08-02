"""Configuration dataclasses for pure TiSASRec."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TiSASRecArgs:
    maxlen: int = 50
    hidden_units: int = 64
    num_blocks: int = 2
    num_heads: int = 1
    dropout_rate: float = 0.2
    l2_emb: float = 1e-4
    time_span: int = 256
    # RecBole FFN width. ``None`` → hidden→hidden (Kang / Amazon default).
    inner_size: int | None = None
    # Relative-time unit. ``None`` → per-user min-gap (Li et al. / Amazon
    # default). Travels with the checkpoint so inference rebuilds histories on
    # the same scale the model trained on.
    time_unit_seconds: int | None = None
