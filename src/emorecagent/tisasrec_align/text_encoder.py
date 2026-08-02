"""Frozen sentence-transformer text encoder."""

from __future__ import annotations

from typing import Protocol

import torch


class TextEncoderBackend(Protocol):
    def encode(self, texts: list[str], *, device: torch.device) -> torch.Tensor: ...


class SentenceTransformerEncoder:
    """Lazy-load all-mpnet-base-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: list[str], *, device: torch.device) -> torch.Tensor:
        model = self._ensure_model()
        emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return emb.to(device).detach()


class HashEncoder:
    """Deterministic mock encoder for tests without sentence-transformers."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def encode(self, texts: list[str], *, device: torch.device) -> torch.Tensor:
        out = []
        for t in texts:
            h = hash(t) % 10_000
            vec = torch.zeros(self.dim, device=device)
            vec[h % self.dim] = 1.0
            vec[(h * 7) % self.dim] = 0.5
            out.append(vec)
        return torch.stack(out, dim=0)
