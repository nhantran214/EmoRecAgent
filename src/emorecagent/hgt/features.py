"""Text feature encoders for HGT node initialization."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


class HashTextEncoder:
    """Deterministic bag-of-words hash embedding (CI-friendly, no GPU)."""

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = _hash_embed(text or "", self._dim, self._seed)
        return out


class SentenceTransformerEncoder:
    """Optional sentence-transformers backend (requires ``[hgt]`` extra)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        sample = self._model.encode(["."], convert_to_numpy=True)
        self._dim = int(sample.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


def build_text_encoder(name: str, *, dim: int = 64, seed: int = 0) -> TextEncoder:
    if name == "hash":
        return HashTextEncoder(dim=dim, seed=seed)
    return SentenceTransformerEncoder(model_name=name)


def _hash_embed(text: str, dim: int, seed: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.md5(f"{seed}:{token}".encode(), usedforsecurity=False).hexdigest()
        h = int(digest, 16)
        idx = h % dim
        sign = 1.0 if (h >> 1) & 1 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec
