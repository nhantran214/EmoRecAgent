"""Paper §III.F / Fig. 4 helpers: latent fusion confidence and (N_u, M_u).

Implements Eqs. (19)–(21) from EmoRecAgent (Guarded Reranking Agent).
Used by Option B (``guardrail_mode: context_dependent``) only.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def stage1_margin_confidence(
    stage1_ranked: Sequence[str],
    stage1_scores: dict[str, float],
    *,
    n0: int = 5,
) -> float:
    """Sigmoid of Stage-1 score gap between rank-1 and rank-``n0`` (Eq. 19 ``m_u``)."""
    if not stage1_ranked:
        return 0.0
    top = stage1_ranked[0]
    head_idx = min(max(n0, 1), len(stage1_ranked)) - 1
    head = stage1_ranked[head_idx]
    margin = float(stage1_scores.get(top, 0.0)) - float(
        stage1_scores.get(head, 0.0)
    )
    return float(sigmoid(margin))


def compute_alignment_confidence(
    *,
    s_u: np.ndarray | None,
    p_u: np.ndarray | None,
    stage1_ranked: Sequence[str],
    stage1_scores: dict[str, float],
    n0: int = 5,
    omega: float = 0.7,
) -> float:
    """Eq. (19): c_u = ω · c_align + (1-ω) · m_u."""
    if p_u is not None and s_u is not None:
        c_align = 0.5 * (1.0 + cosine_similarity(s_u, p_u))
    else:
        # No latent manifesto → rely on sequential margin only.
        c_align = 0.5
        omega = 0.0

    if not stage1_ranked:
        return float(c_align)
    m_u = stage1_margin_confidence(stage1_ranked, stage1_scores, n0=n0)
    return float(omega * c_align + (1.0 - omega) * m_u)


def should_invoke_llm(
    *,
    c_u: float,
    stage1_margin_conf: float,
    enabled: bool = True,
    min_c_u: float = 0.45,
    max_stage1_margin: float = 0.85,
) -> tuple[bool, str]:
    """Confidence gate (C): skip LLM when Stage-1 is sure or T_u poorly aligned.

    Returns ``(invoke, reason)``.
    """
    if not enabled:
        return True, "gate_off"
    if float(c_u) < float(min_c_u):
        return False, "low_c_u"
    if float(stage1_margin_conf) > float(max_stage1_margin):
        return False, "high_stage1_margin"
    return True, "ok"


def context_dependent_window(
    c_u: float,
    *,
    n0: int = 5,
    m0: int = 10,
    gamma_n: float = 3.0,
    gamma_m: float = 5.0,
    n_min: int = 3,
    n_max: int = 8,
    m_min: int = 8,
    m_max: int = 15,
) -> tuple[int, int]:
    """Eq. (20): (N_u, M_u) from alignment confidence c_u."""
    c = min(1.0, max(0.0, float(c_u)))
    n_u = int(round(n0 + gamma_n * (1.0 - c)))
    m_u = int(round(m0 + gamma_m * c))
    n_u = max(n_min, min(n_max, n_u))
    m_u = max(m_min, min(m_max, m_u))
    return n_u, m_u


def fuse_user_vector(
    s_u: np.ndarray,
    p_u: np.ndarray | None,
    *,
    alpha: float = 0.7,
) -> np.ndarray:
    """Eq. (16): x_u = α s_u + (1-α) p_u (α=1 when p_u unavailable)."""
    s = np.asarray(s_u, dtype=np.float64).reshape(-1)
    if p_u is None:
        return s
    a = float(alpha)
    if a >= 1.0 - 1e-9:
        return s
    p = np.asarray(p_u, dtype=np.float64).reshape(-1)
    if p.shape != s.shape:
        raise ValueError(f"p_u dim {p.shape} != s_u dim {s.shape}")
    return a * s + (1.0 - a) * p


def fused_pool_scores(
    x_u: np.ndarray,
    item_embs: dict[str, np.ndarray],
) -> dict[str, float]:
    """Eq. (17) base term: score(i) = e_i^⊤ x_u (boost applied separately)."""
    x = np.asarray(x_u, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    for item, emb in item_embs.items():
        e = np.asarray(emb, dtype=np.float64).reshape(-1)
        out[item] = float(np.dot(e, x))
    return out
