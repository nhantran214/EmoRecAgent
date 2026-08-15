"""Complementary item-potential score φ(u,i) for Stage-2 swap (no LLM).

v1 mixed 90% weak residual / 10% seq and lost to π¹ on Beauty valid.
v1.1: φ = z_seq + α · residual (keep Stage-1 backbone).
v2: listwise softmax LTR, user-grouped OOF. Features = π¹ rank + residual
(text, co, last-co, hist, category, popularity, tail interactions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cross_user_lookup import CrossUserLookup, lookup_co_items
from .item_metadata import ItemMeta
from .stage2_paper_guard import cosine_similarity
from .stage2_reason_promote import item_token_hit_count, preference_tokens

DEFAULT_WEIGHTS: dict[str, float] = {
    "text": 0.45,
    "co": 0.25,
    "hist": 0.20,
    "seq": 0.10,
}

# Residual mix on top of z_seq / π¹ rank. v1 put 90% mass here and failed.
DEFAULT_RESIDUAL_WEIGHTS: dict[str, float] = {
    "text": 0.25,
    "co": 0.15,
    "hist": 0.15,
    "last_co": 0.25,
    "cat": 0.20,
}

LTR_FEATURE_NAMES: tuple[str, ...] = (
    "neg_log_rank",
    "inv_rank",
    "z_seq",
    "z_text",
    "z_co",
    "z_hist",
    "z_last_co",
    "z_cat",
    "z_pop",
    "z_markov",
    "z_brand",
    "seq_gap_r10",
    "z_cat_unpop",
    "z_text_tail",
    "z_co_tail",
    "z_hist_tail",
    "z_last_co_tail",
    "z_cat_tail",
    "z_markov_tail",
    "z_brand_tail",
    "z_cat_unpop_tail",
)

RANK_FEATURE_NAMES: tuple[str, ...] = ("neg_log_rank", "inv_rank", "z_seq")
CACHE_V1_DIM: int = 14


def ltr_feature_vector(
    *,
    rank_1indexed: int,
    z_seq: float,
    z_text: float,
    z_co: float,
    z_hist: float,
    z_last_co: float = 0.0,
    z_cat: float = 0.0,
    z_pop: float = 0.0,
    z_markov: float = 0.0,
    z_brand: float = 0.0,
    seq_gap_r10: float = 0.0,
    tail_rank: int = 20,
) -> np.ndarray:
    """One row of the valid-gold LTR channel (π¹ backbone + residual × tail)."""
    r = max(int(rank_1indexed), 1)
    tail = 1.0 if r > int(tail_rank) else 0.0
    cat_unpop = float(z_cat) * (-float(z_pop))
    return np.asarray(
        [
            -math.log(float(r)),
            1.0 / float(r),
            float(z_seq),
            float(z_text),
            float(z_co),
            float(z_hist),
            float(z_last_co),
            float(z_cat),
            float(z_pop),
            float(z_markov),
            float(z_brand),
            float(seq_gap_r10),
            cat_unpop,
            float(z_text) * tail,
            float(z_co) * tail,
            float(z_hist) * tail,
            float(z_last_co) * tail,
            float(z_cat) * tail,
            float(z_markov) * tail,
            float(z_brand) * tail,
            cat_unpop * tail,
        ],
        dtype=np.float64,
    )


def ltr_feature_matrix(
    pool: list[str],
    scored: PotentialScores,
    *,
    tail_rank: int = 20,
) -> np.ndarray:
    """Stack LTR rows in π¹ pool order (row i = rank i+1)."""
    z10 = 0.0
    if len(pool) >= 10:
        z10 = float(scored.z_seq.get(pool[9], 0.0))
    rows = [
        ltr_feature_vector(
            rank_1indexed=i + 1,
            z_seq=float(scored.z_seq.get(item, 0.0)),
            z_text=float(scored.z_text.get(item, 0.0)),
            z_co=float(scored.z_co.get(item, 0.0)),
            z_hist=float(scored.z_hist.get(item, 0.0)),
            z_last_co=float(scored.z_last_co.get(item, 0.0)),
            z_cat=float(scored.z_cat.get(item, 0.0)),
            z_pop=float(scored.z_pop.get(item, 0.0)),
            z_markov=float(scored.z_markov.get(item, 0.0)),
            z_brand=float(scored.z_brand.get(item, 0.0)),
            seq_gap_r10=float(scored.z_seq.get(item, 0.0)) - z10,
            tail_rank=tail_rank,
        )
        for i, item in enumerate(pool)
    ]
    return np.stack(rows, axis=0) if rows else np.zeros((0, len(LTR_FEATURE_NAMES)))


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def _user_slices(groups: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) slices; ``groups`` must be sorted by user."""
    n = int(groups.shape[0])
    if n == 0:
        return []
    slices: list[tuple[int, int]] = []
    start = 0
    cur = groups[0]
    for i in range(1, n):
        if groups[i] != cur:
            slices.append((start, i))
            start = i
            cur = groups[i]
    slices.append((start, n))
    return slices


def listwise_nll(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    slices: list[tuple[int, int]],
) -> float:
    """Mean per-user −log P(gold | softmax over pool)."""
    nll, _ = listwise_nll_and_grad(w, X, y, slices)
    return nll


def listwise_nll_and_grad(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    slices: list[tuple[int, int]],
) -> tuple[float, np.ndarray]:
    s = X @ w
    g = np.zeros_like(w, dtype=np.float64)
    total = 0.0
    n_ok = 0
    for a, b in slices:
        yy = y[a:b]
        if not np.any(yy > 0):
            continue
        ss = s[a:b]
        xx = X[a:b]
        m = float(ss.max())
        ex = np.exp(ss - m)
        z = float(ex.sum())
        gold = float(ex[yy > 0].sum())
        total += -math.log(max(gold, 1e-12) / max(z, 1e-12))
        p = ex / max(z, 1e-12)
        pg = np.zeros_like(p)
        mask = yy > 0
        pg[mask] = ex[mask] / max(gold, 1e-12)
        g += xx.T @ (p - pg)
        n_ok += 1
    n_ok = max(n_ok, 1)
    return total / n_ok, g / n_ok


def fit_listwise_weights(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    l2: float = 0.1,
    drop_rank: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit listwise softmax weights. Returns ``(w, mu, sd)`` on training rows."""
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_in = int(X.shape[1])
    rank_cols = 3 if n_in > 3 else 0
    if drop_rank and rank_cols:
        X = X[:, rank_cols:]
    Xs, mu, sd = _standardize_fit(X)
    slices = _user_slices(groups)
    n_f = Xs.shape[1]
    w0 = np.zeros(n_f, dtype=np.float64)
    # Prefer π¹ when rank features exist: start −log_rank in the direction of gold.
    if not drop_rank and n_f >= 1:
        w0[0] = 1.0

    def obj(w: np.ndarray) -> tuple[float, np.ndarray]:
        nll, grad = listwise_nll_and_grad(w, Xs, y, slices)
        return nll + float(l2) * float(np.dot(w, w)), grad + 2.0 * float(l2) * w

    res = minimize(obj, w0, method="L-BFGS-B", jac=True, options={"maxiter": 80})
    w = np.asarray(res.x, dtype=np.float64)
    if drop_rank and rank_cols:
        full = np.zeros(n_in, dtype=np.float64)
        full[rank_cols:] = w
        mu_full = np.zeros(n_in, dtype=np.float64)
        sd_full = np.ones(n_in, dtype=np.float64)
        mu_full[rank_cols:] = mu
        sd_full[rank_cols:] = sd
        return full, mu_full, sd_full
    return w, mu, sd


def apply_listwise_scores(
    X: np.ndarray,
    w: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    Xs = _standardize_apply(np.asarray(X, dtype=np.float64), mu, sd)
    return Xs @ np.asarray(w, dtype=np.float64)


def oof_listwise_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    l2: float = 0.1,
    drop_rank: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """User-grouped OOF listwise scores (no valid-item leakage across folds)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    n = X.shape[0]
    oof = np.zeros(n, dtype=np.float64)
    uniq = np.unique(groups)
    n_splits = min(int(n_splits), max(len(uniq), 1))
    if n_splits < 2:
        w, mu, sd = fit_listwise_weights(X, y, groups, l2=l2, drop_rank=drop_rank)
        return apply_listwise_scores(X, w, mu, sd)
    rng = np.random.RandomState(int(seed))
    order = np.arange(len(uniq))
    rng.shuffle(order)
    fold_of = {}
    for i, idx in enumerate(order):
        fold_of[uniq[idx]] = i % n_splits
    group_fold = np.asarray([fold_of[g] for g in groups], dtype=np.int64)
    for fold in range(n_splits):
        te = np.where(group_fold == fold)[0]
        tr = np.where(group_fold != fold)[0]
        w, mu, sd = fit_listwise_weights(
            X[tr], y[tr], groups[tr], l2=l2, drop_rank=drop_rank
        )
        oof[te] = apply_listwise_scores(X[te], w, mu, sd)
    return oof


def augment_v1_cache_features(
    X: np.ndarray,
    ranks: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Add seq_gap_r10, z_cat×(−z_pop), and tail interaction to 14-col v1 cache."""
    X = np.asarray(X, dtype=np.float64)
    ranks = np.asarray(ranks)
    extra = np.zeros((X.shape[0], 3), dtype=np.float64)
    if X.shape[1] < 9:
        return X
    z_seq = X[:, 2]
    z_cat = X[:, 7]
    z_pop = X[:, 8]
    for a, b in _user_slices(groups):
        rr = ranks[a:b]
        zs = z_seq[a:b]
        hit = np.where(rr == 10)[0]
        z10 = float(zs[int(hit[0])]) if len(hit) else 0.0
        extra[a:b, 0] = zs - z10
        extra[a:b, 1] = z_cat[a:b] * (-z_pop[a:b])
        extra[a:b, 2] = extra[a:b, 1] * (rr.astype(np.float64) > 20.0)
    return np.hstack([X, extra])


def oof_hgb_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    drop_rank: bool = True,
    seed: int = 42,
    pos_weight: float = 25.0,
) -> np.ndarray:
    """User-grouped OOF HistGradientBoosting P(gold). Default: residual columns only."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.asarray(X, dtype=np.float64)
    yb = (np.asarray(y) > 0).astype(np.int32)
    groups = np.asarray(groups)
    if drop_rank and X.shape[1] > 3:
        X = X[:, 3:]
    n = X.shape[0]
    oof = np.zeros(n, dtype=np.float64)
    uniq = np.unique(groups)
    n_splits = min(int(n_splits), max(len(uniq), 1))
    rng = np.random.RandomState(int(seed))
    order = np.arange(len(uniq))
    rng.shuffle(order)
    fold_of = {uniq[idx]: i % n_splits for i, idx in enumerate(order)}
    group_fold = np.asarray([fold_of[g] for g in groups], dtype=np.int64)
    sw = np.where(yb > 0, float(pos_weight), 1.0)
    for fold in range(max(n_splits, 1)):
        if n_splits < 2:
            tr = np.arange(n)
            te = tr
        else:
            te = np.where(group_fold == fold)[0]
            tr = np.where(group_fold != fold)[0]
        clf = HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=60,
            learning_rate=0.08,
            min_samples_leaf=80,
            l2_regularization=0.15,
            random_state=int(seed) + fold,
        )
        clf.fit(X[tr], yb[tr], sample_weight=sw[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        if n_splits < 2:
            break
    return oof


def score_pool_ltr(
    pool: list[str],
    scored: PotentialScores,
    *,
    w: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    tail_rank: int = 20,
) -> dict[str, float]:
    """φ_v2: listwise LTR over v1 channels + π¹ rank."""
    X = ltr_feature_matrix(pool, scored, tail_rank=tail_rank)
    if int(w.shape[0]) != int(X.shape[1]):
        raise ValueError(
            f"LTR weight dim {w.shape[0]} != feature dim {X.shape[1]}; "
            "re-fit weights after a feature change"
        )
    s = apply_listwise_scores(X, w, mu, sd)
    return {item: float(s[i]) for i, item in enumerate(pool)}


def load_listwise_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load fit-on-valid ``(w, mu, sd)`` saved by ``analyze_item_potential``."""
    data = np.load(Path(path), allow_pickle=True)
    return (
        np.asarray(data["w"], dtype=np.float64),
        np.asarray(data["mu"], dtype=np.float64),
        np.asarray(data["sd"], dtype=np.float64),
    )


def rerank_pool_by_phi(
    pool: list[str],
    phi: dict[str, float],
) -> list[str]:
    """Reorder π¹ pool by φ descending; ties keep Stage-1 order."""
    pos = {item: i for i, item in enumerate(pool)}
    return sorted(pool, key=lambda x: (-float(phi.get(x, 0.0)), pos[x]))


@dataclass(frozen=True)
class PotentialScores:
    """Per-item φ and z-scored / raw channels, keyed by item id."""

    phi: dict[str, float]
    z_text: dict[str, float]
    z_co: dict[str, float]
    z_hist: dict[str, float]
    z_seq: dict[str, float]
    z_last_co: dict[str, float]
    z_cat: dict[str, float]
    z_pop: dict[str, float]
    z_markov: dict[str, float]
    z_brand: dict[str, float]
    raw_text: dict[str, float]
    raw_co: dict[str, float]
    raw_hist: dict[str, float]
    raw_seq: dict[str, float]
    raw_last_co: dict[str, float]
    raw_cat: dict[str, float]
    raw_pop: dict[str, float]
    raw_markov: dict[str, float]
    raw_brand: dict[str, float]
    weights_used: dict[str, float]


def mix_backbone_residual(
    pool: list[str],
    scored: PotentialScores,
    *,
    alpha: float = 0.15,
    residual_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """φ = z_seq + α · residual. α=0 copies π¹ order inside the pool.

    v1 mixed 90% residual / 10% seq and scored below π¹. This keeps Stage-1
    as the backbone and treats text/co/hist/last-co/cat as a correction.
    """
    rw = dict(DEFAULT_RESIDUAL_WEIGHTS if residual_weights is None else residual_weights)
    channels = {
        "text": scored.z_text,
        "co": scored.z_co,
        "hist": scored.z_hist,
        "last_co": scored.z_last_co,
        "cat": scored.z_cat,
    }
    used = _renormalize_weights(channels, rw)
    out: dict[str, float] = {}
    a = float(alpha)
    for item in pool:
        resid = 0.0
        for name, w in used.items():
            resid += w * float(channels[name].get(item, 0.0))
        out[item] = float(scored.z_seq.get(item, 0.0)) + a * resid
    return out


def zscore_map(raw: dict[str, float]) -> dict[str, float]:
    """Standardize values inside one user's pool. All-equal → zeros."""
    if not raw:
        return {}
    xs = np.asarray(list(raw.values()), dtype=np.float64)
    mu = float(xs.mean())
    sd = float(xs.std())
    if sd < 1e-12:
        return {k: 0.0 for k in raw}
    return {k: (float(v) - mu) / sd for k, v in raw.items()}


def rank_penalty(rank_1indexed: int, head_n: int = 20) -> float:
    """Cost of evicting π¹ rank ``r`` from the head. Rank 1 is costliest."""
    n = max(int(head_n), 1)
    r = min(max(int(rank_1indexed), 1), n)
    return float(n + 1 - r) / float(n)


def swap_delta(
    phi_challenger: float,
    phi_incumbent: float,
    incumbent_rank: int,
    *,
    tau: float = 0.25,
    head_n: int = 20,
) -> float:
    """Δ = φ(c) − φ(d) − τ · penalty(rank(d)). Swap when Δ > γ."""
    return (
        float(phi_challenger)
        - float(phi_incumbent)
        - float(tau) * rank_penalty(incumbent_rank, head_n)
    )


def _overlap_scores(
    pool: list[str],
    t_u: str,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
) -> dict[str, float]:
    q = preference_tokens(t_u)
    if not q:
        return {item: 0.0 for item in pool}
    denom = float(len(q))
    return {
        item: float(
            item_token_hit_count(q, item, item_meta, review_snippets)
        )
        / denom
        for item in pool
    }


def _aligned_cos_scores(
    pool: list[str],
    p_u: np.ndarray | None,
    item_embs: dict[str, np.ndarray],
) -> dict[str, float]:
    if p_u is None:
        return {item: 0.0 for item in pool}
    out: dict[str, float] = {}
    for item in pool:
        emb = item_embs.get(item)
        out[item] = cosine_similarity(p_u, emb) if emb is not None else 0.0
    return out


def _category_tokens(meta: ItemMeta | None) -> set[str]:
    if meta is None or not meta.categories:
        return set()
    return {c.strip().lower() for c in meta.categories.split(",") if c.strip()}


def _name_tokens(meta: ItemMeta | None, *, max_tok: int = 6) -> set[str]:
    if meta is None or not meta.name:
        return set()
    out: list[str] = []
    for raw in meta.name.lower().replace("/", " ").replace("-", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if len(tok) >= 3:
            out.append(tok)
        if len(out) >= max_tok:
            break
    return set(out)


def _brand_overlap_scores(
    pool: list[str],
    history_items: list[str],
    item_meta: dict[str, ItemMeta] | None,
    *,
    last_k: int = 3,
) -> dict[str, float]:
    if not item_meta or not history_items:
        return {item: 0.0 for item in pool}
    hist: set[str] = set()
    for hid in history_items[-max(int(last_k), 1) :]:
        hist |= _name_tokens(item_meta.get(hid))
    if not hist:
        return {item: 0.0 for item in pool}
    out: dict[str, float] = {}
    for item in pool:
        it = _name_tokens(item_meta.get(item))
        if not it:
            out[item] = 0.0
            continue
        out[item] = float(len(hist & it)) / float(len(hist | it))
    return out


def build_next_item_lookup(events: list) -> dict[str, dict[str, float]]:
    """P(next | prev) from consecutive train events (not basket co-purchase)."""
    from collections import defaultdict

    by_user: dict[str, list] = defaultdict(list)
    for it in events:
        by_user[str(it.user_id)].append(it)
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for rows in by_user.values():
        rows = sorted(rows, key=lambda r: (int(r.timestamp), str(r.item)))
        for a, b in zip(rows, rows[1:]):
            ia, ib = str(a.item), str(b.item)
            if ia != ib:
                counts[ia][ib] += 1.0
    out: dict[str, dict[str, float]] = {}
    for prev, cm in counts.items():
        z = float(sum(cm.values()))
        if z <= 0:
            continue
        out[prev] = {nxt: c / z for nxt, c in cm.items()}
    return out


def markov_scores(
    pool: list[str],
    history_items: list[str],
    lookup: dict[str, dict[str, float]] | None,
    *,
    last_k: int = 3,
    decay: float = 0.6,
) -> dict[str, float]:
    if not history_items or not lookup:
        return {item: 0.0 for item in pool}
    acc = {item: 0.0 for item in pool}
    k = min(max(int(last_k), 1), len(history_items))
    wsum = 0.0
    for j, hid in enumerate(history_items[-k:]):
        w = float(decay) ** (k - 1 - j)
        cm = lookup.get(hid) or {}
        wsum += w
        for item in pool:
            acc[item] += w * float(cm.get(item, 0.0))
    if wsum <= 0:
        return acc
    return {item: v / wsum for item, v in acc.items()}


def _cat_overlap_scores(
    pool: list[str],
    history_items: list[str],
    item_meta: dict[str, ItemMeta] | None,
    *,
    last_k: int = 5,
) -> dict[str, float]:
    if not item_meta or not history_items:
        return {item: 0.0 for item in pool}
    hist_cats: set[str] = set()
    for hid in history_items[-max(int(last_k), 1) :]:
        hist_cats |= _category_tokens(item_meta.get(hid))
    if not hist_cats:
        return {item: 0.0 for item in pool}
    out: dict[str, float] = {}
    for item in pool:
        ic = _category_tokens(item_meta.get(item))
        if not ic:
            out[item] = 0.0
            continue
        out[item] = float(len(hist_cats & ic)) / float(len(hist_cats | ic))
    return out


def _hist_affinity_scores(
    pool: list[str],
    history_items: list[str],
    item_embs: dict[str, np.ndarray],
    *,
    decay: float = 0.9,
) -> dict[str, float]:
    """Recency-weighted mean cosine vs history item embeddings (oldest→newest)."""
    if not history_items:
        return {item: 0.0 for item in pool}
    n = len(history_items)
    weighted: list[tuple[float, np.ndarray]] = []
    for k, hid in enumerate(history_items):
        emb = item_embs.get(hid)
        if emb is None:
            continue
        w = float(decay) ** (n - 1 - k)
        weighted.append((w, emb))
    if not weighted:
        return {item: 0.0 for item in pool}
    wsum = sum(w for w, _ in weighted)
    if wsum <= 0:
        return {item: 0.0 for item in pool}
    out: dict[str, float] = {}
    for item in pool:
        ei = item_embs.get(item)
        if ei is None:
            out[item] = 0.0
            continue
        acc = 0.0
        for w, ej in weighted:
            acc += w * cosine_similarity(ei, ej)
        out[item] = acc / wsum
    return out


def _renormalize_weights(
    z_channels: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, float]:
    """Drop all-zero channels (cold-start) and renormalize the rest."""
    active: dict[str, float] = {}
    for name, w in weights.items():
        ch = z_channels.get(name) or {}
        if w <= 0 or not ch:
            continue
        if any(abs(v) > 1e-12 for v in ch.values()):
            active[name] = float(w)
    total = sum(active.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in active.items()}


def score_pool_potential(
    pool: list[str],
    *,
    t_u: str,
    p_u: np.ndarray | None,
    item_embs: dict[str, np.ndarray],
    seq_scores: dict[str, float],
    history_items: list[str],
    anchor_items: list[str],
    lookup: CrossUserLookup,
    item_meta: dict[str, ItemMeta] | None = None,
    review_snippets: dict[str, list[str]] | None = None,
    item_pop: dict[str, float] | None = None,
    markov_lookup: dict[str, dict[str, float]] | None = None,
    weights: dict[str, float] | None = None,
    hist_decay: float = 0.9,
    text_align_mix: float = 0.5,
) -> PotentialScores:
    """Score every item in ``pool`` (typically π¹[:300]).

    ``history_items`` / ``anchor_items`` are train (and valid-prefix) ids
    *before* the query timestamp, oldest→newest. ``p_u`` is MLP(enc(T_u))
    in the same space as ``item_embs``; None skips the aligned-cosine term.
    """
    pool = list(dict.fromkeys(pool))
    wcfg = dict(DEFAULT_WEIGHTS if weights is None else weights)

    overlap = _overlap_scores(pool, t_u, item_meta, review_snippets)
    aligned = _aligned_cos_scores(pool, p_u, item_embs)
    mix = min(max(float(text_align_mix), 0.0), 1.0)
    if p_u is None:
        raw_text = overlap
    else:
        raw_text = {
            item: mix * aligned[item] + (1.0 - mix) * overlap[item]
            for item in pool
        }

    pool_set = set(pool)
    raw_co_full = lookup_co_items(anchor_items, pool_set, lookup)
    raw_co = {item: float(raw_co_full.get(item, 0.0)) for item in pool}
    last_anchor = history_items[-1:] if history_items else []
    raw_last_full = lookup_co_items(last_anchor, pool_set, lookup)
    raw_last_co = {item: float(raw_last_full.get(item, 0.0)) for item in pool}
    raw_hist = _hist_affinity_scores(
        pool, history_items, item_embs, decay=hist_decay
    )
    raw_seq = {item: float(seq_scores.get(item, 0.0)) for item in pool}
    raw_cat = _cat_overlap_scores(pool, history_items, item_meta)
    pop = item_pop or {}
    raw_pop = {item: math.log1p(float(pop.get(item, 0.0))) for item in pool}
    raw_markov = markov_scores(pool, history_items, markov_lookup)
    raw_brand = _brand_overlap_scores(pool, history_items, item_meta)

    z_text = zscore_map(raw_text)
    z_co = zscore_map(raw_co)
    z_hist = zscore_map(raw_hist)
    z_seq = zscore_map(raw_seq)
    z_last_co = zscore_map(raw_last_co)
    z_cat = zscore_map(raw_cat)
    z_pop = zscore_map(raw_pop)
    z_markov = zscore_map(raw_markov)
    z_brand = zscore_map(raw_brand)
    z_channels = {"text": z_text, "co": z_co, "hist": z_hist, "seq": z_seq}
    used = _renormalize_weights(z_channels, wcfg)

    phi: dict[str, float] = {item: 0.0 for item in pool}
    for name, w in used.items():
        ch = z_channels[name]
        for item in pool:
            phi[item] += w * ch[item]

    return PotentialScores(
        phi=phi,
        z_text=z_text,
        z_co=z_co,
        z_hist=z_hist,
        z_seq=z_seq,
        z_last_co=z_last_co,
        z_cat=z_cat,
        z_pop=z_pop,
        z_markov=z_markov,
        z_brand=z_brand,
        raw_text=raw_text,
        raw_co=raw_co,
        raw_hist=raw_hist,
        raw_seq=raw_seq,
        raw_last_co=raw_last_co,
        raw_cat=raw_cat,
        raw_pop=raw_pop,
        raw_markov=raw_markov,
        raw_brand=raw_brand,
        weights_used=used,
    )


def greedy_potential_swaps(
    pool_order: list[str],
    phi: dict[str, float],
    *,
    head_n: int = 20,
    focus_k: int = 50,
    tau: float = 0.25,
    gamma: float = 0.0,
    max_swaps: int | None = 10,
) -> tuple[list[str], int]:
    """Swap high-φ focus items into the π¹ head when Δ > γ.

    Focus = top-``focus_k`` by φ over ``pool_order``. Challengers are focus
    items currently outside the head. Each accepted pair swaps positions.
    """
    if not pool_order:
        return [], 0
    order = list(pool_order)
    n_head = min(max(int(head_n), 0), len(order))
    if n_head <= 0:
        return order, 0
    s1_pos = {item: i for i, item in enumerate(order)}
    by_phi = sorted(
        order,
        key=lambda x: (-float(phi.get(x, 0.0)), s1_pos[x]),
    )
    focus = by_phi[: min(max(int(focus_k), 0), len(by_phi))]
    n_swaps = 0
    cap = len(order) if max_swaps is None else max(int(max_swaps), 0)

    while n_swaps < cap:
        head = order[:n_head]
        head_set = set(head)
        pos = {item: i for i, item in enumerate(order)}
        challengers = [c for c in focus if c not in head_set]
        best: tuple[float, str, str] | None = None
        for c in challengers:
            for d in head:
                rank_d = pos[d] + 1
                delta = swap_delta(
                    float(phi.get(c, 0.0)),
                    float(phi.get(d, 0.0)),
                    rank_d,
                    tau=tau,
                    head_n=n_head,
                )
                if delta <= float(gamma):
                    continue
                if best is None or delta > best[0]:
                    best = (delta, c, d)
        if best is None:
            break
        _, c, d = best
        i_c, i_d = pos[c], pos[d]
        order[i_c], order[i_d] = order[i_d], order[i_c]
        n_swaps += 1
    return order, n_swaps
