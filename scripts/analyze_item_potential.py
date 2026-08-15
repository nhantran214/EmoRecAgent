#!/usr/bin/env python3
"""Validate complementary φ(u,i) on the valid split (no LLM).

Reports, vs Stage-1 π¹:
  - AUROC / recall@50 of φ for gold ∈ π¹[:K]
  - whether gold in 21–K lands in φ-focus-50
  - oracle hr@10/@20 if every Δ>γ swap is applied

History = train prefix only on valid (no valid leakage). On test, history =
train+valid and φ_v2 uses fit-on-valid LTR weights. Example::

  "$ERA_PY" scripts/analyze_item_potential.py --config "$CONFIG" --split "$SPLIT"

  "$ERA_PY" scripts/analyze_item_potential.py --config "$CONFIG" --eval-split test
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _auroc(scores: dict[str, float], labels: dict[str, int]) -> float | None:
    """Mann–Whitney AUROC; None if no positive or no negative."""
    pos = [scores[i] for i, y in labels.items() if y == 1 and i in scores]
    neg = [scores[i] for i, y in labels.items() if y == 0 and i in scores]
    if not pos or not neg:
        return None
    # P(score_pos > score_neg) + 0.5 P(eq)
    wins = 0.0
    n = 0
    for p in pos:
        for q in neg:
            n += 1
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / n if n else None


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx = np.argsort(np.argsort(np.asarray(xs, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(ys, dtype=np.float64)))
    rx = rx.astype(np.float64)
    ry = ry.astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if den < 1e-12:
        return None
    return float(np.dot(rx, ry) / den)


def _best_rank(order: list[str], relevant: set[str]) -> int | None:
    pos = {item: i + 1 for i, item in enumerate(order)}
    ranks = [pos[g] for g in relevant if g in pos]
    return min(ranks) if ranks else None


def _hit(order: list[str], relevant: set[str], k: int) -> bool:
    top = set(order[:k])
    return any(g in top for g in relevant)


def _index_tu_by_user(tu_cache: dict) -> dict[str, list]:
    by_user: dict[str, list] = defaultdict(list)
    for row in tu_cache.values():
        by_user[str(row.user_id)].append(row)
    for rows in by_user.values():
        rows.sort(key=lambda r: int(r.query_ts_ms))
    return by_user


def _resolve_t_u(rows_for_user: list, query_ts: int) -> str:
    if not rows_for_user:
        return ""
    best_t = ""
    for row in rows_for_user:
        if int(row.query_ts_ms) <= query_ts:
            best_t = str(row.T_u or "")
        else:
            break
    if best_t:
        return best_t
    # No prefix key (valid ts ≠ cached test ts): use earliest available T_u.
    return str(rows_for_user[0].T_u or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/categories/Beauty_and_Personal_Care.yaml",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="processed split dir (default: data.out_dir from --config)",
    )
    parser.add_argument("--pool-k", type=int, default=300)
    parser.add_argument("--head-n", type=int, default=20)
    parser.add_argument("--focus-k", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--max-swaps", type=int, default=10)
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="0 = all valid users",
    )
    parser.add_argument(
        "--no-alignment",
        action="store_true",
        help="skip MLP(enc(T_u)); text channel = lexical overlap only",
    )
    parser.add_argument(
        "--features-cache",
        default=None,
        help="save/load pooled features so LTR can rerun without Stage-1 "
        "(default: results/<category>/item_potential_valid_features.npz)",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="skip Stage-1 extract; load --features-cache",
    )
    parser.add_argument("--ltr-splits", type=int, default=5)
    parser.add_argument("--ltr-l2", type=float, default=0.1)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.15,
        help="v1.1 φ = z_seq + alpha * residual (0 copies π¹ order in-pool)",
    )
    parser.add_argument(
        "--eval-split",
        choices=("valid", "test"),
        default="valid",
        help="valid = fit/OOF gate; test = rerank π¹[:K] with fit-on-valid weights",
    )
    parser.add_argument(
        "--ltr-weights",
        default="",
        help="npz from --ltr-out; required for --eval-split test",
    )
    parser.add_argument(
        "--ltr-out",
        default=None,
        help="fit-on-all-valid weights for later Stage-2 "
        "(default: tisasrec_align.item_potential_ltr_path from --config)",
    )
    args = parser.parse_args()
    from emorecagent.config import load_config

    _cfg = load_config(args.config)
    if not args.split:
        args.split = _cfg.data.out_dir
    if not args.ltr_out:
        args.ltr_out = _cfg.tisasrec_align.item_potential_ltr_path
    if not args.features_cache:
        args.features_cache = (
            f"results/{_cfg.data.category}/item_potential_valid_features.npz"
        )
    if args.eval_split == "test" and args.reuse_cache:
        raise SystemExit("--reuse-cache is valid-only; omit it for --eval-split test")
    if args.reuse_cache:
        return _run_cached(args)

    from emorecagent.data.loader import load_split_jsonl
    from emorecagent.tisasrec_align.cross_user_lookup import load_lookup
    from emorecagent.tisasrec_align.item_metadata import load_stage2_item_metadata
    from emorecagent.tisasrec_align.item_potential import (
        build_next_item_lookup,
        greedy_potential_swaps,
        load_listwise_npz,
        ltr_feature_matrix,
        mix_backbone_residual,
        rerank_pool_by_phi,
        score_pool_ltr,
        score_pool_potential,
    )
    from emorecagent.tisasrec_align.review_context import (
        item_review_snippets_from_index,
        load_review_text_index,
    )
    from emorecagent.tisasrec_align.stage1_factory import build_stage1_recommender
    from emorecagent.tisasrec_align.tu_cache import load_tu_cache

    cfg = load_config(args.config)
    ta = cfg.tisasrec_align
    split = Path(args.split)
    train = load_split_jsonl(split / "train.jsonl")
    valid = load_split_jsonl(split / "valid.jsonl")
    if not valid:
        raise SystemExit(f"no valid split at {split / 'valid.jsonl'}")
    eval_rows = valid
    if args.eval_split == "test":
        eval_rows = load_split_jsonl(split / "test.jsonl")
        if not eval_rows:
            raise SystemExit(f"no test split at {split / 'test.jsonl'}")
    ltr_w = ltr_mu = ltr_sd = None
    if args.eval_split == "test":
        wpath = Path(args.ltr_weights or args.ltr_out)
        if not wpath.is_file():
            raise SystemExit(f"--eval-split test needs LTR weights at {wpath}")
        ltr_w, ltr_mu, ltr_sd = load_listwise_npz(wpath)
        print(f"loaded φ_v2 weights from {wpath}  dim={ltr_w.shape[0]}")

    # Frozen Stage-1. Valid: history = train only. Test: train+valid (no test leak).
    history_src = train if args.eval_split == "valid" else train + valid
    stage1 = build_stage1_recommender(cfg, history_src, force_stage1_only=True)
    stage1.fit(history_src)
    catalog = stage1.catalog_items()
    keep = set(catalog)

    item_meta = {}
    meta_root = cfg.data.meta_path or cfg.data.inter_path or cfg.data.review_path
    if meta_root:
        try:
            item_meta = load_stage2_item_metadata(meta_root, keep_ids=keep)
        except (FileNotFoundError, ValueError) as exc:
            print(f"warning: item meta unavailable: {exc}")

    review_snippets: dict[str, list[str]] = {}
    if cfg.data.review_path:
        try:
            idx = load_review_text_index(cfg.data.review_path)
            allowed_reviews = {
                (it.user_id, it.item, int(it.timestamp)) for it in history_src
            }
            review_snippets = item_review_snippets_from_index(
                idx,
                keep_ids=keep,
                allowed_reviews=allowed_reviews,
                max_chars=ta.llm_card_max_review_chars,
                max_per_item=max(1, int(ta.llm_card_review_candidates)),
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"warning: review snippets unavailable: {exc}")

    lookup = load_lookup(ta.cross_user_lookup_path)
    tu_cache = load_tu_cache(ta.tu_cache_path)
    tu_by_user = _index_tu_by_user(tu_cache)

    alignment_mlp = None
    text_encoder = None
    if not args.no_alignment:
        import torch

        from emorecagent.tisasrec_align.alignment_mlp import AlignmentMLP
        from emorecagent.tisasrec_align.text_encoder import (
            HashEncoder,
            SentenceTransformerEncoder,
        )

        align_path = Path(ta.alignment_checkpoint_path)
        if align_path.is_file():
            text_encoder = (
                HashEncoder(dim=ta.text_encoder_dim)
                if ta.use_hash_encoder
                else SentenceTransformerEncoder()
            )
            if hasattr(text_encoder, "warm_up"):
                text_encoder.warm_up()
            payload = torch.load(align_path, map_location="cpu", weights_only=False)
            meta = payload.get("meta") or {}
            hidden = int(meta.get("hidden_dim", ta.hidden_units))
            act = str(meta.get("activation", ta.alignment_activation))
            alignment_mlp = AlignmentMLP(
                ta.text_encoder_dim, hidden, activation=act  # type: ignore[arg-type]
            )
            alignment_mlp.load_state_dict(payload["model"])
            alignment_mlp.eval()
            print(f"loaded alignment MLP from {align_path}")
        else:
            print(f"warning: no alignment MLP at {align_path}; text=overlap only")

    train_by_user: dict[str, list] = defaultdict(list)
    train_items: dict[str, set[str]] = defaultdict(set)
    item_pop: dict[str, float] = defaultdict(float)
    for it in train:
        item_pop[str(it.item)] += 1.0
    for it in history_src:
        uid = str(it.user_id)
        train_by_user[uid].append(it)
        train_items[uid].add(str(it.item))
    markov_lookup = build_next_item_lookup(train)
    print(f"markov next-item transitions: {len(markov_lookup)} anchors")

    by_user: dict[str, list] = defaultdict(list)
    for row in eval_rows:
        by_user[str(row.user_id)].append(row)
    uids = sorted(by_user)
    if args.max_users > 0:
        uids = uids[: args.max_users]

    pool_k = int(args.pool_k)
    head_n = int(args.head_n)
    focus_k = int(args.focus_k)

    n = 0
    n_gold_in_pool = 0
    auroc_phi: list[float] = []
    auroc_s1: list[float] = []
    auroc_text: list[float] = []
    auroc_co: list[float] = []
    auroc_hist: list[float] = []
    rec50_phi = 0
    rec50_s1 = 0
    best_phi: list[int] = []
    best_s1: list[int] = []
    spearman_vs_s1: list[float] = []
    # gold in π¹ 21–K
    n_tail = 0
    tail_in_focus = 0
    # oracle
    hr10_s1 = 0
    hr20_s1 = 0
    hr50_s1 = 0
    hr100_s1 = 0
    hr10_phi_focus = 0  # gold in φ top-50 (not a ranking of catalog)
    hr10_oracle = 0
    hr20_oracle = 0
    hr10_v2 = 0
    hr20_v2 = 0
    hr50_v2 = 0
    hr100_v2 = 0
    auroc_v2: list[float] = []
    rec50_v2 = 0
    best_v2: list[int] = []
    tail_in_focus_v2 = 0
    n_swaps_sum = 0
    n_oracle_users_swapped = 0
    p_u_cache: dict[str, np.ndarray] = {}
    feat_X: list[np.ndarray] = []
    feat_y: list[np.ndarray] = []
    feat_g: list[np.ndarray] = []
    feat_items: list[np.ndarray] = []
    feat_ranks: list[np.ndarray] = []
    feat_phi: list[np.ndarray] = []
    feat_phi_bb: list[np.ndarray] = []

    import torch

    def _project_tu(t_u: str) -> np.ndarray | None:
        if alignment_mlp is None or text_encoder is None or not t_u.strip():
            return None
        cached = p_u_cache.get(t_u)
        if cached is not None:
            return cached
        with torch.no_grad():
            emb = text_encoder.encode([t_u], device=torch.device("cpu"))
            vec = alignment_mlp(emb).squeeze(0).detach().cpu().numpy()
        p_u_cache[t_u] = vec
        return vec

    full_emb = None
    item_to_idx = getattr(stage1, "_item_to_idx", None)
    model = getattr(stage1, "_model", None)
    if model is not None and item_to_idx is not None and hasattr(model, "item_embedding"):
        full_emb = model.item_embedding.weight.detach().cpu().numpy()

    def _item_embs(ids: list[str]) -> dict[str, np.ndarray]:
        if full_emb is None or item_to_idx is None:
            if hasattr(stage1, "item_embeddings"):
                return stage1.item_embeddings(ids)
            return {}
        out: dict[str, np.ndarray] = {}
        for item in ids:
            loc = item_to_idx.get(item)
            if loc is None:
                continue
            out[item] = full_emb[int(loc)].astype(np.float64)
        return out

    for uid in uids:
        rows = by_user[uid]
        relevant = {str(r.item) for r in rows}
        query_ts = max(int(r.timestamp) for r in rows)
        hist_events = sorted(
            (it for it in train_by_user.get(uid, []) if int(it.timestamp) < query_ts),
            key=lambda it: (int(it.timestamp), str(it.item)),
        )
        history_items = [str(it.item) for it in hist_events]
        seen = set(train_items.get(uid, set()))
        candidates = [item for item in catalog if item not in seen]
        for g in relevant:
            if g not in candidates:
                candidates.append(g)

        if hasattr(stage1, "prepare_user_query"):
            stage1.prepare_user_query(uid, query_ts)
        ranked = stage1.rank(uid, candidates, query_ts_ms=query_ts)
        pool = ranked[: min(pool_k, len(ranked))]
        if not pool:
            continue
        n += 1
        gold_in_pool = {g for g in relevant if g in pool}
        if gold_in_pool:
            n_gold_in_pool += 1

        seq_scores = stage1.score(uid, pool, query_ts_ms=query_ts)
        need_ids = list(dict.fromkeys(list(pool) + history_items))
        item_embs = _item_embs(need_ids)
        t_u = _resolve_t_u(tu_by_user.get(uid) or [], query_ts)
        p_u = _project_tu(t_u)
        scored = score_pool_potential(
            pool,
            t_u=t_u,
            p_u=p_u,
            item_embs=item_embs,
            seq_scores=seq_scores,
            history_items=history_items,
            anchor_items=history_items,
            lookup=lookup,
            item_meta=item_meta or None,
            review_snippets=review_snippets or None,
            item_pop=item_pop,
            markov_lookup=markov_lookup,
        )
        phi_bb = mix_backbone_residual(pool, scored, alpha=float(args.alpha))
        phi_v2: dict[str, float] | None = None
        by_v2: list[str] | None = None
        if ltr_w is not None and ltr_mu is not None and ltr_sd is not None:
            phi_v2 = score_pool_ltr(
                pool, scored, w=ltr_w, mu=ltr_mu, sd=ltr_sd
            )
            by_v2 = rerank_pool_by_phi(pool, phi_v2)
        if args.eval_split != "test":
            feat_X.append(ltr_feature_matrix(pool, scored))
            feat_y.append(
                np.asarray(
                    [1.0 if it in relevant else 0.0 for it in pool], dtype=np.float64
                )
            )
            feat_g.append(np.full(len(pool), n - 1, dtype=np.int64))
            feat_items.append(np.asarray(pool, dtype=object))
            feat_ranks.append(np.arange(1, len(pool) + 1, dtype=np.int32))
            feat_phi.append(
                np.asarray([float(scored.phi[it]) for it in pool], dtype=np.float64)
            )
            feat_phi_bb.append(
                np.asarray([float(phi_bb[it]) for it in pool], dtype=np.float64)
            )
        labels = {item: 1 if item in relevant else 0 for item in pool}
        # Higher π¹ is better → invert rank as score.
        s1_score = {item: -float(i + 1) for i, item in enumerate(pool)}

        def _add_auroc(bucket: list[float], sc: dict[str, float]) -> None:
            val = _auroc(sc, labels)
            if val is not None:
                bucket.append(val)

        _add_auroc(auroc_phi, scored.phi)
        _add_auroc(auroc_s1, s1_score)
        _add_auroc(auroc_text, scored.z_text)
        _add_auroc(auroc_co, scored.z_co)
        _add_auroc(auroc_hist, scored.z_hist)

        by_phi = sorted(
            pool, key=lambda x: (-scored.phi.get(x, 0.0), s1_score[x])
        )
        if gold_in_pool:
            rec50_phi += int(_hit(by_phi, gold_in_pool, 50))
            rec50_s1 += int(_hit(pool, gold_in_pool, 50))
            bp = _best_rank(by_phi, gold_in_pool)
            bs = _best_rank(pool, gold_in_pool)
            if bp is not None:
                best_phi.append(bp)
            if bs is not None:
                best_s1.append(bs)
            sp = _spearman(
                [scored.phi[i] for i in pool],
                [s1_score[i] for i in pool],
            )
            if sp is not None:
                spearman_vs_s1.append(sp)

        br_s1 = _best_rank(pool, relevant) or 10**9
        if 21 <= br_s1 <= len(pool):
            n_tail += 1
            focus_set = set(by_phi[: min(focus_k, len(by_phi))])
            if any(g in focus_set for g in gold_in_pool):
                tail_in_focus += 1

        hr10_s1 += int(_hit(pool, relevant, 10))
        hr20_s1 += int(_hit(pool, relevant, 20))
        hr50_s1 += int(_hit(pool, relevant, 50))
        hr100_s1 += int(_hit(pool, relevant, 100))
        hr10_phi_focus += int(_hit(by_phi, relevant, 10))
        if phi_v2 is not None and by_v2 is not None:
            _add_auroc(auroc_v2, phi_v2)
            if gold_in_pool:
                rec50_v2 += int(_hit(by_v2, gold_in_pool, 50))
                bv = _best_rank(by_v2, gold_in_pool)
                if bv is not None:
                    best_v2.append(bv)
            if 21 <= br_s1 <= len(pool):
                if any(
                    g in set(by_v2[: min(focus_k, len(by_v2))]) for g in gold_in_pool
                ):
                    tail_in_focus_v2 += 1
            hr10_v2 += int(_hit(by_v2, relevant, 10))
            hr20_v2 += int(_hit(by_v2, relevant, 20))
            hr50_v2 += int(_hit(by_v2, relevant, 50))
            hr100_v2 += int(_hit(by_v2, relevant, 100))

        if args.eval_split == "test":
            if n % 200 == 0:
                print(f"… {n}/{len(uids)}", flush=True)
            continue

        new_order, n_swaps = greedy_potential_swaps(
            pool,
            scored.phi,
            head_n=head_n,
            focus_k=focus_k,
            tau=args.tau,
            gamma=args.gamma,
            max_swaps=args.max_swaps,
        )
        n_swaps_sum += n_swaps
        if n_swaps:
            n_oracle_users_swapped += 1
        hr10_oracle += int(_hit(new_order, relevant, 10))
        hr20_oracle += int(_hit(new_order, relevant, 20))

        if n % 200 == 0:
            print(f"… {n}/{len(uids)}", flush=True)

    def _pct(num: int, den: int) -> str:
        if den <= 0:
            return "n/a"
        return f"{num}/{den} ({100.0 * num / den:.1f}%)"

    print(f"\nn_users={n}  gold∈π¹[:{pool_k}]={_pct(n_gold_in_pool, n)}  split={args.eval_split}")
    if args.eval_split == "test":
        den = max(n, 1)
        print("=== test φ_v2 pool rerank (fit-on-valid weights; no LLM, no swap) ===")
        print(
            f"  AUROC π¹={_mean(auroc_s1):.4f}  φ_v2={_mean(auroc_v2):.4f}  "
            f"Δ={_mean(auroc_v2) - _mean(auroc_s1):+.4f}"
        )
        print("=== recall@50 among users with gold in pool ===")
        print(f"  π¹[:50]: {_pct(rec50_s1, n_gold_in_pool)}")
        print(f"  φ_v2 top-50: {_pct(rec50_v2, n_gold_in_pool)}")
        print(
            f"  best gold rank: π¹ p50={_median(best_s1)}  "
            f"φ_v2 p50={_median(best_v2)}"
        )
        print(f"=== tail gold (π¹ 21–{pool_k}) in φ_v2-focus-{focus_k} ===")
        print(f"  {_pct(tail_in_focus_v2, n_tail)}")
        print("=== hr (full-catalog equivalent for k≤pool; user_mean) ===")
        payload = {
            "split": "test",
            "n_users": n,
            "n_gold_in_pool": n_gold_in_pool,
            "auroc_s1": _mean(auroc_s1),
            "auroc_v2": _mean(auroc_v2),
            "recall50_s1": rec50_s1 / max(n_gold_in_pool, 1),
            "recall50_v2": rec50_v2 / max(n_gold_in_pool, 1),
        }
        for k, s1c, v2c in (
            (10, hr10_s1, hr10_v2),
            (20, hr20_s1, hr20_v2),
            (50, hr50_s1, hr50_v2),
            (100, hr100_s1, hr100_v2),
        ):
            s1v = s1c / den
            v2v = v2c / den
            print(f"  hr@{k:<3} π¹={s1v:.4f}  φ_v2={v2v:.4f}  Δ={v2v - s1v:+.4f}")
            payload[f"hr@{k}_s1"] = s1v
            payload[f"hr@{k}_v2"] = v2v
            payload[f"hr@{k}_delta"] = v2v - s1v
        out_path = Path("results/Beauty_and_Personal_Care/item_potential_v2_test.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
        return 0

    print("=== ranking quality (macro AUROC on pool items; gold=1) ===")
    print(f"  π¹ rank:     {_mean(auroc_s1):.4f}  (n={len(auroc_s1)})")
    print(f"  φ:           {_mean(auroc_phi):.4f}  (n={len(auroc_phi)})")
    print(f"  φ text only: {_mean(auroc_text):.4f}")
    print(f"  φ co only:   {_mean(auroc_co):.4f}")
    print(f"  φ hist only: {_mean(auroc_hist):.4f}")
    print(
        f"  Spearman(φ, −π¹_rank): {_mean(spearman_vs_s1):.3f}  "
        "(high ⇒ φ copies Stage-1)"
    )
    print("=== recall@50 among users with gold in pool ===")
    print(f"  π¹[:50]: {_pct(rec50_s1, n_gold_in_pool)}")
    print(f"  φ top-50: {_pct(rec50_phi, n_gold_in_pool)}")
    print(
        f"  best gold rank: π¹ p50={_median(best_s1)}  φ p50={_median(best_phi)}"
    )
    print(f"=== tail gold (π¹ 21–{pool_k}) in φ-focus-{focus_k} ===")
    print(f"  {_pct(tail_in_focus, n_tail)}")
    print("=== hr (all users; oracle = greedy Δ>γ swaps, no LLM) ===")
    print(
        f"  hr@10  π¹={hr10_s1 / n:.4f}  φ-as-ranker={hr10_phi_focus / n:.4f}  "
        f"oracle_swap={hr10_oracle / n:.4f}  "
        f"Δ_oracle={hr10_oracle / n - hr10_s1 / n:+.4f}"
    )
    print(
        f"  hr@20  π¹={hr20_s1 / n:.4f}  oracle_swap={hr20_oracle / n:.4f}  "
        f"Δ_oracle={hr20_oracle / n - hr20_s1 / n:+.4f}"
    )
    print(
        f"  swaps: users={n_oracle_users_swapped}/{n}  "
        f"mean_swaps={n_swaps_sum / max(n, 1):.2f}  "
        f"tau={args.tau} gamma={args.gamma} head={head_n} focus={focus_k}"
    )
    print(
        "gate: φ must beat π¹ AUROC/recall@50 on tail gold, and oracle Δhr@10>0, "
        "before wiring LLM."
    )
    if feat_X:
        X = np.vstack(feat_X)
        yv = np.concatenate(feat_y)
        gv = np.concatenate(feat_g)
        items_a = np.concatenate(feat_items)
        ranks_a = np.concatenate(feat_ranks)
        phi_a = np.concatenate(feat_phi)
        phi_bb_a = np.concatenate(feat_phi_bb)
        cache_path = Path(args.features_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X,
            y=yv,
            groups=gv,
            items=items_a,
            ranks=ranks_a,
            phi_v1=phi_a,
            phi_v1b=phi_bb_a,
            alpha=np.asarray([float(args.alpha)]),
        )
        print(f"wrote features cache {cache_path}  rows={X.shape[0]}")
        _run_ltr_on_arrays(
            X,
            yv,
            gv,
            items_a,
            ranks_a,
            phi_a,
            phi_bb_a,
            args=args,
        )
    return 0


def _median(xs: list[int]) -> str:
    if not xs:
        return "n/a"
    ys = sorted(xs)
    return str(ys[len(ys) // 2])


def _pct(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{num}/{den} ({100.0 * num / den:.1f}%)"


def _user_blocks(
    groups: np.ndarray,
) -> list[tuple[int, int]]:
    n = int(groups.shape[0])
    if n == 0:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    cur = groups[0]
    for i in range(1, n):
        if groups[i] != cur:
            out.append((start, i))
            start = i
            cur = groups[i]
    out.append((start, n))
    return out


def _eval_scores_on_blocks(
    *,
    blocks: list[tuple[int, int]],
    items: np.ndarray,
    y: np.ndarray,
    ranks: np.ndarray,
    scores: np.ndarray,
    head_n: int,
    focus_k: int,
    tau: float,
    gamma: float,
    max_swaps: int,
    label: str,
    silent: bool = False,
    skip_swap: bool = False,
) -> dict[str, float]:
    from emorecagent.tisasrec_align.item_potential import greedy_potential_swaps

    n = 0
    n_gold = 0
    aurocs: list[float] = []
    aurocs_s1: list[float] = []
    rec50 = 0
    rec50_s1 = 0
    best: list[int] = []
    best_s1: list[int] = []
    n_tail = 0
    tail_in_focus = 0
    hr10 = 0
    hr20 = 0
    hr10_s1 = 0
    hr20_s1 = 0
    hr10_swap = 0
    hr20_swap = 0
    n_swaps_sum = 0
    n_swapped_users = 0

    for a, b in blocks:
        n += 1
        sl_items = [str(x) for x in items[a:b]]
        sl_y = y[a:b]
        sl_rank = ranks[a:b]
        sl_s = scores[a:b]
        relevant = {sl_items[i] for i, v in enumerate(sl_y) if v > 0}
        if not sl_items:
            continue
        order_s1 = [p for _, p in sorted(zip(sl_rank.tolist(), sl_items))]
        order_phi = [
            p
            for _, _, p in sorted(
                zip((-sl_s).tolist(), sl_rank.tolist(), sl_items)
            )
        ]
        labels = {sl_items[i]: int(sl_y[i] > 0) for i in range(len(sl_items))}
        sc = {sl_items[i]: float(sl_s[i]) for i in range(len(sl_items))}
        s1sc = {sl_items[i]: -float(sl_rank[i]) for i in range(len(sl_items))}
        gold_in_pool = relevant.intersection(sl_items)
        if gold_in_pool:
            n_gold += 1
            av = _auroc(sc, labels)
            if av is not None:
                aurocs.append(av)
            av1 = _auroc(s1sc, labels)
            if av1 is not None:
                aurocs_s1.append(av1)
            rec50 += int(_hit(order_phi, gold_in_pool, 50))
            rec50_s1 += int(_hit(order_s1, gold_in_pool, 50))
            bp = _best_rank(order_phi, gold_in_pool)
            bs = _best_rank(order_s1, gold_in_pool)
            if bp is not None:
                best.append(bp)
            if bs is not None:
                best_s1.append(bs)
        br = _best_rank(order_s1, relevant) or 10**9
        if 21 <= br <= len(order_s1):
            n_tail += 1
            if any(g in set(order_phi[: min(focus_k, len(order_phi))]) for g in gold_in_pool):
                tail_in_focus += 1
        hr10 += int(_hit(order_phi, relevant, 10))
        hr20 += int(_hit(order_phi, relevant, 20))
        hr10_s1 += int(_hit(order_s1, relevant, 10))
        hr20_s1 += int(_hit(order_s1, relevant, 20))
        n_swaps = 0
        if not skip_swap:
            phi_map = {sl_items[i]: float(sl_s[i]) for i in range(len(sl_items))}
            new_order, n_swaps = greedy_potential_swaps(
                order_s1,
                phi_map,
                head_n=head_n,
                focus_k=focus_k,
                tau=tau,
                gamma=gamma,
                max_swaps=max_swaps,
            )
            hr10_swap += int(_hit(new_order, relevant, 10))
            hr20_swap += int(_hit(new_order, relevant, 20))
        else:
            hr10_swap += int(_hit(order_s1, relevant, 10))
            hr20_swap += int(_hit(order_s1, relevant, 20))
        n_swaps_sum += n_swaps
        if n_swaps:
            n_swapped_users += 1

    den = max(n, 1)
    gden = max(n_gold, 1)
    out = {
        "n": float(n),
        "n_gold": float(n_gold),
        "auroc": _mean(aurocs),
        "auroc_s1": _mean(aurocs_s1),
        "rec50": rec50 / gden,
        "rec50_s1": rec50_s1 / gden,
        "tail_focus": (tail_in_focus / n_tail) if n_tail else 0.0,
        "n_tail": float(n_tail),
        "hr10": hr10 / den,
        "hr10_s1": hr10_s1 / den,
        "hr20": hr20 / den,
        "hr20_s1": hr20_s1 / den,
        "hr10_swap": hr10_swap / den,
        "d_hr10_swap": hr10_swap / den - hr10_s1 / den,
        "mean_swaps": n_swaps_sum / den,
    }
    if not silent:
        print(f"\n=== {label} (n_users={n}, gold∈pool={_pct(n_gold, n)}) ===")
        print(
            f"  AUROC:     {out['auroc']:.4f}  vs π¹ {out['auroc_s1']:.4f}  "
            f"Δ={out['auroc'] - out['auroc_s1']:+.4f}"
        )
        print(
            f"  recall@50: {_pct(rec50, n_gold)}  vs π¹ {_pct(rec50_s1, n_gold)}  "
            f"best-gold p50={_median(best)} (π¹ p50={_median(best_s1)})"
        )
        print(f"  tail gold in focus-{focus_k}: {_pct(tail_in_focus, n_tail)}")
        print(
            f"  hr@10 ranker={out['hr10']:.4f}  π¹={out['hr10_s1']:.4f}  "
            f"Δ={out['hr10'] - out['hr10_s1']:+.4f}"
        )
        print(
            f"  oracle_swap hr@10={out['hr10_swap']:.4f}  "
            f"Δ={out['d_hr10_swap']:+.4f}  "
            f"mean_swaps={out['mean_swaps']:.2f} users_swapped={n_swapped_users}/{n}"
        )
    return out


def _blend_rank_residual(
    blocks: list[tuple[int, int]],
    ranks: np.ndarray,
    residual: np.ndarray,
    beta: float,
) -> np.ndarray:
    out = np.zeros_like(residual, dtype=np.float64)
    for a, b in blocks:
        r = np.maximum(ranks[a:b].astype(np.float64), 1.0)
        z = residual[a:b].astype(np.float64)
        sd = float(z.std())
        zz = (z - float(z.mean())) / sd if sd > 1e-12 else np.zeros_like(z)
        out[a:b] = -np.log(r) + float(beta) * zz
    return out


def _print_gate(name: str, m: dict[str, float]) -> bool:
    beat_auroc = m["auroc"] > m["auroc_s1"] + 1e-4
    beat_rec = m["rec50"] > m["rec50_s1"] + 1e-4
    beat_hr = m["d_hr10_swap"] > 0.0
    ok = beat_auroc and beat_rec
    print(
        f"GATE {name}: AUROC {'>' if beat_auroc else '≤'} π¹, "
        f"recall@50 {'>' if beat_rec else '≤'} π¹, "
        f"oracle Δhr@10={m['d_hr10_swap']:+.4f} "
        f"→ {'PASS' if ok else 'FAIL'}"
        + (" (swap also +)" if ok and beat_hr else "")
    )
    return ok


def _error_analysis_tail(
    X: np.ndarray,
    y: np.ndarray,
    ranks: np.ndarray,
    groups: np.ndarray,
) -> None:
    """Tail gold (rank 21–300) vs π¹[:10] mean on v1 cache columns 7/8 (cat/pop)."""
    if X.shape[1] < 9:
        return
    z_cat = X[:, 7]
    z_pop = X[:, 8]
    n_head = n_tail = 0
    cat_gt = pop_lt = 0
    cat_tail: list[float] = []
    pop_tail: list[float] = []
    cat_head: list[float] = []
    pop_head: list[float] = []
    for a, b in _user_blocks(groups):
        yy = y[a:b]
        rr = ranks[a:b]
        if not np.any(yy > 0):
            continue
        gi = int(np.argmin(np.where(yy > 0, rr, 10**9)))
        gr = int(rr[gi])
        if gr <= 10:
            n_head += 1
            continue
        if gr > 300:
            continue
        n_tail += 1
        cat_tail.append(float(z_cat[a + gi]))
        pop_tail.append(float(z_pop[a + gi]))
        hm = rr <= 10
        if np.any(hm):
            ch = float(z_cat[a:b][hm].mean())
            ph = float(z_pop[a:b][hm].mean())
            cat_head.append(ch)
            pop_head.append(ph)
            if z_cat[a + gi] > ch:
                cat_gt += 1
            if z_pop[a + gi] < ph:
                pop_lt += 1
    print("\n--- tail-gold vs π¹[:10] (valid cache) ---")
    print(f"  gold in π¹[:10]: {n_head}   gold rank 21–300: {n_tail}")
    if n_tail and cat_head:
        print(
            f"  mean z_cat: tail-gold={_mean(cat_tail):+.3f}  "
            f"head={_mean(cat_head):+.3f}"
        )
        print(
            f"  mean z_pop: tail-gold={_mean(pop_tail):+.3f}  "
            f"head={_mean(pop_head):+.3f}"
        )
        print(
            f"  tail-gold z_cat > head mean: {cat_gt}/{n_tail} "
            f"({100.0 * cat_gt / n_tail:.1f}%)"
        )
        print(
            f"  tail-gold z_pop < head mean: {pop_lt}/{n_tail} "
            f"({100.0 * pop_lt / n_tail:.1f}%)"
        )


def _run_ltr_on_arrays(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    items: np.ndarray,
    ranks: np.ndarray,
    phi_v1: np.ndarray,
    phi_v1b: np.ndarray | None,
    *,
    args: argparse.Namespace,
) -> None:
    from emorecagent.tisasrec_align.item_potential import (
        LTR_FEATURE_NAMES,
        fit_listwise_weights,
        oof_listwise_scores,
    )

    blocks = _user_blocks(groups)
    kw = dict(
        blocks=blocks,
        items=items,
        y=y,
        ranks=ranks,
        head_n=int(args.head_n),
        focus_k=int(args.focus_k),
        tau=float(args.tau),
        gamma=float(args.gamma),
        max_swaps=int(args.max_swaps),
    )
    print("\n--- valid-gold channel (user-grouped OOF; no LLM) ---")
    m_v1 = _eval_scores_on_blocks(
        scores=phi_v1,
        label="φ_v1 hand mix (90% residual / 10% seq)",
        **kw,
    )
    _print_gate("φ_v1", m_v1)
    if phi_v1b is not None:
        m_bb = _eval_scores_on_blocks(
            scores=phi_v1b,
            label=f"φ_v1.1 backbone  z_seq + α={args.alpha:g}·residual",
            **kw,
        )
        _print_gate("φ_v1.1", m_bb)

    print("\nfitting listwise LTR OOF (π¹ rank + residual)…", flush=True)
    oof = oof_listwise_scores(
        X,
        y,
        groups,
        n_splits=int(args.ltr_splits),
        l2=float(args.ltr_l2),
        drop_rank=False,
        seed=42,
    )
    m_ltr = _eval_scores_on_blocks(
        scores=oof,
        label="φ_v2 listwise LTR OOF (π¹ rank + residual)",
        **kw,
    )
    _print_gate("φ_v2 LTR", m_ltr)

    print("\nfitting residual-only LTR OOF…", flush=True)
    oof_nr = oof_listwise_scores(
        X,
        y,
        groups,
        n_splits=int(args.ltr_splits),
        l2=float(args.ltr_l2),
        drop_rank=True,
        seed=42,
    )
    m_nr = _eval_scores_on_blocks(
        scores=oof_nr,
        label="LTR OOF residual-only (no π¹ rank / z_seq)",
        **kw,
    )
    _print_gate("residual-only", m_nr)

    print("\n--- blend −log(rank) + β · z(residual OOF), per-user z ---")
    best_beta = 0.0
    best_m: dict[str, float] | None = None
    for beta in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
        blended = _blend_rank_residual(blocks, ranks, oof_nr, beta)
        m = _eval_scores_on_blocks(
            scores=blended,
            label=f"blend β={beta:g}",
            silent=True,
            skip_swap=True,
            **kw,
        )
        print(
            f"  β={beta:<4g}  AUROC={m['auroc']:.4f} (Δ{m['auroc']-m['auroc_s1']:+.4f})  "
            f"R@50={m['rec50']:.3f} (Δ{m['rec50']-m['rec50_s1']:+.3f})  "
            f"swapΔhr@10={m['d_hr10_swap']:+.4f}"
        )
        key = (m["rec50"] - m["rec50_s1"], m["auroc"] - m["auroc_s1"], m["d_hr10_swap"])
        if best_m is None or key > (
            best_m["rec50"] - best_m["rec50_s1"],
            best_m["auroc"] - best_m["auroc_s1"],
            best_m["d_hr10_swap"],
        ):
            best_beta = beta
            best_m = m
    assert best_m is not None
    blended = _blend_rank_residual(blocks, ranks, oof_nr, best_beta)
    m_blend = _eval_scores_on_blocks(
        scores=blended,
        label=f"φ_v2 blend −log(rank)+β={best_beta:g}·z(residual OOF)",
        **kw,
    )
    _print_gate(f"blend β={best_beta:g}", m_blend)

    _error_analysis_tail(X, y, ranks, groups)

    from emorecagent.tisasrec_align.item_potential import (
        CACHE_V1_DIM,
        augment_v1_cache_features,
        oof_hgb_scores,
    )

    X_hgb = X
    if int(X.shape[1]) == CACHE_V1_DIM:
        X_hgb = augment_v1_cache_features(X, ranks, groups)
        print(
            f"\naugmented v1 cache {X.shape[1]} → {X_hgb.shape[1]} cols "
            "(seq_gap_r10, z_cat_unpop, z_cat_unpop_tail)",
            flush=True,
        )
    print("\nfitting HGB residual-only OOF (drop rank/z_seq)…", flush=True)
    hgb_resid = oof_hgb_scores(
        X_hgb, y, groups, n_splits=int(args.ltr_splits), drop_rank=True, seed=42
    )
    print("--- blend −log(rank) + β · z(HGB residual OOF) ---")
    best_hgb_beta = 0.0
    best_hgb: dict[str, float] | None = None
    for beta in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
        blended = _blend_rank_residual(blocks, ranks, hgb_resid, beta)
        m = _eval_scores_on_blocks(
            scores=blended,
            label=f"hgb β={beta:g}",
            silent=True,
            skip_swap=True,
            **kw,
        )
        print(
            f"  β={beta:<4g}  AUROC={m['auroc']:.4f} (Δ{m['auroc']-m['auroc_s1']:+.4f})  "
            f"R@50={m['rec50']:.3f} (Δ{m['rec50']-m['rec50_s1']:+.3f})  "
            f"hr@10={m['hr10']:.4f} (Δ{m['hr10']-m['hr10_s1']:+.4f})"
        )
        key = (m["rec50"] - m["rec50_s1"], m["auroc"] - m["auroc_s1"], m["hr10"] - m["hr10_s1"])
        if best_hgb is None or key > (
            best_hgb["rec50"] - best_hgb["rec50_s1"],
            best_hgb["auroc"] - best_hgb["auroc_s1"],
            best_hgb["hr10"] - best_hgb["hr10_s1"],
        ):
            best_hgb_beta = beta
            best_hgb = m
    assert best_hgb is not None
    hgb_blend = _blend_rank_residual(blocks, ranks, hgb_resid, best_hgb_beta)
    m_hgb = _eval_scores_on_blocks(
        scores=hgb_blend,
        label=f"φ_v3 HGB blend −log(rank)+β={best_hgb_beta:g}·z(HGB residual)",
        skip_swap=True,
        **kw,
    )
    _print_gate(f"HGB blend β={best_hgb_beta:g}", m_hgb)

    print("\nfitting HGB OOF with rank features…", flush=True)
    hgb_full = oof_hgb_scores(
        X_hgb, y, groups, n_splits=int(args.ltr_splits), drop_rank=False, seed=42
    )
    m_hgb_full = _eval_scores_on_blocks(
        scores=hgb_full,
        label="φ_v3 HGB OOF (rank + residual, nonlinear)",
        skip_swap=True,
        **kw,
    )
    _print_gate("HGB full", m_hgb_full)

    w, mu, sd = fit_listwise_weights(
        X, y, groups, l2=float(args.ltr_l2), drop_rank=False
    )
    out = Path(args.ltr_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        w=w,
        mu=mu,
        sd=sd,
        names=np.asarray(LTR_FEATURE_NAMES),
        best_beta=np.asarray([best_beta]),
    )
    print(f"\nfit-on-all-valid weights → {out}  dim={len(w)}")
    names = (
        LTR_FEATURE_NAMES[: len(w)]
        if len(w) <= len(LTR_FEATURE_NAMES)
        else tuple(f"f{i}" for i in range(len(w)))
    )
    if int(X.shape[1]) == 14:
        names = (
            "neg_log_rank",
            "inv_rank",
            "z_seq",
            "z_text",
            "z_co",
            "z_hist",
            "z_last_co",
            "z_cat",
            "z_pop",
            "z_text_tail",
            "z_co_tail",
            "z_hist_tail",
            "z_last_co_tail",
            "z_cat_tail",
        )
    for name, coef in zip(names, w.tolist()):
        print(f"  {name:16s} {coef:+.4f}")
    print(
        "OOF metrics are the gate. Fit-on-all weights are for later test/Stage-2 only."
    )
    print("Do not wire LLM unless a channel above PASSes AUROC and recall@50 vs π¹.")


def _run_cached(args: argparse.Namespace) -> int:
    path = Path(args.features_cache)
    if not path.is_file():
        raise SystemExit(f"--reuse-cache but missing {path}")
    data = np.load(path, allow_pickle=True)
    print(f"loaded features cache {path}  rows={data['X'].shape[0]}")
    phi_bb = data["phi_v1b"] if "phi_v1b" in data.files else None
    _run_ltr_on_arrays(
        data["X"],
        data["y"],
        data["groups"],
        data["items"],
        data["ranks"],
        data["phi_v1"],
        phi_bb,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
