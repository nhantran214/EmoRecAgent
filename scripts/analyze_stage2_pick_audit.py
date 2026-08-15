#!/usr/bin/env python3
"""Audit Stage-2 promote_swap pick quality (near-miss shortlist vs hybrid gate).

Two modes:

1) **Metrics-only** (always): helped/hurt + near-miss conversion from experiment
   JSONs — no Stage-1 reload.

2) **Shortlist coverage** (``--config`` + ``--split``): reload frozen Stage-1 +
   item cards / T_u, and for near-miss users report whether gold was in the
   LLM pool / lexical narrow shortlist, and whether the hybrid lexical gate
   would allow promoting gold over the displacee. Does **not** call the LLM
   (cannot recover the exact scorecard pick from a past run).

Example::

  python3 scripts/analyze_stage2_pick_audit.py \\
    --baseline results/.../emorecagent_align_option_b_stage1_only.json \\
    --fused results/.../emorecagent_align_option_b.json

  python3 scripts/analyze_stage2_pick_audit.py \\
    --baseline ..._stage1_only.json --fused ..._option_b.json \\
    --config configs/categories/Beauty_and_Personal_Care.yaml \\
    --split data/processed/Beauty_and_Personal_Care
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as scripts/*.py without install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _per_user_lists(payload: dict) -> tuple[list[str], dict[str, list[float]]]:
    user_ids = [str(u) for u in (payload.get("user_ids") or [])]
    per_user = payload.get("per_user") or {}
    out: dict[str, list[float]] = {}
    for key, vals in per_user.items():
        if not isinstance(vals, list):
            continue
        if user_ids and len(vals) != len(user_ids):
            raise SystemExit(
                f"{key}: len(per_user)={len(vals)} != len(user_ids)={len(user_ids)}"
            )
        out[key] = [float(x) for x in vals]
    if not user_ids and "hr@10" in out:
        user_ids = [str(i) for i in range(len(out["hr@10"]))]
    return user_ids, out


def _print_metrics_audit(base: dict, fused: dict) -> None:
    uids, pu_b = _per_user_lists(base)
    _, pu_f = _per_user_lists(fused)
    h10 = pu_b["hr@10"]
    h20 = pu_b["hr@20"]
    h50 = pu_b.get("hr@50") or []
    f10 = pu_f["hr@10"]
    f20 = pu_f.get("hr@20") or []
    n = len(h10)

    helped = sum(1 for a, b in zip(h10, f10) if a == 0 and b > 0)
    hurt = sum(1 for a, b in zip(h10, f10) if a > 0 and b == 0)
    nm_11_20 = [i for i, (a, b) in enumerate(zip(h10, h20)) if a == 0 and b > 0]
    nm_21_50 = [
        i
        for i, (a, b) in enumerate(zip(h20, h50))
        if h50 and a == 0 and b > 0
    ]

    def _cohort_help(idxs: list[int]) -> tuple[int, int, float]:
        if not idxs:
            return 0, 0, 0.0
        h = sum(1 for i in idxs if h10[i] == 0 and f10[i] > 0)
        # "rescued" among near-miss: stage2 hr@10 hit
        hit = sum(1 for i in idxs if f10[i] > 0)
        return h, hit, hit / len(idxs)

    h_nm, hit_nm, rate_nm = _cohort_help(nm_11_20)
    h_nm2, hit_nm2, rate_nm2 = _cohort_help(nm_21_50)

    meta = (fused.get("metadata") or {}).get("tisasrec_align") or {}
    print("=== Metrics audit (no Stage-1 reload) ===")
    print(f"n_users={n}")
    print(
        f"hr@10: stage1={_mean(h10):.4f} stage2={_mean(f10):.4f} "
        f"delta={_mean(f10) - _mean(h10):+.4f}"
    )
    print(f"flips: helped={helped} hurt={hurt} net={helped - hurt:+d}")
    print(
        f"near-miss 11–20: n={len(nm_11_20)} stage2_hr@10_hits={hit_nm} "
        f"rescue_rate={rate_nm:.3f} (helped_from_zero={h_nm})"
    )
    if h50:
        print(
            f"near-miss 21–50: n={len(nm_21_50)} stage2_hr@10_hits={hit_nm2} "
            f"rescue_rate={rate_nm2:.3f} (helped_from_zero={h_nm2})"
        )
    if f20:
        # Did Stage-2 push near-miss 11–20 out of top-20 without entering top-10?
        lost20 = sum(
            1 for i in nm_11_20 if h20[i] > 0 and f20[i] == 0
        )
        print(f"near-miss 11–20 lost hr@20 under Stage-2: {lost20}")
    print(
        "metadata: "
        f"swaps={meta.get('n_stage2_swaps')} "
        f"empty={meta.get('n_stage2_empty_picks')} "
        f"hybrid_blocked={meta.get('n_stage2_hybrid_blocked')} "
        f"hybrid_first_filtered={meta.get('n_stage2_hybrid_first_filtered')} "
        f"lexical_argmax={meta.get('n_stage2_lexical_argmax')} "
        f"lexical_first={meta.get('n_stage2_lexical_first')} "
        f"pick_mode={meta.get('llm_pick_mode')} "
        f"depth={meta.get('llm_reason_depth')} "
        f"hybrid_gate={meta.get('llm_hybrid_gate_enabled')} "
        f"hybrid_first={meta.get('llm_hybrid_first_enabled')}"
    )
    print(
        "interpretation: high swaps + low rescue_rate ⇒ scorecard picks are "
        "noisy; shortlist miss vs pick miss needs --config/--split below."
    )


def _shortlist_audit(
    *,
    config_path: str,
    split_dir: Path,
    base: dict,
    fused: dict,
    max_users: int,
) -> None:
    from emorecagent.config import load_config
    from emorecagent.data.loader import load_split_jsonl
    from emorecagent.tisasrec_align.item_metadata import load_stage2_item_metadata
    from emorecagent.tisasrec_align.review_context import (
        item_review_snippets_from_index,
        load_review_text_index,
    )
    from emorecagent.tisasrec_align.stage1_factory import build_stage1_recommender
    from emorecagent.tisasrec_align.stage2_reason_promote import (
        hybrid_lexical_allows,
        match_snippets_to_tu,
        narrow_llm_shortlist,
    )
    from emorecagent.tisasrec_align.tu_cache import cache_key, load_tu_cache

    cfg = load_config(config_path)
    ta = cfg.tisasrec_align
    train = load_split_jsonl(split_dir / "train.jsonl")
    valid_path = split_dir / "valid.jsonl"
    valid = load_split_jsonl(valid_path) if valid_path.exists() else []
    test = load_split_jsonl(split_dir / "test.jsonl")
    fit_rows = list(train)
    if ta.test_history == "train_valid" and valid:
        fit_rows = list(train) + list(valid)

    print("\n=== Shortlist / hybrid coverage (Stage-1 reload, no LLM) ===")
    stage1 = build_stage1_recommender(cfg, fit_rows, force_stage1_only=True)
    stage1.fit(fit_rows)
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
    if ta.llm_card_review_snippets and cfg.data.review_path:
        try:
            idx = load_review_text_index(cfg.data.review_path)
            allowed_reviews = {
                (it.user_id, it.item, int(it.timestamp)) for it in fit_rows
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

    tu_cache = load_tu_cache(ta.tu_cache_path)

    # Group test relevants + query timestamp (max ts among user test rows).
    by_user: dict[str, list] = defaultdict(list)
    for row in test:
        by_user[str(row.user_id)].append(row)

    uids, pu_b = _per_user_lists(base)
    _, pu_f = _per_user_lists(fused)
    h10, h20 = pu_b["hr@10"], pu_b["hr@20"]
    f10 = pu_f["hr@10"]
    nm_idx = [i for i, (a, b) in enumerate(zip(h10, h20)) if a == 0 and b > 0]
    if max_users > 0:
        nm_idx = nm_idx[: max_users]

    protect_n = int(ta.llm_protect_n)
    promote_k = int(ta.llm_promote_k)
    narrow_cap = int(ta.llm_narrow_cap)
    pool_k = int(ta.rerank_pool_k)
    llm_cap = int(ta.llm_pool_cap)
    head_end = protect_n + promote_k

    n = 0
    gold_in_pool = 0
    gold_in_llm_cap = 0
    gold_in_narrow = 0
    gold_hybrid_ok = 0
    gold_hybrid_first_eligible = 0  # in narrow ∩ hybrid_ok (what hybrid-first shows LLM)
    shortlist_miss = 0
    hybrid_miss = 0  # in narrow but fails hybrid
    pick_miss = 0  # hybrid-first eligible but Stage-2 did not rescue
    stage2_rescued = 0
    rank_sum = 0
    rank_n = 0

    for i in nm_idx:
        uid = uids[i]
        rows = by_user.get(uid) or []
        if not rows:
            continue
        relevants = {str(r.item) for r in rows}
        query_ts = max(int(r.timestamp) for r in rows)
        stage1.prepare_user_query(uid, query_ts)
        ranked = stage1.rank(uid, catalog, query_ts_ms=query_ts)
        # Any relevant in ranks 11–20 (hr@20 hit with hr@10 miss); prefer best.
        band_hits: list[tuple[int, str]] = []
        for rnk, item in enumerate(ranked, start=1):
            if rnk > 20:
                break
            if item in relevants and rnk >= 11:
                band_hits.append((rnk, item))
        if not band_hits:
            # Metric said near-miss but reload disagrees — count separately.
            continue
        gold_rank, gold_id = min(band_hits, key=lambda t: t[0])

        n += 1
        rank_sum += gold_rank
        rank_n += 1
        rescued = f10[i] > 0
        if rescued:
            stage2_rescued += 1

        pool = ranked[: min(pool_k, len(ranked))]
        ranks = {item: r + 1 for r, item in enumerate(ranked)}
        in_pool = gold_id in set(pool)
        gold_in_pool += int(in_pool)
        # Runtime narrows inside the LLM shortlist (C), not the full K pool.
        llm_subset = list(pool[: min(llm_cap, len(pool))])
        in_llm = gold_id in set(llm_subset)
        gold_in_llm_cap += int(in_llm)

        t_u = ""
        row = tu_cache.get(cache_key(uid, query_ts))
        if row is not None:
            t_u = str(row.T_u or "")

        matched = match_snippets_to_tu(
            review_snippets,
            t_u,
            item_ids=list(llm_subset),
            max_chars=ta.llm_card_max_review_chars,
        )
        shortlist = narrow_llm_shortlist(
            llm_subset,
            t_u=t_u,
            stage1_ranks=ranks,
            item_meta=item_meta or None,
            review_snippets=matched or review_snippets,
            protect_n=protect_n,
            promote_k=promote_k,
            narrow_cap=narrow_cap,
        )
        head_set = set(llm_subset[: min(head_end, len(llm_subset))])
        eligible = [x for x in shortlist if x not in head_set]
        in_narrow = gold_id in set(eligible)
        gold_in_narrow += int(in_narrow)

        displacee = list(llm_subset[protect_n : min(head_end, len(llm_subset))])
        hybrid_ok = hybrid_lexical_allows(
            gold_id,
            displacee_ids=displacee,
            t_u=t_u,
            preference_facts=None,
            item_meta=item_meta or None,
            review_snippets=matched or review_snippets,
            stage1_ranks=ranks,
            overlap_delta=int(ta.llm_hybrid_overlap_delta),
            overlap_delta_out_of_band=int(ta.llm_hybrid_overlap_delta_out_of_band),
            rank_lo=int(ta.llm_hybrid_rank_lo),
            rank_hi=int(ta.llm_hybrid_rank_hi),
            min_overlap=int(getattr(ta, "llm_hybrid_min_overlap", 0) or 0),
        )
        gold_hybrid_ok += int(hybrid_ok)

        if not in_narrow:
            shortlist_miss += 1
        elif not hybrid_ok:
            hybrid_miss += 1
        else:
            gold_hybrid_first_eligible += 1
            if not rescued:
                pick_miss += 1

    if n == 0:
        print("no consistent near-miss 11–20 users found for shortlist audit")
        return

    def pct(x: int) -> str:
        return f"{100.0 * x / n:.1f}%"

    print(f"audited near-miss 11–20 users: {n} (max_users={max_users})")
    print(f"  mean gold π¹ rank: {rank_sum / max(rank_n, 1):.1f}")
    print(f"  gold ∈ π¹[:K={pool_k}]:     {gold_in_pool}/{n} ({pct(gold_in_pool)})")
    print(f"  gold ∈ LLM cap C={llm_cap}: {gold_in_llm_cap}/{n} ({pct(gold_in_llm_cap)})")
    print(
        f"  gold ∈ narrow eligible:   {gold_in_narrow}/{n} ({pct(gold_in_narrow)}) "
        f"— shortlist miss if low"
    )
    print(
        f"  gold passes hybrid gate:  {gold_hybrid_ok}/{n} ({pct(gold_hybrid_ok)}) "
        f"— lexical signal vs displacee (T_u only; no extract facts)"
    )
    print(
        f"  gold hybrid-first eligible (narrow ∩ hybrid): "
        f"{gold_hybrid_first_eligible}/{n} ({pct(gold_hybrid_first_eligible)})"
    )
    print(
        f"  Stage-2 actually rescued (hr@10): {stage2_rescued}/{n} "
        f"({pct(stage2_rescued)})"
    )
    print("\n=== Funnel decomposition (near-miss 11–20) ===")
    print(f"  shortlist_miss (not in narrow):     {shortlist_miss}/{n} ({pct(shortlist_miss)})")
    print(f"  hybrid_miss (narrow but fail gate): {hybrid_miss}/{n} ({pct(hybrid_miss)})")
    print(
        f"  pick_miss (eligible, not rescued):  {pick_miss}/{n} ({pct(pick_miss)}) "
        f"— LLM chose other / empty / gate / skipped"
    )
    print(
        f"  rescued among eligible: "
        f"{gold_hybrid_first_eligible - pick_miss}/{gold_hybrid_first_eligible} "
        f"({100.0 * (gold_hybrid_first_eligible - pick_miss) / max(gold_hybrid_first_eligible, 1):.1f}% "
        f"of eligible)"
    )
    print(
        "diagnosis: "
        "shortlist_miss = gold not in narrow; "
        "hybrid_miss = gold loses lexical margin vs displacee; "
        "pick_miss = gold was hybrid-first-visible but Stage-2 did not promote it."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Stage-1-only JSON")
    parser.add_argument("--fused", required=True, help="Stage-2 fused JSON")
    parser.add_argument(
        "--config",
        default="",
        help="optional category YAML for shortlist/hybrid coverage audit",
    )
    parser.add_argument(
        "--split",
        default="",
        help="processed split dir (train/test[/valid].jsonl); required with --config",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=300,
        help="cap near-miss users for Stage-1 shortlist audit (0 = all)",
    )
    args = parser.parse_args()

    base = _load(Path(args.baseline))
    fused = _load(Path(args.fused))
    _print_metrics_audit(base, fused)

    if args.config or args.split:
        if not args.config or not args.split:
            raise SystemExit("--config and --split must be provided together")
        _shortlist_audit(
            config_path=args.config,
            split_dir=Path(args.split),
            base=base,
            fused=fused,
            max_users=int(args.max_users),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
