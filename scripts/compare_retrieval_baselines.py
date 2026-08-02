#!/usr/bin/env python3
"""Compare retrieval baselines (popularity vs HetTiSASRec valid pool@50)."""

from __future__ import annotations

import argparse
import sys

from emorecagent.config import load_config
from emorecagent.hettisasrec.aspect_graph import AspectGraphBundle
from emorecagent.hettisasrec.model import HetTiSASRecArgs, HetTiSASRecModel
from emorecagent.hettisasrec.sequence_data import (
    build_train_pairs,
    build_valid_eval_cases,
)
from emorecagent.hettisasrec.train import load_train_valid_from_config
from emorecagent.hettisasrec.valid_eval import (
    evaluate_valid_cases,
    popularity_pool_recall,
)
from emorecagent.sequential.id_maps import build_id_maps_from_interactions


def _resolve_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pure-tisasrec", action="store_true")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("torch required (pip install -e '.[torch]')", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    ht = cfg.hettisasrec
    train, valid = load_train_valid_from_config(cfg.data.out_dir)
    id_maps = build_id_maps_from_interactions(train, valid)
    graph = AspectGraphBundle.load(ht.aspect_graph_path)

    valid_cases = build_valid_eval_cases(train, valid, id_maps)
    train_pairs = build_train_pairs(train, id_maps)
    device = _resolve_device(ht.device)

    pop = popularity_pool_recall(
        valid_cases,
        train_pairs,
        list(graph.item_ids),
        pool_size=ht.pool_size,
        max_pairs=ht.valid_eval_max_pairs,
        seed=cfg.experiment.seed,
        mask_train_seen=ht.valid_mask_train_seen,
    )
    print(f"[compare_retrieval] popularity pool@{ht.pool_size}={pop:.4f}")

    use_aspect = ht.use_aspect_enrichment and not args.pure_tisasrec
    model_args = HetTiSASRecArgs(
        maxlen=ht.maxlen,
        hidden_units=ht.hidden_units,
        num_blocks=ht.num_blocks,
        num_heads=ht.num_heads,
        dropout_rate=ht.dropout_rate,
        l2_emb=ht.l2_emb,
        time_span=ht.time_span,
        use_aspect_enrichment=use_aspect,
        aspect_mp_layers=ht.aspect_mp_layers,
    )
    model = HetTiSASRecModel(
        len(id_maps.user_to_idx),
        len(id_maps.item_to_idx),
        model_args,
        graph,
    ).to(device)

    ckpt = torch.load(ht.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = evaluate_valid_cases(
        model,
        valid_cases,
        list(graph.item_ids),
        train_pairs,
        device=device,
        pool_size=ht.pool_size,
        max_pairs=ht.valid_eval_max_pairs,
        seed=cfg.experiment.seed,
        mask_train_seen=ht.valid_mask_train_seen,
        maxlen=ht.maxlen,
        time_span=ht.time_span,
        eval_batch_size=ht.valid_eval_batch_size,
    )
    label = "pure_tisasrec" if args.pure_tisasrec else "hettisasrec"
    print(
        f"[compare_retrieval] mode={label} pool_recall@50={metrics.pool_recall:.4f} "
        f"mrr={metrics.link_mrr:.4f} n_pairs={metrics.n_pairs_eval}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
