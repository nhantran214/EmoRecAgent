#!/usr/bin/env python3
"""Verify prerequisites for the emorecagent_align pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from emorecagent.config import load_config, validate_category_path_isolation


def _require(path: Path, label: str, missing: list[str]) -> None:
    if not path.exists():
        missing.append(f"{label}: {path}")


def _check_category_isolation(cfg, missing: list[str]) -> None:
    for err in validate_category_path_isolation(cfg):
        missing.append(f"category path isolation: {err}")


def _check_data_absa(cfg, missing: list[str]) -> None:
    out_dir = Path(cfg.data.out_dir)
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        _require(out_dir / name, f"make data ({name})", missing)
    # ID-only / no-review tracks (e.g. Yelp_AC) skip ABSA and raw reviews.
    if not cfg.absa.enabled:
        if cfg.data.inter_path:
            inter = Path(cfg.data.inter_path)
            if not inter.exists():
                missing.append(f"RecBole inter source (inter_path): {inter}")
        return
    _require(Path(cfg.absa.cache_path), "make absa (cache)", missing)
    _require(Path(cfg.data.review_path), "raw reviews (review_path)", missing)


def _check_stage1_test(cfg, missing: list[str], steps: list[str]) -> None:
    ta = cfg.tisasrec_align
    backend = getattr(ta, "stage1_backend", "era")
    if backend == "recbole":
        bundle = Path(ta.recbole_bundle_path)
        if not bundle.is_file():
            missing.append(f"RecBole Stage-1 bundle: {bundle}")
            steps.append(
                "python3 scripts/train_yelp_ac_recbole_stage1.py "
                "(or --skip-train --checkpoint <recbole.pth>)"
            )
        return
    if not Path(ta.stage1_checkpoint_path).exists():
        missing.append(f"Stage 1 checkpoint: {ta.stage1_checkpoint_path}")
        steps.append("make train-emorecagent")
    if not Path(ta.e_i_matrix_path).exists():
        missing.append(f"E_I matrix: {ta.e_i_matrix_path}")
        steps.append("make train-emorecagent")


def _alignment_encoder_meta(path: Path) -> bool | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    if "use_hash_encoder" not in meta:
        return None
    return bool(meta["use_hash_encoder"])


def _check_encoder_consistency(cfg, missing: list[str]) -> None:
    ta = cfg.tisasrec_align
    align_path = Path(ta.alignment_checkpoint_path)
    trained_hash = _alignment_encoder_meta(align_path)
    if trained_hash is None:
        return
    infer_hash = bool(ta.use_hash_encoder)
    if trained_hash != infer_hash:
        mode = "HashEncoder" if infer_hash else "SentenceTransformer"
        trained_mode = "HashEncoder" if trained_hash else "SentenceTransformer"
        missing.append(
            f"Encoder mismatch: alignment checkpoint trained with {trained_mode}, "
            f"config expects {mode} (tisasrec_align.use_hash_encoder={infer_hash})"
        )


def _needs_alignment_checkpoint(ta) -> bool:
    """Fusion eval needs alignment_mlp.pt; rerank and stage1_only do not."""
    if ta.stage1_only:
        return False
    if ta.stage2_mode == "rerank":
        return False
    return True


def _check_eval(cfg, missing: list[str], steps: list[str]) -> None:
    ta = cfg.tisasrec_align
    if _needs_alignment_checkpoint(ta) and not Path(
        ta.alignment_checkpoint_path
    ).exists():
        missing.append(f"Stage 2 alignment checkpoint: {ta.alignment_checkpoint_path}")
    if ta.stage2_mode == "rerank":
        lookup_path = Path(ta.cross_user_lookup_path)
        if not lookup_path.exists():
            missing.append(f"Cross-user lookup: {lookup_path}")
    tu_path = Path(ta.tu_cache_path)
    if not tu_path.exists():
        missing.append(f"T_u cache: {tu_path}")
    elif tu_path.stat().st_size == 0:
        missing.append(f"T_u cache is empty: {tu_path}")

    if any("Stage 2" in m or "T_u cache" in m or "Cross-user lookup" in m for m in missing):
        if ta.stage2_mode == "rerank":
            steps.extend(
                [
                    "make precompute-tu-emorecagent SPLIT=test NO_LLM=1",
                    "make build-cross-user-lookup-emorecagent",
                ]
            )
        else:
            steps.extend(
                [
                    "make precompute-tu-emorecagent SPLIT=train NO_LLM=1",
                    "make train-align-emorecagent USE_HASH_ENCODER=1",
                    "make precompute-tu-emorecagent SPLIT=test NO_LLM=1",
                ]
            )
    if any("Encoder mismatch" in m for m in missing):
        steps.append(
            "retrain: make train-align-emorecagent USE_HASH_ENCODER=1 "
            "(or set tisasrec_align.use_hash_encoder to match checkpoint meta)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--stage1-test",
        action="store_true",
        help="require Stage 1 checkpoint only (for test-emorecagent)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="require full alignment artifacts (for experiment-emorecagent)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.data.out_dir)
    missing: list[str] = []
    steps: list[str] = []

    _check_category_isolation(cfg, missing)
    _check_data_absa(cfg, missing)
    if args.stage1_test:
        _check_stage1_test(cfg, missing, steps)
    if args.eval:
        _check_stage1_test(cfg, missing, steps)
        _check_eval(cfg, missing, steps)
        ta = cfg.tisasrec_align
        if _needs_alignment_checkpoint(ta):
            _check_encoder_consistency(cfg, missing)

    if missing:
        if args.eval:
            label = "emorecagent_align experiment"
        elif args.stage1_test:
            label = "Stage 1 test eval (test-emorecagent)"
        else:
            label = "emorecagent_align pipeline"
        print(f"Missing prerequisites for {label}:", file=sys.stderr)
        for line in missing:
            print(f"  - {line}", file=sys.stderr)
        if steps:
            print("\nRun these steps in order:", file=sys.stderr)
            for step in steps:
                print(f"  {step}", file=sys.stderr)
        elif not args.stage1_test and not args.eval:
            if cfg.absa.enabled:
                print("\nRun first: make data && make absa", file=sys.stderr)
            else:
                print("\nRun first: make data", file=sys.stderr)
        return 1

    if args.eval:
        ta = cfg.tisasrec_align
        print("OK: full eval artifacts present for emorecagent_align.")
        print(f"  stage1: {Path(ta.stage1_checkpoint_path).resolve()}")
        if _needs_alignment_checkpoint(ta):
            print(f"  alignment: {Path(ta.alignment_checkpoint_path).resolve()}")
        print(f"  tu_cache: {Path(ta.tu_cache_path).resolve()}")
        print(f"  use_hash_encoder: {ta.use_hash_encoder}")
        print(f"  stage2_mode: {ta.stage2_mode}")
        if ta.stage2_mode == "rerank":
            print(f"  cross_user_lookup: {Path(ta.cross_user_lookup_path).resolve()}")
        if _needs_alignment_checkpoint(ta):
            trained_hash = _alignment_encoder_meta(Path(ta.alignment_checkpoint_path))
            if trained_hash is not None:
                print(f"  alignment meta.use_hash_encoder: {trained_hash}")
    elif args.stage1_test:
        ta = cfg.tisasrec_align
        print("OK: Stage 1 checkpoint present for test-emorecagent.")
        print(f"  stage1: {Path(ta.stage1_checkpoint_path).resolve()}")
        print(f"  e_i: {Path(ta.e_i_matrix_path).resolve()}")
    elif not cfg.absa.enabled:
        print("OK: data artifacts present (ABSA disabled for this track).")
        print(f"  splits: {out_dir.resolve()}")
        print(f"  absa.enabled: false")
    else:
        print("OK: data + ABSA artifacts present (read-only inputs for alignment pipeline).")
        print(f"  splits: {out_dir.resolve()}")
        print(f"  absa cache: {Path(cfg.absa.cache_path).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
