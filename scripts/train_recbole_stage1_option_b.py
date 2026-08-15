#!/usr/bin/env python3
"""Train Option B Stage-1 (RecBole TiSASRec CE) and export Stage-2 bundle.

Reads an ERA category config (``configs/categories/{Cat}.yaml``) for artifact
paths and ``tisasrec_align.recbole_train_config``, then trains + exports:

  data/processed/{Cat}/tisasrec_option_b/recbole_stage1_bundle.pt
  data/processed/{Cat}/tisasrec_option_b/e_i_matrix.pt
  data/processed/{Cat}/tisasrec_option_b/item_token_to_idx.json

Option A (ERA) stays ``scripts/train_tisasrec_stage1.py`` under ``configs/legacy/``.

Examples::

  python scripts/train_recbole_stage1_option_b.py \\
    --config configs/categories/Yelp.yaml

  python scripts/train_recbole_stage1_option_b.py \\
    --config configs/categories/Beauty_and_Personal_Care.yaml --skip-train
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT_ROOT / "baseline" / "RecBole-TiSASRec"
DEFAULT_ERA_CONFIG = PROJECT_ROOT / "configs" / "categories" / "Yelp.yaml"


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    files = sorted(ckpt_dir.glob("TiSASRec-*.pth"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no TiSASRec-*.pth under {ckpt_dir}")
    return files[-1]


def _checkpoint_dir_from_recbole_config(recbole_config: Path, category: str) -> Path:
    """Resolve RecBole checkpoint_dir from the experiment YAML (else default)."""
    default = BASELINE / "checkpoints" / category
    try:
        payload = yaml.safe_load(recbole_config.read_text(encoding="utf-8")) or {}
    except OSError:
        return default
    model = payload.get("model") or {}
    raw = model.get("checkpoint_dir")
    if not raw:
        return default
    path = Path(raw)
    if not path.is_absolute():
        path = BASELINE / path
    return path


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_era_paths(era_config: Path) -> tuple[str, Path, Path]:
    """Return (category, recbole_train_config, bundle_out) from ERA YAML."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from emorecagent.config import load_config

    cfg = load_config(era_config)
    ta = cfg.tisasrec_align
    category = cfg.data.category
    if ta.stage1_backend != "recbole":
        raise SystemExit(
            f"config {era_config} has stage1_backend={ta.stage1_backend!r}; "
            "Option B train expects stage1_backend=recbole"
        )
    recbole_raw = (ta.recbole_train_config or "").strip()
    if not recbole_raw:
        raise SystemExit(
            f"config {era_config} missing tisasrec_align.recbole_train_config"
        )
    bundle_raw = (ta.recbole_bundle_path or "").strip()
    if not bundle_raw:
        raise SystemExit(
            f"config {era_config} missing tisasrec_align.recbole_bundle_path"
        )
    return category, _resolve_path(recbole_raw), _resolve_path(bundle_raw)


def _copy_metrics_as_paper(src: Path, dst: Path, *, category: str, split: str) -> None:
    """Mirror RecBole metrics into the paper Stage-1 JSON shape when possible."""
    if not src.is_file():
        return
    payload = json.loads(src.read_text(encoding="utf-8"))
    means = dict(payload.get("means_per_user") or payload.get("means") or {})
    metrics = {
        "link_recall_at_10": float(means.get("recall@10", 0.0)),
        "link_recall_at_20": float(means.get("recall@20", 0.0)),
        "link_ndcg_at_10": float(means.get("ndcg@10", 0.0)),
        "link_ndcg_at_20": float(means.get("ndcg@20", 0.0)),
        "link_hr_at_10": float(means.get("hr@10", means.get("recall@10", 0.0))),
        "link_hr_at_20": float(means.get("hr@20", means.get("recall@20", 0.0))),
        "link_mrr_at_10": float(means.get("mrr@10", 0.0)),
        "link_mrr_at_20": float(means.get("mrr@20", 0.0)),
        "n_pairs_eval": int(payload.get("n_test_users") or 0),
    }
    out = {
        "method": "recbole_tisasrec_stage1",
        "eval_protocol": "recbole_full_loo",
        "split": split,
        "metrics": metrics,
        "source": str(src),
        "recbole": payload.get("recbole"),
        "category": category,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_ERA_CONFIG),
        help="ERA category YAML (default: configs/categories/Yelp.yaml)",
    )
    parser.add_argument(
        "--recbole-config",
        default=None,
        help="Override RecBole train YAML (default: from ERA config)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter with RecBole deps (default: current)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only export bundle from an existing RecBole .pth",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Train/eval only; skip ERA Stage-2 bundle export",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--bundle-out",
        default=None,
        help="Override bundle path (default: from ERA config)",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="RecBole metrics JSON (default: results/{Cat}/recbole_tisasrec_option_b.json)",
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs (smoke tests)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override RecBole train_batch_size",
    )
    args = parser.parse_args()

    era_config = _resolve_path(args.config)
    category, recbole_from_era, bundle_from_era = _load_era_paths(era_config)
    recbole_config = (
        _resolve_path(args.recbole_config)
        if args.recbole_config
        else recbole_from_era
    )
    bundle_out = (
        _resolve_path(args.bundle_out) if args.bundle_out else bundle_from_era
    )
    metrics_out = (
        _resolve_path(args.metrics_out)
        if args.metrics_out
        else PROJECT_ROOT / "results" / category / "recbole_tisasrec_option_b.json"
    )
    log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "logs" / category
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    ckpt_dir = _checkpoint_dir_from_recbole_config(recbole_config, category)

    if not args.skip_train:
        log_file = args.log_file or str(log_dir / "recbole_tisasrec_option_b_stage1.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(BASELINE / "amazon" / "run_experiment.py"),
            "--config",
            str(recbole_config),
            "--out",
            str(metrics_out),
            "--log-file",
            log_file,
        ]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.train_batch_size is not None:
            cmd.extend(["--train-batch-size", str(args.train_batch_size)])
        print("[train] ", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
        if rc != 0:
            return rc

    if args.skip_export:
        print(f"[done] metrics={metrics_out} (bundle export skipped)", flush=True)
        return 0

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from emorecagent.tisasrec_align.recbole_backend import export_recbole_bundle

    ckpt = (
        Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(ckpt_dir)
    )
    if not ckpt.is_absolute():
        ckpt = (PROJECT_ROOT / ckpt).resolve()
    print(f"[export] checkpoint={ckpt}", flush=True)
    print(f"[export] bundle_out={bundle_out}", flush=True)
    bundle_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = export_recbole_bundle(
        recbole_checkpoint=ckpt,
        out_path=bundle_out,
        vendor_root=BASELINE / "vendor",
    )
    pointer = bundle_out.with_suffix(".source.txt")
    pointer.write_text(str(ckpt.resolve()) + "\n", encoding="utf-8")

    # E_I table for InfoNCE Alignment MLP (§III.F Eq. 14–15).
    from emorecagent.tisasrec_align.recbole_backend import (
        _build_model,
        load_recbole_bundle,
    )

    rb = load_recbole_bundle(bundle_out)
    model = _build_model(rb, device=torch.device("cpu"), vendor_root=BASELINE / "vendor")
    e_i_path = bundle_out.with_name("e_i_matrix.pt")
    torch.save(model.item_embedding.weight.detach().cpu(), e_i_path)
    item_map_path = bundle_out.with_name("item_token_to_idx.json")
    item_map = {
        tok: i for i, tok in enumerate(rb.item_tokens) if tok and tok != "[PAD]"
    }
    item_map_path.write_text(json.dumps(item_map) + "\n", encoding="utf-8")

    if metrics_out.is_file():
        shutil.copy2(metrics_out, bundle_out.with_name("recbole_metrics.json"))
        if category == "Yelp_AC":
            paper_metrics = (
                PROJECT_ROOT / "results" / "Yelp_AC" / "tisasrec_paper_stage1_test.json"
            )
            _copy_metrics_as_paper(
                metrics_out,
                paper_metrics,
                category=category,
                split=f"data/processed/{category}",
            )

    meta = {
        "stage1_backend": "recbole",
        "category": category,
        "option": "B",
        "era_config": str(era_config),
        "recbole_config": str(recbole_config),
        "source_checkpoint": str(ckpt.resolve()),
        "bundle": str(bundle_out.resolve()),
        "e_i_matrix": str(e_i_path.resolve()),
        "item_token_to_idx": str(item_map_path.resolve()),
        "metrics": str(metrics_out.resolve()) if metrics_out.is_file() else None,
    }
    bundle_out.with_name("export_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] Stage-2 bundle → {bundle}", flush=True)
    print(f"[done] E_I → {e_i_path}", flush=True)
    print(f"[done] metrics={metrics_out}", flush=True)
    rel_cfg = (
        era_config.relative_to(PROJECT_ROOT)
        if era_config.is_relative_to(PROJECT_ROOT)
        else era_config
    )
    print(
        "[next] Alignment MLP + Stage-2 (§III.F):\n"
        f"  python3 scripts/train_alignment_stage2_option_b.py --config {rel_cfg}\n"
        f"  python3 scripts/run_experiment.py --config {rel_cfg} --stage1-only …",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
