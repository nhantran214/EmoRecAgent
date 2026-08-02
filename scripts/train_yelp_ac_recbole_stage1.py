#!/usr/bin/env python3
"""Train RecBole TiSASRec for Yelp_AC and export an EmoRecAgent Stage-1 bundle.

Yelp_AC paper track only. Amazon / Yelp-review keep ``scripts/train_tisasrec_stage1.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT_ROOT / "baseline" / "RecBole-TiSASRec"
DEFAULT_BUNDLE = (
    PROJECT_ROOT / "data/processed/Yelp_AC/tisasrec_paper/recbole_stage1_bundle.pt"
)
DEFAULT_METRICS = PROJECT_ROOT / "results/Yelp_AC/recbole_tisasrec.json"
PAPER_METRICS = PROJECT_ROOT / "results/Yelp_AC/tisasrec_paper_stage1_test.json"


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    files = sorted(ckpt_dir.glob("TiSASRec-*.pth"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no TiSASRec-*.pth under {ckpt_dir}")
    return files[-1]


def _copy_metrics_as_paper(src: Path, dst: Path) -> None:
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
        "split": "data/processed/Yelp_AC",
        "metrics": metrics,
        "source": str(src),
        "recbole": payload.get("recbole"),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recbole-config",
        default=str(BASELINE / "configs/paper_tisasrec_yelp_ac.yaml"),
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
        "--checkpoint",
        default=None,
        help="RecBole .pth (default: latest under baseline/.../checkpoints)",
    )
    parser.add_argument("--bundle-out", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--metrics-out", default=str(DEFAULT_METRICS))
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "logs/Yelp_AC"))
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from emorecagent.tisasrec_align.recbole_backend import export_recbole_bundle

    ckpt_dir = BASELINE / "checkpoints"
    if not args.skip_train:
        log_file = args.log_file or str(
            Path(args.log_dir) / "recbole_tisasrec_stage1.log"
        )
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(BASELINE / "amazon/run_experiment.py"),
            "--config",
            args.recbole_config,
            "--out",
            args.metrics_out,
            "--log-file",
            log_file,
        ]
        print("[train] ", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
        if rc != 0:
            return rc

    ckpt = (
        Path(args.checkpoint)
        if args.checkpoint
        else _latest_checkpoint(ckpt_dir)
    )
    print(f"[export] checkpoint={ckpt}", flush=True)
    bundle = export_recbole_bundle(
        recbole_checkpoint=ckpt,
        out_path=args.bundle_out,
        vendor_root=BASELINE / "vendor",
    )
    # Keep a pointer next to the bundle for provenance.
    pointer = Path(args.bundle_out).with_suffix(".source.txt")
    pointer.write_text(str(ckpt.resolve()) + "\n", encoding="utf-8")
    _copy_metrics_as_paper(Path(args.metrics_out), PAPER_METRICS)
    # Convenience copy of metrics beside bundle.
    shutil.copy2(args.metrics_out, Path(args.bundle_out).with_name("recbole_metrics.json"))
    print(f"[done] bundle={bundle}", flush=True)
    print(f"[done] paper metrics → {PAPER_METRICS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
