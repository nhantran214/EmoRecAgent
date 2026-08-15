"""Train AC-TSR / RecBole TiSASRec (CE) on EmoRecAgent processed splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

BASELINE_ROOT = Path(__file__).resolve().parent.parent
AMAZON_DIR = BASELINE_ROOT / "amazon"
VENDOR_ROOT = BASELINE_ROOT / "vendor"
PROJECT_ROOT = BASELINE_ROOT.parent.parent
DEFAULT_LOG_DIR = BASELINE_ROOT / "logs"
DATASET_ROOT = BASELINE_ROOT / "dataset"
CHECKPOINT_DIR = BASELINE_ROOT / "checkpoints"

sys.path.insert(0, str(AMAZON_DIR))
sys.path.insert(0, str(VENDOR_ROOT))  # prefer AC-TSR RecBole fork over site-packages

from export_inter import export_combined_inter  # noqa: E402
from logging_utils import configure_run_logging, default_log_path, tee_stdout_stderr  # noqa: E402


def _resolve_processed_dir(category: str, processed_dir: str | None) -> Path:
    if processed_dir:
        path = Path(processed_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return PROJECT_ROOT / "data" / "processed" / category


def _resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), PROJECT_ROOT, BASELINE_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    if path.parts and path.parts[0] == "configs":
        return (BASELINE_ROOT / path).resolve()
    return (Path.cwd() / path).resolve()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _flatten_recbole_result(result: dict) -> dict[str, float]:
    """Normalize RecBole metric dict → flat recall/ndcg/hr/mrr @K keys."""
    out: dict[str, float] = {}
    for key, value in result.items():
        name = str(key).lower().replace(" ", "")
        if name.startswith("hit@"):
            k = name.split("@", 1)[1]
            out[f"hr@{k}"] = float(value)
            out.setdefault(f"recall@{k}", float(value))
        elif name.startswith("recall@"):
            k = name.split("@", 1)[1]
            out[f"recall@{k}"] = float(value)
            out.setdefault(f"hr@{k}", float(value))
        elif name.startswith("ndcg@"):
            out[name] = float(value)
        elif name.startswith("mrr@"):
            out[name] = float(value)
    return out


def _write_emorec_json(
    out_path: Path,
    *,
    category: str,
    metrics: dict[str, float],
    recbole_valid: dict,
    recbole_test: dict,
    n_users: int | None,
) -> None:
    payload = {
        "method": "recbole_tisasrec",
        "category": category,
        "eval_protocol": "recbole_full_loo",
        "aggregation": "user_mean",
        "protocol": "full_catalog",
        "n_test_users": n_users,
        "n_test_rows": n_users,
        "means_per_user": metrics,
        "means": metrics,
        "recbole": {
            "valid": {str(k): float(v) for k, v in recbole_valid.items()},
            "test": {str(k): float(v) for k, v in recbole_test.items()},
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def run_recbole_tisasrec(
    *,
    dataset_name: str,
    config_files: list[Path],
    config_dict: dict,
    logger,
) -> tuple[dict, dict]:
    from recbole.quick_start import run_recbole  # noqa: WPS433

    logger.info("RecBole model=TiSASRec dataset=%s", dataset_name)
    logger.info("config_files=%s", [str(p) for p in config_files])
    logger.info("config_dict overrides=%s", config_dict)

    result = run_recbole(
        model="TiSASRec",
        dataset=dataset_name,
        config_file_list=[str(p) for p in config_files],
        config_dict=config_dict,
    )
    # run_recbole returns (model, dataset, train, valid, test) in some versions,
    # or a metric dict — normalize both shapes.
    if isinstance(result, tuple) and len(result) >= 5:
        _model, _dataset, _train, valid_result, test_result = result[:5]
    elif isinstance(result, dict):
        valid_result = result.get("best_valid_result") or result.get("valid") or {}
        test_result = result.get("test_result") or result.get("test") or {}
    else:
        raise RuntimeError(f"unexpected run_recbole return type: {type(result)}")

    return dict(valid_result), dict(test_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RecBole TiSASRec (CE) on EmoRecAgent processed splits."
    )
    parser.add_argument(
        "--config",
        default="configs/paper_tisasrec_yelp_ac.yaml",
        help="Experiment YAML (default: configs/paper_tisasrec_yelp_ac.yaml)",
    )
    parser.add_argument("--category", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override RecBole train_batch_size",
    )
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse existing dataset/*.inter files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = _resolve_project_path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config not found: {config_path}")
        return 1
    exp_cfg = _load_yaml(config_path)
    data_cfg = exp_cfg.get("data", {})
    model_cfg = exp_cfg.get("model", {})
    log_cfg = exp_cfg.get("logging", {})

    category = args.category or data_cfg.get("category", "Yelp_AC")
    dataset_name = data_cfg.get("dataset_name", category)
    data_dir = _resolve_processed_dir(
        category, args.processed_dir or data_cfg.get("processed_dir")
    )
    if not (data_dir / "train.jsonl").exists():
        print(f"ERROR: train.jsonl not found in {data_dir}")
        return 1

    log_dir = args.log_dir or log_cfg.get("log_dir") or str(DEFAULT_LOG_DIR)
    if not Path(log_dir).is_absolute():
        log_dir = str(BASELINE_ROOT / log_dir)
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = default_log_path(f"recbole_tisasrec_{category}", log_dir)

    logger, _ = configure_run_logging(f"recbole_tisasrec_{category}", log_file=log_path)
    logger.info("log file: %s", log_path.resolve())

    with tee_stdout_stderr(log_path):
        t0 = time.monotonic()
        loss_label = model_cfg.get("loss_type") or "CE"
        logger.info("=== RecBole TiSASRec (%s) × %s ===", loss_label, category)
        logger.info("processed_dir: %s", data_dir)

        dataset_dir = DATASET_ROOT / dataset_name
        if not args.skip_export:
            counts = export_combined_inter(data_dir, dataset_dir, dataset_name)
            logger.info("exported RecBole .inter: %s → %s", counts, dataset_dir)
        else:
            logger.info("skip export; using %s", dataset_dir)

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        # Runtime paths must be absolute so RecBole finds the exported dataset.
        # Prefer per-track checkpoint dirs (Yelp Option B vs Yelp_AC) when set.
        ckpt_dir = CHECKPOINT_DIR
        raw_ckpt = (
            model_cfg.get("checkpoint_dir")
            or data_cfg.get("checkpoint_dir")
            or None
        )
        if raw_ckpt:
            ckpt_dir = Path(raw_ckpt)
            if not ckpt_dir.is_absolute():
                ckpt_dir = BASELINE_ROOT / ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "data_path": str(DATASET_ROOT),
            "checkpoint_dir": str(ckpt_dir),
            "seed": exp_cfg.get("experiment", {}).get("seed", 42),
        }
        epochs = args.epochs if args.epochs is not None else model_cfg.get("epochs")
        if epochs is not None:
            config_dict["epochs"] = int(epochs)
        for key in (
            "n_layers",
            "n_heads",
            "hidden_size",
            "inner_size",
            "train_batch_size",
            "learning_rate",
            "loss_type",
            "hidden_dropout_prob",
            "attn_dropout_prob",
            "time_span",
            "stopping_step",
            "weight_decay",
            "neg_sampling",
        ):
            if key in model_cfg and model_cfg[key] is not None:
                config_dict[key] = model_cfg[key]
        # CLI overrides YAML (must run after the model_cfg loop).
        if args.train_batch_size is not None:
            config_dict["train_batch_size"] = int(args.train_batch_size)

        # Dataset overlay: Yelp_AC default; Option B Yelp reviews sets
        # data.recbole_base_config: configs/yelp_reviews.yaml
        base_name = data_cfg.get("recbole_base_config") or "configs/yelp_ac.yaml"
        base_cfg = _resolve_project_path(base_name)
        if not base_cfg.is_file():
            # Also resolve relative to baseline root (configs/...).
            alt = BASELINE_ROOT / base_name
            if alt.is_file():
                base_cfg = alt
            else:
                alt2 = BASELINE_ROOT / Path(base_name).name
                base_cfg = alt2 if alt2.is_file() else base_cfg
        config_files = [
            base_cfg,
            BASELINE_ROOT / "configs" / "config_t.yaml",
            BASELINE_ROOT / "configs" / "TiSASRec.yaml",
        ]
        for path in config_files:
            if not path.is_file():
                print(f"ERROR: missing RecBole config: {path}")
                return 1

        valid_result, test_result = run_recbole_tisasrec(
            dataset_name=dataset_name,
            config_files=config_files,
            config_dict=config_dict,
            logger=logger,
        )

        metrics = _flatten_recbole_result(test_result)
        n_users = None
        manifest = data_dir / "manifest.json"
        if manifest.is_file():
            with manifest.open(encoding="utf-8") as fh:
                n_users = json.load(fh).get("n_test_users")

        out_path = Path(args.out) if args.out else (
            PROJECT_ROOT / "results" / category / "recbole_tisasrec.json"
        )
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        _write_emorec_json(
            out_path,
            category=category,
            metrics=metrics,
            recbole_valid=valid_result,
            recbole_test=test_result,
            n_users=n_users,
        )

        logger.info("[RecBole metrics] wrote %s", out_path)
        logger.info("  valid: %s", valid_result)
        logger.info("  test:  %s", test_result)
        for key in ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "hr@10", "hr@20"):
            if key in metrics:
                logger.info("  %s: %.4f", key, metrics[key])
        logger.info("=== done in %.1fs ===", time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
