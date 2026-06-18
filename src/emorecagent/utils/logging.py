"""Structured experiment run logging."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def manifest_hash(paths: list[str | Path]) -> str:
    """SHA-256 over sorted file paths that exist (content hashes for small files)."""
    h = hashlib.sha256()
    for path in sorted(str(p) for p in paths):
        pth = Path(path)
        if not pth.exists():
            continue
        h.update(path.encode())
        if pth.is_file() and pth.stat().st_size <= 5_000_000:
            h.update(pth.read_bytes())
    return h.hexdigest()[:16]


class RunLogger:
    """Write resolved config + dataset manifest beside result JSON."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        *,
        run_id: str | None = None,
        config: Any,
        result_path: str | Path,
        extra: dict[str, Any] | None = None,
        manifest_paths: list[str | Path] | None = None,
        absa_quality_path: str | Path | None = None,
        compare_paths: list[str | Path] | None = None,
    ) -> Path:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "config": _to_jsonable(config),
            "result_path": str(result_path),
            "dataset_manifest": manifest_hash(manifest_paths or []),
            "ranking": str(result_path),
        }
        if absa_quality_path:
            payload["absa_quality"] = str(absa_quality_path)
        if compare_paths:
            payload["compare"] = [str(p) for p in compare_paths]
        if extra:
            payload["extra"] = extra
        log_name = f"{run_id}_manifest.json" if run_id else "run_manifest.json"
        log_path = self.out_dir / log_name
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return log_path
