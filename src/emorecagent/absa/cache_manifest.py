"""Sidecar manifest for ABSA cache pipeline version (R8)."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ConfigError


def manifest_path_for(cache_path: str | Path) -> Path:
    p = Path(cache_path)
    return p.with_name(f"{p.stem}.cache_manifest.json")


def read_manifest(cache_path: str | Path) -> dict | None:
    path = manifest_path_for(cache_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(cache_path: str | Path, pipeline_version: str) -> Path:
    path = manifest_path_for(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pipeline_version": pipeline_version}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def ensure_cache_manifest(cache_path: str | Path, pipeline_version: str) -> None:
    """Refuse stale caches when version mismatches config."""
    cache_p = Path(cache_path)
    manifest = read_manifest(cache_p)
    if cache_p.exists():
        if manifest is None:
            raise ConfigError(
                f"ABSA cache exists without manifest: {cache_p}\n"
                "Run: make clean-absa"
            )
        if manifest.get("pipeline_version") != pipeline_version:
            raise ConfigError(
                f"ABSA cache pipeline_version={manifest.get('pipeline_version')!r} "
                f"!= config {pipeline_version!r}\n"
                "Run: make clean-absa"
            )
