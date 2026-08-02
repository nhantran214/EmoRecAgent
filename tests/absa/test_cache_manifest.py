"""Cache manifest version checks."""

from __future__ import annotations

import pytest

from emorecagent.absa.cache import AbsaCache
from emorecagent.absa.cache_manifest import ensure_cache_manifest, write_manifest
from emorecagent.config import ConfigError


def test_manifest_mismatch_raises(tmp_path) -> None:
    cache_path = tmp_path / "absa_cache.sqlite"
    AbsaCache(cache_path).close()
    write_manifest(cache_path, "v1")
    with pytest.raises(ConfigError, match="clean-absa"):
        ensure_cache_manifest(cache_path, "v2")


def test_fresh_cache_without_manifest_raises(tmp_path) -> None:
    cache_path = tmp_path / "absa_cache.sqlite"
    AbsaCache(cache_path).close()
    with pytest.raises(ConfigError, match="without manifest"):
        ensure_cache_manifest(cache_path, "v1")
