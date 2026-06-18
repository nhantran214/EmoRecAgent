"""U4 ABSA cache tests."""

from __future__ import annotations

from emorecagent.absa.cache import AbsaCache
from emorecagent.llm.schemas import AbsaTriple, TripleSet


def test_cache_put_get_round_trip(tmp_path) -> None:
    cache = AbsaCache(tmp_path / "absa.sqlite")
    triples = TripleSet(
        triples=[AbsaTriple(aspect="scent", opinion="nice", sentiment="positive")]
    )
    cache.put("r1", triples)
    assert cache.contains("r1")
    hit = cache.get("r1")
    assert hit is not None
    assert hit.triples[0].aspect == "scent"
    cache.close()


def test_cache_miss_returns_none(tmp_path) -> None:
    cache = AbsaCache(tmp_path / "absa.sqlite")
    assert cache.get("missing") is None
    cache.close()


def test_cache_delete(tmp_path) -> None:
    cache = AbsaCache(tmp_path / "absa.sqlite")
    cache.put("r1", TripleSet(triples=[]))
    assert cache.contains("r1")
    assert cache.delete("r1") is True
    assert not cache.contains("r1")
    assert cache.delete("r1") is False
    cache.close()
