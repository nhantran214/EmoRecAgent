"""Classical ABSA tool adapter tests."""

from __future__ import annotations

import pytest

from emorecagent.absa.classical import (
    MockClassicalAbsaTool,
    _patch_pyabsa_tokenizer,
    _rows_from_pyabsa_output,
    require_absa_ml,
)
from emorecagent.config import ConfigError
from emorecagent.llm.schemas import AbsaTriple


def test_mock_tool_returns_configured_triples() -> None:
    tool = MockClassicalAbsaTool(
        [
            AbsaTriple(
                aspect="scent",
                opinion="",
                sentiment="positive",
                confidence=0.9,
            )
        ]
    )
    out = tool.predict("smells great")
    assert len(out) == 1
    assert out[0].aspect == "scent"
    assert out[0].sentiment == "positive"


def test_rows_from_pyabsa_list_aspects() -> None:
    raw = {
        "aspect": ["scent", "bottle"],
        "sentiment": ["Positive", "Negative"],
        "confidence": [0.99, 0.98],
    }
    rows = _rows_from_pyabsa_output(raw)
    assert len(rows) == 2
    assert rows[0]["aspect"] == "scent"
    assert rows[1]["sentiment"] == "Negative"


def test_patch_pyabsa_tokenizer_adds_split_special_tokens() -> None:
    class _Tok:
        pass

    class _Ext:
        tokenizer = _Tok()

    _patch_pyabsa_tokenizer(_Ext())
    assert _Ext.tokenizer.split_special_tokens is False


def test_mock_tool_empty_text() -> None:
    tool = MockClassicalAbsaTool([])
    assert tool.predict("  ") == []


def test_require_absa_ml_raises_without_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "pyabsa":
            raise ImportError("no pyabsa")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ConfigError, match="absa-ml"):
        require_absa_ml()
