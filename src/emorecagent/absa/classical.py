"""Classical ABSA tool (PyABSA) adapter and test double."""

from __future__ import annotations

import inspect
import time
from typing import Protocol

from ..config import AbsaCfg, ConfigError
from ..llm.schemas import AbsaTriple

_SENTIMENT_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
}


def _patch_update_checker_compat() -> None:
    """Allow metric_visualizer to import with update-checker 1.x installed."""
    try:
        from update_checker import UpdateChecker

        params = list(inspect.signature(UpdateChecker.check).parameters.values())
        if len(params) >= 3 and params[1].kind is inspect.Parameter.KEYWORD_ONLY:
            orig = UpdateChecker.check

            def _compat(self, package_name, package_version):  # noqa: ANN001
                return orig(
                    self,
                    package_name=package_name,
                    package_version=package_version,
                )

            UpdateChecker.check = _compat  # type: ignore[method-assign]
    except Exception:
        pass


def require_absa_ml() -> None:
    _patch_update_checker_compat()
    try:
        import pyabsa  # noqa: F401
    except ImportError as exc:
        raise ConfigError(
            "Hybrid ABSA requires optional ML dependencies.\n"
            "Install with: pip install -e \".[absa-ml]\""
        ) from exc
    except (TypeError, ModuleNotFoundError) as exc:
        raise ConfigError(
            "PyABSA dependency versions are incompatible.\n"
            "Reinstall pinned extras: pip install -e \".[absa-ml]\" --force-reinstall"
        ) from exc


class ClassicalAbsaTool(Protocol):
    def predict(self, text: str) -> list[AbsaTriple]: ...


def _normalize_sentiment(raw: object) -> str | None:
    if raw is None:
        return None
    return _SENTIMENT_MAP.get(str(raw).strip().lower())


def _confidence_from_row(row: dict) -> float:
    for key in ("confidence", "probability"):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)) and val:
            try:
                return max(float(x) for x in val)
            except (TypeError, ValueError):
                continue
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            continue
    return 0.75


def _patch_pyabsa_tokenizer(extractor: object) -> None:
    """Unpickled PyABSA tokenizers may lack attrs added in newer transformers."""
    tok = getattr(extractor, "tokenizer", None)
    if tok is None:
        return
    if not hasattr(tok, "split_special_tokens"):
        tok.split_special_tokens = False


def _confidence_at(row: dict, index: int) -> object:
    for key in ("confidence", "probability"):
        vals = row.get(key)
        if isinstance(vals, list) and index < len(vals):
            return vals[index]
    probs = row.get("probs")
    if isinstance(probs, list) and index < len(probs):
        p = probs[index]
        if isinstance(p, (list, tuple)) and p:
            return max(p)
    return None


def _rows_from_pyabsa_output(raw: object) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        aspects = raw.get("aspect") or raw.get("aspects") or []
        if isinstance(aspects, str):
            aspects = [aspects]
        sentiments = raw.get("sentiment") or raw.get("sentiments") or []
        if isinstance(aspects, list) and aspects:
            rows: list[dict] = []
            for i, aspect in enumerate(aspects):
                rows.append(
                    {
                        "aspect": aspect,
                        "sentiment": sentiments[i] if i < len(sentiments) else None,
                        "confidence": _confidence_at(raw, i),
                    }
                )
            return rows
        if "aspect" in raw:
            return [raw]
        return []
    if isinstance(raw, list):
        if not raw:
            return []
        if isinstance(raw[0], dict):
            return list(raw)
        return []
    return []


class PyAbsaClassicalTool:
    """PyABSA ATEPC wrapper; loads model once in ``__init__``."""

    def __init__(
        self,
        *,
        checkpoint: str = "multilingual",
        checkpoint_path: str | None = None,
        device: str = "auto",
    ) -> None:
        require_absa_ml()
        from pyabsa import AspectTermExtraction as ATEPC  # type: ignore import-untyped

        t0 = time.perf_counter()
        load_target = checkpoint_path or checkpoint
        auto_device = device == "auto"
        device_arg = "cuda" if device == "cuda" else "cpu"
        self._extractor = ATEPC.AspectExtractor(
            load_target,
            auto_device=auto_device if device == "auto" else False,
            device=device_arg if not auto_device else None,
        )
        _patch_pyabsa_tokenizer(self._extractor)
        self.warmup_seconds = time.perf_counter() - t0

    @classmethod
    def from_config(cls, cfg: AbsaCfg) -> PyAbsaClassicalTool:
        return cls(
            checkpoint=cfg.classical_checkpoint,
            checkpoint_path=cfg.classical_checkpoint_path,
            device=cfg.classical_device,
        )

    def predict(self, text: str) -> list[AbsaTriple]:
        if not text.strip():
            return []
        raw = self._extractor.predict(
            text,
            print_result=False,
            save_result=False,
            ignore_error=True,
            pred_sentiment=True,
        )
        triples: list[AbsaTriple] = []
        for row in _rows_from_pyabsa_output(raw):
            aspect = str(row.get("aspect") or "").strip().lower()
            sentiment = _normalize_sentiment(row.get("sentiment"))
            if not aspect or sentiment is None:
                continue
            triples.append(
                AbsaTriple(
                    aspect=aspect,
                    opinion="",
                    sentiment=sentiment,  # type: ignore[arg-type]
                    confidence=_confidence_from_row(row),
                )
            )
        return triples


class MockClassicalAbsaTool:
    """Deterministic tool for unit tests (no PyABSA)."""

    def __init__(
        self,
        triples: list[AbsaTriple] | None = None,
        *,
        warmup_seconds: float = 0.0,
    ) -> None:
        self._triples = triples or []
        self.warmup_seconds = warmup_seconds

    def predict(self, text: str) -> list[AbsaTriple]:
        if not text.strip():
            return []
        return list(self._triples)
