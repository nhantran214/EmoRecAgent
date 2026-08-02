"""Classical ABSA tool (PyABSA) adapter and test double."""

from __future__ import annotations

import inspect
import threading
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

    def predict_batch(self, texts: list[str]) -> list[list[AbsaTriple]]: ...


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


_SPECIAL_TOKEN_NAMES = (
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
)


def _ensure_tokenizer_special_token_attrs(tok: object) -> None:
    """Repair pickled tokenizers for transformers>=4.45 + PyABSA.

    PyABSA checkpoints pickle DebertaV2TokenizerFast with private ``_bos_token``
    fields. Newer transformers only expose ``bos_token`` via
    ``_special_tokens_map``, so ATEPCProcessor crashes with AttributeError.
    """
    if tok is None:
        return
    if tok.__dict__.get("_special_tokens_map") is None:
        mapping: dict[str, object] = {}
        for name in _SPECIAL_TOKEN_NAMES:
            priv = f"_{name}"
            if priv in tok.__dict__:
                mapping[name] = tok.__dict__[priv]
        if "_additional_special_tokens" in tok.__dict__:
            mapping["additional_special_tokens"] = tok.__dict__[
                "_additional_special_tokens"
            ]
        if mapping:
            tok.__dict__["_special_tokens_map"] = mapping
    if not hasattr(tok, "split_special_tokens"):
        tok.split_special_tokens = False


def _patch_pyabsa_atepc_processor() -> None:
    """Run tokenizer repair before PyABSA ATEPCProcessor reads bos_token."""
    try:
        from pyabsa.tasks.AspectTermExtraction.dataset_utils.__lcf__ import (  # type: ignore import-untyped
            data_utils_for_inference as dui,
        )
    except Exception:
        return
    processor_cls = getattr(dui, "ATEPCProcessor", None)
    if processor_cls is None or getattr(
        processor_cls, "_emorecagent_special_tokens_patched", False
    ):
        return
    orig_init = processor_cls.__init__

    def _init(self, tokenizer):  # noqa: ANN001
        _ensure_tokenizer_special_token_attrs(tokenizer)
        return orig_init(self, tokenizer)

    processor_cls.__init__ = _init  # type: ignore[method-assign]
    processor_cls._emorecagent_special_tokens_patched = True


def _patch_pyabsa_tokenizer(extractor: object) -> None:
    """Unpickled PyABSA tokenizers may lack attrs added in newer transformers."""
    _ensure_tokenizer_special_token_attrs(getattr(extractor, "tokenizer", None))


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


def _triples_from_pyabsa_raw(raw: object) -> list[AbsaTriple]:
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


def _resolve_pyabsa_auto_device(device: str) -> bool | str:
    """Map our device setting onto PyABSA's ``auto_device`` kwarg.

    PyABSA ignores a separate ``device=`` kwarg; it only reads ``auto_device``.
    Passing ``auto_device=False`` forces CPU — so CUDA must be the string
    ``\"cuda\"``, not a boolean False + device kwarg.
    """
    if device == "auto":
        return True
    if device == "cuda":
        return "cuda"
    return False


def _require_cuda_available() -> None:
    try:
        import torch
    except ImportError as exc:
        raise ConfigError(
            "classical ABSA on GPU requires torch. Install: pip install -e \".[absa-ml]\""
        ) from exc
    if not torch.cuda.is_available():
        raise ConfigError(
            "classical ABSA requires a CUDA GPU (classical_device=cuda) but "
            "torch.cuda.is_available() is False.\n"
            "Check nvidia-smi / CUDA drivers, or pass --device cpu only for debugging."
        )


class PyAbsaClassicalTool:
    """PyABSA ATEPC wrapper; loads model once in ``__init__``."""

    def __init__(
        self,
        *,
        checkpoint: str = "multilingual",
        checkpoint_path: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        require_absa_ml()
        from pyabsa import AspectTermExtraction as ATEPC  # type: ignore import-untyped

        if device == "cuda":
            _require_cuda_available()

        _patch_pyabsa_atepc_processor()
        t0 = time.perf_counter()
        load_target = checkpoint_path or checkpoint
        auto_device = _resolve_pyabsa_auto_device(device)
        self._extractor = ATEPC.AspectExtractor(
            load_target,
            auto_device=auto_device,
        )
        _patch_pyabsa_tokenizer(self._extractor)
        self._infer_lock = threading.Lock()
        self._batch_size = max(1, int(batch_size))
        self.device = str(getattr(getattr(self._extractor, "config", None), "device", device))
        self.warmup_seconds = time.perf_counter() - t0
        if device == "cuda" and not str(self.device).startswith("cuda"):
            raise ConfigError(
                f"Requested classical_device=cuda but PyABSA loaded on {self.device!r}"
            )

    @classmethod
    def from_config(cls, cfg: AbsaCfg) -> PyAbsaClassicalTool:
        return cls(
            checkpoint=cfg.classical_checkpoint,
            checkpoint_path=cfg.classical_checkpoint_path,
            device=cfg.classical_device,
            batch_size=cfg.classical_batch_size,
        )

    def predict(self, text: str) -> list[AbsaTriple]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[list[AbsaTriple]]:
        if not texts:
            return []
        out: list[list[AbsaTriple]] = [[] for _ in texts]
        nonempty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not nonempty:
            return out
        with self._infer_lock:
            raw = self._extractor.predict(
                [t for _, t in nonempty],
                print_result=False,
                save_result=False,
                ignore_error=True,
                pred_sentiment=True,
                eval_batch_size=self._batch_size,
            )
        if isinstance(raw, dict):
            rows = [raw]
        elif isinstance(raw, list):
            rows = raw
        else:
            rows = []
        if len(rows) != len(nonempty):
            # Fall back to one-by-one if alignment fails.
            for i, text in nonempty:
                with self._infer_lock:
                    one = self._extractor.predict(
                        text,
                        print_result=False,
                        save_result=False,
                        ignore_error=True,
                        pred_sentiment=True,
                        eval_batch_size=1,
                    )
                out[i] = _triples_from_pyabsa_raw(one)
            return out
        for (i, _), row in zip(nonempty, rows):
            out[i] = _triples_from_pyabsa_raw(row)
        return out


class MockClassicalAbsaTool:
    """Deterministic tool for unit tests (no PyABSA)."""

    def __init__(
        self,
        triples: list[AbsaTriple] | None = None,
        *,
        warmup_seconds: float = 0.0,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self._triples = triples or []
        self.warmup_seconds = warmup_seconds
        self._batch_size = batch_size
        self.device = device

    def predict(self, text: str) -> list[AbsaTriple]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[list[AbsaTriple]]:
        return [list(self._triples) if t.strip() else [] for t in texts]