"""Swappable TGI LLM client with structured output and retries."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Protocol, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from ..config import Config, LlmCfg, TgiCfg, resolve_llm_model, resolve_tgi_base_url
from .schemas import (
    HybridAbsaVerdict,
    TripleSet,
    coerce_hybrid_verdict,
    coerce_triple_set,
    parse_batch_ranking_json,
    parse_ranking_json,
    ranking_max_tokens,
    salvage_hybrid_verdict_from_error,
    salvage_ranking_verdict_from_error,
    salvage_triple_set_from_error,
)

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM fails after all retries or returns unparseable output."""


class ChatClient(Protocol):
    """Minimal interface shared by real and fake clients."""

    def invoke(self, messages: list[BaseMessage]) -> AIMessage: ...

    def with_structured_output(self, schema: type[T]) -> Any: ...


class FakeLLM:
    """Deterministic test double: returns scripted text or structured objects."""

    def __init__(
        self,
        responses: list[str | BaseModel] | None = None,
        *,
        raise_on: int | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._idx = 0
        self._raise_on = raise_on
        self.temperature = 0.0
        self.model = "fake"

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        del kwargs
        if self._raise_on is not None and self._idx == self._raise_on:
            raise RuntimeError("simulated LLM failure")
        if self._idx >= len(self._responses):
            return AIMessage(content="{}")
        resp = self._responses[self._idx]
        self._idx += 1
        if isinstance(resp, BaseModel):
            return AIMessage(content=resp.model_dump_json())
        return AIMessage(content=resp)

    def with_structured_output(self, schema: type[T]) -> "_FakeStructured[T]":
        return _FakeStructured(self, schema)


class _FakeStructured:
    def __init__(self, llm: FakeLLM, schema: type[T]) -> None:
        self._llm = llm
        self._schema = schema

    def invoke(self, messages: list[BaseMessage]) -> T:
        msg = self._llm.invoke(messages)
        raw = msg.content
        if isinstance(raw, str):
            return self._schema.model_validate_json(raw)
        raise LLMError("FakeLLM returned non-JSON content for structured call")


def _normalize_openai_base_url(url: str) -> str:
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _build_tgi_chat(
    llm_cfg: LlmCfg, tgi_cfg: TgiCfg, *, for_absa: bool
) -> ChatClient:
    from langchain_openai import ChatOpenAI

    base_url = resolve_tgi_base_url(tgi_cfg, llm_cfg, for_absa=for_absa)
    if not base_url:
        raise LLMError("TGI base URL is required")
    api_key = tgi_cfg.api_key or os.environ.get("TGI_API_KEY") or "tgi-local"
    kwargs: dict[str, Any] = {
        "model": resolve_llm_model(llm_cfg, for_absa=for_absa),
        "base_url": _normalize_openai_base_url(base_url),
        "api_key": api_key,
        "temperature": llm_cfg.temperature,
        "timeout": llm_cfg.request_timeout_s,
        "max_retries": 0,
    }
    # ABSA uses JSON grammar decode; uncapped max_tokens (~1024) causes 60–120s+
    # generations and client timeouts under high concurrency.
    if for_absa:
        kwargs["max_tokens"] = 1024
    return ChatOpenAI(**kwargs)


class LLMClient:
    """Production client wrapping TGI (OpenAI-compatible API) with retry helpers."""

    def __init__(
        self,
        llm: ChatClient,
        *,
        max_retries: int = 3,
        request_timeout_s: int = 120,
        backoff_s: float = 1.0,
    ) -> None:
        self._llm = llm
        self.max_retries = max_retries
        self.request_timeout_s = request_timeout_s
        self.backoff_s = backoff_s

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        for_absa: bool = False,
    ) -> "LLMClient":
        chat = _build_tgi_chat(cfg.llm, cfg.tgi, for_absa=for_absa)
        return cls(
            chat,
            max_retries=cfg.llm.max_retries,
            request_timeout_s=cfg.llm.request_timeout_s,
        )

    def invoke_text(self, prompt: str, *, max_tokens: int | None = None) -> str:
        msg = self._invoke_with_retry(
            [HumanMessage(content=prompt)], max_tokens=max_tokens
        )
        content = msg.content
        if not isinstance(content, str):
            raise LLMError("Expected string content from LLM")
        return content

    def invoke_structured(
        self, prompt: str, schema: type[T], *, max_tokens: int | None = None
    ) -> T:
        """Structured output with bounded retry on parse/validation failures."""
        structured = self._llm.with_structured_output(schema)
        if max_tokens is not None and hasattr(structured, "bind"):
            structured = structured.bind(max_tokens=max_tokens)
        use_json_prompt = False
        last_err: Exception | None = None
        timeout_failures = 0
        for attempt in range(self.max_retries):
            try:
                if use_json_prompt:
                    raw = self.invoke_text(
                        prompt + self._json_schema_prompt_suffix(schema),
                        max_tokens=max_tokens,
                    )
                    return self._coerce_structured(raw, schema)
                result = structured.invoke([HumanMessage(content=prompt)])
                return self._coerce_structured_result(result, schema)
            except Exception as exc:
                last_err = exc
                if not use_json_prompt and self._prefer_json_prompt_fallback(exc):
                    use_json_prompt = True
                    if attempt + 1 < self.max_retries:
                        time.sleep(self.backoff_s * (attempt + 1))
                    continue
                salvaged = self._try_salvage_structured(exc, schema)
                if salvaged is not None:
                    return salvaged
                if isinstance(exc, (ValidationError, ValueError, TypeError)):
                    use_json_prompt = True
                    if attempt + 1 < self.max_retries:
                        time.sleep(self.backoff_s * (attempt + 1))
                    continue
                if attempt + 1 < self.max_retries and _is_transient_llm_error(exc):
                    # Timeouts are usually server stalls; fail fast for caller fallback.
                    if _is_timeout_error(exc):
                        timeout_failures += 1
                        if timeout_failures >= 2:
                            break
                    time.sleep(self.backoff_s * (attempt + 1))
                    continue
                break
        raise LLMError(
            f"Structured output failed after {self.max_retries} attempts: {last_err}"
        ) from last_err

    def invoke_ranking_json(
        self,
        prompt: str,
        *,
        pool_ids: list[str],
        max_tokens: int | None = None,
        suffix: str = "",
    ) -> list[str]:
        """Text-mode ranking call (TGI-safe); salvages truncated JSON."""
        from .prompts import RANKING_JSON_SUFFIX

        budget = max_tokens or ranking_max_tokens([len(pool_ids)])
        raw = self.invoke_text(prompt + (suffix or RANKING_JSON_SUFFIX), max_tokens=budget)
        try:
            ranked = parse_ranking_json(raw, pool_ids=pool_ids)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise LLMError(f"Ranking JSON parse failed: {exc}") from exc
        if len(ranked) < len(pool_ids):
            logger.warning(
                "ranking_partial_salvage=true pool=%s recovered=%s",
                len(pool_ids),
                len(ranked),
            )
        return ranked

    def invoke_batch_ranking_json(
        self,
        prompt: str,
        *,
        pools_by_row: dict[str, list[str]],
        max_tokens: int | None = None,
        suffix: str = "",
    ) -> dict[str, list[str]]:
        from .prompts import BATCH_RANKING_JSON_SUFFIX

        pool_sizes = [len(pools_by_row[k]) for k in sorted(pools_by_row)]
        budget = max_tokens or ranking_max_tokens(pool_sizes)
        raw = self.invoke_text(
            prompt + (suffix or BATCH_RANKING_JSON_SUFFIX),
            max_tokens=budget,
        )
        try:
            return parse_batch_ranking_json(raw, pools_by_row=pools_by_row)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise LLMError(f"Batch ranking JSON parse failed: {exc}") from exc

    @staticmethod
    def _prefer_json_prompt_fallback(exc: Exception) -> bool:
        """Native structured output unavailable (e.g. TGI rejects json_schema)."""
        if isinstance(exc, OutputParserException):
            return True
        status = getattr(exc, "status_code", None)
        if status in (400, 422, 501):
            return True
        msg = str(exc).lower()
        return "json_schema" in msg or "response_format" in msg

    def _json_schema_prompt_suffix(self, schema: type[T]) -> str:
        return (
            "\n\nRespond with a single JSON object only (no markdown fences) "
            f"matching this JSON schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )

    def _coerce_structured_result(self, result: object, schema: type[T]) -> T:
        if isinstance(result, schema):
            parsed = result
        else:
            parsed = schema.model_validate(result)
        if schema is TripleSet:
            return coerce_triple_set(parsed)  # type: ignore[return-value]
        if schema is HybridAbsaVerdict:
            return coerce_hybrid_verdict(parsed)  # type: ignore[return-value]
        return parsed

    def _coerce_structured(self, raw: str, schema: type[T]) -> T:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return self._coerce_structured_result(schema.model_validate_json(text), schema)
        except ValidationError as exc:
            salvaged = self._try_salvage_structured(exc, schema)
            if salvaged is not None:
                return salvaged
            raise

    def _try_salvage_structured(self, exc: Exception, schema: type[T]) -> T | None:
        if schema is TripleSet:
            return salvage_triple_set_from_error(exc)  # type: ignore[return-value]
        if schema is HybridAbsaVerdict:
            return salvage_hybrid_verdict_from_error(exc)  # type: ignore[return-value]
        from .schemas import ReasoningRankingVerdict

        if schema is ReasoningRankingVerdict:
            salvaged = salvage_ranking_verdict_from_error(exc, pool_ids=[])
            if salvaged is not None:
                return ReasoningRankingVerdict(ranked_item_ids=salvaged)  # type: ignore[return-value]
        return None

    def _invoke_with_retry(
        self,
        messages: list[BaseMessage],
        *,
        max_tokens: int | None = None,
    ) -> AIMessage:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if max_tokens is not None:
                    return self._llm.invoke(messages, max_tokens=max_tokens)
                return self._llm.invoke(messages)
            except Exception as exc:
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise LLMError(
            f"LLM invoke failed after {self.max_retries} attempts: {last_err}"
        ) from last_err


def _is_timeout_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return True
    msg = str(exc).lower()
    return "timeout" in msg or "timed out" in msg


def _is_transient_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (LLMError, TimeoutError, ConnectionError, OSError)):
        return True
    name = type(exc).__name__.lower()
    if any(
        token in name
        for token in (
            "timeout",
            "connection",
            "ratelimit",
            "unavailable",
            "internalserver",
        )
    ):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "timeout",
            "timed out",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "503",
            "502",
            "429",
        )
    )
