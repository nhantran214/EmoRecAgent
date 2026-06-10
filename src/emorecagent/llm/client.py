"""Swappable local LLM client with structured output and retries (U3)."""

from __future__ import annotations

import time
from typing import Any, Protocol, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from ..config import Config, LlmCfg

T = TypeVar("T", bound=BaseModel)


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

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
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


class LLMClient:
    """Production client wrapping ChatOllama with retry + structured helpers."""

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
    def from_config(cls, cfg: Config | LlmCfg, *, host: str | None = None) -> "LLMClient":
        llm_cfg = cfg if isinstance(cfg, LlmCfg) else cfg.llm
        host = host or (cfg.ollama.host if isinstance(cfg, Config) else None)
        if host is None:
            raise LLMError("OLLAMA_HOST is required to build LLMClient")
        chat = ChatOllama(
            model=llm_cfg.model,
            base_url=host,
            temperature=llm_cfg.temperature,
            num_predict=2048,
            timeout=llm_cfg.request_timeout_s,
        )
        return cls(
            chat,
            max_retries=llm_cfg.max_retries,
            request_timeout_s=llm_cfg.request_timeout_s,
        )

    def invoke_text(self, prompt: str) -> str:
        msg = self._invoke_with_retry([HumanMessage(content=prompt)])
        content = msg.content
        if not isinstance(content, str):
            raise LLMError("Expected string content from LLM")
        return content

    def invoke_structured(self, prompt: str, schema: type[T]) -> T:
        """Structured output with bounded retry on parse/validation failures."""
        structured = self._llm.with_structured_output(schema)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                result = structured.invoke([HumanMessage(content=prompt)])
                if isinstance(result, schema):
                    return result
                return schema.model_validate(result)
            except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise LLMError(
            f"Structured output failed after {self.max_retries} attempts: {last_err}"
        ) from last_err

    def _invoke_with_retry(self, messages: list[BaseMessage]) -> AIMessage:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._llm.invoke(messages)
            except Exception as exc:
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise LLMError(
            f"LLM invoke failed after {self.max_retries} attempts: {last_err}"
        ) from last_err
