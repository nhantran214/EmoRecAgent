"""U3 LLM client tests: FakeLLM, structured output, retries."""

from __future__ import annotations

import pytest
from langchain_core.exceptions import OutputParserException

from emorecagent.llm.client import FakeLLM, LLMClient, LLMError
from emorecagent.llm.schemas import AbsaTriple, TripleSet


def test_fake_llm_returns_scripted_structured_json() -> None:
    triples = TripleSet(
        triples=[
            AbsaTriple(aspect="battery", opinion="long lasting", sentiment="positive")
        ]
    )
    client = LLMClient(FakeLLM([triples.model_dump_json()]))
    out = client.invoke_structured("extract", TripleSet)
    assert len(out.triples) == 1
    assert out.triples[0].aspect == "battery"


def test_structured_retry_then_success() -> None:
    bad = "not json"
    good = TripleSet(triples=[]).model_dump_json()
    fake = FakeLLM([bad, good])
    client = LLMClient(fake, max_retries=3, backoff_s=0.0)
    out = client.invoke_structured("x", TripleSet)
    assert out.triples == []


def test_structured_exhausted_retries_raises_llm_error() -> None:
    fake = FakeLLM(["{broken", "{also-broken"])
    client = LLMClient(fake, max_retries=2, backoff_s=0.0)
    with pytest.raises(LLMError, match="Structured output failed"):
        client.invoke_structured("x", TripleSet)


def test_invoke_text_retries_on_failure() -> None:
    class _Flaky:
        def __init__(self) -> None:
            self.n = 0

        def invoke(self, messages):  # noqa: ANN001
            from langchain_core.messages import AIMessage

            self.n += 1
            if self.n == 1:
                raise RuntimeError("transient")
            return AIMessage(content="hello")

        def with_structured_output(self, schema):  # noqa: ANN001
            raise NotImplementedError

    client = LLMClient(_Flaky(), max_retries=2, backoff_s=0.0)
    assert client.invoke_text("hi") == "hello"


def test_fake_llm_structured_direct() -> None:
    payload = TripleSet(
        triples=[AbsaTriple(aspect="scent", opinion="nice", sentiment="positive")]
    )
    fake = FakeLLM([payload])
    out = fake.with_structured_output(TripleSet).invoke([])
    assert out.triples[0].sentiment == "positive"


def test_from_config_uses_chat_openai() -> None:
    from unittest.mock import MagicMock, patch

    from emorecagent.config import Config, LlmCfg, Neo4jCfg, TgiCfg

    cfg = Config(
        experiment={"name": "t", "seed": 1},
        data={
            "category": "c",
            "review_path": "r",
            "meta_path": "m",
            "out_dir": "o",
        },
        scoring={},
        cf={},
        absa={
            "targets_path": "t",
            "cache_path": "c",
            "gold_path": "g",
        },
        agents={},
        eval={},
        llm=LlmCfg(
            model="Qwen/Qwen2.5-7B-Instruct",
            model_small="Qwen/Qwen2.5-3B-Instruct",
        ),
        neo4j=Neo4jCfg(uri="bolt://x", user="u", password="p"),
        tgi=TgiCfg(
            base_url="http://localhost:8080/",
            base_url_small="http://localhost:8081/",
        ),
    )
    with patch("langchain_openai.ChatOpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        client = LLMClient.from_config(cfg, for_absa=True)
        mock_openai.assert_called_once()
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"] == "http://localhost:8081/v1"
        assert kwargs["model"] == "Qwen/Qwen2.5-3B-Instruct"
        assert client.max_retries == cfg.llm.max_retries


def test_tgi_structured_json_fallback() -> None:
    payload = TripleSet(
        triples=[AbsaTriple(aspect="scent", opinion="nice", sentiment="positive")]
    )

    class _TgiLike:
        def with_structured_output(self, schema):  # noqa: ANN001
            class _Broken:
                def invoke(self, messages):  # noqa: ANN001
                    raise OutputParserException("tool binding unsupported")

            return _Broken()

        def invoke(self, messages):  # noqa: ANN001
            from langchain_core.messages import AIMessage

            return AIMessage(content=payload.model_dump_json())

    client = LLMClient(_TgiLike(), max_retries=2, backoff_s=0.0)
    out = client.invoke_structured("extract", TripleSet)
    assert out.triples[0].aspect == "scent"


def test_tgi_422_json_schema_fallback() -> None:
    payload = TripleSet(
        triples=[AbsaTriple(aspect="battery", opinion="ok", sentiment="neutral")]
    )

    class _UnprocessableEntityError(Exception):
        status_code = 422

    class _Tgi422:
        def with_structured_output(self, schema):  # noqa: ANN001
            class _Broken:
                def invoke(self, messages):  # noqa: ANN001
                    raise _UnprocessableEntityError(
                        "unknown variant `json_schema`, expected json_object"
                    )

            return _Broken()

        def invoke(self, messages):  # noqa: ANN001
            from langchain_core.messages import AIMessage

            return AIMessage(content=payload.model_dump_json())

    client = LLMClient(_Tgi422(), max_retries=2, backoff_s=0.0)
    out = client.invoke_structured("extract", TripleSet)
    assert out.triples[0].aspect == "battery"


def test_structured_timeout_retries_then_wraps_llm_error() -> None:
    class _TimeoutAlways:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema):  # noqa: ANN001
            parent = self

            class _Broken:
                def invoke(self, messages):  # noqa: ANN001
                    parent.calls += 1
                    raise TimeoutError("timed out")

            return _Broken()

        def invoke(self, messages):  # noqa: ANN001
            self.calls += 1
            raise TimeoutError("timed out")

    flaky = _TimeoutAlways()
    client = LLMClient(flaky, max_retries=5, backoff_s=0.0)
    with pytest.raises(LLMError, match="Structured output failed"):
        client.invoke_structured("extract", TripleSet)
    # Cap timeout retries at 2 attempts even when max_retries is higher.
    assert flaky.calls == 2


def test_is_transient_recognizes_api_timeout_name() -> None:
    from emorecagent.llm.client import _is_timeout_error, _is_transient_llm_error

    class APITimeoutError(Exception):
        pass

    assert _is_transient_llm_error(APITimeoutError("Request timed out."))
    assert _is_timeout_error(APITimeoutError("Request timed out."))
    assert _is_transient_llm_error(TimeoutError("timed out"))
    assert not _is_transient_llm_error(ValueError("bad schema"))
