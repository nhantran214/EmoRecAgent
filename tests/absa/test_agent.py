"""Hybrid ABSA agent tests with MockClassicalAbsaTool + FakeLLM."""

from __future__ import annotations

from emorecagent.absa.agent import HybridAbsaAgent
from emorecagent.absa.classical import MockClassicalAbsaTool
from emorecagent.llm.client import LLMClient, FakeLLM
from emorecagent.llm.schemas import AbsaTriple, HybridAbsaVerdict, TripleSet


def _verdict_ok() -> str:
    return HybridAbsaVerdict(
        triples=[
            AbsaTriple(
                aspect="battery",
                opinion="lasts long",
                sentiment="positive",
                confidence=0.95,
            ),
            AbsaTriple(
                aspect="charging",
                opinion="slow",
                sentiment="negative",
                confidence=0.9,
            ),
        ],
        needs_repair=False,
        missing_aspect_hints=[],
    ).model_dump_json()


def _verdict_needs_repair() -> str:
    return HybridAbsaVerdict(
        triples=[
            AbsaTriple(
                aspect="battery",
                opinion="lasts long",
                sentiment="positive",
                confidence=0.95,
            ),
        ],
        needs_repair=True,
        missing_aspect_hints=["packaging"],
    ).model_dump_json()


def _repair_triples() -> str:
    return TripleSet(
        triples=[
            AbsaTriple(
                aspect="battery",
                opinion="lasts long",
                sentiment="positive",
                confidence=0.95,
            ),
            AbsaTriple(
                aspect="packaging",
                opinion="nice box",
                sentiment="positive",
                confidence=0.8,
            ),
        ]
    ).model_dump_json()


def test_agent_validate_only_path() -> None:
    tool = MockClassicalAbsaTool(
        [
            AbsaTriple(
                aspect="battery",
                opinion="",
                sentiment="positive",
                confidence=0.95,
            ),
            AbsaTriple(
                aspect="charging",
                opinion="",
                sentiment="negative",
                confidence=0.9,
            ),
        ]
    )
    fake = FakeLLM([_verdict_ok()])
    agent = HybridAbsaAgent(tool, LLMClient(fake), classical_min_confidence=0.85)
    out = agent.process("battery lasts long but charging is slow")
    assert len(out.triples) == 2
    assert agent.stats.llm_calls == 1
    assert agent.stats.repair_rate == 0.0


def test_agent_repair_path() -> None:
    tool = MockClassicalAbsaTool(
        [
            AbsaTriple(
                aspect="battery",
                opinion="",
                sentiment="positive",
                confidence=0.95,
            ),
        ]
    )
    fake = FakeLLM([_verdict_needs_repair(), _repair_triples()])
    agent = HybridAbsaAgent(tool, LLMClient(fake), repair_on_gap=True)
    out = agent.process("battery lasts long, nice packaging")
    aspects = {t.aspect for t in out.triples}
    assert "packaging" in aspects
    assert agent.stats.llm_calls == 2
    assert agent.stats.repair_calls == 1


def test_agent_filters_low_confidence() -> None:
    tool = MockClassicalAbsaTool([])
    low = HybridAbsaVerdict(
        triples=[
            AbsaTriple(
                aspect="scent",
                opinion="weak",
                sentiment="negative",
                confidence=0.2,
            )
        ],
        needs_repair=False,
    ).model_dump_json()
    fake = FakeLLM([low])
    agent = HybridAbsaAgent(tool, LLMClient(fake), min_confidence=0.5)
    out = agent.process("weak scent")
    assert out.triples == []
