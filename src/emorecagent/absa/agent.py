"""Hybrid ABSA agent: classical tool + LLM validate/repair."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..llm.client import LLMClient
from ..llm.prompts import (
    ABSA_AGENT_REPAIR_V1,
    ABSA_AGENT_VALIDATE_FAST_V1,
    ABSA_AGENT_VALIDATE_V1,
    format_prompt,
)
from ..llm.schemas import HybridAbsaVerdict, TripleSet
from .classical import ClassicalAbsaTool


@dataclass
class HybridAgentStats:
    llm_calls: int = 0
    repair_calls: int = 0
    fast_path_calls: int = 0
    reviews: int = 0

    @property
    def repair_rate(self) -> float:
        if self.reviews == 0:
            return 0.0
        return self.repair_calls / self.reviews

    @property
    def validate_only_rate(self) -> float:
        if self.reviews == 0:
            return 0.0
        return (self.reviews - self.repair_calls) / self.reviews

    @property
    def llm_calls_mean(self) -> float:
        if self.reviews == 0:
            return 0.0
        return self.llm_calls / self.reviews


@dataclass
class HybridAbsaAgent:
    classical: ClassicalAbsaTool
    client: LLMClient
    min_confidence: float = 0.5
    classical_min_confidence: float = 0.85
    repair_on_gap: bool = True
    stats: HybridAgentStats = field(default_factory=HybridAgentStats)

    def process(self, text: str) -> TripleSet:
        self.stats.reviews += 1
        candidates = self.classical.predict(text)
        cand_json = json.dumps(
            [c.model_dump() for c in candidates], ensure_ascii=False
        )

        use_fast = bool(candidates) and all(
            c.confidence >= self.classical_min_confidence for c in candidates
        )
        if use_fast:
            self.stats.fast_path_calls += 1
            prompt = format_prompt(
                ABSA_AGENT_VALIDATE_FAST_V1,
                candidates_json=cand_json,
            )
        else:
            prompt = format_prompt(
                ABSA_AGENT_VALIDATE_V1,
                review_text=text,
                candidates_json=cand_json,
            )

        verdict = self._invoke_verdict(prompt)
        triples = list(verdict.triples)

        if verdict.needs_repair and self.repair_on_gap:
            self.stats.repair_calls += 1
            hints = ", ".join(verdict.missing_aspect_hints) or "(none)"
            repair_prompt = format_prompt(
                ABSA_AGENT_REPAIR_V1,
                review_text=text,
                candidates_json=cand_json,
                missing_hints=hints,
            )
            repaired = self.client.invoke_structured(repair_prompt, TripleSet)
            self.stats.llm_calls += 1
            triples = list(repaired.triples)
        elif not triples and not candidates:
            self.stats.repair_calls += 1
            repair_prompt = format_prompt(
                ABSA_AGENT_REPAIR_V1,
                review_text=text,
                candidates_json="[]",
                missing_hints="extract all supported aspects",
            )
            repaired = self.client.invoke_structured(repair_prompt, TripleSet)
            self.stats.llm_calls += 1
            triples = list(repaired.triples)

        kept = [t for t in triples if t.confidence >= self.min_confidence]
        return TripleSet(triples=kept)

    def _invoke_verdict(self, prompt: str) -> HybridAbsaVerdict:
        self.stats.llm_calls += 1
        return self.client.invoke_structured(prompt, HybridAbsaVerdict)
