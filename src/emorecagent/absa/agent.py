"""Hybrid ABSA agent: classical tool + LLM validate/repair."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ..llm.client import LLMClient, LLMError
from ..llm.prompts import (
    ABSA_AGENT_REPAIR_V1,
    ABSA_AGENT_VALIDATE_FAST_V1,
    ABSA_AGENT_VALIDATE_V1,
    format_prompt,
)
from ..llm.schemas import AbsaTriple, HybridAbsaVerdict, TripleSet
from .classical import ClassicalAbsaTool

logger = logging.getLogger(__name__)


@dataclass
class HybridAgentStats:
    llm_calls: int = 0
    repair_calls: int = 0
    fast_path_calls: int = 0
    llm_fallbacks: int = 0
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

        try:
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
                triples = list(self._invoke_triples(repair_prompt))
            elif not triples and not candidates:
                self.stats.repair_calls += 1
                repair_prompt = format_prompt(
                    ABSA_AGENT_REPAIR_V1,
                    review_text=text,
                    candidates_json="[]",
                    missing_hints="extract all supported aspects",
                )
                triples = list(self._invoke_triples(repair_prompt))
        except LLMError as exc:
            self.stats.llm_fallbacks += 1
            logger.warning("hybrid_llm_fallback classical_only: %s", exc)
            triples = list(candidates)

        kept = [t for t in triples if t.confidence >= self.min_confidence]
        return TripleSet(triples=kept)

    # Validate is shorter; repair TripleSet lists need more room (384 truncated).
    _ABSA_VERDICT_MAX_TOKENS = 512
    _ABSA_TRIPLES_MAX_TOKENS = 1024

    def _invoke_verdict(self, prompt: str) -> HybridAbsaVerdict:
        self.stats.llm_calls += 1
        return self.client.invoke_structured(
            prompt, HybridAbsaVerdict, max_tokens=self._ABSA_VERDICT_MAX_TOKENS
        )

    def _invoke_triples(self, prompt: str) -> list[AbsaTriple]:
        self.stats.llm_calls += 1
        return list(
            self.client.invoke_structured(
                prompt, TripleSet, max_tokens=self._ABSA_TRIPLES_MAX_TOKENS
            ).triples
        )
