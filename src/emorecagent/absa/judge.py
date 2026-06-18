"""Stage 2: LLM-as-judge validation."""

from __future__ import annotations

import json

from ..llm.client import LLMClient
from ..llm.prompts import ABSA_JUDGE_V1, format_prompt
from ..llm.schemas import AbsaTriple, TripleSet


class AbsaJudge:
    def __init__(self, client: LLMClient, *, min_confidence: float = 0.5) -> None:
        self._client = client
        self.min_confidence = min_confidence

    def judge(self, review_text: str, candidates: list[AbsaTriple]) -> TripleSet:
        if not candidates:
            return TripleSet(triples=[])
        cand_json = json.dumps(
            [c.model_dump() for c in candidates], ensure_ascii=False
        )
        prompt = format_prompt(
            ABSA_JUDGE_V1,
            review_text=review_text,
            candidates_json=cand_json,
        )
        validated = self._client.invoke_structured(prompt, TripleSet)
        kept = [
            t for t in validated.triples if t.confidence >= self.min_confidence
        ]
        return TripleSet(triples=kept)
