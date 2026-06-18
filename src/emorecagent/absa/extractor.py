"""Stage 1: candidate triple extraction."""

from __future__ import annotations

from ..llm.client import LLMClient
from ..llm.prompts import ABSA_EXTRACT_V1, format_prompt
from ..llm.schemas import TripleSet


class AbsaExtractor:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, review_text: str) -> TripleSet:
        prompt = format_prompt(ABSA_EXTRACT_V1, review_text=review_text)
        return self._client.invoke_structured(prompt, TripleSet)
