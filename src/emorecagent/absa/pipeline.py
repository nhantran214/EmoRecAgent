"""ABSA pipeline: text processor + normalize + cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import AbsaCfg, Config, ConfigError
from ..llm.client import LLMClient
from ..llm.schemas import AbsaTriple, TripleSet
from .agent import HybridAbsaAgent, HybridAgentStats
from .cache import AbsaCache
from .cache_manifest import ensure_cache_manifest, write_manifest
from .classical import ClassicalAbsaTool, MockClassicalAbsaTool, PyAbsaClassicalTool, require_absa_ml
from .extractor import AbsaExtractor
from .judge import AbsaJudge
from .normalize import normalize_aspect


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    review_id: str
    text: str


class AbsaTextProcessor(Protocol):
    def process_text(self, text: str) -> list[AbsaTriple]: ...


@dataclass
class LlmOnlyProcessor:
    extractor: AbsaExtractor
    judge: AbsaJudge

    def process_text(self, text: str) -> list[AbsaTriple]:
        candidates = self.extractor.extract(text).triples
        return self.judge.judge(text, candidates).triples


@dataclass
class HybridProcessor:
    agent: HybridAbsaAgent

    def process_text(self, text: str) -> list[AbsaTriple]:
        return self.agent.process(text).triples

    @property
    def stats(self) -> HybridAgentStats:
        return self.agent.stats


@dataclass
class ClassicalOnlyProcessor:
    """PyABSA / mock classical tool only — no LLM validate/repair."""

    tool: ClassicalAbsaTool
    min_confidence: float = 0.5

    def process_text(self, text: str) -> list[AbsaTriple]:
        return [
            t for t in self.tool.predict(text) if t.confidence >= self.min_confidence
        ]

    def process_texts(self, texts: list[str]) -> list[list[AbsaTriple]]:
        batched = self.tool.predict_batch(texts)
        return [
            [t for t in triples if t.confidence >= self.min_confidence]
            for triples in batched
        ]


def _finalize_triples(triples: list[AbsaTriple]) -> TripleSet:
    normalized = [
        AbsaTriple(
            aspect=normalize_aspect(t.aspect),
            opinion=t.opinion,
            sentiment=t.sentiment,
            confidence=t.confidence,
        )
        for t in triples
    ]
    best: dict[tuple[str, str], AbsaTriple] = {}
    for t in normalized:
        key = (t.aspect, t.sentiment)
        if key not in best or t.confidence > best[key].confidence:
            best[key] = t
    return TripleSet(triples=list(best.values()))


class AbsaPipeline:
    def __init__(
        self,
        processor: AbsaTextProcessor,
        cache: AbsaCache | None = None,
    ) -> None:
        self.processor = processor
        self.cache = cache

    def process(self, record: ReviewRecord, *, use_cache: bool = True) -> TripleSet:
        if use_cache and self.cache is not None:
            hit = self.cache.get(record.review_id)
            if hit is not None:
                return hit

        validated = self.processor.process_text(record.text)
        result = _finalize_triples(validated)

        if use_cache and self.cache is not None:
            self.cache.put(record.review_id, result)
        return result

    def process_batch(
        self,
        records: list[ReviewRecord],
        *,
        use_cache: bool = True,
    ) -> list[TripleSet]:
        """GPU-friendly path for classical-only: one PyABSA call per chunk."""
        if not records:
            return []
        process_texts = getattr(self.processor, "process_texts", None)
        if process_texts is None:
            return [self.process(rec, use_cache=use_cache) for rec in records]

        texts = [rec.text for rec in records]
        all_triples = process_texts(texts)
        results: list[TripleSet] = []
        for rec, triples in zip(records, all_triples):
            result = _finalize_triples(triples)
            if use_cache and self.cache is not None:
                self.cache.put(rec.review_id, result)
            results.append(result)
        return results


def build_absa_pipeline(
    cfg: Config,
    client: LLMClient | None = None,
    *,
    cache: AbsaCache | None = None,
    classical_tool: ClassicalAbsaTool | None = None,
    skip_manifest_check: bool = False,
) -> AbsaPipeline:
    absa = cfg.absa
    if not skip_manifest_check:
        ensure_cache_manifest(absa.cache_path, absa.pipeline_version)

    if absa.backend == "llm_only":
        if client is None:
            raise ConfigError("absa.backend=llm_only requires an LLMClient")
        processor: AbsaTextProcessor = LlmOnlyProcessor(
            AbsaExtractor(client),
            AbsaJudge(client, min_confidence=absa.min_confidence),
        )
    elif absa.backend == "hybrid":
        if client is None:
            raise ConfigError("absa.backend=hybrid requires an LLMClient")
        if classical_tool is not None:
            tool = classical_tool
        else:
            require_absa_ml()
            tool = PyAbsaClassicalTool.from_config(absa)
        agent = HybridAbsaAgent(
            tool,
            client,
            min_confidence=absa.min_confidence,
            classical_min_confidence=absa.classical_min_confidence,
            repair_on_gap=absa.repair_on_gap,
        )
        processor = HybridProcessor(agent)
    elif absa.backend == "classical":
        if classical_tool is not None:
            tool = classical_tool
        else:
            require_absa_ml()
            tool = PyAbsaClassicalTool.from_config(absa)
        processor = ClassicalOnlyProcessor(tool, min_confidence=absa.min_confidence)
    else:
        raise ConfigError(f"Unknown absa.backend: {absa.backend!r}")

    cache_obj = cache or AbsaCache(absa.cache_path)
    write_manifest(absa.cache_path, absa.pipeline_version)
    return AbsaPipeline(processor, cache=cache_obj)

def build_mock_hybrid_pipeline(
    client: LLMClient,
    classical_tool: MockClassicalAbsaTool,
    *,
    cache: AbsaCache | None = None,
    min_confidence: float = 0.5,
    classical_min_confidence: float = 0.85,
    repair_on_gap: bool = True,
) -> AbsaPipeline:
    agent = HybridAbsaAgent(
        classical_tool,
        client,
        min_confidence=min_confidence,
        classical_min_confidence=classical_min_confidence,
        repair_on_gap=repair_on_gap,
    )
    return AbsaPipeline(HybridProcessor(agent), cache=cache)
