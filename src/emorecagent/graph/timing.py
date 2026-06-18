"""Per-stage timing accumulators for the LangGraph pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _StageAccum:
    count: int = 0
    total_s: float = 0.0

    def record(self, duration_s: float) -> None:
        self.count += 1
        self.total_s += duration_s

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0


@dataclass
class GraphStageTimer:
    """Accumulates wall-clock time per graph node invocation."""

    _stages: dict[str, _StageAccum] = field(default_factory=dict)
    graph_invokes: int = 0

    def record(self, stage: str, duration_s: float) -> None:
        self._stages.setdefault(stage, _StageAccum()).record(duration_s)

    def record_graph_invoke(self) -> None:
        self.graph_invokes += 1

    def summary_lines(self) -> list[str]:
        if not self._stages:
            return []
        order = ("absa", "profiling", "reasoning", "reflection", "explanation")
        lines: list[str] = []
        prefix = f"invokes={self.graph_invokes}" if self.graph_invokes else "invokes=0"
        parts: list[str] = []
        seen: set[str] = set()
        for name in order:
            acc = self._stages.get(name)
            if acc is None:
                continue
            seen.add(name)
            parts.append(
                f"{name}: n={acc.count} total={acc.total_s:.2f}s mean={acc.mean_s:.3f}s"
            )
        for name in sorted(self._stages):
            if name in seen:
                continue
            acc = self._stages[name]
            parts.append(
                f"{name}: n={acc.count} total={acc.total_s:.2f}s mean={acc.mean_s:.3f}s"
            )
        lines.append(f"stages ({prefix}): " + "; ".join(parts))
        total = sum(acc.total_s for acc in self._stages.values())
        lines.append(f"stages total wall={total:.2f}s")
        return lines
