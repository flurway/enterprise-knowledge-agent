"""
Retrieval budgeting for long-term memory.

Long-term memory is a mixed bucket: preferences, constraints, research
summaries, episodes, hypotheses, and conflict records share one vector index.
This selector keeps durable user guidance from being crowded out by lower-value
but semantically similar memories.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from memory.taxonomy import get_memory_type_spec


DEFAULT_TYPE_LIMITS = {
    "preference": 2,
    "constraint": 2,
    "research_summary": 2,
    "episode": 1,
    "hypothesis": 1,
    "conflict": 0,
}


@dataclass
class MemoryRetrievalBudget:
    max_items: int = 3
    min_score: float = 0.3
    per_type_limits: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TYPE_LIMITS))

    @classmethod
    def from_config(cls, memory_config) -> "MemoryRetrievalBudget":
        return cls(
            max_items=memory_config.max_long_term_results,
            min_score=getattr(memory_config, "retrieval_min_score", 0.3),
            per_type_limits=dict(getattr(memory_config, "retrieval_type_limits", DEFAULT_TYPE_LIMITS)),
        )

    def to_dict(self) -> dict:
        return {
            "max_items": self.max_items,
            "min_score": self.min_score,
            "per_type_limits": dict(self.per_type_limits),
        }

    def select(self, memories: list[dict]) -> list[dict]:
        eligible = []
        for memory in memories:
            memory_type = memory.get("memory_type", "")
            if memory.get("score", 0.0) < self.min_score:
                continue
            if self.per_type_limits.get(memory_type, 0) <= 0:
                continue
            eligible.append(memory)

        eligible.sort(key=self._rank_key)

        selected = []
        used_by_type: dict[str, int] = {}
        for memory in eligible:
            if len(selected) >= self.max_items:
                break
            memory_type = memory.get("memory_type", "")
            used = used_by_type.get(memory_type, 0)
            if used >= self.per_type_limits.get(memory_type, 0):
                continue
            selected.append(memory)
            used_by_type[memory_type] = used + 1
        return selected

    @staticmethod
    def _rank_key(memory: dict) -> tuple:
        try:
            priority = get_memory_type_spec(memory.get("memory_type", "episode")).retrieval_priority
        except Exception:
            priority = 0
        return (-priority, -float(memory.get("score", 0.0)), float(memory.get("age_days", 0.0)))
