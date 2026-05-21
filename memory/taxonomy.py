"""
Memory taxonomy for long-term memory.

The taxonomy separates durable user preferences from grounded research
summaries and low-confidence episodes. This keeps the memory system from
becoming one undifferentiated vector bucket.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    RESEARCH_SUMMARY = "research_summary"
    EPISODE = "episode"
    HYPOTHESIS = "hypothesis"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class MemoryTypeSpec:
    memory_type: MemoryType
    description: str
    default_confidence: float
    retrieval_priority: int
    ttl_days: int | None = None
    requires_source_grounding: bool = False
    durable: bool = True

    def to_dict(self) -> dict:
        return {
            "memory_type": self.memory_type.value,
            "description": self.description,
            "default_confidence": self.default_confidence,
            "retrieval_priority": self.retrieval_priority,
            "ttl_days": self.ttl_days,
            "requires_source_grounding": self.requires_source_grounding,
            "durable": self.durable,
        }


DEFAULT_MEMORY_TYPE_SPECS: dict[MemoryType, MemoryTypeSpec] = {
    MemoryType.PREFERENCE: MemoryTypeSpec(
        MemoryType.PREFERENCE,
        "Stable user preference, such as preferred answer style or workflow habit",
        default_confidence=0.9,
        retrieval_priority=100,
        durable=True,
    ),
    MemoryType.CONSTRAINT: MemoryTypeSpec(
        MemoryType.CONSTRAINT,
        "Explicit user constraint that should guide future runs",
        default_confidence=0.9,
        retrieval_priority=95,
        durable=True,
    ),
    MemoryType.RESEARCH_SUMMARY: MemoryTypeSpec(
        MemoryType.RESEARCH_SUMMARY,
        "Source-grounded summary of a completed research interaction",
        default_confidence=0.85,
        retrieval_priority=60,
        ttl_days=90,
        requires_source_grounding=True,
        durable=True,
    ),
    MemoryType.EPISODE: MemoryTypeSpec(
        MemoryType.EPISODE,
        "Low-confidence memory about an interaction episode",
        default_confidence=0.45,
        retrieval_priority=25,
        ttl_days=14,
        durable=False,
    ),
    MemoryType.HYPOTHESIS: MemoryTypeSpec(
        MemoryType.HYPOTHESIS,
        "Uncertain claim that needs future verification",
        default_confidence=0.5,
        retrieval_priority=30,
        ttl_days=30,
        durable=False,
    ),
    MemoryType.CONFLICT: MemoryTypeSpec(
        MemoryType.CONFLICT,
        "Unresolved memory conflict record",
        default_confidence=0.3,
        retrieval_priority=10,
        ttl_days=None,
        durable=True,
    ),
}


def get_memory_type_spec(memory_type: str | MemoryType) -> MemoryTypeSpec:
    key = memory_type if isinstance(memory_type, MemoryType) else MemoryType(memory_type)
    return DEFAULT_MEMORY_TYPE_SPECS[key]
