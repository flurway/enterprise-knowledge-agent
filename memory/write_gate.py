"""
Long-term memory write gate.

The gate prevents every assistant response from being stored as long-term
memory. It keeps durable memory reserved for grounded, useful research
summaries and leaves chitchat, clarification prompts, unsafe/invalid answers,
and very short responses in short-term history only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.taxonomy import MemoryType, get_memory_type_spec


@dataclass(frozen=True)
class MemoryWriteDecision:
    should_write: bool
    reason: str
    memory_type: str = "research_summary"
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_write": self.should_write,
            "reason": self.reason,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class MemoryWriteGate:
    def __init__(self, min_response_chars: int = 120):
        self.min_response_chars = min_response_chars

    def decide(self, query: str, result: dict[str, Any], session_id: str = "") -> MemoryWriteDecision:
        intent = str(result.get("intent", ""))
        response = str(result.get("response", ""))
        citations = result.get("citations") or {}
        validation = result.get("answer_validation") or {}

        base_metadata = {
            "session_id": session_id,
            "intent": intent,
            "has_citations": bool(citations),
            "query_preview": query[:160],
        }

        if result.get("needs_clarification"):
            return MemoryWriteDecision(False, "clarification response is not durable memory", metadata=base_metadata)

        if intent == "chitchat":
            return MemoryWriteDecision(False, "chitchat should stay in short-term memory", metadata=base_metadata)

        if not response.strip():
            return MemoryWriteDecision(False, "empty response", metadata=base_metadata)

        if len(response.strip()) < self.min_response_chars:
            return MemoryWriteDecision(False, "response too short for durable memory", metadata=base_metadata)

        if citations and validation and not validation.get("ok", True):
            return MemoryWriteDecision(False, "answer validation failed", metadata={**base_metadata, "answer_validation": validation})

        if citations:
            spec = get_memory_type_spec(MemoryType.RESEARCH_SUMMARY)
            return MemoryWriteDecision(
                True,
                "grounded research answer with citations",
                memory_type=spec.memory_type.value,
                confidence=spec.default_confidence,
                metadata={
                    **base_metadata,
                    "source_count": len(citations),
                    "source_grounded": True,
                    "ttl_days": spec.ttl_days,
                    "retrieval_priority": spec.retrieval_priority,
                    "requires_source_grounding": spec.requires_source_grounding,
                },
            )

        spec = get_memory_type_spec(MemoryType.EPISODE)
        return MemoryWriteDecision(
            True,
            "ungrounded fallback answer retained with lower confidence",
            memory_type=spec.memory_type.value,
            confidence=spec.default_confidence,
            metadata={
                **base_metadata,
                "source_grounded": False,
                "ttl_days": spec.ttl_days,
                "retrieval_priority": spec.retrieval_priority,
                "durable": spec.durable,
            },
        )
