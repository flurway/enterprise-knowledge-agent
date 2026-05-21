"""
Deterministic memory candidate extractor.

This extractor only captures explicit user preferences and constraints. It is
deliberately conservative: it should not infer hidden preferences from ordinary
questions or save domain knowledge as user memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from memory.taxonomy import MemoryType, get_memory_type_spec


@dataclass(frozen=True)
class MemoryCandidate:
    content: str
    memory_type: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "keywords": self.keywords,
        }


class ExplicitMemoryExtractor:
    def __init__(self, max_candidate_chars: int = 220):
        self.max_candidate_chars = max_candidate_chars
        self.preference_patterns = [
            re.compile(r"(?:我|本人)?(?:更)?(?:喜欢|偏好|习惯|倾向于)(?P<value>[^。！？\n]+)"),
            re.compile(r"(?:以后|后续|之后).{0,8}(?:优先|尽量)(?P<value>[^。！？\n]+)"),
            re.compile(r"I (?:prefer|like|usually|tend to) (?P<value>[^.\n]+)", re.IGNORECASE),
        ]
        self.constraint_patterns = [
            re.compile(r"(?:以后|后续|之后|请|麻烦)?(?:都|务必|必须|请)?(?P<value>用中文回答|使用中文|先给结论|先说结论|不要联网|不要使用网络|不要调用外部搜索|不要太长|简洁回答)"),
            re.compile(r"(?:请|以后|后续|之后).{0,8}(?:不要|别|禁止)(?P<value>[^。！？\n]+)"),
            re.compile(r"(?:always|never|do not|don't) (?P<value>[^.\n]+)", re.IGNORECASE),
        ]

    def extract(self, text: str, session_id: str = "") -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        normalized = text.strip()
        if not normalized:
            return candidates

        candidates.extend(self._extract_by_patterns(
            normalized,
            self.preference_patterns,
            MemoryType.PREFERENCE,
            session_id,
        ))
        candidates.extend(self._extract_by_patterns(
            normalized,
            self.constraint_patterns,
            MemoryType.CONSTRAINT,
            session_id,
        ))
        return self._dedupe(candidates)

    def _extract_by_patterns(
        self,
        text: str,
        patterns: list[re.Pattern],
        memory_type: MemoryType,
        session_id: str,
    ) -> list[MemoryCandidate]:
        spec = get_memory_type_spec(memory_type)
        candidates: list[MemoryCandidate] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = self._clean_value(match.group("value"))
                if not value:
                    continue
                content = self._format_content(memory_type, value)
                candidates.append(MemoryCandidate(
                    content=content,
                    memory_type=memory_type.value,
                    confidence=spec.default_confidence,
                    metadata={
                        "session_id": session_id,
                        "source": "explicit_user_message",
                        "extraction_method": "deterministic_pattern",
                        "retrieval_priority": spec.retrieval_priority,
                        "ttl_days": spec.ttl_days,
                        "durable": spec.durable,
                    },
                    keywords=self._keywords(value),
                ))
        return candidates

    def _clean_value(self, value: str) -> str:
        cleaned = value.strip(" ：:，,。.!！?？ \t")
        return cleaned[:self.max_candidate_chars]

    @staticmethod
    def _format_content(memory_type: MemoryType, value: str) -> str:
        if memory_type == MemoryType.PREFERENCE:
            return f"用户偏好: {value}"
        if memory_type == MemoryType.CONSTRAINT:
            return f"用户约束: {value}"
        return value

    @staticmethod
    def _keywords(value: str) -> list[str]:
        tokens = [token.strip(" ，,。.!！?？") for token in re.split(r"\s+", value) if token.strip()]
        if len(tokens) > 1:
            return tokens[:5]
        return [value[:20]] if value else []

    @staticmethod
    def _dedupe(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        seen: set[tuple[str, str]] = set()
        deduped: list[MemoryCandidate] = []
        for candidate in candidates:
            key = (candidate.memory_type, candidate.content)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped
