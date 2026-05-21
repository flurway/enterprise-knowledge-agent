"""
Answer-level validation helpers.

These checks are intentionally deterministic. They do not decide whether the
answer is semantically perfect; they catch cheap, high-signal failures such as
unknown citation ids, missing citations when sources exist, and compliance with
obvious prompt-injection instructions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AnswerValidationResult:
    ok: bool
    cited_ids: list[str] = field(default_factory=list)
    unknown_citation_ids: list[str] = field(default_factory=list)
    missing_citations: bool = False
    unsafe_compliance_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "cited_ids": self.cited_ids,
            "unknown_citation_ids": self.unknown_citation_ids,
            "missing_citations": self.missing_citations,
            "unsafe_compliance_signals": self.unsafe_compliance_signals,
        }


CITATION_PATTERN = re.compile(r"\[(?:来源|source)?\s*(\d+)\]", re.IGNORECASE)

UNSAFE_COMPLIANCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("claims_no_citations_needed", r"(不需要|无需|不用).{0,12}(引用|来源|cite|citation)"),
    ("mentions_revealing_prompt", r"(系统提示词|system prompt).{0,20}(如下|是|:|：)"),
    ("mentions_secret_exfiltration", r"(api[_ -]?key|token|secret|密钥|密码).{0,20}(是|如下|:|：)"),
    ("claims_tool_execution_from_doc", r"(我已|已经).{0,12}(调用|执行|运行).{0,12}(工具|shell|命令)"),
)


def extract_citation_ids(answer: str) -> list[str]:
    ids = CITATION_PATTERN.findall(answer)
    seen = set()
    ordered = []
    for cid in ids:
        normalized = cid.strip()
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def detect_unsafe_compliance(answer: str) -> list[str]:
    signals = []
    for name, pattern in UNSAFE_COMPLIANCE_PATTERNS:
        if re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL):
            signals.append(name)
    return signals


def validate_answer(answer: str, citations: dict, require_citation_when_sources_exist: bool = True) -> AnswerValidationResult:
    cited_ids = extract_citation_ids(answer)
    known_ids = {str(cid) for cid in citations.keys()}
    unknown = [cid for cid in cited_ids if cid not in known_ids]
    missing = bool(citations) and require_citation_when_sources_exist and not cited_ids
    unsafe_signals = detect_unsafe_compliance(answer)
    ok = not unknown and not missing and not unsafe_signals
    return AnswerValidationResult(
        ok=ok,
        cited_ids=cited_ids,
        unknown_citation_ids=unknown,
        missing_citations=missing,
        unsafe_compliance_signals=unsafe_signals,
    )

