"""
Deterministic prompt-injection scanner for retrieved/web content.

This is not meant to be a perfect classifier. It is a cheap first-pass guard
that labels suspicious external text before it reaches the model or tool layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class InjectionPattern:
    name: str
    regex: str
    severity: float
    description: str


@dataclass
class InjectionFinding:
    name: str
    matched_text: str
    severity: float
    description: str


@dataclass
class InjectionScanResult:
    risk_score: float
    findings: list[InjectionFinding] = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        return self.risk_score >= 0.5

    def summary(self) -> str:
        if not self.findings:
            return "no suspicious prompt-injection patterns detected"
        names = ", ".join(f.name for f in self.findings[:5])
        return f"risk={self.risk_score:.2f}; findings={names}"


PATTERNS: tuple[InjectionPattern, ...] = (
    InjectionPattern(
        "ignore_previous_instructions",
        r"(ignore|disregard|forget).{0,40}(previous|above|prior).{0,40}(instruction|prompt|message)s?",
        0.8,
        "Attempts to override earlier instructions",
    ),
    InjectionPattern(
        "chinese_ignore_instructions",
        r"(忽略|无视|忘记).{0,20}(之前|以上|上面|所有).{0,20}(指令|提示|规则)",
        0.8,
        "Chinese instruction override attempt",
    ),
    InjectionPattern(
        "system_prompt_exfiltration",
        r"(reveal|print|show|dump|泄露|输出|显示).{0,40}(system prompt|developer message|系统提示|开发者指令)",
        0.9,
        "Attempts to exfiltrate hidden prompts",
    ),
    InjectionPattern(
        "tool_abuse",
        r"(call|invoke|run|execute|调用|运行|执行).{0,40}(tool|function|shell|bash|命令|工具)",
        0.7,
        "Attempts to drive tool execution from external text",
    ),
    InjectionPattern(
        "credential_theft",
        r"(api[_ -]?key|token|password|secret|凭证|密钥|密码).{0,40}(send|exfiltrate|leak|上传|发送|泄露)",
        0.95,
        "Attempts to steal secrets",
    ),
    InjectionPattern(
        "citation_suppression",
        r"(do not|don't|不要|无需).{0,30}(cite|citation|引用|来源)",
        0.45,
        "Attempts to suppress citations",
    ),
)


def scan_text(text: str, max_findings: int = 10) -> InjectionScanResult:
    findings: list[InjectionFinding] = []
    for pattern in PATTERNS:
        for match in re.finditer(pattern.regex, text, flags=re.IGNORECASE | re.DOTALL):
            matched_text = " ".join(match.group(0).split())
            findings.append(
                InjectionFinding(
                    name=pattern.name,
                    matched_text=matched_text[:160],
                    severity=pattern.severity,
                    description=pattern.description,
                )
            )
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break

    if not findings:
        return InjectionScanResult(risk_score=0.0)

    strongest = max(f.severity for f in findings)
    density_bonus = min(0.2, 0.03 * (len(findings) - 1))
    return InjectionScanResult(risk_score=min(1.0, strongest + density_bonus), findings=findings)

