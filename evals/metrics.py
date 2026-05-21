from __future__ import annotations


def accuracy(items: list[bool]) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"

