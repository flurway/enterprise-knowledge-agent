# Memory Conflict and Context Pollution

中文锚点：长期记忆相似但矛盾时，系统应该记录 unresolved conflict record，preserve both pieces of evidence，保留新旧证据，而不是静默覆盖旧记忆。这个页面专门回答“长期记忆相似但矛盾时应该怎么处理”。

## Memory Types

Long-term memory should separate stable facts, user preferences, research episodes, uncertain hypotheses, and conflict records. Treating every summary as the same memory type makes it difficult to reason about trust and freshness.

## Conflict Handling

When a new memory is very similar to an old memory but appears to contradict it, the system should not silently overwrite the old entry. It should create an unresolved conflict record, preserve both pieces of evidence, and delay the final decision until a stronger source or user confirmation is available.

## Context Pollution

Long-context pollution happens when irrelevant previous conversation, stale memory, or unrelated tool output changes the model's answer. Useful defenses include retrieval budgets, dependency-scoped step context, memory confidence scores, source grounding, and explicit conflict metadata.

## Important Claim

The safest initial behavior for uncertain memory conflict detection is to record the conflict first and decide later. A heuristic detector can be used as a cheap prefilter, but it should not be presented as a final semantic contradiction classifier.
