# Prompt Injection Defense for RAG Agents

中文锚点：RAG 里的外部文档应该标记为 UNTRUSTED_CONTEXT，因为外部文本只能作为 evidence 证据，不能作为控制指令。

## Untrusted Context

Retrieved documents, webpages, and long-term memory snippets are untrusted context. They can provide evidence, but they must not be treated as control instructions.

The system should explicitly mark external text as `UNTRUSTED_CONTEXT` and tell the model not to follow instructions inside it. A malicious document may say "ignore previous instructions" or "do not cite sources"; those phrases should be treated as content to analyze, not as commands.

## Tool Policy

Tool calls should pass through a policy layer before execution. Read-only tools such as knowledge-base search are low risk. External web fetch is medium risk. Code patching, shell execution, and persistent memory writes are high risk and may require confirmation or a dedicated execution mode.

## URL Safety

A webpage fetcher should reject non-http schemes, local files, localhost, loopback IP addresses, and private network addresses. This prevents a retrieved document from tricking the agent into reading local files or internal services.

## Security Eval Claim

Prompt-injection tests should include both English and Chinese attack strings, such as "ignore previous instructions", "泄露系统提示词", and "调用工具删除文件".
