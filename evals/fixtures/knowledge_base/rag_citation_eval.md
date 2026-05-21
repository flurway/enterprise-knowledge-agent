# RAG Citation Evaluation

## Hybrid Retrieval

A useful research knowledge base should test both semantic and lexical retrieval. Dense retrieval is good at paraphrase matching, while BM25 is good at exact terms such as `IndexFlatIP`, `RRF`, `MCP`, and `UNTRUSTED_CONTEXT`.

## Citation Accuracy

Citation accuracy means the answer's cited source actually supports the claim. A system can retrieve a relevant document but still cite the wrong chunk if citation identifiers are reused across multiple searches.

## Global Citation Numbering

When an agent performs multiple searches in one run, citation identifiers should be globally unique within the run. If every retrieval starts again from `[来源1]`, later citations can overwrite earlier ones and make the final answer unverifiable.

## Evaluation Cases

Good RAG evaluation cases should specify the query, the expected source document, and one or more phrases that must appear in the retrieved context. This makes regressions easier to catch.

