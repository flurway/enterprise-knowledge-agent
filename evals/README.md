# Evals

This directory contains lightweight evaluation cases and runners.

The first harness is intentionally offline: it checks deterministic safety and
context-boundary behavior without calling an LLM or web search provider. Later
iterations can add model-backed RAG and agent-task evals next to these cases.

Current security cases cover prompt-injection detection, URL/response safety,
and cross-step untrusted-flow propagation into tool policy blocking.

## Run

```bash
python evals/run_all.py --no-report
python evals/run_evals.py --suite security
python evals/run_kb_eval.py --no-report
python evals/run_answer_eval.py --no-report
python evals/run_memory_write_eval.py --no-report
python evals/run_memory_conflict_eval.py --no-report
python evals/run_memory_retrieval_eval.py --no-report
python evals/run_kb_eval.py --kb-dir evals/fixtures/enterprise_knowledge_base --cases evals/cases/enterprise_kb_cases.jsonl --top-k 3 --no-report
```

Reports are written to `evals/reports/`.

`run_all.py` is the Phase 1 baseline command. It runs:

- security eval
- KB lexical retrieval smoke test
- RAG pipeline eval
- answer-level validation eval
- memory write-gate eval
- memory conflict-detector eval
- memory retrieval-budget eval

## Knowledge-Base Fixture

`evals/fixtures/knowledge_base/` contains a small synthetic knowledge base for
testing the project itself:

- agent runtime trace and replan reliability
- memory conflict and context pollution
- prompt-injection defense and tool policy
- citation accuracy and global citation numbering
- backend schema, MySQL index, and network reliability talking points
- one adversarial document that should be treated as untrusted context

The offline KB eval uses a deterministic lexical retriever so it can run without
LLM calls, web search, FAISS, or sentence-transformers downloads.

`evals/fixtures/enterprise_knowledge_base/` contains a larger synthetic
enterprise business knowledge base for testing the target scenario of a
reliable collaboration agent over company documents. It models a fictional SaaS
company with product, sales, legal, customer success, support, security,
finance, operations, release-note, competitor, and adversarial partner
documents. The cases intentionally include cross-department dependencies,
version conflicts, approval boundaries, compliance wording, private deployment
constraints, and prompt-injection-like external text.

Run the enterprise fixture with:

```bash
python evals/run_kb_eval.py --kb-dir evals/fixtures/enterprise_knowledge_base --cases evals/cases/enterprise_kb_cases.jsonl --top-k 3 --no-report
```

`run_rag_pipeline_eval.py` runs the small technical fixture through the project
retrieval pipeline shape: chunking, dense retrieval, BM25, RRF fusion, and
citation context generation. It uses deterministic local embeddings so no model
download is needed. When FAISS is available, it uses `faiss.IndexFlatIP`;
otherwise it falls back to a tiny in-memory inner-product index.

```bash
conda run -n common-env python evals/run_rag_pipeline_eval.py --no-report
```

`run_answer_eval.py` checks answer-level contracts: cited source ids must exist,
answers should not omit citations when sources are available, and obvious
compliance with prompt-injection instructions is flagged.

`run_memory_conflict_eval.py` checks the current deterministic conflict
detector against positive and negative cases. It is intentionally a regression
suite for the current policy, not a claim that surface heuristics solve all
semantic contradiction detection.

`run_memory_retrieval_eval.py` checks the retrieval budget policy that selects
which long-term memories enter ordinary answer context. It gives durable user
preferences and constraints priority, caps repeated memory types, and keeps
conflict records out of normal recall.
