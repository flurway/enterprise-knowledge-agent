from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from evals.kb_eval import load_cases
from rag.chunker import StructuredChunker
from rag.reranker import CitationTracer
from rag.retriever import BM25Index, HybridRetriever, RetrievalResult


class SimpleIPIndex:
    """Tiny in-memory inner-product index with a FAISS-like interface."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.size == 0:
            return
        self.vectors = np.vstack([self.vectors, embeddings.astype(np.float32)])

    def search(self, query_embeddings: np.ndarray, top_k: int):
        if self.ntotal == 0:
            return np.zeros((len(query_embeddings), 0), dtype=np.float32), np.zeros((len(query_embeddings), 0), dtype=np.int64)
        scores = query_embeddings @ self.vectors.T
        k = min(top_k, self.ntotal)
        indices = np.argsort(-scores, axis=1)[:, :k]
        sorted_scores = np.take_along_axis(scores, indices, axis=1)
        return sorted_scores.astype(np.float32), indices.astype(np.int64)


class DeterministicHybridRetriever(HybridRetriever):
    """
    HybridRetriever variant for evals.

    It keeps the production search path (dense + BM25 + RRF) but replaces
    sentence-transformers/FAISS with deterministic local hashing so the eval can
    run offline and never mutate the real index.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__(embedding_dim=embedding_dim)
        self.index_backend = "uninitialized"

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.embedding_dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, idx] += sign
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors

    def build_index(self, chunks):
        self.chunks = chunks
        embeddings = self.get_embeddings([chunk.content for chunk in chunks])
        try:
            import faiss

            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index_backend = "faiss.IndexFlatIP"
        except ImportError:
            self.faiss_index = SimpleIPIndex(embeddings.shape[1])
            self.index_backend = "simple_ip_fallback"
        self.faiss_index.add(embeddings)
        self.bm25_index = BM25Index()
        self.bm25_index.add_documents(chunks)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
        cn_bigrams = [cn_chars[i] + cn_chars[i + 1] for i in range(len(cn_chars) - 1)]
        return words + cn_bigrams


def build_fixture_retriever(kb_dir: Path) -> DeterministicHybridRetriever:
    chunker = StructuredChunker(chunk_size=2000, chunk_overlap=80)
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            chunker.chunk_document(
                text=text,
                doc_id=path.stem,
                doc_title=path.name,
                metadata={"source": "fixture_kb", "filename": path.name},
            )
        )
    retriever = DeterministicHybridRetriever()
    retriever.build_index(chunks)
    return retriever


async def run_rag_pipeline_eval(kb_dir: Path, cases_path: Path, top_k: int = 5) -> dict:
    retriever = build_fixture_retriever(kb_dir)
    cases = load_cases(cases_path)
    tracer = CitationTracer()
    results = []

    for case in cases:
        hits = await retriever.search(
            query=case["query"],
            top_k_dense=top_k,
            top_k_sparse=top_k,
        )
        hits = hits[:top_k]
        expected_doc = case["expected_doc"]
        rank = next((i + 1 for i, hit in enumerate(hits) if hit.chunk.doc_title == expected_doc), None)
        retrieved_context = "\n\n".join(hit.chunk.content for hit in hits)
        phrase_checks = {
            phrase: phrase.lower() in retrieved_context.lower()
            for phrase in case.get("must_include", [])
        }
        citation_context, citations = tracer.build_citation_context(hits[:2], start_index=5)
        results.append({
            "id": case["id"],
            "query": case["query"],
            "expected_doc": expected_doc,
            "top_doc": hits[0].chunk.doc_title if hits else "",
            "rank": rank,
            "top1_pass": rank == 1,
            "topk_pass": rank is not None,
            "must_include_pass": all(phrase_checks.values()),
            "phrase_checks": phrase_checks,
            "citation_start_pass": "[来源5]" in citation_context and "5" in citations,
            "hits": [
                {
                    "doc": hit.chunk.doc_title,
                    "chunk_id": hit.chunk.chunk_id,
                    "score": round(hit.score, 6),
                    "source": hit.source,
                }
                for hit in hits
            ],
        })

    total = len(results)
    return {
        "suite": "rag_pipeline",
        "index_backend": retriever.index_backend,
        "total": total,
        "top1_accuracy": sum(1 for r in results if r["top1_pass"]) / max(total, 1),
        "topk_recall": sum(1 for r in results if r["topk_pass"]) / max(total, 1),
        "must_include_accuracy": sum(1 for r in results if r["must_include_pass"]) / max(total, 1),
        "citation_start_accuracy": sum(1 for r in results if r["citation_start_pass"]) / max(total, 1),
        "results": results,
    }


def report_summary(report: dict) -> dict:
    return {
        "suite": report["suite"],
        "index_backend": report["index_backend"],
        "total": report["total"],
        "top1_accuracy": report["top1_accuracy"],
        "topk_recall": report["topk_recall"],
        "must_include_accuracy": report["must_include_accuracy"],
        "citation_start_accuracy": report["citation_start_accuracy"],
    }
