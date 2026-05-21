from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path


@dataclass
class EvalDocument:
    doc_id: str
    title: str
    content: str


@dataclass
class RetrievalHit:
    doc: EvalDocument
    score: float
    matched_terms: list[str]


STOPWORDS = {
    "应该", "怎么", "该怎", "时应", "么处", "什么", "哪些", "为什么",
    "文档", "系统", "一个", "这个", "时候", "可以",
    "agent", "tool",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cn_bigrams = [cn_chars[i] + cn_chars[i + 1] for i in range(len(cn_chars) - 1)]
    return [token for token in words + cn_bigrams if token not in STOPWORDS]


def load_documents(kb_dir: Path) -> list[EvalDocument]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        docs.append(EvalDocument(doc_id=path.name, title=path.name, content=path.read_text(encoding="utf-8")))
    return docs


class LexicalRetriever:
    def __init__(self, docs: list[EvalDocument]):
        self.docs = docs
        self.doc_tokens = [tokenize(doc.content + "\n" + doc.title) for doc in docs]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.doc_freq: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freq[token] += 1

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        query_terms = tokenize(query)
        scored = []
        for doc, tokens, doc_len in zip(self.docs, self.doc_tokens, self.doc_lens):
            tf = Counter(tokens)
            score = 0.0
            matched_terms = []
            for term in query_terms:
                if term not in tf:
                    continue
                matched_terms.append(term)
                df = self.doc_freq.get(term, 0)
                idf = log((len(self.docs) - df + 0.5) / (df + 0.5) + 1)
                score += idf * tf[term] * 2.5 / (tf[term] + 1.5 * (0.25 + 0.75 * doc_len / self.avg_doc_len))
            scored.append(RetrievalHit(doc=doc, score=score, matched_terms=sorted(set(matched_terms))[:20]))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_kb_eval(kb_dir: Path, cases_path: Path, top_k: int = 3) -> dict:
    docs = load_documents(kb_dir)
    retriever = LexicalRetriever(docs)
    cases = load_cases(cases_path)
    results = []
    for case in cases:
        hits = retriever.search(case["query"], top_k=top_k)
        top_doc = hits[0].doc.doc_id if hits else ""
        expected_doc = case["expected_doc"]
        rank = next((i + 1 for i, hit in enumerate(hits) if hit.doc.doc_id == expected_doc), None)
        expected_text = next((doc.content for doc in docs if doc.doc_id == expected_doc), "")
        phrase_checks = {
            phrase: phrase.lower() in expected_text.lower()
            for phrase in case.get("must_include", [])
        }
        results.append({
            "id": case["id"],
            "query": case["query"],
            "expected_doc": expected_doc,
            "top_doc": top_doc,
            "rank": rank,
            "top1_pass": rank == 1,
            "topk_pass": rank is not None,
            "must_include_pass": all(phrase_checks.values()),
            "phrase_checks": phrase_checks,
            "hits": [
                {"doc": hit.doc.doc_id, "score": round(hit.score, 4), "matched_terms": hit.matched_terms}
                for hit in hits
            ],
        })

    total = len(results)
    top1 = sum(1 for r in results if r["top1_pass"])
    topk = sum(1 for r in results if r["topk_pass"])
    phrase = sum(1 for r in results if r["must_include_pass"])
    return {
        "suite": "kb_retrieval",
        "total": total,
        "top1_accuracy": top1 / max(total, 1),
        "topk_recall": topk / max(total, 1),
        "must_include_accuracy": phrase / max(total, 1),
        "results": results,
    }
