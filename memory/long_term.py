"""
长期记忆 — FAISS 向量检索 + 时间衰减 + 去重
"""
import os
import json
import time
import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    content: str
    memory_type: str
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    confidence: float = 1.0
    status: str = "active"
    superseded_by: Optional[int] = None
    valid_until: Optional[float] = None


@dataclass
class MemoryConflict:
    existing_index: int
    existing_content: str
    incoming_content: str
    similarity: float
    incoming_index: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    status: str = "unresolved"
    reason: str = ""
    conflict_type: str = "contradiction"
    resolution_strategy: str = "manual_review"


class LongTermMemory:
    def __init__(self, index_path: str, embedding_dim: int = 512):
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.entries: list[MemoryEntry] = []
        self.conflicts: list[MemoryConflict] = []
        self.index = None
        os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
        self._load()

    def _load(self):
        meta_path = f"{self.index_path}_meta.json"
        index_file = f"{self.index_path}.faiss"
        try:
            import faiss
            if os.path.exists(index_file) and os.path.exists(meta_path):
                self.index = faiss.read_index(index_file)
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = [MemoryEntry(**e) for e in data.get("entries", [])]
                self.conflicts = [MemoryConflict(**c) for c in data.get("conflicts", [])]
            else:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
        except ImportError:
            logger.warning("FAISS not installed, long-term memory disabled")
            self.index = None

    def save(self):
        if self.index is None:
            return
        import faiss
        os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else ".", exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "entries": [
                    {"content": e.content, "memory_type": e.memory_type, "timestamp": e.timestamp,
                     "session_id": e.session_id, "keywords": e.keywords, "metadata": e.metadata,
                     "confidence": e.confidence, "status": e.status,
                     "superseded_by": e.superseded_by, "valid_until": e.valid_until}
                    for e in self.entries
                ],
                "conflicts": [
                    {"existing_index": c.existing_index, "existing_content": c.existing_content,
                     "incoming_content": c.incoming_content, "similarity": c.similarity,
                     "incoming_index": c.incoming_index,
                     "timestamp": c.timestamp, "status": c.status, "reason": c.reason,
                     "conflict_type": c.conflict_type,
                     "resolution_strategy": c.resolution_strategy}
                    for c in self.conflicts
                ],
            }, f, ensure_ascii=False, indent=2)

    async def add_memory(self, content: str, memory_type: str, embedding: np.ndarray,
                         session_id: str = "", keywords: list[str] = None,
                         metadata: dict = None, confidence: float = 1.0):
        if self.index is None:
            return
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        embedding = embedding.astype(np.float32).reshape(1, -1)
        metadata = metadata or {}
        incoming_status = "active"

        if self.index.ntotal > 0:
            scores, indices = self.index.search(embedding, 1)
            if scores[0][0] > 0.9:
                idx = indices[0][0]
                existing = self.entries[idx]
                if self._looks_conflicting(existing.content, content):
                    incoming_index = len(self.entries)
                    conflict_type, conflict_status, resolution_strategy = self._classify_conflict(
                        existing, memory_type
                    )
                    metadata = {
                        **metadata,
                        "memory_conflict": True,
                        "conflict_with": int(idx),
                        "conflict_type": conflict_type,
                    }
                    if conflict_type == "temporal_update":
                        self._mark_superseded(existing, incoming_index)
                        metadata["supersedes"] = int(idx)
                    else:
                        existing.status = "conflicted"
                        existing.metadata = {**existing.metadata, "memory_conflict": True}
                        incoming_status = "conflicted"
                    self.conflicts.append(MemoryConflict(
                        existing_index=int(idx),
                        existing_content=existing.content,
                        incoming_content=content,
                        similarity=float(scores[0][0]),
                        incoming_index=incoming_index,
                        status=conflict_status,
                        reason="High semantic similarity but contradictory surface cues were detected",
                        conflict_type=conflict_type,
                        resolution_strategy=resolution_strategy,
                    ))
                else:
                    existing.content = content
                    existing.timestamp = time.time()
                    existing.keywords = keywords or []
                    existing.metadata = metadata
                    existing.confidence = confidence
                    existing.status = "active"
                    existing.superseded_by = None
                    existing.valid_until = None
                    self.save()
                    return

        self.entries.append(MemoryEntry(
            content=content, memory_type=memory_type, session_id=session_id,
            keywords=keywords or [], metadata=metadata, confidence=confidence,
            status=incoming_status,
        ))
        self.index.add(embedding)
        self.save()

    @staticmethod
    def _classify_conflict(existing: MemoryEntry, incoming_memory_type: str) -> tuple[str, str, str]:
        if existing.memory_type == incoming_memory_type and incoming_memory_type in {"preference", "constraint"}:
            return "temporal_update", "resolved", "supersede_old_memory"
        return "contradiction", "unresolved", "manual_review"

    @staticmethod
    def _mark_superseded(entry: MemoryEntry, incoming_index: int) -> None:
        now = time.time()
        entry.status = "superseded"
        entry.superseded_by = incoming_index
        entry.valid_until = now
        entry.metadata = {
            **entry.metadata,
            "memory_status": "superseded",
            "superseded_by": incoming_index,
            "superseded_at": now,
            "superseded_reason": "temporal_update",
        }

    @staticmethod
    def _status_weight(status: str) -> float:
        if status == "conflicted":
            return 0.2
        if status in {"superseded", "archived"}:
            return 0.0
        return 1.0

    def get_conflict_reviews(self, status: str = "unresolved", limit: int = 5) -> list[dict]:
        reviews = []
        for conflict_index, conflict in enumerate(self.conflicts):
            if status and conflict.status != status:
                continue
            existing = self._entry_at(conflict.existing_index)
            incoming = self._entry_at(conflict.incoming_index)
            reviews.append({
                "conflict_index": conflict_index,
                "status": conflict.status,
                "conflict_type": conflict.conflict_type,
                "resolution_strategy": conflict.resolution_strategy,
                "similarity": conflict.similarity,
                "reason": conflict.reason,
                "existing_index": conflict.existing_index,
                "incoming_index": conflict.incoming_index,
                "existing_content": conflict.existing_content,
                "incoming_content": conflict.incoming_content,
                "existing_status": existing.status if existing else "",
                "incoming_status": incoming.status if incoming else "",
            })
            if len(reviews) >= limit:
                break
        return reviews

    def resolve_conflict(self, conflict_index: int, resolution: str) -> dict:
        if conflict_index < 0 or conflict_index >= len(self.conflicts):
            return {"resolved": False, "reason": "conflict index not found"}
        conflict = self.conflicts[conflict_index]
        if conflict.status != "unresolved":
            return {"resolved": False, "reason": f"conflict already {conflict.status}"}

        existing = self._entry_at(conflict.existing_index)
        incoming = self._entry_at(conflict.incoming_index)
        if existing is None or incoming is None:
            return {"resolved": False, "reason": "conflict entries are missing"}

        if resolution == "use_existing":
            existing.status = "active"
            incoming.status = "superseded"
            incoming.superseded_by = conflict.existing_index
            incoming.valid_until = time.time()
        elif resolution == "use_incoming":
            self._mark_superseded(existing, conflict.incoming_index or 0)
            incoming.status = "active"
            incoming.superseded_by = None
            incoming.valid_until = None
        elif resolution == "keep_both":
            existing.status = "active"
            incoming.status = "active"
        else:
            return {"resolved": False, "reason": f"unsupported resolution: {resolution}"}

        conflict.status = "resolved"
        conflict.resolution_strategy = resolution
        self.save()
        return {
            "resolved": True,
            "conflict_index": conflict_index,
            "resolution": resolution,
            "existing_index": conflict.existing_index,
            "incoming_index": conflict.incoming_index,
        }

    def _entry_at(self, index: Optional[int]) -> Optional[MemoryEntry]:
        if index is None or index < 0 or index >= len(self.entries):
            return None
        return self.entries[index]

    @staticmethod
    def _looks_conflicting(old: str, new: str) -> bool:
        old_lower = old.lower()
        new_lower = new.lower()
        contradiction_pairs = [
            ("支持", "不支持"),
            ("可以", "不可以"),
            ("推荐", "不推荐"),
            ("有效", "无效"),
            ("安全", "不安全"),
            ("is", "is not"),
            ("are", "are not"),
            ("can", "cannot"),
            ("should", "should not"),
            ("recommended", "not recommended"),
        ]
        for positive, negative in contradiction_pairs:
            old_pos = positive in old_lower
            old_neg = negative in old_lower
            new_pos = positive in new_lower
            new_neg = negative in new_lower
            if (old_pos and not old_neg and new_neg) or (new_pos and not new_neg and old_neg):
                return True

        negation_markers = ["不", "没有", "无法", "不能", "not", "never", "cannot", "n't"]
        old_has_negation = any(marker in old_lower for marker in negation_markers)
        new_has_negation = any(marker in new_lower for marker in negation_markers)
        shared_terms = set(old_lower.split()) & set(new_lower.split())
        return bool(shared_terms) and old_has_negation != new_has_negation

    async def search(self, query_embedding: np.ndarray, top_k: int = 5,
                     memory_type: Optional[str] = None, decay_days: int = 30) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        now = time.time()
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.entries):
                continue
            entry = self.entries[idx]
            if memory_type and entry.memory_type != memory_type:
                continue
            status_weight = self._status_weight(entry.status)
            if status_weight <= 0:
                continue
            age_days = (now - entry.timestamp) / 86400
            decay_factor = np.exp(-age_days / decay_days)
            results.append({
                "content": entry.content, "memory_type": entry.memory_type,
                "score": float(score) * decay_factor * entry.confidence * status_weight,
                "age_days": round(age_days, 1),
                "confidence": entry.confidence,
                "status": entry.status,
                "index": int(idx),
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict:
        return {
            "total_memories": len(self.entries),
            "index_size": self.index.ntotal if self.index else 0,
            "conflicts": len(self.conflicts),
            "unresolved_conflicts": len([c for c in self.conflicts if c.status == "unresolved"]),
            "superseded_memories": len([e for e in self.entries if e.status == "superseded"]),
            "conflicted_memories": len([e for e in self.entries if e.status == "conflicted"]),
        }
