import unittest

from memory.long_term import LongTermMemory, MemoryConflict, MemoryEntry


class MemoryConflictTests(unittest.TestCase):
    def test_surface_conflict_detector_flags_negated_claim(self):
        self.assertTrue(
            LongTermMemory._looks_conflicting(
                "RAG can improve citation accuracy",
                "RAG cannot improve citation accuracy",
            )
        )

    def test_surface_conflict_detector_allows_compatible_claim(self):
        self.assertFalse(
            LongTermMemory._looks_conflicting(
                "RAG can improve citation accuracy",
                "RAG can improve answer grounding",
            )
        )

    def test_preference_conflict_is_classified_as_temporal_update(self):
        existing = MemoryEntry("用户偏好: 使用中文回答", "preference")
        conflict_type, status, strategy = LongTermMemory._classify_conflict(existing, "preference")

        self.assertEqual(conflict_type, "temporal_update")
        self.assertEqual(status, "resolved")
        self.assertEqual(strategy, "supersede_old_memory")

    def test_research_summary_conflict_stays_unresolved(self):
        existing = MemoryEntry("RAG can improve citation accuracy", "research_summary")
        conflict_type, status, strategy = LongTermMemory._classify_conflict(existing, "research_summary")

        self.assertEqual(conflict_type, "contradiction")
        self.assertEqual(status, "unresolved")
        self.assertEqual(strategy, "manual_review")

    def test_mark_superseded_records_memory_state(self):
        entry = MemoryEntry("用户偏好: 先看细节", "preference", metadata={"source": "user"})
        LongTermMemory._mark_superseded(entry, incoming_index=3)

        self.assertEqual(entry.status, "superseded")
        self.assertEqual(entry.superseded_by, 3)
        self.assertEqual(entry.metadata["memory_status"], "superseded")

    def test_status_weight_downranks_conflicted_and_skips_superseded(self):
        self.assertEqual(LongTermMemory._status_weight("active"), 1.0)
        self.assertEqual(LongTermMemory._status_weight("conflicted"), 0.2)
        self.assertEqual(LongTermMemory._status_weight("superseded"), 0.0)

    def test_resolve_conflict_can_use_incoming_memory(self):
        memory = LongTermMemory.__new__(LongTermMemory)
        memory.entries = [
            MemoryEntry("旧结论", "research_summary", status="conflicted"),
            MemoryEntry("新结论", "research_summary", status="conflicted"),
        ]
        memory.conflicts = [
            MemoryConflict(
                existing_index=0,
                incoming_index=1,
                existing_content="旧结论",
                incoming_content="新结论",
                similarity=0.95,
            )
        ]
        memory.save = lambda: None

        result = memory.resolve_conflict(0, "use_incoming")

        self.assertTrue(result["resolved"])
        self.assertEqual(memory.conflicts[0].status, "resolved")
        self.assertEqual(memory.entries[0].status, "superseded")
        self.assertEqual(memory.entries[1].status, "active")

    def test_get_conflict_reviews_exposes_pending_review_payload(self):
        memory = LongTermMemory.__new__(LongTermMemory)
        memory.entries = [
            MemoryEntry("旧结论", "research_summary", status="conflicted"),
            MemoryEntry("新结论", "research_summary", status="conflicted"),
        ]
        memory.conflicts = [
            MemoryConflict(
                existing_index=0,
                incoming_index=1,
                existing_content="旧结论",
                incoming_content="新结论",
                similarity=0.95,
            )
        ]

        reviews = memory.get_conflict_reviews()

        self.assertEqual(reviews[0]["conflict_index"], 0)
        self.assertEqual(reviews[0]["existing_status"], "conflicted")
        self.assertEqual(reviews[0]["incoming_status"], "conflicted")


if __name__ == "__main__":
    unittest.main()
