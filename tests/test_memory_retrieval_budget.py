import unittest

from memory.retrieval_budget import MemoryRetrievalBudget


class MemoryRetrievalBudgetTests(unittest.TestCase):
    def test_prioritizes_preferences_and_constraints_over_research_summaries(self):
        memories = [
            {"content": "研究摘要 A", "memory_type": "research_summary", "score": 0.95, "age_days": 0},
            {"content": "用户偏好 A", "memory_type": "preference", "score": 0.65, "age_days": 3},
            {"content": "用户约束 A", "memory_type": "constraint", "score": 0.62, "age_days": 1},
        ]

        selected = MemoryRetrievalBudget(max_items=2).select(memories)

        self.assertEqual([m["memory_type"] for m in selected], ["preference", "constraint"])

    def test_enforces_per_type_limit(self):
        memories = [
            {"content": "偏好 A", "memory_type": "preference", "score": 0.9, "age_days": 0},
            {"content": "偏好 B", "memory_type": "preference", "score": 0.8, "age_days": 0},
            {"content": "偏好 C", "memory_type": "preference", "score": 0.7, "age_days": 0},
            {"content": "摘要 A", "memory_type": "research_summary", "score": 0.6, "age_days": 0},
        ]

        selected = MemoryRetrievalBudget(max_items=4).select(memories)

        self.assertEqual([m["content"] for m in selected], ["偏好 A", "偏好 B", "摘要 A"])

    def test_filters_low_score_and_conflict_records_from_normal_retrieval(self):
        memories = [
            {"content": "低分偏好", "memory_type": "preference", "score": 0.1, "age_days": 0},
            {"content": "冲突记录", "memory_type": "conflict", "score": 0.9, "age_days": 0},
            {"content": "摘要 A", "memory_type": "research_summary", "score": 0.7, "age_days": 0},
        ]

        selected = MemoryRetrievalBudget(max_items=3).select(memories)

        self.assertEqual([m["content"] for m in selected], ["摘要 A"])

    def test_budget_can_be_built_from_config_object(self):
        class DummyMemoryConfig:
            max_long_term_results = 4
            retrieval_min_score = 0.5
            retrieval_type_limits = {"preference": 1, "constraint": 1, "research_summary": 1}

        budget = MemoryRetrievalBudget.from_config(DummyMemoryConfig())

        self.assertEqual(budget.max_items, 4)
        self.assertEqual(budget.min_score, 0.5)
        self.assertEqual(budget.per_type_limits["preference"], 1)
        self.assertEqual(budget.to_dict()["max_items"], 4)


if __name__ == "__main__":
    unittest.main()
