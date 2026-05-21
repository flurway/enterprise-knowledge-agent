import unittest
from types import SimpleNamespace

from security.untrusted_flow import suspicious_dependency_ids


class UntrustedFlowTest(unittest.TestCase):
    def test_detects_suspicious_injection_scan_dependency(self):
        execution_context = {
            1: SimpleNamespace(data={"injection_scan": {"is_suspicious": True}}),
            2: SimpleNamespace(data={"injection_scan": {"is_suspicious": False}}),
        }

        self.assertEqual(suspicious_dependency_ids(execution_context, [1, 2]), [1])

    def test_propagates_already_untrusted_dependency(self):
        execution_context = {
            1: SimpleNamespace(data={"triggered_by_untrusted_context": True}),
            2: SimpleNamespace(data={}),
        }

        self.assertEqual(suspicious_dependency_ids(execution_context, [1, 2]), [1])

    def test_ignores_missing_dependencies(self):
        self.assertEqual(suspicious_dependency_ids({}, [99]), [])


if __name__ == "__main__":
    unittest.main()
