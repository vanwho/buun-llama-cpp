"""Small negative tests for completion guardrails; run with unittest discovery."""
import hashlib
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("v10_validate", Path(__file__).with_name("validate.py"))
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ReceiptTests(unittest.TestCase):
    def receipt(self):
        path = Path(__file__).resolve()
        return {"schema_version": 1, "task": "49-02", "source_commit": "a" * 40,
                "checks": {"cuda_cold_eligibility": {"status": "pass", "exit_code": 0,
                    "command": ["test-kv-page-select", "--backend", "CUDA0"],
                    "artifacts": [{"path": str(path),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}}}

    def check(self, receipt):
        return validator.check_receipt(Path(__file__).parent,
            {"id": "49-02", "required_proofs": ["cuda_cold_eligibility"]}, receipt)

    def test_valid_shape(self):
        self.assertFalse(self.check(self.receipt()))

    def test_missing_proof(self):
        receipt = self.receipt()
        receipt["checks"] = {}
        self.assertTrue(self.check(receipt))

    def test_deferred_is_not_pass(self):
        receipt = self.receipt()
        receipt["checks"]["cuda_cold_eligibility"]["status"] = "deferred"
        self.assertTrue(self.check(receipt))

    def test_failed_command(self):
        receipt = self.receipt()
        receipt["checks"]["cuda_cold_eligibility"]["exit_code"] = 1
        self.assertTrue(self.check(receipt))

    def test_changed_raw_output(self):
        receipt = self.receipt()
        receipt["checks"]["cuda_cold_eligibility"]["artifacts"][0]["sha256"] = "0" * 64
        self.assertTrue(self.check(receipt))

    def test_missing_raw_output(self):
        receipt = self.receipt()
        receipt["checks"]["cuda_cold_eligibility"]["artifacts"][0]["path"] = "missing-v10-test-output"
        self.assertTrue(self.check(receipt))

    def test_unmet_review_needs_real_tasks(self):
        self.assertTrue(validator.check_review({"tasks": []}, {"goal_met": False, "next_task_ids": []}))

    def test_goal_true_needs_capability_fields(self):
        self.assertTrue(validator.check_review({"tasks": []}, {"goal_met": True}))


if __name__ == "__main__":
    unittest.main()
