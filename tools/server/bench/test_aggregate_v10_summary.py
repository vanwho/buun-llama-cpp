"""Regression tests for the v10 summary's anti-stale-data guardrails."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("aggregate_v10_summary", HERE / "aggregate-v10-summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = HERE.parents[2] / ".wiretail/execution/evidence"
        cls.small = json.loads((root / "v10-51-02/V10_SMALL.json").read_text())

    def test_current_bundle_is_accepted(self):
        self.assertFalse(summary.validate_small(self.small))

    def test_phase47_style_stale_bundle_is_rejected(self):
        fixture = copy.deepcopy(self.small)
        for mode in summary.MODES:
            fixture["matrix"][mode]["identity"]["bundle_identity"] = "/srv/ai/paged-kv/results/v9/43-02/candidate/bin/llama-server"
        self.assertTrue(summary.validate_small(fixture))

    def test_thirty_token_workload_is_rejected(self):
        fixture = copy.deepcopy(self.small)
        fixture["configuration"]["prompt_tokens"] = 30
        fixture["matrix"]["selective-v2"]["identity"]["runtime"]["measured_request_tokens"] = 30
        self.assertTrue(summary.validate_small(fixture))

    def test_h_page_count_mismatch_is_rejected(self):
        fixture = copy.deepcopy(self.small)
        fixture["matrix"]["selective-v2"]["identity"]["runtime"]["hot_capacity_pages"] = 15
        self.assertTrue(summary.validate_small(fixture))

    def test_unmatched_question_keeps_ratio_null(self):
        fixture = copy.deepcopy(self.small)
        fixture["matrix"]["cpu-main"]["questions"]["1"]["cases"].pop()
        errors = summary.validate_small(fixture)
        ratios = summary.paired_ratios(fixture, errors)
        self.assertTrue(errors)
        self.assertIsNone(ratios["1"]["selective-v2_over_cpu-main"])


if __name__ == "__main__":
    unittest.main()
