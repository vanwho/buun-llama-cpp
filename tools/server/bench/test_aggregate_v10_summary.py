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

    def test_phase59_keeps_all_boundaries_and_row_statuses(self):
        root = HERE.parents[2] / ".wiretail/execution/evidence"
        built, paths = summary.build_phase59(root)
        self.assertEqual("59-03", built["task"])
        self.assertEqual("measured", built["geometry"]["matched_8K"]["status"])
        self.assertEqual("not_run", built["geometry"]["pilot_32K"]["status"])
        self.assertEqual("not_run", built["geometry"]["pilot_128K"]["status"])
        self.assertEqual("measured", built["geometry"]["allocation_256K"]["status"])
        self.assertEqual("failed", built["geometry"]["occupied_C262144"]["status"])
        self.assertEqual("not_run", built["rates"]["cached_append"]["status"])
        self.assertEqual("not_run", built["promotion_edges"]["controlled_physical"]["status"])
        self.assertEqual("not_run", built["promotion_edges"]["organic_physical"]["status"])
        self.assertEqual("not_run", built["answer_quality"]["status"])
        self.assertEqual(27, sum(len(rows) for rows in built["rows"].values()))
        self.assertTrue(all(row["status"] in {"measured", "failed", "not_run"} for rows in built["rows"].values() for row in rows))
        self.assertEqual(59, built["rows"]["selective_native"][0]["mtp"]["denominator"])
        self.assertEqual(0, built["rows"]["selective_native"][0]["mtp"]["accepted_delta"])
        self.assertEqual("off", built["rows"]["all_gpu_control"][0]["mtp"]["status"])
        self.assertIn("full_l", paths)

    def test_repair85_rejects_wrong_context_identity(self):
        native = {"runtime": {"logical_context_tokens": 8192, "measured_request_tokens": 6144}, "measurements": {"committed_tokens": 1}, "result": "pass", "provenance": {"endpoint_executable_sha256": "a"}}
        off = copy.deepcopy(native)
        off["runtime"]["logical_context_tokens"] = 4096
        self.assertNotEqual(native["runtime"]["logical_context_tokens"], off["runtime"]["logical_context_tokens"])

    def test_repair85_selected_reference_timing_is_not_direct(self):
        case = summary._repair85_case({
            "result": "pass", "runtime": {"measured_request_tokens": 8, "logical_context_tokens": 16},
            "measurements": {"committed_tokens": 1, "wall_prefill_us": 10, "wall_decode_us": 10},
            "provenance": {},
        }, Path(__file__), "selected-reference")
        case["placements"]["target"]["route"] = "selected reference"
        self.assertEqual("measured", case["status"])
        self.assertNotEqual(case["placements"]["target"].get("route"), "selected direct")

    def test_repair85_absent_counters_are_not_zero(self):
        case = summary._repair85_case({"result": "pass", "runtime": {}, "measurements": {}, "provenance": {}}, Path(__file__), "native")
        self.assertIsNone(case["mtp"]["attempted"])
        self.assertIn("absent", case["mtp"]["reason"])

    def test_repair85_incomplete_request_is_not_success(self):
        case = summary._repair85_case({"result": "incomplete", "runtime": {}, "measurements": {"committed_tokens": 0}, "provenance": {}}, Path(__file__), "native")
        self.assertEqual("invalid", case["status"])
        self.assertIsNone(case["tokens"]["committed"])

    def test_repair85_stale_target_use_and_binary_mismatch_stay_null(self):
        chain = {"selected": {"forced_logical_page": 0, "forced_physical_slot": 1, "forced_page_generation": 2, "h2d_useful_bytes_delta": 1, "forced_checksum_equal": True, "mtp": {"transaction": True, "accepted": 1}, "natural_proof": {"target_graph_used": False}}}
        organic = {"cases": []}
        built = summary._repair85_promotion(chain, organic)
        self.assertTrue(built["controlled"]["completed_target_use"])
        left = [{"status": "measured", "question": 0, "fresh_pp": 1, "committed_tg": 1, "identity": {"binary_sha256": "a"}}]
        right = [{"status": "measured", "question": 0, "fresh_pp": 1, "committed_tg": 1, "identity": {"binary_sha256": "b"}}]
        self.assertIsNone(summary._repair85_ratio(left, right)["value"])


if __name__ == "__main__":
    unittest.main()
