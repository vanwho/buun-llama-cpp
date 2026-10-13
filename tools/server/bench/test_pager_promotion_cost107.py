"""Focused deterministic contracts for the verified-cold107 protocol."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


HERE = pathlib.Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("promotion_cost107", HERE / "run-pager-promotion-cost.py")
assert SPEC is not None and SPEC.loader is not None
cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cost)
VSPEC = importlib.util.spec_from_file_location("validate_cost107", HERE / "validate-pager-promotion-cost.py")
assert VSPEC is not None and VSPEC.loader is not None
validator = importlib.util.module_from_spec(VSPEC)
VSPEC.loader.exec_module(validator)


class VerifiedCold107Tests(unittest.TestCase):
    def test_exact_geometry_argv_and_interior_source_page_boundaries(self) -> None:
        expected = [(8192, 4096, 16, 6144, 5120), (16384, 8192, 32, 14336, 9216),
                    (32768, 16384, 64, 30720, 17408), (65536, 32768, 128, 61440, 33792),
                    (77824, 65536, 256, 73728, 66560), (131072, 65536, 256, 126976, 66560),
                    (196608, 65536, 256, 192512, 66560), (262144, 65536, 256, 258048, 66560)]
        self.assertEqual(expected, [(g["L"], g["H"], g["hot_pages"], g["C"], g["a_min"])
                                    for g in cost.VERIFIED_COLD107_GEOMETRIES])
        for geometry in cost.VERIFIED_COLD107_GEOMETRIES:
            argv = cost.verified_cold107_argv(geometry)
            self.assertEqual(str(cost.VERIFIED_COLD107_SERVER), argv[0])
            self.assertEqual(str(geometry["L"]), argv[argv.index("-c") + 1])
            self.assertEqual(str(geometry["hot_pages"]), argv[argv.index("--kv-hot-pages") + 1])
            self.assertEqual("1024", argv[argv.index("-b") + 1])
            self.assertEqual("256", argv[argv.index("-ub") + 1])
            cost.validate_verified_cold107_geometry(geometry, 3000, 200)
        inv = [
            {"logical_page_id": 0, "position_begin": 0, "position_end": 256,
             "valid_length": 256, "resident": True, "host_backed": True, "summary_ready": True},
            {"logical_page_id": 1, "position_begin": 256, "position_end": 512,
             "valid_length": 256, "resident": False, "host_backed": True, "summary_ready": True},
            {"logical_page_id": 2, "position_begin": 512, "position_end": 768,
             "valid_length": 256, "resident": False, "host_backed": True, "summary_ready": True},
            {"logical_page_id": 3, "position_begin": 768, "position_end": 1024,
             "valid_length": 256, "resident": False, "host_backed": True, "summary_ready": False},
        ]
        snap = {"slots": [{"id": 0, "pager_metrics": {"page_inventory": inv}}]}
        self.assertEqual([2], [p["logical_page_id"] for p in cost.verified_cold_source_pages(
            snap, [{"token_begin": 512, "token_end": 768}])])
        self.assertEqual([], cost.verified_cold_source_pages(
            snap, [{"token_begin": 384, "token_end": 640}]))

    def test_separate_final_query_and_identical_cold_hot_request_boundary(self) -> None:
        base = [{"role": "user", "content": "frozen repository history"}]
        question = "Using PATH and SYMBOL, explain its behavior."
        messages = base + [{"role": "user", "content": question}]
        body = cost.DRIVER.body(cost.ALIAS, messages, 400, True, ignore_eos=False)
        self.assertEqual(base[0]["content"], body["messages"][0]["content"])
        self.assertEqual(question, body["messages"][-1]["content"])
        cold = {"status": "pass", "request_sha256": cost.canonical_hash(body), "base_sha256": "base",
                "ttft_ms": 40, "client_wall_ms": 100}
        hot = {**cold, "target_resident_before": True, "historical_h2d_pages": 0}
        self.assertTrue(cost.verified_cold107_pair_valid(
            cold, hot, same_request=True, same_base=True, target_promoted=True,
            no_historical_h2d=True))
        self.assertFalse(cost.verified_cold107_pair_valid(
            cold, hot, same_request=True, same_base=False, target_promoted=True,
            no_historical_h2d=True))

    def test_semantic_miss_or_mixed_hot_control_cannot_create_penalty_or_pass_report(self) -> None:
        cold = {"status": "completed", "ttft_ms": 40, "client_wall_ms": 100}
        hot = {"status": "completed", "ttft_ms": 35, "client_wall_ms": 90}
        self.assertIsNone(cost.verified_cold_pair_penalty(cold, hot, False, True)["ttft_ms"])
        self.assertIsNone(cost.verified_cold_pair_penalty(cold, hot, True, False)["wall_ms"])
        report = {"schema": "aco107-promotion-cost-v1", "task": "107-01",
                  "geometries": [{"name": f"row-{n}", "status": "filled",
                                  "cold_requests": [{"http_request": False,
                                                     "pre_request_target": {"host_backed_cold": False}}],
                                  "pairs": []} for n in range(8)]}
        self.assertTrue(any("cold-preconditioned HTTP request" in error
                            for error in validator.validate_report(report)))
        captured_fault = {"http_request": True, "status": "runtime_fault",
            "request_path": "/raw/request.json", "request_sha256": "request-hash",
            "raw_path": "/raw/response.sse", "raw_sha256": "response-hash",
            "error": "SSE error [500]: Unable to commit selected KV history",
            "ttft_ms": None, "pre_request_target": {"host_backed_cold": True}}
        self.assertTrue(validator._cold_http_request_evidence(captured_fault))
        self.assertFalse(validator._cold_http_request_evidence(
            {**captured_fault, "raw_path": None}))
        fault_report = {"schema": "aco107-promotion-cost-v1", "task": "107-01",
            "geometries": [{"name": f"row-{n}", "status": "filled",
                "cold_requests": [captured_fault], "pairs": []} for n in range(8)]}
        self.assertEqual([], validator.validate_report(fault_report))
        cold_miss = {"status": "filled", "verified_cold_request_count": 3,
            "cold_requests": [{"http_request": True, "status": "pass",
                "pre_request_target": {"host_backed_cold": True}} for _ in range(3)],
            "physical_target_promotions": 0, "valid_pair_count": 0}
        self.assertTrue(cost._resumable_verified_row(cold_miss))
        self.assertFalse(cost._resumable_verified_row({**cold_miss,
            "cold_requests": cold_miss["cold_requests"][:2]}))
        source = cost.repo_context.CorpusChunk("a.py", "digest", 20, 1, 1,
                                                "source", 0, 20)
        scheduled_row = {"A_fill_curve": [{"source_ranges": [{"path": "a.py",
            "sha256": "digest", "start_byte": 0, "end_byte": 6}]}],
            "B_fill_curve": [{"source_ranges": [{"path": "b.md",
            "sha256": "other", "start_byte": 0, "end_byte": 3}]}]}
        other = cost.repo_context.CorpusChunk("b.md", "other", 20, 1, 1,
                                               "doc", 0, 20)
        self.assertTrue(cost._row_source_ranges_match_schedule(
            scheduled_row, [source], [other]))
        self.assertFalse(cost._row_source_ranges_match_schedule(
            scheduled_row, [source], []))


if __name__ == "__main__":
    unittest.main()
