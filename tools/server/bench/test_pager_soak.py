#!/usr/bin/env python3
"""Portable context-boundary tests for the lifecycle soak harness."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("run_pager_soak", HERE / "run-pager-soak.py")
assert SPEC and SPEC.loader
soak = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(soak)


class PagerSoakContextTests(unittest.TestCase):
    def test_response_oracle_rejects_http_200_server_error(self) -> None:
        response = {"choices": [{"message": {"content": "Error: No valid data found."}}]}
        result = soak.classify_completion_response(200, None, response, "MARKER_A")
        self.assertEqual("ok", result["transport_status"])
        self.assertEqual("server_semantic_error", result["semantic_status"])
        self.assertEqual("not_evaluated", result["marker_status"])

    def test_response_oracle_requires_exact_marker(self) -> None:
        response = {"choices": [{"message": {"content": "MARKER_B\n"}}]}
        result = soak.classify_completion_response(200, None, response, "MARKER_A")
        self.assertEqual("ok", result["semantic_status"])
        self.assertEqual("mismatch", result["marker_status"])

    def test_response_oracle_preserves_http_failure(self) -> None:
        result = soak.classify_completion_response(501, "HTTPError:501", {
            "error": {"type": "not_supported_error"}}, "MARKER_A")
        self.assertEqual("error", result["transport_status"])
        self.assertEqual("not_evaluated", result["semantic_status"])
        self.assertEqual("HTTPError:501", result["error"])

    def test_marker_grammar_is_single_literal(self) -> None:
        self.assertEqual('root ::= "MARKER_A"', soak.marker_grammar("MARKER_A"))

    def test_concurrent_oracle_reports_slot_and_semantic_blockers(self) -> None:
        results = [
            {"label": "concurrent-a", "slot_id": None,
             "transport_status": "ok", "semantic_status": "server_semantic_error",
             "marker_status": "not_evaluated"},
            {"label": "concurrent-b", "slot_id": None,
             "transport_status": "error", "semantic_status": "not_evaluated",
             "marker_status": "not_evaluated"},
        ]
        oracle = soak.build_concurrent_oracle([0], results)
        self.assertEqual("blocked", oracle["status"])
        self.assertIn("requires_at_least_two_independent_slots", oracle["blockers"])
        self.assertIn("concurrent-a:semantic_server_semantic_error", oracle["blockers"])
        self.assertIn("concurrent-b:transport_failed", oracle["blockers"])

    def test_concurrent_oracle_passes_only_independent_exact_results(self) -> None:
        results = [
            {"label": "concurrent-a", "slot_id": 0,
             "transport_status": "ok", "semantic_status": "ok", "marker_status": "exact"},
            {"label": "concurrent-b", "slot_id": 1,
             "transport_status": "ok", "semantic_status": "ok", "marker_status": "exact"},
        ]
        oracle = soak.build_concurrent_oracle([0, 1], results)
        self.assertEqual("passed", oracle["status"])
        self.assertEqual([], oracle["blockers"])

    def test_startup_probe_classifies_unhealthy_service_before_soak(self) -> None:
        self.assertEqual("ready", soak.classify_startup_probe(
            {"8080": True, "8091": True}, 0))
        self.assertEqual("ready", soak.classify_startup_probe(
            {"8080": True}, 0))
        self.assertEqual("runtime_crash_or_unavailable", soak.classify_startup_probe(
            {"8080": True}, 0, identity_stable=False))
        self.assertEqual("runtime_crash_or_unavailable", soak.classify_startup_probe(
            {"8080": False, "8091": False}, 0))
        self.assertEqual("restart_failed", soak.classify_startup_probe(
            {"8080": False, "8091": False}, 1))

    def test_dry_run_uses_corpus_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(soak.sys, "argv", [
                "run-pager-soak.py", directory, "--dry-run"]):
            self.assertEqual(0, soak.main())
            summary = json.loads((pathlib.Path(directory) / "run-summary.json").read_text())
            self.assertEqual(22016, summary["context"]["resolved"])
            self.assertEqual("acceptance", summary["context"]["mode"])
            self.assertEqual("not_run_dry_run", summary["prompt"]["token_sizing"])
            self.assertIsNone(summary["prompt"]["occupied_prompt_tokens"])

    def test_sub_ceiling_soak_requires_diagnostic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(soak.sys, "argv", [
                "run-pager-soak.py", directory, "--context", "16384"]):
            self.assertEqual(2, soak.main())

    def test_diagnostic_odd_tail_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(soak.sys, "argv", [
                "run-pager-soak.py", directory, "--dry-run", "--context", "6401",
                "--diagnostic"]):
            self.assertEqual(0, soak.main())
            summary = json.loads((pathlib.Path(directory) / "run-summary.json").read_text())
            self.assertTrue(summary["context"]["diagnostic_only"])
            self.assertEqual(1, summary["prompt"]["tail_tokens"])
            self.assertEqual("turbo4", summary["placement"]["draft_kv"])
            self.assertEqual("gpu", summary["placement"]["draft_backend"])


if __name__ == "__main__":
    unittest.main()
