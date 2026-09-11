#!/usr/bin/env python3
"""Portable tests for the shared evidence and resume contracts."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pager_benchmark_contract import (
    CaseStateStore,
    CampaignDeadlines,
    ResumeError,
    case_key,
    classify_timeout,
    stream_metrics,
    validate_evidence,
    validate_speed_evidence,
)


class ResumeContractTests(unittest.TestCase):
    def test_case_key_includes_causal_inputs_but_not_runtime_status(self) -> None:
        base = {
            "bundle_manifest_sha256": "bundle-a", "model_sha256": "model-a",
            "tokenizer_template_sha256": "template-a", "corpus_sha256": "corpus-a",
            "config_sha256": "config-a", "prompt_hash": "prompt-a",
            "request_hash": "request-a", "source_release": "release-a",
            "mode": "selective", "context_tokens": 22016,
            "sampling": {"temperature": 0, "seed": 42},
            "cache_condition": "cold", "trial_index": 1,
            "status": "started", "attempt_id": "first",
        }
        original = case_key(base)
        base["status"] = "completed"
        base["attempt_id"] = "second"
        self.assertEqual(original, case_key(base))
        for field, value in (("request_hash", "request-b"),
                             ("cache_condition", "warm"), ("trial_index", 2),
                             ("source_release", "release-b")):
            changed = dict(base)
            changed[field] = value
            self.assertNotEqual(original, case_key(changed))

    def test_two_successes_then_interruption_resume_without_duplicate_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            campaign = {"source_release": "bundle-a", "corpus_sha256": "corpus-a"}
            store = CaseStateStore(root, campaign)
            raw = root / "raw.json"
            raw.write_text("raw\n")
            first = {"request_hash": "one", "trial_index": 1}
            second = {"request_hash": "two", "trial_index": 1}
            key_one, skipped = store.start(first)
            self.assertFalse(skipped)
            attempt_one = store.states[key_one]["attempt_id"]
            store.complete(key_one, attempt_one, success=True,
                           record={"id": "one", "status": "pass"}, raw_paths=[raw])
            key_two, skipped = store.start(second)
            self.assertFalse(skipped)
            attempt_two = store.states[key_two]["attempt_id"]
            store.interrupted(key_two, attempt_two, reason="operator_interrupt")

            resumed = CaseStateStore(root, campaign, resume=True)
            self.assertTrue(resumed.completed(key_one))
            self.assertFalse(resumed.completed(key_two))
            self.assertEqual(1, len(resumed.completed_records()))
            key_one_again, skipped = resumed.start(first)
            self.assertEqual(key_one, key_one_again)
            self.assertTrue(skipped)
            _, skipped = resumed.start(second)
            self.assertFalse(skipped)
            self.assertNotEqual(attempt_two, resumed.states[key_two]["attempt_id"])
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(1, progress["completed_successes"])

    def test_resume_rejects_changed_campaign_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            CaseStateStore(root, {"source_release": "release-a"})
            with self.assertRaises(ResumeError):
                CaseStateStore(root, {"source_release": "release-b"}, resume=True)

    def test_deadlines_timeout_classes_and_mixed_token_chunks(self) -> None:
        deadlines = CampaignDeadlines(total_seconds=900)
        self.assertEqual(900, deadlines.total_seconds)
        self.assertEqual("prefill_no_progress_timeout", classify_timeout("prefill"))
        self.assertEqual("decode_no_progress_timeout", classify_timeout("decode", progress_observed=True))
        metrics = stream_metrics(10.0, [
            {"timestamp": 10.25, "token_count": 2},
            {"timestamp": 10.50, "token_count": 1},
        ])
        self.assertEqual(0.25, round(metrics["ttft_us"] / 1_000_000, 2))
        self.assertEqual(3, metrics["completion_tokens"])
        self.assertEqual("sse_chunk_timestamps", metrics["timing_basis"])

    def test_existing_receipt_envelopes_validate(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[3] / ".wiretail" / "execution" / "evidence"
        for name in ("18-01_RUNTIME.json", "18-02_SIZING.json"):
            with self.subTest(name=name):
                receipt = json.loads((root / name).read_text())
                self.assertEqual([], validate_evidence(receipt))

    def test_speed_receipt_allows_optional_missing_stage_metrics(self) -> None:
        receipt = {
            "schema": "pager-speed-v6", "schema_version": 1, "task_id": "25-02",
            "experiment_id": "experiment", "procedure": "short", "started_utc": "start",
            "finished_utc": "finish", "result": "pass",
            "provenance": {field: "value" for field in (
                "source_commit", "source_diff_sha256", "bundle_identity",
                "bundle_manifest_sha256", "model_sha256", "tokenizer_template_sha256",
                "config_sha256", "gpu", "driver", "build")},
            "runtime": {
                "logical_context_tokens": 4096, "prompt_tokens": 512,
                "cached_rows": 0, "effective_batch": 256, "cuda_query_tile": 64,
                "cache_condition": "cold-prefill", "target_placement": "CUDA",
                "mtp_placement": "gpu", "target_type_k": "turbo4",
                "target_type_v": "turbo4", "mtp_type_k": "turbo4",
                "mtp_type_v": "turbo4", "allocated_context_rows": 4096,
                "hot_rows": 4096, "attended_rows": 768,
            },
            "measurements": {
                "generated_tokens": 16, "committed_tokens": 16,
                "mtp_proposed_tokens": 12, "mtp_accepted_tokens": 8,
                "target_gpu_bytes": 100, "host_committed_rows": 512,
                "host_committed_bytes": 200, "pinned_ring_bytes": 0,
                "wall_prefill_us": 1000, "wall_decode_us": 2000,
                "ttft_us": 1100, "completion_latency_us": 3000,
                "optional": {
                    "cuda_launch_count": {"value": None, "reason": "not enabled"},
                },
            },
            "raw_index": [{"id": "case", "sha256": "a" * 64}],
        }
        self.assertEqual([], validate_speed_evidence(receipt))
        receipt["runtime"]["target_type_k"] = "q8_0"
        self.assertIn("runtime.target_type_k_not_turbo4", validate_speed_evidence(receipt))
        receipt["runtime"]["target_type_k"] = "turbo4"
        receipt["runtime"]["target_placement"] = None
        receipt["runtime"]["target_placement_reason"] = "not measured"
        self.assertIn("runtime.target_placement", validate_speed_evidence(receipt))


if __name__ == "__main__":
    unittest.main()
