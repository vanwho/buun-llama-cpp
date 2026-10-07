#!/usr/bin/env python3
"""Synthetic contract tests for the task-specific 105-03 findings checker."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import check_105_03_findings as checker


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FindingsCheckerTests(unittest.TestCase):
    def make_report(self, root: Path, *, answer: str, goal_status: str = "goal_miss") -> dict:
        dsos = {
            "/build/libllama-server-impl.so": "1" * 64,
            "/build/libllama.so.0": "2" * 64,
            "/build/libggml-cuda.so.0": "3" * 64,
        }
        command = ("llama-server -m /models/model.gguf --alias test-model -c 262144 -np 1 "
                   "-b 1024 -ub 256 -ctk turbo4 -ctv turbo4 --kv-pager selective "
                   "--kv-router probe-rerank --kv-page-size 256 --kv-hot-pages 200 "
                   "--no-context-shift --ctx-checkpoints 0 --device CUDA0 "
                   "--spec-draft-kv-device gpu --spec-type draft-mtp --spec-draft-n-max 2 "
                   "--spec-draft-type-k turbo4 --spec-draft-type-v turbo4")
        identity = {
            "binary_sha256": "a" * 64, "model": "/models/model.gguf",
            "model_sha256": "b" * 64, "loaded_dso_sha256": dsos,
            "command": command, "context": "262144", "hot_pages": "200",
            "page_size_tokens": "256", "batch": "1024", "ubatch": "256",
            "target_kv_placement": "gpu", "mtp_placement": "gpu",
            "mtp_type_k": "turbo4", "mtp_type_v": "turbo4", "spec_draft_n_max": "2",
        }
        fingerprint = "f" * 64
        model_alias = "test-model"

        def save_json(name: str, value: object) -> dict[str, str]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(value, sort_keys=True).encode("utf-8")
            path.write_bytes(data)
            return {"path": str(path), "sha256": sha256(data)}

        def save_raw(name: str) -> dict[str, str]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = b"data: synthetic-http-200\n\n"
            path.write_bytes(data)
            return {"path": str(path), "sha256": sha256(data)}

        recall_question = "What factual behavior is stated?"
        schedule = [
            {"stage": "A1", "user": "A1 source payload"},
            {"stage": "B", "user": "B source payload"},
            {"stage": "repo_continuation_2", "user": "next source payload"},
            {"stage": "A2", "user": recall_question},
        ]
        records, history = [], []
        before = 0
        for index, item in enumerate(schedule):
            after = (index + 1) * 15000
            request = {"model": model_alias, "messages": [{"role": "user", "content": item["user"]}],
                       "max_tokens": 400, "n_ctx": 262144}
            request_ref = save_json(f"request-{index}.json", request)
            response_ref = save_raw(f"response-{index}.sse")
            fresh = after - before
            record = {
                "request_index": index, "stage": item["stage"], "status": "pass",
                "http_status": 200, "committed": True, "within_fresh_limit": True,
                "identity_fingerprint": fingerprint, "candidate_identity": identity,
                "request_path": request_ref["path"], "request_sha256": request_ref["sha256"],
                "raw_path": response_ref["path"], "raw_sha256": response_ref["sha256"],
                "artifacts": {"request": request_ref, "response": response_ref},
                "fresh_tokens": fresh, "cached_rows": 0,
                "frontier_delta_tokens": after - before,
                "response_prompt_tokens": fresh, "response_completion_tokens": 1,
                "timings": {"prompt_n": fresh, "prompt_ms": 10.0 + index,
                            "draft_n": 2, "draft_n_accepted": 1},
                "mtp": {"draft_tokens": 2, "accepted_tokens": 1},
                "answer_text": answer if item["stage"] == "A2" else "short response",
            }
            records.append(record)
            history.append({"request_index": index, "status": "pass",
                            "within_fresh_limit": True,
                            "occupied_before_tokens": before,
                            "occupied_after_tokens": after,
                            "fresh_tokens": fresh,
                            "frontier_delta_tokens": after - before})
            before = after

        base_messages = [{"role": "assistant", "content": "retained history"}]
        base_hash = sha256(json.dumps(base_messages, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode())
        probe_records = []
        for probe_index in range(12):
            prompt_index, repetition = divmod(probe_index, 4)
            prompt = checker.CANONICAL_PROMPTS[prompt_index]
            budget = 40 if repetition == 0 else 400
            request = {"model": model_alias,
                       "messages": base_messages + [{"role": "user", "content": prompt}],
                       "max_tokens": budget, "n_ctx": 262144}
            request_ref = save_json(f"canonical-request-{probe_index}.json", request)
            response_ref = save_raw(f"canonical-response-{probe_index}.sse")
            probe_records.append({
                "probe_index": probe_index, "prompt_index": prompt_index,
                "prompt": prompt, "repetition": repetition,
                "kind": "warmup" if repetition == 0 else "measured",
                "max_tokens": budget, "status": "pass", "http_status": 200,
                "actual_output_tokens": 2,
                "mtp_drafted_tokens": 0 if repetition == 0 else 3,
                "mtp_accepted_tokens": 0 if repetition == 0 else 1,
                "request_path": request_ref["path"], "request_sha256": request_ref["sha256"],
                "raw_path": response_ref["path"], "raw_sha256": response_ref["sha256"],
                "request_artifact": request_ref, "response_artifact": response_ref,
            })

        return {
            "campaign": "occupied-frontier-v1", "status": "incomplete",
            "execution_status": "complete", "goal_status": goal_status,
            "request_completed": False, "measurement_valid": True,
            "candidate_identity": identity, "identity_fingerprint": fingerprint,
            "geometry": {**checker.EXPECTED, "target_tokens": 250000,
                         "slot_id": 0, "completion_threshold_tokens": 249744},
            "effective_geometry": {"context": 262144, "hot_pages": 200,
                                   "page_size_tokens": 256, "batch": 1024, "ubatch": 256},
            "provenance": {"model_alias": model_alias},
            "repo_preflight": {
                "source_identity": {"commit": "source-id", "dirty_fingerprint": "c" * 64},
                "schedule": schedule,
                "recall_question": {"text": recall_question, "override": True,
                                    "sha256": sha256(recall_question.encode())},
                "requested_A2_prompt_frontier_tokens": 250000,
                "planner_A2_prompt_frontier_tokens": 250000,
                "projected_A2_prompt_tokens": 248000,
            },
            "records": records,
            "frontier": {"committed_tokens": before, "live_tokens": before,
                         "completion_threshold_tokens": 249744, "history": history},
            "slot": {"slot_id": 0, "generation": 4},
            "post_load_probes": {
                "complete": True, "identity_fingerprint": fingerprint,
                "base_messages": base_messages, "base_messages_sha256": base_hash,
                "completed": list(range(12)), "records": probe_records,
            },
            "final_snapshot": {"metrics": {}},
        }

    def validate(self, answer: str, goal_status: str = "goal_miss") -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.make_report(root, answer=answer, goal_status=goal_status)
            return checker.validate_report(report, root)

    def test_completed_short_eos_goal_miss_is_accepted(self) -> None:
        result = self.validate("A concise answer.")
        self.assertEqual("pass", result["status"])
        self.assertEqual("complete", result["execution_status"])
        self.assertEqual("goal_miss", result["goal_status"])
        self.assertEqual(60000, result["frontier_findings"]["achieved_C_tokens"])
        self.assertEqual(190000, result["frontier_findings"]["requested_C_target_gap_tokens"])
        self.assertEqual("unavailable", result["gpu_memory"]["status"])
        self.assertIsNone(result["gpu_memory"]["gpu_used_bytes"])

    def test_completed_semantic_miss_is_findings_not_a_gate(self) -> None:
        result = self.validate("A different recalled answer.")
        self.assertEqual("complete", result["execution_status"])
        self.assertEqual("goal_miss", result["goal_status"])
        self.assertFalse(result["factual_recall"]["fffe_marker_observed"])

    def test_rejects_incomplete_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.make_report(root, answer="short", goal_status="goal_miss")
            report["execution_status"] = "incomplete"
            with self.assertRaisesRegex(checker.FindingsError, "planned execution is incomplete"):
                checker.validate_report(report, root)

    def test_rejects_mixed_candidate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.make_report(root, answer="short", goal_status="goal_miss")
            mixed_identity = dict(report["records"][1]["candidate_identity"])
            mixed_identity["binary_sha256"] = "d" * 64
            report["records"][1]["candidate_identity"] = mixed_identity
            with self.assertRaisesRegex(checker.FindingsError, "candidate identity is mixed"):
                checker.validate_report(report, root)

    def test_rejects_missing_mandatory_occupancy_or_canonical_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.make_report(root, answer="short", goal_status="goal_miss")
            report["records"].pop()
            with self.assertRaisesRegex(checker.FindingsError, "mandatory occupancy request records"):
                checker.validate_report(report, root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.make_report(root, answer="short", goal_status="goal_miss")
            report["post_load_probes"]["records"].pop()
            with self.assertRaisesRegex(checker.FindingsError, "all 12 post-load canonical requests"):
                checker.validate_report(report, root)


if __name__ == "__main__":
    unittest.main()
