#!/usr/bin/env python3
"""Portable tests for the shared evidence and resume contracts."""

from __future__ import annotations

import json
import importlib.util
import pathlib
import threading
import tempfile
import unittest
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

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
    resolve_batch_tokens,
    resolve_hot_capacity,
)


class ResumeContractTests(unittest.TestCase):
    def test_incremental_endpoint_disconnect_resume_and_frontier_guard(self) -> None:
        """Completed turns checkpoint once; resume sends only the missing turn."""
        path = HERE / "run-incremental-scale.py"
        spec = importlib.util.spec_from_file_location("run_incremental_scale_test", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class EndpointState:
            def __init__(self) -> None:
                self.frontier = 0
                self.generation = 1
                self.disconnect_turn = 2
                self.disconnect = True
                self.erase_calls = 0
                self.requests: list[int] = []

        endpoint_state = EndpointState()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                pass

            def _send(self, status: int, payload: object) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/metrics":
                    self._send(200, {})
                elif self.path == "/slots":
                    self._send(200, [{
                        "id": 0, "is_processing": False,
                        "n_prompt_tokens": endpoint_state.frontier,
                        "lifecycle": {"session_generation": endpoint_state.generation},
                    }])
                else:
                    self._send(404, {})

            def do_POST(self) -> None:  # noqa: N802
                if self.path.startswith("/slots/0"):
                    endpoint_state.frontier = 0
                    endpoint_state.generation += 1
                    endpoint_state.erase_calls += 1
                    self._send(200, {"ok": True})
                    return
                if self.path != "/v1/chat/completions":
                    self._send(404, {})
                    return
                size = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(size))
                messages = request["messages"]
                turn = (len(messages) - 1) // 2
                endpoint_state.requests.append(turn)
                prompt_tokens = sum(len(str(item["content"]).split()) + 4 for item in messages)
                if turn == endpoint_state.disconnect_turn and endpoint_state.disconnect:
                    endpoint_state.disconnect = False
                    self.close_connection = True
                    return
                cached = endpoint_state.frontier
                endpoint_state.frontier = prompt_tokens
                events = [
                    {"choices": [{"delta": {"content": f"answer-{turn}"}}]},
                    {"choices": [], "usage": {"prompt_tokens": prompt_tokens,
                                                   "completion_tokens": 1,
                                                   "prompt_tokens_details": {"cached_tokens": cached}},
                     "timings": {"prompt_ms": 1.0, "predicted_ms": 1.0,
                                 "prompt_per_second": 100.0, "predicted_per_second": 100.0}},
                ]
                body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
                body += b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        class Renderer:
            template_id = "fake-template"
            tokenizer_id = "fake-tokenizer"

            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def __call__(self, messages: list[dict[str, str]]) -> dict[str, object]:
                count = sum(len(str(item["content"]).split()) + 4 for item in messages)
                return SimpleNamespace(token_ids=tuple(range(count)))

        key_file = pathlib.Path(tempfile.mkdtemp()) / "key"
        key_file.write_text("test-key\n")

        def run(output: pathlib.Path, *session_flags: str) -> int:
            argv = ["run-incremental-scale.py", *session_flags,
                    "--output", str(output), "--endpoint",
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    "--api-key-file", str(key_file), "--context", "4096",
                    "--hot-pages", "16", "--target-tokens", "1536",
                    "--turn-delta", "512", "--max-tokens", "1"]
            with patch.object(module, "ServerPromptRenderer", Renderer), \
                    patch.object(module, "_runtime_identity", lambda *_args: {
                        "model": "fake-model", "model_path": "fake-model.gguf",
                        "binary": "fake-server", "loaded_dsos": [], "slot_id": 0,
                    }), patch.object(sys, "argv", argv):
                return module.main()

        try:
            interrupted = pathlib.Path(tempfile.mkdtemp())
            self.assertEqual(1, run(interrupted, "--new-session"))
            state = json.loads((interrupted / "incremental-state.json").read_text())
            self.assertEqual(2, state["frontier"]["next_turn_index"])
            self.assertEqual([0, 1, 2], endpoint_state.requests)
            self.assertEqual(1, endpoint_state.erase_calls)

            self.assertEqual(0, run(interrupted, "--resume"))
            resumed = json.loads((interrupted / "incremental-state.json").read_text())
            self.assertEqual(3, resumed["frontier"]["next_turn_index"])
            self.assertEqual([0, 1, 2, 2], endpoint_state.requests)
            self.assertEqual(1, endpoint_state.erase_calls)

            uninterrupted = pathlib.Path(tempfile.mkdtemp())
            endpoint_state.frontier = 0
            self.assertEqual(0, run(uninterrupted, "--new-session"))
            complete = json.loads((uninterrupted / "incremental-state.json").read_text())
            self.assertEqual(resumed["messages"], complete["messages"])
            self.assertEqual(
                [(item["turn"], item["assistant_text"], item["final_sse"])
                 for item in resumed["turns"]],
                [(item["turn"], item["assistant_text"], item["final_sse"])
                 for item in complete["turns"]])

            endpoint_state.frontier += 1
            requests_before = list(endpoint_state.requests)
            with self.assertRaises(module.ResumeStateError):
                run(interrupted, "--resume")
            self.assertEqual(requests_before, endpoint_state.requests)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_hot_capacity_uses_resolved_page_size_and_keeps_auto_typed(self) -> None:
        resolved = resolve_hot_capacity(73216, 256, 286)
        self.assertEqual(286, resolved["hot_capacity_pages"])
        self.assertEqual(73216, resolved["hot_capacity_tokens"])
        automatic = resolve_hot_capacity(8192, 128, "auto")
        self.assertTrue(automatic["automatic"])
        self.assertIsNone(automatic["hot_capacity_pages"])
        with self.assertRaises(ValueError):
            resolve_hot_capacity(8192, 256, 33)

    def test_batch_contract_rejects_invalid_widths(self) -> None:
        self.assertEqual({"batch_tokens": 128, "ubatch_tokens": 64},
                         resolve_batch_tokens(128, 64))
        with self.assertRaises(ValueError):
            resolve_batch_tokens(63, 64)

    def test_long_prompt_fitter_expands_neutral_padding(self) -> None:
        path = HERE / "run-final-curve.py"
        spec = importlib.util.spec_from_file_location("run_final_curve_test", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Renderer:
            template_id = "test-template"
            tokenizer_id = "test-tokenizer"

            def __call__(self, messages: list[dict[str, str]]) -> dict[str, object]:
                text = messages[0]["content"]
                return {"text": text, "token_ids": tuple(range(text.count("This is neutral"))),
                        "template_id": self.template_id, "tokenizer_id": self.tokenizer_id}

        target = 12_000
        fit = module._fit_prompt(Renderer(), "keep this final question", target, 8)
        self.assertEqual(target, fit.token_count)
        self.assertIn("keep this final question", fit.rendered_text)

    def test_sse_error_is_retained_as_runtime_error(self) -> None:
        path = HERE / "run-final-curve.py"
        spec = importlib.util.spec_from_file_location("run_final_curve_test", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            "SSE error [500]: Context size has been exceeded.",
            module._stream_error({"error": {"code": 500, "message": "Context size has been exceeded."}}),
        )
        self.assertIsNone(module._stream_error({"choices": []}))

    def test_request_sends_fitted_messages_without_rerendering(self) -> None:
        path = HERE / "run-final-curve.py"
        spec = importlib.util.spec_from_file_location("run_final_curve_test", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        captured: dict[str, object] = {}

        def stream(_endpoint: str, _key: str, request_body: dict[str, object], *_args: object,
                   **_kwargs: object) -> tuple[int, dict[str, object], None]:
            captured.update(request_body)
            return 200, {"usage": {"completion_tokens": 1}, "timings": {},
                         "stream_metrics": {}}, None

        module._stream_completion = stream
        module.snapshot = lambda _endpoint, _key: {"metrics": {}}
        messages = [{"role": "user", "content": "already fitted"}]
        result = module.run_request(
            "http://server/v1/chat/completions", "", "model", messages, 8,
            128, "test", 0, 1, 12, 1.0, pathlib.Path(tempfile.gettempdir()) / "run-final-curve-test.sse")
        self.assertEqual(messages, captured["messages"])
        self.assertEqual("pass", result["status"])
        captured.clear()
        module._stream_completion = lambda *_args, **_kwargs: (
            200, {"usage": {}, "timings": {}, "stream_metrics": {}}, None)
        failed = module.run_request(
            "http://server/v1/chat/completions", "", "model", messages, 8,
            128, "test", 0, 1, 12, 1.0, pathlib.Path(tempfile.gettempdir()) / "run-final-curve-empty.sse")
        self.assertEqual("runtime_fault", failed["status"])

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
