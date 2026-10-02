#!/usr/bin/env python3
"""Deterministic resume and evidence tests for occupied-frontier campaigns."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SPEC = importlib.util.spec_from_file_location(
    "run_occupancy_frontier_test", HERE / "run-occupancy-frontier.py")
assert SPEC and SPEC.loader
occupancy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(occupancy)

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_occupied_frontier_test", HERE / "validate-occupied-frontier.py")
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(validator)

REPO_BASELINE_SPEC = importlib.util.spec_from_file_location(
    "run_repo_context_baseline_test", HERE / "run-repo-context-baseline.py")
assert REPO_BASELINE_SPEC and REPO_BASELINE_SPEC.loader
repo_baseline = importlib.util.module_from_spec(REPO_BASELINE_SPEC)
REPO_BASELINE_SPEC.loader.exec_module(repo_baseline)


class FakeRenderer:
    template_id = "fake-template-v1"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __call__(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        count = sum(len(str(item["content"]).split()) + 4 for item in messages)
        return SimpleNamespace(token_ids=tuple(range(count)))


class FakeRuntime:
    def __init__(self) -> None:
        self.frontier = 0
        self.generation = 1
        self.clear_count = 0
        self.request_indices: list[int] = []
        self.endpoint_calls = 0

    def snapshot(self, _endpoint: str, _key: str) -> dict[str, object]:
        self.endpoint_calls += 1
        host_rows = max(0, self.frontier - 16384)
        target_rows = min(self.frontier, 16384)
        return {
            "metrics": {
                "context_tokens": 32768,
                "page_tokens": 256,
                "page_capacity": 64,
                "target_type_k": "turbo4",
                "target_type_v": "turbo4",
                "mtp_backend": "gpu",
                "mtp_rows": 32768,
                "target_allocated_bytes": 12345678,
                "physical_pool_capacity_bytes": 6543210,
                "target_resident_bytes": target_rows * 4,
                "target_valid_bytes": target_rows * 4,
                "target_valid_rows": target_rows,
                "host_valid_rows": host_rows,
                "host_valid_bytes": host_rows * 4,
                "host_pageable_bytes": host_rows * 4,
                "host_pinned_bytes": 4096,
                "host_committed_bytes": host_rows * 4 + 4096,
                "scratch_high_water_bytes": 2048,
                "live_allocation_peak_bytes": 23456789,
                "mtp_allocated_bytes": 32768 * 4,
            },
            "slots": [{"id": 0, "is_processing": False,
                       "n_prompt_tokens": self.frontier,
                       "lifecycle": {"session_generation": self.generation}}],
        }

    def clear_slot(self, _endpoint: str, _key: str, slot_id: int,
                   _timeout: float) -> dict[str, object]:
        self.endpoint_calls += 1
        self.frontier = 0
        self.generation += 1
        self.clear_count += 1
        return {"slot_id": slot_id, "ok": True, "http_status": 200}

    def run_request(self, _endpoint: str, _key: str, _model: str,
                    messages: list[dict[str, str]], _maximum: int,
                    _context: int, _phase: str, request_index: int, _trial: int,
                    prompt_tokens: int, _timeout: float, raw_path: pathlib.Path,
                    **_kwargs: object) -> dict[str, object]:
        self.endpoint_calls += 1
        self.request_indices.append(request_index)
        self.frontier = prompt_tokens
        raw_path.write_text(f"data: fake-response-{request_index}\n", encoding="utf-8")
        return {
            "status": "pass",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1},
            "timings": {"prompt_ms": 1, "predicted_ms": 1},
            "response": {"choices": [{"message": {"content": f"answer-{request_index}"}}]},
            "cached_rows": max(0, self.frontier - prompt_tokens),
        }


def identity(binary_digest: str = "a" * 64) -> dict[str, object]:
    return {
        "main_pid": 321, "pid": 321, "process_start_time_ticks": "87654321",
        "exe": "/fake/llama-server", "binary_sha256": binary_digest,
        "loaded_dso_sha256": {"/fake/libggml-cuda.so": "b" * 64},
        "model": "/fake/model.gguf", "model_sha256": "c" * 64,
        "command": "llama-server -c 32768 -b 1024 -ub 256",
        "context": "32768", "hot_pages": "64", "page_size_tokens": "256",
        "batch": "1024", "ubatch": "256", "target_kv_placement": "gpu",
        "mtp_placement": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
        "spec_draft_n_max": "2",
    }


class OccupancyFrontierTests(unittest.TestCase):
    def test_repo_baseline_requires_real_host_backed_pages(self) -> None:
        geometry = repo_baseline.repo_context.load_manifest()["runtime_requirements"]
        self.assertEqual(16384, geometry["logical_context_tokens"])
        self.assertEqual(8192, geometry["selected_hot_tokens"])
        self.assertEqual(256, geometry["page_tokens"])
        passing = {"slots": [{"id": 0, "is_processing": False,
                              "n_prompt_tokens": 12288,
                              "lifecycle": {"session_generation": 1},
                              "pager_metrics": {
                                  "host_valid_rows": 4096,
                                  "host_valid_bytes": 16384,
                                  "page_inventory": [
                                      {"logical_page_id": 0, "resident": True,
                                       "host_backed": True},
                                      {"logical_page_id": 1, "resident": False,
                                       "host_backed": True},
                                  ]}}]}
        proof = repo_baseline.host_offload_evidence(passing)
        self.assertEqual(4096, proof["host_valid_rows"])
        self.assertEqual(16384, proof["host_valid_bytes"])
        self.assertEqual([1], proof["cold_host_backed_page_ids"])

        no_host_bytes = {"slots": [{"id": 0, "is_processing": False,
                                    "n_prompt_tokens": 12288,
                                    "lifecycle": {"session_generation": 1},
                                    "pager_metrics": {
                                        "host_valid_rows": 0,
                                        "host_valid_bytes": 0,
                                        "page_inventory": []}}]}
        with self.assertRaisesRegex(RuntimeError, "did not prove real host offload"):
            repo_baseline.host_offload_evidence(no_host_bytes)

    def test_slot_task_id_is_a_generation_when_server_omits_lifecycle_generation(self) -> None:
        slot = occupancy.selected_slot({"slots": [{"id": 0, "is_processing": False,
                                                    "id_task": 23,
                                                    "n_prompt_tokens": 4258}]}, 0)
        self.assertEqual(23, slot["generation"])
        self.assertEqual(4258, slot["occupied_tokens"])

    def test_repo_context_manifest_identity_and_a_b_a_reserve_and_frontier(self) -> None:
        manifest = occupancy.repo_context.load_manifest()
        prompts = occupancy.repo_context.load_prompts()
        self.assertIn("`--stdin` wins", prompts["A1"])
        self.assertIn("N_KV", prompts["B"])
        self.assertIn("Windows console", prompts["A2"])
        prompts = {"A1": "A1_QUERY", "B": "B_QUERY", "A2": "A2_QUERY"}
        turns = occupancy.build_repo_content_turns(manifest, prompts)
        self.assertEqual(3, len(turns))
        self.assertIn("A1_QUERY", turns[0]["content"])
        self.assertIn("B_QUERY", turns[1]["content"])
        self.assertEqual("A2_QUERY", turns[2]["content"])
        self.assertNotIn("BEGIN FILE:", turns[2]["content"])
        self.assertNotIn("tools/tokenize/tokenize.cpp", turns[2]["content"])

        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source.txt"
            source.write_text("original\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            # Exercise the same hash refusal path with a deliberately stale identity.
            old_root = occupancy.repo_context.ROOT
            try:
                occupancy.repo_context.ROOT = pathlib.Path(directory)
                occupancy.repo_context._read_repo_file("source.txt", digest)
                with self.assertRaises(occupancy.repo_context.SourceIdentityError):
                    occupancy.repo_context._read_repo_file("source.txt", "0" * 64)
            finally:
                occupancy.repo_context.ROOT = old_root

        class CountingRenderer:
            def __call__(self, messages: list[dict[str, str]]) -> SimpleNamespace:
                count = sum(len(message["content"].split()) + 4 for message in messages)
                return SimpleNamespace(token_ids=tuple(range(count)))

        renderer = CountingRenderer()
        corpus = [
            occupancy.repo_context.CorpusChunk(
                "src/a.cpp", "a" * 64, 8, 1, 1, "int a;\n"),
            occupancy.repo_context.CorpusChunk(
                "tools/b.md", "b" * 64, 8, 1, 1, "word " * 3000),
        ]
        prefix = [{"role": "assistant", "content": "prior answer"}]
        text, tokens, selected = occupancy.repo_context.append_chunks_to_frontier(
            renderer, prefix, "B_QUERY", corpus, context_tokens=8192,
            reserve_tokens=800, max_fresh_tokens=100)
        self.assertIn("src/a.cpp", text)
        self.assertEqual(["src/a.cpp"], [item.path for item in selected])
        self.assertEqual((1, 1), (selected[0].start_line, selected[0].end_line))
        restored = occupancy.repo_context.restore_chunks(
            [occupancy.repo_context.SourceFile("src/a.cpp", "a" * 64, 8, "int a;\n")],
            occupancy.repo_context.selection_record(selected))
        self.assertEqual(selected, restored)
        self.assertEqual(tokens, len(renderer(prefix + [{"role": "user", "content": text}]).token_ids))
        occupancy.repo_context.enforce_reserve(tokens, 400, 800, 8192)
        with self.assertRaises(ValueError):
            occupancy.repo_context.enforce_reserve(7900, 400, 800, 8192)

    def test_context_limit_validation_precedes_runtime_or_endpoint_access(self) -> None:
        for context_tokens in (8192, 32768, 131072, 262144):
            with self.subTest(context_tokens=context_tokens):
                occupancy.validate_requested_geometry(
                    context_tokens, 4096, 256, 1000, 16)

        runtime = FakeRuntime()
        invalid_geometries = (
            (262145, 4096, 256, 1000),  # above the named context ceiling
            (8192, 8192, 256, 1000),    # H must be smaller than L
            (8193, 4096, 256, 1000),    # L must be page aligned
            (8192, 4097, 256, 1000),    # H must be page aligned
            (8192, 4096, 256, 8192),    # target must leave generation room
        )
        for context_tokens, hot_tokens, page_tokens, target_tokens in invalid_geometries:
            with self.subTest(context_tokens=context_tokens, hot_tokens=hot_tokens,
                              target_tokens=target_tokens):
                argv = [
                    "run-occupancy-frontier.py", "--output", "unused-output",
                    "--api-key-file", "unused-key", "--target-tokens", str(target_tokens),
                    "--context-tokens", str(context_tokens), "--hot-tokens", str(hot_tokens),
                    "--page-tokens", str(page_tokens),
                ]
                with patch.object(occupancy, "load_driver", return_value=runtime) as load_driver, \
                        patch.object(occupancy.sys, "argv", argv):
                    with self.assertRaises(SystemExit):
                        occupancy.main()
                load_driver.assert_not_called()

        self.assertEqual(runtime.endpoint_calls, 0)

    def test_committed_interrupt_resumes_with_bound_identity_and_unique_artifacts(self) -> None:
        runtime = FakeRuntime()
        current_identity = identity()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            key = root / "api-key"
            key.write_text("fake-key\n", encoding="utf-8")
            output = root / "campaign"

            def run(extra: list[str] | None = None) -> int:
                argv = [
                    "run-occupancy-frontier.py", "--output", str(output),
                    "--api-key-file", str(key), "--target-tokens", "20000",
                    "--initial-tokens", "1200", "--turn-delta", "10000",
                    "--max-tokens", "1",
                ] + (extra or [])
                with patch.object(occupancy, "load_driver", lambda: runtime), \
                        patch.object(occupancy, "capture_runtime_identity",
                                     lambda _driver: current_identity), \
                        patch.object(occupancy, "ServerPromptRenderer", FakeRenderer), \
                        patch.object(occupancy.sys, "argv", argv), \
                        contextlib.redirect_stdout(io.StringIO()):
                    return occupancy.main()

            self.assertEqual(1, run(["--max-requests", "1"]))
            partial = json.loads((output / "occupied-frontier.json").read_text())
            self.assertFalse(partial["request_completed"])
            partial_proof = validator.validate_report(partial)
            self.assertIn("requested occupancy frontier is incomplete",
                          partial_proof["errors"])
            saved = json.loads((output / "incremental-state.json").read_text())
            self.assertEqual(1, saved["next_request_index"])
            self.assertEqual(1, len(saved["history"]))

            self.assertEqual(0, run(["--resume-state", str(output)]))
            complete = json.loads((output / "occupied-frontier.json").read_text())
            proof = validator.validate_report(complete)
            self.assertEqual("pass", proof["status"], proof["errors"])
            self.assertGreater(complete["frontier"]["committed_tokens"], 16384)
            indices = [row["request_index"] for row in complete["records"]]
            self.assertEqual(list(range(len(indices))), indices)
            self.assertEqual(indices, runtime.request_indices)
            self.assertEqual(1, runtime.clear_count)
            journal = [json.loads(line) for line in
                       (output / "request-journal.jsonl").read_text().splitlines()]
            self.assertEqual(indices, [row["request_index"] for row in journal])

            saved = json.loads((output / "incremental-state.json").read_text())
            slot = occupancy.selected_slot(runtime.snapshot("", ""), 0)
            bad_geometry = dict(saved["geometry"], hot_capacity_tokens=8192,
                                hot_capacity_pages=32)
            with self.assertRaises(occupancy.ResumeStateError):
                occupancy.validate_resume_state(saved, current_identity, bad_geometry, slot)

            original_identity = current_identity.copy()
            current_identity = identity("d" * 64)
            with self.assertRaises(occupancy.ResumeStateError):
                run(["--resume-state", str(output)])
            current_identity = original_identity


if __name__ == "__main__":
    unittest.main()
