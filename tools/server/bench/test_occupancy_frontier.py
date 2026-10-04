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
import subprocess
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
        self.generation += 1
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
    def test_a2_replay_fresh_tokens_use_uncached_prompt_tail(self) -> None:
        usage = {"prompt_tokens_details": {"cached_tokens": 250033}}
        self.assertEqual(140, occupancy._fresh_prompt_tokens(250173, 250436, usage))
        self.assertEqual(11892, occupancy._fresh_prompt_tokens(16181, 4289, {}))

    def test_post_load_canonical_reuses_immutable_prefix_and_mtp_deltas(self) -> None:
        class ProbeRuntime:
            def __init__(self) -> None:
                self.calls: list[tuple[list[dict[str, str]], int]] = []
                self.frontier = 20000
                self.generation = 9
                self.clear_count = 0

            def snapshot(self, _endpoint: str, _key: str) -> dict[str, object]:
                return {"slots": [{"id": 0, "n_prompt_tokens": self.frontier,
                                   "lifecycle": {"session_generation": self.generation}}]}

            def run_request(self, _endpoint: str, _key: str, _model: str,
                            messages: list[dict[str, str]], maximum: int, *args: object,
                            **_kwargs: object) -> dict[str, object]:
                index = len(self.calls)
                raw_path = pathlib.Path(args[6])
                self.calls.append((json.loads(json.dumps(messages)), maximum))
                self.generation += 1
                raw_path.write_text(f"data: probe-{index}\n", encoding="utf-8")
                drafted = 10 + index
                accepted = 2 + index % 5
                return {
                    "status": "pass", "usage": {"completion_tokens": maximum - 1},
                    "output_tokens": maximum - 1,
                    "response": {"choices": [{"message": {"content": f"answer-{index}"}}]},
                    "timings": {"predicted_ms": index + 1},
                    "mtp": {"draft_tokens": drafted, "accepted_tokens": accepted},
                }

        runtime = ProbeRuntime()
        original_base = [{"role": "user", "content": "occupied history"},
                         {"role": "assistant", "content": "retained answer"}]
        base_copy = json.loads(json.dumps(original_base))
        current_identity = identity()
        fingerprint = "f" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            state: dict[str, object] = {}
            pre_snapshot = runtime.snapshot("", "")
            pre_slot = {"slot_id": 0, "generation": 9, "occupied_tokens": 20000}
            result = occupancy.run_post_load_canonical(
                runtime, "", "", "fake", FakeRenderer(), original_base, root,
                state, current_identity, fingerprint, 32768, 0, pre_snapshot,
                pre_slot, prefill_timeout=1, total_timeout=2)

            self.assertEqual(12, len(runtime.calls))
            self.assertEqual([40, 400, 400, 400] * 3,
                             [budget for _messages, budget in runtime.calls])
            self.assertEqual(base_copy, original_base)
            self.assertEqual(0, runtime.clear_count)
            prefixes = [messages[:-1] for messages, _budget in runtime.calls]
            self.assertTrue(all(prefix == base_copy for prefix in prefixes))
            self.assertEqual(12, len({row["mtp_drafted_tokens"] for row in result["records"]}))
            self.assertEqual(3, len(result["prompt_summaries"]))
            self.assertTrue(all(row["median_acceptance_percent"] is not None
                                for row in result["prompt_summaries"]))
            self.assertEqual(list(range(12)), result["completed"])

            # A resumed helper invocation skips every checkpointed probe.
            occupancy.run_post_load_canonical(
                runtime, "", "", "fake", FakeRenderer(), original_base, root,
                state, current_identity, fingerprint, 32768, 0, pre_snapshot,
                pre_slot, prefill_timeout=1, total_timeout=2)
            self.assertEqual(12, len(runtime.calls))

    def test_repo_identity_ignores_wiretail_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Bench Test"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "bench@example.invalid"], check=True)
            source = root / "src" / "sample.cpp"
            state = root / ".wiretail" / "execution" / "WORK_STATE.json"
            source.parent.mkdir(parents=True)
            state.parent.mkdir(parents=True)
            source.write_text("int value = 1;\n", encoding="utf-8")
            state.write_text("{}\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)

            baseline = repo_baseline.repo_context.git_identity(root)
            state.write_text('{"status":"in_progress"}\n', encoding="utf-8")
            handoff = root / ".wiretail" / "execution" / "handoffs" / "102-04.md"
            handoff.parent.mkdir(parents=True)
            handoff.write_text("checkpoint\n", encoding="utf-8")
            self.assertEqual(baseline, repo_baseline.repo_context.git_identity(root))

            source.write_text("int value = 2;\n", encoding="utf-8")
            self.assertNotEqual(baseline, repo_baseline.repo_context.git_identity(root))

    def test_repo_baseline_records_runtime_fault_and_skips_only_dependents(self) -> None:
        completed = [{"request_id": "A1", "status": "pass"}]
        failed_record = {"request_id": "B", "status": "runtime_fault",
                         "error": "no_victim"}
        failure = repo_baseline.RecordedRequestFailure("B", failed_record, "B failed: no_victim")
        result = repo_baseline.failed_sequence_result(
            "warmup", "selected", failure, completed)
        self.assertEqual("runtime_fault", result["status"])
        self.assertEqual(completed, result["completed_records"])
        self.assertEqual(failed_record, result["failed_record"])
        self.assertEqual(["A2"], result["dependent_requests_skipped"])

    def test_repo_request_rejects_reported_prompt_tokens_below_rendered_and_saves_result(self) -> None:
        class SmallRenderer:
            def __call__(self, messages: list[dict[str, str]]) -> SimpleNamespace:
                return SimpleNamespace(token_ids=tuple(range(9)))

        class TruncatedRuntime:
            def run_request(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                return {
                    "status": "pass", "usage": {"prompt_tokens": 2,
                                                  "completion_tokens": 1,
                                                  "total_tokens": 3},
                    "after": {"slots": [{"id": 0, "n_prompt_tokens": 3,
                                          "lifecycle": {"session_generation": 1}}]},
                    "response": {"choices": [{"message": {"content": "answer"}}]},
                }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(repo_baseline, "capture_runtime_identity", return_value=identity()), \
                    patch.object(repo_baseline, "identity_fingerprint", return_value="fingerprint"):
                with self.assertRaisesRegex(RuntimeError, "rendered prompt 9 tokens"):
                    repo_baseline.record_request(
                        TruncatedRuntime(), SmallRenderer(), "endpoint", "key", "model",
                        [{"role": "user", "content": "prompt"}], pathlib.Path(directory),
                        identity(), "fingerprint", "A1", "A1", "gpu", 0, True)
            saved = json.loads((pathlib.Path(directory) / "requests/A1.result.json").read_text())
            self.assertEqual(9, saved["rendered_prompt_tokens"])
            self.assertEqual("fail", saved["prompt_token_validation"]["status"])

    def test_repo_frontier_counts_base_user_against_fixed_request_entry_limit(self) -> None:
        class CountingRenderer:
            def __call__(self, messages: list[dict[str, str]]) -> SimpleNamespace:
                count = sum(len(message["content"].split()) + 4 for message in messages)
                return SimpleNamespace(token_ids=tuple(range(count)))

        renderer = CountingRenderer()
        corpus = [repo_baseline.repo_context.CorpusChunk(
            "src/a.cpp", "a" * 64, 100, 1, 50, "line\n" * 50)]
        content, _tokens, selected = repo_baseline.repo_context.append_chunks_to_frontier(
            renderer, [], "B documentation query", corpus, context_tokens=1000,
            reserve_tokens=0, max_fresh_tokens=25)
        self.assertTrue(selected)
        self.assertLessEqual(
            len(renderer([{"role": "user", "content": content}]).token_ids), 25)

    def test_repo_preflight_reuses_frozen_selection_and_rejects_mismatch(self) -> None:
        runtime = FakeRuntime()
        current = identity()
        current.update({"context": "16384", "hot_pages": "32", "pager_mode": "off",
                        "command": "llama-server -c 16384 -b 1024 -ub 256",
                        "spec_type": "draft-mtp"})
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            key = root / "api-key"
            key.write_text("fake-key\n", encoding="utf-8")
            output = root / "preflight"
            shared = root / "shared"
            argv = ["run-repo-context-baseline.py", "--placement", "gpu",
                    "--output", str(output), "--shared-state", str(shared),
                    "--api-key-file", str(key), "--preflight-only"]
            with patch.object(repo_baseline, "load_driver", return_value=runtime), \
                    patch.object(repo_baseline, "capture_runtime_identity", return_value=current), \
                    patch.object(repo_baseline, "api_key", return_value="fake-key"), \
                    patch.object(repo_baseline, "ServerPromptRenderer", FakeRenderer), \
                    patch.object(repo_baseline.sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, repo_baseline.main())
            self.assertEqual([], runtime.request_indices)
            preflight = json.loads((output / "primary-conversation-preflight.json").read_text())
            self.assertEqual(0, preflight["requests_sent"])
            selection_path = shared / "scale-corpus-selection.json"
            selection_bytes = selection_path.read_bytes()
            selected_ranges = json.loads(selection_bytes)["selected_ranges"]

            # Idempotent resume must recompute the same bounded frontier and
            # leave the shared evidence bytes untouched.
            with patch.object(repo_baseline, "load_driver", return_value=runtime), \
                    patch.object(repo_baseline, "capture_runtime_identity", return_value=current), \
                    patch.object(repo_baseline, "api_key", return_value="fake-key"), \
                    patch.object(repo_baseline, "ServerPromptRenderer", FakeRenderer), \
                    patch.object(repo_baseline.sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, repo_baseline.main())
            self.assertEqual(selected_ranges,
                             json.loads(selection_path.read_bytes())["selected_ranges"])
            self.assertEqual(selection_bytes, selection_path.read_bytes())
            self.assertEqual([], runtime.request_indices)

            changed_source = {"commit": "changed-candidate",
                              "dirty_fingerprint": "f" * 64}
            with patch.object(repo_baseline, "load_driver", return_value=runtime), \
                    patch.object(repo_baseline, "capture_runtime_identity", return_value=current), \
                    patch.object(repo_baseline, "api_key", return_value="fake-key"), \
                    patch.object(repo_baseline, "ServerPromptRenderer", FakeRenderer), \
                    patch.object(repo_baseline.repo_context, "git_identity",
                                 return_value=changed_source), \
                    patch.object(repo_baseline.sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                        repo_baseline.repo_context.SourceIdentityError,
                        "candidate_source changed"):
                    repo_baseline.main()
            self.assertEqual(selection_bytes, selection_path.read_bytes())
            self.assertEqual([], runtime.request_indices)

    def test_repo_baseline_treats_host_residency_as_optional_diagnostic(self) -> None:
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
        proof = repo_baseline.host_residency_observation(passing)
        self.assertEqual("observed", proof["status"])
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
        absent = repo_baseline.host_residency_observation(no_host_bytes)
        self.assertEqual("observed", absent["status"])
        self.assertEqual(0, absent["host_valid_rows"])
        self.assertEqual([], absent["cold_host_backed_page_ids"])

        unavailable = repo_baseline.host_residency_observation({"slots": []})
        self.assertEqual("unavailable", unavailable["status"])
        unavailable = repo_baseline.host_residency_observation({"slots": [{"id": 0}]})
        self.assertEqual("unavailable", unavailable["status"])

    def test_slot_task_id_is_not_a_stable_generation(self) -> None:
        slot = occupancy.selected_slot({"slots": [{"id": 0, "is_processing": False,
                                                    "id_task": 23,
                                                    "n_prompt_tokens": 4258}]}, 0)
        self.assertIsNone(slot["generation"])
        self.assertEqual(23, slot["task_id"])
        self.assertEqual(4258, slot["occupied_tokens"])

    def test_resume_allows_task_id_change_when_frontier_is_preserved(self) -> None:
        state = {"schema_version": 2, "geometry": {"context_tokens": 32768},
                 "identity_fingerprint": "candidate", "slot": {"slot_id": 0,
                 "generation": None}, "frontier": {"live_occupied_tokens": 4258}}
        slot = {"slot_id": 0, "generation": None, "task_id": 24,
                "occupied_tokens": 4258}
        with patch.object(occupancy, "identity_fingerprint", return_value="candidate"):
            occupancy.validate_resume_state(state, {}, {"context_tokens": 32768}, slot)

    def test_adopts_hashed_completed_prefix_ending_at_live_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            records = []
            prior_frontier = 0
            messages = []
            for index, (stage, user, prompt, completion, after) in enumerate((
                    ("A1", "first", 100, 10, 109),
                    ("B", "second", 200, 10, 209))):
                request_path = root / f"request-{index}.json"
                raw_path = root / f"raw-{index}.sse"
                request_messages = messages + [{"role": "user", "content": user}]
                request_path.write_text(json.dumps({"messages": request_messages}),
                                        encoding="utf-8")
                raw_path.write_text(f"response-{index}", encoding="utf-8")
                response = {"choices": [{"message": {"content": f"answer-{index}"}}]}
                records.append({
                    "request_index": index, "status": "pass", "stage": stage,
                    "request_path": str(request_path), "raw_path": str(raw_path),
                    "request_sha256": occupancy.sha256_file(request_path),
                    "raw_sha256": occupancy.sha256_file(raw_path),
                    "request": {}, "response": response,
                    "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
                    "frontier_before_tokens": prior_frontier,
                    "live_slot_after": {"slot_id": 0, "occupied_tokens": after},
                })
                messages = request_messages + [{"role": "assistant",
                                                "content": f"answer-{index}"}]
                prior_frontier = after
            with patch.object(occupancy, "_read_checkpoint", return_value={
                    "identity_fingerprint": "candidate", "records": records}), \
                    patch.object(occupancy, "identity_fingerprint", return_value="candidate"):
                adopted_messages, adopted_records, history, next_index = \
                    occupancy.adopt_completed_prefix(
                        root, {}, "candidate", {"slot_id": 0, "occupied_tokens": 209},
                        [{"stage": "A1", "user": "first"},
                         {"stage": "B", "user": "second", "fresh_token_limit": 100}],
                        1000)
            self.assertEqual(2, len(adopted_records))
            self.assertTrue(all(record["committed"] for record in adopted_records))
            self.assertEqual(2, len(history))
            self.assertEqual(2, next_index)
            self.assertEqual("answer-1", adopted_messages[-1]["content"])

    def test_slot_snapshot_uses_pager_frontier_for_empty_slot(self) -> None:
        slot = occupancy.selected_slot(
            {"slots": [{"id": 0, "is_processing": False,
                        "pager_metrics": {"slot_generation": 7, "valid_rows": 0,
                                          "target_valid_rows": 0, "host_valid_rows": 0}}]}, 0)
        self.assertEqual(7, slot["generation"])
        self.assertEqual(0, slot["occupied_tokens"])
        populated = occupancy.selected_slot(
            {"slots": [{"id": 0, "is_processing": False,
                        "pager_metrics": {"slot_generation": 7, "valid_rows": 19000,
                                          "target_valid_rows": 12000, "host_valid_rows": 7000}}]}, 0)
        self.assertEqual(19000, populated["occupied_tokens"])

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

    def test_occupancy_freezes_repo_content_schedule_before_generation(self) -> None:
        self.assertIn(262144, validator.SUPPORTED_CONTEXTS)
        renderer = FakeRenderer()
        source = occupancy.repo_context.SourceFile(
            "src/corpus.cpp", "a" * 64, 180000, "repository evidence line\n" * 9000)
        chunks = occupancy.repo_context.source_chunks([source])
        with patch.object(occupancy.repo_context, "tracked_inventory", return_value=[source]), \
                patch.object(occupancy.repo_context, "git_identity",
                             return_value={"commit": "candidate-source",
                                           "dirty_fingerprint": "b" * 64}):
            schedule, plan = occupancy.build_repo_schedule(
                renderer, identity(), 32768, 16384, 16000, 30720)
        self.assertEqual("pass", plan["status"])
        self.assertEqual(0, plan["requests_sent"])
        self.assertGreater(len(schedule), 3)
        self.assertEqual("A1", schedule[0]["stage"])
        self.assertEqual("A2", schedule[-1]["stage"])
        self.assertEqual(chunks[0].path, plan["selected_ranges"][0]["path"])
        self.assertNotIn("BEGIN FILE:", schedule[-1]["user"])
        self.assertEqual(16000, plan["geometry"]["requested_max_fresh_tokens"])
        self.assertEqual(11904, plan["geometry"]["max_fresh_tokens"])
        self.assertEqual(4096, plan["geometry"]["planner_fresh_token_headroom"])
        self.assertEqual(2, plan["geometry"]["generation_write_pages"])
        self.assertEqual(62, plan["geometry"]["max_query_pages"])
        self.assertTrue(all(item.get("fresh_token_limit", 0) <= 11904
                            for item in schedule[1:-1]))
        self.assertTrue(all(item.get("planned_fresh_tokens", 0) <= 11904
                            for item in schedule[1:-1]))
        self.assertTrue(all(item["rendered_prompt_tokens"] <= 32768
                            for item in schedule))
        self.assertGreater(schedule[-2]["rendered_prompt_tokens"], 16384 + 2048)
        self.assertEqual(400, occupancy._scheduled_generation_budget("B", 400))
        self.assertEqual(0, occupancy._scheduled_generation_budget("A2", 400))
        self.assertEqual(29806, occupancy._repo_completion_threshold(30720, 16384, schedule))
        self.assertEqual(30720, plan["requested_occupied_target_tokens"])
        self.assertLessEqual(plan["pre_A2_frontier_target_tokens"], 30720)
        self.assertGreater(plan["pre_A2_frontier_target_tokens"], 16384 + 2048)
        self.assertEqual(30720, plan["requested_A2_prompt_frontier_tokens"])
        self.assertEqual(30315, plan["planner_A2_prompt_frontier_tokens"])
        self.assertLessEqual(schedule[-1]["rendered_prompt_tokens"], 30315)
        self.assertLessEqual(
            plan["pre_A2_frontier_target_tokens"] + schedule[-2]["reserve"]["query_tokens"],
            30720)

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
