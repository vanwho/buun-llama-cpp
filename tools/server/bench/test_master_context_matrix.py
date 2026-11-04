"""Offline protocol regression tests for the master context runner."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import os
import pathlib
import json
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = pathlib.Path(__file__).with_name("run-master-context-matrix.py")
SPEC = importlib.util.spec_from_file_location("master_context_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)
import repo_context  # noqa: E402
import repo_prefill_map  # noqa: E402
CELL_PATH = pathlib.Path(__file__).with_name("run-context-matrix-cell.py")
CELL_SPEC = importlib.util.spec_from_file_location("context_matrix_cell", CELL_PATH)
assert CELL_SPEC is not None and CELL_SPEC.loader is not None
cell = importlib.util.module_from_spec(CELL_SPEC)
CELL_SPEC.loader.exec_module(cell)


class MasterContextMatrixTests(unittest.TestCase):
    def test_mtp_controls_never_restore_target_without_their_draft_history(self) -> None:
        self.assertTrue(cell.resident_prefix_required("none", "on"))
        self.assertTrue(cell.resident_prefix_required("off", "on"))
        self.assertTrue(cell.resident_prefix_required("selective", "off"))
        self.assertTrue(cell.resident_prefix_required("none", "off"))
        self.assertTrue(cell.resident_prefix_required("off", "off"))
        self.assertTrue(cell.resident_prefix_required("none", "off", prefill_only=True))

    def test_successful_last_stage_does_not_hide_an_earlier_runtime_failure(self) -> None:
        stages = [{"status": "runtime_or_driver_failure"}, {"status": "complete"}]
        self.assertEqual(cell.completed_series_status(stages, 2), "runtime_or_driver_failure")
        self.assertEqual(cell.completed_series_status(stages[1:], 2), "incomplete_series")
        self.assertEqual(cell.completed_series_status(stages[1:], 1), "complete")

    def test_checkpoint_free_control_keeps_gpu_quant_and_rejects_mtp(self) -> None:
        argv = matrix.argv_for_row(8192, "gpu", mtp=False, checkpoint_free_control=True)
        for name, value in (("-ctk", "turbo4"), ("-ctv", "turbo4"), ("-b", "1024"),
                            ("-ub", "256"), ("--ctx-checkpoints", "0"), ("--cache-ram", "0")):
            self.assertEqual(argv[argv.index(name) + 1], value)
        self.assertIn("--no-cache-idle-slots", argv)
        with self.assertRaises(ValueError):
            matrix.argv_for_row(8192, "gpu", mtp=True, checkpoint_free_control=True)

    def test_sparse_decode_keeps_prefill_checkpoints_without_empty_case_requests(self) -> None:
        selections = {"6144": {}, "12288": {"low": [0, 2]}}
        self.assertEqual(cell.checkpoint_case_options(selections, 6144), ["--prefill-only"])
        options = cell.checkpoint_case_options(selections, 12288)
        self.assertEqual(options[0], "--case-selection")
        selected = matrix.validate_case_selection(json.loads(options[1]))
        cases = matrix.canonical_schedule(selection=selected)
        self.assertEqual(len(cases), 8)
        self.assertEqual({c["prompt_index"] for c in cases}, {0, 2})
        self.assertEqual({c["reasoning"] for c in cases}, {"low"})
        self.assertEqual(cell.checkpoint_case_options(None, 6144), [])
        with self.assertRaisesRegex(ValueError, "lacks target"):
            cell.checkpoint_case_options(selections, 28672)

    def test_mapped_multifile_fill_and_checkpoint_resume_use_map_group_bound(self) -> None:
        class CharacterTokenizer:
            def encode(self, text: str) -> list[int]:
                return [ord(char) for char in text]

            def token_byte_offsets(self, text: str) -> tuple[list[int], list[int]]:
                offsets = [0]
                for char in text:
                    offsets.append(offsets[-1] + len(char.encode("utf-8")))
                return self.encode(text), offsets

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

        tokenizer = CharacterTokenizer()

        class FakeRenderer:
            def __call__(self, messages):
                users = [row["content"] for row in messages if row.get("role") == "user"]
                # Account for deterministic per-turn framing while keeping
                # generated assistant acknowledgments out of fake occupancy.
                count = sum(len(tokenizer.encode(text)) for text in users) + 12 * len(users)
                return SimpleNamespace(token_ids=[0] * count)

        class FakeFinal:
            def __init__(self):
                self.clear_calls = 0

            def clear_slot(self, endpoint, key, slot, timeout):
                self_outer.assertEqual(key, "offline-auth-sentinel")
                self.clear_calls += 1
                return {"ok": True}

        def fake_append(final, endpoint, key, payload, raw_path):
            self.assertEqual(key, "offline-auth-sentinel",
                             "continuation metadata must not overwrite the API key")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text("offline fake SSE\n", encoding="utf-8")
            messages = payload["messages"]
            users = [row["content"] for row in messages if row.get("role") == "user"]
            prompt_n = sum(len(tokenizer.encode(text)) for text in users) + 12 * len(users)
            prior_users = users[:-1]
            cache_n = sum(len(tokenizer.encode(text)) for text in prior_users) + 12 * len(prior_users)
            return {
                "content": "a", "reasoning_content": "",
                "usage": {"prompt_tokens": prompt_n, "completion_tokens": 1,
                          "prompt_tokens_details": {"cached_tokens": cache_n}},
                "timings": {"prompt_n": prompt_n, "prompt_ms": 10.0,
                            "prompt_per_second": prompt_n * 100.0, "cache_n": cache_n},
            }

        self_outer = self
        with tempfile.TemporaryDirectory(prefix="mapped-fill-integration-") as tmp:
            base = pathlib.Path(tmp)
            source = base / "source"
            (source / "src").mkdir(parents=True)
            for index in range(220):
                (source / "src" / f"f{index:03d}.txt").write_text(
                    f"short-{index:03d} é λ\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "src"], check=True)
            env = os.environ.copy()
            env.update({"GIT_AUTHOR_NAME": "Fill Test", "GIT_AUTHOR_EMAIL": "fill@example.invalid",
                        "GIT_COMMITTER_NAME": "Fill Test", "GIT_COMMITTER_EMAIL": "fill@example.invalid"})
            subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"],
                           check=True, env=env)

            map_root = base / "map-payload"
            mapped_manifest = repo_prefill_map.build_map(
                source, map_root, tokenizer, chunk_tokens=128, capacity_tokens=1024,
                model_identity={"test": True})
            map_path = map_root / "map.json"
            mapped = repo_prefill_map.MappedCorpus(map_path)
            final = FakeFinal()
            output1 = base / "run-1"
            output1.mkdir()
            output2 = base / "run-2"
            output2.mkdir()
            bounds = []
            real_validate = matrix.validate_fill_bounds

            def capture_bound(*args, **kwargs):
                bounds.append(kwargs.get("max_fresh_tokens"))
                return real_validate(*args, **kwargs)

            with contextlib.redirect_stdout(io.StringIO()):
                with (patch.object(matrix, "SOURCE_WORKTREE", source),
                      patch.object(matrix, "NativeTokenizer", lambda *args, **kwargs: tokenizer),
                      patch.object(matrix, "_append_request", side_effect=fake_append),
                      patch.object(matrix, "validate_fill_bounds", side_effect=capture_bound)):
                    history1, info1 = matrix.fill_context(
                        final, FakeRenderer(), "http://offline", "offline-auth-sentinel", output1,
                        context_tokens=4096, target_tokens=250, fingerprint="a" * 64,
                        reuse_filled_prefix=True, chunk_map_path=map_path)
                    manifest1 = json.loads((output1 / "source-manifest.json").read_text())
                    history2, info2 = matrix.fill_context(
                        final, FakeRenderer(), "http://offline", "offline-auth-sentinel", output2,
                        context_tokens=4096, target_tokens=500, fingerprint="a" * 64,
                        reuse_filled_prefix=True, initial_history=history1,
                        initial_manifest=manifest1, initial_occupied=info1["occupied_tokens"],
                        chunk_map_path=map_path)

            manifest2 = json.loads((output2 / "source-manifest.json").read_text())
            rows1 = [json.loads(line) for line in (output1 / "schedule.jsonl").read_text().splitlines()]
            rows2 = [json.loads(line) for line in (output2 / "schedule.jsonl").read_text().splitlines()]
            ranges1 = manifest1["selected_ranges"]
            ranges2 = manifest2["selected_ranges"]

            self.assertEqual(info1["occupied_tokens"], 250)
            self.assertEqual(info2["occupied_tokens"], 500)
            self.assertEqual(final.clear_calls, 1, "continuation must not erase the saved prefix")
            self.assertEqual([row["chunk_kind"] for row in rows1],
                             ["full_group", "checkpoint_tail"])
            self.assertEqual([row["content_tokens"] for row in rows1], [128, 98])
            self.assertEqual(rows1[0]["content_tokens"], mapped_manifest["chunk_tokens"])
            self.assertGreaterEqual(len({s["path"] for s in ranges1[0]["source_segments"]}), 2,
                                    "one fixed-size group should combine multiple short files")
            self.assertTrue(all(row["fresh_tokens"] <= 128 +
                                matrix.CANONICAL_SUFFIX_RESERVE_TOKENS
                                for row in rows1 + rows2))
            self.assertEqual(set(bounds), {128 + matrix.CANONICAL_SUFFIX_RESERVE_TOKENS})

            cursor1 = mapped.resume_cursor(ranges1)
            self.assertTrue(any(begin < cursor1 < end for begin, end in
                                zip([0] + mapped.ends[:-1], mapped.ends)),
                            "checkpoint must stop inside a frozen group")
            self.assertEqual(ranges2[:len(ranges1)], ranges1)
            continuation = ranges2[len(ranges1):]
            self.assertEqual([row["chunk_kind"] for row in rows2],
                             ["full_group", "checkpoint_tail"])
            self.assertEqual([row["content_tokens"] for row in rows2], [128, 98])
            self.assertEqual(continuation[0]["corpus_start_byte"], cursor1)
            self.assertEqual(continuation[0]["map_chunk_ids"], [1, 2])
            self.assertEqual(mapped.resume_cursor(ranges2), ranges2[-1]["corpus_end_byte"])
            cursor = 0
            for row in ranges2:
                self.assertEqual(row["corpus_start_byte"], cursor)
                cursor = row["corpus_end_byte"]
            self.assertEqual(cursor, ranges2[-1]["corpus_end_byte"])

            # The old global 4K ceiling is gone: a configured 32K group plus
            # the reserved framing allowance is accepted explicitly.
            max_group_fresh = 32768 + matrix.CANONICAL_SUFFIX_RESERVE_TOKENS
            matrix.validate_fill_bounds(max_group_fresh, max_group_fresh, 1, 40000,
                                        max_fresh_tokens=max_group_fresh)
            with self.assertRaisesRegex(RuntimeError, "fresh/context bound"):
                matrix.validate_fill_bounds(max_group_fresh + 1, max_group_fresh + 1,
                                            1, 40000, max_fresh_tokens=max_group_fresh)

    def test_stock_hybrid_rewind_keeps_checkpoints_enabled(self) -> None:
        argv = matrix.argv_for_row(65536, "gpu", mtp=False)
        self.assertEqual(argv[argv.index("--ctx-checkpoints") + 1], "8")
        self.assertEqual(argv[argv.index("--checkpoint-min-step") + 1], "8192")
        self.assertEqual(argv[argv.index("-b") + 1], "1024")
        self.assertEqual(argv[argv.index("-ub") + 1], "256")

    def test_127k_fill_has_a_real_128k_generation_reserve(self) -> None:
        matrix.row_geometry(131072, 130048, "gpu")
        remaining = (131072 - matrix.canonical_prompt_upper_bound(130048)
                     - 400 - matrix.MTP_REPLAY_TOKENS)
        self.assertGreaterEqual(remaining, matrix.SAFETY_TOKENS)
        with self.assertRaises(ValueError):
            matrix.row_geometry(131072, 131000, "gpu")

    def test_remaining_pairs_do_not_repeat_completed_modes_or_prompts(self) -> None:
        selection = matrix.validate_case_selection({"off": [2], "medium": [0, 1, 2]})
        cases = matrix.canonical_schedule(selection=selection)
        self.assertEqual(len(cases), 16)
        self.assertEqual(sum(c["kind"] == "measured" for c in cases), 12)
        self.assertEqual([c["prompt_index"] for c in cases[:4]], [2] * 4)
        self.assertEqual({c["reasoning"] for c in cases}, {"off", "medium"})
        for bad in ({"high": [0]}, {"off": [3]}, {"off": [True]}, {}):
            with self.assertRaises(ValueError):
                matrix.validate_case_selection(bad)

    def test_two_token_ack_is_not_a_completed_speed_measurement(self) -> None:
        self.assertIsNotNone(matrix.canonical_output_error(
            {"content": "I", "usage": {"completion_tokens": 2}}))
        self.assertIsNone(matrix.canonical_output_error(
            {"content": "def merge(a, b):\n    return sorted(a + b)\n" * 4,
             "usage": {"completion_tokens": 100}}))
        self.assertIsNone(matrix.canonical_output_error(
            {"reasoning_content": "Working through the requested algorithm. " * 50,
             "usage": {"completion_tokens": 400}}))

    def test_backend_work_does_not_inflate_fresh_input_bound(self) -> None:
        # ACO/MTP may report additional backend work while occupancy advanced
        # by a valid source chunk. Only the latter is constrained by the fill
        # chunk limit; backend work and elapsed rates remain separate evidence.
        matrix.validate_fill_bounds(3116, 5456, 1, 10000)
        rates = matrix.prefill_measurements(
            3116,
            {"prompt_n": 6219, "prompt_ms": 4554.479,
             "prompt_per_second": 1365.469},
            5.0,
        )
        self.assertEqual(rates["backend_prefill_tokens"], 6219)
        self.assertEqual(rates["backend_prefill_tok_s"], 1365.469)
        self.assertAlmostEqual(rates["fresh_prefill_tok_s"], 3116 / 4.554479)
        self.assertAlmostEqual(rates["wall_prefill_tok_s"], 3116 / 5.0)
        with self.assertRaisesRegex(RuntimeError, "fresh/context bound"):
            matrix.validate_fill_bounds(4097, 5456, 1, 10000, max_fresh_tokens=4096)

    def test_pager_off_adapts_to_upstream_without_pager_cli(self) -> None:
        upstream_help = "--ctx-size N --no-kv-offload"
        fork_help = "--ctx-size N --kv-pager off|selective"
        self.assertEqual(matrix.resolve_pager_mode("off", upstream_help), "none")
        self.assertEqual(matrix.resolve_pager_mode("none", upstream_help), "none")
        self.assertEqual(matrix.resolve_pager_mode("off", fork_help), "off")
        with self.assertRaisesRegex(RuntimeError, "does not support requested selective"):
            matrix.resolve_pager_mode("selective", upstream_help)

    def test_exact_48_case_schedule(self) -> None:
        cases = matrix.canonical_schedule()
        self.assertEqual(len(cases), 48)
        self.assertEqual(sum(case["kind"] == "warmup" for case in cases), 12)
        self.assertEqual(sum(case["kind"] == "measured" for case in cases), 36)
        self.assertEqual([case["max_tokens"] for case in cases[:4]], [40, 400, 400, 400])
        self.assertEqual({case["reasoning"] for case in cases}, set(matrix.MODES))
        self.assertEqual({case["prompt_index"] for case in cases}, {0, 1, 2})

    def test_render_cache_key_reuses_only_same_mode_and_messages(self) -> None:
        messages = [{"role": "user", "content": "same frozen context and prompt"}]
        self.assertEqual(matrix.canonical_render_key("low", messages),
                         matrix.canonical_render_key("low", messages))
        self.assertNotEqual(matrix.canonical_render_key("low", messages),
                            matrix.canonical_render_key("high", messages))
        self.assertNotEqual(matrix.canonical_render_key("low", messages),
                            matrix.canonical_render_key("low", messages + [
                                {"role": "assistant", "content": "new history"}]))

    def test_prompt_sizing_timeout_scales_for_full_context(self) -> None:
        self.assertEqual(matrix.prompt_sizing_timeout(8192), 60.0)
        self.assertEqual(matrix.prompt_sizing_timeout(80896), 79.0)
        self.assertEqual(matrix.prompt_sizing_timeout(262144), 256.0)
        self.assertEqual(matrix.prompt_sizing_timeout(524288), 300.0)

    def test_filled_prompt_preflight_uses_only_reserved_suffix(self) -> None:
        # L-C=2048 leaves 1536 safety + 400 output + 3 MTP replay + 96 suffix.
        estimate = matrix.canonical_prompt_upper_bound(78847)
        self.assertEqual(estimate, 78943)
        self.assertGreaterEqual(80896 - estimate - 400 - matrix.MTP_REPLAY_TOKENS,
                                matrix.SAFETY_TOKENS)

    def test_geometry_argv_mapping_has_no_pager_flags(self) -> None:
        self.assertEqual(matrix.row_geometry(8192, 6144, "gpu"),
                         {"context_tokens": 8192, "target_tokens": 6144, "placement": "gpu"})
        gpu = matrix.argv_for_row(8192, "gpu")
        cpu = matrix.argv_for_row(8192, "cpu")
        self.assertIn("turbo4", gpu)
        self.assertNotIn("--no-kv-offload", gpu)
        self.assertIn("--no-kv-offload", cpu)
        self.assertEqual(cpu[cpu.index("-ctk") + 1], "q4_0")
        self.assertEqual(cpu[cpu.index("-ctv") + 1], "q4_0")
        for argv in (gpu, cpu):
            self.assertEqual(argv[argv.index("--spec-draft-type-k") + 1], "turbo4")
            self.assertEqual(argv[argv.index("--spec-draft-type-v") + 1], "turbo4")
            self.assertFalse(any("pager" in item for item in argv))
        self.assertEqual(matrix.row_geometry(131072, 126976, "gpu"),
                         {"context_tokens": 131072, "target_tokens": 126976,
                          "placement": "gpu"})
        with self.assertRaisesRegex(ValueError, "reserve"):
            matrix.row_geometry(131072, 130200, "gpu")

    def test_selective_pager_uses_turbo4_hot_cache_even_for_host_backing(self) -> None:
        argv = matrix.argv_for_row(90112, "cpu", pager="selective", hot_pages=256,
                                   mtp=True, draft_device="gpu")
        self.assertEqual(argv[argv.index("-ctk") + 1], "turbo4")
        self.assertEqual(argv[argv.index("-ctv") + 1], "turbo4")
        self.assertNotIn("--no-kv-offload", argv)
        self.assertEqual(argv[argv.index("--kv-hot-pages") + 1], "256")
        self.assertEqual(argv[argv.index("--spec-draft-kv-device") + 1], "gpu")

    def test_benchmark_launch_pins_parallelism_and_known_clean_cache_profile(self) -> None:
        argv = matrix.argv_for_row(262144, "cpu", pager="selective", hot_pages=256,
                                   mtp=True, draft_device="gpu")
        for option, expected in (("-t", "16"), ("-tb", "16"), ("-b", "1024"),
                                 ("-ub", "256"), ("-c", "262144"),
                                 ("--cache-ram", "0"), ("--ctx-checkpoints", "0")):
            self.assertEqual(argv[argv.index(option) + 1], expected, option)
        self.assertIn("--slot-save-path", argv)
        self.assertIn("--no-cache-idle-slots", argv)
        self.assertNotIn("--cache-idle-slots", argv)
        self.assertEqual(argv[argv.index("--kv-router") + 1], "probe-rerank")
        self.assertEqual(argv[argv.index("--spec-draft-kv-device") + 1], "gpu")

    def test_cached_prefix_gate_rejects_reprefill_before_scoring_speed(self) -> None:
        matrix.validate_cached_prefix(200000, 199996, label="test", tolerance=8)
        for cached in (None, 0, 100000):
            with self.subTest(cached=cached):
                with self.assertRaisesRegex(RuntimeError, "did not reuse"):
                    matrix.validate_cached_prefix(200000, cached, label="test", tolerance=8)

    def test_filled_series_prefills_once_then_varies_reasoning_per_request(self) -> None:
        all_modes = ("off", "low", "medium", "xhigh")
        self.assertEqual(matrix.repository_fill_reasoning_mode(), "off")
        scheduled = matrix.canonical_schedule(all_modes, (0, 1, 2))
        self.assertEqual({case["reasoning"] for case in scheduled}, set(all_modes))
        self.assertEqual(len(scheduled), 48)
        self.assertEqual(matrix.canonical_payload(
            scheduled[0], [{"role": "user", "content": "base"}])[
                "reasoning_effort"], "none")
        low_case = next(case for case in scheduled if case["reasoning"] == "low")
        self.assertEqual(matrix.canonical_payload(
            low_case, [{"role": "user", "content": "base"}])[
                "reasoning_effort"], "low")

    def test_process_memory_parser_accepts_two_field_threads_line(self) -> None:
        parsed = cell.parse_process_memory_status(
            "VmRSS:\t123 kB\nVmHWM:\t456 kB\nVmSwap:\t0 kB\nThreads:\t16\n")
        self.assertEqual(parsed, {"VmRSS": 123, "VmHWM": 456,
                                  "VmSwap": 0, "Threads": 16})

    def test_series_continuation_uses_only_unseen_repository_bytes(self) -> None:
        text = "first line\nsecond line\nthird line\n"
        chunk = repo_context.CorpusChunk("src/example.cpp", "a" * 64,
            len(text.encode()), 1, 3, text, 0, len(text.encode()))
        rendered = repo_context.render_chunk(chunk).encode("utf-8")
        cut = len(rendered) // 2
        while cut < len(rendered) and (rendered[cut] & 0xC0) == 0x80:
            cut += 1
        ranges = [{"kind": "tracked_source", "path": chunk.path,
                   "sha256": chunk.sha256, "rendered_start_byte": 0,
                   "rendered_end_byte": cut}]
        remaining = matrix.remaining_source_chunks([chunk], ranges)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0][0].encode("utf-8"), rendered[cut:])
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            matrix.remaining_source_chunks([chunk], [{**ranges[0],
                "rendered_start_byte": 1}])

    def test_resume_fingerprint_refuses_changes_and_never_overwrites(self) -> None:
        identity = {"geometry": {"context_tokens": 8192, "target_tokens": 6144},
                    "source": "source-a", "server": "server-a"}
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            fingerprint = matrix.validate_resume(output, identity, resume=False)
            raw = output / "raw.sse"
            matrix._write_unique(raw, b"immutable response")
            self.assertEqual(matrix.validate_resume(output, identity, resume=True), fingerprint)
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                matrix.validate_resume(output, {**identity, "source": "source-b"}, resume=True)
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                matrix.validate_resume(output, {**identity, "geometry": {"context_tokens": 16384}}, resume=True)
            with self.assertRaises(FileExistsError):
                matrix._write_unique(raw, b"replacement")
            self.assertEqual(raw.read_bytes(), b"immutable response")
            with tempfile.TemporaryDirectory() as unconfigured:
                dirty = pathlib.Path(unconfigured)
                (dirty / "raw.sse").write_bytes(b"keep")
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    matrix.validate_resume(dirty, identity, resume=False)
                self.assertEqual((dirty / "raw.sse").read_bytes(), b"keep")
            with tempfile.TemporaryDirectory() as diagnostic_only:
                retry = pathlib.Path(diagnostic_only)
                (retry / "driver-failure-00.log").write_text("preflight failure")
                captures = retry / "runtime-captures"
                captures.mkdir()
                (captures / "runtime-20261013T032947-2678228.json").write_text("{}")
                matrix.validate_resume(retry, identity, resume=False)
                self.assertTrue((retry / "driver-failure-00.log").is_file())
                self.assertTrue((retry / "run-config.json").is_file())

    def test_cpu_host_residency_accepts_unknown_host_device_only(self) -> None:
        self.assertTrue(matrix.residency_rows_match(
            [("unknown", "true", "5570953216")],
            [("CUDA0", "false", "82313216")], True))
        self.assertFalse(matrix.residency_rows_match(
            [("CUDA0", "false", "5570953216")],
            [("CUDA0", "false", "82313216")], True))
        self.assertTrue(matrix.residency_rows_match(
            [("CUDA0", "false", "1000")],
            [("CUDA0", "false", "100")], False))

    def test_live_identity_rejects_argv_and_binary_mismatch(self) -> None:
        executable = pathlib.Path("/frozen/bin/llama-server")
        argv = [str(executable), "-c", "8192"]
        matrix.validate_live_identity(argv, argv, executable, executable)
        with self.assertRaisesRegex(RuntimeError, "argv mismatch"):
            matrix.validate_live_identity(argv, [*argv, "--no-kv-offload"], executable, executable)
        with self.assertRaisesRegex(RuntimeError, "executable mismatch"):
            matrix.validate_live_identity(argv, argv, executable, pathlib.Path("/other/server"))

    def test_frozen_source_ranges_replay_and_allow_only_suffix(self) -> None:
        baseline = {"git_identity": {"commit": "base"}, "fixture_sha256": "f",
                    "inventory": [{"path": "a", "sha256": "1"}],
                    "source_schedule": [{"path": "a", "sha256": "1"}],
                    "selected_ranges": [{"path": "a", "sha256": "1", "end": 3}]}
        matrix.validate_source_replay(baseline, baseline)
        extended = {**baseline, "selected_ranges": baseline["selected_ranges"] +
                    [{"path": "b", "sha256": "2", "end": 4}]}
        matrix.validate_source_replay(extended, baseline, allow_suffix=True)
        with self.assertRaisesRegex(ValueError, "selected range"):
            matrix.validate_source_replay(extended, baseline)
        changed = {**baseline, "selected_ranges": [{"path": "a", "sha256": "different"}]}
        with self.assertRaisesRegex(ValueError, "selected range"):
            matrix.validate_source_replay(changed, baseline)

    def test_preserved_prefix_can_use_exact_tail_closure(self) -> None:
        self.assertEqual(matrix._tail_target_for_frontier(61440, 61438), 61440)
        self.assertEqual(matrix._tail_target_for_frontier(126976, 126976), None)
        self.assertEqual(matrix._tail_target_for_frontier(126976, 126950), None)

    def test_cohort_report_inputs_are_explicit_and_do_not_target_freeze(self) -> None:
        report = pathlib.Path(__file__).with_name("build-master106-cohort-report.py")
        source = report.read_text(encoding="utf-8")
        for option in ("--gpu-json", "--cpu-json", "--freeze-json", "--output-json", "--output-md"):
            self.assertIn(option, source)
        self.assertNotIn("MASTER106_FREEZE.json", source)


if __name__ == "__main__":
    unittest.main()
