"""Offline protocol regression tests for the master context runner."""

from __future__ import annotations

import importlib.util
import pathlib
import json
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("run-master-context-matrix.py")
SPEC = importlib.util.spec_from_file_location("master_context_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class MasterContextMatrixTests(unittest.TestCase):
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
            matrix.validate_fill_bounds(4097, 5456, 1, 10000)

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
            matrix.row_geometry(131072, 130000, "gpu")

    def test_selective_pager_uses_turbo4_hot_cache_even_for_host_backing(self) -> None:
        argv = matrix.argv_for_row(90112, "cpu", pager="selective", hot_pages=256,
                                   mtp=True, draft_device="gpu")
        self.assertEqual(argv[argv.index("-ctk") + 1], "turbo4")
        self.assertEqual(argv[argv.index("-ctv") + 1], "turbo4")
        self.assertNotIn("--no-kv-offload", argv)
        self.assertEqual(argv[argv.index("--kv-hot-pages") + 1], "256")
        self.assertEqual(argv[argv.index("--spec-draft-kv-device") + 1], "gpu")

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
