#!/usr/bin/env python3
"""Portable lifecycle and telemetry contract tests for the pager adapter."""

from __future__ import annotations

import json
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULE_SPEC = importlib.util.spec_from_file_location(
    "run_pager_profile_benchmark", HERE / "run-pager-profile-benchmark.py")
assert MODULE_SPEC and MODULE_SPEC.loader
adapter = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(adapter)


def identity(profile: str, pid: int, *, binary: str = "/opt/llama-server") -> dict[str, object]:
    return {
        "profile": profile,
        "pid": pid,
        "binary": binary,
        "model": "/models/qwen.gguf",
        "context": "22016",
        "pager_mode": "selective",
        "page_size_tokens": "256",
        "mtp_placement": "gpu",
        "mtp_type_k": "turbo4",
        "mtp_type_v": "turbo4",
    }


def snapshot(profile: str, pid: int, *, binary: str = "/opt/llama-server") -> dict[str, object]:
    observed = identity(profile, pid, binary=binary)
    return {
        "profile": profile,
        "pid": pid,
        "identity": observed,
        "health": {"http_code": 200, "body": "ok"},
    }


def telemetry() -> dict[str, object]:
    return {
        "page_tokens": 256,
        "context_tokens": 16384,
        "target_bytes": 4096,
        "host_budget_bytes": 8192,
        "vram_budget_bytes": 16384,
        "router_top_k": 8,
        "router_explore": 2,
        "pin_recent_tokens": 512,
        "prefetch_depth": 2,
        "route": "selected",
        "mtp_backend": "gpu",
        "target_type_k": "q4_0",
        "target_type_v": "q4_0",
        "mtp_type_k": "turbo4",
        "mtp_type_v": "turbo4",
        "mtp_rows": 2,
        "mtp_bytes": 2048,
        "faults": 1,
        "prefetch_hits": 3,
        "evictions": 1,
        "attention_stale_dropped": 0,
        "d2h_useful_bytes": 100,
        "d2h_aligned_bytes": 128,
        "h2d_useful_bytes": 200,
        "h2d_aligned_bytes": 256,
        "attention_publish_time_us": 7,
        "attention_d2h_time_us": 8,
        "waits": 1,
        "resident_pages": 4,
        "table_epoch": 2,
        "host_pageable_bytes": 4096,
        "host_pinned_bytes": 1024,
        "mode": "selective",
    }


class AdapterContractTests(unittest.TestCase):
    def write_canonical_artifacts(self, output: pathlib.Path, expected: dict[str, object]) -> None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run-config.json").write_text(json.dumps({
            "runtime_identity": {"candidate": expected},
        }) + "\n")
        (output / "records.jsonl").write_text(json.dumps({
            "error": False, "http_code": 200, "timings": {},
        }) + "\n")
        (output / "summary.json").write_text("[]\n")

    def run_main(self, output: pathlib.Path, snapshots: list[dict[str, object]],
                 metrics: list[dict[str, object] | None], *, runner_rc: int = 0,
                 restore: dict[str, object] | None = None) -> int:
        expected = identity("candidate", 202)

        def canonical(command: list[str], **_: object) -> object:
            self.write_canonical_artifacts(pathlib.Path(command[2]), expected)
            return adapter.subprocess.CompletedProcess(command, runner_rc)

        with patch.dict(adapter.os.environ, {
            "BENCH_ENDPOINT": "http://127.0.0.1:8080/v1",
            "CANONICAL_BENCHMARK_RUNNER": "/fake/canonical-runner",
            "LLAMA_ACTIVE_PROFILE": "/fake/active-profile",
            "BENCH_RESTORE_PROFILE": "0",
        }, clear=True), patch.object(adapter, "service_snapshot", side_effect=snapshots), \
                patch.object(adapter, "read_server_metrics", side_effect=[(item, None) for item in metrics]), \
                patch.object(adapter, "restore_profile", return_value=restore), \
                patch.object(adapter.subprocess, "run", side_effect=canonical), \
                patch.object(adapter.sys, "argv", ["run-pager-profile-benchmark.py", "fast", "short", str(output)]):
            return adapter.main()

    def test_missing_queue_and_mtp_not_present_are_explicit(self) -> None:
        missing_queue = telemetry()
        missing_queue.pop("attention_publish_time_us")
        self.assertIn("queue_us", adapter.missing_pager_fields(missing_queue))

        not_present = telemetry()
        not_present["mtp_backend"] = "not_present"
        envelope = adapter.pager_envelope("short", not_present)
        self.assertEqual(envelope["mtp_placement"], "not_present")
        self.assertEqual(envelope["mtp_backend"], "not_present")

    def test_dry_run_derives_v4_context_and_acceptance_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(adapter.os.environ, {}, clear=True), \
                patch.object(adapter.sys, "argv", [
                    "run-pager-profile-benchmark.py", "fast", "stable-focus",
                    directory, "--dry-run"]):
            self.assertEqual(0, adapter.main())
            config = json.loads((pathlib.Path(directory) / "run-config.json").read_text())
            self.assertEqual(22016, config["context"]["resolved"])
            self.assertEqual("acceptance", config["context"]["mode"])
            self.assertFalse(config["context"]["diagnostic_only"])
            self.assertEqual(22016, config["launcher"]["context"])
            self.assertEqual(10752, config["launcher"]["prompt_context_words"])

    def test_sub_ceiling_requires_explicit_diagnostic_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(adapter.os.environ, {}, clear=True), \
                patch.object(adapter.sys, "argv", [
                    "run-pager-profile-benchmark.py", "fast", "stable-focus",
                    directory, "--dry-run", "--context", "16384"]):
            self.assertEqual(2, adapter.main())

    def test_diagnostic_dry_run_records_odd_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(adapter.os.environ, {}, clear=True), \
                patch.object(adapter.sys, "argv", [
                    "run-pager-profile-benchmark.py", "fast", "stable-focus",
                    directory, "--dry-run", "--context", "6401", "--diagnostic"]):
            self.assertEqual(0, adapter.main())
            config = json.loads((pathlib.Path(directory) / "run-config.json").read_text())
            self.assertEqual(1, config["prompt"]["tail_tokens"])
            self.assertTrue(config["context"]["diagnostic_only"])

    def test_http_401_and_403_are_non_secret_auth_errors(self) -> None:
        for status in (401, 403):
            with self.subTest(status=status), patch.dict(adapter.os.environ, {"BENCH_API_KEY": "secret-value"}, clear=True), \
                    patch.object(adapter, "urlopen", side_effect=HTTPError(
                        "http://metrics", status, "rejected", {}, None)):
                values, error = adapter.read_server_metrics("http://127.0.0.1:8080/v1")
                self.assertIsNone(values)
                self.assertIsNotNone(error)
                self.assertIn("authentication", error or "")
                self.assertNotIn("secret-value", error or "")

    def test_identity_mismatch_is_reported(self) -> None:
        mismatches = adapter.identity_mismatches(
            identity("candidate", 202, binary="/opt/other-server"), identity("candidate", 202))
        self.assertIn("binary", mismatches)

    def test_runtime_identity_names_main_pid_exe_and_loaded_dsos(self) -> None:
        pid = os.getpid()
        with patch.object(adapter, "_managed_server_pid", return_value=pid):
            observed = adapter.runtime_identity(None)
        self.assertEqual(pid, observed["main_pid"])
        self.assertEqual(pid, observed["pid"])
        self.assertEqual(observed["binary"], observed["exe"])
        self.assertIn(observed["binary"], observed["proc_maps"])
        self.assertEqual(
            ["/bundle/libllama.so", "/bundle/llama-server"],
            adapter._loaded_project_dsos([
                "/usr/lib/libc.so.6", "/bundle/llama-server", "/bundle/libllama.so",
            ]))

    def test_restoration_requires_exact_transient_overrides(self) -> None:
        before = snapshot("prior", 101)
        after = snapshot("prior", 303)
        before["transient_overrides"] = {
            "AI_BENCHMARK_KV_HOT_PAGES": "4",
            "AI_BENCHMARK_SERVER_BIN": "/opt/frozen/llama-server",
        }
        after["transient_overrides"] = {
            "AI_BENCHMARK_KV_HOT_PAGES": "8",
            "AI_BENCHMARK_SERVER_BIN": "/opt/frozen/llama-server",
        }
        errors = adapter.verify_restoration(
            before, after, {"attempted": True, "state": "restored"})
        self.assertIn("restore_verification_failed:transient_overrides", errors)

    def test_bundle_manifest_hashes_relative_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "bundle"
            (root / "bin").mkdir(parents=True)
            executable = root / "bin" / "llama-server"
            executable.write_bytes(b"candidate")
            executable.chmod(0o555)
            output = pathlib.Path(directory) / "result"
            manifest = adapter.write_bundle_manifest(output, str(executable))
            self.assertIsNotNone(manifest)
            files = manifest["files"]
            self.assertEqual("bin/llama-server", files[0]["path"])
            self.assertEqual(adapter._sha256_file(executable), files[0]["sha256"])
            self.assertEqual("bin/llama-server", manifest["executable"])

    def test_restoration_rejects_wrong_runtime_identity(self) -> None:
        errors = adapter.verify_restoration(
            snapshot("prior", 101),
            snapshot("prior", 303, binary="/opt/other-server"),
            {"attempted": True, "state": "restored", "profile": "prior", "exit_code": 0},
        )
        self.assertIn("restore_verification_failed:identity:binary", errors)

    def test_success_keeps_candidate_loaded_and_separates_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            rc = self.run_main(output, [snapshot("prior", 101), snapshot("candidate", 202)],
                               [telemetry(), telemetry()])
            self.assertEqual(rc, 0)
            lifecycle = json.loads((output / "lifecycle-state.json").read_text())
            self.assertEqual(lifecycle["canonical_exit_code"], 0)
            self.assertEqual(lifecycle["adapter_validation"], "passed")
            self.assertIsNone(lifecycle["failure_class"])
            self.assertIsNone(lifecycle["restoration"])
            self.assertEqual(lifecycle["profile_after"], "candidate")

    def test_missing_queue_restores_and_verifies_prior_service(self) -> None:
        missing_queue = telemetry()
        missing_queue.pop("attention_publish_time_us")
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            restoration = {"attempted": True, "state": "restored", "profile": "prior", "exit_code": 0}
            rc = self.run_main(output,
                               [snapshot("prior", 101), snapshot("candidate", 202), snapshot("prior", 303)],
                               [telemetry(), missing_queue], restore=restoration)
            self.assertEqual(rc, 1)
            lifecycle = json.loads((output / "lifecycle-state.json").read_text())
            self.assertEqual(lifecycle["adapter_validation"], "failed")
            self.assertEqual(lifecycle["failure_class"], "missing_runtime_telemetry")
            self.assertIn("missing_pager_telemetry:queue_us", lifecycle["validation_errors"])
            self.assertEqual(lifecycle["restoration"]["state"], "restored")
            self.assertEqual(lifecycle["profile_after"], "prior")
            self.assertEqual(lifecycle["server_pid_after"], 303)

    def test_candidate_mismatch_restores_prior_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            restoration = {"attempted": True, "state": "restored", "profile": "prior", "exit_code": 0}
            rc = self.run_main(output,
                               [snapshot("prior", 101), snapshot("candidate", 202, binary="/opt/other-server"),
                                snapshot("prior", 303)],
                               [telemetry(), telemetry()], restore=restoration)
            self.assertEqual(rc, 1)
            lifecycle = json.loads((output / "lifecycle-state.json").read_text())
            self.assertTrue(any(error.startswith("runtime_identity_mismatch:")
                                for error in lifecycle["validation_errors"]))
            self.assertEqual(lifecycle["profile_after"], "prior")
            self.assertEqual(lifecycle["server_pid_after"], 303)

    def test_canonical_failure_has_distinct_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            restoration = {"attempted": True, "state": "restored", "profile": "prior", "exit_code": 0}
            rc = self.run_main(output, [snapshot("prior", 101), snapshot("prior", 101), snapshot("prior", 303)],
                               [telemetry(), telemetry()], runner_rc=3, restore=restoration)
            self.assertEqual(rc, 3)
            lifecycle = json.loads((output / "lifecycle-state.json").read_text())
            self.assertEqual(lifecycle["canonical_exit_code"], 3)
            self.assertEqual(lifecycle["adapter_validation"], "failed")
            self.assertEqual(lifecycle["failure_class"], "canonical_runner_failure")


if __name__ == "__main__":
    unittest.main()
