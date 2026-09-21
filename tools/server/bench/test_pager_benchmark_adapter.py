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

from pager_benchmark_contract import validate_authenticated_slot_progress

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
        "batch": "128",
        "ubatch": "128",
        "pager_mode": "selective",
        "page_size_tokens": "256",
        "target_kv_placement": "gpu",
        "no_kv_offload": False,
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
        "route": "selected direct",
        "target_backend": "CUDA0",
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
        "queue_time_us": 7,
        "copy_time_us": 8,
        "wait_time_us": 9,
        "waits": 1,
        "selected_page_count": 4,
        "target_valid_rows": 1024,
        "target_valid_bytes": 4096,
        "target_resident_bytes": 4096,
        "physical_pool_capacity_bytes": 16384,
        "target_allocated_bytes": 16384,
        "live_allocation_peak_bytes": 16384,
        "host_valid_rows": 512,
        "host_valid_bytes": 2048,
        "emitted_tokens": 5,
        "predicted_tokens": 4,
        "accepted_tokens": 3,
        "acceptance_denominator": 4,
        "prefill_dense_routes": 1,
        "prefill_reference_routes": 0,
        "prefill_direct_routes": 0,
        "decode_dense_routes": 0,
        "decode_reference_routes": 0,
        "decode_direct_routes": 2,
        "mtp_verify_dense_routes": 0,
        "mtp_verify_reference_routes": 1,
        "mtp_verify_direct_routes": 0,
        "requested_tokens": 16384,
        "admitted_tokens": 1024,
        "allocated_bytes": 16384,
        "valid_rows": 1024,
        "valid_bytes": 4096,
        "selected_pages": 4,
        "snapshot_monotonic_us": 123456,
        "request_generation": 2,
        "slot_generation": 3,
        "config_generation": 1,
        "reset_epoch": 1,
        "table_epoch": 2,
        "table_epoch_changes": 1,
        "host_pageable_bytes": 4096,
        "host_pinned_bytes": 1024,
        "mode": "selective",
    }


def native_identity() -> dict[str, object]:
    return {
        "command": "/opt/llama-server --spec-type draft-mtp --spec-draft-n-max 2 "
                   "--spec-draft-type-k turbo4 --spec-draft-type-v turbo4 "
                   "--spec-draft-kv-device gpu",
        "mtp_placement": "gpu",
        "mtp_type_k": "turbo4",
        "mtp_type_v": "turbo4",
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 2,
    }


def startup_observation() -> dict[str, object]:
    return {
        "draft_backend": "gpu", "type_k": "turbo4", "type_v": "turbo4",
        "reserved_rows": 8192, "reserved_bytes": 8781824,
    }


def mtp_record(draft: int | None = 12, accepted: int | None = 8,
               rate: float | None = 66.6666667) -> dict[str, object]:
    before = {
        "llamacpp:spec_decode_num_draft_tokens_total": 100,
        "llamacpp:spec_decode_num_accepted_tokens_total": 50,
    }
    after = {
        "llamacpp:spec_decode_num_draft_tokens_total": 100 + (draft or 0),
        "llamacpp:spec_decode_num_accepted_tokens_total": 50 + (accepted or 0),
    }
    return {
        "phase": "measured", "prompt_index": 0, "case_key": "q0",
        "request_start": "2026-09-14T10:00:00+00:00",
        "request_end": "2026-09-14T10:00:01+00:00",
        "mtp_source": "prometheus_counter_delta",
        "mtp_counters": {"before": before, "after": after,
                          "delta": {
                              "llamacpp:spec_decode_num_draft_tokens_total": 12,
                              "llamacpp:spec_decode_num_accepted_tokens_total": accepted or 0,
                          }},
        "mtp_journal_excerpt_sha256": "a" * 64,
        "mtp": {"mode": "native", "draft_tokens": draft,
                "accepted_tokens": accepted, "acceptance_percent": rate,
                "journal_case_key": "q0"},
        "timings": {}, "error": False, "http_code": 200,
    }


class AdapterContractTests(unittest.TestCase):
    def test_authenticated_slot_progress_requires_monotonic_request_progress(self) -> None:
        def sample(processed: int, route: str = "selected reference") -> dict[str, object]:
            return {"slots": [{
                "id": 0,
                "is_processing": True,
                "n_prompt_tokens": processed + 128,
                "n_prompt_tokens_processed": processed,
                "pager_metrics": {
                    "status": "ok", "context_tokens": 65536,
                    "page_tokens": 256, "mtp_rows": 65536,
                    "mtp_backend": "gpu", "mtp_type_k": "turbo4",
                    "mtp_type_v": "turbo4", "route": route,
                },
            }]}

        errors, summary = validate_authenticated_slot_progress(
            [sample(1024), sample(2048)], expected_context_tokens=65536,
            expected_page_tokens=256, expected_mtp_rows=65536)
        self.assertEqual([], errors)
        self.assertEqual(1024, summary["first_processed"])
        self.assertEqual(2048, summary["last_processed"])
        self.assertTrue(summary["reference_or_fallback_observed"])
        self.assertFalse(summary["production_cold_success"])

    def test_authenticated_slot_progress_rejects_stalled_or_reduced_geometry(self) -> None:
        sample = {"slots": [{
            "id": 0, "is_processing": True,
            "n_prompt_tokens": 65536, "n_prompt_tokens_processed": 0,
            "pager_metrics": {
                "status": "ok", "context_tokens": 8192,
                "page_tokens": 256, "mtp_rows": 8192,
                "mtp_backend": "gpu", "mtp_type_k": "turbo4",
                "mtp_type_v": "turbo4", "route": "selected reference",
            },
        }]}
        errors, _ = validate_authenticated_slot_progress(
            [sample, sample], expected_context_tokens=65536,
            expected_page_tokens=256, expected_mtp_rows=65536)
        self.assertIn("slot_progress_context_geometry_mismatch", errors)
        self.assertIn("slot_progress_mtp_rows_mismatch", errors)
        self.assertIn("slot_progress_no_authenticated_token_progress", errors)

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
                 restore: dict[str, object] | None = None,
                 mode: str = "selective") -> int:
        expected = identity("candidate", 202)

        def canonical(command: list[str], **_: object) -> object:
            self.assertEqual(_["env"]["BENCH_ENDPOINT"],
                             "http://127.0.0.1:8080/v1/chat/completions")
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
                patch.object(adapter.sys, "argv", ["run-pager-profile-benchmark.py", "fast", "short", str(output),
                                                    "--mode", mode]):
            return adapter.main()

    def test_missing_queue_and_mtp_not_present_are_explicit(self) -> None:
        missing_queue = telemetry()
        missing_queue.pop("queue_time_us")
        self.assertIn("queue_us", adapter.missing_pager_fields(missing_queue))

        not_present = telemetry()
        not_present["mtp_backend"] = "not_present"
        envelope = adapter.pager_envelope("short", not_present)
        self.assertEqual(envelope["mtp_placement"], "not_present")
        self.assertEqual(envelope["mtp_backend"], "not_present")

        errors = adapter.validate_live_telemetry(not_present)
        self.assertIn("mtp_rows_fabricated_without_mtp", errors)

    def test_native_mtp_rows_reject_missing_and_invalid_observations(self) -> None:
        self.assertEqual([], adapter.validate_native_mtp_record(mtp_record()))
        zero = mtp_record(12, 0, 0.0)
        self.assertEqual([], adapter.validate_native_mtp_record(zero))
        fixtures = {
            "missing draft": mtp_record(None, 8, 66.6666667),
            "zero draft": mtp_record(0, 0, 0.0),
            "accepted exceeds": mtp_record(12, 13, 108.3333),
            "bad percentage": mtp_record(12, 8, 66.0),
        }
        for name, record in fixtures.items():
            with self.subTest(name=name):
                self.assertTrue(adapter.validate_native_mtp_record(record))

    def test_native_mtp_counters_reject_reset_and_stale_journal(self) -> None:
        reset = mtp_record()
        reset["mtp_counters"]["after"]["llamacpp:spec_decode_num_draft_tokens_total"] = 99
        self.assertIn("mtp_counter_reset", adapter.validate_native_mtp_record(reset))
        stale = mtp_record()
        stale["mtp"]["journal_case_key"] = "q1"
        self.assertIn("mtp_journal_stale", adapter.validate_native_mtp_record(stale))

    def test_native_mtp_identity_rejects_wrong_device_and_spec_none(self) -> None:
        self.assertEqual([], adapter.validate_native_runtime_identity(
            native_identity(), startup_observation()))
        wrong_device = native_identity()
        wrong_device["command"] = wrong_device["command"].replace(
            "--spec-draft-kv-device gpu", "--spec-draft-kv-device cpu")
        self.assertIn("mtp_spec_draft_device_invalid",
                      adapter.validate_native_runtime_identity(wrong_device, startup_observation()))
        spec_none = native_identity()
        spec_none["command"] = spec_none["command"].replace(
            "--spec-type draft-mtp", "--spec-type none")
        self.assertIn("mtp_spec_type_not_draft_mtp",
                      adapter.validate_native_runtime_identity(spec_none, startup_observation()))

    def test_cumulative_native_counters_keep_three_prompt_rows_independent(self) -> None:
        parsed = adapter.parse_mtp_counters(
            '# HELP ignored\n'
            'llamacpp:spec_decode_num_draft_tokens_total{slot="0"} 100\n'
            'llamacpp:spec_decode_num_accepted_tokens_total{slot="0"} 50\n')
        self.assertEqual({
            "llamacpp:spec_decode_num_draft_tokens_total": 100,
            "llamacpp:spec_decode_num_accepted_tokens_total": 50,
        }, parsed)
        counters = [
            ({"llamacpp:spec_decode_num_draft_tokens_total": 100,
              "llamacpp:spec_decode_num_accepted_tokens_total": 50},
             {"llamacpp:spec_decode_num_draft_tokens_total": 112,
              "llamacpp:spec_decode_num_accepted_tokens_total": 58}),
            ({"llamacpp:spec_decode_num_draft_tokens_total": 112,
              "llamacpp:spec_decode_num_accepted_tokens_total": 58},
             {"llamacpp:spec_decode_num_draft_tokens_total": 124,
              "llamacpp:spec_decode_num_accepted_tokens_total": 58}),
            ({"llamacpp:spec_decode_num_draft_tokens_total": 124,
              "llamacpp:spec_decode_num_accepted_tokens_total": 58},
             {"llamacpp:spec_decode_num_draft_tokens_total": 130,
              "llamacpp:spec_decode_num_accepted_tokens_total": 61}),
        ]
        rows = []
        for index, (before, after) in enumerate(counters):
            delta, errors = adapter.mtp_counter_delta(before, after)
            self.assertEqual([], errors)
            self.assertIsNotNone(delta)
            row = mtp_record(delta["llamacpp:spec_decode_num_draft_tokens_total"],
                             delta["llamacpp:spec_decode_num_accepted_tokens_total"],
                             100 * delta["llamacpp:spec_decode_num_accepted_tokens_total"] /
                             delta["llamacpp:spec_decode_num_draft_tokens_total"])
            row["prompt_index"] = index
            row["case_key"] = f"q{index}"
            row["mtp"]["journal_case_key"] = f"q{index}"
            row["mtp_counters"] = {"before": before, "after": after, "delta": delta}
            rows.append(row)
        self.assertEqual([0, 1, 2], [row["prompt_index"] for row in rows])
        self.assertEqual([12, 12, 6], [row["mtp"]["draft_tokens"] for row in rows])
        self.assertEqual([8, 0, 3], [row["mtp"]["accepted_tokens"] for row in rows])

    def test_feature_off_is_explicit_and_old_records_are_rejected(self) -> None:
        off = {"phase": "measured", "timings": {}, "error": False,
               "mtp_source": "off", "mtp": {"mode": "off"},
               "prompt_index": 0}
        self.assertEqual([], adapter.validate_native_mtp_record(off, mtp_requested=False))
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            (output / "run-config.json").write_text(json.dumps({
                "launcher": {"mtp": "native"},
                "runtime_identity": {"active": native_identity(),
                                      "startup_observation": startup_observation()},
                "resume": {"case_indexes": [0]},
            }) + "\n")
            old = {"phase": "measured", "prompt_index": 0,
                   "timings": {}, "error": False, "http_code": 200}
            (output / "records.jsonl").write_text(json.dumps(old) + "\n")
            self.assertIn("mtp_observation_missing", adapter.record_validation_errors(output))

    def test_native_campaign_requires_three_valid_requested_prompt_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            (output / "run-config.json").write_text(json.dumps({
                "launcher": {"mtp": "native"},
                "runtime_identity": {"active": native_identity(),
                                      "startup_observation": startup_observation()},
                "request": {"prompts": ["q0", "q1", "q2"]},
            }) + "\n")
            records = []
            for index in range(3):
                record = mtp_record()
                record["prompt_index"] = index
                record["case_key"] = f"q{index}"
                record["mtp"]["journal_case_key"] = f"q{index}"
                records.append(record)
            (output / "records.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            self.assertEqual([], adapter.record_validation_errors(output))

    def test_measured_fields_do_not_use_estimate_or_counter_substitutes(self) -> None:
        envelope = adapter.pager_envelope("short", telemetry())
        self.assertEqual(envelope["hot_tokens"], 1024)
        self.assertEqual(envelope["queue_us"], 7)
        self.assertEqual(envelope["copy_us"], 8)
        self.assertEqual(envelope["wait_us"], 9)
        self.assertEqual(envelope["selected_page_count"], 4)
        self.assertEqual(envelope["peak_vram_bytes"], 16384)
        self.assertEqual(envelope["steady_vram_bytes"], 4096)
        self.assertEqual(envelope["target_placement"], "CUDA0")

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
            self.assertEqual("exact-rendered-token-preflight", config["launcher"]["token_sizing"])
            self.assertEqual("gpu", config["launcher"]["target_kv_placement"])
            self.assertFalse(config["launcher"]["no_kv_offload"])
            self.assertIsNone(config["prompt"]["occupied_prompt_tokens"])

    def test_cpu_main_kv_dry_run_records_independent_gpu_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(adapter.os.environ, {}, clear=True), \
                patch.object(adapter.sys, "argv", [
                    "run-pager-profile-benchmark.py", "fast", "short",
                    directory, "--dry-run", "--no-kv-offload"]):
            self.assertEqual(0, adapter.main())
            config = json.loads((pathlib.Path(directory) / "run-config.json").read_text())
            self.assertEqual("cpu", config["launcher"]["target_kv_placement"])
            self.assertTrue(config["launcher"]["no_kv_offload"])
            self.assertEqual("gpu", config["launcher"]["mtp_placement"])
            self.assertTrue(config["placement"]["no_kv_offload"])

    def test_sub_ceiling_requires_explicit_diagnostic_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(adapter.os.environ, {}, clear=True), \
                patch.object(adapter.sys, "argv", [
                    "run-pager-profile-benchmark.py", "fast", "stable-focus",
                    directory, "--dry-run", "--context", "16384"]):
            self.assertEqual(2, adapter.main())

    def test_one_case_and_one_trial_are_forwarded_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(adapter.os.environ, {}, clear=True), \
                patch.object(adapter.sys, "argv", [
                    "run-pager-profile-benchmark.py", "fast", "short", directory,
                    "--dry-run", "--one-case", "--one-trial",
                    "--generation-length", "16", "--progress-bound", "7"]):
            self.assertEqual(0, adapter.main())
            config = json.loads((pathlib.Path(directory) / "run-config.json").read_text())
            self.assertEqual([0], config["resume"]["case_indexes"])
            self.assertEqual("16", os.environ["BENCH_GENERATION_LENGTH"])
            self.assertEqual("7.0", os.environ["BENCH_PROGRESS_BOUND"])

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

    def test_requested_and_observed_microbatch_mismatch_is_reported(self) -> None:
        observed = identity("candidate", 202)
        observed["ubatch"] = "64"
        mismatches = adapter.identity_mismatches(observed, identity("candidate", 202))
        self.assertIn("ubatch", mismatches)

    def test_model_and_loaded_dso_mismatches_are_reported(self) -> None:
        observed = identity("candidate", 202)
        expected = identity("candidate", 202)
        observed["model"] = "/models/other.gguf"
        observed["loaded_dsos"] = ["/bundle/libllama.so"]
        expected["loaded_dsos"] = ["/bundle/libllama.so", "/bundle/libggml.so"]
        mismatches = adapter.identity_mismatches(observed, expected)
        self.assertIn("model", mismatches)
        self.assertIn("loaded_dsos", mismatches)

    def test_restoration_rejects_identity_pid_disagreement(self) -> None:
        before = snapshot("prior", 101)
        after = snapshot("prior", 303)
        after["identity"]["pid"] = 404
        errors = adapter.verify_restoration(
            before, after, {"attempted": True, "state": "restored", "profile": "prior", "exit_code": 0})
        self.assertIn("restore_verification_failed:pid", errors)

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
            adapter.write_build_receipt(root, str(executable))
            output = pathlib.Path(directory) / "result"
            manifest = adapter.write_bundle_manifest(output, str(executable))
            self.assertIsNotNone(manifest)
            files = manifest["files"]
            self.assertEqual("bin/llama-server", files[0]["path"])
            self.assertEqual(adapter._sha256_file(executable), files[0]["sha256"])
            self.assertEqual("bin/llama-server", manifest["executable"])

    def test_missing_receipt_is_not_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "bundle"
            root.mkdir()
            executable = root / "llama-server"
            executable.write_bytes(b"candidate")
            executable.chmod(0o555)
            with self.assertRaisesRegex(ValueError, "missing immutable build-receipt"):
                adapter.write_bundle_manifest(pathlib.Path(directory) / "result", str(executable))

    def test_changed_project_dso_invalidates_build_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "bundle"
            (root / "bin").mkdir(parents=True)
            executable = root / "bin/llama-server"
            dso = root / "libllama.so"
            executable.write_bytes(b"candidate")
            dso.write_bytes(b"dso-v1")
            executable.chmod(0o555)
            adapter.write_build_receipt(root, str(executable))
            dso.write_bytes(b"dso-v2")
            with self.assertRaisesRegex(ValueError, "does not match bundled"):
                adapter.write_bundle_manifest(pathlib.Path(directory) / "result", str(executable))

    def test_comparative_source_identity_does_not_use_invoking_checkout(self) -> None:
        manifest = {"source": {"head": "old", "fingerprint_sha256": "old-build"},
                    "invocation_source": {"head": "new"}}
        self.assertEqual([], adapter.comparative_identity_errors(manifest, "old-build"))
        self.assertEqual(["built_source_fingerprint_mismatch"],
                         adapter.comparative_identity_errors(manifest, "new-build"))

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
        missing_queue.pop("queue_time_us")
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

    def test_feature_off_baseline_allows_optional_pager_gap(self) -> None:
        missing_queue = telemetry()
        missing_queue.pop("queue_time_us")
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            rc = self.run_main(output,
                               [snapshot("prior", 101), snapshot("candidate", 202)],
                               [telemetry(), missing_queue], mode="off")
            self.assertEqual(0, rc)
            lifecycle = json.loads((output / "lifecycle-state.json").read_text())
            self.assertEqual("passed", lifecycle["adapter_validation"])

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
