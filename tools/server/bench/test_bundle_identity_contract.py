#!/usr/bin/env python3
"""Deterministic immutable-bundle and measured-row identity checks."""

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
SPEC = importlib.util.spec_from_file_location(
    "run_pager_profile_benchmark_identity", HERE / "run-pager-profile-benchmark.py")
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)

CURVE_SPEC = importlib.util.spec_from_file_location(
    "run_final_curve_identity", HERE / "run-final-curve.py")
assert CURVE_SPEC and CURVE_SPEC.loader
curve = importlib.util.module_from_spec(CURVE_SPEC)
CURVE_SPEC.loader.exec_module(curve)


class BundleIdentityContractTests(unittest.TestCase):
    def _bundle(self, directory: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        root = directory / "bundle"
        (root / "bin").mkdir(parents=True)
        executable = root / "bin" / "llama-server"
        dso = root / "libllama.so"
        executable.write_bytes(b"candidate-v1")
        dso.write_bytes(b"dso-v1")
        for path in (executable, dso):
            path.chmod(0o555)
        adapter.write_build_receipt(root, str(executable))
        root.chmod(0o555)
        return root, executable

    def _identity(self, executable: pathlib.Path, dso: pathlib.Path) -> dict[str, object]:
        return {
            "main_pid": 123,
            "pid": 123,
            "binary": str(executable),
            "exe": str(executable),
            "model": "/models/current.gguf",
            "loaded_dsos": [str(dso)],
            "loaded_file_hashes": {
                str(executable): adapter._sha256_file(executable),
                str(dso): adapter._sha256_file(dso),
            },
        }

    def test_matching_read_only_bundle_records_manifest_and_loaded_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            root, executable = self._bundle(directory)
            dso = root / "libllama.so"
            manifest = adapter.write_bundle_manifest(directory / "result", str(executable))
            self.assertIsNotNone(manifest)
            identity = self._identity(executable, dso)
            self.assertEqual([], adapter.bundle_identity_errors(identity, manifest, "model-hash"))
            self.assertEqual(
                {str(executable), str(dso)},
                set(identity["loaded_file_hashes"]),
            )
            self.assertTrue(manifest["manifest_sha256"])

    def test_missing_candidate_fails_closed_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "requires --server-bin"):
                curve._runtime_identity(
                    type("Adapter", (), {"runtime_identity": lambda self, profile: {}})(),
                    pathlib.Path(temporary), None)

    def test_changed_loaded_dso_fails_closed_and_cannot_be_a_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            root, executable = self._bundle(directory)
            dso = root / "libllama.so"
            manifest = adapter.write_bundle_manifest(directory / "result", str(executable))
            identity = self._identity(executable, dso)
            identity["loaded_file_hashes"] = dict(identity["loaded_file_hashes"])
            identity["loaded_file_hashes"][str(dso)] = "0" * 64
            errors = adapter.bundle_identity_errors(identity, manifest, "model-hash")
            self.assertIn("bundle_loaded_file_hash_mismatch", errors)
            self.assertIn(
                "bundle_loaded_file_not_manifested",
                adapter.bundle_identity_errors(
                    {**identity, "loaded_file_hashes": {**identity["loaded_file_hashes"], "/tmp/rogue.so": "1" * 64}},
                    manifest,
                    "model-hash",
                ),
            )

    def test_each_case_provenance_retains_endpoint_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            root, executable = self._bundle(directory)
            dso = root / "libllama.so"
            identity = self._identity(executable, dso)
            manifest = adapter.write_bundle_manifest(directory / "result", str(executable))
            args = type("Args", (), {
                "reset_mode": "fresh", "cache_condition": "cold-prefill", "context": 128,
                "max_tokens": 1, "page_size": 16, "hot_pages": "auto", "batch_tokens": None,
                "ubatch_tokens": None, "mode": "selective", "prefill_policy": "runtime",
            })()
            record = {"case_id": "case", "status": "pass", "fit": {"token_count": 4},
                      "before": {}, "after": {"metrics": {"context_tokens": 128}}, "usage": {},
                      "speed_measurements": {}}
            result = curve._case_record(
                record, record["fit"], args, identity, "template", "model-hash", "config",
                manifest, {"gpu": "test", "driver": "test", "memory_total_mib": "1"})
            provenance = result["provenance"]
            self.assertEqual(123, provenance["endpoint_pid"])
            self.assertEqual(adapter._sha256_file(executable), provenance["endpoint_executable_sha256"])
            self.assertEqual({str(dso): adapter._sha256_file(dso)},
                             provenance["endpoint_loaded_project_dso_hashes"])
            self.assertEqual("model-hash", provenance["resolved_model_sha256"])


if __name__ == "__main__":
    unittest.main()
