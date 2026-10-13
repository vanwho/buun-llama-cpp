"""Offline protocol regression tests for the master context runner."""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("run-master-context-matrix.py")
SPEC = importlib.util.spec_from_file_location("master_context_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class MasterContextMatrixTests(unittest.TestCase):
    def test_exact_48_case_schedule(self) -> None:
        cases = matrix.canonical_schedule()
        self.assertEqual(len(cases), 48)
        self.assertEqual(sum(case["kind"] == "warmup" for case in cases), 12)
        self.assertEqual(sum(case["kind"] == "measured" for case in cases), 36)
        self.assertEqual([case["max_tokens"] for case in cases[:4]], [40, 400, 400, 400])
        self.assertEqual({case["reasoning"] for case in cases}, set(matrix.MODES))
        self.assertEqual({case["prompt_index"] for case in cases}, {0, 1, 2})

    def test_geometry_argv_mapping_has_no_pager_flags(self) -> None:
        self.assertEqual(matrix.row_geometry(8192, 6144, "gpu"),
                         {"context_tokens": 8192, "target_tokens": 6144, "placement": "gpu"})
        gpu = matrix.argv_for_row(8192, "gpu")
        cpu = matrix.argv_for_row(8192, "cpu")
        self.assertIn("turbo4", gpu)
        self.assertNotIn("--no-kv-offload", gpu)
        self.assertIn("--no-kv-offload", cpu)
        self.assertEqual(cpu[cpu.index("-ctk") + 1], "f16")
        self.assertEqual(cpu[cpu.index("-ctv") + 1], "f16")
        for argv in (gpu, cpu):
            self.assertEqual(argv[argv.index("--spec-draft-type-k") + 1], "turbo4")
            self.assertEqual(argv[argv.index("--spec-draft-type-v") + 1], "turbo4")
            self.assertFalse(any("pager" in item for item in argv))
        with self.assertRaises(ValueError):
            matrix.row_geometry(131072, 126976, "gpu")

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


if __name__ == "__main__":
    unittest.main()
