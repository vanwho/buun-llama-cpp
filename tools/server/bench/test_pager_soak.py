#!/usr/bin/env python3
"""Portable context-boundary tests for the lifecycle soak harness."""

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
SPEC = importlib.util.spec_from_file_location("run_pager_soak", HERE / "run-pager-soak.py")
assert SPEC and SPEC.loader
soak = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(soak)


class PagerSoakContextTests(unittest.TestCase):
    def test_dry_run_uses_corpus_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(soak.sys, "argv", [
                "run-pager-soak.py", directory, "--dry-run"]):
            self.assertEqual(0, soak.main())
            summary = json.loads((pathlib.Path(directory) / "run-summary.json").read_text())
            self.assertEqual(22016, summary["context"]["resolved"])
            self.assertEqual("acceptance", summary["context"]["mode"])
            self.assertEqual(10752, summary["prompt"]["synthetic_context_words"])

    def test_sub_ceiling_soak_requires_diagnostic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(soak.sys, "argv", [
                "run-pager-soak.py", directory, "--context", "16384"]):
            self.assertEqual(2, soak.main())

    def test_diagnostic_odd_tail_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(soak.sys, "argv", [
                "run-pager-soak.py", directory, "--dry-run", "--context", "6401",
                "--diagnostic"]):
            self.assertEqual(0, soak.main())
            summary = json.loads((pathlib.Path(directory) / "run-summary.json").read_text())
            self.assertTrue(summary["context"]["diagnostic_only"])
            self.assertEqual(1, summary["prompt"]["tail_tokens"])
            self.assertEqual("turbo4", summary["placement"]["draft_kv"])
            self.assertEqual("gpu", summary["placement"]["draft_backend"])


if __name__ == "__main__":
    unittest.main()
