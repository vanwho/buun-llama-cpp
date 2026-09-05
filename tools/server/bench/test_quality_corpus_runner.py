#!/usr/bin/env python3
"""Portable tests for the bounded quality-corpus request runner."""

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
    "run_quality_corpus", HERE / "run-quality-corpus.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class QualityCorpusRunnerTests(unittest.TestCase):
    def test_fixture_is_v4_and_context_ceiling_is_not_16k(self) -> None:
        corpus = json.loads((HERE / "fixtures/pager-corpus-v4.json").read_text())
        self.assertEqual([], runner.validate_corpus(corpus))
        self.assertEqual(22016, runner.corpus_context_ceiling(corpus))

    def test_context_ceiling_is_derived_from_cases(self) -> None:
        corpus = {"cases": [{"context_tokens": 22016}]}
        self.assertEqual(22016, runner.corpus_context_ceiling(corpus))

    def test_derived_context_is_acceptance_context(self) -> None:
        resolved = runner.resolve_context("derived", 22016)
        self.assertEqual(22016, resolved["resolved"])
        self.assertEqual("corpus_ceiling", resolved["source"])
        self.assertEqual("acceptance", resolved["mode"])
        self.assertFalse(resolved["diagnostic_only"])

    def test_sub_ceiling_context_is_rejected_without_diagnostic_mode(self) -> None:
        with self.assertRaises(runner.ContextResolutionError):
            runner.resolve_context(16384, 22016)

    def test_sub_ceiling_diagnostic_context_is_explicit(self) -> None:
        resolved = runner.resolve_context(6401, 22016, diagnostic=True)
        self.assertEqual(6401, resolved["resolved"])
        self.assertEqual("diagnostic", resolved["mode"])
        self.assertTrue(resolved["diagnostic_only"])
        self.assertTrue(resolved["sub_ceiling"])

    def test_odd_tail_is_preserved_in_diagnostic_records_and_provenance(self) -> None:
        corpus_path = HERE / "fixtures/pager-corpus-v4.json"
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "request_case",
                side_effect=lambda *_args: (200, {"choices": [{"message": {
                    "content": "middle-needle-29"}}]}, None)):
            output = pathlib.Path(directory)
            with patch.object(sys, "argv", [
                    "run-quality-corpus.py", str(corpus_path), str(output),
                    "--endpoint", "http://example/v1/chat/completions",
                    "--model", "qwen", "--mode", "selective", "--context", "6401",
                    "--diagnostic"]):
                self.assertEqual(1, runner.main())
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(6401, provenance["context"])
            self.assertTrue(provenance["diagnostic_only"])
            records = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
            self.assertEqual(4, sum(record["tail_tokens"] == 1 for record in records))
            self.assertEqual(18, sum(record["status"] == "skipped_context" for record in records))

    def test_exact_checker_normalizes_whitespace(self) -> None:
        case = {"expected_answer": "the answer", "checker": {"type": "exact"}}
        self.assertEqual((True, "pass"), runner.score_case(case, " the   answer\n"))
        self.assertEqual((False, "answer_mismatch"), runner.score_case(case, "other"))

    def test_request_names_are_index_safe(self) -> None:
        request = runner.build_request(
            {"prompt": "fact", "id": "same", "context_tokens": 256},
            "qwen", 32, 42)
        self.assertEqual("fact", request["messages"][0]["content"])
        self.assertFalse("api_key" in json.dumps(request))

if __name__ == "__main__":
    unittest.main()
