#!/usr/bin/env python3
"""Portable tests for the bounded quality-corpus request runner."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import unittest


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
