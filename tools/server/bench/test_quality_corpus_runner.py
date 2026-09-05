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

from pager_benchmark_contract import (
    MandatoryPromptTooLarge,
    PromptFit,
    RenderedPrompt,
    fit_prompt,
)


SPEC = importlib.util.spec_from_file_location(
    "run_quality_corpus", HERE / "run-quality-corpus.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class QualityCorpusRunnerTests(unittest.TestCase):
    @staticmethod
    def deterministic_renderer(messages: list[dict[str, object]]) -> RenderedPrompt:
        rendered = "<bos>" + "".join(
            f"<{message['role']}>{message.get('content', '')}<eom>"
            for message in messages
        )
        return RenderedPrompt(rendered, tuple(rendered.encode("utf-8")),
                               "template:test-v1", "tokenizer:unicode-bytes-v1")

    def test_exact_fitter_preserves_facts_unicode_tools_and_offsets(self) -> None:
        messages = [
            {"role": "system", "content": "prefix: supplied facts stay unchanged"},
            {"role": "tool", "content": {"name": "lookup", "arguments": "{}"}},
            {"role": "user", "content": "FACT π punctuation?! {{PADDING}} Final question?"},
        ]
        fit = fit_prompt(
            messages, "x🙂, x🙂, x🙂, x🙂", 256, 8,
            self.deterministic_renderer,
            protected_facts=("FACT π punctuation?!", "Final question?"),
        )
        self.assertLessEqual(fit.token_count + fit.generation_reserve, 256)
        self.assertEqual("template:test-v1", fit.template_id)
        self.assertEqual("tokenizer:unicode-bytes-v1", fit.tokenizer_id)
        self.assertEqual(2, len(fit.fact_offsets))
        self.assertIn("Final question?", fit.messages[-1]["content"])
        self.assertNotIn("{{PADDING}}", fit.messages[-1]["content"])

    def test_exact_fitter_empty_and_oversized_mandatory_prompt(self) -> None:
        fit = fit_prompt(
            [{"role": "user", "content": "prefix {{PADDING}} suffix"}],
            "", 64, 8, self.deterministic_renderer,
        )
        self.assertEqual(0, fit.padding_characters)
        with self.assertRaises(MandatoryPromptTooLarge):
            fit_prompt(
                [{"role": "user", "content": "mandatory text " * 20 + "{{PADDING}}"}],
                "", 16, 4, self.deterministic_renderer,
            )

    def test_exact_fitter_accepts_page_tail_boundaries_without_live_kv(self) -> None:
        for capacity in (16384, 22016, 262144):
            with self.subTest(capacity=capacity):
                fit = fit_prompt(
                    [{"role": "user", "content": "tail {{PADDING}}"}],
                    "", capacity, 32, self.deterministic_renderer,
                )
                self.assertLessEqual(fit.token_count + 32, capacity)

    def test_marker_heavy_53797_overflow_is_fitted_before_submission(self) -> None:
        def marker_renderer(messages: list[dict[str, object]]) -> RenderedPrompt:
            text = "<bos>" + " ".join(str(message.get("content", "")) for message in messages)
            return RenderedPrompt(text, tuple(range(len(text.split()))),
                                  "template:marker-v1", "tokenizer:marker-v1")

        padding = "marker?! [logical-page] x " * 53797
        naive = marker_renderer([{"role": "user", "content": padding}])
        self.assertGreater(len(naive.token_ids), 22016)
        fit = fit_prompt(
            [{"role": "user", "content": "FACT candidate-a {{PADDING}} Final question?"}],
            padding, 22016, 32, marker_renderer,
            protected_facts=("FACT candidate-a", "Final question?"),
        )
        self.assertLessEqual(fit.token_count + 32, 22016)
        self.assertIn("candidate-a", fit.messages[0]["content"])

    def test_fixture_is_v4_and_context_ceiling_is_not_16k(self) -> None:
        corpus = json.loads((HERE / "fixtures/pager-corpus-v4.json").read_text())
        self.assertEqual([], runner.validate_corpus(corpus))
        self.assertEqual(22016, runner.corpus_context_ceiling(corpus))

    def test_quality_case_builder_uses_explicit_padding_tail(self) -> None:
        corpus = json.loads((HERE / "fixtures/pager-corpus-v4.json").read_text())
        fit = runner.fit_case_prompt(corpus["cases"][0], self.deterministic_renderer, 4096, 8)
        self.assertLessEqual(fit.token_count + 8, 4096)
        self.assertNotIn("{{PAGER_PADDING}}", fit.messages[0]["content"])
        self.assertGreaterEqual(len(fit.fact_offsets), 2)

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
                side_effect=lambda *_args: (200, {"usage": {"prompt_tokens": 1},
                    "choices": [{"message": {"content": "middle-needle-29"}}]}, None)), \
                patch.object(runner, "ServerPromptRenderer", return_value=type(
                    "FakeRenderer", (), {"template_id": "template:test",
                                         "tokenizer_id": "tokenizer:test"})()), \
                patch.object(runner, "fit_case_prompt", return_value=PromptFit(
                    messages=[{"role": "user", "content": "fitted"}],
                    rendered_text="fitted", token_ids=(1,), token_count=1,
                    desired_occupancy=6401, generation_reserve=32,
                    padding_characters=0, template_id="template:test",
                    tokenizer_id="tokenizer:test", fact_offsets=(),
                    request_token_sha256="0" * 64)):
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
