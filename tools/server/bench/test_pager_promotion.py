#!/usr/bin/env python3
"""Offline repeatability tests for file-backed pager-promotion prompts."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from pager_promotion import (
    DEFAULT_B_COUNT,
    DEFAULT_TARGET_FIXTURE_ID,
    FIXTURE_ROOT,
    assess_natural_retrieval,
    build_case_plan,
    build_promotion_steps,
    load_fixture_catalog,
    messages_for_step,
    normalize_answer,
    response_budget,
    select_b_fixtures,
    write_plan,
)


class PagerPromotionPromptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_fixture_catalog(FIXTURE_ROOT)
        cls.by_id = {item.fixture_id: item for item in cls.catalog}

    def test_manifest_and_source_files_are_complete_and_hashed(self) -> None:
        self.assertEqual(24, len(self.catalog))
        self.assertEqual(3, len({item.category for item in self.catalog}))
        self.assertTrue(all(item.token_count_no_bos == 1024 for item in self.catalog))
        for item in self.catalog:
            self.assertEqual(64, len(item.sha256))
            self.assertEqual((FIXTURE_ROOT / item.relative_path).read_bytes().decode("utf-8"),
                             item.body)

    def test_retrieval_answer_comparison_accepts_only_benign_outer_formatting(self) -> None:
        expected = "A retrieval fact with enough tokens to exercise a complete answer."
        self.assertEqual(expected, normalize_answer(expected))
        self.assertEqual(expected, normalize_answer(f"RETRIEVAL_KEY: {expected}"))
        self.assertEqual(expected, normalize_answer(f'Retrieval key: "{expected}"'))
        self.assertNotEqual(expected, normalize_answer(f"RETRIEVAL_KEY: {expected} extra"))

    def test_response_budget_uses_available_context_not_answer_length(self) -> None:
        self.assertEqual(8192 - 100 - 128, response_budget(100))
        self.assertGreater(response_budget(1000), 6000)
        with self.assertRaises(ValueError):
            response_budget(-1)
        with self.assertRaises(ValueError):
            response_budget(8192)

    def test_each_target_has_deterministic_same_family_context_documents(self) -> None:
        for target in self.catalog:
            first = select_b_fixtures(self.catalog, target.fixture_id)
            second = select_b_fixtures(self.catalog, target.fixture_id)
            self.assertEqual(first, second)
            self.assertEqual(DEFAULT_B_COUNT, len(first))
            self.assertTrue(all(item.category == target.category for item in first))
            self.assertNotIn(target.fixture_id, {item.fixture_id for item in first})
            self.assertEqual(4, len({item.fixture_id for item in first}))

    def test_primary_case_is_natural_six_turn_a_b_a_sequence(self) -> None:
        steps = build_promotion_steps(self.catalog, DEFAULT_TARGET_FIXTURE_ID)
        self.assertEqual(6, len(steps))
        self.assertEqual("ingest_A_ack", steps[0].stage)
        self.assertEqual("append_B_ack", steps[1].stage)
        self.assertEqual("query_A_again", steps[-1].stage)
        self.assertEqual("", steps[0].expected_answer_local_only)
        self.assertIn("Acknowledge briefly", steps[0].user_content)
        self.assertTrue(steps[0].user_content.endswith(steps[0].question))
        self.assertTrue(steps[1].user_content.startswith(
            "Read the following file as context (merge_sorted_lists_02.py):"))
        self.assertIn("acknowledge briefly without summarizing", steps[1].user_content)
        self.assertNotIn("Copy every character", steps[1].user_content)
        self.assertTrue(steps[1].user_content.endswith(steps[1].question))
        self.assertIn("which input's value", steps[-1].question)
        self.assertIn("current values are equal", steps[-1].question)
        self.assertIn("merge_sorted_lists_01.py", steps[-1].question)
        self.assertIn("PY_MERGE_01", steps[-1].question)
        self.assertIn("Answer naturally", steps[-1].question)
        self.assertNotIn("RETRIEVAL_KEY", steps[-1].user_content)
        self.assertIn(self.by_id["PY_MERGE_01"].body, steps[0].user_content)
        self.assertEqual(self.by_id["PY_MERGE_01"].expected_answer,
                         steps[-1].expected_answer_local_only)
        self.assertTrue(all(step.cache_prompt for step in steps[1:]))
        self.assertFalse(steps[0].cache_prompt)

    def test_all_families_use_a_free_form_ingest_acknowledgement(self) -> None:
        for fixture_id in ("PY_MERGE_01", "MMAP_READ_01", "BASH_WATCH_01"):
            question = build_promotion_steps(self.catalog, fixture_id)[0].question
            self.assertIn("Acknowledge briefly", question)
            self.assertNotIn("YES or NO", question)

    def test_each_continuation_preserves_prior_user_and_actual_assistant_turns(self) -> None:
        steps = build_promotion_steps(self.catalog, "MMAP_READ_03")
        prior_answers: list[str] = []
        previous_messages: list[dict[str, str]] = []
        for index, step in enumerate(steps):
            messages = messages_for_step(steps, index, prior_answers)
            self.assertEqual("user", messages[-1]["role"])
            self.assertEqual(step.user_content, messages[-1]["content"])
            if index:
                self.assertEqual(previous_messages + [
                    {"role": "assistant", "content": prior_answers[-1]},
                    {"role": "user", "content": step.user_content},
                ], messages)
            previous_messages = messages
            prior_answers.append("Understood; I have read this file.")

    def test_free_form_actual_prior_answers_are_preserved(self) -> None:
        steps = build_promotion_steps(self.catalog, "BASH_WATCH_01")
        messages = messages_for_step(steps, 1, ["Sure, I have read it."])
        self.assertEqual("Sure, I have read it.", messages[-2]["content"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            messages_for_step(steps, 2, ["YES"])

    def test_natural_retrieval_check_accepts_paraphrase_not_exact_sentence(self) -> None:
        result = assess_natural_retrieval(
            "PY_MERGE_01", "On equal values, the item from the left input comes first.")
        self.assertEqual("pass", result["status"])
        self.assertTrue(result["matched"])
        self.assertEqual("fail", assess_natural_retrieval(
            "PY_MERGE_01", "The right input is selected first.")["status"])

    def test_b_count_is_bounded_for_natural_pressure_escalation(self) -> None:
        self.assertEqual(8, len(build_promotion_steps(
            self.catalog, "PY_MERGE_02", b_count=6)))
        with self.assertRaises(ValueError):
            select_b_fixtures(self.catalog, "PY_MERGE_02", b_count=7)

    def test_plan_contains_prompts_and_local_expectations_separately(self) -> None:
        plan = build_case_plan(self.catalog, "PY_MERGE_01")
        self.assertEqual({"server_context_tokens": 8192, "gpu_hot_tokens": 4096,
                          "page_size_tokens": 256, "hot_pages": 16}, plan["geometry"])
        final = plan["steps"][-1]
        self.assertIn("which input's value", final["user_content"])
        self.assertIn("current values are equal", final["user_content"])
        self.assertIn("merge_sorted_lists_01.py", final["user_content"])
        self.assertEqual(self.by_id["PY_MERGE_01"].expected_answer,
                         final["expected_answer_local_only"])
        self.assertNotIn("RETRIEVAL_KEY", final["user_content"])

    def test_manifest_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text())
            (root / "manifest.json").write_text(json.dumps(manifest))
            for item in self.catalog:
                destination = root / item.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(item.body, encoding="utf-8")
            first = root / self.catalog[0].relative_path
            first.write_text(first.read_text() + "tampered\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_fixture_catalog(root)

    def test_plan_writer_refuses_to_overwrite(self) -> None:
        plan = build_case_plan(self.catalog, "MMAP_READ_01")
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "plan.json"
            write_plan(output, [plan])
            self.assertEqual(plan, json.loads(output.read_text()))
            with self.assertRaises(FileExistsError):
                write_plan(output, [plan])


if __name__ == "__main__":
    unittest.main()
