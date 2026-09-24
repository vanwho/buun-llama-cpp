#!/usr/bin/env python3
"""Offline checks for the bounded two-topic pager-promotion prompt."""

from __future__ import annotations

import pathlib
import unittest

from pager_promotion import (
    BASH_QUESTION, BASH_WINNER, DEFAULT_TARGET_FIXTURE_ID, FIXTURE_ROOT,
    PYTHON_QUESTION, PYTHON_WINNER, assess_natural_retrieval, build_case_plan,
    build_promotion_steps, load_fixture_catalog, messages_for_step,
    pages_are_cold, pages_overlapping_token_range, refresh_page_versions,
    response_budget,
)


class PagerPromotionPromptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_fixture_catalog(FIXTURE_ROOT)
        cls.by_id = {item.fixture_id: item for item in cls.catalog}

    def test_fixture_manifest_and_hashes(self) -> None:
        self.assertEqual(24, len(self.catalog))
        for fixture in self.catalog:
            self.assertEqual(1024, fixture.token_count_no_bos)
            self.assertEqual((FIXTURE_ROOT / fixture.relative_path).read_bytes().decode(),
                             fixture.body)
        selected = load_fixture_catalog(FIXTURE_ROOT, [
            *(f"PY_MERGE_{n:02d}" for n in range(1, 6)),
            *(f"BASH_WATCH_{n:02d}" for n in range(1, 6))])
        self.assertEqual(10, len(selected))
        self.assertEqual("PY_MERGE_03", selected[2].fixture_id)
        self.assertEqual("BASH_WATCH_01", selected[5].fixture_id)

    def test_exact_three_user_turns_and_fixture_order(self) -> None:
        steps = build_promotion_steps(self.catalog)
        self.assertEqual(3, len(steps))
        self.assertEqual(["compare_python", "compare_bash", "repeat_python"],
                         [step.stage for step in steps])
        python_ids = tuple(f"PY_MERGE_{n:02d}" for n in range(1, 6))
        bash_ids = tuple(f"BASH_WATCH_{n:02d}" for n in range(1, 6))
        self.assertEqual(python_ids, steps[0].appended_fixture_ids)
        self.assertEqual(bash_ids, steps[1].appended_fixture_ids)
        for fixture_id in python_ids:
            self.assertIn(self.by_id[fixture_id].body, steps[0].user_content)
        for fixture_id in bash_ids:
            self.assertIn(self.by_id[fixture_id].body, steps[1].user_content)
        self.assertEqual(PYTHON_QUESTION, steps[0].question)
        self.assertEqual(BASH_QUESTION, steps[1].question)
        self.assertEqual(steps[0].question, steps[2].question)
        self.assertEqual(PYTHON_QUESTION, steps[2].user_content)
        self.assertEqual(PYTHON_WINNER, steps[0].expected_answer_local_only)
        self.assertEqual(BASH_WINNER, steps[1].expected_answer_local_only)
        self.assertEqual(PYTHON_WINNER, steps[2].expected_answer_local_only)
        self.assertTrue(steps[1].cache_prompt and steps[2].cache_prompt)
        self.assertNotIn("RETRIEVAL_KEY", steps[0].question + steps[1].question + steps[2].question)

    def test_cumulative_messages_keep_real_prior_assistant_responses(self) -> None:
        steps = build_promotion_steps(self.catalog)
        replies = ["merge_sorted_lists_03.py", "watch_directory_new_files_01.sh"]
        messages = messages_for_step(steps, 2, replies)
        self.assertEqual(["user", "assistant", "user", "assistant", "user"],
                         [message["role"] for message in messages])
        self.assertEqual(replies[0], messages[1]["content"])
        self.assertEqual(replies[1], messages[3]["content"])
        self.assertEqual(steps[0].user_content, messages[0]["content"])
        self.assertEqual(steps[1].user_content, messages[2]["content"])
        self.assertEqual(PYTHON_QUESTION, messages[-1]["content"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            messages_for_step(steps, 2, replies[:1])

    def test_response_budget_and_filename_scoring(self) -> None:
        self.assertEqual(16384 - 100 - 128, response_budget(100))
        self.assertTrue(assess_natural_retrieval(
            PYTHON_WINNER, '"merge_sorted_lists_03.py"')['matched'])
        self.assertFalse(assess_natural_retrieval(
            PYTHON_WINNER, "merge_sorted_lists_02.py")['matched'])
        with self.assertRaises(ValueError):
            response_budget(16384)

    def test_every_overlapping_page_and_mutable_version_are_tracked(self) -> None:
        inventory = [{"logical_page_id": index, "generation": 10 + index,
                      "content_version": 20 + index, "sequence_id": 0,
                      "sequence_generation": 1, "position_begin": index * 256,
                      "position_end": (index + 1) * 256, "valid_length": 256,
                      "host_backed": True, "resident": index < 2}
                     for index in range(5)]
        pages = pages_overlapping_token_range(inventory, 100, 1024 + 100)
        self.assertEqual(list(range(5)), [page["logical_page_id"] for page in pages])
        changed_versions = [dict(page, content_version=100 + index, resident=False)
                            for index, page in enumerate(inventory)]
        refreshed = refresh_page_versions(changed_versions, pages)
        self.assertEqual(list(range(100, 105)),
                         [page["content_version"] for page in refreshed])
        self.assertTrue(pages_are_cold(refreshed, refreshed, require_complete=True))

    def test_case_plan_locks_new_geometry_and_fixture_set(self) -> None:
        plan = build_case_plan(self.catalog, DEFAULT_TARGET_FIXTURE_ID)
        self.assertEqual({"server_context_tokens": 16384, "gpu_hot_tokens": 4096,
                          "page_size_tokens": 256, "hot_pages": 16}, plan["geometry"])
        self.assertEqual(3, plan["request_count"])
        self.assertEqual(PYTHON_WINNER, plan["steps"][0]["expected_answer_local_only"])
        self.assertEqual(PYTHON_WINNER, plan["steps"][2]["expected_answer_local_only"])
        with self.assertRaisesRegex(ValueError, "fixed"):
            build_promotion_steps(self.catalog, "PY_MERGE_01")


if __name__ == "__main__":
    unittest.main()
