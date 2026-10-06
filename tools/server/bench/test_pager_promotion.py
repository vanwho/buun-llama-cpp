#!/usr/bin/env python3
"""Offline checks for the bounded two-topic pager-promotion prompt."""

from __future__ import annotations

import pathlib
import importlib.util
import sys
import unittest

from pager_promotion import (
    DEFAULT_PRESSURE_FIXTURE_IDS, DEFAULT_SOURCE_FIXTURE_IDS,
    DEFAULT_TARGET_FIXTURE_ID, FIXTURE_ROOT, GENERATION_COMPLETION_LIMIT_TOKENS,
    PLAN_FORMAT_AND_QUERY_RESERVE_TOKENS, SERVER_CONTEXT_TOKENS, GPU_HOT_TOKENS,
    assess_natural_retrieval, build_case_plan, build_promotion_steps,
    load_fixture_catalog, messages_for_step,
    pages_are_cold, pages_overlapping_token_range, refresh_page_versions,
    response_budget,
)

_DRIVER_PATH = pathlib.Path(__file__).with_name("run-pager-promotion.py")
sys.path.insert(0, str(_DRIVER_PATH.parent))
_DRIVER_SPEC = importlib.util.spec_from_file_location("run_pager_promotion", _DRIVER_PATH)
_DRIVER = importlib.util.module_from_spec(_DRIVER_SPEC)
_DRIVER_SPEC.loader.exec_module(_DRIVER)
_promotion_for_page = _DRIVER._promotion_for_page
_assess_content_retrieval = _DRIVER.assess_content_retrieval
_case_acceptance_status = _DRIVER.case_acceptance_status
_completed_pressure_tail_pages = _DRIVER.completed_pressure_tail_pages
_generation_start_index = _DRIVER.generation_start_index


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
        fixture_ids = DEFAULT_SOURCE_FIXTURE_IDS + DEFAULT_PRESSURE_FIXTURE_IDS
        selected = load_fixture_catalog(FIXTURE_ROOT, fixture_ids)
        self.assertEqual(5, len(selected))
        self.assertEqual(fixture_ids, tuple(item.fixture_id for item in selected))

    def test_generation_window_starts_after_changed_query_is_frozen(self) -> None:
        samples = [
            {"query_replay_count": 0, "frozen_history_generation": 0},
            {"query_replay_count": 1, "frozen_history_generation": 0},
            {"query_replay_count": 1, "frozen_history_generation": 7},
            {"query_replay_count": 1, "frozen_history_generation": 7},
        ]
        self.assertEqual(2, _generation_start_index(samples))
        self.assertIsNone(_generation_start_index(samples[:2]))

    def test_answer_quality_markers_follow_each_question(self) -> None:
        source = _assess_content_retrieval(
            "Python01 uses append and extend.",
            "merge_sorted_lists_01.py uses append() and extend().",
            ("merge_sorted_lists_01.py", "append", "extend"))
        final = _assess_content_retrieval(
            "The preallocated merge writes each output position exactly once.",
            "The preallocated output writes each output position exactly once.")
        self.assertTrue(source["matched"])
        self.assertTrue(final["matched"])

    def test_answer_quality_is_required_for_case_acceptance(self) -> None:
        slash_filler = _assess_content_retrieval(
            "The preallocated merge writes each output position exactly once.",
            "// / /// **/***", ("preallocated", "output position", "once"))
        coherent = _assess_content_retrieval(
            "The preallocated merge writes each output position exactly once.",
            "The preallocated merge writes each output position exactly once.",
            ("preallocated", "output position", "once"))
        self.assertEqual("diagnostic_incomplete",
                         _case_acceptance_status(True, True, slash_filler))
        self.assertEqual("pass", _case_acceptance_status(True, True, coherent))

    def test_exact_three_user_turns_and_fixture_order(self) -> None:
        steps = build_promotion_steps(self.catalog)
        self.assertEqual(3, len(steps))
        self.assertEqual(["source_file", "bash_pressure", "natural_recall"],
                         [step.stage for step in steps])
        python_ids = DEFAULT_SOURCE_FIXTURE_IDS
        bash_ids = DEFAULT_PRESSURE_FIXTURE_IDS
        self.assertEqual(python_ids, steps[0].appended_fixture_ids)
        self.assertEqual(bash_ids, steps[1].appended_fixture_ids)
        for fixture_id in python_ids:
            self.assertIn(self.by_id[fixture_id].body, steps[0].user_content)
        for fixture_id in bash_ids:
            self.assertIn(self.by_id[fixture_id].body, steps[1].user_content)
        self.assertIn("merge_sorted_lists_01.py", steps[0].question)
        self.assertIn("merge_sorted_lists_03.py", steps[2].question)
        self.assertEqual(steps[2].question, steps[2].user_content)
        self.assertTrue(steps[0].user_content.endswith(steps[0].question))
        self.assertIn("merge_sorted_lists_01.py",
                      steps[0].expected_answer_local_only)
        self.assertEqual(self.by_id["PY_MERGE_03"].expected_answer,
                         steps[2].expected_answer_local_only)
        self.assertTrue(steps[1].cache_prompt)
        self.assertTrue(steps[2].cache_prompt)
        self.assertNotIn("RETRIEVAL_KEY", steps[0].question + steps[1].question + steps[2].question)
        self.assertNotIn("exactly which filename", steps[0].question.lower())
        self.assertIn("RETRIEVAL_KEY: The preallocated merge writes each output position exactly once.",
                      self.by_id["PY_MERGE_03"].body)

    def test_cumulative_messages_keep_real_prior_assistant_responses(self) -> None:
        steps = build_promotion_steps(self.catalog)
        replies = ["It preallocates and writes each output position once.", "Reviewed."]
        messages = messages_for_step(steps, 2, replies)
        self.assertEqual(["user", "assistant", "user", "assistant", "user"],
                         [message["role"] for message in messages])
        self.assertEqual(replies[0], messages[1]["content"])
        self.assertEqual(replies[1], messages[3]["content"])
        self.assertEqual(steps[0].user_content, messages[0]["content"])
        self.assertEqual(steps[1].user_content, messages[2]["content"])
        self.assertEqual(steps[2].question, messages[-1]["content"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            messages_for_step(steps, 2, replies[:1])

    def test_response_budget_and_filename_scoring(self) -> None:
        self.assertEqual(GENERATION_COMPLETION_LIMIT_TOKENS, response_budget(100))
        self.assertEqual(GENERATION_COMPLETION_LIMIT_TOKENS,
                         response_budget(SERVER_CONTEXT_TOKENS - 128 -
                                         GENERATION_COMPLETION_LIMIT_TOKENS - 1))
        self.assertTrue(assess_natural_retrieval(
            "merge_sorted_lists_03.py", '"merge_sorted_lists_03.py"')['matched'])
        self.assertFalse(assess_natural_retrieval(
            "merge_sorted_lists_03.py", "merge_sorted_lists_02.py")['matched'])
        with self.assertRaises(ValueError):
            response_budget(SERVER_CONTEXT_TOKENS - 128 - 256)
        with self.assertRaises(ValueError):
            response_budget(SERVER_CONTEXT_TOKENS)

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
        changed_versions[-1].update(position_end=1408, valid_length=384)
        refreshed = refresh_page_versions(changed_versions, pages)
        self.assertEqual(list(range(100, 105)),
                         [page["content_version"] for page in refreshed])
        self.assertEqual(1408, refreshed[-1]["position_end"])
        self.assertTrue(pages_are_cold(refreshed, refreshed, require_complete=True))

    def test_case_plan_locks_new_geometry_and_fixture_set(self) -> None:
        plan = build_case_plan(self.catalog, DEFAULT_TARGET_FIXTURE_ID)
        self.assertEqual({"server_context_tokens": SERVER_CONTEXT_TOKENS, "gpu_hot_tokens": GPU_HOT_TOKENS,
                          "page_size_tokens": 256, "hot_pages": 16}, plan["geometry"])
        self.assertEqual(3, plan["request_count"])
        self.assertEqual(list(DEFAULT_SOURCE_FIXTURE_IDS), plan["selected_python_fixture_ids"])
        self.assertEqual(list(DEFAULT_PRESSURE_FIXTURE_IDS), plan["selected_bash_fixture_ids"])
        self.assertEqual(list(DEFAULT_SOURCE_FIXTURE_IDS),
                         plan["steps"][0]["appended_fixture_ids_local_only"])
        self.assertEqual(list(DEFAULT_PRESSURE_FIXTURE_IDS),
                         plan["steps"][1]["appended_fixture_ids_local_only"])
        budget = plan["token_budget"]
        self.assertEqual(5120, budget["selected_fixture_tokens_no_bos"])
        self.assertEqual(2 * GENERATION_COMPLETION_LIMIT_TOKENS,
                         budget["planned_prior_reply_tokens"])
        self.assertEqual(GENERATION_COMPLETION_LIMIT_TOKENS,
                         budget["final_completion_reserve_tokens"])
        self.assertEqual(1024, PLAN_FORMAT_AND_QUERY_RESERVE_TOKENS)
        self.assertEqual(5120 + 3 * GENERATION_COMPLETION_LIMIT_TOKENS + 128 + 1024,
                         budget["estimated_required_tokens"])
        self.assertTrue(budget["fits_with_context_reserve"])
        self.assertTrue(budget["fixture_pressure_exceeds_hot_capacity"])
        self.assertEqual("source_file", plan["steps"][0]["stage"])
        self.assertEqual("bash_pressure", plan["steps"][1]["stage"])
        self.assertEqual("natural_recall", plan["steps"][2]["stage"])

    def test_live_sequence_is_bounded_and_uses_natural_content_questions(self) -> None:
        selected = load_fixture_catalog(FIXTURE_ROOT,
                                        DEFAULT_SOURCE_FIXTURE_IDS + DEFAULT_PRESSURE_FIXTURE_IDS)
        target = next(item for item in selected if item.fixture_id == "PY_MERGE_03")
        steps = build_promotion_steps(selected)
        self.assertEqual(["source_file", "bash_pressure", "natural_recall"],
                         [step.stage for step in steps])
        self.assertEqual(DEFAULT_SOURCE_FIXTURE_IDS, steps[0].appended_fixture_ids)
        self.assertEqual(DEFAULT_PRESSURE_FIXTURE_IDS, steps[1].appended_fixture_ids)
        self.assertIn("merge_sorted_lists_01.py", steps[0].question)
        self.assertIn(target.filename, steps[2].question)
        self.assertIn("allocate", steps[2].question.lower())
        self.assertIn("PY_MERGE_03", steps[2].question)
        self.assertIn("`out[write]`", steps[2].question)
        self.assertIn("how many times", steps[2].question.lower())
        self.assertIn("cursor increments", steps[2].question)
        self.assertIn("preallocated output", steps[2].question)
        self.assertIn("how many times", steps[2].question.lower())
        self.assertIn("output position", steps[2].question)
        self.assertIn(target.filename, steps[2].question)
        self.assertNotIn("RETRIEVAL_KEY", steps[2].question)
        self.assertTrue(steps[1].cache_prompt)
        self.assertTrue(steps[2].cache_prompt)
        self.assertEqual((), steps[2].appended_fixture_ids)
        self.assertEqual(3, len(steps))
        self.assertEqual(8192, _DRIVER.CONTEXT)
        self.assertEqual(4096, _DRIVER.HOT_TOKENS)
        self.assertEqual(1024, _DRIVER.BATCH)
        self.assertEqual(256, _DRIVER.UBATCH)

    def test_pressure_readiness_covers_only_newly_completed_full_pages(self) -> None:
        inventory = [
            {"logical_page_id": index, "position_begin": index * 256,
             "position_end": (index + 1) * 256, "valid_length": 256}
            for index in range(7)]
        pages = _completed_pressure_tail_pages(inventory, 1379, 1792)
        self.assertEqual([5, 6], [page["logical_page_id"] for page in pages])
        self.assertEqual([], _completed_pressure_tail_pages(inventory, 1792, 1900))

    def test_empty_natural_proof_keeps_selector_nomination_unknown(self) -> None:
        page = {"logical_page_id": 7, "generation": 12, "content_version": 31,
                "resident": False, "host_backed": True}
        record = {"request_id": "req-1", "request_generation": 4,
                  "pager_after": {"selector_trace": {"enabled": False}}}
        report = _promotion_for_page(page, {}, record, [])
        self.assertIsNone(report["selector_nominated"])
        self.assertEqual("promotion_chain_incomplete", report["selector_outcome"])
        self.assertFalse(report["claimed_promoted"])

    def test_direct_shortlist_is_required_for_selector_nomination(self) -> None:
        page = {"logical_page_id": 7, "generation": 12, "content_version": 31,
                "resident": False, "host_backed": True}
        record = {"request_id": "req-1", "request_generation": 4,
                  "pager_after": {"selector_trace": {
                      "enabled": True, "raw_selector_output_valid": True,
                      "raw_cold_logical_pages": [7, 9], "outcome": "selected_pending"}}}
        report = _promotion_for_page(page, {}, record, [])
        self.assertTrue(report["selector_nominated"])
        self.assertEqual("raw_selector_output", report["selector_evidence_source"])
        record["pager_after"]["selector_trace"]["raw_cold_logical_pages"] = [9]
        report = _promotion_for_page(page, {}, record, [])
        self.assertIsNone(report["selector_nominated"])
        self.assertEqual("bounded_raw_selector_output", report["selector_evidence_source"])

    def test_authenticated_candidate_proves_nomination_beyond_bounded_raw_ids(self) -> None:
        page = {"logical_page_id": 7, "generation": 12, "content_version": 31,
                "resident": False, "host_backed": True}
        record = {"request_id": "req-1", "request_generation": 4,
                  "pager_after": {"selector_trace": {
                      "enabled": True, "target_candidate_nominated": True,
                      "raw_selector_output_valid": True,
                      "raw_cold_logical_pages": [8, 9], "outcome": "selected_pending"}}}
        report = _promotion_for_page(page, {}, record, [])
        self.assertTrue(report["selector_nominated"])
        self.assertEqual("authenticated_selector_candidate",
                         report["selector_evidence_source"])

    def test_live_trace_uses_captured_selector_result_before_later_reset(self) -> None:
        page = {"logical_page_id": 7, "generation": 12, "content_version": 31,
                "resident": False, "host_backed": True}
        record = {"request_id": "req-1", "request_generation": 4,
                  "pager_after": {"selector_trace": {"enabled": True,
                      "target_logical_page": 7, "outcome": "selector_not_run"}}}
        snapshots = [
            {"observed_monotonic_ns": 1, "trace": {"enabled": True,
                "target_logical_page": 7, "outcome": "selector_not_run"}},
            {"observed_monotonic_ns": 2, "trace": {"enabled": True,
                "target_logical_page": 7, "target_found": True,
                "target_eligible": True, "target_page_generation": 12,
                "target_content_version": 31, "raw_selector_output_valid": True,
                "raw_cold_logical_pages": [7, 9], "outcome": "selected_pending"}},
        ]
        report = _promotion_for_page(page, {}, record, [], snapshots)
        self.assertTrue(report["selector_nominated"])
        self.assertEqual("raw_selector_output", report["selector_evidence_source"])
        self.assertEqual("selected_pending", report["selector_diagnostic"]["outcome"])

    def test_explicit_selector_boundary_supports_negative_nomination(self) -> None:
        page = {"logical_page_id": 7, "generation": 12, "content_version": 31,
                "resident": False, "host_backed": True}
        for outcome in ("selector_not_run", "no_eligible_cold_page"):
            with self.subTest(outcome=outcome):
                record = {"request_id": "req-1", "request_generation": 4,
                          "pager_after": {"selector_trace": {"enabled": True,
                              "outcome": outcome}}}
                report = _promotion_for_page(page, {}, record, [])
                self.assertFalse(report["selector_nominated"])


if __name__ == "__main__":
    unittest.main()
