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
    GENERATION_CONTEXT_RESERVE_TOKENS,
    PLAN_FORMAT_AND_QUERY_RESERVE_TOKENS, SERVER_CONTEXT_TOKENS, GPU_HOT_TOKENS,
    assess_natural_retrieval, build_case_plan, build_promotion_steps,
    load_fixture_catalog, messages_for_step, build_frozen_schedule,
    frozen_schedule_hash, write_or_validate_schedule,
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
_answer_trace_page = _DRIVER.answer_trace_page
_completed_pressure_tail_pages = _DRIVER.completed_pressure_tail_pages
_generation_start_index = _DRIVER.generation_start_index
_request_local_mtp = _DRIVER.request_local_mtp
_classify_sequence = _DRIVER.classify_sequence
_profile_setting_mismatches = _DRIVER.profile_setting_mismatches
_validate_summary = _DRIVER.validate_summary


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

    def test_answer_page_boundary_is_recorded_without_dropping_outcome(self) -> None:
        page, wholly_within, answer_pages = _answer_trace_page(
            anchor_start=2092, anchor_end=2102,
            fixture_start=1078, fixture_end=2103)
        self.assertEqual(8, page)
        self.assertFalse(wholly_within)
        self.assertEqual([8], answer_pages)

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
        self.assertEqual("I have reviewed the Python merge examples.",
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
        self.assertEqual(400, response_budget(100))
        self.assertEqual(400, response_budget(SERVER_CONTEXT_TOKENS - 401))
        self.assertTrue(assess_natural_retrieval(
            "merge_sorted_lists_03.py", '"merge_sorted_lists_03.py"')['matched'])
        self.assertFalse(assess_natural_retrieval(
            "merge_sorted_lists_03.py", "merge_sorted_lists_02.py")['matched'])
        with self.assertRaises(ValueError):
            response_budget(SERVER_CONTEXT_TOKENS - 399)
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
        self.assertEqual(0, budget["planned_prior_reply_tokens"])
        self.assertEqual(GENERATION_COMPLETION_LIMIT_TOKENS,
                         budget["final_completion_reserve_tokens"])
        self.assertEqual(1024, PLAN_FORMAT_AND_QUERY_RESERVE_TOKENS)
        self.assertEqual(8080, budget["estimated_required_tokens"])
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
        self.assertIn("allocate", steps[2].expected_answer_local_only.lower())
        self.assertIn("PY_MERGE_03", steps[2].question)
        self.assertIn("what implementation behavior", steps[2].question.lower())
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

    def test_bash_target_uses_same_two_python_three_bash_sequence(self) -> None:
        ids = DEFAULT_SOURCE_FIXTURE_IDS + DEFAULT_PRESSURE_FIXTURE_IDS
        catalog = load_fixture_catalog(FIXTURE_ROOT, ids)
        target = next(item for item in catalog if item.fixture_id == "BASH_WATCH_02")
        steps = build_promotion_steps(catalog, target.fixture_id,
                                      python_fixture_ids=DEFAULT_SOURCE_FIXTURE_IDS,
                                      bash_fixture_ids=DEFAULT_PRESSURE_FIXTURE_IDS)
        self.assertEqual(DEFAULT_SOURCE_FIXTURE_IDS, steps[0].appended_fixture_ids)
        self.assertEqual(DEFAULT_PRESSURE_FIXTURE_IDS, steps[1].appended_fixture_ids)
        self.assertIn(target.filename, steps[2].question)
        self.assertEqual(target.expected_answer, steps[2].expected_answer_local_only)

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

    def test_frozen_schedule_targets_and_a2_does_not_reinsert_source(self) -> None:
        schedule = build_frozen_schedule(self.catalog, FIXTURE_ROOT)
        self.assertEqual(["PY_MERGE_03", "PY_MERGE_01", "BASH_WATCH_01"],
                         [row["target_fixture_id"] for row in schedule["sequences"]])
        python_sequence = schedule["sequences"][0]
        self.assertEqual(2, len(python_sequence["turns"][0]["appended_fixture_ids_local_only"]))
        self.assertEqual(3, len(python_sequence["turns"][1]["appended_fixture_ids_local_only"]))
        bash_sequence = schedule["sequences"][2]
        self.assertEqual(2, len(bash_sequence["turns"][0]["appended_fixture_ids_local_only"]))
        self.assertEqual(3, len(bash_sequence["turns"][1]["appended_fixture_ids_local_only"]))
        self.assertEqual([], bash_sequence["turns"][2]["appended_fixture_ids_local_only"])
        self.assertNotIn(self.by_id["BASH_WATCH_01"].body,
                         bash_sequence["turns"][2]["user_content"])
        self.assertEqual(947300, schedule["seed_base"])
        self.assertEqual(1536, schedule["reserve_tokens"])

    def test_shared_schedule_is_create_once_and_source_bound(self) -> None:
        import tempfile
        schedule = build_frozen_schedule(self.catalog, FIXTURE_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "frozen.json"
            first_hash = write_or_validate_schedule(path, schedule)
            self.assertEqual(first_hash, write_or_validate_schedule(path, schedule))
            changed = dict(schedule)
            changed["seed_base"] = 1
            with self.assertRaisesRegex(ValueError, "differs"):
                write_or_validate_schedule(path, changed)
        schedule["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            _DRIVER._schedule_sources_valid(schedule, self.catalog, FIXTURE_ROOT)

    def test_actual_reply_is_retained_at_its_full_length(self) -> None:
        steps = build_promotion_steps(self.catalog)
        reply = "actual prior response " + ("detail " * 700)
        messages = messages_for_step(steps, 1, [reply])
        self.assertEqual(reply, messages[1]["content"])
        self.assertGreater(len(messages[1]["content"]), 4000)
        self.assertEqual(1536, GENERATION_CONTEXT_RESERVE_TOKENS)

    def test_profile_matching_enforces_selective_and_dense_flags(self) -> None:
        base = {"--kv-pager": "selective", "--kv-router": "legacy",
                "--kv-page-size": "256", "--kv-hot-pages": "16"}
        self.assertEqual([], _profile_setting_mismatches(base, "legacy"))
        self.assertTrue(_profile_setting_mismatches(base, "probe-rerank"))
        self.assertTrue(_profile_setting_mismatches(base, "dense"))
        dense = {"--kv-pager": "off", "--kv-router": None, "--kv-hot-pages": None}
        self.assertEqual([], _profile_setting_mismatches(dense, "dense"))

    def test_response_local_mtp_counts_win_over_zero_slot_deltas(self) -> None:
        observation = _request_local_mtp(
            {"timings": {"draft_n": 398, "draft_n_accepted": 0,
                         "prompt_per_second": 1200.0}},
            {"mtp_request_counters": {"draft_n": 8, "draft_n_accepted": 3}},
            {"mtp_request_counters": {"draft_n": 8, "draft_n_accepted": 3}})
        self.assertEqual(398, observation["drafted"])
        self.assertEqual(0, observation["accepted"])
        self.assertEqual("response.timings", observation["origin"])
        self.assertEqual(0.0, observation["acceptance_percent"])
        self.assertEqual(1200.0, observation["raw_response_timings"]["prompt_per_second"])
        no_draft = _request_local_mtp({"timings": {"draft_n": 0,
            "draft_n_accepted": 0}},
            {"mtp_request_counters": {"draft_n": 4, "draft_n_accepted": 2}},
            {"mtp_request_counters": {"draft_n": 4, "draft_n_accepted": 2}})
        self.assertEqual(0, no_draft["drafted"])
        self.assertEqual(0, no_draft["accepted"])
        self.assertEqual("response.timings", no_draft["origin"])
        self.assertIsNone(no_draft["acceptance_percent"])

    def test_missing_mtp_counts_remain_unknown(self) -> None:
        observation = _request_local_mtp({}, {}, {})
        self.assertIsNone(observation["drafted"])
        self.assertIsNone(observation["accepted"])
        self.assertIsNone(observation["acceptance_percent"])
        self.assertIsNone(observation["origin"])

    def test_http500_is_execution_incomplete_and_miss_is_valid(self) -> None:
        requests = [{"http_status": 200} for _ in range(3)]
        requests[-1] = {"http_status": 500, "runtime_error": "query_finalize"}
        self.assertEqual("execution_incomplete",
                         _classify_sequence("probe-rerank", requests, None, None))
        self.assertEqual("semantic_miss",
                         _classify_sequence("probe-rerank", [{"http_status": 200}] * 3,
                                            False, None))
        self.assertEqual("dense_inconclusive",
                         _classify_sequence("dense", [{"http_status": 200}] * 3,
                                            False, None))

    def test_summary_validator_accepts_miss_dense_and_rejects_missing_raw_request(self) -> None:
        import hashlib
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            records = []
            for index in range(3):
                request = root / f"request-{index}.json"
                response = root / f"response-{index}.json"
                payload = {"messages": [{"role": "user", "content": f"u{turn} ",}
                                        for turn in range(index + 1)],
                           "seed": 947300 + index, "temperature": 0,
                           "top_p": 1, "max_tokens": 400,
                           "model": "qwen38-fast-turbo4-mtp",
                           "reasoning_effort": "none",
                           "chat_template_kwargs": {"enable_thinking": False},
                           "cache_prompt": index > 0}
                request.write_text(json.dumps(payload))
                response.write_text("{}")
                records.append({"http_status": 200,
                    "raw_request_artifact": {"path": str(request),
                        "sha256": hashlib.sha256(request.read_bytes()).hexdigest()},
                    "raw_response_artifact": {"path": str(response),
                        "sha256": hashlib.sha256(response.read_bytes()).hexdigest()}})
            frozen = root / "schedule.json"
            schedule = {"schema": "pager-promotion-frozen-schedule-v1",
                "seed_base": 947300,
                "geometry": {"context_tokens": 8192, "hot_tokens": 4096,
                    "page_tokens": 256, "batch": 1024, "ubatch": 256},
                "request_options": {"model_alias": "qwen38-fast-turbo4-mtp",
                    "temperature": 0, "top_p": 1,
                    "max_tokens": 400, "reasoning_effort": "none",
                    "enable_thinking": False, "stop": None},
                "sources": [{"fixture_id": "x", "sha256": "a" * 64}],
                "sequences": [{"target_fixture_id": "T", "turns": [
                    {"user_content": "u0 ", "cache_prompt": False},
                    {"user_content": "u1 ", "cache_prompt": True},
                    {"user_content": "u2 ", "cache_prompt": True}]},
                    {"target_fixture_id": "U", "turns": [
                    {"user_content": "v0 ", "cache_prompt": False},
                    {"user_content": "v1 ", "cache_prompt": True},
                    {"user_content": "v2 ", "cache_prompt": True}]}]}
            frozen.write_text(json.dumps(schedule))
            schedule_hash = hashlib.sha256(frozen.read_bytes()).hexdigest()
            identity = {"binary_sha256": "a", "model_sha256": "b",
                        "candidate_dsos": [{"sha256": "c"}]}
            shared = {"geometry": schedule["geometry"], "seed_base": 947300,
                      "request_options": schedule["request_options"]}
            summary = {"schema": "pager-promotion-profile-summary-v1", "profile": "dense",
                "candidate_identity": identity,
                "candidate_identity_sha256": hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "schedule_path": str(frozen), "schedule_sha256": schedule_hash,
                "shared_options": shared,
                "shared_options_sha256": hashlib.sha256(
                    json.dumps(shared, sort_keys=True).encode()).hexdigest(),
                "selected_targets": ["T"],
                "sequences": [{"profile": "dense", "fixture_id": "T",
                    "schedule_sha256": schedule_hash,
                    "classification": "dense_inconclusive", "semantic_match": False,
                    "requests": records}]}
            path = root / "summary.json"
            path.write_text(json.dumps(summary))
            self.assertTrue(_validate_summary(path)["valid"])
            summary["sequences"][0]["classification"] = "execution_incomplete"
            summary["sequences"][0]["requests"] = records[:1]
            summary["sequences"][0]["requests"][0]["http_status"] = 0
            path.write_text(json.dumps(summary))
            self.assertTrue(_validate_summary(path)["valid"])
            summary["selected_targets"] = ["U", "T"]
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "subset|reordered"):
                _validate_summary(path)
            summary["selected_targets"] = ["T"]
            summary["sequences"][0]["classification"] = "dense_inconclusive"
            summary["sequences"][0]["requests"] = records[:2]
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "expected raw requests"):
                _validate_summary(path)


if __name__ == "__main__":
    unittest.main()
