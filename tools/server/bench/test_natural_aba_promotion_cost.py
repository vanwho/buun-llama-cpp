"""Focused contracts for the natural A/B/A promotion helper."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("natural_aba", HERE / "run-natural-aba-promotion-cost.py")
assert SPEC is not None and SPEC.loader is not None
natural = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(natural)


class NaturalAbaTests(unittest.TestCase):
    def test_cold_target_transition_with_singleton_witness_for_other_page(self) -> None:
        target = {"logical_page_id": 5, "sequence_id": 0, "sequence_generation": 1,
            "generation": 9, "content_version": 256, "position_begin": 1280,
            "position_end": 1536, "resident": False, "host_backed": True}
        promoted = {**target, "resident": True}
        before = {"pager": {"page_inventory": [target], "h2d_useful_bytes": 0}}
        after = {"pager": {"page_inventory": [promoted], "h2d_useful_bytes": 100,
            "selected_page_ids": [5], "natural_proof": {"logical_page_id": 4}}}
        result = natural.classify_target(before, after, target, {"semantic_answer": ""})
        self.assertTrue(result["target_transition"])
        self.assertTrue(result["target_selected"])
        self.assertTrue(result["completed_batch_transfer"])
        self.assertFalse(result["matching_full_target_witness"])
        self.assertEqual("natural_target_promotion", result["physical_classification"])
        self.assertEqual(4, result["natural_proof"]["logical_page_id"])

        matching_after = {"pager": {"page_inventory": [promoted], "h2d_useful_bytes": 100,
            "selected_page_ids": [5], "natural_proof": {"logical_page": 5}}}
        matching = natural.classify_target(before, matching_after, target, {"semantic_answer": ""})
        self.assertTrue(matching["matching_full_target_witness"])

    def test_target_hot_secondary_transfer_is_mixed_control(self) -> None:
        target = {"logical_page_id": 5, "resident": True, "host_backed": True,
            "sequence_id": 0, "sequence_generation": 1, "generation": 9,
            "content_version": 256, "position_begin": 1280, "position_end": 1536}
        before = {"pager": {"page_inventory": [target], "h2d_useful_bytes": 100}}
        after = {"pager": {"page_inventory": [target], "h2d_useful_bytes": 200,
            "selected_page_ids": [5, 17]}}
        cls = natural.classify_target(before, after, target, {})
        self.assertEqual("target_hot_mixed_control", natural.summarize_repeat({}, {}, cls))

    def test_failed_fill_response_cannot_enter_measurement(self) -> None:
        self.assertFalse(natural.fill_response_admissible({"status": "runtime_failure", "http_status": 200}))
        self.assertFalse(natural.fill_response_admissible({"status": "completed", "http_status": 500}))
        self.assertTrue(natural.fill_response_admissible({"status": "completed", "http_status": 200}))

    def test_prompt_renderer_retries_only_transient_transport_failures(self) -> None:
        class Flaky:
            calls = 0

            def __call__(self, messages):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("server prompt endpoint failed: URLError")
                return messages

        flaky = Flaky()
        self.assertEqual([], natural.TransientRetryRenderer(flaky, attempts=2)([]))
        self.assertEqual(2, flaky.calls)

        class Rejected:
            calls = 0

            def __call__(self, messages):
                self.calls += 1
                raise ValueError("server prompt endpoint failed: HTTP 400")

        rejected = Rejected()
        with self.assertRaisesRegex(ValueError, "HTTP 400"):
            natural.TransientRetryRenderer(rejected, attempts=3)([])
        self.assertEqual(1, rejected.calls)

    def test_resume_reuses_exact_schedule_and_body(self) -> None:
        frozen = {"argv": ["server", "-c", "8192"], "schedule": [{"path": "a.c", "start_byte": 0}]}
        self.assertTrue(natural.frozen_inputs_match(frozen, dict(frozen)))
        self.assertFalse(natural.frozen_inputs_match(frozen, {**frozen, "argv": ["server", "-c", "16384"]}))
        body = b'{"model":"qwen","messages":[]}'
        self.assertTrue(natural.exact_body_match(body, body))
        self.assertFalse(natural.exact_body_match(body, body + b" "))
        self.assertFalse(natural.resumable_row_terminal({"status": "setup_failed", "requests": []}))
        self.assertFalse(natural.resumable_row_terminal({"status": "runtime_outcome",
            "requests": [{"stage": "A1"}, {"stage": "B"}]}))
        self.assertTrue(natural.resumable_row_terminal({"status": "runtime_outcome",
            "requests": [{"stage": "A2"}]}))
        history = [{"role": "user", "content": "A1"},
                   {"role": "assistant", "content": "actual A1 reply"},
                   {"role": "user", "content": "B"}]
        appended = natural.append_actual_reply(history, "actual B reply")
        self.assertEqual(["user", "assistant", "user", "assistant"],
                         [message["role"] for message in appended])
        self.assertEqual("B", appended[-2]["content"])

        class Rendered:
            def __init__(self, count: int):
                self.token_ids = tuple(range(count))

        class Renderer:
            def __call__(self, messages):
                return Rendered(sum(len(item["content"]) for item in messages))

        near_target = [{"role": "user", "content": "x" * 13646}]
        self.assertEqual(1024, natural.pressure_append_budget(
            Renderer(), near_target, 14336, "q" * 100))


if __name__ == "__main__":
    unittest.main()
