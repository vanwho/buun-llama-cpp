"""Regression tests for semantic forward-review policy.

The production checker is deliberately separate from the legacy phase-review
validator. These tests exercise the declarative policy contract only.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import forward_review as checker


CAPABILITIES = (
    "build_identity_valid",
    "required_turbo4_placements",
    "semantic_cold_recall",
    "replay_frozen_gpu_attention",
    "larger_occupied_context",
)


class ForwardReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifact = self.root / "evidence.json"
        self.artifact.write_text('{"evidence":"summary only"}\n', encoding="utf-8")
        self.sha256 = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        self.task = {
            "id": "105-05",
            "review_policy": {"type": "semantic_forward_findings_v1"},
        }
        self.state = {"tasks": [
            {"id": self.task["id"], "status": "in_progress"},
            {"id": "105-06", "status": "todo"},
            {"id": "105-07", "status": "todo"},
            {"id": "105-08", "status": "done"},
        ]}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def artifact_ref(self) -> dict:
        return {"path": str(self.artifact), "sha256": self.sha256}

    def pass_row(self) -> dict:
        return {"status": "pass", "artifacts": [self.artifact_ref()]}

    def review(self, *, goal_met: bool = True) -> dict:
        review = {
            "task": self.task["id"],
            "goal_met": goal_met,
            "capabilities": {key: self.pass_row() for key in CAPABILITIES},
            "bulk_prefill": {
                **self.pass_row(),
                "uncached_input_tokens": 2048,
                "fresh_input_tok_s": 610.5,
                "candidate_fingerprint": "a" * 64,
            },
            "matched_decode": {
                **self.pass_row(),
                "selected_tok_s": 90.0,
                "cpu_ram_tok_s": 72.0,
                "identity_and_workload_matched": True,
            },
            "canonical_mtp": {
                **self.pass_row(),
                "prompts": {
                    f"prompt_{index}": {
                        "drafted": 100,
                        "accepted": 50,
                        "acceptance_percent": 50.0,
                        "artifacts": [self.artifact_ref()],
                    }
                    for index in range(1, 4)
                },
            },
            "next_task_ids": [],
            # Informational only: semantic acceptance must not depend on this.
            "optional_physical_witnesses": {"status": "unknown"},
        }
        if not goal_met:
            review["capabilities"]["larger_occupied_context"] = {
                "status": "unknown", "reason": "The larger run is still in progress.",
                "artifacts": [],
            }
            review["next_task_ids"] = ["105-06", "105-07"]
        return review

    def errors(self, review: dict) -> list[str]:
        return checker.check_forward_review(self.state, review, self.task, self.root)

    def test_complete_semantic_review_accepts_valid_hashed_evidence(self) -> None:
        self.assertEqual([], self.errors(self.review()))

    def test_goal_met_requires_every_capability_to_pass(self) -> None:
        review = self.review()
        review["capabilities"]["semantic_cold_recall"] = {
            "status": "unknown", "reason": "No cold-recall finding yet.",
            "artifacts": [],
        }
        self.assertTrue(self.errors(review))

    def test_goal_miss_requires_reason_and_ordered_unfinished_successors(self) -> None:
        review = self.review(goal_met=False)
        review["capabilities"]["larger_occupied_context"].pop("reason")
        errors = self.errors(review)
        self.assertTrue(errors)
        self.assertTrue(any("reason" in error.lower() for error in errors))

        review = self.review(goal_met=False)
        review["next_task_ids"] = ["105-07", "105-06"]
        errors = self.errors(review)
        self.assertTrue(errors)
        self.assertTrue(any("order" in error.lower() or "successor" in error.lower()
                            for error in errors))

    def test_goal_miss_requires_at_least_one_existing_unfinished_successor(self) -> None:
        review = self.review(goal_met=False)
        review["next_task_ids"] = ["not-a-task"]
        self.assertTrue(self.errors(review))

        review = self.review(goal_met=False)
        review["next_task_ids"] = ["105-08"]  # exists, but is already done
        self.assertTrue(self.errors(review))

    def test_goal_met_does_not_require_successors(self) -> None:
        self.assertEqual([], self.errors(self.review(goal_met=True)))

    def test_task_identity_must_match(self) -> None:
        review = self.review()
        review["task"] = "105-04"
        self.assertTrue(self.errors(review))

    def test_pass_artifact_must_exist_and_match_sha256(self) -> None:
        review = self.review()
        review["bulk_prefill"]["artifacts"][0]["sha256"] = "0" * 64
        self.assertTrue(self.errors(review))

        review = self.review()
        review["matched_decode"]["artifacts"][0]["path"] = "missing.json"
        self.assertTrue(self.errors(review))

    def test_nonpass_findings_need_a_reason(self) -> None:
        review = self.review(goal_met=False)
        review["bulk_prefill"] = {
            "status": "fail", "reason": "Measured below the required input rate.",
            "uncached_input_tokens": 2048, "fresh_input_tok_s": 420.0,
            "candidate_fingerprint": "a" * 64, "artifacts": [],
        }
        self.assertEqual([], self.errors(review))
        del review["bulk_prefill"]["reason"]
        self.assertTrue(self.errors(review))

    def test_bulk_prefill_pass_has_minimum_work_and_rate_and_identity(self) -> None:
        for field, value in (("uncached_input_tokens", 1023),
                             ("fresh_input_tok_s", 499.99),
                             ("candidate_fingerprint", "")):
            with self.subTest(field=field):
                review = self.review()
                review["bulk_prefill"][field] = value
                self.assertTrue(self.errors(review))

    def test_matched_decode_pass_requires_matched_positive_rates_and_win(self) -> None:
        for changes in (
            {"selected_tok_s": 72.0},
            {"selected_tok_s": 0.0},
            {"cpu_ram_tok_s": 0.0},
            {"identity_and_workload_matched": False},
        ):
            with self.subTest(changes=changes):
                review = self.review()
                review["matched_decode"].update(changes)
                self.assertTrue(self.errors(review))

    def test_mtp_pass_requires_exact_three_prompt_denominators_and_threshold(self) -> None:
        review = self.review()
        del review["canonical_mtp"]["prompts"]["prompt_3"]
        self.assertTrue(self.errors(review))

        review = self.review()
        review["canonical_mtp"]["prompts"]["prompt_2"]["drafted"] = 0
        self.assertTrue(self.errors(review))

        review = self.review()
        row = review["canonical_mtp"]["prompts"]["prompt_1"]
        row.update(accepted=39, acceptance_percent=39.0)
        self.assertTrue(self.errors(review))

        review = self.review()
        row = review["canonical_mtp"]["prompts"]["prompt_1"]
        row.update(accepted=101, acceptance_percent=101.0)
        self.assertTrue(self.errors(review))

    def test_all_numeric_fields_reject_nonfinite_and_boolean_values(self) -> None:
        mutations = (
            ("bulk_prefill", "fresh_input_tok_s", float("nan")),
            ("bulk_prefill", "fresh_input_tok_s", float("inf")),
            ("bulk_prefill", "fresh_input_tok_s", True),
            ("bulk_prefill", "uncached_input_tokens", True),
            ("matched_decode", "selected_tok_s", float("nan")),
            ("matched_decode", "cpu_ram_tok_s", float("inf")),
        )
        for section, field, value in mutations:
            with self.subTest(section=section, field=field, value=value):
                review = self.review()
                review[section][field] = value
                self.assertTrue(self.errors(review))

        review = self.review()
        review["canonical_mtp"]["prompts"]["prompt_1"]["drafted"] = True
        self.assertTrue(self.errors(review))

    def test_physical_witness_telemetry_is_not_an_acceptance_gate(self) -> None:
        review = self.review()
        review["optional_physical_witnesses"] = {
            "status": "unknown", "reason": "Not required by this review policy."
        }
        self.assertEqual([], self.errors(review))


if __name__ == "__main__":
    unittest.main()
